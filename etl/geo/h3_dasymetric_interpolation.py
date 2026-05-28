"""
ETL: CNEFE 2022 household data → dasymetric base parquet (base_metadata).

Distributes census-tract household counts across H3 hexagons using dasymetric
weighting.  The resulting peso_dom column represents the fraction of a census
tract's households that fall within each hexagon:

    HexagonValue = TractValue × peso_dom

Input:  cfg.RAW_DIR / h3_past / br_h3_res9_v1.parquet  (H3↔sector grid)
        cfg.RAW_DIR / h3_past / chunks_uf_cnefe_domicilios / *.parquet  (CNEFE)
Output: cfg.FILES_H3["base_metadata"]  (one row per inhabited hexagon)
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# ==============================================================================
# 1. ENVIRONMENT CONFIGURATION
# ==============================================================================
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pipeline.py").exists())
sys.path.insert(0, str(_ROOT))

from src import config as cfg

# ==============================================================================
# 2. PATHS DEFINITION
# ==============================================================================
# Pointing to the old/temporary folders inside the RAW_DIR (inputs/raw/)
path_v1 = cfg.RAW_DIR / 'h3_past' / 'br_h3_res9_v1.parquet'
chunks_dir = cfg.RAW_DIR / 'h3_past' / 'chunks_uf_cnefe_domicilios'

# The final output is our global metadata base!
output_path = cfg.FILES_H3["base_metadata"]

# ==============================================================================
# 3. BASE GRID LOADING
# ==============================================================================
def main() -> None:
    print("Starting dasymetric consolidation...")

    print("1/3 - Loading base grid...")
    cols_base = ['h3_id', 'cd_setor', 'cd_mun', 'nm_mun', 'cd_uf', 'nm_uf']
    df_base = pd.read_parquet(path_v1, columns=cols_base)
    print(f"   Base grid loaded: {len(df_base):,} hexagons.")

    print("2/3 - Loading CNEFE household chunks...")
    list_households = []
    for file_path in chunks_dir.glob('*.parquet'):
        temp_df = pd.read_parquet(file_path)
        temp_df = temp_df.rename(columns={'qtd_domicilios': 'qtd_dom'})
        list_households.append(temp_df)

    df_dom = pd.concat(list_households, ignore_index=True)
    print(f"   Households loaded: {len(df_dom):,} records.")

    print("3/3 - Calculating dasymetric weights and saving...")
    df_final = pd.merge(df_base, df_dom, on='h3_id', how='inner')

    # Sum of households per census tract
    df_final['total_dom_setor'] = df_final.groupby('cd_setor')['qtd_dom'].transform('sum')
    # Weight = fraction of the tract's households that fall within each hexagon
    df_final['peso_dom'] = df_final['qtd_dom'] / df_final['total_dom_setor']

    # Division by zero when total_dom_setor=0 produces inf, not NaN; replace both.
    df_final['peso_dom'] = df_final['peso_dom'].replace([np.inf, -np.inf], 0).fillna(0)
    df_final = df_final.drop(columns=['total_dom_setor'])
    df_final = df_final[df_final['qtd_dom'] > 0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(output_path, index=False, compression='snappy')

    print(f"   ✓ Saved: {output_path.name}")
    print(f"   Columns: {df_final.columns.tolist()}")
    print(f"   Total inhabited hexagons: {len(df_final):,}")
    print("Done!")


if __name__ == "__main__":
    main()