"""
ETL: NASA LHASA / GEE → Indicator e1 (landslide susceptibility).

Fonte: Stanley & Kirschbaum (2017) — Global Landslide Susceptibility Map
       Modelo multicritério (slope + geology + forest loss + roads + faults).

Métrica principal: lhasa_high_frac (fração da área do hexágono com LHASA >= 4,
i.e. classes "High" ou "Very High"). Equivalente conceitual ao "alta_media"
do método de slope, mas calibrado com múltiplos critérios.

Input:  cfg.RAW_DIR state-level CSVs com colunas h3_id, lhasa_mean, lhasa_high_frac
Output: cfg.FILES_H3["e1"] parquet
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pipeline.py").exists())
sys.path.insert(0, str(_ROOT))
from src import config as cfg
from src import utils

# ==============================================================================
# 1. PATHS
# ==============================================================================
GEE_DIR = cfg.RAW_DIR / cfg.INDICATORS["e1"]["source"]["dir"]

now = datetime.now().strftime("%Y%m%d_%H%M%S")
DIAGNOSTIC_TXT = cfg.DIAGNOSE_DIR / f"diagnostic_h3_e1_deslizamentos_lhasa_{now}.txt"

col_e1_norm = cfg.COLUMN_MAP["e1"]
col_e1_abs  = col_e1_norm.replace("_norm", "_abs")


# ==============================================================================
# 2. MAIN
# ==============================================================================
def main():
    print("=" * 60)
    print("ETL: NASA LHASA — E1 (Landslide Susceptibility)")
    print(f"Source: {GEE_DIR}")
    print("=" * 60)

    # Load all UF CSVs
    print("\n1/4 - Loading GEE CSVs...")
    csv_files = sorted(GEE_DIR.glob("h3_susc_desliz_lhasa_v1_uf_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in: {GEE_DIR}")
    print(f"   Files found: {len(csv_files)}")

    parts = []
    for path in csv_files:
        df = utils.read_csv_columns(path, ["h3_id", "lhasa_mean", "lhasa_high_frac"])
        parts.append(df)
        print(f"   ✓ {path.name}  ({len(df):,} hexagons)")

    df_all = pd.concat(parts, ignore_index=True)
    print(f"\n   Total: {len(df_all):,} hexagons loaded")

    # Border hexagons may appear in multiple UF-level GEE exports — deduplicate.
    dupes = df_all["h3_id"].duplicated().sum()
    if dupes > 0:
        print(f"   NOTE: {dupes:,} border hexagons in multiple UF exports ({100*dupes/len(df_all):.1f}%) — keeping first")
        df_all = df_all.drop_duplicates(subset="h3_id", keep="first")

    # ==============================================================================
    # 3. INDICATOR CALCULATION
    # ==============================================================================
    print("\n2/4 - Calculating indicator...")

    df_all["lhasa_mean"]      = pd.to_numeric(df_all["lhasa_mean"],      errors="coerce")
    df_all["lhasa_high_frac"] = pd.to_numeric(df_all["lhasa_high_frac"], errors="coerce")

    # Fill NaN with 0 (hexágonos sem dado LHASA = sem informação = assumir sem risco)
    df_all["lhasa_high_frac"] = df_all["lhasa_high_frac"].fillna(0)

    # e1_abs = fração da área do hexágono em classe High/Very High do LHASA (0–1)
    # Não há ponderação por domicílios — a métrica é diretamente a extensão espacial
    # da suscetibilidade alta no hexágono.
    df_all[col_e1_abs] = df_all["lhasa_high_frac"]

    # winsorize=False: suscetibilidade é geograficamente concentrada
    # (Amazônia + áreas tropicais úmidas com desmatamento). P99 colapsaria a cauda.
    df_all[col_e1_norm] = utils.normalize_minmax(df_all[col_e1_abs], winsorize=False)

    n_risk = (df_all["lhasa_high_frac"] > 0).sum()
    print(f"   Hexagons with LHASA high/very-high (> 0): {n_risk:,}")
    print(f"   Mean fraction at risk: {df_all[col_e1_abs].mean():.4f}")
    print(f"   Mean LHASA value: {df_all['lhasa_mean'].mean():.2f}")

    # ==============================================================================
    # 4. MERGE WITH H3 BASE AND SAVE
    # ==============================================================================
    print("\n3/4 - Merging with H3 base grid...")
    df_h3 = pd.read_parquet(cfg.FILES_H3["base_metadata"], columns=["h3_id"])

    df_final = df_h3.merge(
        df_all[["h3_id", col_e1_abs, col_e1_norm]],
        on="h3_id", how="left"
    )

    print(f"   Hexagons with risk: {(df_final[col_e1_abs] > 0).sum():,}")
    print(f"   Hexagons without e1 data (NaN): {df_final[col_e1_abs].isna().sum():,}")

    print("\n4/4 - Saving parquet...")
    utils.save_parquet(df_final, cfg.FILES_H3["e1"])
    print(f"   ✓ Saved: {cfg.FILES_H3['e1'].name}")

    _write_diagnostic(df_all, df_final, csv_files)
    print(f"\nDiagnostic: {DIAGNOSTIC_TXT}")
    print("Done!")


def _write_diagnostic(df_all, df_final, csv_files):
    with open(DIAGNOSTIC_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("NASA LHASA — E1 Landslide Susceptibility ETL Diagnostic\n")
        f.write(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"GEE directory : {GEE_DIR}\n")
        f.write(f"Files read    : {len(csv_files)}\n\n")

        f.write("--- lhasa_mean (raw LHASA value 1–5) ---\n")
        s_mean = df_all["lhasa_mean"].dropna()
        f.write(f"  mean   = {s_mean.mean():.4f}\n")
        f.write(f"  median = {s_mean.median():.4f}\n")
        f.write(f"  min    = {s_mean.min():.4f}\n")
        f.write(f"  max    = {s_mean.max():.4f}\n")
        f.write(f"  count  = {len(s_mean):,}\n\n")

        f.write("--- lhasa_high_frac (fraction of hex area with LHASA >= 4) ---\n")
        s_raw = df_all["lhasa_high_frac"].dropna()
        f.write(f"  mean   = {s_raw.mean():.6f}\n")
        f.write(f"  median = {s_raw.median():.6f}\n")
        f.write(f"  min    = {s_raw.min():.6f}\n")
        f.write(f"  max    = {s_raw.max():.6f}\n")
        f.write(f"  = 0    = {(s_raw == 0).sum():,} hexagons\n")
        f.write(f"  > 0    = {(s_raw > 0).sum():,} hexagons\n\n")

        for col in [col_e1_abs, col_e1_norm]:
            s = df_final[col]
            f.write(f"--- {col} (full grid) ---\n")
            f.write(f"  mean   = {s.mean():.6f}\n")
            f.write(f"  median = {s.median():.6f}\n")
            f.write(f"  min    = {s.min():.6f}\n")
            f.write(f"  max    = {s.max():.6f}\n")
            f.write(f"  zeros  = {(s == 0).sum():,}\n")
            f.write(f"  nulls  = {s.isna().sum():,}\n\n")


if __name__ == "__main__":
    main()
