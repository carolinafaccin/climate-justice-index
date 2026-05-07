"""
ETL: Copernicus DEM / GEE → Indicator e3 (sea-level rise exposure).

Input:  cfg.RAW_DIR state-level CSVs with columns h3_id, risco_slr, qtd_dom
Output: cfg.FILES_H3["e3"] parquet
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.append(PROJECT_ROOT)
from src import config as cfg
from src import utils

# ==============================================================================
# 1. PATHS
# ==============================================================================
GEE_DIR = cfg.RAW_DIR / cfg.INDICATORS["e3"]["source"]["dir"]

now = datetime.now().strftime("%Y%m%d_%H%M%S")
DIAGNOSTIC_TXT = cfg.DIAGNOSE_DIR / f"diagnostic_h3_e3_mar_{now}.txt"

col_e3_norm = cfg.COLUMN_MAP["e3"]
col_e3_abs  = col_e3_norm.replace("_norm", "_abs")


# ==============================================================================
# 2. MAIN
# ==============================================================================
def main():
    print("=" * 60)
    print("ETL: GEE Copernicus DEM — E3 (Sea Level Rise Susceptibility)")
    print(f"Source: {GEE_DIR}")
    print("=" * 60)

    # Load all coastal UF CSVs
    print("\n1/4 - Loading GEE CSVs...")
    csv_files = sorted(GEE_DIR.glob("h3_susc_mar_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in: {GEE_DIR}")
    print(f"   Files found: {len(csv_files)}")

    parts = []
    for path in csv_files:
        try:
            df = pd.read_csv(path, usecols=["h3_id", "qtd_dom", "risco_slr"])
        except ValueError:
            df = pd.read_csv(path)
            df.columns = df.columns.str.lower()
            df = df[["h3_id", "qtd_dom", "risco_slr"]]
        parts.append(df)
        print(f"   ✓ {path.name}  ({len(df):,} hexagons)")

    df_all = pd.concat(parts, ignore_index=True)
    print(f"\n   Total: {len(df_all):,} coastal hexagons loaded")

    dupes = df_all["h3_id"].duplicated().sum()
    if dupes > 0:
        print(f"   WARNING: {dupes:,} duplicate h3_ids — keeping first value.")
        df_all = df_all.drop_duplicates(subset="h3_id", keep="first")

    # ==============================================================================
    # 3. INDICATOR CALCULATION
    # ==============================================================================
    print("\n2/4 - Calculating indicator...")

    df_all["risco_slr"] = pd.to_numeric(df_all["risco_slr"], errors="coerce").fillna(0)
    df_all["qtd_dom"]   = pd.to_numeric(df_all["qtd_dom"],   errors="coerce").fillna(0)

    # e3_abs = households in sea-level-risk hexagons (0 for non-risk hexagons)
    # risco_slr is binary (0/1) from Reducer.first() — see GEE script
    # If re-exported with Reducer.mean(), risco_slr becomes fraction of area at risk (0–1),
    # and this formula still holds correctly without any changes.
    df_all[col_e3_abs] = df_all["qtd_dom"] * df_all["risco_slr"]

    # winsorize=False: coastal indicator (<1% of hexagons), P99=0 would collapse normalisation
    df_all[col_e3_norm] = utils.normalize_minmax(df_all[col_e3_abs], winsorize=False)

    n_risk = (df_all["risco_slr"] > 0).sum()
    print(f"   Hexagons with risk > 0: {n_risk:,}")
    print(f"   Total households at risk: {int(df_all.loc[df_all['risco_slr'] > 0, 'qtd_dom'].sum()):,}")

    # ==============================================================================
    # 4. MERGE WITH H3 BASE AND SAVE
    # ==============================================================================
    print("\n3/4 - Merging with H3 base grid...")
    df_h3 = pd.read_parquet(cfg.FILES_H3["base_metadata"], columns=["h3_id"])

    df_final = df_h3.merge(
        df_all[["h3_id", col_e3_abs, col_e3_norm]],
        on="h3_id", how="left"
    )

    # Inland hexagons (not in coastal CSVs) are not at risk → fill with 0
    df_final[[col_e3_abs, col_e3_norm]] = df_final[[col_e3_abs, col_e3_norm]].fillna(0)

    print(f"   Hexagons with households at risk: {(df_final[col_e3_abs] > 0).sum():,}")

    print("\n4/4 - Saving parquet...")
    utils.save_parquet(df_final, cfg.FILES_H3["e3"])
    print(f"   ✓ Saved: {cfg.FILES_H3['e3'].name}")

    _write_diagnostic(df_all, df_final, csv_files)
    print(f"\nDiagnostic: {DIAGNOSTIC_TXT}")
    print("Done!")


def _write_diagnostic(df_all, df_final, csv_files):
    with open(DIAGNOSTIC_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("GEE Copernicus DEM — E3 Sea Level Rise ETL Diagnostic\n")
        f.write(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"GEE directory : {GEE_DIR}\n")
        f.write(f"Files read    : {len(csv_files)}\n\n")

        f.write("--- risco_slr (coastal hexagons) ---\n")
        vc = df_all["risco_slr"].value_counts().sort_index()
        for v, c in vc.items():
            f.write(f"  risco_slr={v}: {c:,} hexagons\n")
        f.write(f"\n  Total households at risk: {int(df_all.loc[df_all['risco_slr'] > 0, 'qtd_dom'].sum()):,}\n\n")

        for col in [col_e3_abs, col_e3_norm]:
            s = df_final[col]
            f.write(f"--- {col} (full grid, inland=0) ---\n")
            f.write(f"  mean   = {s.mean():.6f}\n")
            f.write(f"  median = {s.median():.6f}\n")
            f.write(f"  min    = {s.min():.6f}\n")
            f.write(f"  max    = {s.max():.6f}\n")
            f.write(f"  zeros  = {(s == 0).sum():,}\n")
            f.write(f"  nulls  = {s.isna().sum():,}\n\n")


if __name__ == "__main__":
    main()
