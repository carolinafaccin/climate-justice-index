"""
ETL: Copernicus DEM / GEE → Indicator e1 (landslide susceptibility via slope).

Input:  cfg.RAW_DIR state-level CSVs with columns h3_id, alta_media
Output: cfg.FILES_H3["e1"] parquet
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(PROJECT_ROOT)
from src import config as cfg
from src import utils

# ==============================================================================
# 1. PATHS
# ==============================================================================
GEE_DIR = cfg.RAW_DIR / cfg.INDICATORS["e1"]["source"]["dir"]

now = datetime.now().strftime("%Y%m%d_%H%M%S")
DIAGNOSTIC_TXT = cfg.DIAGNOSE_DIR / f"diagnostic_h3_e1_deslizamentos_slope_{now}.txt"

col_e1_norm = cfg.COLUMN_MAP["e1"]
col_e1_abs  = col_e1_norm.replace("_norm", "_abs")


# ==============================================================================
# 2. MAIN
# ==============================================================================
def main():
    print("=" * 60)
    print("ETL: GEE Copernicus DEM — E1 (Landslide Susceptibility)")
    print(f"Source: {GEE_DIR}")
    print("=" * 60)

    # Load all UF CSVs
    print("\n1/4 - Loading GEE CSVs...")
    csv_files = sorted(GEE_DIR.glob("h3_susc_desliz_slope_v1_uf_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in: {GEE_DIR}")
    print(f"   Files found: {len(csv_files)}")

    parts = []
    for path in csv_files:
        try:
            df = pd.read_csv(path, usecols=["h3_id", "alta_media"])
        except ValueError:
            df = pd.read_csv(path)
            df.columns = df.columns.str.lower()
            df = df[["h3_id", "alta_media"]]
        parts.append(df)
        print(f"   ✓ {path.name}  ({len(df):,} hexagons)")

    df_all = pd.concat(parts, ignore_index=True)
    print(f"\n   Total: {len(df_all):,} hexagons loaded")

    dupes = df_all["h3_id"].duplicated().sum()
    if dupes > 0:
        print(f"   WARNING: {dupes:,} duplicate h3_ids — keeping first value.")
        df_all = df_all.drop_duplicates(subset="h3_id", keep="first")

    # ==============================================================================
    # 3. INDICATOR CALCULATION
    # ==============================================================================
    print("\n2/4 - Calculating indicator...")

    df_all["alta_media"] = pd.to_numeric(df_all["alta_media"], errors="coerce").fillna(0)

    # e1_abs = fraction of hex area with high or medium slope susceptibility (0–1)
    # No weighting by households — the metric is directly the spatial extent of slope risk
    df_all[col_e1_abs] = df_all["alta_media"]

    # winsorize=False: landslide susceptibility is geographically concentrated
    # (mostly mountainous/hilly areas). P99 would collapse to ~0 and normalize poorly.
    df_all[col_e1_norm] = utils.normalize_minmax(df_all[col_e1_abs], winsorize=False)

    n_risk = (df_all["alta_media"] > 0).sum()
    print(f"   Hexagons with risk > 0: {n_risk:,}")
    print(f"   Mean fraction at risk: {df_all[col_e1_abs].mean():.4f}")

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
        f.write("GEE Copernicus DEM — E1 Landslide Susceptibility ETL Diagnostic\n")
        f.write(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"GEE directory : {GEE_DIR}\n")
        f.write(f"Files read    : {len(csv_files)}\n\n")

        f.write("--- alta_media (fraction of hex area with high/medium slope) ---\n")
        s_raw = df_all["alta_media"].dropna()
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
