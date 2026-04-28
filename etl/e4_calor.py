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
GEE_DIR = cfg.RAW_DIR / cfg.INDICATORS["e4"]["source"]["dir"]

now = datetime.now().strftime("%Y%m%d_%H%M%S")
DIAGNOSTIC_TXT = cfg.DIAGNOSE_DIR / f"diagnostic_h3_e4_calor_{now}.txt"

col_e4_norm = cfg.COLUMN_MAP["e4"]
col_e4_abs  = col_e4_norm.replace("_norm", "_abs")


# ==============================================================================
# 2. MAIN
# ==============================================================================
def main():
    print("=" * 60)
    print("ETL: GEE Landsat — E4 (Extreme Heat / LST Anomaly)")
    print(f"Source: {GEE_DIR}")
    print("=" * 60)

    # Load all UF CSVs
    print("\n1/4 - Loading GEE CSVs...")
    csv_files = sorted(GEE_DIR.glob("h3_anomalia_calor_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in: {GEE_DIR}")
    print(f"   Files found: {len(csv_files)}")

    parts = []
    for path in csv_files:
        try:
            df = pd.read_csv(path, usecols=["h3_id", "anomalia_temp"])
        except ValueError:
            # fallback: read all columns and select manually
            df = pd.read_csv(path)
            df.columns = df.columns.str.lower()
            df = df[["h3_id", "anomalia_temp"]]
        parts.append(df)
        print(f"   ✓ {path.name}  ({len(df):,} hexagons)")

    df_all = pd.concat(parts, ignore_index=True)
    print(f"\n   Total: {len(df_all):,} hexagons loaded")

    # Sanity check for duplicates
    dupes = df_all["h3_id"].duplicated().sum()
    if dupes > 0:
        print(f"   WARNING: {dupes:,} duplicate h3_ids — keeping first value.")
        df_all = df_all.drop_duplicates(subset="h3_id", keep="first")

    # ==============================================================================
    # 3. INDICATOR CALCULATION
    # ==============================================================================
    print("\n2/4 - Calculating indicator...")

    df_all["anomalia_temp"] = pd.to_numeric(df_all["anomalia_temp"], errors="coerce")

    # e4_abs: positive anomaly only — cooling or stable hexagons score 0
    df_all[col_e4_abs] = df_all["anomalia_temp"].clip(lower=0)

    # e4_norm: min-max with winsorization
    df_all[col_e4_norm] = utils.normalize_minmax(df_all[col_e4_abs], winsorize=True)

    n_warming   = (df_all[col_e4_abs] > 0).sum()
    n_stable    = (df_all[col_e4_abs] == 0).sum()
    n_null      = df_all[col_e4_abs].isna().sum()
    print(f"   Hexagons warming (anomaly > 0): {n_warming:,}")
    print(f"   Hexagons stable/cooling (≤ 0):  {n_stable:,}")
    print(f"   Hexagons without data (NaN):    {n_null:,}")

    # ==============================================================================
    # 4. MERGE WITH H3 BASE AND SAVE
    # ==============================================================================
    print("\n3/4 - Merging with H3 base grid...")
    df_h3 = pd.read_parquet(cfg.FILES_H3["base_metadata"], columns=["h3_id"])
    df_final = df_h3.merge(
        df_all[["h3_id", col_e4_abs, col_e4_norm]],
        on="h3_id", how="left"
    )

    n_matched  = df_final[col_e4_abs].notna().sum()
    n_unmatched = df_final[col_e4_abs].isna().sum()
    print(f"   Hexagons with value:    {n_matched:,}")
    print(f"   Hexagons without match: {n_unmatched:,}")

    print("\n4/4 - Saving parquet...")
    utils.save_parquet(df_final, cfg.FILES_H3["e4"])
    print(f"   ✓ Saved: {cfg.FILES_H3['e4'].name}")

    _write_diagnostic(df_all, df_final, csv_files)
    print(f"\nDiagnostic: {DIAGNOSTIC_TXT}")
    print("Done!")


def _write_diagnostic(df_all, df_final, csv_files):
    with open(DIAGNOSTIC_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("GEE Landsat — E4 Extreme Heat ETL Diagnostic\n")
        f.write(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"GEE directory : {GEE_DIR}\n")
        f.write(f"Files read    : {len(csv_files)}\n\n")

        f.write("--- anomalia_temp (raw, before clip) ---\n")
        s_raw = df_all["anomalia_temp"].dropna()
        f.write(f"  mean   = {s_raw.mean():.4f} °C\n")
        f.write(f"  median = {s_raw.median():.4f} °C\n")
        f.write(f"  min    = {s_raw.min():.4f} °C\n")
        f.write(f"  max    = {s_raw.max():.4f} °C\n")
        f.write(f"  < 0    = {(s_raw < 0).sum():,} hexagons\n")
        f.write(f"  = 0    = {(s_raw == 0).sum():,} hexagons\n")
        f.write(f"  > 0    = {(s_raw > 0).sum():,} hexagons\n\n")

        for col in [col_e4_abs, col_e4_norm]:
            s = df_final[col].dropna()
            f.write(f"--- {col} ---\n")
            f.write(f"  mean   = {s.mean():.6f}\n")
            f.write(f"  median = {s.median():.6f}\n")
            f.write(f"  min    = {s.min():.6f}\n")
            f.write(f"  max    = {s.max():.6f}\n")
            f.write(f"  zeros  = {(s == 0).sum():,}\n")
            f.write(f"  nulls  = {df_final[col].isna().sum():,}\n\n")


if __name__ == "__main__":
    main()
