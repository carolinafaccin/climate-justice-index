import pandas as pd
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
ICM_DIR = cfg.RAW_DIR / cfg.INDICATORS["g7"]["source"]["dir"]
SRC_COL  = cfg.INDICATORS["g7"]["source"]["col"]   # "v7"

now = datetime.now().strftime("%Y%m%d_%H%M%S")
DIAGNOSTIC_TXT = cfg.DIAGNOSE_DIR / f"diagnostic_h3_g7_midr_{now}.txt"

col_g7_norm = cfg.COLUMN_MAP["g7"]
col_g7_abs  = col_g7_norm.replace("_norm", "_abs")

# ==============================================================================
# 2. LOAD AND COMBINE ALL LISTS (lista-a.csv … lista-d.csv)
# ==============================================================================
print("=" * 60)
print("ETL: MIDR/ICM — G7 (Registry of families in risk areas)")
print(f"Source: {ICM_DIR}")
print("=" * 60)

print("\n1/4 - Loading ICM CSVs...")
csv_files = sorted(ICM_DIR.glob("lista-*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No 'lista-*.csv' CSV found in: {ICM_DIR}")

parts = []
for path in csv_files:
    try:
        df = pd.read_csv(path, sep=';', usecols=["cod_mun", SRC_COL], dtype={"cod_mun": str})
    except Exception:
        df = pd.read_csv(path, sep=';', encoding='latin1', usecols=["cod_mun", SRC_COL], dtype={"cod_mun": str})
    parts.append(df)
    print(f"   ✓ {path.name}  ({len(df):,} municipalities)")

df_icm = pd.concat(parts, ignore_index=True)
print(f"\n   Total before deduplication: {len(df_icm):,} rows")

# A municipality could appear in more than one list — keep the highest v7 value
df_icm["cod_mun"] = df_icm["cod_mun"].astype(str).str.strip()
df_icm[SRC_COL]   = pd.to_numeric(df_icm[SRC_COL], errors="coerce")
df_icm = df_icm.groupby("cod_mun", as_index=False)[SRC_COL].max()
print(f"   Unique municipalities: {len(df_icm):,}")

# ==============================================================================
# 3. INDICATOR — binary: 1 = has registry, 0 = doesn't
# ==============================================================================
print("\n2/4 - Calculating indicator...")
df_icm[col_g7_abs]  = df_icm[SRC_COL]
df_icm[col_g7_norm] = df_icm[col_g7_abs]  # already 0/1 — no normalization needed

n_yes = (df_icm[col_g7_abs] == 1).sum()
n_no  = (df_icm[col_g7_abs] == 0).sum()
print(f"   v7=1 (has registry):   {n_yes:,} municipalities")
print(f"   v7=0 (no registry):    {n_no:,} municipalities")

# ==============================================================================
# 4. MERGE WITH H3 BASE
# ==============================================================================
print("\n3/4 - Merging with H3 base grid...")
df_h3 = pd.read_parquet(cfg.FILES_H3["base_metadata"], columns=["h3_id", "cd_mun"])
df_h3["cd_mun"] = df_h3["cd_mun"].astype(str).str.strip()

df_final = df_h3.merge(
    df_icm[["cod_mun", col_g7_abs, col_g7_norm]].rename(columns={"cod_mun": "cd_mun"}),
    on="cd_mun", how="left"
)

n_matched   = df_final[col_g7_abs].notna().sum()
n_unmatched = df_final[col_g7_abs].isna().sum()
print(f"   Hexagons with data:          {n_matched:,}")
print(f"   Hexagons without data (NaN): {n_unmatched:,}")

# ==============================================================================
# 5. SAVE
# ==============================================================================
print("\n4/4 - Saving parquet...")
df_export = df_final[["h3_id", col_g7_abs, col_g7_norm]]
utils.save_parquet(df_export, cfg.FILES_H3["g7"])
print(f"   ✓ Saved: {cfg.FILES_H3['g7'].name}")

# ==============================================================================
# 6. DIAGNOSTIC
# ==============================================================================
with open(DIAGNOSTIC_TXT, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("MIDR/ICM — G7 Families in Risk Areas Registry ETL Diagnostic\n")
    f.write(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"ICM directory : {ICM_DIR}\n")
    f.write(f"Files read    : {[p.name for p in csv_files]}\n")
    f.write(f"Source column : {SRC_COL}\n\n")
    f.write(f"Unique municipalities processed: {len(df_icm):,}\n")
    f.write(f"  v7=1 (has registry)    : {n_yes:,}\n")
    f.write(f"  v7=0 (no registry)     : {n_no:,}\n\n")
    f.write(f"Hexagons with data       : {n_matched:,}\n")
    f.write(f"Hexagons without data (NaN): {n_unmatched:,}\n")

print(f"\nDiagnostic: {DIAGNOSTIC_TXT}")
print("Done!")
