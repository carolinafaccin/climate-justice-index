import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# ==============================================================================
# 1. ENVIRONMENT CONFIGURATION
# ==============================================================================
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.append(PROJECT_ROOT)

from src import config as cfg
from src import utils

# ==============================================================================
# 2. PATHS AND DIAGNOSTIC DEFINITION
# ==============================================================================
now = datetime.now().strftime("%Y%m%d_%H%M%S")
DIAGNOSTIC_TXT = cfg.DIAGNOSE_DIR / f'diagnostic_munic_h3_{now}.txt'

# Columns used as join keys in MUNIC CSVs
ID_COLS = ['cd_mun', 'sigla_uf', 'cd_uf', 'nm_mun']

# Indicators driven by this script: all those with a 'file' key in their source
# MUNIC sources are identified by having both 'file' and a column spec ('col' or 'cols')
MUNIC_INDICATORS = {k: v for k, v in cfg.INDICATORS.items()
                    if 'source' in v and 'file' in v['source']
                    and ('col' in v['source'] or 'cols' in v['source'])}


# ==============================================================================
# 3. HELPER
# ==============================================================================
def load_and_select(path: Path, extra_cols: list) -> pd.DataFrame:
    """Loads a MUNIC CSV (handles encoding) and returns ID columns + target columns."""
    if not path.exists():
        raise FileNotFoundError(f"Error: File not found -> {path}")
    try:
        df = pd.read_csv(path, sep=',', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=';', encoding='latin1')
    df.columns = df.columns.str.lower()
    final_cols = [c for c in ID_COLS if c in df.columns] + [c.lower() for c in extra_cols]
    return df[final_cols]


# ==============================================================================
# 4. PROCESSING — data-driven loop via indicators.json etl_source
# ==============================================================================
print("Starting ETL Pipeline - IBGE MUNIC...")
print(f"1/3 - Indicators to process: {list(MUNIC_INDICATORS.keys())}")

df_h3 = pd.read_parquet(cfg.FILES_H3["base_metadata"], columns=['h3_id', 'cd_mun'])
df_h3['cd_mun'] = df_h3['cd_mun'].astype(str)

generated_files = {}

for ind_key, meta in MUNIC_INDICATORS.items():
    source   = meta['source']
    col_norm = cfg.COLUMN_MAP[ind_key]
    src_path = cfg.RAW_DIR / source['file']

    print(f"\n  [{ind_key}] {col_norm} — {src_path.name}")

    if source['type'] == 'binary':
        if 'cols' in source:
            src_cols = source['cols']
            df_src = load_and_select(src_path, src_cols)
            for c in src_cols:
                if c in df_src.columns:
                    df_src[c] = df_src[c].map({'Sim': 1, 'Não': 0})
            existing_cols = [c for c in src_cols if c in df_src.columns]
            df_src[col_norm] = df_src[existing_cols].max(axis=1)
        else:
            src_col = source['col']
            df_src  = load_and_select(src_path, [src_col])
            df_src[src_col]  = df_src[src_col].map({'Sim': 1, 'Não': 0})
            df_src[col_norm] = df_src[src_col]
        df_merge = df_src[['cd_mun', col_norm]]

    elif source['type'] == 'count':
        src_cols     = source['cols']
        df_src       = load_and_select(src_path, src_cols)
        for c in src_cols:
            if c in df_src.columns:
                df_src[c] = df_src[c].map({'Sim': 1, 'Não': 0})
        col_abs          = col_norm.replace('_norm', '_abs')
        existing_cols    = [c for c in src_cols if c in df_src.columns]
        df_src[col_abs]  = df_src[existing_cols].sum(axis=1, min_count=1)
        df_src[col_norm] = utils.normalize_minmax(df_src[col_abs], winsorize=True)
        df_merge = df_src[['cd_mun', col_abs, col_norm]]

    else:
        print(f"  Unknown type '{source['type']}' for {ind_key}, skipping.")
        continue

    df_merge['cd_mun'] = df_merge['cd_mun'].astype(str)
    df_final = df_h3.merge(df_merge, on='cd_mun', how='left')

    out_path = cfg.FILES_H3[ind_key]
    df_final.to_parquet(out_path, index=False)
    generated_files[ind_key] = (out_path.name, df_final)
    print(f"  ✓ Saved: {out_path.name}")

# ==============================================================================
# 5. DIAGNOSTICS AND EXPORT (.txt)
# ==============================================================================
print("\n3/3 - Generating diagnostic file...")

with open(DIAGNOSTIC_TXT, 'w', encoding='utf-8') as f:
    f.write("=== PARQUET FILES DIAGNOSTIC (H3 Level) ===\n")
    f.write("Note: The counts below reflect the number of hexagons, not municipalities.\n\n")

    for key, (file_name, df_diag) in generated_files.items():
        f.write(f"--- {key.upper()} : {file_name} ---\n")
        value_cols = [c for c in df_diag.columns if c not in ['h3_id', 'cd_mun']]

        for col in value_cols:
            nulls = df_diag[col].isna().sum()
            if 'norm' in col:
                f.write(f"Column: {col}\n")
                f.write(f"  > Value 1 (Yes/Maximum): {(df_diag[col] == 1).sum()}\n")
                f.write(f"  > Value 0 (No/Minimum): {(df_diag[col] == 0).sum()}\n")
                f.write(f"  > Null Values (NaN): {nulls}\n")
            if 'abs' in col:
                f.write(f"Column: {col}\n")
                f.write(f"  > Mean: {df_diag[col].mean():.2f}\n")
                f.write(f"  > Median: {df_diag[col].median()}\n")
                f.write(f"  > Maximum: {df_diag[col].max()}\n")
                f.write(f"  > Minimum: {df_diag[col].min()}\n")
                f.write(f"  > Null Values (NaN): {nulls}\n")
        f.write("-" * 50 + "\n")

print(f"\n✅ Pipeline completed successfully! Diagnostic saved at: {DIAGNOSTIC_TXT}")
