import pandas as pd
import numpy as np
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
# 2. BUSINESS PARAMETERS
# ==============================================================================
# Change these variables to easily adjust the indicator's logic in the future
TARGET_YEARS = range(2015, 2025) # 2015 to 2024 (10 years)
TARGET_COLUMN = "Despesas Liquidadas"
TARGET_ACCOUNT = "18 - Gestão Ambiental" # Exact string from the IBGE/Siconfi file

# ==============================================================================
# 3. PATHS, COLUMNS AND DIAGNOSTIC DEFINITION
# ==============================================================================
input_dir = cfg.RAW_DIR / cfg.INDICATORS["g1"]["source"]["dir"]

h3_path = cfg.FILES_H3["base_metadata"]
output_path = cfg.FILES_H3["g1"]

# Dynamic column names (from indicators.json)
col_norm = cfg.COLUMN_MAP["g1"]
col_abs  = col_norm.replace('_norm', '_abs')
col_log  = col_norm.replace('_norm', '_log')

# Diagnostic log configuration
now = datetime.now().strftime("%Y%m%d_%H%M%S")
DIAGNOSTIC_TXT = cfg.DIAGNOSE_DIR / f'diagnostic_h3_g1_siconfi_{now}.txt'

# ==============================================================================
# 4. HELPER FUNCTIONS
# ==============================================================================
def standardize_columns(col):
    """Formats column names to lowercase and removes accents/spaces."""
    col = str(col).lower()
    col = col.replace('ç', 'c').replace('ã', 'a').replace('é', 'e').replace('õ', 'o')
    col = col.replace(' ', '_')
    return col

# ==============================================================================
# 5. IN-MEMORY EXTRACTION, CLEANING AND FILTERING (T0 -> Memory)
# ==============================================================================
all_dfs = []

print("Starting Unified ETL Pipeline - SICONFI...")
print(f"1/4 - Reading and cleaning raw data from {min(TARGET_YEARS)} to {max(TARGET_YEARS)}...")

for year in TARGET_YEARS:
    file_path = input_dir / f'finbra_mun_despesas-por-funcao_{year}.csv'
    
    if file_path.exists():
        # 1. Read skipping 3 rows, using original encoding (latin1) and treating decimals
        df = pd.read_csv(file_path, skiprows=3, sep=';', encoding='latin1', decimal=',')
        
        # 2. Standardize column names
        df.columns = [standardize_columns(c) for c in df.columns]
        
        # 3. Filter only the target Column and target Account
        if 'coluna' in df.columns and 'conta' in df.columns:
            mask = (df['coluna'] == TARGET_COLUMN) & (df['conta'].str.contains(TARGET_ACCOUNT, case=False, na=False))
            df_filtered = df[mask].copy()
            
            # 4. Handle numeric columns safely
            for col in ['valor', 'populacao']:
                if col in df_filtered.columns:
                    df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce').fillna(0)
            
            # 5. Calculate Per Capita
            if 'valor' in df_filtered.columns and 'populacao' in df_filtered.columns:
                df_filtered['valor_per_capita'] = 0.0
                pop_mask = df_filtered['populacao'] > 0
                df_filtered.loc[pop_mask, 'valor_per_capita'] = df_filtered.loc[pop_mask, 'valor'] / df_filtered.loc[pop_mask, 'populacao']
                
                # 6. Standardize IBGE code column name and keep only what matters
                df_filtered = df_filtered.rename(columns={'cod.ibge': 'cd_mun'})
                
                # Append to the main list
                all_dfs.append(df_filtered[['cd_mun', 'valor_per_capita']])
                print(f"  ✓ Processed: {year} | Rows extracted: {len(df_filtered)}")
        else:
            print(f"  ⚠️ Missing required columns ('coluna' or 'conta') in {year}.")
    else:
        print(f"  ⚠️ Warning: Raw file not found for the year {year}")

# Average annual per capita across available years
# (mean is robust to missing years; sum would penalize municipalities with fewer data points)
print("  -> Aggregating mean annual per capita per municipality...")
df_siconfi = pd.concat(all_dfs).groupby('cd_mun')['valor_per_capita'].mean().reset_index()

# Rename dynamically using the col_abs variable
df_siconfi.rename(columns={'valor_per_capita': col_abs}, inplace=True)

# ==============================================================================
# 6. H3 LOADING AND MERGING
# ==============================================================================
print("2/4 - Loading H3 base and merging data...")
df_h3 = pd.read_parquet(h3_path)

# Standardize join key to 7-digit string (IBGE standard)
# Siconfi sometimes exports 6-digit codes (without check digit) — pad to 7 if needed
df_siconfi['cd_mun'] = df_siconfi['cd_mun'].astype(str).str.strip()
df_h3['cd_mun']      = df_h3['cd_mun'].astype(str).str.strip()

siconfi_len = df_siconfi['cd_mun'].str.len().mode()[0]
h3_len      = df_h3['cd_mun'].str.len().mode()[0]
if siconfi_len != h3_len:
    print(f"  ⚠️ cd_mun length mismatch: Siconfi={siconfi_len} digits, H3={h3_len} digits — truncating to {min(siconfi_len, h3_len)}.")
    trunc = min(siconfi_len, h3_len)
    df_siconfi['cd_mun'] = df_siconfi['cd_mun'].str[:trunc]
    df_h3['cd_mun']      = df_h3['cd_mun'].str[:trunc]

# Join between H3 and Siconfi (filling NA dynamically)
df_final = df_h3.merge(df_siconfi, on='cd_mun', how='left').fillna({col_abs: 0})

n_matched = df_final[col_abs].gt(0).sum()
print(f"   Hexágonos com investimento > 0: {n_matched:,} / {len(df_final):,}")

# ==============================================================================
# 7. OUTLIER TREATMENT AND NORMALIZATION
# ==============================================================================
print("3/4 - Treating outliers and normalizing...")

# log1p compresses the right tail (most municipalities spend little; a few spend a lot)
df_final[col_log]  = np.log1p(df_final[col_abs])
df_final[col_norm] = utils.normalize_minmax(df_final[col_log], winsorize=True)
df_final = df_final.drop(columns=[col_log])

# ==============================================================================
# 8. EXPORT (Parquet)
# ==============================================================================
df_export = df_final[['h3_id', col_abs, col_norm]]
df_export.to_parquet(output_path, index=False)

print(f"  ✓ File successfully saved at: {output_path.name}")

# ==============================================================================
# 9. DIAGNOSTICS AND EXPORT (.txt)
# ==============================================================================
print("4/4 - Generating diagnostic file...")

with open(DIAGNOSTIC_TXT, 'w', encoding='utf-8') as f:
    f.write("=== PARQUET FILES DIAGNOSTIC (H3 Level) ===\n")
    f.write("Note: The counts below reflect the number of hexagons, not municipalities.\n\n")
    
    f.write(f"--- G1 (Siconfi Investment: {TARGET_ACCOUNT}) ---\n")
    f.write(f"Period analyzed: {min(TARGET_YEARS)} to {max(TARGET_YEARS)}\n")
    f.write(f"File generated: {output_path.name}\n\n")
    
    # Uses the dynamic column list for the log
    value_columns = [col_abs, col_norm]
    
    for col in value_columns:
        nulls = df_export[col].isna().sum()
        
        if 'norm' in col:
            ones = (df_export[col] == 1).sum()
            zeros = (df_export[col] == 0).sum()
            f.write(f"Column: {col}\n")
            f.write(f"  > Exact value 1 (Maximum): {ones}\n")
            f.write(f"  > Exact value 0 (Minimum): {zeros}\n")
            f.write(f"  > Distribution mean: {df_export[col].mean():.4f}\n")
            f.write(f"  > Null Values (NaN): {nulls}\n")
        
        if 'abs' in col:
            f.write(f"Column: {col}\n")
            f.write(f"  > Investment mean (BRL per capita): {df_export[col].mean():.2f}\n")
            f.write(f"  > Median: {df_export[col].median():.2f}\n")
            f.write(f"  > Maximum (post-winsorizing): {df_export[col].max():.2f}\n")
            f.write(f"  > Minimum: {df_export[col].min():.2f}\n")
            f.write(f"  > Null Values (NaN): {nulls}\n")
            
    f.write("-" * 50 + "\n")

print(f"\n✅ Pipeline completed successfully! Diagnostic saved at: {DIAGNOSTIC_TXT}")