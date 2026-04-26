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
# 2. BUSINESS RULES (CENSUS LOGIC) — derived from indicators.json
# ==============================================================================
CENSUS_LOGIC = {
    k: v["source"]
    for k, v in cfg.INDICATORS.items()
    if "source" in v and "source_dir" in v["source"]
}

# Collect all raw variable names referenced directly from indicators.json
# v06004_v06001 is a computed column (product), not present in CSVs — exclude it and include v06004 instead
_all_refs = {col for logic in CENSUS_LOGIC.values() for col in logic["num_cols"] + logic["den_cols"]}
REQUIRED_RAW_VARS = sorted((_all_refs - {"v06004_v06001"}) | {"v06004"})

# ==============================================================================
# 3. PATHS — source_dir read from first census source entry in indicators.json
# ==============================================================================
_source_dir = next(v["source"]["source_dir"] for v in cfg.INDICATORS.values() if "source" in v and "source_dir" in v["source"])
input_dir = cfg.RAW_DIR / _source_dir
h3_path = cfg.FILES_H3["base_metadata"]
now = datetime.now().strftime("%Y%m%d_%H%M%S")
DIAGNOSTIC_TXT = cfg.DIAGNOSE_DIR / f'diagnostic_h3_censo2022_{now}.txt'

# ==============================================================================
# 4. DATA EXTRACTION (READING CSVS)
# ==============================================================================
print("\n1/4 - Reading and consolidating raw CSVs...")
all_csvs = list(input_dir.glob('*.csv'))
df_censo = pd.DataFrame(columns=['cd_setor'])

for f in all_csvs:
    try:
        with open(f, 'r', encoding='utf-8', errors='ignore') as tf:
            line = tf.readline()
            sep = ';' if ';' in line else ','
        
        df_temp = pd.read_csv(f, sep=sep, dtype={'cd_setor': str}, encoding='utf-8')
        df_temp.columns = df_temp.columns.str.lower()
        
        if 'cd_setor' not in df_temp.columns: continue
            
        # BLINDAGEM: Pega apenas variáveis que ainda não estão no df_censo
        cols_already_have = df_censo.columns.tolist()
        vars_to_extract = [c for c in REQUIRED_RAW_VARS if c in df_temp.columns and c not in cols_already_have]
        
        if vars_to_extract:
            df_temp = df_temp[['cd_setor'] + vars_to_extract]
            df_censo = pd.merge(df_censo, df_temp, on='cd_setor', how='outer')
            print(f"  OK: {f.name} ({len(vars_to_extract)} new variables)")
                
    except Exception as e:
        print(f"  ERRO: {f.name}: {e}")

# v06004 usa vírgula como separador decimal no arquivo do IBGE — converte antes do merge
# v06004_v06001 = produto usado no numerador da média ponderada de renda: Σ(renda×responsáveis×peso)/Σ(responsáveis×peso)
if 'v06004' in df_censo.columns:
    df_censo['v06004'] = pd.to_numeric(df_censo['v06004'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
    if 'v06001' in df_censo.columns:
        df_censo['v06004_v06001'] = df_censo['v06004'] * df_censo['v06001']

print(f"Consolidation completed! Master shape: {df_censo.shape}")

# ==============================================================================
# 5. MERGE AND DASYMETRIC WEIGHTING
# ==============================================================================
print("2/4 - Merging with H3 grid and applying weights...")
df_h3 = pd.read_parquet(h3_path)

df_censo['cd_setor'] = df_censo['cd_setor'].astype(str)
df_h3['cd_setor'] = df_h3['cd_setor'].astype(str)

df_merged = pd.merge(df_h3[['h3_id', 'cd_setor', 'peso_dom']], df_censo, on='cd_setor', how='inner')

# Identifica todas as colunas numéricas que precisam de peso
columns_to_weight = [c for c in df_merged.columns if c in REQUIRED_RAW_VARS or c == 'v06004_v06001']

for col in columns_to_weight:
    df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce').fillna(0)
    df_merged[col] = df_merged[col] * df_merged['peso_dom']

print("3/4 - Aggregating values per H3 Hexagon...")
# LINHA ADICIONADA: Sem isso a Seção 6 quebra
df_hex = df_merged.groupby('h3_id')[columns_to_weight].sum().reset_index()

# ==============================================================================
# 6. INDICATORS CALCULATION
# ==============================================================================
print("4/4 - Calculating ratios and exporting...")
generated_files = {}

for ind_key, logic in CENSUS_LOGIC.items():
    col_norm = cfg.COLUMN_MAP.get(ind_key)
    if not col_norm: continue
        
    col_abs = col_norm.replace('_norm', '_abs')
    
    # Razão
    num = df_hex[[c for c in logic['num_cols'] if c in df_hex.columns]].sum(axis=1)
    den = df_hex[[c for c in logic['den_cols'] if c in df_hex.columns]].sum(axis=1)
    
# BLINDAGEM: Trata divisão por zero e limita o teto em 1.0 (100%)
    with np.errstate(divide='ignore', invalid='ignore'):
        abs_val = num / den
        abs_val = abs_val.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Normalização (com inversão opcional para indicadores onde alto = menos vulnerável)
    norm_val = utils.normalize_minmax(abs_val, winsorize=True)
    if logic.get("invert_norm", False):
        norm_val = 1.0 - norm_val

    df_export = pd.DataFrame({'h3_id': df_hex['h3_id'], col_abs: abs_val, col_norm: norm_val})
    
    out_path = cfg.FILES_H3[ind_key]
    utils.save_parquet(df_export, out_path)
    generated_files[ind_key] = (out_path.name, df_export, col_abs, col_norm)
    print(f"  Saved: {ind_key.upper()}")

# ==============================================================================
# 7. EXPORT DIAGNOSTIC LOG (.txt)
# ==============================================================================
print("Generating diagnostic report...")

with open(DIAGNOSTIC_TXT, 'w', encoding='utf-8') as f:
    f.write("=== CENSUS 2022 PARQUET DIAGNOSTIC (H3 Level) ===\n")
    f.write("Note: Dasymetric interpolation successfully applied.\n\n")
    
    for ind_key, (file_name, df_diag, col_abs, col_norm) in generated_files.items():
        f.write(f"--- {ind_key.upper()} : {file_name} ---\n")
        
        # Absolute stats
        f.write(f"Column (Absolute/Percentage): {col_abs}\n")
        f.write(f"  > Mean: {df_diag[col_abs].mean():.4f}\n")
        f.write(f"  > Median: {df_diag[col_abs].median():.4f}\n")
        f.write(f"  > Maximum: {df_diag[col_abs].max():.4f}\n")
        f.write(f"  > Minimum: {df_diag[col_abs].min():.4f}\n\n")

        # Normalized stats
        f.write(f"Column (Normalized): {col_norm}\n")
        ones = (df_diag[col_norm] >= 0.999).sum()
        zeros = (df_diag[col_norm] <= 0.001).sum()
        f.write(f"  > Extreme Value (~1): {ones} hexagons\n")
        f.write(f"  > Extreme Value (~0): {zeros} hexagons\n")
        f.write(f"  > Mean: {df_diag[col_norm].mean():.4f}\n")
        f.write("-" * 50 + "\n")

print(f"\nCensus ETL Pipeline completed! Diagnostic saved at: {DIAGNOSTIC_TXT}")