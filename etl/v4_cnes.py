import pandas as pd
import numpy as np
import h3
import geopandas as gpd
from scipy.spatial import cKDTree
import sys
from pathlib import Path
from datetime import datetime

"""
Modelo Gravitacional de Acessibilidade a Saúde (v4 — CNES).

Mede a "força de atração" da infraestrutura de saúde sobre cada hexágono.
Estabelecimentos maiores (mais serviços) atraem de mais longe; quanto maior
a distância, menor a influência do estabelecimento sobre o hexágono.

Fórmula:
    v4_sau_abs = Σ [ capacity_score_j / (distância_j_metros + 100) ]

    O buffer de 100m evita divisão por zero quando o hexágono coincide com
    o estabelecimento. Considera as 3 unidades mais próximas (k=3).

Direção do indicador:
    v4_sau_abs alto  →  boa acessibilidade  →  MENOS vulnerável.
    Para ser consistente com os demais indicadores de vulnerabilidade
    (onde alto = mais vulnerável), o col_norm é INVERTIDO: 1 − norm.
    Assim, col_norm alto = inacessibilidade alta = mais vulnerável.
"""

# ==============================================================================
# 1. ENVIRONMENT CONFIGURATION
# ==============================================================================
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.append(PROJECT_ROOT)

from src import config as cfg
from src import utils

# ==============================================================================
# 2. PATHS, COLUMNS AND DIAGNOSTIC DEFINITION
# ==============================================================================
# Input paths (Pulled directly from config.py)
h3_path = cfg.FILES_H3["base_metadata"]
cnes_path = cfg.RAW_DIR / cfg.INDICATORS["v4"]["source"]["file"]

# Output path
output_path = cfg.FILES_H3["v4"]

# Dynamic column names (Pulled straight from indicators.json via config)
# If col_norm is 'v4_sau_norm', the code automatically creates 'v4_sau_abs' and 'v4_sau_log'
col_norm = cfg.COLUMN_MAP["v4"]
col_abs = col_norm.replace('_norm', '_abs')
col_log = col_norm.replace('_norm', '_log')

# Diagnostic log configuration
now = datetime.now().strftime("%Y%m%d_%H%M%S")
DIAGNOSTIC_TXT = cfg.DIAGNOSE_DIR / f'diagnostic_h3_v4_cnes_{now}.txt'

# ==============================================================================
# BUSINESS PARAMETERS
# ==============================================================================
CNES_SERVICES = [
    'st_centro_cirurgico', 'st_centro_obstetrico', 'st_centro_neonatal',
    'st_atend_hospitalar', 'st_servico_apoio', 'st_atend_ambulatorial'
]
N_NEAREST_FACILITIES = 3   # k in kd-tree query
DISTANCE_BUFFER_M    = 100 # added to distance to avoid division by zero

# =====================================================================
# 3. DATA LOADING AND STANDARDIZATION
# =====================================================================
print("Starting ETL Pipeline - V4 CNES...")
print("1/5 - Loading data...")

df_h3 = pd.read_parquet(h3_path)
df_h3.columns = df_h3.columns.str.lower()

if not cnes_path.exists():
    raise FileNotFoundError(f"⚠️ Error: CNES file not found at: {cnes_path}")

df_cnes = pd.read_csv(cnes_path, sep=';', low_memory=False, encoding='latin1')
df_cnes.columns = df_cnes.columns.str.lower()

# =====================================================================
# 4. CNES PROCESSING (COORDINATES AND CAPACITY)
# =====================================================================
print("2/5 - Cleaning coordinates and calculating Capacity Score...")
df_cnes['latitude'] = pd.to_numeric(df_cnes['latitude'], errors='coerce')
df_cnes['longitude'] = pd.to_numeric(df_cnes['longitude'], errors='coerce')
df_cnes = df_cnes.dropna(subset=['latitude', 'longitude'])

for col in CNES_SERVICES:
    if col in df_cnes.columns:
        df_cnes[col] = pd.to_numeric(df_cnes[col], errors='coerce').fillna(0)

# Facility Capacity Score (Weight)
df_cnes['capacity_score'] = df_cnes[CNES_SERVICES].sum(axis=1) + 1

# =====================================================================
# 5. SPATIAL PREPARATION (CONVERSION TO METERS - EPSG:5880)
# =====================================================================
print("3/5 - Converting coordinates to metric system (SIRGAS 2000)...")

# Transforms CNES into GeoDataFrame and converts to meters
gdf_cnes = gpd.GeoDataFrame(
    df_cnes, 
    geometry=gpd.points_from_xy(df_cnes.longitude, df_cnes.latitude),
    crs="EPSG:4326"
).to_crs("EPSG:5880")

def get_h3_centroid(h3_id):
    try:
        return h3.cell_to_latlng(h3_id) if hasattr(h3, 'cell_to_latlng') else h3.h3_to_geo(h3_id)
    except:
        return (np.nan, np.nan)

centroids = df_h3['h3_id'].apply(get_h3_centroid)
df_h3['lat'] = [c[0] for c in centroids]
df_h3['lng'] = [c[1] for c in centroids]

gdf_h3 = gpd.GeoDataFrame(
    df_h3,
    geometry=gpd.points_from_xy(df_h3.lng, df_h3.lat),
    crs="EPSG:4326"
).to_crs("EPSG:5880")

# =====================================================================
# 6. EUCLIDEAN DISTANCE AND GRAVITATIONAL MODEL CALCULATION
# =====================================================================
print("4/5 - Calculating Euclidean Distance and Gravitational Score...")

coords_cnes = np.array(list(zip(gdf_cnes.geometry.x, gdf_cnes.geometry.y)))
coords_h3 = np.array(list(zip(gdf_h3.geometry.x, gdf_h3.geometry.y)))
capacities = gdf_cnes['capacity_score'].values

tree = cKDTree(coords_cnes)
distances, indices = tree.query(coords_h3, k=N_NEAREST_FACILITIES)
gravitational_weights = capacities[indices] / (distances + DISTANCE_BUFFER_M)

# Uses the dynamic column name!
df_h3[col_abs] = np.sum(gravitational_weights, axis=1)

df_h3 = df_h3.drop(columns=['lat', 'lng', 'geometry'], errors='ignore')

# =====================================================================
# 7. LOGARITHMIC FUNCTION TREATMENT AND NORMALIZATION
# =====================================================================
print("5/5 - Applying logarithmic function, normalizing and saving...")

# Applies logarithmic function to flatten giant outliers
df_h3[col_log] = np.log1p(df_h3[col_abs])

# Invert: high accessibility → low vulnerability score (consistent with other v* indicators)
df_h3[col_norm] = 1.0 - utils.normalize_minmax(df_h3[col_log], winsorize=True)

# Remove the intermediate log column
df_h3 = df_h3.drop(columns=[col_log])

# =====================================================================
# 8. SAVE FINAL FILE
# =====================================================================
df_export = df_h3[['h3_id', col_abs, col_norm]]
df_export.to_parquet(output_path, index=False)

print(f"  ✓ Parquet file successfully saved at: {output_path.name}")

# =====================================================================
# 9. DATA DIAGNOSTIC (.txt)
# =====================================================================
print("Generating diagnostic file...")

with open(DIAGNOSTIC_TXT, 'w', encoding='utf-8') as f:
    f.write("="*50 + "\n")
    f.write("GRAVITATIONAL MODEL AND DISTANCE DIAGNOSTIC (V4 - CNES)\n")
    f.write("="*50 + "\n\n")
    
    f.write(f"Total CNES facilities processed: {len(df_cnes):,}\n")
    f.write(f"Total H3 hexagons processed: {len(df_h3):,}\n\n")
    
    min_distance = distances[:, 0]
    
    f.write("--- DISTANCE TO NEAREST FACILITY ---\n")
    f.write(f"Average in Brazil: {np.mean(min_distance)/1000:.2f} km\n")
    f.write(f"Median (50% of Brazil is within): {np.median(min_distance)/1000:.2f} km\n")
    f.write(f"Maximum Distance (Most isolated location): {np.max(min_distance)/1000:.2f} km\n\n")
    
    f.write(f"--- NORMALIZED VARIABLE STATISTICS ({col_norm}) ---\n")
    f.write(df_h3[col_norm].describe().to_string() + "\n\n")
    
    cols_show = ['h3_id', col_abs, col_norm]
    if 'nm_mun' in df_h3.columns:
        cols_show.insert(1, 'nm_mun')
    if 'nm_uf' in df_h3.columns:
        cols_show.insert(2, 'nm_uf')
        
    f.write("--- TOP 5 HEXAGONS WITH HIGHEST ACCESSIBILITY (Urban Centers) ---\n")
    f.write(df_h3.sort_values(by=col_norm, ascending=False)[cols_show].head().to_string() + "\n\n")
    
    f.write("--- BOTTOM 5 HEXAGONS WITH LOWEST ACCESSIBILITY (Isolated Locations) ---\n")
    f.write(df_h3.sort_values(by=col_norm, ascending=True)[cols_show].head().to_string() + "\n")

print(f"✅ Diagnostic saved at: {DIAGNOSTIC_TXT}")