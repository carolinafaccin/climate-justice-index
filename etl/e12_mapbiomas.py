import pandas as pd
import numpy as np
import rasterio
import h3
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
RASTER_LANDSLIDES = cfg.RAW_DIR / cfg.INDICATORS["e1"]["source"]["file"]
RASTER_FLOODS     = cfg.RAW_DIR / cfg.INDICATORS["e2"]["source"]["file"]
CNEFE_DIR         = cfg.RAW_DIR / cfg.INDICATORS["e1"]["source"]["cnefe_dir"]
CACHE_DIR         = CNEFE_DIR

H3_RES   = cfg.H3_RES
CHUNK_SZ = 500_000

now = datetime.now().strftime("%Y%m%d_%H%M%S")
DIAGNOSTIC_TXT = cfg.DIAGNOSE_DIR / f"diagnostic_h3_e12_mapbiomas_{now}.txt"

col_e1_norm = cfg.COLUMN_MAP["e1"]
col_e1_abs  = col_e1_norm.replace("_norm", "_abs")
col_e2_norm = cfg.COLUMN_MAP["e2"]
col_e2_abs  = col_e2_norm.replace("_norm", "_abs")

USECOLS = ['latitude', 'longitude', 'cod_especie']

# Detect h3-py version once
try:
    h3.latlng_to_cell(0.0, 0.0, H3_RES)
    def _to_cell(lat, lon): return h3.latlng_to_cell(lat, lon, H3_RES)
except AttributeError:
    def _to_cell(lat, lon): return h3.geo_to_h3(lat, lon, H3_RES)


# ==============================================================================
# 2. HELPERS
# ==============================================================================
def _sample_risk(src, nodata, lons, lats):
    """Point-sample raster at address coordinates; returns int8 array (1=risk, 0=no risk)."""
    coords = list(zip(lons, lats))
    vals = np.array([v[0] for v in src.sample(coords)], dtype=float)
    if nodata is not None:
        vals[vals == nodata] = np.nan
    return (np.nan_to_num(vals, nan=0.0) > 0).astype(np.int8)


def _read_cnefe_chunk(path):
    """Yields filtered chunks from a CNEFE state CSV."""
    try:
        reader = pd.read_csv(
            path, sep=';', usecols=USECOLS,
            dtype={'cod_especie': 'Int64'},
            chunksize=CHUNK_SZ, low_memory=False
        )
    except Exception:
        reader = pd.read_csv(
            path, sep=';', usecols=USECOLS, encoding='latin1',
            dtype={'cod_especie': 'Int64'},
            chunksize=CHUNK_SZ, low_memory=False
        )
    for chunk in reader:
        df = chunk[chunk['cod_especie'] == 1].copy()
        if df.empty:
            continue
        df['latitude']  = pd.to_numeric(df['latitude'],  errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df = df.dropna(subset=['latitude', 'longitude'])
        if not df.empty:
            yield df


# ==============================================================================
# 3. MAIN
# ==============================================================================
def main():
    print("=" * 60)
    print("ETL: MapBiomas — E1 (Deslizamentos) + E2 (Inundações)")
    print("Fonte de domicílios: CNEFE 2022")
    print("=" * 60)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(CNEFE_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nenhum CSV encontrado em: {CNEFE_DIR}")
    print(f"\nArquivos CNEFE encontrados: {len(csv_files)}")

    agg_parts = []
    total_dom = 0

    with rasterio.open(RASTER_LANDSLIDES) as src_l, rasterio.open(RASTER_FLOODS) as src_f:
        nodata_l = src_l.nodata
        nodata_f = src_f.nodata

        for csv_path in csv_files:
            cache_path = CACHE_DIR / f"{csv_path.stem}.parquet"

            if cache_path.exists():
                print(f"  → {csv_path.name}  [cache]")
                agg_parts.append(pd.read_parquet(cache_path))
                continue

            print(f"  → {csv_path.name}", end="", flush=True)
            n_state = 0
            state_parts = []

            for df in _read_cnefe_chunk(csv_path):
                lats = df['latitude'].values
                lons = df['longitude'].values

                df['in_l'] = _sample_risk(src_l, nodata_l, lons, lats)
                df['in_f'] = _sample_risk(src_f, nodata_f, lons, lats)
                df['h3_id'] = [_to_cell(float(la), float(lo)) for la, lo in zip(lats, lons)]

                part = df.groupby('h3_id', sort=False).agg(
                    total=('in_l', 'count'),
                    l_sum=('in_l', 'sum'),
                    f_sum=('in_f', 'sum')
                )
                state_parts.append(part)
                n_state += len(df)

            if state_parts:
                state_agg = pd.concat(state_parts).groupby('h3_id').sum()
                state_agg.to_parquet(cache_path)
                agg_parts.append(state_agg)

            total_dom += n_state
            print(f"  {n_state:,} domicílios")

    # Final aggregation
    print(f"\nTotal de domicílios processados: {total_dom:,}")
    print("Agregando por hexágono H3...")
    df_agg = pd.concat(agg_parts).groupby('h3_id').sum().reset_index()
    print(f"Hexágonos com ao menos 1 domicílio: {len(df_agg):,}")

    df_agg[col_e1_abs] = df_agg['l_sum'] / df_agg['total']
    df_agg[col_e2_abs] = df_agg['f_sum'] / df_agg['total']
    # Proporções já limitadas em [0,1]: winsorize=False evita colapso em indicadores
    # esparsos onde P99 cai em 0 (ex: deslizamentos afetam < 1% dos hexágonos)
    df_agg[col_e1_norm] = utils.normalize_minmax(df_agg[col_e1_abs], winsorize=False)
    df_agg[col_e2_norm] = utils.normalize_minmax(df_agg[col_e2_abs], winsorize=False)

    # Merge with H3 base and save
    print("Mesclando com malha H3 base...")
    df_h3 = pd.read_parquet(cfg.FILES_H3['base_metadata'], columns=['h3_id'])

    df_e1 = df_h3.merge(df_agg[['h3_id', col_e1_abs, col_e1_norm]], on='h3_id', how='left')
    df_e2 = df_h3.merge(df_agg[['h3_id', col_e2_abs, col_e2_norm]], on='h3_id', how='left')

    utils.save_parquet(df_e1, cfg.FILES_H3['e1'])
    utils.save_parquet(df_e2, cfg.FILES_H3['e2'])
    print(f"  ✓ Salvo: {cfg.FILES_H3['e1'].name}")
    print(f"  ✓ Salvo: {cfg.FILES_H3['e2'].name}")

    _write_diagnostic(df_agg, df_e1, df_e2, total_dom)
    print(f"\nDiagnóstico: {DIAGNOSTIC_TXT}")
    print("Concluído!")


def _write_diagnostic(df_agg, df_e1, df_e2, total_dom):
    with open(DIAGNOSTIC_TXT, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("MapBiomas ETL Diagnostic\n")
        f.write(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Raster deslizamentos : {RASTER_LANDSLIDES}\n")
        f.write(f"Raster inundações    : {RASTER_FLOODS}\n")
        f.write(f"CNEFE dir            : {CNEFE_DIR}\n\n")
        f.write(f"Total domicílios processados     : {total_dom:,}\n")
        f.write(f"Hexágonos com ≥1 domicílio       : {len(df_agg):,}\n\n")

        for label, col_abs, col_norm, df_ind in [
            ("E1 - Deslizamentos", col_e1_abs, col_e1_norm, df_e1),
            ("E2 - Inundações",    col_e2_abs, col_e2_norm, df_e2),
        ]:
            f.write(f"--- {label} ---\n")
            for col in [col_abs, col_norm]:
                s = df_ind[col].dropna()
                f.write(
                    f"  {col}:\n"
                    f"    mean   = {s.mean():.6f}\n"
                    f"    median = {s.median():.6f}\n"
                    f"    min    = {s.min():.6f}\n"
                    f"    max    = {s.max():.6f}\n"
                    f"    zeros  = {(s == 0).sum():,}\n"
                    f"    nulls  = {df_ind[col].isna().sum():,}\n"
                )
            f.write("\n")


if __name__ == "__main__":
    main()
