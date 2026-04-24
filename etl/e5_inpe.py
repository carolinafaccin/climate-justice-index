import pandas as pd
import numpy as np
import h3
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.append(PROJECT_ROOT)
from src import config as cfg
from src import utils

# ==============================================================================
# 1. PATHS AND SETTINGS
# ==============================================================================
QUEIMADAS_DIR = cfg.RAW_DIR / cfg.INDICATORS["e5"]["source"]["dir"]
ANOS          = list(range(2016, 2026))   # 2016–2025
K_RING        = 4                         # ~1 km buffer em H3 res9

H3_RES = cfg.H3_RES

now = datetime.now().strftime("%Y%m%d_%H%M%S")
DIAGNOSTIC_TXT = cfg.DIAGNOSE_DIR / f"diagnostic_h3_e5_queimadas_{now}.txt"

col_e5_norm = cfg.COLUMN_MAP["e5"]
col_e5_abs  = col_e5_norm.replace("_norm", "_abs")

# Detect h3-py version once
try:
    h3.latlng_to_cell(0.0, 0.0, H3_RES)
    def _to_cell(lat, lon): return h3.latlng_to_cell(lat, lon, H3_RES)
    def _k_ring(cell, k):   return h3.grid_disk(cell, k)
except AttributeError:
    def _to_cell(lat, lon): return h3.geo_to_h3(lat, lon, H3_RES)
    def _k_ring(cell, k):   return h3.k_ring(cell, k)


# ==============================================================================
# 2. MAIN
# ==============================================================================
def main():
    print("=" * 60)
    print("ETL: INPE Queimadas — E5 (Proximidade a focos de queimadas)")
    print(f"Anos: {ANOS[0]}–{ANOS[-1]}  |  k-ring: {K_RING} (~1 km)")
    print("=" * 60)

    # Load H3 base
    print("\n1/4 - Carregando malha H3 base...")
    df_h3 = pd.read_parquet(cfg.FILES_H3["base_metadata"], columns=["h3_id"])
    all_h3 = set(df_h3["h3_id"].values)
    print(f"   {len(all_h3):,} hexágonos na malha base.")

    # For each year, find exposed hexagons
    print("\n2/4 - Processando focos por ano...")
    year_exposure = {}   # {ano: set of exposed h3_ids}

    for ano in ANOS:
        path = QUEIMADAS_DIR / f"{ano}.csv"
        if not path.exists():
            print(f"   [{ano}] AVISO: arquivo não encontrado — {path.name}")
            continue

        df = pd.read_csv(path, usecols=["latitude", "longitude"], low_memory=False)
        df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df = df.dropna(subset=["latitude", "longitude"])

        # Assign H3 cell to each fire hotspot
        fire_cells = set(
            _to_cell(float(lat), float(lon))
            for lat, lon in zip(df["latitude"], df["longitude"])
        )

        # Expand each fire cell with k-ring to cover ~1 km buffer
        exposed = set()
        for cell in fire_cells:
            exposed.update(_k_ring(cell, K_RING))

        # Intersect with the H3 base grid (Brazil only)
        exposed_in_base = exposed & all_h3
        year_exposure[ano] = exposed_in_base
        print(f"   [{ano}] {len(df):,} focos → {len(fire_cells):,} células únicas → "
              f"{len(exposed_in_base):,} hexágonos expostos")

    n_years = len(year_exposure)
    if n_years == 0:
        raise RuntimeError("Nenhum arquivo de queimadas encontrado. Verifique o caminho.")

    # Aggregate: count how many years each hexagon was exposed
    print(f"\n3/4 - Agregando exposição ({n_years} anos)...")
    exposure_count = pd.Series(0, index=sorted(all_h3), dtype=np.int16)
    for ano, exposed_set in year_exposure.items():
        in_index = exposure_count.index.isin(exposed_set)
        exposure_count[in_index] += 1

    df_agg = exposure_count.reset_index()
    df_agg.columns = ["h3_id", "anos_expostos"]

    # e5_abs = fraction of years exposed (0–1), Opção B (média anual)
    df_agg[col_e5_abs]  = df_agg["anos_expostos"] / n_years
    df_agg[col_e5_norm] = utils.normalize_minmax(df_agg[col_e5_abs], winsorize=True)

    # Merge with H3 base and save
    print("4/4 - Salvando parquet...")
    df_final = df_h3.merge(df_agg[["h3_id", col_e5_abs, col_e5_norm]], on="h3_id", how="left")
    utils.save_parquet(df_final, cfg.FILES_H3["e5"])
    print(f"   ✓ Salvo: {cfg.FILES_H3['e5'].name}")

    _write_diagnostic(df_agg, df_final, n_years, year_exposure)
    print(f"\nDiagnóstico: {DIAGNOSTIC_TXT}")
    print("Concluído!")


def _write_diagnostic(df_agg, df_final, n_years, year_exposure):
    with open(DIAGNOSTIC_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("INPE Queimadas ETL Diagnostic\n")
        f.write(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Diretório queimadas : {QUEIMADAS_DIR}\n")
        f.write(f"Anos processados    : {sorted(year_exposure.keys())}\n")
        f.write(f"k-ring              : {K_RING}\n\n")

        f.write("Hexágonos expostos por ano:\n")
        for ano in sorted(year_exposure):
            f.write(f"  {ano}: {len(year_exposure[ano]):,}\n")

        f.write(f"\nDistribuição de anos_expostos (0–{n_years}):\n")
        dist = df_agg["anos_expostos"].value_counts().sort_index()
        for v, c in dist.items():
            f.write(f"  {v:2d} anos: {c:,} hexágonos\n")

        f.write(f"\n--- {col_e5_abs} ---\n")
        s = df_final[col_e5_abs].dropna()
        f.write(f"  mean   = {s.mean():.6f}\n  median = {s.median():.6f}\n"
                f"  min    = {s.min():.6f}\n  max    = {s.max():.6f}\n"
                f"  zeros  = {(s == 0).sum():,}\n  nulls  = {df_final[col_e5_abs].isna().sum():,}\n")

        f.write(f"\n--- {col_e5_norm} ---\n")
        s = df_final[col_e5_norm].dropna()
        f.write(f"  mean   = {s.mean():.6f}\n  median = {s.median():.6f}\n"
                f"  min    = {s.min():.6f}\n  max    = {s.max():.6f}\n"
                f"  zeros  = {(s == 0).sum():,}\n  nulls  = {df_final[col_e5_norm].isna().sum():,}\n")


if __name__ == "__main__":
    main()
