"""
ETL: Landsat NDVI / GEE → Indicator v6 (cobertura vegetal).

Input:  cfg.RAW_DIR / cfg.INDICATORS["v6"]["source"]["dir"]
        CSVs do GEE com colunas h3_id, ndvi_mean, qtd_dom (um por UF)
Output: cfg.FILES_H3["v6"]  →  br_h3_v6_areas_verdes.parquet

Lógica: NDVI alto → baixa vulnerabilidade. O indicador final é o inverso do
NDVI normalizado: v6_ver_norm = 1 − normalize(ndvi_mean). Hexágonos com
menor cobertura vegetal recebem pontuação maior (maior vulnerabilidade).
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
# 0. CONFIGURAÇÃO
# ==============================================================================
GEE_DIR = cfg.RAW_DIR / cfg.INDICATORS["v6"]["source"]["dir"]

now = datetime.now().strftime("%Y%m%d_%H%M%S")
DIAGNOSTIC_TXT = cfg.DIAGNOSE_DIR / f"diagnostic_h3_v6_areas_verdes_{now}.txt"

col_v6_norm = cfg.COLUMN_MAP["v6"]          # "v6_ver_norm"
col_v6_abs  = col_v6_norm.replace("_norm", "_abs")  # "v6_ver_abs"
COL_NDVI_RAW = "ndvi_mean"


# ==============================================================================
# 1. MAIN
# ==============================================================================
def main():
    print("=" * 60)
    print("ETL: GEE Landsat — V6 (Cobertura Vegetal / NDVI, 2020–2024)")
    print(f"Source: {GEE_DIR}")
    print("=" * 60)

    # --------------------------------------------------------------------------
    # 1/4 — Carregar CSVs por UF
    # --------------------------------------------------------------------------
    print("\n1/4 - Carregando CSVs do GEE...")
    csv_files = sorted(GEE_DIR.glob("h3_areas_verdes_ndvi_uf_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nenhum CSV encontrado em: {GEE_DIR}")
    print(f"   Arquivos encontrados: {len(csv_files)}")

    parts = []
    for path in csv_files:
        try:
            df = pd.read_csv(path, usecols=["h3_id", COL_NDVI_RAW, "qtd_dom"])
        except ValueError:
            df = pd.read_csv(path)
            df.columns = df.columns.str.lower()
            df = df[["h3_id", COL_NDVI_RAW, "qtd_dom"]]
        parts.append(df)
        print(f"   ✓ {path.name}  ({len(df):,} hexágonos)")

    df_all = pd.concat(parts, ignore_index=True)
    print(f"\n   Total: {len(df_all):,} hexágonos carregados")

    dupes = df_all["h3_id"].duplicated().sum()
    if dupes > 0:
        print(f"   AVISO: {dupes:,} h3_ids duplicados — mantendo primeiro valor.")
        df_all = df_all.drop_duplicates(subset="h3_id", keep="first")

    # --------------------------------------------------------------------------
    # 2/4 — Calcular indicador
    # --------------------------------------------------------------------------
    print("\n2/4 - Calculando indicador...")

    df_all[COL_NDVI_RAW] = pd.to_numeric(df_all[COL_NDVI_RAW], errors="coerce")
    df_all["qtd_dom"]    = pd.to_numeric(df_all["qtd_dom"],    errors="coerce").fillna(0)

    n_null = df_all[COL_NDVI_RAW].isna().sum()
    print(f"   Hexágonos sem NDVI (NaN): {n_null:,}")

    # Winsorização p1–p99 (p1 corta sombras/água com NDVI muito negativo)
    p01 = df_all[COL_NDVI_RAW].quantile(0.01)
    p99 = df_all[COL_NDVI_RAW].quantile(0.99)
    ndvi_clipped = df_all[COL_NDVI_RAW].clip(lower=p01, upper=p99)

    ndvi_min = ndvi_clipped.min()
    ndvi_max = ndvi_clipped.max()
    ndvi_norm = (ndvi_clipped - ndvi_min) / (ndvi_max - ndvi_min)

    # Inverte: NDVI alto = baixa vulnerabilidade → 1 − ndvi_norm
    df_all[col_v6_abs] = 1.0 - ndvi_norm

    # Normalização final min-max (garante [0,1] exato após NaNs preservados)
    df_all[col_v6_norm] = utils.normalize_minmax(df_all[col_v6_abs], winsorize=False)

    n_low_green  = (df_all[col_v6_norm] > 0.7).sum()
    n_high_green = (df_all[col_v6_norm] < 0.3).sum()
    print(f"   Baixa cobertura vegetal (v6_norm > 0.7): {n_low_green:,} hexágonos")
    print(f"   Alta cobertura vegetal  (v6_norm < 0.3): {n_high_green:,} hexágonos")
    print(f"   NDVI médio bruto:   {df_all[COL_NDVI_RAW].mean():.4f}")
    print(f"   NDVI mediano bruto: {df_all[COL_NDVI_RAW].median():.4f}")

    # --------------------------------------------------------------------------
    # 3/4 — Merge com grade H3 base
    # --------------------------------------------------------------------------
    print("\n3/4 - Mesclando com grade H3 base...")
    df_h3 = pd.read_parquet(cfg.FILES_H3["base_metadata"], columns=["h3_id"])
    df_final = df_h3.merge(
        df_all[["h3_id", COL_NDVI_RAW, col_v6_abs, col_v6_norm]],
        on="h3_id", how="left"
    )

    n_matched   = df_final[col_v6_norm].notna().sum()
    n_unmatched = df_final[col_v6_norm].isna().sum()
    print(f"   Hexágonos com valor:    {n_matched:,}")
    print(f"   Hexágonos sem match:    {n_unmatched:,}")

    # --------------------------------------------------------------------------
    # 4/4 — Salvar
    # --------------------------------------------------------------------------
    print("\n4/4 - Salvando parquet...")
    utils.save_parquet(df_final, cfg.FILES_H3["v6"])
    print(f"   ✓ Salvo: {cfg.FILES_H3['v6'].name}")

    _write_diagnostic(df_all, df_final, csv_files, p01, p99)
    print(f"\nDiagnóstico: {DIAGNOSTIC_TXT}")

    print("\nGerando figuras de debug...")
    _write_figures(df_final)

    print("Concluído!")


# ==============================================================================
# DIAGNÓSTICO
# ==============================================================================
def _write_diagnostic(df_all, df_final, csv_files, p01, p99):
    with open(DIAGNOSTIC_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("GEE Landsat — V6 Cobertura Vegetal (NDVI) ETL Diagnóstico\n")
        f.write(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Diretório GEE : {GEE_DIR}\n")
        f.write(f"Arquivos lidos: {len(csv_files)}\n\n")

        f.write("--- ndvi_mean (bruto, antes do clip) ---\n")
        s = df_all[COL_NDVI_RAW].dropna()
        f.write(f"  média    = {s.mean():.4f}\n")
        f.write(f"  mediana  = {s.median():.4f}\n")
        f.write(f"  p01      = {p01:.4f}  (limite inferior de winsorização)\n")
        f.write(f"  p99      = {p99:.4f}  (limite superior de winsorização)\n")
        f.write(f"  min      = {s.min():.4f}\n")
        f.write(f"  max      = {s.max():.4f}\n")
        f.write(f"  < 0      = {(s < 0).sum():,} hexágonos (sombras/água)\n")
        f.write(f"  0–0.2    = {((s >= 0) & (s < 0.2)).sum():,} hexágonos (solo exposto/urbano)\n")
        f.write(f"  0.2–0.4  = {((s >= 0.2) & (s < 0.4)).sum():,} hexágonos (vegetação esparsa)\n")
        f.write(f"  ≥ 0.4    = {(s >= 0.4).sum():,} hexágonos (vegetação densa)\n\n")

        f.write("--- qtd_dom ---\n")
        d = df_all["qtd_dom"]
        f.write(f"  soma     = {d.sum():,.0f} domicílios\n")
        f.write(f"  média    = {d.mean():.2f}\n")
        f.write(f"  zeros    = {(d == 0).sum():,} hexágonos\n\n")

        for col in [col_v6_abs, col_v6_norm]:
            s2 = df_final[col].dropna()
            f.write(f"--- {col} ---\n")
            f.write(f"  média    = {s2.mean():.6f}\n")
            f.write(f"  mediana  = {s2.median():.6f}\n")
            f.write(f"  min      = {s2.min():.6f}\n")
            f.write(f"  max      = {s2.max():.6f}\n")
            f.write(f"  zeros    = {(s2 == 0).sum():,}\n")
            f.write(f"  nulos    = {df_final[col].isna().sum():,}\n\n")


# ==============================================================================
# FIGURAS DE DEBUG
# ==============================================================================
def _write_figures(df_final):
    try:
        import matplotlib.pyplot as plt
        import h3
        import geopandas as gpd
        from shapely.geometry import Polygon as ShapelyPolygon
    except ImportError as e:
        print(f"  Figuras ignoradas — dependência ausente: {e}")
        return

    import json as _json

    DEBUG_DIR = cfg.FIGURES_DIR / "debug_areas_verdes"
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    _cities_path = cfg.BASE_DIR / "diagnose" / "cities.json"
    if _cities_path.exists():
        with open(_cities_path, "r", encoding="utf-8") as _f:
            CITIES = _json.load(_f)["cities"]
    else:
        CITIES = [
            {"nm_mun": "São Paulo",      "nm_uf": "São Paulo"},
            {"nm_mun": "Rio de Janeiro", "nm_uf": "Rio de Janeiro"},
            {"nm_mun": "Belo Horizonte", "nm_uf": "Minas Gerais"},
        ]

    df_base = pd.read_parquet(cfg.FILES_H3["base_metadata"])
    df_viz  = df_base.merge(df_final[["h3_id", COL_NDVI_RAW, col_v6_norm]], on="h3_id", how="left")

    def _h3_to_polygon(h3_id):
        try:
            boundary = (
                h3.cell_to_boundary(h3_id)
                if hasattr(h3, "cell_to_boundary")
                else h3.h3_to_geo_boundary(h3_id)
            )
            return ShapelyPolygon([(lng, lat) for lat, lng in boundary])
        except Exception:
            return None

    def _city_map(city_info):
        nm_mun = city_info["nm_mun"]
        nm_uf  = city_info.get("nm_uf")
        label  = nm_mun + (f" ({nm_uf})" if nm_uf else "")
        mask = df_viz["nm_mun"].str.lower() == nm_mun.lower()
        if nm_uf and "nm_uf" in df_viz.columns:
            mask &= df_viz["nm_uf"].str.lower() == nm_uf.lower()
        df_city = df_viz[mask].copy()
        if len(df_city) == 0:
            print(f"  {label} não encontrado — ignorado.")
            return
        df_city["geometry"] = df_city["h3_id"].map(_h3_to_polygon)
        df_city = df_city.dropna(subset=["geometry"])
        if len(df_city) == 0:
            return
        gdf = gpd.GeoDataFrame(df_city, geometry="geometry", crs="EPSG:4326")
        safe = (
            label.lower()
            .replace(" ", "_").replace("(", "").replace(")", "")
            .replace("ã", "a").replace("â", "a").replace("á", "a")
            .replace("ê", "e").replace("é", "e")
            .replace("ó", "o").replace("ú", "u")
        )
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        gdf.plot(
            column=col_v6_norm, ax=axes[0], cmap="RdYlGn_r",
            vmin=0, vmax=1, legend=True,
            legend_kwds={"label": "Vulnerabilidade v6 (0=menor, 1=maior)", "shrink": 0.5},
        )
        axes[0].set_title(f"V6 — Cobertura Vegetal (vulnerabilidade)\n{label}", fontsize=11)
        axes[0].axis("off")

        gdf.plot(
            column=COL_NDVI_RAW, ax=axes[1], cmap="RdYlGn",
            legend=True,
            legend_kwds={"label": "NDVI médio Landsat 8/9 (2020–2024)", "shrink": 0.5},
        )
        axes[1].set_title(f"NDVI bruto\n{label}", fontsize=11)
        axes[1].axis("off")

        for ax in axes:
            minx, miny, maxx, maxy = gdf.total_bounds
            pad_x = (maxx - minx) * 0.05
            pad_y = (maxy - miny) * 0.05
            ax.set_xlim(minx - pad_x, maxx + pad_x)
            ax.set_ylim(miny - pad_y, maxy + pad_y)

        out = DEBUG_DIR / f"v6_areas_verdes_{safe}_{now}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {label}: {out.name}")

    if "nm_mun" in df_viz.columns:
        for city in CITIES:
            _city_map(city)
    else:
        print("  nm_mun ausente na base_metadata — mapas por cidade ignorados.")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(
        df_viz[col_v6_norm].dropna(), bins=100,
        alpha=0.4, color="steelblue", density=True,
        label=f"Brasil (n={df_viz[col_v6_norm].notna().sum():,})",
    )
    ax.axvline(0.5, color="black", linestyle=":", linewidth=1.2, label="limiar 0.5")
    ax.set_xlabel(f"{col_v6_norm}  (0 = menor vulnerabilidade, 1 = maior)")
    ax.set_ylabel("Densidade")
    ax.set_title("Distribuição — V6 Cobertura Vegetal (NDVI Landsat 8/9, 2020–2024)")
    ax.legend()
    out_dist = DEBUG_DIR / f"v6_areas_verdes_distribuicao_{now}.png"
    fig.savefig(out_dist, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Distribuição nacional: {out_dist.name}")


if __name__ == "__main__":
    main()
