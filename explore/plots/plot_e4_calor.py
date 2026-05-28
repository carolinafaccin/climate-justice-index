"""
Debug figures for indicator e4 (extreme heat / LST anomaly).

Reads cfg.FILES_H3["e4"] and generates per-city H3 maps and a national distribution histogram.

Usage:
    python explore/plots/plot_e4_calor.py
"""

import json
import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pipeline.py").exists())
sys.path.insert(0, str(_ROOT))

from src import config as cfg
from src import utils
import pandas as pd

CMAP     = "YlOrRd"
_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

col_norm = cfg.COLUMN_MAP["e4"]
col_abs  = col_norm.replace("_norm", "_abs")

DEBUG_DIR = cfg.FIGURES_DIR / "debug_calor"


def _h3_to_polygon(h3_id):
    try:
        boundary = utils.h3_cell_to_boundary(h3_id)
        from shapely.geometry import Polygon as ShapelyPolygon
        return ShapelyPolygon([(lng, lat) for lat, lng in boundary])
    except Exception:
        return None


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import geopandas as gpd
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return

    e4_path = cfg.FILES_H3["e4"]
    if not e4_path.exists():
        raise FileNotFoundError(
            f"E4 parquet not found at {e4_path} — run etl/exposure/e4_calor.py first"
        )

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    _cities_path = cfg.BASE_DIR / "config" / "cities.json"
    if _cities_path.exists():
        with open(_cities_path, "r", encoding="utf-8") as f:
            CITIES = json.load(f)["cities"]
    else:
        CITIES = [
            {"nm_mun": "Porto Alegre",   "nm_uf": "Rio Grande do Sul"},
            {"nm_mun": "São Paulo",      "nm_uf": "São Paulo"},
            {"nm_mun": "Rio de Janeiro", "nm_uf": "Rio de Janeiro"},
        ]

    df_base = pd.read_parquet(cfg.FILES_H3["base_metadata"])
    df_e4   = pd.read_parquet(e4_path, columns=["h3_id", col_abs, col_norm])
    df_viz  = df_base.merge(df_e4, on="h3_id", how="left")

    def _city_map(city_info):
        nm_mun = city_info["nm_mun"]
        nm_uf  = city_info.get("nm_uf")
        label  = nm_mun + (f" ({nm_uf})" if nm_uf else "")
        mask = df_viz["nm_mun"].str.lower() == nm_mun.lower()
        if nm_uf and "nm_uf" in df_viz.columns:
            mask &= df_viz["nm_uf"].str.lower() == nm_uf.lower()
        df_city = df_viz[mask].copy()
        if len(df_city) == 0:
            print(f"  {label} not found — skipped.")
            return
        df_city["geometry"] = df_city["h3_id"].map(_h3_to_polygon)
        df_city = df_city.dropna(subset=["geometry"])
        if len(df_city) == 0:
            print(f"  {label}: all polygon conversions failed — skipped.")
            return
        gdf = gpd.GeoDataFrame(df_city, geometry="geometry", crs="EPSG:4326")

        safe = (
            label.lower()
            .replace(" ", "_").replace("(", "").replace(")", "")
            .replace("ã", "a").replace("â", "a").replace("á", "a")
            .replace("ê", "e").replace("é", "e")
            .replace("ó", "o").replace("ú", "u")
        )
        fig, ax = plt.subplots(figsize=(12, 12))
        gdf.plot(
            column=col_norm, ax=ax, cmap=CMAP,
            vmin=0, vmax=1, legend=True,
            legend_kwds={"label": "Heat exposure (0 = lower, 1 = higher)", "shrink": 0.5},
        )
        minx, miny, maxx, maxy = gdf.total_bounds
        pad_x = (maxx - minx) * 0.05
        pad_y = (maxy - miny) * 0.05
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
        ax.set_title(f"Indicator e4 — Extreme Heat\n{col_norm}  |  {label}", fontsize=12)
        ax.axis("off")
        out = DEBUG_DIR / f"e4_cal_{safe}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {label}: {out.name}")

    if "nm_mun" in df_viz.columns:
        for city in CITIES:
            _city_map(city)
    else:
        print("  nm_mun absent from base_metadata — city maps skipped.")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(
        df_viz[col_norm].dropna(), bins=100,
        alpha=0.25, color="gray", density=True,
        label=f"Brasil (n={df_viz[col_norm].notna().sum():,})",
    )
    if "nm_mun" in df_viz.columns:
        for idx, city_info in enumerate(CITIES):
            nm_mun = city_info["nm_mun"]
            nm_uf  = city_info.get("nm_uf")
            mask = df_viz["nm_mun"].str.lower() == nm_mun.lower()
            if nm_uf and "nm_uf" in df_viz.columns:
                mask &= df_viz["nm_uf"].str.lower() == nm_uf.lower()
            df_city = df_viz[mask]
            label = nm_mun + (f" ({nm_uf})" if nm_uf else "")
            color = _PALETTE[idx % len(_PALETTE)]
            if len(df_city) > 0:
                ax.hist(
                    df_city[col_norm].dropna(), bins=60,
                    alpha=0.55, color=color, density=True,
                    label=f"{label} (n={len(df_city):,})",
                )
    ax.axvline(0.5, color="black", linestyle=":", linewidth=1.2, label="threshold 0.5")
    ax.set_xlabel(f"{col_norm}  (0 = lower exposure, 1 = higher)")
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of {col_norm} — Extreme Heat")
    ax.legend()
    out_dist = DEBUG_DIR / "e4_cal_distribuicao.png"
    fig.savefig(out_dist, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Distribuição: {out_dist.name}")


if __name__ == "__main__":
    main()
