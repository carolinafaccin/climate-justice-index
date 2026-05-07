"""
ETL: Landsat / GEE LST anomaly → Indicator e4 (extreme heat exposure).

Input:  cfg.RAW_DIR GEE CSVs with columns h3_id, anomalia_temp, qtd_dom (one per UF)
Output: cfg.FILES_H3["e4"] parquet
"""

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
# 0. CONFIGURATION
# ==============================================================================
# True  → Option B: normalise both anomaly AND qtd_dom before multiplying [0,1]×[0,1]
#          Result: high score only where both heat AND households are high
# False → Option A: normalise anomaly only [0,1], multiply by raw qtd_dom
#          Result: magnitude in "heat-weighted households"
NORMALIZE_DOM = True

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
            df = pd.read_csv(path, usecols=["h3_id", "anomalia_temp", "qtd_dom"])
        except ValueError:
            # fallback: read all columns and select manually
            df = pd.read_csv(path)
            df.columns = df.columns.str.lower()
            df = df[["h3_id", "anomalia_temp", "qtd_dom"]]
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
    df_all["qtd_dom"]      = pd.to_numeric(df_all["qtd_dom"],      errors="coerce").fillna(0)

    # Normalize heat anomaly to [0,1] (winsorized at p95)
    anomalia_clipped = df_all["anomalia_temp"].clip(lower=0)
    p95_anomalia     = anomalia_clipped.quantile(0.95)
    if p95_anomalia == 0:
        print("   WARNING: p95 of heat anomaly is 0 — no positive anomaly detected. e4 will be all zeros.")
        anomalia_norm = pd.Series(0.0, index=df_all.index)
    else:
        anomalia_norm = (anomalia_clipped / p95_anomalia).clip(upper=1)

    if NORMALIZE_DOM:
        # Option B: also normalise qtd_dom → score [0,1]×[0,1], high only where both are high
        p95_dom      = df_all["qtd_dom"].quantile(0.95)
        qtd_dom_term = (df_all["qtd_dom"] / p95_dom).clip(upper=1)
        print(f"   Mode: NORMALIZE_DOM=True  (Option B — anomalia_norm × qtd_dom_norm)")
    else:
        # Option A: raw qtd_dom → score in "heat-weighted households"
        qtd_dom_term = df_all["qtd_dom"]
        print(f"   Mode: NORMALIZE_DOM=False (Option A — anomalia_norm × qtd_dom)")

    df_all[col_e4_abs] = anomalia_norm * qtd_dom_term

    # e4_norm: min-max with winsorization
    df_all[col_e4_norm] = utils.normalize_minmax(df_all[col_e4_abs], winsorize=True)

    n_warming   = (df_all[col_e4_abs] > 0).sum()
    n_stable    = (df_all[col_e4_abs] == 0).sum()
    n_null      = df_all[col_e4_abs].isna().sum()
    print(f"   Hexagons with exposure (anomaly > 0 and qtd_dom > 0): {n_warming:,}")
    print(f"   Hexagons with no exposure (cooling or uninhabited):    {n_stable:,}")
    print(f"   Hexagons without data (NaN):                          {n_null:,}")

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

    print("\nGenerating debug figures...")
    _write_figures(df_final)

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

        f.write("--- qtd_dom (households per hexagon) ---\n")
        s_dom = df_all["qtd_dom"]
        f.write(f"  sum    = {s_dom.sum():,.0f} households\n")
        f.write(f"  mean   = {s_dom.mean():.2f}\n")
        f.write(f"  median = {s_dom.median():.2f}\n")
        f.write(f"  zeros  = {(s_dom == 0).sum():,} hexagons\n\n")

        for col in [col_e4_abs, col_e4_norm]:
            s = df_final[col].dropna()
            f.write(f"--- {col} ---\n")
            f.write(f"  mean   = {s.mean():.6f}\n")
            f.write(f"  median = {s.median():.6f}\n")
            f.write(f"  min    = {s.min():.6f}\n")
            f.write(f"  max    = {s.max():.6f}\n")
            f.write(f"  zeros  = {(s == 0).sum():,}\n")
            f.write(f"  nulls  = {df_final[col].isna().sum():,}\n\n")


def _write_figures(df_final):
    try:
        import matplotlib.pyplot as plt
        import h3
        import geopandas as gpd
        from shapely.geometry import Polygon as ShapelyPolygon
    except ImportError as e:
        print(f"  Skipping visualizations — missing dependency: {e}")
        return

    import json as _json

    DEBUG_DIR = cfg.FIGURES_DIR / "debug_calor"
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    CMAP = "YlOrRd"
    _mode_tag = "B" if NORMALIZE_DOM else "A"
    _PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

    _cities_path = cfg.BASE_DIR / "diagnose" / "cities.json"
    if _cities_path.exists():
        with open(_cities_path, "r", encoding="utf-8") as _f:
            CITIES = _json.load(_f)["cities"]
    else:
        CITIES = [
            {"nm_mun": "Porto Alegre",   "nm_uf": "Rio Grande do Sul"},
            {"nm_mun": "São Paulo",      "nm_uf": "São Paulo"},
            {"nm_mun": "Rio de Janeiro", "nm_uf": "Rio de Janeiro"},
        ]

    # Merge scores with full base metadata (nm_mun, nm_uf, etc.)
    df_base = pd.read_parquet(cfg.FILES_H3["base_metadata"])
    df_viz  = df_base.merge(df_final[["h3_id", col_e4_abs, col_e4_norm]], on="h3_id", how="left")

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
            column=col_e4_norm, ax=ax, cmap=CMAP,
            vmin=0, vmax=1, legend=True,
            legend_kwds={"label": "Heat exposure (0 = lower, 1 = higher)", "shrink": 0.5},
        )
        minx, miny, maxx, maxy = gdf.total_bounds
        pad_x = (maxx - minx) * 0.05
        pad_y = (maxy - miny) * 0.05
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
        ax.set_title(
            f"Indicator e4 — Extreme Heat (Option {_mode_tag})\n"
            f"{col_e4_norm}  |  {label}",
            fontsize=12,
        )
        ax.axis("off")
        out = DEBUG_DIR / f"e4_cal_{safe}_op{_mode_tag}_{now}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {label}: {out.name}")

    # ── Mapas por cidade ──────────────────────────────────────────────────
    if "nm_mun" in df_viz.columns:
        for city in CITIES:
            _city_map(city)
    else:
        print("  nm_mun absent from base_metadata — city maps skipped.")

    # ── Distribution: national histogram + highlighted cities ──────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(
        df_viz[col_e4_norm].dropna(), bins=100,
        alpha=0.25, color="gray", density=True,
        label=f"Brasil (n={df_viz[col_e4_norm].notna().sum():,})",
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
                    df_city[col_e4_norm].dropna(), bins=60,
                    alpha=0.55, color=color, density=True,
                    label=f"{label} (n={len(df_city):,})",
                )
    ax.axvline(0.5, color="black", linestyle=":", linewidth=1.2, label="threshold 0.5")
    ax.set_xlabel(f"{col_e4_norm}  (0 = lower exposure, 1 = higher)")
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of {col_e4_norm} — Extreme Heat (Option {_mode_tag})")
    ax.legend()
    out_dist = DEBUG_DIR / f"e4_cal_distribuicao_op{_mode_tag}_{now}.png"
    fig.savefig(out_dist, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Distribuição: {out_dist.name}")


if __name__ == "__main__":
    main()
