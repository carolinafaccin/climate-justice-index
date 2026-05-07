"""
Identifies the municipality with the highest weighted mean score (weighted by qtd_dom)
for the final IIC, each sub-index, and each individual indicator.

Generates H3 maps (IIC + sub-indices + per-dimension indicators) for each identified
municipality, saved under outputs/figures/maps/highlights_analysis/.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap, LinearSegmentedColormap
import matplotlib.gridspec as mgridspec
import geopandas as gpd
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
from src import config as cfg
import utils as diag_utils

# ==============================================================================
# PATHS
# ==============================================================================
_results = sorted(
    cfg.FILES["output"]["results_dir"].glob("br_h3_iic_v2_0_*.parquet"),
    key=lambda p: p.stat().st_mtime, reverse=True,
)
if not _results:
    raise FileNotFoundError(
        f"No results found in:\n  {cfg.FILES['output']['results_dir']}\n"
        "Run `python run.py` first to generate the results."
    )
RESULTS_FILE = _results[0]

HIGHLIGHTS_DIR = cfg.FIGURES_DIR / "maps" / "highlights_analysis"
HIGHLIGHTS_DIR.mkdir(parents=True, exist_ok=True)

DPI = 150
CLASS_BOUNDS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CLASS_LABELS = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

ALL_INDICATORS = diag_utils.ALL_INDICATOR_KEYS
ALL_METRICS    = ["iic_final", "ip", "iv", "ie", "ig"] + ALL_INDICATORS
DIMS           = diag_utils.DIMS

# ==============================================================================
# MAP HELPERS
# ==============================================================================

def _class_colors_from_base(base_color: str) -> list:
    light = np.array(mcolors.to_rgba("#f7f7f7"))
    dark  = np.array(mcolors.to_rgba(base_color))
    return [tuple(light + (dark - light) * i / 4) for i in range(5)]


def _class_colors_from_cmap(cmap_name: str) -> list:
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i / 4) for i in range(5)]


def _plot_classified_panel(gdf, col, ax, title, class_colors,
                            show_legend: bool = False) -> None:
    listed = ListedColormap(class_colors)
    norm   = BoundaryNorm(CLASS_BOUNDS, ncolors=len(class_colors), clip=True)
    gdf_valid = gdf[gdf[col].notna()]
    gdf_na    = gdf[gdf[col].isna()]
    if not gdf_na.empty:
        gdf_na.plot(color="#cccccc", ax=ax, linewidth=0)
    if not gdf_valid.empty:
        gdf_valid.plot(column=col, cmap=listed, norm=norm, ax=ax, linewidth=0)
    if show_legend:
        patches = [
            mpatches.Patch(facecolor=class_colors[i], edgecolor="#999999",
                           lw=0.3, label=CLASS_LABELS[i])
            for i in range(len(class_colors))
        ]
        ax.legend(handles=patches, loc="lower left", fontsize=6,
                  framealpha=0.9, handlelength=1.2, handleheight=1.0,
                  borderpad=0.5, labelspacing=0.3)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    ax.axis("off")


def _shared_legend(fig, class_colors, y_anchor: float = 0.01) -> None:
    patches = [
        mpatches.Patch(facecolor=class_colors[i], edgecolor="#999999",
                       lw=0.3, label=CLASS_LABELS[i])
        for i in range(len(class_colors))
    ]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, y_anchor), framealpha=0.9,
               handlelength=1.5, handleheight=1.0)


_build_gdf = diag_utils.build_gdf


def _city_slug(nm_mun: str, nm_uf: str) -> str:
    import unicodedata
    slug = unicodedata.normalize("NFKD", nm_mun).encode("ascii", "ignore").decode()
    slug = slug.lower().replace(" ", "_")
    uf   = nm_uf.split()[-1].lower()
    return f"{slug}_{uf}"


def _save(fig, name: str) -> None:
    path = HIGHLIGHTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path.name}")


# ==============================================================================
# FIGURAS
# ==============================================================================

def fig_map_city(nm_mun: str, nm_uf: str, gdf: gpd.GeoDataFrame) -> None:
    PLOTS = [
        ("iic_final", "IIC Final",              _class_colors_from_cmap("RdYlGn_r")),
        ("ip",        "IP - Grupos Prioritários", _class_colors_from_base(DIMS["ip"]["color"])),
        ("iv",        "IV - Vulnerabilidade",     _class_colors_from_base(DIMS["iv"]["color"])),
        ("ie",        "IE - Exposição",           _class_colors_from_base(DIMS["ie"]["color"])),
        ("ig",        "IG - Gestão Municipal",    _class_colors_from_base(DIMS["ig"]["color"])),
    ]
    fig = plt.figure(figsize=(22, 13))
    gs  = mgridspec.GridSpec(2, 3, figure=fig, hspace=0.18, wspace=0.06)
    ax_main  = fig.add_subplot(gs[:, 0])
    axes_sub = [fig.add_subplot(gs[r, c]) for r, c in [(0,1),(0,2),(1,1),(1,2)]]

    for ax, (col, title, class_colors) in zip([ax_main] + axes_sub, PLOTS):
        _plot_classified_panel(gdf, col, ax, title, class_colors, show_legend=True)

    fig.suptitle(
        f"{nm_mun} / {nm_uf}\nH3 Spatial Distribution (res. 9) — IIC and Sub-indices",
        fontsize=14, fontweight="bold", y=1.01,
    )
    _save(fig, f"map_{_city_slug(nm_mun, nm_uf)}")


def fig_map_city_indicators(nm_mun: str, nm_uf: str,
                             gdf: gpd.GeoDataFrame) -> None:
    for dim_key, meta in DIMS.items():
        indicators   = meta["indicators"]
        class_colors = _class_colors_from_base(meta["color"])
        n_cols = 3
        n_rows = (len(indicators) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 5.5, n_rows * 5.5 + 0.8))
        axes_flat = np.array(axes).flatten()

        for i, ind_key in enumerate(indicators):
            if ind_key not in gdf.columns:
                axes_flat[i].set_visible(False)
                continue
            _plot_classified_panel(gdf, ind_key, axes_flat[i],
                                   title=meta["ind_labels"][ind_key],
                                   class_colors=class_colors, show_legend=False)
        for j in range(len(indicators), len(axes_flat)):
            axes_flat[j].set_visible(False)

        _shared_legend(fig, class_colors, y_anchor=0.01)
        fig.suptitle(
            f"{nm_mun} / {nm_uf}  —  {meta['label']}\nIndividual indicators (5 classes, 0–1)",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0.07, 1, 0.93])
        _save(fig, f"map_{_city_slug(nm_mun, nm_uf)}_ind_{dim_key}")


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

def load_and_aggregate() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the full results parquet and aggregate by municipality (qtd_dom weighted mean)."""
    print(f"Loading {RESULTS_FILE.name} ...")
    needed = ["h3_id", "nm_mun", "nm_uf", "qtd_dom"] + ALL_METRICS
    needed = [c for c in needed if c]  # remove None
    df = pd.read_parquet(RESULTS_FILE, columns=needed)
    df = df.dropna(subset=["nm_mun", "nm_uf"])
    print(f"  {len(df):,} hexagons loaded.")

    def _wmean_series(grp: pd.DataFrame) -> pd.Series:
        w = grp["qtd_dom"].fillna(1.0).clip(lower=0.001)
        result = {}
        for m in ALL_METRICS:
            if m not in grp.columns:
                result[m] = np.nan
                continue
            v = grp[m]
            mask = v.notna()
            result[m] = np.average(v[mask], weights=w[mask]) if mask.sum() > 0 else np.nan
        result["n_hex"] = len(grp)
        return pd.Series(result)

    print("Aggregating by municipality (household-count weighted mean)...")
    agg = df.groupby(["nm_mun", "nm_uf"], sort=False).apply(_wmean_series).reset_index()
    print(f"  {len(agg):,} municipalities.")
    return df, agg


def find_highlights(agg: pd.DataFrame) -> pd.DataFrame:
    """Para cada métrica, retorna o município com maior valor médio."""
    rows = []
    for m in ALL_METRICS:
        if m not in agg.columns:
            continue
        sub = agg[["nm_mun", "nm_uf", "n_hex", m]].dropna(subset=[m])
        if sub.empty:
            continue
        top = sub.loc[sub[m].idxmax()]
        rows.append({
            "metric":       m,
            "municipality": top["nm_mun"],
            "uf":           top["nm_uf"],
            "value":        top[m],
            "n_hex":        int(top["n_hex"]),
        })
    return pd.DataFrame(rows)


def print_table(highlights: pd.DataFrame) -> None:
    print(f"\n{'─' * 78}")
    print(f"{'Metric':<12} {'Municipality':<35} {'UF':<5} {'Value':>7}  {'Hex':>6}")
    print(f"{'─' * 78}")
    for _, row in highlights.iterrows():
        print(
            f"{row['metric']:<12} {row['municipality']:<35} "
            f"{row['uf']:<5} {row['value']:>7.4f}  {row['n_hex']:>6,}"
        )
    print()


def load_city_data(nm_mun: str, nm_uf: str) -> pd.DataFrame:
    needed = ["h3_id", "nm_mun", "nm_uf", "iic_final",
              "ip", "iv", "ie", "ig"] + ALL_INDICATORS
    df = pd.read_parquet(RESULTS_FILE, columns=needed)
    mask = (df["nm_mun"] == nm_mun) & (df["nm_uf"] == nm_uf)
    result = df[mask].drop_duplicates(subset="h3_id").reset_index(drop=True)
    print(f"  {nm_mun} / {nm_uf}: {len(result):,} hexagons")
    return result


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("Highlights Analysis — Municipalities with Highest Scores")
    print(f"File: {RESULTS_FILE.name}")
    print("=" * 60)

    df_full, agg = load_and_aggregate()
    highlights   = find_highlights(agg)

    print_table(highlights)

    # Unique municipalities to map
    unique_cities = (
        highlights[["municipality", "uf"]]
        .drop_duplicates()
        .rename(columns={"municipality": "nm_mun", "uf": "nm_uf"})
        .to_dict("records")
    )
    print(f"{len(unique_cities)} unique municipalit{'y' if len(unique_cities) == 1 else 'ies'} identified.")
    print(f"Maps will be saved to: {HIGHLIGHTS_DIR}\n")

    for city in unique_cities:
        nm_mun, nm_uf = city["nm_mun"], city["nm_uf"]
        metrics_here  = highlights.loc[
            (highlights["municipality"] == nm_mun) & (highlights["uf"] == nm_uf),
            "metric"
        ].tolist()
        print(f"→ {nm_mun} / {nm_uf}  (top in: {', '.join(metrics_here)})")

        city_df = load_city_data(nm_mun, nm_uf)
        if city_df.empty:
            print("  [WARNING] No data found, skipping.")
            continue

        print(f"  Converting {len(city_df):,} hexagons to geometry...")
        gdf = _build_gdf(city_df)
        fig_map_city(nm_mun, nm_uf, gdf)
        fig_map_city_indicators(nm_mun, nm_uf, gdf)

    print(f"\nDone! {len(list(HIGHLIGHTS_DIR.glob('*.png')))} maps in {HIGHLIGHTS_DIR}")


if __name__ == "__main__":
    main()
