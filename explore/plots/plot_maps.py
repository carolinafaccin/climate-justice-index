"""
H3 maps of IIC v2.0 by municipality.

Generates one PNG per map for each city in config/cities.json:
  - 1 map for the final IIC index
  - 4 maps for sub-indices (IP, IV, IE, IG)
  - 23 maps for individual indicators

Each map:
  - Is zoomed to the municipality boundary (from IBGE 2024 mesh, EPSG:5880)
  - Overlays the municipal boundary line
  - Includes a scale bar (km) and north arrow

Outputs: cfg.FIGURES_DIR / "maps" / map_*_{ts}.png
"""

import json
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
import geopandas as gpd
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR.parent))
from src import config as cfg
import utils as diag_utils

# ==============================================================================
# PATHS AND PARAMETERS
# ==============================================================================
DPI          = 150
FIG_SIZE     = (9, 9)
CLASS_BOUNDS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CLASS_LABELS = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]

MAPS_DIR = cfg.FIGURES_DIR / "maps"
MAPS_DIR.mkdir(parents=True, exist_ok=True)

MUNICIPALITIES_GPKG = cfg.RAW_DIR / "ibge" / "malha_municipal" / "2024" / "municipios.gpkg"

with open(PROJECT_ROOT / "config" / "cities.json", encoding="utf-8") as _f:
    CITIES = json.load(_f).get("cities", [])

# Most recent final IIC parquet (exclude dashboard variants)
_results = sorted(
    [
        p for p in cfg.FILES['output']['results_dir'].glob("br_h3_iic_v2_0_*.parquet")
        if "dashboard" not in p.name
    ],
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
if not _results:
    raise FileNotFoundError(
        f"No results found in:\n  {cfg.FILES['output']['results_dir']}\n"
        "Run `python run_index.py` first to generate the results."
    )
RESULTS_FILE = _results[0]

_ts_match = re.search(r"(\d{8}_\d{6})", RESULTS_FILE.stem)
FILE_TS = _ts_match.group(1) if _ts_match else RESULTS_FILE.stem

ALL_INDICATORS = diag_utils.ALL_INDICATOR_KEYS
DIMS           = diag_utils.DIMS


# ==============================================================================
# MUNICIPAL BOUNDARY MESH
# ==============================================================================
_MUNICIPALITIES_GDF = None


def _load_municipalities() -> gpd.GeoDataFrame | None:
    global _MUNICIPALITIES_GDF
    if _MUNICIPALITIES_GDF is not None:
        return _MUNICIPALITIES_GDF
    if not MUNICIPALITIES_GPKG.exists():
        print(f"  [WARNING] Municipal mesh not found: {MUNICIPALITIES_GPKG}")
        return None
    try:
        print("Loading IBGE 2024 municipal mesh...")
        gdf = gpd.read_file(MUNICIPALITIES_GPKG)
        _MUNICIPALITIES_GDF = gdf.to_crs("EPSG:4326")
        print(f"  {len(_MUNICIPALITIES_GDF):,} municipalities loaded.")
        return _MUNICIPALITIES_GDF
    except Exception as e:
        print(f"  [WARNING] Failed to load municipal mesh: {e}")
        return None


def _get_municipality_boundary(cd_mun) -> gpd.GeoDataFrame | None:
    muns = _load_municipalities()
    if muns is None or cd_mun is None:
        return None
    try:
        cd   = str(int(float(cd_mun))).zfill(7)
        mask = muns["cd_mun"].astype(str).str.zfill(7) == cd
        result = muns[mask]
        return result if not result.empty else None
    except Exception:
        return None


# ==============================================================================
# COLOR HELPERS
# ==============================================================================

def _class_colors_from_base(base_color: str) -> list:
    """Build 5-step sequential palette from near-white to the given base colour."""
    light = np.array(mcolors.to_rgba("#f7f7f7"))
    dark  = np.array(mcolors.to_rgba(base_color))
    return [tuple(light + (dark - light) * i / 4) for i in range(5)]


def _class_colors_from_cmap(cmap_name: str) -> list:
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i / 4) for i in range(5)]


# ==============================================================================
# MAP DECORATION: EXTENT, SCALE BAR, NORTH ARROW, LEGEND
# ==============================================================================

def _set_extent(ax, boundary: gpd.GeoDataFrame, buffer_frac: float = 0.05) -> None:
    """Zoom the axis to the municipality boundary plus a proportional buffer."""
    minx, miny, maxx, maxy = boundary.total_bounds
    bx = (maxx - minx) * buffer_frac
    by = (maxy - miny) * buffer_frac
    ax.set_xlim(minx - bx, maxx + bx)
    ax.set_ylim(miny - by, maxy + by)


def _add_scale_bar(ax) -> None:
    """Draw a two-tone (black/white) scale bar with km labels at the lower-left."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lat_c = (y0 + y1) / 2

    # Convert degrees-longitude to km at the central latitude
    km_per_deg = np.cos(np.radians(lat_c)) * 111.32
    map_w_km   = (x1 - x0) * km_per_deg

    # Pick a "nice" bar length ~20% of the map width
    nice = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    bar_km  = min(nice, key=lambda d: abs(d - map_w_km * 0.20))
    bar_deg = bar_km / km_per_deg

    # Anchor: lower-left with 5% margin
    mx = (x1 - x0) * 0.05
    my = (y1 - y0) * 0.05
    bx0 = x0 + mx
    by0 = y0 + my
    bar_h = (y1 - y0) * 0.012   # visual height of the bar

    # Left half (black) and right half (white with black border)
    ax.add_patch(Rectangle(
        (bx0,              by0), bar_deg / 2, bar_h,
        facecolor="black", edgecolor="black", lw=0.5, zorder=20, transform=ax.transData,
    ))
    ax.add_patch(Rectangle(
        (bx0 + bar_deg / 2, by0), bar_deg / 2, bar_h,
        facecolor="white", edgecolor="black", lw=0.5, zorder=20, transform=ax.transData,
    ))
    # Outer border
    ax.add_patch(Rectangle(
        (bx0, by0), bar_deg, bar_h,
        facecolor="none", edgecolor="black", lw=0.9, zorder=21, transform=ax.transData,
    ))

    # Labels above the bar
    ty = by0 + bar_h * 1.4
    fs = 6.5
    ax.text(bx0,            ty, "0",              ha="center", va="bottom",
            fontsize=fs, zorder=22, transform=ax.transData)
    ax.text(bx0 + bar_deg,  ty, f"{bar_km:.4g} km", ha="center", va="bottom",
            fontsize=fs, zorder=22, transform=ax.transData)


def _add_north_arrow(ax) -> None:
    """Draw a north arrow (filled arrowhead + 'N' label) in the upper-right corner."""
    # All coordinates in axes fraction (0–1)
    x, y_base, y_tip = 0.92, 0.84, 0.93

    ax.annotate(
        "", xy=(x, y_tip), xytext=(x, y_base),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            color="black",
            lw=1.6,
            mutation_scale=16,
        ),
        zorder=30,
    )
    ax.text(
        x, y_tip + 0.025, "N",
        ha="center", va="bottom",
        fontsize=10, fontweight="bold",
        transform=ax.transAxes, zorder=31,
    )


def _add_legend(ax, class_colors: list) -> None:
    """Compact classified-colour legend at the lower-right of the axis."""
    handles = [
        mpatches.Patch(facecolor=class_colors[i], edgecolor="#777777",
                       lw=0.3, label=CLASS_LABELS[i])
        for i in range(len(class_colors))
    ]
    handles.append(
        mpatches.Patch(facecolor="#cccccc", edgecolor="#777777", lw=0.3, label="Sem dado")
    )
    ax.legend(
        handles=handles, loc="lower right", fontsize=6.5,
        framealpha=0.88, handlelength=1.2, handleheight=1.0,
        borderpad=0.5, labelspacing=0.3,
    )


# ==============================================================================
# CORE SINGLE-MAP RENDERER
# ==============================================================================

def _plot_single_map(
    gdf: gpd.GeoDataFrame,
    col: str,
    title: str,
    subtitle: str,
    class_colors: list,
    boundary: gpd.GeoDataFrame | None,
    out_name: str,
) -> None:
    """Render one variable as a standalone classified map and save to PNG."""
    listed = ListedColormap(class_colors)
    norm   = BoundaryNorm(CLASS_BOUNDS, ncolors=len(class_colors), clip=True)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Plot NA hexagons first (grey), then valid values
    gdf_na    = gdf[gdf[col].isna()]
    gdf_valid = gdf[gdf[col].notna()]
    if not gdf_na.empty:
        gdf_na.plot(color="#cccccc", ax=ax, linewidth=0)
    if not gdf_valid.empty:
        gdf_valid.plot(column=col, cmap=listed, norm=norm, ax=ax, linewidth=0)

    # Municipal boundary line + extent clipping
    if boundary is not None and not boundary.empty:
        boundary.boundary.plot(ax=ax, color="#1a1a1a", linewidth=1.4, zorder=10)
        _set_extent(ax, boundary)

    # Decorations (scale bar must come after extent is set)
    _add_scale_bar(ax)
    _add_north_arrow(ax)
    _add_legend(ax, class_colors)

    ax.set_title(subtitle, fontsize=9, pad=4)
    ax.axis("off")
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.005)

    path = MAPS_DIR / f"{out_name}_{FILE_TS}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ==============================================================================
# UTILITIES
# ==============================================================================

def _city_slug(nm_mun: str, nm_uf: str) -> str:
    import unicodedata
    slug = unicodedata.normalize("NFKD", nm_mun).encode("ascii", "ignore").decode()
    return f"{slug.lower().replace(' ', '_')}_{nm_uf.split()[-1].lower()}"


_build_gdf = diag_utils.build_gdf


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_city_data(nm_mun: str, nm_uf: str) -> pd.DataFrame:
    cols = (
        ["h3_id", "nm_mun", "nm_uf", "cd_mun", "iic_final", "ip", "iv", "ie", "ig"]
        + ALL_INDICATORS
    )
    df   = pd.read_parquet(RESULTS_FILE, columns=cols)
    mask = (df["nm_mun"] == nm_mun) & (df["nm_uf"] == nm_uf)
    out  = df[mask].drop_duplicates(subset="h3_id").reset_index(drop=True)
    print(f"  {nm_mun}/{nm_uf}: {len(out):,} hexagons loaded.")
    return out


# ==============================================================================
# PER-CITY MAP GENERATION
# ==============================================================================

def generate_city_maps(
    nm_mun: str,
    nm_uf: str,
    gdf: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame | None,
) -> None:
    """Generate all 28 individual maps for one city."""
    slug  = _city_slug(nm_mun, nm_uf)
    label = f"{nm_mun} / {nm_uf}"

    # --- 1. Final index ---
    _plot_single_map(
        gdf, "iic_final",
        title=f"{label} — IIC Final",
        subtitle="Índice de Injustiça Climática  |  5 classes (0–1)",
        class_colors=_class_colors_from_cmap("RdYlGn_r"),
        boundary=boundary,
        out_name=f"map_{slug}_iic_final",
    )

    # --- 2. Sub-indices (one map each) ---
    sub_labels = {
        "ip": "IP – Grupos Prioritários",
        "iv": "IV – Vulnerabilidade",
        "ie": "IE – Exposição a Riscos Climáticos",
        "ig": "IG – Gestão Municipal",
    }
    for dim_key, meta in DIMS.items():
        col = dim_key
        if col not in gdf.columns:
            print(f"  [SKIP] Sub-index '{col}' not in data.")
            continue
        _plot_single_map(
            gdf, col,
            title=f"{label} — {sub_labels.get(col, meta['label'])}",
            subtitle=f"Sub-índice {col.upper()}  |  5 classes (0–1)",
            class_colors=_class_colors_from_base(meta["color"]),
            boundary=boundary,
            out_name=f"map_{slug}_{col}",
        )

    # --- 3. Individual indicators (one map each) ---
    for dim_key, meta in DIMS.items():
        class_colors = _class_colors_from_base(meta["color"])
        for ind_key in meta["indicators"]:
            if ind_key not in gdf.columns:
                print(f"  [SKIP] Indicator '{ind_key}' not in data.")
                continue
            _plot_single_map(
                gdf, ind_key,
                title=f"{label} — {meta['ind_labels'][ind_key]}",
                subtitle=f"Indicador {ind_key.upper()}  |  5 classes (0–1)",
                class_colors=class_colors,
                boundary=boundary,
                out_name=f"map_{slug}_{ind_key}",
            )


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    n_maps_per_city = 1 + len(DIMS) + len(ALL_INDICATORS)
    print("=" * 60)
    print(f"IIC v2.0 Maps  |  {RESULTS_FILE.name}")
    print(f"Timestamp     : {FILE_TS}")
    print(f"Cities        : {len(CITIES)}")
    print(f"Maps per city : {n_maps_per_city}  (1 index + 4 sub-indices + {len(ALL_INDICATORS)} indicators)")
    print(f"Total expected: {len(CITIES) * n_maps_per_city}")
    print("=" * 60)

    for city in CITIES:
        nm_mun = city["nm_mun"]
        nm_uf  = city["nm_uf"]
        print(f"\n→ {nm_mun} / {nm_uf}")

        city_df = load_city_data(nm_mun, nm_uf)
        if city_df.empty:
            print("  [WARNING] No data found, skipping.")
            continue

        cd_mun   = city_df["cd_mun"].iloc[0] if "cd_mun" in city_df.columns else None
        boundary = _get_municipality_boundary(cd_mun)
        print(f"  Municipal boundary: {'loaded' if boundary is not None else 'not found'} (cd_mun={cd_mun})")

        print(f"  Converting {len(city_df):,} hexagons to geometry...")
        gdf = _build_gdf(city_df)

        generate_city_maps(nm_mun, nm_uf, gdf, boundary=boundary)

    n_saved = len(list(MAPS_DIR.glob("*.png")))
    print(f"\nDone! {n_saved} total PNGs in:\n  {MAPS_DIR}")


if __name__ == "__main__":
    main()
