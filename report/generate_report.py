"""
Generate a static HTML report for the IIC v2.0.

Usage:
    python report/generate_report.py

Outputs to <repo_root>/docs/:
    index.html, assets/style.css, imgs/{city_slug}/...
"""

import json
import sys
import unicodedata
from datetime import date
from pathlib import Path

import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import geopandas as gpd
from PIL import Image

plt.rcParams.update({
    "font.family":     "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
})

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "explore"))

from src import config as cfg
import utils as diag_utils

# ==============================================================================
# PATHS
# ==============================================================================
DOCS_DIR   = PROJECT_ROOT / "docs"
IMGS_DIR   = DOCS_DIR / "imgs"
ASSETS_DIR = DOCS_DIR / "assets"
TMPL_DIR   = SCRIPT_DIR / "templates"

MUNICIPALITIES_GPKG  = cfg.RAW_DIR / "ibge" / "malha_municipal" / "2024" / "municipios.gpkg"
TIPOLOGIAS_CSV       = cfg.RAW_DIR / "ibge" / "tipologias" / "tipologias_municipios_brasil.csv"
COASTAL_MUNS_CSV     = cfg.RAW_DIR / "ibge" / "malha_municipal" / "2024" / "municipios_defrontantes_com_o_mar.csv"

with open(PROJECT_ROOT / "config" / "cities.json", encoding="utf-8") as _f:
    CITIES = json.load(_f)["cities"]

# ==============================================================================
# VISUAL CONSTANTS
# ==============================================================================
DPI              = 120
OVERVIEW_FIG_SIZE = (15, 7.5)  # landscape: map left + radar/hist stacked right
IND_FIG_SIZE      = (12, 5.5)  # landscape combined indicator figure

WRI_YELLOW = "#F0AB00"
WRI_GREEN  = "#32864B"

DIM_COLORS = {
    "ip": "#eb8026",  # orange  – Grupos Prioritários
    "iv": "#3855a3",  # blue    – Vulnerabilidade
    "ie": "#32864b",  # green   – Exposição
    "ig": "#9b216c",  # purple  – Capacidade de Gestão Municipal
}
DIM_ORDER    = ["ip", "iv", "ie", "ig"]
ALL_IND_KEYS = diag_utils.ALL_INDICATOR_KEYS
ABBR_TO_DIM  = {meta["abbr"].lower(): dim for dim, meta in cfg.DIMENSION_META.items()}

WEBP_QUALITY = 82  # 0-100; 82 gives ~68% size reduction vs PNG with negligible quality loss

def _save_webp(fig, path: Path, dpi: int = DPI) -> None:
    """Save a matplotlib figure as WebP via Pillow (significantly smaller than PNG)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    Image.open(buf).save(path.with_suffix(".webp"), "WEBP", quality=WEBP_QUALITY, method=4)
    plt.close(fig)

# ==============================================================================
# FILE DISCOVERY
# ==============================================================================
_results = sorted(
    [p for p in cfg.FILES['output']['results_dir'].glob("br_h3_iic_v2_0_*.parquet") if "dashboard" not in p.name],
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
if not _results:
    # Fallback: use main dashboard parquet (not the dimension-split ones)
    _results = sorted(
        [p for p in cfg.FILES['output']['dashboard_dir'].glob("br_h3_iic_v2_0_dashboard_*.parquet")
         if "_dim_" not in p.name],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
if not _results:
    raise FileNotFoundError(f"No results found in {cfg.FILES['output']['results_dir']}.\nRun `python run_index.py` first.")
RESULTS_FILE = _results[0]

# ==============================================================================
# HELPERS
# ==============================================================================
def fmt3(x) -> str:
    """Format float with exactly 3 decimal places."""
    if x is None:
        return "—"
    return f"{float(x):.3f}"


def _slug(nm_mun: str, nm_uf: str) -> str:
    s = unicodedata.normalize("NFKD", nm_mun).encode("ascii", "ignore").decode()
    u = unicodedata.normalize("NFKD", nm_uf).encode("ascii", "ignore").decode()
    return f"{s.lower().replace(' ', '_')}_{u.split()[-1].lower()}"


def _class_colors_from_base(base_color: str) -> list:
    light = np.array(mcolors.to_rgba("#f7f7f7"))
    dark  = np.array(mcolors.to_rgba(base_color))
    return [tuple(light + (dark - light) * i / 4) for i in range(5)]


def _iic_colors() -> list:
    cmap = plt.get_cmap("RdYlGn_r")
    return [cmap(i / 4) for i in range(5)]

# ==============================================================================
# TIPOLOGIAS + PORTE QUINTILES
# ==============================================================================
def load_coastal_muns() -> set[str]:
    """Returns set of cd_mun (7-digit strings) for coastal municipalities."""
    if not COASTAL_MUNS_CSV.exists():
        print(f"  [WARNING] Coastal municipalities CSV not found: {COASTAL_MUNS_CSV}")
        return set()
    df = pd.read_csv(COASTAL_MUNS_CSV, sep=";", dtype=str)
    col = "municipios_defrontantes_com_o_mar"
    coastal = df[df[col].str.strip().str.lower().isin(["true", "1", "yes", "sim"])]["cd_mun"]
    return set(coastal.apply(lambda x: str(int(float(x))).zfill(7)))


def load_porte_map() -> dict[str, str]:
    """Returns {cd_mun_7digits: porte_munic}."""
    if not TIPOLOGIAS_CSV.exists():
        print(f"  [WARNING] Tipologias CSV not found: {TIPOLOGIAS_CSV}")
        return {}
    df = pd.read_csv(TIPOLOGIAS_CSV, sep=";", usecols=["cd_mun", "porte_munic"], dtype=str)
    df["cd_mun"] = df["cd_mun"].str.zfill(7)
    return dict(zip(df["cd_mun"], df["porte_munic"]))


def compute_porte_quintiles(
    df_full: pd.DataFrame,
    porte_map: dict[str, str],
    cols: list[str],
) -> dict[str, dict[str, list[float]]]:
    """Return {porte: {col: [q0, q20, q40, q60, q80, q100]}}."""
    df = df_full.copy()
    cd_str     = df["cd_mun"].apply(lambda x: str(int(float(x))).zfill(7) if pd.notna(x) else None)
    df["_porte"] = cd_str.map(porte_map)

    result: dict = {}
    for porte, grp in df.groupby("_porte", dropna=True):
        result[porte] = {}
        for col in cols:
            if col not in grp.columns:
                continue
            vals = grp[col].dropna().values
            if len(vals) < 10:
                continue
            q = np.quantile(vals, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).tolist()
            for i in range(1, len(q)):
                if q[i] <= q[i - 1]:
                    q[i] = q[i - 1] + 1e-6
            result[porte][col] = q
    return result


def get_bounds_labels(
    porte: str | None,
    col: str,
    quintile_data: dict,
) -> tuple[list[float], list[str]]:
    """Return (class_bounds, class_labels) from porte quintiles or fixed fallback."""
    fallback = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    if porte and porte in quintile_data and col in quintile_data[porte]:
        q = quintile_data[porte][col]
        return q, [f"{q[i]:.3f}–{q[i+1]:.3f}" for i in range(5)]
    return fallback, [f"{fallback[i]:.1f}–{fallback[i+1]:.1f}" for i in range(5)]

# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_full_data() -> pd.DataFrame:
    print(f"Loading {RESULTS_FILE.name} ...")
    try:
        import pyarrow.parquet as pq
        available = set(pq.read_schema(RESULTS_FILE).names)
    except Exception:
        available = None

    want = (
        ["h3_id", "nm_mun", "nm_uf", "cd_mun", "iic_final", "ip", "iv", "ie", "ig"]
        + ALL_IND_KEYS
    )
    cols = [c for c in want if available is None or c in available]
    df   = pd.read_parquet(RESULTS_FILE, columns=cols)
    print(f"  {len(df):,} hexagons loaded.")
    return df


def compute_national_stats(df: pd.DataFrame) -> dict:
    stats: dict = {}
    for col in ["iic_final", "ip", "iv", "ie", "ig"] + ALL_IND_KEYS:
        if col in df.columns:
            stats[col] = {"mean": float(df[col].dropna().mean())}
    stats["_iic_all"] = df["iic_final"].dropna().values
    return stats


_MUNICIPALITIES_GDF: gpd.GeoDataFrame | None = None


def load_municipalities() -> gpd.GeoDataFrame | None:
    global _MUNICIPALITIES_GDF
    if _MUNICIPALITIES_GDF is not None:
        return _MUNICIPALITIES_GDF
    if not MUNICIPALITIES_GPKG.exists():
        print(f"  [WARNING] Municipal mesh not found: {MUNICIPALITIES_GPKG}")
        return None
    print("Loading IBGE municipal mesh...")
    gdf = gpd.read_file(MUNICIPALITIES_GPKG).to_crs("EPSG:4326")
    _MUNICIPALITIES_GDF = gdf
    print(f"  {len(gdf):,} municipalities loaded.")
    return gdf


def get_boundary(cd_mun, municipalities: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame | None:
    if municipalities is None or cd_mun is None:
        return None
    try:
        cd   = str(int(float(cd_mun))).zfill(7)
        mask = municipalities["cd_mun"].astype(str).str.zfill(7) == cd
        r    = municipalities[mask]
        return r if not r.empty else None
    except Exception:
        return None

# ==============================================================================
# MAP DECORATION
# ==============================================================================
def _set_extent(ax, boundary: gpd.GeoDataFrame, buf: float = 0.05) -> None:
    minx, miny, maxx, maxy = boundary.total_bounds
    bx, by = (maxx - minx) * buf, (maxy - miny) * buf
    ax.set_xlim(minx - bx, maxx + bx)
    ax.set_ylim(miny - by, maxy + by)


def _add_scale_bar(ax) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    km_per_deg = np.cos(np.radians((y0 + y1) / 2)) * 111.32
    nice   = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    bar_km = min(nice, key=lambda d: abs(d - (x1 - x0) * km_per_deg * 0.20))
    bar_deg = bar_km / km_per_deg
    bx0 = x0 + (x1 - x0) * 0.05
    by0 = y0 + (y1 - y0) * 0.05
    bar_h = (y1 - y0) * 0.012
    ax.add_patch(Rectangle((bx0, by0), bar_deg / 2, bar_h, fc="black", ec="black", lw=0.5, zorder=20, transform=ax.transData))
    ax.add_patch(Rectangle((bx0 + bar_deg / 2, by0), bar_deg / 2, bar_h, fc="white", ec="black", lw=0.5, zorder=20, transform=ax.transData))
    ax.add_patch(Rectangle((bx0, by0), bar_deg, bar_h, fc="none", ec="black", lw=0.9, zorder=21, transform=ax.transData))
    ty = by0 + bar_h * 1.4
    ax.text(bx0,           ty, "0",              ha="center", va="bottom", fontsize=6, zorder=22, transform=ax.transData)
    ax.text(bx0 + bar_deg, ty, f"{bar_km:.4g} km", ha="center", va="bottom", fontsize=6, zorder=22, transform=ax.transData)


def _add_north_arrow(ax) -> None:
    ax.annotate("", xy=(0.93, 0.94), xytext=(0.93, 0.85),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=14), zorder=30)
    ax.text(0.93, 0.96, "N", ha="center", va="bottom", fontsize=9, fontweight="bold",
            transform=ax.transAxes, zorder=31)


def _legend_handles(class_colors: list, labels: list) -> list:
    handles = [mpatches.Patch(fc=class_colors[i], ec="#888", lw=0.4, label=labels[i]) for i in range(5)]
    handles.append(mpatches.Patch(fc="#cccccc", ec="#888", lw=0.4, label="Sem dado"))
    return handles

# ==============================================================================
# MAP CORE RENDERER
# ==============================================================================
def _draw_hexagons(ax, gdf: gpd.GeoDataFrame, col: str,
                   class_colors: list, class_bounds: list,
                   boundary: gpd.GeoDataFrame | None) -> None:
    listed = ListedColormap(class_colors)
    norm   = BoundaryNorm(class_bounds, ncolors=len(class_colors), clip=True)
    gdf_na    = gdf[gdf[col].isna()]
    gdf_valid = gdf[gdf[col].notna()]
    if not gdf_na.empty:
        gdf_na.plot(color="#cccccc", ax=ax, linewidth=0)
    if not gdf_valid.empty:
        gdf_valid.plot(column=col, cmap=listed, norm=norm, ax=ax, linewidth=0)
    if boundary is not None and not boundary.empty:
        boundary.boundary.plot(ax=ax, color="#1a1a1a", linewidth=1.2, zorder=10)
        _set_extent(ax, boundary)
    _add_scale_bar(ax)
    _add_north_arrow(ax)
    ax.axis("off")


def _save_overview_figure(
    gdf: gpd.GeoDataFrame,
    class_bounds_iic: list,
    boundary: gpd.GeoDataFrame | None,
    iic_vals: np.ndarray,
    iic_mean: float,
    nat_iic_mean: float,
    city_dim_means: list,
    national_dim_means: list,
    dim_labels: list,
    city_name: str,
    porte: str | None,
    out_path: Path,
) -> None:
    """Landscape overview: IIC map (left) | radar + histogram stacked (right)."""
    class_colors = _iic_colors()
    labels = [f"{class_bounds_iic[i]:.3f}–{class_bounds_iic[i+1]:.3f}" for i in range(5)]
    porte_label  = porte if porte else "não identificado"

    fig = plt.figure(figsize=OVERVIEW_FIG_SIZE)
    gs  = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], hspace=0.38, wspace=0.12)
    ax_map  = fig.add_subplot(gs[:, 0])
    ax_rad  = fig.add_subplot(gs[0, 1], projection="polar")
    ax_hist = fig.add_subplot(gs[1, 1])

    # Left: IIC map — legend placed below the axes to avoid overlapping geometry
    _draw_hexagons(ax_map, gdf, "iic_final", class_colors, class_bounds_iic, boundary)
    ax_map.set_title("Índice de Injustiça Climática", fontsize=9, pad=4)
    handles = _legend_handles(class_colors, labels)
    ax_map.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                  ncol=3, fontsize=6,
                  title=f"Classificação por quintis · porte municipal: {porte_label}",
                  title_fontsize=6,
                  framealpha=1.0, edgecolor="#ccc",
                  handlelength=1.0, handleheight=0.85, borderpad=0.4, labelspacing=0.2)

    # Top right: radar
    N      = len(dim_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]
    cv     = city_dim_means + city_dim_means[:1]
    nv     = national_dim_means + national_dim_means[:1]
    ax_rad.plot(angles, cv, "o-", color=WRI_YELLOW, lw=2, label=city_name, zorder=3)
    ax_rad.fill(angles, cv, alpha=0.25, color=WRI_YELLOW, zorder=2)
    ax_rad.plot(angles, nv, "o--", color="#888", lw=1.5, label="Brasil", zorder=3)
    ax_rad.fill(angles, nv, alpha=0.12, color="#888", zorder=1)
    ax_rad.set_xticks(angles[:-1])
    ax_rad.set_xticklabels(dim_labels, size=8)
    rad_max = min(round(max(city_dim_means + national_dim_means) * 1.15, 1), 1.0)
    rad_ticks = [v for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] if v < rad_max]
    ax_rad.set_ylim(0, rad_max)
    ax_rad.set_yticks(rad_ticks)
    ax_rad.set_yticklabels([f"{v:.1f}" for v in rad_ticks], size=5.5)
    ax_rad.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=7.5)
    ax_rad.set_title("Perfil por Dimensão", fontsize=8.5, pad=8)

    # Bottom right: IIC histogram — x-axis adapts to actual data range
    iic_valid = iic_vals[~np.isnan(iic_vals)]
    x_max = min(round(float(np.percentile(iic_valid, 99)) * 1.1 + 0.05, 1), 1.0)
    x_max = max(x_max, iic_mean * 1.2)  # always show the city mean with room
    ax_hist.hist(iic_valid, bins=25, range=(0, x_max), color="#fde68a", edgecolor="none", alpha=0.85)
    ax_hist.axvline(iic_mean,     color=WRI_YELLOW, lw=2,   label=f"Município: {iic_mean:.3f}")
    ax_hist.axvline(nat_iic_mean, color="#444444",  lw=1.5, ls="--", label=f"Brasil: {nat_iic_mean:.3f}")
    ax_hist.set_xlabel("IIC Final", fontsize=8)
    ax_hist.set_ylabel("Hexágonos", fontsize=8)
    ax_hist.set_xlim(0, x_max)
    ax_hist.tick_params(labelsize=7)
    ax_hist.legend(fontsize=7.5, framealpha=0.85)
    ax_hist.spines[["top", "right"]].set_visible(False)
    ax_hist.set_title("Distribuição do IIC Final", fontsize=8.5, pad=4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_webp(fig, out_path)

# ==============================================================================
# COMBINED INDICATOR FIGURE (map left + histogram right)
# ==============================================================================
def _save_indicator_figure(
    gdf: gpd.GeoDataFrame,
    col: str,
    display_name: str,
    class_colors: list,
    class_bounds: list,
    boundary: gpd.GeoDataFrame | None,
    city_vals: np.ndarray,
    city_mean: float,
    nat_mean: float,
    dim_color: str,
    porte: str | None,
    out_path: Path,
) -> None:
    labels = [f"{class_bounds[i]:.3f}–{class_bounds[i+1]:.3f}" for i in range(5)]
    porte_label = porte if porte else "não identificado"

    fig = plt.figure(figsize=IND_FIG_SIZE)
    # Map spans full height on the left; histogram sits in the lower-right only,
    # creating a narrow horizontal strip that doesn't compete with the map visually.
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.35, 1],
        height_ratios=[1, 0.52],
        hspace=0.08,
        wspace=0.15,
    )
    ax_map  = fig.add_subplot(gs[:, 0])  # full-height left column
    ax_hist = fig.add_subplot(gs[1, 1])  # bottom-right only

    # Left: map
    _draw_hexagons(ax_map, gdf, col, class_colors, class_bounds, boundary)
    ax_map.set_title(display_name, fontsize=8, pad=4, color="#444")
    handles = _legend_handles(class_colors, labels)
    ax_map.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                  ncol=3, fontsize=6,
                  title=f"Classificação por quintis · porte municipal: {porte_label}",
                  title_fontsize=6,
                  framealpha=1.0, edgecolor="#ccc",
                  handlelength=1.0, handleheight=0.85, borderpad=0.4, labelspacing=0.2)

    # Right: histogram — narrow horizontal strip
    valid = city_vals[~np.isnan(city_vals)]
    counts, edges = np.histogram(valid, bins=20, range=(0, 1))
    bar_w = (edges[1] - edges[0]) * 0.78
    centers = (edges[:-1] + edges[1:]) / 2
    ax_hist.bar(centers, counts, width=bar_w, color=dim_color, alpha=0.55, edgecolor="none")
    ax_hist.axvline(city_mean, color=WRI_YELLOW, lw=1.8, label=f"Município: {city_mean:.3f}")
    ax_hist.axvline(nat_mean,  color="#666",     lw=1.2, ls="--", label=f"Brasil: {nat_mean:.3f}")
    ax_hist.set_xlabel(display_name, fontsize=7, color="#555")
    ax_hist.set_ylabel("Hexágonos",  fontsize=7, color="#555")
    ax_hist.set_xlim(0, 1)
    ax_hist.tick_params(labelsize=6.5, colors="#999")
    ax_hist.spines[["top", "right", "left"]].set_visible(False)
    ax_hist.spines["bottom"].set_color("#ccc")
    ax_hist.yaxis.grid(True, alpha=0.35, linestyle="-", lw=0.5, color="#bbb")
    ax_hist.set_axisbelow(True)
    ax_hist.set_title("Distribuição", fontsize=7, pad=3, color="#999", fontweight="normal")
    ax_hist.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.3),
        ncol=2, fontsize=6.5, framealpha=0,
        handlelength=1.2, handletextpad=0.4, columnspacing=1.0,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_webp(fig, out_path)

# ==============================================================================
# PER-CITY GENERATION
# ==============================================================================
def generate_city(
    city_row: dict,
    df_full: pd.DataFrame,
    nat: dict,
    municipalities: gpd.GeoDataFrame | None,
    porte_map: dict[str, str],
    quintile_data: dict,
    coastal_muns: set[str] | None = None,
) -> dict | None:
    nm_mun = city_row["nm_mun"]
    nm_uf  = city_row["nm_uf"]
    slug   = _slug(nm_mun, nm_uf)
    city_dir = IMGS_DIR / slug

    print(f"\n→ {nm_mun} / {nm_uf}")

    df = df_full[(df_full["nm_mun"] == nm_mun) & (df_full["nm_uf"] == nm_uf)].copy()
    df = df.drop_duplicates(subset="h3_id").reset_index(drop=True)
    if df.empty:
        print("  [WARNING] No data, skipping.")
        return None
    print(f"  {len(df):,} hexagons")

    gdf      = diag_utils.build_gdf(df)
    cd_mun   = df["cd_mun"].iloc[0] if "cd_mun" in df.columns else None
    boundary = get_boundary(cd_mun, municipalities)
    cd_str   = str(int(float(cd_mun))).zfill(7) if cd_mun is not None else None
    porte    = porte_map.get(cd_str) if cd_str else None
    print(f"  porte_munic: {porte}")

    iic_vals = df["iic_final"].dropna().values
    iic_mean = float(np.nanmean(iic_vals))
    nat_pct  = float((nat["_iic_all"] < iic_mean).mean() * 100)

    # --- Overview figure (map + radar + histogram combined) ---
    bounds_iic, _      = get_bounds_labels(porte, "iic_final", quintile_data)
    dim_labels         = [cfg.DIMENSION_META[ABBR_TO_DIM[a]]["abbr"] for a in DIM_ORDER]
    city_dim_means     = [float(df[a].mean()) if a in df.columns else 0.0 for a in DIM_ORDER]
    national_dim_means = [nat.get(a, {}).get("mean", 0.0) for a in DIM_ORDER]
    _save_overview_figure(
        gdf, bounds_iic, boundary,
        iic_vals, iic_mean, nat["iic_final"]["mean"],
        city_dim_means, national_dim_means, dim_labels, nm_mun,
        porte,
        city_dir / "overview.png",
    )
    print("  overview.png")

    # --- Per-dimension ---
    dims_data: dict = {}
    for abbr in DIM_ORDER:
        dim_key  = ABBR_TO_DIM[abbr]
        dim_meta = cfg.DIMENSION_META[dim_key]
        color    = DIM_COLORS[abbr]
        dim_dir  = city_dir / abbr

        dim_city_mean = float(df[abbr].mean()) if abbr in df.columns else None

        inds_data: dict = {}
        for ind_key in cfg.DIMENSIONS[dim_key]:
            if ind_key not in df.columns:
                continue
            # e3 (sea-level rise) is only applicable to coastal municipalities
            if ind_key == "e3" and coastal_muns is not None and cd_str not in coastal_muns:
                print(f"  {abbr}/{ind_key} skipped — {nm_mun} is not a coastal municipality")
                continue
            ind_vals = df[ind_key].dropna().values
            if len(ind_vals) == 0:
                continue

            ind_mean = float(np.nanmean(ind_vals))
            nat_ind  = nat.get(ind_key, {}).get("mean", 0.0)
            bounds_ind, _ = get_bounds_labels(porte, ind_key, quintile_data)

            _save_indicator_figure(
                gdf, ind_key,
                cfg.INDICATORS[ind_key]["display_name"],
                _class_colors_from_base(color), bounds_ind,
                boundary, ind_vals, ind_mean, nat_ind, color,
                porte,
                dim_dir / f"{ind_key}.png",
            )
            print(f"  {abbr}/{ind_key}.png")

            inds_data[ind_key] = {
                "label":         cfg.INDICATORS[ind_key]["display_name"],
                "description":   cfg.INDICATORS[ind_key]["description"],
                "city_mean":     fmt3(ind_mean),
                "national_mean": fmt3(nat_ind),
                "fig":           f"imgs/{slug}/{abbr}/{ind_key}.webp",
            }

        dims_data[abbr] = {
            "label":         dim_meta["display_name"],
            "abbr_upper":    dim_meta["abbr"],
            "invert":        dim_meta["invert"],
            "color":         color,
            "city_mean":     fmt3(dim_city_mean),
            "national_mean": fmt3(nat.get(abbr, {}).get("mean", 0.0)),
            "indicators":    inds_data,
        }

    return {
        "nm_mun":       nm_mun,
        "nm_uf":        nm_uf,
        "slug":         slug,
        "porte":        porte or "—",
        "n_hexagons":   len(df),
        "iic_mean":     fmt3(iic_mean),
        "iic_max":      fmt3(float(np.nanmax(iic_vals))),
        "iic_min":      fmt3(float(np.nanmin(iic_vals))),
        "national_pct": round(nat_pct, 1),
        "group":        city_row.get("group", ""),
        "dims":         dims_data,
        "imgs": {
            "overview": f"imgs/{slug}/overview.webp",
        },
    }

# ==============================================================================
# NATIONAL DISTRIBUTION FIGURE
# ==============================================================================
def _save_national_distribution(df_full: pd.DataFrame, nat_iic_mean: float, out_path: Path) -> None:
    vals = df_full["iic_final"].dropna().values
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(vals, bins=50, range=(0, 1), color="#fde68a", edgecolor="none", alpha=0.9)
    ax.axvline(nat_iic_mean, color=WRI_YELLOW, lw=2, label=f"Média Brasil: {nat_iic_mean:.3f}")
    ax.set_xlabel("Índice de Injustiça Climática", fontsize=10)
    ax.set_ylabel("Número de hexágonos", fontsize=10)
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=9)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x)))
    )
    ax.legend(fontsize=9.5, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Distribuição nacional do Índice de Injustiça Climática", fontsize=11, pad=8)
    ax.text(0.99, 0.97, f"Total: {len(vals):,} hexágonos",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="#777")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_webp(fig, out_path)
    print(f"  Saved: {out_path.with_suffix('.webp').name}")


# ==============================================================================
# SANKEY JSON (Plotly, embedded in HTML via CDN)
# ==============================================================================
def _build_sankey_json() -> str:
    ip_w = round(25 / 5, 4)   # 5.0
    iv_w = round(25 / 5, 4)   # 5.0
    ie_w = round(25 / 5, 4)   # 5.0
    ig_w = round(25 / 8, 4)   # 3.125

    node_labels = [
        "IIC",
        "IP — Grupos Prioritários", "IV — Vulnerabilidade",
        "IE — Exposição Climática", "IG — Gestão Municipal",
        # IP (5–9)
        "Mulheres negras chefes de domicílio", "População negra",
        "Indígenas e quilombolas", "Idosos (60 anos ou mais)", "Crianças (9 anos ou menos)",
        # IV (10–14)
        "Baixa renda", "Moradia precária", "Educação", "Acesso à saúde", "Infraestrutura",
        # IE (15–19)
        "Deslizamentos de terra", "Inundações, alagamentos e enxurradas",
        "Elevação do nível do mar", "Calor extremo", "Focos de queimadas",
        # IG (20–27)
        "Investimento", "Planejamento", "Participação", "Governança",
        "Resposta", "Informação", "Reconhecimento", "Reparação",
    ]
    node_colors = (
        ["#2c2c2c"]               # IIC
        + ["#eb8026", "#3855a3", "#32864b", "#9b216c"]   # dims
        + ["#f5c896"] * 5         # IP indicators
        + ["#adb8dc"] * 5         # IV indicators
        + ["#a2cbb0"] * 5         # IE indicators
        + ["#d4a3be"] * 8         # IG indicators
    )

    # Flow: indicators → dimensions → IIC
    tgt = [0,0,0,0] + [1]*5 + [2]*5 + [3]*5 + [4]*8
    src = (
        [1,2,3,4]
        + list(range(5,10)) + list(range(10,15))
        + list(range(15,20)) + list(range(20,28))
    )
    val = ([25]*4
           + [ip_w]*5 + [iv_w]*5 + [ie_w]*5 + [ig_w]*8)
    lnk_colors = (
        ["rgba(235,128,38,0.4)", "rgba(56,85,163,0.4)",
         "rgba(50,134,75,0.4)", "rgba(155,33,108,0.4)"]
        + ["rgba(235,128,38,0.22)"] * 5
        + ["rgba(56,85,163,0.22)"]  * 5
        + ["rgba(50,134,75,0.22)"]  * 5
        + ["rgba(155,33,108,0.22)"] * 8
    )

    fig = {
        "data": [{
            "type": "sankey", "orientation": "h",
            "node": {
                "pad": 14, "thickness": 18,
                "line": {"color": "white", "width": 0.3},
                "label": node_labels,
                "color": node_colors,
            },
            "link": {"source": src, "target": tgt, "value": val, "color": lnk_colors},
        }],
        "layout": {
            "font": {"family": "Arial, sans-serif", "size": 12},
            "paper_bgcolor": "white",
            "margin": {"l": 10, "r": 10, "t": 10, "b": 10},
            "height": 520,
        },
    }
    return json.dumps(fig)


# ==============================================================================
# METHODOLOGICAL NOTES
# ==============================================================================
def _render_methodological_notes() -> str:
    notes_path = SCRIPT_DIR / "methodological_notes.md"
    if not notes_path.exists():
        print("  [WARNING] methodological_notes.md not found — skipping notes section.")
        return ""
    text = notes_path.read_text(encoding="utf-8")
    try:
        import markdown as md_lib
        return md_lib.markdown(text, extensions=["tables", "fenced_code"])
    except ImportError:
        print("  [WARNING] `markdown` package not installed — notes will render as plain text. Run: pip install markdown")
        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f'<div class="notes-plain"><pre>{escaped}</pre></div>'


# ==============================================================================
# HTML RENDERING
# ==============================================================================
def render_html(cities_data: list, sankey_json: str, nat_dist_img: str, notes_html: str) -> str:
    from jinja2 import Environment, FileSystemLoader
    env  = Environment(loader=FileSystemLoader(str(TMPL_DIR)), autoescape=True)
    tmpl = env.get_template("report.html")
    return tmpl.render(
        cities=cities_data,
        generated_at=date.today().strftime("%d/%m/%Y"),
        sankey_json=sankey_json,
        nat_dist_img=nat_dist_img,
        notes_html=notes_html,
    )


CSS = """\
:root {
  --sidebar-w: 240px;
  --yellow: #F0AB00;
  --green: #32864B;
  --dark: #1a1a2e;
  --text: #222;
  --muted: #666;
  --border: #e0e0e0;
  --font: Arial, Helvetica, sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: var(--font); color: var(--text); background: #f2f2f2; }

/* ---- SIDEBAR ---- */
#sidebar {
  position: fixed; top: 0; left: 0;
  width: var(--sidebar-w); height: 100vh;
  overflow-y: auto;
  background: #fff;
  border-right: 1px solid var(--border);
}
.sidebar-brand {
  padding: 1rem 1rem 0.75rem;
  border-bottom: 3px solid var(--yellow);
  margin-bottom: 0.5rem;
}
.sidebar-brand .brand-wri {
  display: block; font-size: 10px; font-weight: 700;
  color: var(--green); letter-spacing: 0.1em; text-transform: uppercase;
}
.sidebar-brand .brand-title {
  display: block; font-size: 12px; color: #333; margin-top: 3px; line-height: 1.4;
}
.toc { list-style: none; }
.toc > li { border-bottom: 1px solid #f0f0f0; }
.toc a {
  display: block; padding: 0.38rem 1rem;
  color: #444; text-decoration: none; font-size: 12px;
  transition: background 0.1s;
}
.toc a:hover { background: #fef9ed; color: #111; }
.toc-city > a {
  font-weight: 700; font-size: 12.5px; color: #222;
  display: flex; justify-content: space-between; align-items: center;
}
.toc-city > a::after { content: "▸"; font-size: 9px; color: #bbb; flex-shrink: 0; }
.toc-city.open > a::after { content: "▾"; }
.toc-sub {
  list-style: none; background: #fafafa;
  max-height: 0; overflow: hidden;
  transition: max-height 0.25s ease;
}
.toc-city.open .toc-sub { max-height: 300px; }
.toc-sub a { padding: 0.27rem 1rem 0.27rem 1.6rem; font-size: 11px; color: #777; }
.toc-sub a:hover { color: #222; background: #f5f0e8; }

/* ---- MAIN ---- */
#content { margin-left: var(--sidebar-w); }
.page {
  padding: 3rem 3.5rem;
  border-bottom: 2px solid var(--border);
  background: #fff;
  max-width: 1100px;
}

/* ---- COVER ---- */
.cover-page {
  min-height: 100vh; background: #fff;
  border-left: 8px solid var(--yellow);
  border-bottom: 2px solid var(--border);
  display: flex; align-items: center; max-width: none;
}
.cover-inner { max-width: 700px; }
.cover-tag {
  font-size: 11px; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--green); margin-bottom: 1.5rem;
}
.cover-page h1 {
  font-size: 2.5rem; font-weight: 700; color: var(--dark);
  line-height: 1.2; margin-bottom: 1rem;
}
.cover-subtitle { font-size: 1.05rem; color: #555; margin-bottom: 0.4rem; }
.cover-meta     { font-size: 0.85rem; color: #999; margin-bottom: 2.5rem; }
.cover-cities   { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.city-chip {
  background: #f4f4f4; border: 1px solid #ddd; color: #444;
  border-radius: 20px; padding: 5px 14px; font-size: 12px;
  text-decoration: none; transition: background 0.12s, color 0.12s;
}
.city-chip:hover { background: var(--yellow); color: #fff; border-color: var(--yellow); }

/* ---- INTRO ---- */
.intro-text { max-width: 780px; }
.intro-text p { font-size: 0.95rem; line-height: 1.7; color: #333; margin-bottom: 1rem; }
.dim-cards {
  display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1.75rem;
}
.dim-card { border-radius: 8px; padding: 1rem 1.25rem; border-left: 4px solid; }
.dim-card h4 { font-size: 0.85rem; font-weight: 700; margin-bottom: 0.35rem; }
.dim-card p  { font-size: 0.82rem; color: #555; line-height: 1.55; }

/* ---- SECTION TITLES ---- */
h1.section-title {
  font-size: 1.75rem; font-weight: 700;
  margin-bottom: 1.5rem; padding-bottom: 0.5rem;
  border-bottom: 3px solid var(--yellow);
}
h2.section-title {
  font-size: 1.25rem; font-weight: 700;
  margin-bottom: 1.25rem; padding-bottom: 0.4rem;
  border-bottom: 3px solid;
}

/* ---- CITY HEADER ---- */
.city-header-page {
  background: #fff;
  border-left: 8px solid var(--yellow);
  border-bottom: 2px solid var(--border);
  min-height: 26vh; display: flex; align-items: center; max-width: none;
}
.city-header-inner { max-width: 900px; }
.city-uf {
  font-size: 11px; color: var(--green); font-weight: 700;
  letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.4rem;
}
.city-header-page h1 { font-size: 2.1rem; color: var(--dark); margin-bottom: 1.5rem; }
.city-quick-stats { display: flex; gap: 2.5rem; flex-wrap: wrap; }
.quick-stat { display: flex; flex-direction: column; }
.stat-value {
  font-size: 1.7rem; font-weight: 700; color: var(--dark);
  font-variant-numeric: tabular-nums; letter-spacing: -0.01em;
}
.stat-label { font-size: 11px; color: #888; margin-top: 2px; }

/* ---- OVERVIEW ---- */
.overview-block img {
  width: 100%; display: block;
  border-radius: 6px; border: 1px solid var(--border);
}

/* ---- DIMENSION SCORE BAR ---- */
.dim-score-bar {
  display: flex; gap: 2rem; align-items: center; flex-wrap: wrap;
  background: #f8f8f8; border-radius: 6px;
  padding: 0.7rem 1rem; margin-bottom: 1.5rem; font-size: 0.9rem;
}
.dim-score-bar span strong { font-size: 1.05rem; font-variant-numeric: tabular-nums; }

/* ---- INDICATOR BLOCK ---- */
.indicator-block {
  margin-bottom: 2rem; padding-bottom: 2rem;
  border-bottom: 1px solid var(--border);
}
.indicator-block:last-child { border-bottom: none; margin-bottom: 0; }
.ind-header { display: flex; align-items: center; gap: 0.65rem; margin-bottom: 0.35rem; flex-wrap: wrap; }
.ind-key {
  display: inline-block; color: #fff; font-size: 11px; font-weight: 700;
  border-radius: 4px; padding: 3px 9px; flex-shrink: 0;
}
.ind-label { font-weight: 700; font-size: 0.95rem; }
.ind-means { font-size: 11px; color: var(--muted); margin-left: auto; font-variant-numeric: tabular-nums; }
.ind-description { font-size: 0.82rem; color: var(--muted); line-height: 1.55; margin-bottom: 0.75rem; }
.ind-fig img { width: 100%; display: block; border-radius: 6px; border: 1px solid var(--border); }
.invert-note { font-size: 12px; color: var(--muted); font-weight: 400; margin-left: 0.5rem; }

/* ---- NATIONAL DISTRIBUTION ---- */
.nat-dist-block { margin-top: 2.5rem; }
.nat-dist-block img {
  width: 100%; display: block;
  border-radius: 6px; border: 1px solid var(--border);
  margin-top: 0.75rem;
}
.subsection-title {
  font-size: 1rem; font-weight: 700; color: var(--dark);
  margin-bottom: 0.25rem;
}

/* ---- SANKEY ---- */
.sankey-wrap {
  border: 1px solid var(--border); border-radius: 6px;
  padding: 0.5rem; background: #fff; margin-top: 1rem;
}
.sankey-note {
  font-size: 0.82rem; color: var(--muted); margin-bottom: 1rem; line-height: 1.55;
}

/* ---- PDF BUTTON ---- */
.btn-pdf {
  display: inline-flex; align-items: center; gap: 0.4rem;
  background: var(--yellow); color: #fff;
  border: none; border-radius: 6px;
  padding: 0.5rem 1.1rem; font-size: 0.85rem; font-weight: 600;
  cursor: pointer; text-decoration: none; font-family: var(--font);
  transition: background 0.15s; margin-top: 1.5rem;
}
.btn-pdf:hover { background: #d49600; }
.sidebar-pdf {
  padding: 0.85rem 1rem;
  border-top: 1px solid var(--border);
}
.sidebar-pdf .btn-pdf {
  width: 100%; justify-content: center;
  margin-top: 0; font-size: 0.8rem; padding: 0.45rem;
}

/* ---- CITY GROUPS (sidebar + table) ---- */
.toc-group-label {
  padding: 0.5rem 1rem 0.2rem;
  font-size: 10px; font-weight: 700; color: var(--green);
  letter-spacing: 0.08em; text-transform: uppercase;
  border-bottom: none;
}
.ct-group-row td {
  background: #f5f5f5; font-size: 11px; font-weight: 700;
  color: var(--green); letter-spacing: 0.06em; text-transform: uppercase;
  padding: 0.4rem 0.9rem; border-bottom: 1px solid var(--border);
}
.cover-group-label {
  width: 100%; font-size: 10px; font-weight: 700; color: var(--green);
  letter-spacing: 0.1em; text-transform: uppercase;
  margin-top: 1rem; margin-bottom: 0.3rem;
}
.cover-group-label:first-child { margin-top: 0; }

/* ---- ACTIVE SIDEBAR LINK ---- */
.toc a.active {
  background: #fef3cc;
  color: #111;
  border-left: 3px solid var(--yellow);
  padding-left: calc(1rem - 3px);
  font-weight: 600;
}
.toc-sub a.active {
  background: #fef3cc;
  color: #333;
  border-left: 3px solid var(--yellow);
  padding-left: calc(1.6rem - 3px);
  font-weight: 600;
}

/* ---- BACK TO TOP ---- */
#back-to-top {
  position: fixed; bottom: 2rem; right: 2rem; z-index: 1000;
  width: 42px; height: 42px;
  background: var(--yellow); color: white; border: none;
  border-radius: 50%; cursor: pointer; font-size: 20px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.18);
  opacity: 0; pointer-events: none; transition: opacity 0.2s;
  display: flex; align-items: center; justify-content: center;
}
#back-to-top.visible { opacity: 1; pointer-events: auto; }
#back-to-top:hover { background: #d49600; }

/* ---- COMPARISON TABLE ---- */
.comparison-table-wrap { overflow-x: auto; margin-top: 1rem; }
.comparison-table {
  width: 100%; border-collapse: collapse; font-size: 0.85rem;
}
.comparison-table th {
  background: var(--dark); color: white;
  padding: 0.55rem 0.9rem; text-align: left;
  font-weight: 600; white-space: nowrap;
}
.comparison-table td {
  padding: 0.5rem 0.9rem; border-bottom: 1px solid var(--border);
  vertical-align: middle;
}
.comparison-table tr:last-child td { border-bottom: none; }
.comparison-table tr:hover td { background: #fef9ed; }
.ct-city { font-weight: 700; }
.ct-num { font-variant-numeric: tabular-nums; }
.ct-iic { font-variant-numeric: tabular-nums; font-weight: 700; font-size: 0.95rem; }
.ct-dim { font-variant-numeric: tabular-nums; font-size: 0.84rem; font-weight: 600; }

/* ---- LIGHTBOX ---- */
#lightbox {
  display: none; position: fixed; inset: 0; z-index: 4000;
  background: rgba(0,0,0,0.82);
  align-items: center; justify-content: center; cursor: zoom-out;
}
#lightbox.active { display: flex; }
#lightbox img {
  max-width: 92vw; max-height: 90vh;
  border-radius: 6px; box-shadow: 0 8px 40px rgba(0,0,0,0.5);
  cursor: default;
}
#lightbox-close {
  position: absolute; top: 1rem; right: 1.5rem;
  color: white; font-size: 2.2rem; cursor: pointer;
  background: none; border: none; line-height: 1; opacity: 0.8;
}
#lightbox-close:hover { opacity: 1; }
.ind-fig img, .overview-block img { cursor: zoom-in; }

/* ---- SANKEY PRINT FALLBACK (hidden on screen) ---- */
.sankey-print-img { display: none; }

/* ---- PRINT ---- */
@media print {
  /* Hide UI chrome */
  #sidebar, #back-to-top, #lightbox, #pw-overlay { display: none !important; }
  .btn-pdf { display: none !important; }
  #content { margin-left: 0 !important; }
  body { font-size: 10pt; color: #111; }

  .page { border-bottom: 1px solid #ddd; padding: 1.4rem 2rem; max-width: none; box-shadow: none; }

  /* Cover alone on page 1 */
  .cover-page {
    min-height: 100vh; border-left: 4px solid var(--yellow);
    break-after: page; display: flex; align-items: center;
  }

  /* Each city starts on a new page */
  .city-header-page { break-before: page; min-height: auto; border-left: 4px solid var(--yellow); }

  /* Notas start on a new page */
  .print-page-break { break-before: page; }

  /* Avoid orphaned content inside blocks */
  .indicator-block { break-inside: avoid; }
  .overview-block { break-inside: avoid; }
  .ind-fig { break-inside: avoid; }
  .nat-dist-block { break-inside: avoid; }
  .dim-score-bar { break-inside: avoid; }
  h1, h2 { break-after: avoid; }

  /* Sankey: hide Plotly, show static image */
  .sankey-wrap { display: none !important; }
  .sankey-print-img {
    display: block !important; width: 100%;
    border-radius: 6px; border: 1px solid var(--border); margin-top: 1rem;
  }

  /* Comparison table: compact */
  .comparison-table { font-size: 8pt; }
  .comparison-table th, .comparison-table td { padding: 0.3rem 0.5rem; }

  /* Dim cards: 1 column */
  .dim-cards { grid-template-columns: 1fr; }
}

/* ---- METHODOLOGICAL NOTES ---- */
.notes-body h1 { font-size: 1.3rem; font-weight: 700; color: var(--dark); margin: 2rem 0 0.75rem; padding-bottom: 0.4rem; border-bottom: 2px solid var(--border); }
.notes-body h2 { font-size: 1rem; font-weight: 700; color: var(--dark); margin: 1.5rem 0 0.5rem; }
.notes-body p  { font-size: 0.88rem; line-height: 1.7; color: #333; margin-bottom: 0.75rem; }
.notes-body strong { color: var(--dark); }
.notes-body em { color: #555; }
.notes-body code {
  font-family: "Courier New", monospace; font-size: 0.82rem;
  background: #f4f4f4; border-radius: 3px; padding: 1px 5px;
}
.notes-body table {
  width: 100%; border-collapse: collapse; font-size: 0.82rem; margin: 1rem 0;
}
.notes-body th {
  background: var(--dark); color: white;
  padding: 0.45rem 0.75rem; text-align: left; font-weight: 600;
}
.notes-body td { padding: 0.4rem 0.75rem; border-bottom: 1px solid var(--border); vertical-align: top; }
.notes-body tr:hover td { background: #fef9ed; }
.notes-body hr { border: none; border-top: 2px solid var(--border); margin: 2rem 0; }
.notes-body ul, .notes-body ol { padding-left: 1.4rem; margin-bottom: 0.75rem; }
.notes-body li { font-size: 0.88rem; line-height: 1.7; color: #333; }

/* ---- FOOTER ---- */
.report-footer {
  text-align: center; font-size: 11px; color: #999;
  padding: 1.5rem; background: #fafafa; max-width: none;
}
"""

# ==============================================================================
# MAIN
# ==============================================================================
def main() -> None:
    for d in [DOCS_DIR, IMGS_DIR, ASSETS_DIR, TMPL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    df_full = load_full_data()
    nat     = compute_national_stats(df_full)
    muns    = load_municipalities()

    print("\nLoading coastal municipalities...")
    coastal_muns = load_coastal_muns()
    print(f"  {len(coastal_muns):,} coastal municipalities loaded.")

    print("\nLoading porte_munic mapping...")
    porte_map = load_porte_map()
    print(f"  {len(porte_map):,} municipalities mapped.")

    print("Computing porte quintiles...")
    ind_cols = ["iic_final", "ip", "iv", "ie", "ig"] + ALL_IND_KEYS
    quintile_data = compute_porte_quintiles(df_full, porte_map, ind_cols)
    print(f"  {len(quintile_data)} porte groups computed.")

    print("\nGenerating national distribution figure...")
    nat_dist_path = IMGS_DIR / "national_distribution.webp"
    _save_national_distribution(df_full, nat["iic_final"]["mean"], nat_dist_path)

    print("Building Sankey JSON...")
    sankey_json = _build_sankey_json()

    cities_data = []
    for city_row in CITIES:
        data = generate_city(city_row, df_full, nat, muns, porte_map, quintile_data, coastal_muns)
        if data:
            cities_data.append(data)

    print("\nRendering methodological notes...")
    notes_html = _render_methodological_notes()

    print("\nRendering HTML...")
    html = render_html(cities_data, sankey_json, "imgs/national_distribution.webp", notes_html)
    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")
    (ASSETS_DIR / "style.css").write_text(CSS, encoding="utf-8")

    print(f"\nDone! {len(cities_data)} cities.")
    print(f"Report → {DOCS_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
