"""
Mapas H3 do IIC v2.0 por município.

Lê as cidades de diagnose/cities.json e gera para cada uma:
  - IIC final + 4 sub-índices (5 classes de cor discretas)
  - Indicadores individuais por dimensão

Sobrepõe o contorno municipal via malha IBGE 2024 (municipios.gpkg).

Saídas: cfg.FIGURES_DIR / "maps" / map_*_{ts}.png
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
import matplotlib.gridspec as mgridspec
import geopandas as gpd
from shapely.geometry import Polygon
from pathlib import Path
import h3

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src import config as cfg

# ==============================================================================
# CAMINHOS E PARÂMETROS
# ==============================================================================
DPI          = 150
CLASS_BOUNDS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CLASS_LABELS = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

MAPS_DIR = cfg.FIGURES_DIR / "maps"
MAPS_DIR.mkdir(parents=True, exist_ok=True)

MUNICIPALITIES_GPKG = cfg.RAW_DIR / "ibge" / "malha_municipal" / "2024" / "municipios.gpkg"

with open(SCRIPT_DIR / "cities.json", encoding="utf-8") as _f:
    CITIES = json.load(_f).get("cities", [])

_results = sorted(
    cfg.FILES["output"]["results_dir"].glob("br_h3_iic_v2_0_*.parquet"),
    key=lambda p: p.stat().st_mtime, reverse=True,
)
if not _results:
    raise FileNotFoundError(
        f"Nenhum resultado encontrado em:\n  {cfg.FILES['output']['results_dir']}\n"
        "Execute `python run.py` primeiro para gerar os resultados."
    )
RESULTS_FILE = _results[0]

_ts_match = re.search(r"(\d{8}_\d{6})", RESULTS_FILE.stem)
FILE_TS = _ts_match.group(1) if _ts_match else RESULTS_FILE.stem

ALL_INDICATORS = [
    "p1", "p2", "p3", "p4", "p5",
    "v1", "v2", "v3", "v4", "v5",
    "e1", "e2", "e3", "e4", "e5",
    "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8",
]

DIMS = {
    "ip": {
        "label": "IP - Grupos Prioritários",
        "color": "#C0392B",
        "indicators": ["p1", "p2", "p3", "p4", "p5"],
        "ind_labels": {
            "p1": "p1 - Mulheres negras chefes de domicílio",
            "p2": "p2 - População negra",
            "p3": "p3 - Indígenas e quilombolas",
            "p4": "p4 - Idosos (60 anos ou mais)",
            "p5": "p5 - Crianças (9 anos ou menos)",
        },
    },
    "iv": {
        "label": "IV - Vulnerabilidade Socioeconômica",
        "color": "#E67E22",
        "indicators": ["v1", "v2", "v3", "v4", "v5"],
        "ind_labels": {
            "v1": "v1 - Baixa renda",
            "v2": "v2 - Moradia precária",
            "v3": "v3 - Educação",
            "v4": "v4 - Acesso à saúde",
            "v5": "v5 - Infraestrutura",
        },
    },
    "ie": {
        "label": "IE - Exposição a Riscos Climáticos",
        "color": "#27AE60",
        "indicators": ["e1", "e2", "e3", "e4", "e5"],
        "ind_labels": {
            "e1": "e1 - Deslizamentos de terra",
            "e2": "e2 - Inundações, alagamentos e enxurradas",
            "e3": "e3 - Elevação do nível do mar",
            "e4": "e4 - Calor extremo",
            "e5": "e5 - Queimadas",
        },
    },
    "ig": {
        "label": "IG - Gestão Municipal (invertida no IIC)",
        "color": "#2980B9",
        "indicators": ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8"],
        "ind_labels": {
            "g1": "g1 - Investimento ambiental",
            "g2": "g2 - Plano de contingência",
            "g3": "g3 - Participação em NUPDECs",
            "g4": "g4 - Conselhos municipais",
            "g5": "g5 - Sistemas de alerta",
            "g6": "g6 - Mapeamento e zoneamento de risco",
            "g7": "g7 - Cadastro de famílias em risco",
            "g8": "g8 - Políticas de direitos humanos",
        },
    },
}

# ==============================================================================
# MALHA MUNICIPAL
# ==============================================================================
_MUNICIPALITIES_GDF = None


def _load_municipalities() -> gpd.GeoDataFrame | None:
    global _MUNICIPALITIES_GDF
    if _MUNICIPALITIES_GDF is not None:
        return _MUNICIPALITIES_GDF
    if not MUNICIPALITIES_GPKG.exists():
        print(f"  [AVISO] Malha municipal não encontrada: {MUNICIPALITIES_GPKG}")
        return None
    try:
        print("Carregando malha municipal IBGE 2024...")
        gdf = gpd.read_file(MUNICIPALITIES_GPKG)
        _MUNICIPALITIES_GDF = gdf.to_crs("EPSG:4326")
        print(f"  {len(_MUNICIPALITIES_GDF):,} municípios carregados.")
        return _MUNICIPALITIES_GDF
    except Exception as e:
        print(f"  [AVISO] Erro ao carregar malha municipal: {e}")
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
# HELPERS DE MAPA
# ==============================================================================

def _class_colors_from_base(base_color: str) -> list:
    light = np.array(mcolors.to_rgba("#f7f7f7"))
    dark  = np.array(mcolors.to_rgba(base_color))
    return [tuple(light + (dark - light) * i / 4) for i in range(5)]


def _class_colors_from_cmap(cmap_name: str) -> list:
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i / 4) for i in range(5)]


def _plot_classified_panel(gdf, col, ax, title, class_colors,
                            show_legend: bool = False,
                            boundary: gpd.GeoDataFrame = None) -> None:
    listed = ListedColormap(class_colors)
    norm   = BoundaryNorm(CLASS_BOUNDS, ncolors=len(class_colors), clip=True)

    gdf_valid = gdf[gdf[col].notna()]
    gdf_na    = gdf[gdf[col].isna()]
    if not gdf_na.empty:
        gdf_na.plot(color="#cccccc", ax=ax, linewidth=0)
    if not gdf_valid.empty:
        gdf_valid.plot(column=col, cmap=listed, norm=norm, ax=ax, linewidth=0)

    if boundary is not None and not boundary.empty:
        boundary.boundary.plot(ax=ax, color="#1a1a1a", linewidth=1.2, zorder=10)

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


def _h3_to_polygon(h3_id: str) -> Polygon:
    coords = h3.cell_to_boundary(h3_id)
    return Polygon([(lng, lat) for lat, lng in coords])


def _build_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    df = df.copy()
    df["geometry"] = df["h3_id"].apply(_h3_to_polygon)
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")


def _city_slug(nm_mun: str, nm_uf: str) -> str:
    import unicodedata
    slug = unicodedata.normalize("NFKD", nm_mun).encode("ascii", "ignore").decode()
    return f"{slug.lower().replace(' ', '_')}_{nm_uf.split()[-1].lower()}"


def _save_map(fig: plt.Figure, name: str) -> None:
    path = MAPS_DIR / f"{name}_{FILE_TS}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {path.name}")


# ==============================================================================
# CARREGAMENTO
# ==============================================================================

def load_city_data(nm_mun: str, nm_uf: str) -> pd.DataFrame:
    cols = (["h3_id", "nm_mun", "nm_uf", "cd_mun", "iic_final",
             "ip", "iv", "ie", "ig"] + ALL_INDICATORS)
    df     = pd.read_parquet(RESULTS_FILE, columns=cols)
    mask   = (df["nm_mun"] == nm_mun) & (df["nm_uf"] == nm_uf)
    result = df[mask].drop_duplicates(subset="h3_id").reset_index(drop=True)
    print(f"  {nm_mun}/{nm_uf}: {len(result):,} hexágonos carregados.")
    return result


# ==============================================================================
# FIGURAS
# ==============================================================================

def fig_map_city(nm_mun: str, nm_uf: str, gdf: gpd.GeoDataFrame,
                 boundary: gpd.GeoDataFrame = None) -> None:
    PLOTS = [
        ("iic_final", "IIC Final",               _class_colors_from_cmap("RdYlGn_r")),
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
        _plot_classified_panel(gdf, col, ax, title, class_colors,
                               show_legend=True, boundary=boundary)

    fig.suptitle(
        f"{nm_mun} / {nm_uf}\n"
        f"Distribuição Espacial H3 (res. 9) — IIC e Sub-índices",
        fontsize=14, fontweight="bold", y=1.01,
    )
    _save_map(fig, f"map_{_city_slug(nm_mun, nm_uf)}")


def fig_map_city_indicators(nm_mun: str, nm_uf: str,
                             gdf: gpd.GeoDataFrame,
                             boundary: gpd.GeoDataFrame = None) -> None:
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
                                   class_colors=class_colors, show_legend=False,
                                   boundary=boundary)
        for j in range(len(indicators), len(axes_flat)):
            axes_flat[j].set_visible(False)

        _shared_legend(fig, class_colors, y_anchor=0.01)
        fig.suptitle(
            f"{nm_mun} / {nm_uf}  —  {meta['label']}\n"
            f"Indicadores individuais (5 classes, 0–1)",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0.07, 1, 0.93])
        _save_map(fig, f"map_{_city_slug(nm_mun, nm_uf)}_ind_{dim_key}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print(f"Mapas IIC v2.0  |  {RESULTS_FILE.name}")
    print(f"Timestamp: {FILE_TS}")
    print("=" * 60)

    print(f"\nCidades configuradas: {len(CITIES)}")
    for city in CITIES:
        nm_mun = city["nm_mun"]
        nm_uf  = city["nm_uf"]
        print(f"\n→ {nm_mun} / {nm_uf}")

        city_df = load_city_data(nm_mun, nm_uf)
        if city_df.empty:
            print("  [AVISO] Nenhum dado encontrado, pulando.")
            continue

        cd_mun   = city_df["cd_mun"].iloc[0] if "cd_mun" in city_df.columns else None
        boundary = _get_municipality_boundary(cd_mun)
        print(f"  Contorno municipal: {'carregado' if boundary is not None else 'não encontrado'} (cd_mun={cd_mun})")

        print(f"  Convertendo {len(city_df):,} hexágonos para geometria...")
        gdf = _build_gdf(city_df)

        fig_map_city(nm_mun, nm_uf, gdf, boundary=boundary)
        fig_map_city_indicators(nm_mun, nm_uf, gdf, boundary=boundary)

    n_maps = len(list(MAPS_DIR.glob("*.png")))
    print(f"\nConcluído! {n_maps} PNGs em {MAPS_DIR}")


if __name__ == "__main__":
    main()
