"""
diagnose/diagnose.py
--------------------
Diagnosticos estatisticos e espaciais do IIC v2.0.

Cidades a mapear: definidas em diagnose/config.json.

Gera automaticamente a partir do parquet de resultados mais recente:
  [scatter] Cada indicador vs. seu sub-indice pai
  [scatter] Cada sub-indice vs. IIC final
  [scatter] Distribuicao do IIC final
  [scatter] Matriz de dispersao entre sub-indices
  [map]     Mapa H3 da cidade - IIC final e sub-indices (5 classes)
  [map]     Mapa H3 da cidade - indicadores individuais por dimensao

Saidas:
  cfg.FIGURES_DIR / "graphs" / scatter_*_{ts}.png
  cfg.FIGURES_DIR / "maps"   / map_*_{ts}.png
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
from matplotlib.colors import BoundaryNorm, ListedColormap, LinearSegmentedColormap
import matplotlib.gridspec as mgridspec
from pathlib import Path
from scipy import stats
import h3
import geopandas as gpd
from shapely.geometry import Polygon

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src import config as cfg

# -- Config de cidades ---------------------------------------------------------
with open(SCRIPT_DIR / "cities.json", encoding="utf-8") as _f:
    _DIAG_CONFIG = json.load(_f)
CITIES = _DIAG_CONFIG.get("cities", [])

# -- Parametros ----------------------------------------------------------------
SAMPLE_N     = 300_000
SEED         = 42
DPI          = 150
CLASS_BOUNDS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CLASS_LABELS = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

GRAPHS_DIR = cfg.FIGURES_DIR / "graphs"
MAPS_DIR   = cfg.FIGURES_DIR / "maps"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
MAPS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = sorted(
    cfg.FILES["output"]["results_dir"].glob("br_h3_iic_v2_0_*.parquet"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)[0]

_ts_match = re.search(r"(\d{8}_\d{6})", RESULTS_FILE.stem)
FILE_TS = _ts_match.group(1) if _ts_match else RESULTS_FILE.stem

# -- Metadados de dimensoes ----------------------------------------------------
DIMS = {
    "ip": {
        "label"     : "IP - Grupos Prioritarios",
        "color"     : "#C0392B",
        "indicators": ["p1", "p2", "p3", "p4", "p5"],
        "ind_labels": {
            "p1": "p1 - Mulheres pretas/pardas chefes de domicilio",
            "p2": "p2 - Populacao negra (pretas e pardas)",
            "p3": "p3 - Indigenas e quilombolas",
            "p4": "p4 - Idosos (60+ anos)",
            "p5": "p5 - Criancas (< 14 anos)",
        },
    },
    "iv": {
        "label"     : "IV - Vulnerabilidade Socioeconomica",
        "color"     : "#E67E22",
        "indicators": ["v1", "v2", "v3", "v4", "v5"],
        "ind_labels": {
            "v1": "v1 - Renda (invertida: menor renda = maior vulnerabilidade)",
            "v2": "v2 - Moradia precaria",
            "v3": "v3 - Educacao",
            "v4": "v4 - Acesso a saude",
            "v5": "v5 - Infraestrutura basica",
        },
    },
    "ie": {
        "label"     : "IE - Exposicao a Riscos Climaticos",
        "color"     : "#27AE60",
        "indicators": ["e1", "e2", "e3", "e4", "e5"],
        "ind_labels": {
            "e1": "e1 - Deslizamentos",
            "e2": "e2 - Inundacoes/alagamentos",
            "e3": "e3 - Elevacao do nivel do mar",
            "e4": "e4 - Calor extremo",
            "e5": "e5 - Queimadas",
        },
    },
    "ig": {
        "label"     : "IG - Gestao Municipal (invertida no IIC)",
        "color"     : "#2980B9",
        "indicators": ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8"],
        "ind_labels": {
            "g1": "g1 - Investimento ambiental",
            "g2": "g2 - Plano de contingencia",
            "g3": "g3 - Participacao NUPDEC",
            "g4": "g4 - Conselhos municipais",
            "g5": "g5 - Sistema de alerta",
            "g6": "g6 - Mapeamento/zoneamento de risco",
            "g7": "g7 - Cadastro de familias em risco",
            "g8": "g8 - Politicas de direitos humanos",
        },
    },
}

SUB_LABELS = {
    "ip": "IP - Grupos Prioritarios",
    "iv": "IV - Vulnerabilidade",
    "ie": "IE - Exposicao",
    "ig": "IG - Gestao Municipal (pre-inversao)",
}

ALL_INDICATORS = [
    "p1", "p2", "p3", "p4", "p5",
    "v1", "v2", "v3", "v4", "v5",
    "e1", "e2", "e3", "e4", "e5",
    "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8",
]


# =============================================================================
# Helpers scatter
# =============================================================================

def _cmap_for(base_color: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("custom", ["#f7f7f7", base_color], N=256)


def _add_stats(ax, x: np.ndarray, y: np.ndarray, color: str) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym = x[mask], y[mask]
    if len(xm) < 2 or np.unique(xm).size < 2:
        return
    slope, intercept, r, p, _ = stats.linregress(xm, ym)
    xfit = np.array([xm.min(), xm.max()])
    ax.plot(xfit, slope * xfit + intercept, color=color, lw=1.8, zorder=5)
    pstr = "< 0.001" if p < 0.001 else f"= {p:.3f}"
    ax.text(
        0.04, 0.95, f"r = {r:.3f}  (p {pstr})",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8),
    )


def _hexbin_scatter(ax, x: np.ndarray, y: np.ndarray, color: str,
                    xlabel: str, ylabel: str, title: str) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym = x[mask], y[mask]
    cmap = _cmap_for(color)
    hb = ax.hexbin(xm, ym, gridsize=60, cmap=cmap, mincnt=1,
                   linewidths=0.1, extent=[0, 1, 0, 1])
    plt.colorbar(hb, ax=ax, label="Nr de hexagonos", pad=0.02, shrink=0.85)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=8)
    ax.grid(True, lw=0.4, alpha=0.4)
    _add_stats(ax, xm, ym, color)


def _save_scatter(fig: plt.Figure, name: str) -> None:
    path = GRAPHS_DIR / f"{name}_{FILE_TS}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {path.name}")


def _save_map(fig: plt.Figure, name: str) -> None:
    path = MAPS_DIR / f"{name}_{FILE_TS}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {path.name}")


# =============================================================================
# Helpers de mapa – 5 classes de cor
# =============================================================================

def _class_colors_from_base(base_color: str) -> list:
    """5 cores uniformemente espassadas de quase-branco ate base_color."""
    light = np.array(mcolors.to_rgba("#f7f7f7"))
    dark  = np.array(mcolors.to_rgba(base_color))
    return [tuple(light + (dark - light) * i / 4) for i in range(5)]


def _class_colors_from_cmap(cmap_name: str) -> list:
    """5 cores amostradas de um colormap nomeado do matplotlib."""
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i / 4) for i in range(5)]


def _plot_classified_panel(gdf, col, ax, title, class_colors,
                            show_legend: bool = False) -> None:
    """Painel coroplético com 5 classes de cor discretas."""
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
    """Legenda compartilhada na base da figura."""
    patches = [
        mpatches.Patch(facecolor=class_colors[i], edgecolor="#999999",
                       lw=0.3, label=CLASS_LABELS[i])
        for i in range(len(class_colors))
    ]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, y_anchor), framealpha=0.9,
               handlelength=1.5, handleheight=1.0)


# =============================================================================
# Carga de dados
# =============================================================================

def load_sample(n: int) -> pd.DataFrame:
    print(f"Carregando {RESULTS_FILE.name} ...")
    cols = ALL_INDICATORS + ["ip", "iv", "ie", "ig", "iic_final"]
    df = pd.read_parquet(RESULTS_FILE, columns=cols)
    df = df.dropna(subset=["iic_final"])
    sample = df.sample(min(n, len(df)), random_state=SEED)
    print(f"  Amostra: {len(sample):,} de {len(df):,} hexagonos.")
    return sample


def load_city_data(nm_mun: str, nm_uf: str) -> pd.DataFrame:
    cols = (["h3_id", "nm_mun", "nm_uf", "iic_final", "ip", "iv", "ie", "ig"]
            + ALL_INDICATORS)
    df = pd.read_parquet(RESULTS_FILE, columns=cols)
    mask = (df["nm_mun"] == nm_mun) & (df["nm_uf"] == nm_uf)
    result = df[mask].drop_duplicates(subset="h3_id").reset_index(drop=True)
    print(f"  {nm_mun}/{nm_uf}: {len(result):,} hexagonos carregados.")
    return result


# =============================================================================
# Figuras scatter
# =============================================================================

def fig_dimension(dim_key: str, meta: dict, df: pd.DataFrame) -> None:
    indicators = meta["indicators"]
    color      = meta["color"]
    n_ind      = len(indicators)
    n_cols     = 3
    n_rows     = (n_ind + 1 + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 4.2))
    axes_flat = np.array(axes).flatten()

    x_sub = df[dim_key].to_numpy(dtype="float64")
    y_iic = df["iic_final"].to_numpy(dtype="float64")

    for i, ind_key in enumerate(indicators):
        _hexbin_scatter(
            axes_flat[i],
            df[ind_key].to_numpy(dtype="float64"), x_sub,
            xlabel=f"{ind_key} (normalizado 0-1)",
            ylabel=f"{dim_key.upper()} (0-1)",
            title=meta["ind_labels"][ind_key],
            color=color,
        )

    note = "  (IG invertida no IIC)" if dim_key == "ig" else ""
    _hexbin_scatter(
        axes_flat[n_ind], x_sub, y_iic,
        xlabel=f"{dim_key.upper()} (0-1)",
        ylabel="IIC final (0-1)",
        title=f"{dim_key.upper()} -> IIC final{note}",
        color="#555555",
    )

    for j in range(n_ind + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Diagnostico de Dispersao - {meta['label']}\n"
        f"Amostra: {len(df):,} hexagonos | hexbin (densidade de pontos)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_scatter(fig, f"scatter_{dim_key}")


def fig_subindices_vs_iic(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    for ax, (sub, color) in zip(axes.flat, [(k, DIMS[k]["color"]) for k in DIMS]):
        note = "  (invertida no IIC)" if sub == "ig" else ""
        _hexbin_scatter(
            ax,
            df[sub].to_numpy(dtype="float64"),
            df["iic_final"].to_numpy(dtype="float64"),
            xlabel=f"{sub.upper()} (0-1)",
            ylabel="IIC final (0-1)",
            title=f"{SUB_LABELS[sub]}{note}",
            color=color,
        )
    fig.suptitle(f"Sub-indices vs. IIC Final\nAmostra: {len(df):,} hexagonos",
                 fontsize=13, fontweight="bold")
    _save_scatter(fig, "scatter_subindices_vs_iic")


def fig_iic_distribution(df: pd.DataFrame) -> None:
    iic = df["iic_final"].dropna().to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    ax1 = axes[0]
    ax1.hist(iic, bins=100, color="#8E44AD", alpha=0.85, edgecolor="none")
    ax1.axvline(np.mean(iic),   color="red",  lw=1.5, ls="--",
                label=f"Media = {np.mean(iic):.3f}")
    ax1.axvline(np.median(iic), color="navy", lw=1.5, ls="--",
                label=f"Mediana = {np.median(iic):.3f}")
    ax1.set_xlabel("IIC final (0-1)", fontsize=10)
    ax1.set_ylabel("Numero de hexagonos", fontsize=10)
    ax1.set_title("Distribuicao do IIC Final - Histograma", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, lw=0.4, alpha=0.4)

    ax2 = axes[1]
    sorted_iic = np.sort(iic)
    ax2.scatter(sorted_iic, np.linspace(0, 1, len(sorted_iic)),
                s=0.5, alpha=0.3, color="#8E44AD", rasterized=True)
    ax2.set_xlabel("IIC final (0-1)", fontsize=10)
    ax2.set_ylabel("Percentil acumulado", fontsize=10)
    ax2.set_title("Distribuicao Acumulada do IIC Final", fontsize=11, fontweight="bold")
    ax2.grid(True, lw=0.4, alpha=0.4)
    ax2.text(
        0.04, 0.96,
        f"n = {len(iic):,}\nMedia = {np.mean(iic):.4f}\nMediana = {np.median(iic):.4f}\n"
        f"DP = {np.std(iic):.4f}\nP10 = {np.percentile(iic, 10):.4f}\n"
        f"P90 = {np.percentile(iic, 90):.4f}",
        transform=ax2.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
    )
    fig.suptitle("Diagnostico - IIC Final", fontsize=13, fontweight="bold")
    _save_scatter(fig, "scatter_iic_final")


def fig_subindices_matrix(df: pd.DataFrame) -> None:
    subs = ["ip", "iv", "ie", "ig"]
    n    = len(subs)
    fig, axes = plt.subplots(n, n, figsize=(15, 15), constrained_layout=True)

    for i, si in enumerate(subs):
        for j, sj in enumerate(subs):
            ax = axes[i][j]
            xi = df[si].to_numpy(dtype="float64")
            xj = df[sj].to_numpy(dtype="float64")
            if i == j:
                mask = np.isfinite(xi)
                ax.hist(xi[mask], bins=80, color="#999999", edgecolor="none", alpha=0.8)
                ax.set_title(si.upper(), fontsize=10, fontweight="bold")
            else:
                mask = np.isfinite(xi) & np.isfinite(xj)
                ax.hexbin(xi[mask], xj[mask], gridsize=40,
                          cmap=_cmap_for(DIMS[si]["color"]), mincnt=1,
                          linewidths=0.05, extent=[0, 1, 0, 1])
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                if mask.sum() > 1:
                    r, _ = stats.pearsonr(xi[mask], xj[mask])
                    ax.text(0.05, 0.93, f"r = {r:.3f}", transform=ax.transAxes,
                            fontsize=8, va="top",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                      ec="none", alpha=0.8))
            if i == n - 1:
                ax.set_xlabel(sj.upper(), fontsize=9)
            if j == 0:
                ax.set_ylabel(si.upper(), fontsize=9)
            ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Matriz de Dispersao entre Sub-indices\nAmostra: {len(df):,} hexagonos",
        fontsize=13, fontweight="bold",
    )
    _save_scatter(fig, "scatter_matrix_subindices")


# =============================================================================
# Mapa H3
# =============================================================================

def _h3_to_polygon(h3_id: str) -> Polygon:
    coords = h3.cell_to_boundary(h3_id)   # [(lat, lng), ...]
    return Polygon([(lng, lat) for lat, lng in coords])


def _build_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    df = df.copy()
    df["geometry"] = df["h3_id"].apply(_h3_to_polygon)
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")


def _city_slug(nm_mun: str, nm_uf: str) -> str:
    slug = nm_mun.lower().replace(" ", "_")
    uf   = nm_uf.split()[-1].lower()
    return f"{slug}_{uf}"


def fig_map_city(nm_mun: str, nm_uf: str, gdf: gpd.GeoDataFrame) -> None:
    """IIC final + 4 sub-indices em 5 classes de cor discretas."""
    PLOTS = [
        ("iic_final", "IIC Final",
         _class_colors_from_cmap("RdYlGn_r")),
        ("ip", "IP - Grupos Prioritarios",
         _class_colors_from_base(DIMS["ip"]["color"])),
        ("iv", "IV - Vulnerabilidade",
         _class_colors_from_base(DIMS["iv"]["color"])),
        ("ie", "IE - Exposicao",
         _class_colors_from_base(DIMS["ie"]["color"])),
        ("ig", "IG - Gestao Municipal\n(pos-inversao no IIC)",
         _class_colors_from_base(DIMS["ig"]["color"])),
    ]

    fig = plt.figure(figsize=(22, 13))
    gs  = mgridspec.GridSpec(2, 3, figure=fig, hspace=0.18, wspace=0.06)

    ax_main  = fig.add_subplot(gs[:, 0])
    axes_sub = [
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]

    for ax, (col, title, class_colors) in zip([ax_main] + axes_sub, PLOTS):
        _plot_classified_panel(gdf, col, ax, title, class_colors, show_legend=True)

    fig.suptitle(
        f"{nm_mun} / {nm_uf}\n"
        f"Distribuicao Espacial H3 (res. 9) - IIC e Sub-indices",
        fontsize=14, fontweight="bold", y=1.01,
    )
    _save_map(fig, f"map_{_city_slug(nm_mun, nm_uf)}")


def fig_map_city_indicators(nm_mun: str, nm_uf: str,
                             gdf: gpd.GeoDataFrame) -> None:
    """Uma figura por dimensao com mapas de cada indicador individual."""
    for dim_key, meta in DIMS.items():
        indicators  = meta["indicators"]
        color       = meta["color"]
        class_colors = _class_colors_from_base(color)
        n_ind  = len(indicators)
        n_cols = 3
        n_rows = (n_ind + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 5.5, n_rows * 5.5 + 0.8),
        )
        axes_flat = np.array(axes).flatten()

        for i, ind_key in enumerate(indicators):
            if ind_key not in gdf.columns:
                axes_flat[i].set_visible(False)
                continue
            _plot_classified_panel(
                gdf, ind_key, axes_flat[i],
                title=meta["ind_labels"][ind_key],
                class_colors=class_colors,
                show_legend=False,
            )

        for j in range(n_ind, len(axes_flat)):
            axes_flat[j].set_visible(False)

        _shared_legend(fig, class_colors, y_anchor=0.01)

        fig.suptitle(
            f"{nm_mun} / {nm_uf}  -  {meta['label']}\n"
            f"Indicadores individuais (5 classes, 0-1)",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0.07, 1, 0.93])
        _save_map(fig, f"map_{_city_slug(nm_mun, nm_uf)}_ind_{dim_key}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print(f"Diagnostico IIC v2.0  |  {RESULTS_FILE.name}")
    print(f"Timestamp: {FILE_TS}")
    print("=" * 60)

    df = load_sample(SAMPLE_N)

    print("\n[1/5] Scatter por dimensao (indicadores vs sub-indice)...")
    for dim_key, meta in DIMS.items():
        print(f"  -> {dim_key.upper()}")
        fig_dimension(dim_key, meta, df)

    print("\n[2/5] Sub-indices vs IIC final...")
    fig_subindices_vs_iic(df)

    print("\n[3/5] Distribuicao do IIC final...")
    fig_iic_distribution(df)

    print("\n[4/5] Matriz de dispersao entre sub-indices...")
    fig_subindices_matrix(df)

    print(f"\n[5/5] Mapas por cidade ({len(CITIES)} cidade(s) em config.json)...")
    for city in CITIES:
        nm_mun = city["nm_mun"]
        nm_uf  = city["nm_uf"]
        print(f"  -> {nm_mun} / {nm_uf}")
        city_df = load_city_data(nm_mun, nm_uf)
        if city_df.empty:
            print(f"  [AVISO] Nenhum dado para {nm_mun} / {nm_uf}")
            continue
        print(f"  Convertendo {len(city_df):,} hexagonos para geometria ...")
        gdf = _build_gdf(city_df)
        fig_map_city(nm_mun, nm_uf, gdf)
        fig_map_city_indicators(nm_mun, nm_uf, gdf)

    n_graphs = len(list(GRAPHS_DIR.glob("*.png")))
    n_maps   = len(list(MAPS_DIR.glob("*.png")))
    print(f"\nConcluido!")
    print(f"  Graficos: {n_graphs} PNGs em {GRAPHS_DIR}")
    print(f"  Mapas:    {n_maps} PNGs em {MAPS_DIR}")


if __name__ == "__main__":
    main()
