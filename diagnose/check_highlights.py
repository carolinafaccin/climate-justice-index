"""
Identifica o município com maior pontuação média (ponderada por qtd_dom) para
o IIC final, cada sub-índice e cada indicador individual.

Gera os mapas H3 (IIC + sub-índices + indicadores por dimensão) de cada
município identificado, salvos em outputs/figures/maps/highlights_analysis/.
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
from shapely.geometry import Polygon
from pathlib import Path
import h3

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src import config as cfg

# ==============================================================================
# CAMINHOS
# ==============================================================================
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

HIGHLIGHTS_DIR = cfg.FIGURES_DIR / "maps" / "highlights_analysis"
HIGHLIGHTS_DIR.mkdir(parents=True, exist_ok=True)

DPI = 150
CLASS_BOUNDS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CLASS_LABELS = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

ALL_INDICATORS = [
    "p1", "p2", "p3", "p4", "p5",
    "v1", "v2", "v3", "v4", "v5",
    "e1", "e2", "e3", "e4", "e5",
    "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8",
]

ALL_METRICS = ["iic_final", "ip", "iv", "ie", "ig"] + ALL_INDICATORS

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
    slug = slug.lower().replace(" ", "_")
    uf   = nm_uf.split()[-1].lower()
    return f"{slug}_{uf}"


def _save(fig, name: str) -> None:
    path = HIGHLIGHTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Salvo: {path.name}")


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
        f"{nm_mun} / {nm_uf}\nDistribuição Espacial H3 (res. 9) — IIC e Sub-índices",
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
            f"{nm_mun} / {nm_uf}  —  {meta['label']}\nIndicadores individuais (5 classes, 0–1)",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0.07, 1, 0.93])
        _save(fig, f"map_{_city_slug(nm_mun, nm_uf)}_ind_{dim_key}")


# ==============================================================================
# ANÁLISE PRINCIPAL
# ==============================================================================

def load_and_aggregate() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega o parquet completo e agrega por município (média pond. por qtd_dom)."""
    print(f"Carregando {RESULTS_FILE.name} ...")
    needed = ["h3_id", "nm_mun", "nm_uf", "qtd_dom"] + ALL_METRICS
    needed = [c for c in needed if c]  # remove None
    df = pd.read_parquet(RESULTS_FILE, columns=needed)
    df = df.dropna(subset=["nm_mun", "nm_uf"])
    print(f"  {len(df):,} hexágonos carregados.")

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

    print("Agregando por município (média ponderada por qtd_dom)...")
    agg = df.groupby(["nm_mun", "nm_uf"], sort=False).apply(_wmean_series).reset_index()
    print(f"  {len(agg):,} municípios.")
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
            "métrica":  m,
            "município": top["nm_mun"],
            "uf":        top["nm_uf"],
            "valor":     top[m],
            "n_hex":     int(top["n_hex"]),
        })
    return pd.DataFrame(rows)


def print_table(highlights: pd.DataFrame) -> None:
    print(f"\n{'─' * 78}")
    print(f"{'Métrica':<12} {'Município':<35} {'UF':<5} {'Valor':>7}  {'Hex':>6}")
    print(f"{'─' * 78}")
    for _, row in highlights.iterrows():
        print(
            f"{row['métrica']:<12} {row['município']:<35} "
            f"{row['uf']:<5} {row['valor']:>7.4f}  {row['n_hex']:>6,}"
        )
    print()


def load_city_data(nm_mun: str, nm_uf: str) -> pd.DataFrame:
    needed = ["h3_id", "nm_mun", "nm_uf", "iic_final",
              "ip", "iv", "ie", "ig"] + ALL_INDICATORS
    df = pd.read_parquet(RESULTS_FILE, columns=needed)
    mask = (df["nm_mun"] == nm_mun) & (df["nm_uf"] == nm_uf)
    result = df[mask].drop_duplicates(subset="h3_id").reset_index(drop=True)
    print(f"  {nm_mun} / {nm_uf}: {len(result):,} hexágonos")
    return result


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("Análise de Destaques — Municípios com Maiores Scores")
    print(f"Arquivo: {RESULTS_FILE.name}")
    print("=" * 60)

    df_full, agg = load_and_aggregate()
    highlights   = find_highlights(agg)

    print_table(highlights)

    # Municípios únicos a mapear
    unique_cities = (
        highlights[["município", "uf"]]
        .drop_duplicates()
        .rename(columns={"município": "nm_mun", "uf": "nm_uf"})
        .to_dict("records")
    )
    print(f"{len(unique_cities)} município(s) único(s) identificado(s).")
    print(f"Mapas serão salvos em: {HIGHLIGHTS_DIR}\n")

    for city in unique_cities:
        nm_mun, nm_uf = city["nm_mun"], city["nm_uf"]
        metrics_here  = highlights.loc[
            (highlights["município"] == nm_mun) & (highlights["uf"] == nm_uf),
            "métrica"
        ].tolist()
        print(f"→ {nm_mun} / {nm_uf}  (destaque em: {', '.join(metrics_here)})")

        city_df = load_city_data(nm_mun, nm_uf)
        if city_df.empty:
            print("  [AVISO] Nenhum dado encontrado, pulando.")
            continue

        print(f"  Convertendo {len(city_df):,} hexágonos para geometria...")
        gdf = _build_gdf(city_df)
        fig_map_city(nm_mun, nm_uf, gdf)
        fig_map_city_indicators(nm_mun, nm_uf, gdf)

    print(f"\nConcluído! {len(list(HIGHLIGHTS_DIR.glob('*.png')))} mapas em {HIGHLIGHTS_DIR}")


if __name__ == "__main__":
    main()
