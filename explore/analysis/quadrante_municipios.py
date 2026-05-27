"""
Quadrant analysis dividing Brazilian municipalities into 4 groups based on:

  IPI_s = simple mean(IP, IV, IE)  — Índice de Potencial de Injustiça
  ICG_m = IG sub-index              — Índice de Capacidade de Governança

Threshold: national median of each index (computed across all municipalities).
Using the median ensures a balanced 50/50 split on each axis and is robust
to the skewed distributions typical of IP, IV, IE and IG.

Quadrants:
  A — Alta IPI + Fraca ICG  → Potencial de Injustiça  (maior preocupação)
  B — Alta IPI + Forte ICG  → Potencial de Adaptação
  C — Baixa IPI + Fraca ICG → Risco Latente
  D — Baixa IPI + Forte ICG → Cenário Ideal

Outputs in {DIAGNOSE_DIR}/analysis/:
  quadrantes_municipios.csv
  quadrante_scatter.png
  quadrante_mapa.png

Usage:
    python explore/analysis/quadrante_municipios.py

Dependencies: pandas, matplotlib, geopandas (all in project venv)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.patches import Rectangle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src import config as cfg

# ─── Settings ─────────────────────────────────────────────────────────────────
IPI_DIMS = ["ip", "iv", "ie"]  # equal-weight components of IPI_s
ICG_COL  = "ig"                # governance capacity (higher = better)

# Quadrant metadata.
# color     — used in scatter plot (vivid, for legibility at small dot size)
# map_color — used in choropleth map (pastel, avoids visual heaviness over large areas)
QUAD_META = {
    "A": {"label": "Potencial de Injustiça",  "color": "#c0392b", "map_color": "#e89080"},
    "B": {"label": "Potencial de Adaptação",  "color": "#e07010", "map_color": "#f0ba78"},
    "C": {"label": "Risco Latente",            "color": "#c8a800", "map_color": "#e8d870"},
    "D": {"label": "Cenário Ideal",            "color": "#27ae60", "map_color": "#88c898"},
}

# ─── Paths ────────────────────────────────────────────────────────────────────
_results = sorted(
    cfg.FILES["output"]["results_dir"].glob(f"{cfg.IIC_FILE_PREFIX}_*.parquet"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
if not _results:
    raise FileNotFoundError(
        f"No results found in {cfg.FILES['output']['results_dir']}\n"
        "Run `python run_index.py` first."
    )
RESULTS_FILE = _results[0]

MALHA_DIR       = cfg.RAW_DIR / "ibge" / "malha_municipal" / "2024"
MUNICIPIOS_GPKG = MALHA_DIR / "municipios.gpkg"
UF_GPKG         = MALHA_DIR / "uf.gpkg"

OUT_DIR     = cfg.DIAGNOSE_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV           = OUT_DIR / "quadrantes_municipios.csv"
OUT_SCATTER       = OUT_DIR / "quadrante_scatter.png"
OUT_SCATTER_CLEAN = OUT_DIR / "quadrante_scatter_clean.png"
OUT_MAP           = OUT_DIR / "quadrante_mapa.png"


# ─── Step 1: Load & aggregate to municipality level ───────────────────────────
def load_and_aggregate() -> pd.DataFrame:
    print(f"\nLoading {RESULTS_FILE.name} ...")
    needed = ["cd_mun", "nm_mun", "nm_uf", "qtd_dom", "iic_final"] + IPI_DIMS + [ICG_COL]
    df = pd.read_parquet(RESULTS_FILE, columns=needed)
    df = df.dropna(subset=["nm_mun"])
    print(f"  {len(df):,} hexagons loaded.")

    def _wmean(grp: pd.DataFrame) -> pd.Series:
        w = grp["qtd_dom"].fillna(1.0).clip(lower=0.001)
        out: dict = {}
        for col in ["iic_final"] + IPI_DIMS + [ICG_COL]:
            v    = grp[col]
            mask = v.notna()
            out[col] = float(np.average(v[mask], weights=w[mask])) if mask.any() else np.nan
        out["n_hex"]   = len(grp)
        out["qtd_dom"] = float(grp["qtd_dom"].sum())
        return pd.Series(out)

    print("Aggregating by municipality (qtd_dom-weighted mean)...")
    agg = (
        df.groupby(["cd_mun", "nm_mun", "nm_uf"], sort=False)
        .apply(_wmean, include_groups=False)
        .reset_index()
    )
    print(f"  {len(agg):,} municipalities aggregated.")
    return agg


# ─── Step 2: Compute IPI_s and assign quadrants ───────────────────────────────
def compute_quadrants(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    df = df.copy()

    df["ipi_s"] = df[IPI_DIMS].mean(axis=1)
    df["icg_m"] = df[ICG_COL]

    ipi_med = float(df["ipi_s"].median())
    icg_med = float(df["icg_m"].median())

    print(f"\nThresholds (national medians):")
    print(f"  IPI_s  median : {ipi_med:.4f}  (mean={df['ipi_s'].mean():.4f})")
    print(f"  ICG_m  median : {icg_med:.4f}  (mean={df['icg_m'].mean():.4f})")

    high_ipi = df["ipi_s"] >= ipi_med
    high_icg = df["icg_m"] >= icg_med

    df["quadrante"] = pd.NA
    df.loc[ high_ipi & ~high_icg, "quadrante"] = "A"
    df.loc[ high_ipi &  high_icg, "quadrante"] = "B"
    df.loc[~high_ipi & ~high_icg, "quadrante"] = "C"
    df.loc[~high_ipi &  high_icg, "quadrante"] = "D"

    print(f"\nQuadrant distribution:")
    for q in ["A", "B", "C", "D"]:
        sub = df[df["quadrante"] == q]
        print(f"  {q} — {QUAD_META[q]['label']}: {len(sub):,} municipalities")

    # IPI component diagnostics
    print(f"\nIPI_s component means (to check balance):")
    for dim in IPI_DIMS:
        print(f"  {dim}: mean={df[dim].mean():.4f}  median={df[dim].median():.4f}")

    return df, ipi_med, icg_med


# ─── Step 3: Export CSV ───────────────────────────────────────────────────────
def save_csv(df: pd.DataFrame) -> None:
    cols = [
        "quadrante", "cd_mun", "nm_mun", "nm_uf",
        "iic_final", "ipi_s", "icg_m", "ip", "iv", "ie", "ig",
        "n_hex", "qtd_dom",
    ]
    present = [c for c in cols if c in df.columns]
    df.sort_values(["quadrante", "ipi_s"], ascending=[True, False])[present].to_csv(
        OUT_CSV, index=False, encoding="utf-8-sig",
    )
    print(f"\nSaved: {OUT_CSV}")


# ─── Step 4: Scatter plot IPI_s × ICG_m ──────────────────────────────────────
def save_scatter(df: pd.DataFrame, ipi_med: float, icg_med: float) -> None:
    margin = 0.02
    x0 = df["ipi_s"].min() - margin
    x1 = df["ipi_s"].max() + margin
    y0 = df["icg_m"].min() - margin
    y1 = df["icg_m"].max() + margin

    fig, ax = plt.subplots(figsize=(8, 7))

    # Quadrant background rectangles
    quad_boxes = {
        "A": (ipi_med, y0,       x1 - ipi_med,  icg_med - y0),
        "B": (ipi_med, icg_med,  x1 - ipi_med,  y1 - icg_med),
        "C": (x0,      y0,       ipi_med - x0,  icg_med - y0),
        "D": (x0,      icg_med,  ipi_med - x0,  y1 - icg_med),
    }
    for q, (rx, ry, rw, rh) in quad_boxes.items():
        ax.add_patch(Rectangle(
            (rx, ry), rw, rh,
            facecolor=QUAD_META[q]["color"], alpha=0.13,
            linewidth=0, zorder=0,
        ))

    # Median threshold lines
    ax.axvline(ipi_med, color="#444444", linewidth=1.0, linestyle="--", alpha=0.6, zorder=1)
    ax.axhline(icg_med, color="#444444", linewidth=1.0, linestyle="--", alpha=0.6, zorder=1)

    # Scatter points per quadrant
    for q in ["A", "B", "C", "D"]:
        sub = df[df["quadrante"] == q]
        ax.scatter(
            sub["ipi_s"], sub["icg_m"],
            c=QUAD_META[q]["color"], s=7, alpha=0.55, linewidths=0, zorder=2,
            label=f"{q} — {QUAD_META[q]['label']}  (n={len(sub):,})",
        )

    # Quadrant name labels centred in each zone
    label_xy = {
        "A": (ipi_med + (x1 - ipi_med) / 2, y0       + (icg_med - y0) / 2),
        "B": (ipi_med + (x1 - ipi_med) / 2, icg_med  + (y1 - icg_med) / 2),
        "C": (x0      + (ipi_med - x0) / 2, y0       + (icg_med - y0) / 2),
        "D": (x0      + (ipi_med - x0) / 2, icg_med  + (y1 - icg_med) / 2),
    }
    for q, (cx, cy) in label_xy.items():
        ax.text(cx, cy, QUAD_META[q]["label"],
                ha="center", va="center", fontsize=8.5,
                color=QUAD_META[q]["color"], fontweight="bold", alpha=0.75, zorder=3)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel(
        "IPI$_s$ — Índice de Potencial de Injustiça\n"
        f"(média de IP, IV, IE  |  mediana nacional = {ipi_med:.3f})",
        fontsize=10,
    )
    ax.set_ylabel(
        "ICG$_m$ — Índice de Capacidade de Governança\n"
        f"(IG  |  mediana nacional = {icg_med:.3f})",
        fontsize=10,
    )
    ax.set_title("Quadrantes de Injustiça Climática Municipal — Brasil", fontsize=12, pad=12)
    ax.legend(fontsize=8.5, framealpha=0.85, loc="upper left")
    ax.grid(linestyle="--", alpha=0.25, zorder=0)

    fig.tight_layout()
    fig.savefig(OUT_SCATTER, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_SCATTER}")


def save_scatter_clean(df: pd.DataFrame, ipi_med: float, icg_med: float) -> None:
    """Slide-ready version: no title, no quadrant labels, no legend."""
    margin = 0.02
    x0 = df["ipi_s"].min() - margin
    x1 = df["ipi_s"].max() + margin
    y0 = df["icg_m"].min() - margin
    y1 = df["icg_m"].max() + margin

    fig, ax = plt.subplots(figsize=(8, 7))

    quad_boxes = {
        "A": (ipi_med, y0,       x1 - ipi_med,  icg_med - y0),
        "B": (ipi_med, icg_med,  x1 - ipi_med,  y1 - icg_med),
        "C": (x0,      y0,       ipi_med - x0,  icg_med - y0),
        "D": (x0,      icg_med,  ipi_med - x0,  y1 - icg_med),
    }
    for q, (rx, ry, rw, rh) in quad_boxes.items():
        ax.add_patch(Rectangle(
            (rx, ry), rw, rh,
            facecolor=QUAD_META[q]["color"], alpha=0.13,
            linewidth=0, zorder=0,
        ))

    ax.axvline(ipi_med, color="#444444", linewidth=1.0, linestyle="--", alpha=0.6, zorder=1)
    ax.axhline(icg_med, color="#444444", linewidth=1.0, linestyle="--", alpha=0.6, zorder=1)

    for q in ["A", "B", "C", "D"]:
        sub = df[df["quadrante"] == q]
        ax.scatter(
            sub["ipi_s"], sub["icg_m"],
            c=QUAD_META[q]["color"], s=7, alpha=0.55, linewidths=0, zorder=2,
        )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("Índice de Potencial de Injustiça\n(IP + IV + IE) / 3", fontsize=11)
    ax.set_ylabel("IG", fontsize=11)
    ax.grid(linestyle="--", alpha=0.25, zorder=0)

    fig.tight_layout()
    fig.savefig(OUT_SCATTER_CLEAN, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_SCATTER_CLEAN}")


# ─── Step 5: Choropleth map ───────────────────────────────────────────────────
def save_map(df: pd.DataFrame) -> None:
    try:
        import geopandas as gpd
    except ImportError:
        print("geopandas not available — skipping map.")
        return

    print("Building choropleth map (loading geopackages)...")
    gdf_uf  = gpd.read_file(UF_GPKG)
    gdf_mun = gpd.read_file(MUNICIPIOS_GPKG, columns=["cd_mun", "geometry"])

    df_join = df[["cd_mun", "quadrante"]].copy()
    df_join["cd_mun"] = df_join["cd_mun"].astype(str).str.zfill(7)
    gdf_mun["cd_mun"] = gdf_mun["cd_mun"].astype(str).str.zfill(7)
    gdf_joined = gdf_mun.merge(df_join, on="cd_mun", how="left")
    gdf_joined["_color"] = gdf_joined["quadrante"].map(
        {q: m["map_color"] for q, m in QUAD_META.items()}
    ).fillna("#cccccc")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("#cfe2f3")

    gdf_joined.plot(ax=ax, color=gdf_joined["_color"],
                    linewidth=0, edgecolor="none", zorder=2)
    gdf_uf.boundary.plot(ax=ax, color="white", linewidth=0.7, zorder=3)

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(
        "Quadrantes de Injustiça Climática por Município — Brasil\n"
        "(threshold: mediana nacional de IPI$_s$ e ICG$_m$)",
        fontsize=12, pad=10,
    )

    legend_patches = [
        mpatches.Patch(
            facecolor=QUAD_META[q]["map_color"],
            label=f"{q} — {QUAD_META[q]['label']}  (n={len(df[df['quadrante']==q]):,})",
        )
        for q in ["A", "B", "C", "D"]
    ]
    ax.legend(handles=legend_patches, title="Cenário", fontsize=9,
              framealpha=0.9, loc="lower right", title_fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_MAP, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_MAP}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 65)
    print("Quadrant Analysis — Municipal Injustice Framework")
    print(f"Results file : {RESULTS_FILE.name}")
    print("=" * 65)

    agg = load_and_aggregate()
    df, ipi_med, icg_med = compute_quadrants(agg)
    save_csv(df)
    save_scatter(df, ipi_med, icg_med)
    save_scatter_clean(df, ipi_med, icg_med)
    save_map(df)

    print(f"\nDone. Outputs in:\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
