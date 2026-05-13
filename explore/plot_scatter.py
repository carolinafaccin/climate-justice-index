"""
IIC v2.0 scatter diagnostic.

Generates hexbin scatter plots between indicators, sub-indices, and the final IIC.

Outputs: cfg.FIGURES_DIR / "graphs" / scatter_*_{ts}.png
"""

import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from scipy import stats

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
from src import config as cfg
import utils as diag_utils

# ==============================================================================
# PATHS AND PARAMETERS
# ==============================================================================
SAMPLE_N = 300_000
SEED     = 42
DPI      = 150

GRAPHS_DIR = cfg.FIGURES_DIR / "graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

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

_ts_match = re.search(r"(\d{8}_\d{6})", RESULTS_FILE.stem)
FILE_TS = _ts_match.group(1) if _ts_match else RESULTS_FILE.stem

ALL_INDICATORS = diag_utils.ALL_INDICATOR_KEYS
DIMS           = diag_utils.DIMS

SUB_LABELS = {k: v["label"] for k, v in DIMS.items()}

# ==============================================================================
# HELPERS
# ==============================================================================

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
    ax.text(0.04, 0.95, f"r = {r:.3f}  (p {pstr})", transform=ax.transAxes,
            fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))


def _hexbin_scatter(ax, x: np.ndarray, y: np.ndarray, color: str,
                    xlabel: str, ylabel: str, title: str) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym = x[mask], y[mask]
    hb = ax.hexbin(xm, ym, gridsize=60, cmap=_cmap_for(color), mincnt=1,
                   linewidths=0.1, extent=[0, 1, 0, 1])
    plt.colorbar(hb, ax=ax, label="Nr of hexagons", pad=0.02, shrink=0.85)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=8)
    ax.grid(True, lw=0.4, alpha=0.4)
    _add_stats(ax, xm, ym, color)


def _save(fig: plt.Figure, name: str) -> None:
    path = GRAPHS_DIR / f"{name}_{FILE_TS}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_sample(n: int) -> pd.DataFrame:
    print(f"Loading {RESULTS_FILE.name} ...")
    cols = ALL_INDICATORS + ["ip", "iv", "ie", "ig", "iic_final"]
    df = pd.read_parquet(RESULTS_FILE, columns=cols).dropna(subset=["iic_final"])
    sample = df.sample(min(n, len(df)), random_state=SEED)
    print(f"  Sample: {len(sample):,} of {len(df):,} hexagons.")
    return sample


# ==============================================================================
# FIGURES
# ==============================================================================

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

    note = "  (IG inverted in IIC)" if dim_key == "ig" else ""
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
        f"Scatter Diagnostic — {meta['label']}\n"
        f"Sample: {len(df):,} hexagons | hexbin (point density)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, f"scatter_{dim_key}")


def fig_subindices_vs_iic(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    for ax, (sub, color) in zip(axes.flat, [(k, DIMS[k]["color"]) for k in DIMS]):
        _hexbin_scatter(
            ax,
            df[sub].to_numpy(dtype="float64"),
            df["iic_final"].to_numpy(dtype="float64"),
            xlabel=f"{sub.upper()} (0-1)",
            ylabel="IIC final (0-1)",
            title=SUB_LABELS[sub],
            color=color,
        )
    fig.suptitle(f"Sub-indices vs. IIC Final\nSample: {len(df):,} hexagons",
                 fontsize=13, fontweight="bold")
    _save(fig, "scatter_subindices_vs_iic")


def fig_iic_distribution(df: pd.DataFrame) -> None:
    iic = df["iic_final"].dropna().to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    ax1 = axes[0]
    ax1.hist(iic, bins=100, color="#8E44AD", alpha=0.85, edgecolor="none")
    ax1.axvline(np.mean(iic),   color="red",  lw=1.5, ls="--",
                label=f"Mean = {np.mean(iic):.3f}")
    ax1.axvline(np.median(iic), color="navy", lw=1.5, ls="--",
                label=f"Median = {np.median(iic):.3f}")
    ax1.set_xlabel("IIC final (0-1)", fontsize=10)
    ax1.set_ylabel("Number of hexagons", fontsize=10)
    ax1.set_title("IIC Final Distribution — Histogram", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, lw=0.4, alpha=0.4)

    ax2 = axes[1]
    sorted_iic = np.sort(iic)
    ax2.scatter(sorted_iic, np.linspace(0, 1, len(sorted_iic)),
                s=0.5, alpha=0.3, color="#8E44AD", rasterized=True)
    ax2.set_xlabel("IIC final (0-1)", fontsize=10)
    ax2.set_ylabel("Cumulative percentile", fontsize=10)
    ax2.set_title("IIC Final Cumulative Distribution", fontsize=11, fontweight="bold")
    ax2.grid(True, lw=0.4, alpha=0.4)
    ax2.text(
        0.04, 0.96,
        f"n = {len(iic):,}\nMean = {np.mean(iic):.4f}\nMedian = {np.median(iic):.4f}\n"
        f"SD = {np.std(iic):.4f}\nP10 = {np.percentile(iic, 10):.4f}\n"
        f"P90 = {np.percentile(iic, 90):.4f}",
        transform=ax2.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
    )
    fig.suptitle("Diagnostic — IIC Final", fontsize=13, fontweight="bold")
    _save(fig, "scatter_iic_final")


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
        f"Sub-index Scatter Matrix\nSample: {len(df):,} hexagons",
        fontsize=13, fontweight="bold",
    )
    _save(fig, "scatter_matrix_subindices")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print(f"IIC v2.0 Scatter Plots  |  {RESULTS_FILE.name}")
    print(f"Timestamp: {FILE_TS}")
    print("=" * 60)

    df = load_sample(SAMPLE_N)

    print("\n[1/4] Scatter by dimension (indicators vs sub-index)...")
    for dim_key, meta in DIMS.items():
        print(f"  -> {dim_key.upper()}")
        fig_dimension(dim_key, meta, df)

    print("\n[2/4] Sub-indices vs IIC final...")
    fig_subindices_vs_iic(df)

    print("\n[3/4] IIC final distribution...")
    fig_iic_distribution(df)

    print("\n[4/4] Sub-index scatter matrix...")
    fig_subindices_matrix(df)

    n_graphs = len(list(GRAPHS_DIR.glob("*.png")))
    print(f"\nDone! {n_graphs} PNGs in {GRAPHS_DIR}")


if __name__ == "__main__":
    main()
