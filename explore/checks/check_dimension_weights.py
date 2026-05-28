"""
Sensitivity analysis: equal-weight vs variance-equalised dimension weights.

Rationale: sub-indices with lower std contribute less real discrimination to
IIC even with equal weights. Variance equalisation inverts this by upweighting
low-std dimensions — in particular IE (climate exposure), which is conceptually
central to a climate justice index but has lower spatial variance than the
socioeconomic dimensions.

  scale_i  = 1 / std(sub-index_i)
  peso_i   = scale_i / Σ scale_j       ← higher weight for lower-std dimensions

Outputs (terminal):
  - Descriptive stats per sub-index
  - Resulting variance-equalised weights
  - Pearson r between equal-weight and equalised IIC

Output (figure):
  - Violin distributions of ip, iv, ie, ig
  - Bar chart of stds and resulting weights
  - Scatter: equal-weight IIC × variance-equalised IIC
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src import config as cfg

DIM_ABBRS   = ["ip", "iv", "ie", "ig"]
DIM_LABELS  = {
    abbr: f"{cfg.DIMENSION_META[dim]['abbr']} — {cfg.DIMENSION_META[dim]['display_name']}"
    for abbr, (dim, meta) in zip(
        DIM_ABBRS,
        [(dim, cfg.DIMENSION_META[dim]) for dim in cfg.DIMENSION_META]
    )
}
_ABBR_TO_DIM = {meta["abbr"].lower(): dim for dim, meta in cfg.DIMENSION_META.items()}
DIM_COLORS   = {abbr: cfg.DIMENSION_META[_ABBR_TO_DIM[abbr]]["color"] for abbr in DIM_ABBRS}
MAX_SAMPLE  = 300_000

# ==============================================================================
# 1. LOAD LATEST DASHBOARD FILE (has ip, iv, ie, ig, iic_final)
# ==============================================================================
dashboard_files = sorted(
    f for f in cfg.RESULTS_DASHBOARD_DIR.glob(f"{cfg.DASHBOARD_FILE_PREFIX}_*.parquet")
    if "_dim_" not in f.name
)
if not dashboard_files:
    sys.exit(f"No dashboard parquet found in {cfg.RESULTS_DASHBOARD_DIR}")

latest = dashboard_files[-1]
print(f"Loading: {latest.name}")

df = pd.read_parquet(latest, columns=["h3_id", "ip", "iv", "ie", "ig", "iic_final"])
df = df.dropna(subset=DIM_ABBRS)
print(f"Hexagons with all 4 sub-indices: {len(df):,}\n")

# ==============================================================================
# 2. DESCRIPTIVE STATS + VARIANCE WEIGHTS
# ==============================================================================
stats = df[DIM_ABBRS].describe().T[["mean", "std", "min", "max"]]

stds         = stats["std"]
inv_stds     = 1.0 / stds
weight_eq    = pd.Series({a: 0.25 for a in DIM_ABBRS})
weight_eql   = inv_stds / inv_stds.sum()   # equalização: maior peso para menor std

print("─" * 72)
print(f"{'Sub-index':<6}  {'mean':>7}  {'std':>7}  {'min':>7}  {'max':>7}  {'w_equal':>8}  {'w_eql':>7}")
print("─" * 72)
for abbr in DIM_ABBRS:
    s = stats.loc[abbr]
    print(
        f"{abbr.upper():<6}  {s['mean']:>7.4f}  {s['std']:>7.4f}  "
        f"{s['min']:>7.4f}  {s['max']:>7.4f}  {0.25:>8.4f}  {weight_eql[abbr]:>7.4f}"
    )
print("─" * 72)
print()

print("Variance-equalised weights  (1/std, normalised):")
for abbr in DIM_ABBRS:
    bar = "█" * int(weight_eql[abbr] * 40)
    delta = weight_eql[abbr] - 0.25
    print(f"  {abbr.upper()}  {weight_eql[abbr]:.4f}  ({delta:+.4f} vs equal)  {bar}")
print()

# ==============================================================================
# 3. WEIGHTED IIC
# ==============================================================================
df["iic_equalised"] = sum(df[a] * weight_eql[a] for a in DIM_ABBRS)
df["iic_delta"]     = df["iic_equalised"] - df["iic_final"]

r_pearson = df["iic_final"].corr(df["iic_equalised"])
print(f"Pearson r (equal vs variance-equalised): {r_pearson:.4f}")

q5, q95 = df["iic_delta"].quantile([0.05, 0.95])
print(f"IIC delta (equalised − equal):  p5={q5:+.4f}  p95={q95:+.4f}")
print(f"  n hexagons gaining > 0.05:   {(df['iic_delta'] >  0.05).sum():,}")
print(f"  n hexagons losing  > 0.05:   {(df['iic_delta'] < -0.05).sum():,}")
print()

# Sample AFTER computing derived columns so df_s has them too
if len(df) > MAX_SAMPLE:
    df_s = df.sample(MAX_SAMPLE, random_state=42)
    sample_note = f"sample of {MAX_SAMPLE:,}"
else:
    df_s = df
    sample_note = f"{len(df):,} hexagons"

# ==============================================================================
# 4. FIGURE
# ==============================================================================
print("Generating figure...")

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ax_viol  = fig.add_subplot(gs[0, :2])   # violin — top-left (spans 2 cols)
ax_bar   = fig.add_subplot(gs[0, 2])    # weight bars — top-right
ax_scat  = fig.add_subplot(gs[1, :2])   # scatter — bottom-left (spans 2 cols)
ax_delta = fig.add_subplot(gs[1, 2])    # delta histogram — bottom-right

# ── 4a. Violin distributions ──────────────────────────────────────────────────
data_viol = [df_s[a].dropna().values for a in DIM_ABBRS]
vp = ax_viol.violinplot(data_viol, positions=range(len(DIM_ABBRS)),
                        showmedians=True, showextrema=False)
for i, (body, abbr) in enumerate(zip(vp["bodies"], DIM_ABBRS)):
    body.set_facecolor(DIM_COLORS[abbr])
    body.set_alpha(0.7)
vp["cmedians"].set_color("#333")

ax_viol.set_xticks(range(len(DIM_ABBRS)))
ax_viol.set_xticklabels(
    [f"{a.upper()}\nstd={stds[a]:.3f}" for a in DIM_ABBRS], fontsize=9
)
ax_viol.set_ylabel("Sub-index value [0, 1]", fontsize=9)
ax_viol.set_title("Distribution of sub-indices across hexagons", fontsize=10)
ax_viol.yaxis.grid(True, linestyle="--", alpha=0.4)

# ── 4b. Weight comparison bar chart ──────────────────────────────────────────
x     = np.arange(len(DIM_ABBRS))
width = 0.35
ax_bar.bar(x - width / 2, [0.25] * 4, width,
           color=[DIM_COLORS[a] for a in DIM_ABBRS], alpha=0.4,
           label="Equal (0.25)")
ax_bar.bar(x + width / 2, [weight_eql[a] for a in DIM_ABBRS], width,
           color=[DIM_COLORS[a] for a in DIM_ABBRS], alpha=0.9,
           label="Var.-equalised (1/std)")

ax_bar.axhline(0.25, color="#999", linestyle="--", linewidth=0.8)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels([a.upper() for a in DIM_ABBRS], fontsize=9)
ax_bar.set_ylabel("Weight", fontsize=9)
ax_bar.set_title("Weights: equal vs variance-norm.", fontsize=10)
ax_bar.legend(fontsize=8, frameon=False)
ax_bar.set_ylim(0, max(weight_eql.max() * 1.3, 0.35))
ax_bar.yaxis.grid(True, linestyle="--", alpha=0.4)

# ── 4c. Scatter: equal-weight vs variance-weighted ────────────────────────────
sc = ax_scat.hexbin(
    df_s["iic_final"], df_s["iic_equalised"],
    gridsize=80, cmap="YlOrRd", mincnt=1, linewidths=0.1
)
fig.colorbar(sc, ax=ax_scat, label="hexagon count")
lims = [
    min(ax_scat.get_xlim()[0], ax_scat.get_ylim()[0]),
    max(ax_scat.get_xlim()[1], ax_scat.get_ylim()[1]),
]
ax_scat.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
ax_scat.set_xlabel("IIC — equal weights (1/4 each)", fontsize=9)
ax_scat.set_ylabel("IIC — variance-equalised weights (1/std)", fontsize=9)
ax_scat.set_title(f"Equal vs variance-equalised IIC  |  r = {r_pearson:.4f}", fontsize=10)

# ── 4d. Delta distribution ────────────────────────────────────────────────────
ax_delta.hist(df_s["iic_delta"], bins=80, color="#7f8c8d", alpha=0.8, edgecolor="none")
ax_delta.axvline(0, color="k", linewidth=0.8, linestyle="--")
ax_delta.axvline(q5,  color="#e74c3c", linewidth=0.9, linestyle=":", label=f"p5 = {q5:+.3f}")
ax_delta.axvline(q95, color="#e74c3c", linewidth=0.9, linestyle=":", label=f"p95 = {q95:+.3f}")
ax_delta.set_xlabel("Δ IIC (weighted − equal)", fontsize=9)
ax_delta.set_ylabel("Hexagon count", fontsize=9)
ax_delta.set_title("IIC change from variance equalisation", fontsize=10)
ax_delta.legend(fontsize=8, frameon=False)
ax_delta.yaxis.grid(True, linestyle="--", alpha=0.4)

fig.suptitle(
    f"Variance equalisation sensitivity — IIC {cfg.INDEX_VERSION}  |  {sample_note}",
    fontsize=12, y=1.01
)

now        = datetime.now().strftime(cfg.TS_FORMAT_FILE)
graphs_dir = cfg.FIGURES_DIR / "graphs"
graphs_dir.mkdir(parents=True, exist_ok=True)
out_path   = graphs_dir / f"dimension_weights_{now}.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved: {out_path}")
