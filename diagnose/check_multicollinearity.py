"""
Multicollinearity diagnostic between normalized IIC indicators.

Produces:
  1. Spearman correlation heatmap saved to outputs/figures/
  2. List of pairs with |r| >= THRESHOLD printed to terminal

Uses the _norm columns (values that enter the index, after inversion),
with random sampling to keep computation feasible at ~4.5M hexagons.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src import config as cfg

THRESHOLD  = 0.70     # pairs above this |r| are flagged as multicollinear
MAX_SAMPLE = 300_000  # hexagons sampled for performance

# ==============================================================================
# 1. LOAD NORMALISED INDICATORS
# ==============================================================================
print("Loading normalised indicators...")

dfs = {}
for key, path in cfg.FILES_H3.items():
    if key == "base_metadata":
        continue
    if not path.exists():
        print(f"  ✗ {key:<4} — file not found")
        continue
    col_norm = cfg.COLUMN_MAP.get(key)
    if not col_norm:
        continue
    try:
        df_tmp = pd.read_parquet(path, columns=["h3_id", col_norm])
        s = df_tmp.set_index("h3_id")[col_norm]
        dfs[key] = s[~s.index.duplicated(keep="first")]
        print(f"  ✓ {key:<4} — {col_norm}")
    except Exception as e:
        print(f"  ✗ {key:<4} — ERROR: {e}")

# Order by dimension as in indicators.json
ordered_keys = [
    key
    for dim in cfg.DIMENSIONS
    for key in cfg.DIMENSIONS[dim]
    if key in dfs
]

df_all = pd.concat([dfs[k].rename(k) for k in ordered_keys], axis=1)
print(f"\nTotal hexagons: {len(df_all):,}  |  Indicators: {len(ordered_keys)}")

# Sampling
if len(df_all) > MAX_SAMPLE:
    df_sample = df_all.sample(MAX_SAMPLE, random_state=42)
    sample_note = f"sample of {MAX_SAMPLE:,} hexagons"
else:
    df_sample = df_all
    sample_note = f"{len(df_all):,} hexagons"

# ==============================================================================
# 2. SPEARMAN CORRELATION
# ==============================================================================
print("Computing Spearman correlation...")
corr = df_sample.corr(method="spearman")

# ==============================================================================
# 3. TERMINAL — PAIRS WITH |r| >= THRESHOLD
# ==============================================================================
key_meta = {}
for key in ordered_keys:
    ind = cfg.INDICATORS.get(key, {})
    key_meta[key] = {
        "dim":  ind.get("dimension", "?"),
        "abbr": ind.get("abbr", key),
        "name": ind.get("name", key),
    }

pairs = []
for i, k1 in enumerate(ordered_keys):
    for k2 in ordered_keys[i + 1:]:
        r = corr.loc[k1, k2]
        if abs(r) >= THRESHOLD:
            pairs.append((abs(r), r, k1, k2))
pairs.sort(reverse=True)

print(f"\n{'─' * 72}")
print(f"Pairs with |r| ≥ {THRESHOLD}  ({len(pairs)} found):")
print(f"{'─' * 72}")
if pairs:
    for _, r, k1, k2 in pairs:
        m1, m2 = key_meta[k1], key_meta[k2]
        print(
            f"  r={r:+.3f}  {k1} ({m1['dim'].upper()}) × {k2} ({m2['dim'].upper()})"
            f"  —  {m1['name']} × {m2['name']}"
        )
else:
    print("  No pairs above threshold.")
print()

# ==============================================================================
# 4. FIGURE — HEATMAP
# ==============================================================================
print("Generating heatmap...")

# One color per dimension, generated dynamically
dim_list  = [d for d in cfg.DIMENSIONS if any(k in dfs for k in cfg.DIMENSIONS[d])]
palette   = sns.color_palette("tab10", len(dim_list))
dim_color = {dim: palette[i] for i, dim in enumerate(dim_list)}

# Axis labels: "p1 / mul"
labels = [f"{k} / {key_meta[k]['abbr']}" for k in ordered_keys]

# Dimension blocks (start position + size)
dim_blocks = []
pos = 0
for dim in dim_list:
    size = len([k for k in cfg.DIMENSIONS[dim] if k in dfs])
    if size > 0:
        dim_blocks.append((dim, pos, size))
        pos += size

n        = len(ordered_keys)
fig_size = max(14, n * 0.72)
fig, ax  = plt.subplots(figsize=(fig_size, fig_size * 0.90))

sns.heatmap(
    corr,
    ax=ax,
    cmap="RdBu_r",
    vmin=-1, vmax=1,
    center=0,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 7},
    linewidths=0.4,
    linecolor="#e0e0e0",
    square=True,
    xticklabels=labels,
    yticklabels=labels,
    cbar_kws={"shrink": 0.55, "label": "Spearman Correlation"},
)

# Coloured borders per dimension
for dim, start, size in dim_blocks:
    color = dim_color[dim]
    ax.add_patch(mpatches.Rectangle(
        (start, start), size, size,
        linewidth=2.5, edgecolor=color, facecolor="none", zorder=5,
    ))

# Gold border on cells with |r| >= THRESHOLD (excluding diagonal)
for i, k1 in enumerate(ordered_keys):
    for j, k2 in enumerate(ordered_keys):
        if i != j and abs(corr.loc[k1, k2]) >= THRESHOLD:
            ax.add_patch(mpatches.Rectangle(
                (j, i), 1, 1,
                linewidth=1.6, edgecolor="#e6b800", facecolor="none", zorder=6,
            ))

# Colour tick labels by indicator dimension
for tick, key in zip(ax.get_xticklabels(), ordered_keys):
    tick.set_color(dim_color.get(key_meta[key]["dim"], "#333333"))
    tick.set_fontsize(8)
for tick, key in zip(ax.get_yticklabels(), ordered_keys):
    tick.set_color(dim_color.get(key_meta[key]["dim"], "#333333"))
    tick.set_fontsize(8)

# Dimension legend
legend_handles = [
    mpatches.Patch(
        color=dim_color[dim],
        label=f"{dim.upper()} — {cfg.DIMENSION_META[dim]['name']}"
    )
    for dim in dim_list
]
legend_handles.append(
    mpatches.Patch(facecolor="none", edgecolor="#e6b800", linewidth=1.6,
                   label=f"|r| ≥ {THRESHOLD}")
)
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=len(dim_list) + 1,
    bbox_to_anchor=(0.5, -0.03),
    fontsize=9,
    frameon=False,
)

ax.set_title(
    f"Multicollinearity among normalised indicators — IIC {cfg.INDEX_VERSION}\n"
    f"Spearman Correlation  |  {sample_note}  |  gold highlight: |r| ≥ {THRESHOLD}",
    fontsize=12, pad=16,
)

plt.tight_layout()

now       = datetime.now().strftime("%Y%m%d_%H%M%S")
graphs_dir = cfg.FIGURES_DIR / "graphs"
graphs_dir.mkdir(parents=True, exist_ok=True)
out_path  = graphs_dir / f"multicolinearidade_{now}.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved: {out_path}")
