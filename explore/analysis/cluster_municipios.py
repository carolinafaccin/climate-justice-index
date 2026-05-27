"""
Cluster analysis for selecting representative municipalities for the article case studies.

Strategy:
  1. Aggregate results parquet to municipality level (qtd_dom-weighted mean)
  2. K-Means (k=N_CLUSTERS) on the 4 sub-indices (ip, iv, ie, ig), standardised
  3. Re-label clusters 0..k-1 by mean iic_final descending (0 = most injust)
  4. Auto-describe each cluster by dimensional z-scores relative to the full dataset
  5. Merge IBGE tipologias (regiao, porte, rural/urban typology)
  6. Filter to top tercile of iic_final; select candidates with regional diversity
  7. Export (filenames include _k{N_CLUSTERS} so different runs never overwrite):
       {DIAGNOSE_DIR}/analysis/clusters_municipios_k{N_CLUSTERS}.csv
       {DIAGNOSE_DIR}/analysis/candidatos_estudo_de_caso_k{N_CLUSTERS}.xlsx
       {DIAGNOSE_DIR}/analysis/cluster_heatmap_k{N_CLUSTERS}.png
       {DIAGNOSE_DIR}/analysis/cluster_pca_k{N_CLUSTERS}.png
       {DIAGNOSE_DIR}/analysis/cluster_boxplots_k{N_CLUSTERS}.png
       {DIAGNOSE_DIR}/analysis/cluster_mapa_k{N_CLUSTERS}.png

Usage:
    python diagnose/analysis/cluster_municipios.py          # default k=4
    # edit N_CLUSTERS below and re-run for a different k

Dependencies: pandas, scikit-learn, matplotlib, seaborn, openpyxl (all in project venv)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src import config as cfg

# ─── Settings ─────────────────────────────────────────────────────────────────
DIMS_COLS    = ["ip", "iv", "ie", "ig"]
N_CLUSTERS   = 4
RANDOM_SEED  = 42
N_CANDIDATES = 50      # max candidates per cluster in the Excel output
IIC_TERCILE  = 0.5     # pool: municipalities above this IIC quantile (0.5 = top half)

REGIOES_ORDER = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
PORTES_ORDER  = [
    "Pequeno Porte I",
    "Pequeno Porte II",
    "Médio Porte",
    "Grande Porte",
    "Metrópole",
]

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
RESULTS_FILE    = _results[0]
TIPOLOGIAS_FILE = cfg.RAW_DIR / "ibge" / "tipologias" / "tipologias_municipios_brasil.csv"

OUT_DIR = cfg.DIAGNOSE_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV      = OUT_DIR / f"clusters_municipios_k{N_CLUSTERS}.csv"
OUT_EXCEL    = OUT_DIR / f"candidatos_estudo_de_caso_k{N_CLUSTERS}.xlsx"
OUT_HEATMAP  = OUT_DIR / f"cluster_heatmap_k{N_CLUSTERS}.png"
OUT_PCA      = OUT_DIR / f"cluster_pca_k{N_CLUSTERS}.png"
OUT_BOXPLOTS = OUT_DIR / f"cluster_boxplots_k{N_CLUSTERS}.png"
OUT_MAP      = OUT_DIR / f"cluster_mapa_k{N_CLUSTERS}.png"

MALHA_DIR       = cfg.RAW_DIR / "ibge" / "malha_municipal" / "2024"
MUNICIPIOS_GPKG = MALHA_DIR / "municipios.gpkg"
UF_GPKG         = MALHA_DIR / "uf.gpkg"
PAIS_GPKG       = MALHA_DIR / "pais.gpkg"


# ─── Step 1: Load & aggregate to municipality level ───────────────────────────
def load_and_aggregate() -> pd.DataFrame:
    print(f"\nLoading {RESULTS_FILE.name} ...")
    needed = ["cd_mun", "nm_mun", "nm_uf", "qtd_dom", "iic_final"] + DIMS_COLS
    df = pd.read_parquet(RESULTS_FILE, columns=needed)
    df = df.dropna(subset=["nm_mun"])
    print(f"  {len(df):,} hexagons loaded.")

    def _wmean(grp: pd.DataFrame) -> pd.Series:
        w = grp["qtd_dom"].fillna(1.0).clip(lower=0.001)
        out: dict = {}
        for col in ["iic_final"] + DIMS_COLS:
            v    = grp[col]
            mask = v.notna()
            out[col] = float(np.average(v[mask], weights=w[mask])) if mask.any() else np.nan
        out["n_hex"]   = len(grp)
        out["qtd_dom"] = float(grp["qtd_dom"].sum())
        return pd.Series(out)

    print("Aggregating by municipality (qtd_dom-weighted mean)...")
    group_cols = ["cd_mun", "nm_mun", "nm_uf"]
    agg = (
        df.groupby(group_cols, sort=False)
        .apply(_wmean, include_groups=False)
        .reset_index()
    )
    print(f"  {len(agg):,} municipalities aggregated.")
    return agg


# ─── Step 2: K-Means clustering on 4 dimensions ───────────────────────────────

# Dimension metadata used in cluster descriptions
_DIM_META = {
    "ip": {
        "label": "Grupos Prioritarios (IP)",
        "high_means": "alta concentracao de grupos prioritarios",
        "low_means":  "baixa concentracao de grupos prioritarios",
    },
    "iv": {
        "label": "Vulnerab. Socioeconomica (IV)",
        "high_means": "alta vulnerabilidade socioecon.",
        "low_means":  "baixa vulnerabilidade socioecon.",
    },
    "ie": {
        "label": "Exposicao Climatica (IE)",
        "high_means": "alta exposicao a riscos climaticos",
        "low_means":  "baixa exposicao climatica",
    },
    "ig": {
        "label": "Gestao Municipal (IG)",
        # ig stored in parquet = 1 − raw_governance (inversion applied before saving).
        # Therefore: high stored ig = poor governance; low stored ig = good governance.
        "high_means": "deficit de gestao municipal (pior contribuicao ao IIC)",
        "low_means":  "boa capacidade de gestao municipal",
    },
}


def _auto_label(cl_means: pd.Series, overall_means: pd.Series, overall_stds: pd.Series) -> str:
    """Generate a short textual label from z-scores of dimensional means."""
    highlights = []
    for dim in DIMS_COLS:
        z = (cl_means[dim] - overall_means[dim]) / max(overall_stds[dim], 1e-9)
        if abs(z) < 0.5:
            continue
        meta = _DIM_META[dim]
        desc = meta["high_means"] if z > 0 else meta["low_means"]
        highlights.append((abs(z), desc))
    highlights.sort(reverse=True)
    if not highlights:
        return "perfil proximo da media em todas as dimensoes"
    return "; ".join(d for _, d in highlights[:3])


def run_kmeans(agg: pd.DataFrame) -> pd.DataFrame:
    mask = agg[DIMS_COLS].notna().all(axis=1)
    n_dropped = (~mask).sum()
    if n_dropped:
        print(f"  Dropping {n_dropped} municipalities with missing sub-index values.")
    df = agg[mask].copy()

    X_scaled = StandardScaler().fit_transform(df[DIMS_COLS].values)
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=20)
    raw_labels = km.fit_predict(X_scaled)

    # Re-label: cluster 0 = highest mean iic_final (most injust profile)
    df["cluster_raw"] = raw_labels
    order = (
        df.groupby("cluster_raw")["iic_final"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    remap = {old: new for new, old in enumerate(order)}
    df["cluster"] = df["cluster_raw"].map(remap).astype(int)
    df = df.drop(columns=["cluster_raw"])

    overall_means = df[DIMS_COLS].mean()
    overall_stds  = df[DIMS_COLS].std()

    # ── Profile table ──────────────────────────────────────────────────────────
    print("\nCluster profiles (municipality-level, mean of qtd_dom-weighted sub-index scores):")
    print(f"  Note: stored ig = 1 − raw_governance → higher ig score = worse governance = greater injustice contribution")
    header = f"  {'Cl':>3}  {'n':>6}  {'IIC':>6}  {'IP':>6}  {'IV':>6}  {'IE':>6}  {'IG':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cl in sorted(df["cluster"].unique()):
        sub  = df[df["cluster"] == cl]
        vals = sub[["iic_final"] + DIMS_COLS].mean()
        print(
            f"  {cl:>3}  {len(sub):>6,}  "
            + "  ".join(f"{vals[c]:>6.3f}" for c in ["iic_final"] + DIMS_COLS)
        )

    # ── Z-score heatmap (text) ─────────────────────────────────────────────────
    print(f"\n  Z-scores relative to full dataset (+ = above mean, - = below):")
    print(f"  {'Cl':>3}  {'IP':>6}  {'IV':>6}  {'IE':>6}  {'IG':>6}  Description")
    print("  " + "-" * 80)
    for cl in sorted(df["cluster"].unique()):
        sub   = df[df["cluster"] == cl]
        means = sub[DIMS_COLS].mean()
        zs    = {d: (means[d] - overall_means[d]) / max(overall_stds[d], 1e-9) for d in DIMS_COLS}
        label = _auto_label(means, overall_means, overall_stds)
        print(
            f"  {cl:>3}  "
            + "  ".join(f"{zs[d]:>+6.2f}" for d in DIMS_COLS)
            + f"  {label}"
        )

    return df


# ─── Step 3: Merge IBGE tipologias ───────────────────────────────────────────
def merge_tipologias(df: pd.DataFrame) -> pd.DataFrame:
    tip_cols = [
        "cd_mun", "uf_sigla", "pop22",
        "regiao", "porte_cnas_mds", "porte_munic",
        "tipo_rururb17_3classes",
    ]
    tip = pd.read_csv(TIPOLOGIAS_FILE, sep=";", dtype={"cd_mun": str})[tip_cols]
    tip["cd_mun"] = tip["cd_mun"].str.strip().str.zfill(7)

    df = df.copy()
    df["cd_mun"] = df["cd_mun"].astype(str).str.strip().str.zfill(7)

    merged = df.merge(tip, on="cd_mun", how="left")
    matched = merged["regiao"].notna().sum()
    print(f"\nTipologias merged: {matched:,}/{len(merged):,} municipalities matched.")
    return merged


# ─── Step 4: Select candidates per cluster ───────────────────────────────────
def select_candidates(df: pd.DataFrame) -> pd.DataFrame:
    thresh = df["iic_final"].quantile(IIC_TERCILE)
    pool   = df[df["iic_final"] >= thresh].copy()
    print(
        f"IIC pool threshold (quantile {IIC_TERCILE:.0%}): IIC >= {thresh:.4f} "
        f"-> {len(pool):,} municipalities eligible"
    )

    records = []
    for cl in sorted(pool["cluster"].unique()):
        sub = pool[pool["cluster"] == cl].sort_values("iic_final", ascending=False)

        chosen     = []
        chosen_idx = set()

        # Pass 1: best IIC from each (region × porte) combination
        # Ensures both small and large municipalities appear, from all regions
        for reg in REGIOES_ORDER:
            for porte in PORTES_ORDER:
                cell = sub[
                    (sub["regiao"] == reg) &
                    (sub["porte_cnas_mds"] == porte)
                ]
                if cell.empty:
                    continue
                row = cell.iloc[0]
                if row.name in chosen_idx:
                    continue
                chosen.append(row)
                chosen_idx.add(row.name)

        # Pass 2: fill remaining slots to N_CANDIDATES by IIC rank (no filter)
        for _, row in sub.iterrows():
            if len(chosen) >= N_CANDIDATES:
                break
            if row.name in chosen_idx:
                continue
            chosen.append(row)
            chosen_idx.add(row.name)

        for rank, row in enumerate(chosen, start=1):
            r = row.to_dict()
            r["candidate_rank"] = rank
            records.append(r)

    return pd.DataFrame(records)


# ─── Step 5: Export outputs ──────────────────────────────────────────────────
EXPORT_COLS_BASE = [
    "cd_mun", "nm_mun", "nm_uf", "uf_sigla",
    "regiao", "porte_munic", "porte_cnas_mds", "tipo_rururb17_3classes", "pop22",
    "iic_final", "ip", "iv", "ie", "ig",
    "n_hex", "qtd_dom",
]

EXPORT_COLS_CANDIDATES = ["cluster", "candidate_rank"] + EXPORT_COLS_BASE


def _present_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _build_profile_table(df_all: pd.DataFrame) -> pd.DataFrame:
    """Cluster profile summary with means, z-scores, and auto-description."""
    overall_means = df_all[DIMS_COLS].mean()
    overall_stds  = df_all[DIMS_COLS].std()

    rows = []
    for cl in sorted(df_all["cluster"].unique()):
        sub   = df_all[df_all["cluster"] == cl]
        means = sub[["iic_final"] + DIMS_COLS].mean()
        stds  = sub[["iic_final"] + DIMS_COLS].std()
        zs    = {d: (means[d] - overall_means[d]) / max(overall_stds[d], 1e-9) for d in DIMS_COLS}
        row   = {
            "cluster":      cl,
            "n_municipios": len(sub),
            "iic_final_mean": round(means["iic_final"], 4),
            "iic_final_std":  round(stds["iic_final"], 4),
        }
        for d in DIMS_COLS:
            row[f"{d}_mean"] = round(float(means[d]), 4)
            row[f"{d}_std"]  = round(float(stds[d]), 4)
            row[f"{d}_zscore"] = round(float(zs[d]), 3)
        row["descricao_perfil"] = _auto_label(means, overall_means, overall_stds)
        rows.append(row)
    return pd.DataFrame(rows)


def save_outputs(df_all: pd.DataFrame, candidates: pd.DataFrame) -> None:
    # CSV: all municipalities with cluster label
    csv_cols = ["cluster"] + EXPORT_COLS_BASE
    df_all.sort_values(["cluster", "iic_final"], ascending=[True, False]).to_csv(
        OUT_CSV,
        columns=_present_cols(df_all, csv_cols),
        index=False,
        encoding="utf-8-sig",
    )
    print(f"\nSaved: {OUT_CSV}")

    # Excel: candidates
    cand_cols    = _present_cols(candidates, EXPORT_COLS_CANDIDATES)
    profile_df   = _build_profile_table(df_all)

    with pd.ExcelWriter(OUT_EXCEL, engine="openpyxl") as writer:
        # Sheet 1: cluster profiles with z-scores and auto-description
        profile_df.to_excel(writer, sheet_name="perfis_cluster", index=False)

        # Sheet 2: all candidates
        candidates.sort_values(
            ["cluster", "candidate_rank"]
        )[cand_cols].to_excel(writer, sheet_name="todos_candidatos", index=False)

        # One sheet per cluster
        for cl in sorted(candidates["cluster"].unique()):
            sub = candidates[candidates["cluster"] == cl][cand_cols]
            sub.to_excel(writer, sheet_name=f"cluster_{cl}", index=False)

    print(f"Saved: {OUT_EXCEL}")


# ─── Step 6: Visualisations ──────────────────────────────────────────────────

# Colour palette: 4 rose/pink shades with enough lightness contrast for maps.
_CLUSTER_PALETTE = ["#8c1a2a", "#c93060", "#e87a90", "#f5b8c0", "#1a9641", "#fdae61"]

_DIM_LABELS = {"ip": "IP\n(Grupos Prioritários)", "iv": "IV\n(Vulnerab. Socioecon.)",
               "ie": "IE\n(Exposição Climática)", "ig": "IG\n(Gestão Municipal)"}


def save_plots(df_all: pd.DataFrame) -> None:
    clusters   = sorted(df_all["cluster"].unique())
    n_clusters = len(clusters)
    palette    = _CLUSTER_PALETTE[:n_clusters]

    # ── 1. Z-score heatmap ────────────────────────────────────────────────────
    overall_means = df_all[DIMS_COLS].mean()
    overall_stds  = df_all[DIMS_COLS].std()

    z_matrix = []
    row_labels = []
    for cl in clusters:
        sub = df_all[df_all["cluster"] == cl]
        iic_mean = sub["iic_final"].mean()
        zs = [(sub[d].mean() - overall_means[d]) / max(overall_stds[d], 1e-9) for d in DIMS_COLS]
        z_matrix.append(zs)
        row_labels.append(f"Cluster {cl}  (IIC={iic_mean:.3f}, n={len(sub):,})")

    z_df = pd.DataFrame(z_matrix, index=row_labels,
                        columns=[_DIM_LABELS[d] for d in DIMS_COLS])

    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.9 * n_clusters + 1.5)))
    sns.heatmap(
        z_df, annot=True, fmt=".2f", center=0, cmap="RdYlGn_r",
        linewidths=0.5, linecolor="white", ax=ax,
        cbar_kws={"label": "Z-score  (verde = menor injustiça  |  vermelho = maior injustiça)", "shrink": 0.8},
        vmin=-2, vmax=2,
    )
    ax.set_title(f"Perfis dos Clusters — Z-scores por Dimensão (k={N_CLUSTERS})",
                 fontsize=12, pad=12)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9, rotation=0)
    fig.tight_layout()
    fig.savefig(OUT_HEATMAP, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_HEATMAP}")

    # ── 2. PCA scatter ────────────────────────────────────────────────────────
    mask = df_all[DIMS_COLS].notna().all(axis=1)
    df_pca = df_all[mask].copy()

    X_scaled = StandardScaler().fit_transform(df_pca[DIMS_COLS].values)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    for cl, color in zip(clusters, palette):
        idx = df_pca["cluster"].values == cl
        ax.scatter(
            coords[idx, 0], coords[idx, 1],
            c=color, label=f"Cluster {cl}", alpha=0.45, s=12, linewidths=0,
        )
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% da variância)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% da variância)", fontsize=10)
    ax.set_title(f"Municípios por Cluster — PCA das 4 Dimensões (k={N_CLUSTERS})", fontsize=12)
    ax.legend(title="Cluster", framealpha=0.8, fontsize=9)

    # Annotate loadings
    loadings = pca.components_.T
    scale = 1.8 / max(abs(loadings).max(), 1e-9)
    for i, dim in enumerate(DIMS_COLS):
        ax.annotate(
            "", xy=(loadings[i, 0] * scale, loadings[i, 1] * scale), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        )
        ax.text(loadings[i, 0] * scale * 1.12, loadings[i, 1] * scale * 1.12,
                dim.upper(), ha="center", va="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_PCA, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PCA}")

    # ── 3. Boxplots ───────────────────────────────────────────────────────────
    plot_cols = ["iic_final"] + DIMS_COLS
    col_labels = {
        "iic_final": "IIC Final",
        "ip": "IP", "iv": "IV", "ie": "IE", "ig": "IG",
    }
    n_cols = len(plot_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 4.5), sharey=False)

    for ax, col in zip(axes, plot_cols):
        data_by_cluster = [
            df_all.loc[df_all["cluster"] == cl, col].dropna().values
            for cl in clusters
        ]
        bp = ax.boxplot(
            data_by_cluster,
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(linewidth=1),
            capprops=dict(linewidth=1),
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
            widths=0.55,
        )
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        ax.set_title(col_labels[col], fontsize=11, pad=6)
        ax.set_xticks(range(1, n_clusters + 1))
        ax.set_xticklabels([f"C{cl}" for cl in clusters], fontsize=9)
        ax.set_xlabel("Cluster", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.tick_params(axis="y", labelsize=8)

    legend_patches = [
        mpatches.Patch(facecolor=c, alpha=0.85, label=f"Cluster {cl}")
        for cl, c in zip(clusters, palette)
    ]
    fig.legend(handles=legend_patches, loc="upper right", fontsize=9,
               title="Cluster", framealpha=0.8, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"Distribuição por Cluster — IIC e Sub-índices (k={N_CLUSTERS})",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_BOXPLOTS, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_BOXPLOTS}")

    # ── 4. Brazil choropleth map (full municipality polygons) ────────────────
    try:
        import geopandas as gpd
    except ImportError:
        print("geopandas not available — skipping map.")
        return

    print("Building choropleth map (loading geopackages)...")
    gdf_pais = gpd.read_file(PAIS_GPKG)
    gdf_uf   = gpd.read_file(UF_GPKG)
    gdf_mun  = gpd.read_file(MUNICIPIOS_GPKG, columns=["cd_mun", "geometry"])

    # Join cluster assignments
    df_join = df_all[["cd_mun", "cluster"]].copy()
    df_join["cd_mun"] = df_join["cd_mun"].astype(str).str.zfill(7)
    gdf_mun["cd_mun"] = gdf_mun["cd_mun"].astype(str).str.zfill(7)
    gdf_joined = gdf_mun.merge(df_join, on="cd_mun", how="left")

    color_dict = {cl: palette[i] for i, cl in enumerate(clusters)}
    gdf_joined["_color"] = gdf_joined["cluster"].map(color_dict).fillna("#cccccc")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("#cfe2f3")  # ocean background

    # Municipality fills (no edge — state borders provide the visual separation)
    gdf_joined.plot(ax=ax, color=gdf_joined["_color"],
                    linewidth=0, edgecolor="none", zorder=2)

    # State borders on top for geographic reference
    gdf_uf.boundary.plot(ax=ax, color="white", linewidth=0.7, zorder=3)

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(
        f"Distribuição dos Clusters por Município — Brasil (k={N_CLUSTERS})",
        fontsize=12, pad=10,
    )

    legend_patches = [
        mpatches.Patch(facecolor=color_dict[cl], label=f"Cluster {cl}  (n={len(df_all[df_all['cluster']==cl]):,})")
        for cl in clusters
    ]
    ax.legend(handles=legend_patches, title="Cluster", fontsize=9,
              framealpha=0.9, loc="lower right", title_fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_MAP, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_MAP}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 65)
    print("Cluster Analysis — Case Study Municipality Selection")
    print(f"Results file : {RESULTS_FILE.name}")
    print(f"Tipologias   : {TIPOLOGIAS_FILE.name}")
    print("=" * 65)

    agg        = load_and_aggregate()
    df_km      = run_kmeans(agg)
    df_merged  = merge_tipologias(df_km)
    candidates = select_candidates(df_merged)

    print("\nCandidates selected per cluster:")
    for cl in sorted(candidates["cluster"].unique()):
        sub = candidates[candidates["cluster"] == cl]
        print(f"  Cluster {cl}: {len(sub)} candidates")
        for _, row in sub.iterrows():
            reg   = row.get("regiao", "?")
            porte = row.get("porte_cnas_mds", "?")
            print(
                f"    [{row['candidate_rank']}] {row['nm_mun']} / {row['nm_uf']} "
                f"| {reg} | {porte} | IIC={row['iic_final']:.3f}"
            )

    save_outputs(df_merged, candidates)
    save_plots(df_merged)
    print(f"\nDone. Outputs in:\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
