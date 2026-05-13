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

Usage:
    python diagnose/analysis/cluster_municipios.py          # default k=4
    # edit N_CLUSTERS below and re-run for a different k

Dependencies: pandas, scikit-learn, openpyxl (all in project venv)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
OUT_CSV   = OUT_DIR / f"clusters_municipios_k{N_CLUSTERS}.csv"
OUT_EXCEL = OUT_DIR / f"candidatos_estudo_de_caso_k{N_CLUSTERS}.xlsx"


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
        # IG is inverted in the IIC: low IG = poor governance = more injustice
        "high_means": "boa capacidade de gestao municipal",
        "low_means":  "deficit de gestao municipal (pior contribuicao ao IIC)",
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
    print(f"  Note: IG is inverted in the IIC — lower IG score = greater injustice contribution")
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
    print(f"\nDone. Outputs in:\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
