"""
Validação E1: correlação entre slope (Copernicus GLO-30) e NASA LHASA.

Input:  CSV exportado pelo Step 3 do GEE script (h3_validacao_lhasa_vs_slope_v1.csv)
        Localização esperada: mesma pasta dos CSVs do Step 2
        Colunas: h3_id, slope_alta_media, lhasa_mean, lhasa_high_frac

Output: arquivo de diagnóstico em cfg.DIAGNOSE_DIR
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
from scipy import stats

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(PROJECT_ROOT)
from src import config as cfg

# ==============================================================================
# 1. PATHS
# ==============================================================================
GEE_DIR      = cfg.RAW_DIR / cfg.INDICATORS["e1"]["source"]["dir"]
VALIDATION_CSV = GEE_DIR / "h3_validacao_lhasa_vs_slope_v1.csv"

now = datetime.now().strftime("%Y%m%d_%H%M%S")
DIAGNOSTIC_TXT = cfg.DIAGNOSE_DIR / f"diagnostic_e1_validacao_lhasa_{now}.txt"


# ==============================================================================
# 2. MAIN
# ==============================================================================
def main():
    print("=" * 60)
    print("Validação E1: Slope (GLO-30) × NASA LHASA")
    print(f"Source: {VALIDATION_CSV}")
    print("=" * 60)

    if not VALIDATION_CSV.exists():
        raise FileNotFoundError(
            f"CSV de validação não encontrado: {VALIDATION_CSV}\n"
            f"Execute o Step 3 do GEE script e coloque o resultado em:\n{GEE_DIR}"
        )

    df = pd.read_csv(VALIDATION_CSV)
    df.columns = df.columns.str.lower()
    print(f"\nHexágonos carregados: {len(df):,}")

    required = ["slope_alta_media", "lhasa_mean", "lhasa_high_frac"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}\n  Colunas encontradas: {list(df.columns)}")

    # Converte e remove linhas sem dados
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df_clean = df.dropna(subset=required)
    n_dropped = len(df) - len(df_clean)
    if n_dropped > 0:
        print(f"   {n_dropped:,} linhas removidas por NaN")
    print(f"   Hexágonos válidos para análise: {len(df_clean):,}")

    # ==============================================================================
    # 3. CORRELAÇÕES
    # ==============================================================================
    print("\n--- Correlações ---")

    pairs = [
        ("slope_alta_media", "lhasa_high_frac",  "Slope vs LHASA High Frac (principal)"),
        ("slope_alta_media", "lhasa_mean",        "Slope vs LHASA Mean (1–5)"),
    ]

    results = {}
    for col_a, col_b, label in pairs:
        r_pearson, p_pearson   = stats.pearsonr(df_clean[col_a], df_clean[col_b])
        r_spearman, p_spearman = stats.spearmanr(df_clean[col_a], df_clean[col_b])
        results[(col_a, col_b)] = {
            "label": label,
            "pearson_r": r_pearson, "pearson_p": p_pearson,
            "spearman_r": r_spearman, "spearman_p": p_spearman,
        }
        print(f"\n  {label}")
        print(f"    Pearson  r = {r_pearson:+.3f}  (p = {p_pearson:.2e})")
        print(f"    Spearman r = {r_spearman:+.3f}  (p = {p_spearman:.2e})")

    # ==============================================================================
    # 4. ESTATÍSTICAS DESCRITIVAS
    # ==============================================================================
    print("\n--- Estatísticas descritivas ---")
    print(df_clean[required].describe().to_string())

    # Quantis do slope por classe LHASA
    df_clean["lhasa_class"] = pd.cut(
        df_clean["lhasa_mean"],
        bins=[0, 1.5, 2.5, 3.5, 4.5, 5.1],
        labels=["1-Very Low", "2-Low", "3-Moderate", "4-High", "5-Very High"],
        right=False
    )
    print("\n--- Slope médio por classe LHASA ---")
    tbl = df_clean.groupby("lhasa_class", observed=True)["slope_alta_media"].agg(["mean", "median", "count"])
    print(tbl.to_string())

    _write_diagnostic(df_clean, results)
    print(f"\nDiagnóstico salvo: {DIAGNOSTIC_TXT}")
    print("Done!")


def _write_diagnostic(df, results):
    r_main = results[("slope_alta_media", "lhasa_high_frac")]
    r_main_val = r_main["spearman_r"]

    if abs(r_main_val) >= 0.5:
        interpretation = "Correlação razoável — os métodos concordam."
    elif abs(r_main_val) >= 0.3:
        interpretation = "Correlação moderada — concordância parcial (esperado dado diferença de resolução 30m vs 1km)."
    else:
        interpretation = "Correlação fraca — os métodos capturam padrões distintos (slope = geomorfologia pura; LHASA = multicritério)."

    with open(DIAGNOSTIC_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Validação E1 — Slope (GLO-30) × NASA LHASA\n")
        f.write(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Hexágonos analisados: {len(df):,}\n\n")

        for (col_a, col_b), res in results.items():
            f.write(f"--- {res['label']} ---\n")
            f.write(f"  Pearson  r = {res['pearson_r']:+.4f}  (p = {res['pearson_p']:.2e})\n")
            f.write(f"  Spearman r = {res['spearman_r']:+.4f}  (p = {res['spearman_p']:.2e})\n\n")

        f.write(f"Interpretação: {interpretation}\n\n")

        f.write("--- Estatísticas descritivas ---\n")
        f.write(df[["slope_alta_media", "lhasa_mean", "lhasa_high_frac"]].describe().to_string())
        f.write("\n\n")

        f.write("--- Slope médio por classe LHASA ---\n")
        df2 = df.copy()
        df2["lhasa_class"] = pd.cut(
            df2["lhasa_mean"],
            bins=[0, 1.5, 2.5, 3.5, 4.5, 5.1],
            labels=["1-Very Low", "2-Low", "3-Moderate", "4-High", "5-Very High"],
            right=False
        )
        tbl = df2.groupby("lhasa_class", observed=True)["slope_alta_media"].agg(["mean", "median", "count"])
        f.write(tbl.to_string())
        f.write("\n")


if __name__ == "__main__":
    main()
