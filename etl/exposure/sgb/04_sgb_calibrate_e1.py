#!/usr/bin/env python3
"""
SGB — Calibração do Threshold E1 (NASA LHASA vs SGB Massa)
===========================================================
Compara o indicador E1 (lhasa_high_frac = e1_des_abs) com a cartografia de
suscetibilidade a movimentos de massa do SGB para encontrar o threshold ótimo.

Referência positiva SGB: sgb_alta_mta_frac > 0.3
  (≥30% da área SGB mapeada do hexágono em classes Alta ou Muito Alta)
Filtro de cobertura:   sgb_coverage_frac >= 0.5
  (pelo menos metade do hexágono coberta por dados SGB)

Análises:
  1. Sweep de threshold em e1_des_abs (= lhasa_high_frac)
  2. Sweep em lhasa_mean (se CSVs GEE disponíveis) — variante alternativa
  3. F1 por macrorregião no threshold ótimo

Inputs (via cfg + config.local.json):
  cfg.FILES_H3["e1"]         — br_h3_e1_deslizamentos.parquet
  br_h3_sgb_massa.parquet    — output do 03_sgb_h3_intersect.py
  GEE CSVs de LHASA          — para lhasa_mean (opcional)

Outputs em cfg.DIAGNOSE_DIR:
  diagnostic_e1_calibration_<timestamp>.txt
  diagnostic_e1_calibration_<timestamp>.csv   — sweep completo por threshold

USO:
  python 05_sgb_calibrate_e1.py
  python 05_sgb_calibrate_e1.py --sgb-ref 0.2   # muda threshold SGB (padrão: 0.3)
  python 05_sgb_calibrate_e1.py --min-coverage 0.3
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from src import config as cfg


def _load_data_dir() -> Path:
    config_path = PROJECT_ROOT / "config" / "config.local.json"
    with open(config_path, encoding="utf-8") as f:
        return Path(json.load(f)["data_dir"])


_DATA_DIR        = _load_data_dir()
SGB_MASSA_PATH   = _DATA_DIR / "inputs/clean/br_h3_sgb_massa.parquet"
E1_PATH          = cfg.FILES_H3["e1"]
GEE_DIR          = cfg.RAW_DIR / cfg.INDICATORS["e1"]["source"]["dir"]
DIAGNOSE_DIR     = cfg.DIAGNOSE_DIR

E1_ABS_COL = "e1_des_abs"   # = lhasa_high_frac, range 0–1

MACRORREGIOES = {
    "N":  {"AC", "AM", "AP", "PA", "RO", "RR", "TO"},
    "NE": {"AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"},
    "CO": {"DF", "GO", "MS", "MT"},
    "SE": {"ES", "MG", "RJ", "SP"},
    "S":  {"PR", "RS", "SC"},
}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ══════════════════════════════════════════════════════════════════════════════

def _macrorregiao(uf: str) -> str:
    for macro, states in MACRORREGIOES.items():
        if uf in states:
            return macro
    return "?"


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1}


def threshold_sweep(
    score: pd.Series,
    ref: pd.Series,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        pred = score > t
        m = compute_metrics(ref, pred)
        rows.append({"threshold": round(float(t), 4), **m,
                     "n_pred_pos": int(pred.sum()), "n_ref_pos": int(ref.sum())})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# CARREGAMENTO
# ══════════════════════════════════════════════════════════════════════════════

def load_data(sgb_ref_thresh: float, min_coverage: float) -> pd.DataFrame:
    print(f"\n1/3 — Carregando dados")

    if not SGB_MASSA_PATH.exists():
        raise FileNotFoundError(
            f"SGB massa parquet não encontrado: {SGB_MASSA_PATH}\n"
            "Execute 03_sgb_h3_intersect.py primeiro."
        )
    if not E1_PATH.exists():
        raise FileNotFoundError(
            f"Parquet E1 não encontrado: {E1_PATH}\n"
            "Execute etl/exposure/e1_deslizamentos_lhasa.py primeiro."
        )

    sgb = pd.read_parquet(SGB_MASSA_PATH)
    e1  = pd.read_parquet(E1_PATH, columns=["h3_id", E1_ABS_COL])

    print(f"   SGB massa:  {len(sgb):,} hexágonos com cobertura SGB")
    print(f"   E1:         {len(e1):,} hexágonos (grade nacional)")

    df = sgb.merge(e1, on="h3_id", how="inner")
    print(f"   Após join:  {len(df):,} hexágonos")

    # Aplica filtros
    df = df[df["sgb_coverage_frac"] >= min_coverage].copy()
    print(f"   Cobertura SGB ≥ {min_coverage:.0%}: {len(df):,} hexágonos")

    df[E1_ABS_COL] = pd.to_numeric(df[E1_ABS_COL], errors="coerce").fillna(0.0)
    df["sgb_ref"]  = df["sgb_alta_mta_frac"] > sgb_ref_thresh
    df["macro"]    = df["cd_estado"].map(_macrorregiao)

    pos = df["sgb_ref"].sum()
    print(f"   Referência SGB positiva (alta suscet.): {pos:,} ({pos/len(df):.1%})")

    return df


def load_lhasa_mean() -> pd.DataFrame | None:
    """Carrega lhasa_mean dos CSVs GEE, se disponíveis."""
    csvs = sorted(GEE_DIR.glob("*.csv")) if GEE_DIR.exists() else []
    if not csvs:
        return None
    parts = []
    for p in csvs:
        try:
            df = pd.read_csv(p, usecols=["h3_id", "lhasa_mean"])
            parts.append(df)
        except Exception:
            pass
    if not parts:
        return None
    df_all = pd.concat(parts, ignore_index=True).drop_duplicates("h3_id")
    df_all["lhasa_mean"] = pd.to_numeric(df_all["lhasa_mean"], errors="coerce")
    return df_all[["h3_id", "lhasa_mean"]]


# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISES
# ══════════════════════════════════════════════════════════════════════════════

def analyse_primary(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Sweep de threshold em e1_des_abs (lhasa_high_frac)."""
    print(f"\n2/3 — Calibração (e1_des_abs = lhasa_high_frac)")
    thresholds = np.arange(0.0, 1.01, 0.05)
    sweep = threshold_sweep(df[E1_ABS_COL], df["sgb_ref"], thresholds)

    best_idx  = sweep["f1"].idxmax()
    best_row  = sweep.iloc[best_idx]
    best_t    = best_row["threshold"]

    print(f"   Threshold ótimo : {best_t:.2f}")
    print(f"   Precision        : {best_row['precision']:.3f}")
    print(f"   Recall           : {best_row['recall']:.3f}")
    print(f"   F1               : {best_row['f1']:.3f}")
    print(f"   TP/FP/FN/TN      : {best_row['tp']}/{best_row['fp']}/{best_row['fn']}/{best_row['tn']}")

    # Distribui hexágonos pelo score para mostrar qual fração passa cada threshold
    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        n = (df[E1_ABS_COL] > t).sum()
        print(f"   e1_des_abs > {t:.1f} : {n:,} hexágonos ({n/len(df):.1%})")

    return sweep, {"threshold": best_t, **best_row.to_dict()}


def analyse_lhasa_mean(df: pd.DataFrame) -> dict | None:
    """Teste secundário: lhasa_mean >= t como preditor alternativo."""
    lhasa_df = load_lhasa_mean()
    if lhasa_df is None:
        print("\n2b/3 — lhasa_mean: CSVs GEE não encontrados — análise secundária pulada")
        return None

    df2 = df.merge(lhasa_df, on="h3_id", how="left")
    if df2["lhasa_mean"].isna().all():
        print("\n2b/3 — lhasa_mean: sem dados após join — pulando")
        return None

    n_valid = df2["lhasa_mean"].notna().sum()
    print(f"\n2b/3 — lhasa_mean ({n_valid:,} hexágonos com dado)")

    thresholds = np.arange(1.0, 5.01, 0.25)
    sweep = threshold_sweep(df2["lhasa_mean"].fillna(0), df2["sgb_ref"], thresholds)
    best_row = sweep.iloc[sweep["f1"].idxmax()]

    print(f"   Threshold ótimo lhasa_mean : >= {best_row['threshold']:.2f}")
    print(f"   F1                          : {best_row['f1']:.3f}")
    return {"threshold": best_row["threshold"], **best_row.to_dict(),
            "sweep": sweep}


def analyse_regional(df: pd.DataFrame, best_threshold: float) -> pd.DataFrame:
    """F1 por macrorregião no threshold ótimo."""
    print(f"\n3/3 — Análise regional (threshold {best_threshold:.2f})")

    df["pred"] = df[E1_ABS_COL] > best_threshold
    rows = []
    for macro in sorted(df["macro"].unique()):
        sub  = df[df["macro"] == macro]
        m    = compute_metrics(sub["sgb_ref"], sub["pred"])
        rows.append({"macro": macro, "n": len(sub),
                     "ref_pos_pct": f"{sub['sgb_ref'].mean():.1%}",
                     **m})
        print(f"   {macro:2}  n={len(sub):>5,}  F1={m['f1']:.3f}  "
              f"prec={m['precision']:.3f}  rec={m['recall']:.3f}")

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNÓSTICO
# ══════════════════════════════════════════════════════════════════════════════

def write_diagnostic(
    df: pd.DataFrame,
    sweep: pd.DataFrame,
    best: dict,
    mean_result: dict | None,
    regional: pd.DataFrame,
    sgb_ref_thresh: float,
    min_coverage: float,
    ts: str,
) -> None:
    txt_path = DIAGNOSE_DIR / f"diagnostic_e1_calibration_{ts}.txt"
    csv_path = DIAGNOSE_DIR / f"diagnostic_e1_calibration_{ts}.csv"

    current_threshold = 0.0  # e1_des_abs > 0 é a classificação atual (qualquer LHASA ≥4)

    # Métricas com threshold atual (> 0) para comparação
    m_current = compute_metrics(df["sgb_ref"], df[E1_ABS_COL] > current_threshold)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("SGB — Calibração Threshold E1 (lhasa_high_frac vs SGB Massa)\n")
        f.write(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("--- Parâmetros ---\n")
        f.write(f"  Referência SGB positiva : sgb_alta_mta_frac > {sgb_ref_thresh}\n")
        f.write(f"  Filtro cobertura SGB    : sgb_coverage_frac >= {min_coverage}\n")
        f.write(f"  Hexágonos na análise    : {len(df):,}\n")
        f.write(f"  Referência positiva     : {df['sgb_ref'].sum():,} ({df['sgb_ref'].mean():.1%})\n\n")

        f.write("--- Threshold atual (e1_des_abs > 0, i.e. qualquer LHASA >= 4) ---\n")
        f.write(f"  Precision : {m_current['precision']:.4f}\n")
        f.write(f"  Recall    : {m_current['recall']:.4f}\n")
        f.write(f"  F1        : {m_current['f1']:.4f}\n\n")

        f.write("--- Threshold ótimo (sweep em e1_des_abs) ---\n")
        f.write(f"  Threshold  : e1_des_abs > {best['threshold']:.2f}\n")
        f.write(f"  Precision  : {best['precision']:.4f}\n")
        f.write(f"  Recall     : {best['recall']:.4f}\n")
        f.write(f"  F1         : {best['f1']:.4f}\n")
        f.write(f"  TP/FP/FN/TN: {best['tp']}/{best['fp']}/{best['fn']}/{best['tn']}\n\n")

        improvement = best["f1"] - m_current["f1"]
        if improvement > 0.02:
            f.write(f"  ⚑ RECOMENDAÇÃO: ajustar threshold para {best['threshold']:.2f} "
                    f"(melhora F1 em {improvement:+.3f})\n\n")
        else:
            f.write(f"  ✓ Threshold atual (> 0) adequado — melhora marginal ({improvement:+.3f})\n\n")

        if mean_result:
            f.write("--- Teste secundário: lhasa_mean >= t ---\n")
            f.write(f"  Threshold ótimo : lhasa_mean >= {mean_result['threshold']:.2f}\n")
            f.write(f"  F1              : {mean_result['f1']:.4f}\n")
            delta = mean_result["f1"] - best["f1"]
            if delta > 0.02:
                f.write(f"  ⚑ lhasa_mean supera lhasa_high_frac em {delta:+.3f}\n")
                f.write("    → Considere exportar 'lhasa_med_high_frac' (frac com LHASA>=3) "
                        "do GEE e re-rodar\n\n")
            else:
                f.write(f"  ✓ lhasa_high_frac é a variável adequada (lhasa_mean "
                        f"{'pior' if delta < 0 else 'equivalente'}: {delta:+.3f})\n\n")

        f.write("--- F1 por macrorregião ---\n")
        f.write(regional.to_string(index=False))
        f.write("\n\n")

        min_f1_macro = regional.loc[regional["f1"].idxmin()]
        if regional["f1"].min() < best["f1"] - 0.15:
            f.write(f"  ⚑ Macrorregião {min_f1_macro['macro']} tem F1 substancialmente "
                    f"menor ({min_f1_macro['f1']:.3f}). Avaliar threshold regional.\n\n")
        else:
            f.write("  ✓ F1 relativamente uniforme entre macrorregiões\n\n")

        f.write("--- Sweep completo (top 10 por F1) ---\n")
        f.write(sweep.nlargest(10, "f1").to_string(index=False))
        f.write("\n")

    sweep.to_csv(csv_path, index=False, float_format="%.4f", encoding="utf-8")

    print(f"\n   ✓ Diagnóstico TXT: {txt_path.name}")
    print(f"   ✓ Sweep CSV:       {csv_path.name}")

    if mean_result and mean_result["f1"] - best["f1"] > 0.02:
        print("\n   ⚑ AÇÃO: lhasa_mean performa melhor → re-exportar lhasa_med_high_frac do GEE")
    elif best["threshold"] > 0.02:
        print(f"\n   ⚑ AÇÃO: ajustar threshold de E1 para e1_des_abs > {best['threshold']:.2f} "
              f"em e1_deslizamentos_lhasa.py")
    else:
        print("\n   ✓ Threshold atual adequado — sem ação necessária no pipeline")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SGB — Calibração threshold E1 vs SGB massa"
    )
    parser.add_argument(
        "--sgb-ref", type=float, default=0.3,
        help="Threshold sgb_alta_mta_frac para classificar hexágono como alto risco SGB (padrão: 0.3)"
    )
    parser.add_argument(
        "--min-coverage", type=float, default=0.5,
        help="Cobertura SGB mínima do hexágono (padrão: 0.5)"
    )
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print("SGB — Calibração Threshold E1 (LHASA vs Movimentos de Massa)")
    print("=" * 70)

    df = load_data(args.sgb_ref, args.min_coverage)
    sweep, best = analyse_primary(df)
    mean_result  = analyse_lhasa_mean(df)
    regional     = analyse_regional(df, best["threshold"])

    DIAGNOSE_DIR.mkdir(parents=True, exist_ok=True)
    write_diagnostic(df, sweep, best, mean_result, regional,
                     args.sgb_ref, args.min_coverage, ts)
    print("\nConcluído.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERROMPIDO] Ctrl+C — diagnóstico não salvo.")
        sys.exit(1)
