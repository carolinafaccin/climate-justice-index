#!/usr/bin/env python3
"""
SGB — Validação do Threshold E2 (flood_score vs SGB Inundações)
================================================================
Compara o indicador E2 (flood_score = e2_inu_abs) com a cartografia de
suscetibilidade a inundações/enxurradas do SGB para validar o threshold atual
e analisar falsos negativos.

Referência positiva SGB: sgb_alta_mta_frac > 0.3
Filtro de cobertura:   sgb_coverage_frac >= 0.5

Análises:
  1. Sweep de threshold em e2_inu_abs (flood_score)
  2. Análise de falsos negativos: onde o E2 não captura o que o SGB aponta
  3. F1 por macrorregião no threshold ótimo

Inputs (via cfg + config.local.json):
  cfg.FILES_H3["e2"]              — br_h3_e2_inundacoes.parquet
  br_h3_sgb_inundacoes.parquet   — output do 03_sgb_h3_intersect.py

Outputs em cfg.DIAGNOSE_DIR:
  diagnostic_e2_validation_<timestamp>.txt
  diagnostic_e2_validation_<timestamp>.csv   — sweep completo por threshold

USO:
  python 06_sgb_validate_e2.py
  python 06_sgb_validate_e2.py --sgb-ref 0.2   # muda threshold SGB (padrão: 0.3)
  python 06_sgb_validate_e2.py --min-coverage 0.3
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


_DATA_DIR           = _load_data_dir()
SGB_INUND_PATH      = _DATA_DIR / "inputs/clean/br_h3_sgb_inundacoes.parquet"
E2_PATH             = cfg.FILES_H3["e2"]
DIAGNOSE_DIR        = cfg.DIAGNOSE_DIR

E2_ABS_COL = "e2_inu_abs"   # = flood_score (HAND × JRC), range 0–1

# Threshold atual do E2 no IIC: flood_score > 0 (qualquer sinal de inundação)
CURRENT_E2_THRESHOLD = 0.0

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

    if not SGB_INUND_PATH.exists():
        raise FileNotFoundError(
            f"SGB inundações parquet não encontrado: {SGB_INUND_PATH}\n"
            "Execute 03_sgb_h3_intersect.py --tipo inundacao primeiro."
        )
    if not E2_PATH.exists():
        raise FileNotFoundError(
            f"Parquet E2 não encontrado: {E2_PATH}\n"
            "Execute etl/exposure/e2_inundacoes_hand.py primeiro."
        )

    sgb = pd.read_parquet(SGB_INUND_PATH)
    e2  = pd.read_parquet(E2_PATH, columns=["h3_id", E2_ABS_COL])

    print(f"   SGB inundações: {len(sgb):,} hexágonos com cobertura SGB")
    print(f"   E2:             {len(e2):,} hexágonos (grade nacional)")

    df = sgb.merge(e2, on="h3_id", how="inner")
    print(f"   Após join:      {len(df):,} hexágonos")

    df = df[df["sgb_coverage_frac"] >= min_coverage].copy()
    print(f"   Cobertura SGB ≥ {min_coverage:.0%}: {len(df):,} hexágonos")

    df[E2_ABS_COL] = pd.to_numeric(df[E2_ABS_COL], errors="coerce").fillna(0.0)
    df["sgb_ref"]  = df["sgb_alta_mta_frac"] > sgb_ref_thresh
    df["macro"]    = df["cd_estado"].map(_macrorregiao)

    pos = df["sgb_ref"].sum()
    print(f"   Referência SGB positiva (alta suscet.): {pos:,} ({pos/len(df):.1%})")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISES
# ══════════════════════════════════════════════════════════════════════════════

def analyse_primary(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Sweep de threshold em e2_inu_abs (flood_score)."""
    print(f"\n2/3 — Validação (e2_inu_abs = flood_score)")

    # Métricas com threshold atual para referência
    m_current = compute_metrics(df["sgb_ref"], df[E2_ABS_COL] > CURRENT_E2_THRESHOLD)
    print(f"   Threshold atual (> {CURRENT_E2_THRESHOLD}):")
    print(f"     Precision {m_current['precision']:.3f}  "
          f"Recall {m_current['recall']:.3f}  F1 {m_current['f1']:.3f}")

    # Sweep mais fino (0.01 steps) pois flood_score tem distribuição muito concentrada em 0
    thresholds = np.round(np.arange(0.0, 1.001, 0.01), 3)
    sweep = threshold_sweep(df[E2_ABS_COL], df["sgb_ref"], thresholds)

    best_idx = sweep["f1"].idxmax()
    best_row = sweep.iloc[best_idx]
    best_t   = best_row["threshold"]

    print(f"   Threshold ótimo : {best_t:.3f}")
    print(f"   Precision        : {best_row['precision']:.3f}")
    print(f"   Recall           : {best_row['recall']:.3f}")
    print(f"   F1               : {best_row['f1']:.3f}")

    # Distribuição de flood_score para entender a escala
    print("\n   Distribuição de e2_inu_abs (flood_score):")
    for q in [0.50, 0.75, 0.90, 0.95, 0.99]:
        v = df[E2_ABS_COL].quantile(q)
        print(f"     P{q*100:.0f}: {v:.4f}")

    return sweep, {"threshold": best_t, **best_row.to_dict(),
                   "m_current": m_current}


def analyse_false_negatives(df: pd.DataFrame, best_threshold: float) -> pd.DataFrame:
    """
    Analisa falsos negativos: hexágonos onde o SGB indica alta suscetibilidade
    a inundações mas o E2 não detecta (flood_score <= threshold).
    """
    print(f"\n   Análise de falsos negativos (SGB+ mas E2 ≤ {best_threshold:.3f})")

    df["pred"] = df[E2_ABS_COL] > best_threshold
    fn_mask    = df["sgb_ref"] & ~df["pred"]
    fn         = df[fn_mask].copy()

    print(f"   Falsos negativos: {len(fn):,} de {df['sgb_ref'].sum():,} ref+ "
          f"({len(fn)/max(df['sgb_ref'].sum(),1):.1%})")

    if fn.empty:
        return pd.DataFrame()

    # Distribuição de flood_score nos FN (quão perto estão do threshold?)
    print("\n   flood_score nos falsos negativos:")
    print(f"     Mediana : {fn[E2_ABS_COL].median():.4f}")
    print(f"     Máximo  : {fn[E2_ABS_COL].max():.4f}")
    print(f"     = 0     : {(fn[E2_ABS_COL] == 0).sum():,} ({(fn[E2_ABS_COL] == 0).mean():.1%})")
    print(f"     (0, t]  : {((fn[E2_ABS_COL] > 0) & ~df.loc[fn.index,'pred']).sum():,}")

    # Distribuição por macrorregião
    print("\n   Falsos negativos por macrorregião:")
    macro_fn = (
        fn.groupby("macro")
        .agg(n_fn=("h3_id", "count"),
             sgb_frac_mean=("sgb_alta_mta_frac", "mean"),
             e2_score_mean=(E2_ABS_COL, "mean"),
             e2_zeros=(E2_ABS_COL, lambda x: (x == 0).mean()))
        .reset_index()
        .sort_values("n_fn", ascending=False)
    )
    # Adiciona total de SGB+ por macrorregião para calcular taxa de FN
    ref_pos_by_macro = df[df["sgb_ref"]].groupby("macro").size().rename("n_ref_pos")
    macro_fn = macro_fn.merge(ref_pos_by_macro, on="macro", how="left")
    macro_fn["fn_rate"] = macro_fn["n_fn"] / macro_fn["n_ref_pos"]

    for _, row in macro_fn.iterrows():
        print(f"     {row['macro']:2}  FN={row['n_fn']:>4,}  "
              f"taxa_FN={row['fn_rate']:.1%}  "
              f"sgb_frac={row['sgb_frac_mean']:.2f}  "
              f"e2_score={row['e2_score_mean']:.3f}  "
              f"e2=0: {row['e2_zeros']:.0%}")

    # Distribuição por classe SGB máxima nos FN
    print("\n   Classe SGB máxima nos falsos negativos:")
    class_dist = fn["sgb_max_class"].value_counts().sort_index()
    for cls, count in class_dist.items():
        label = {5: "Muito Alta", 4: "Alta", 3: "Média", 2: "Baixa", 1: "Muito Baixa"}.get(cls, str(cls))
        print(f"     classe {cls} ({label:12}): {count:,} ({count/len(fn):.1%})")

    return macro_fn


def analyse_regional(df: pd.DataFrame, best_threshold: float) -> pd.DataFrame:
    """F1 por macrorregião no threshold ótimo."""
    print(f"\n   F1 por macrorregião (threshold {best_threshold:.3f})")
    df["pred"] = df[E2_ABS_COL] > best_threshold
    rows = []
    for macro in sorted(df["macro"].unique()):
        sub = df[df["macro"] == macro]
        m   = compute_metrics(sub["sgb_ref"], sub["pred"])
        rows.append({"macro": macro, "n": len(sub),
                     "ref_pos_pct": f"{sub['sgb_ref'].mean():.1%}", **m})
        print(f"     {macro:2}  n={len(sub):>5,}  F1={m['f1']:.3f}  "
              f"prec={m['precision']:.3f}  rec={m['recall']:.3f}")
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNÓSTICO
# ══════════════════════════════════════════════════════════════════════════════

def write_diagnostic(
    df: pd.DataFrame,
    sweep: pd.DataFrame,
    best: dict,
    macro_fn: pd.DataFrame,
    regional: pd.DataFrame,
    sgb_ref_thresh: float,
    min_coverage: float,
    ts: str,
) -> None:
    txt_path = DIAGNOSE_DIR / f"diagnostic_e2_validation_{ts}.txt"
    csv_path = DIAGNOSE_DIR / f"diagnostic_e2_validation_{ts}.csv"

    m_current = best.pop("m_current")
    improvement = best["f1"] - m_current["f1"]

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("SGB — Validação Threshold E2 (flood_score vs SGB Inundações)\n")
        f.write(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("--- Parâmetros ---\n")
        f.write(f"  Referência SGB positiva : sgb_alta_mta_frac > {sgb_ref_thresh}\n")
        f.write(f"  Filtro cobertura SGB    : sgb_coverage_frac >= {min_coverage}\n")
        f.write(f"  Hexágonos na análise    : {len(df):,}\n")
        f.write(f"  Referência positiva     : {df['sgb_ref'].sum():,} ({df['sgb_ref'].mean():.1%})\n\n")

        f.write(f"--- Threshold atual (e2_inu_abs > {CURRENT_E2_THRESHOLD}) ---\n")
        f.write(f"  Precision : {m_current['precision']:.4f}\n")
        f.write(f"  Recall    : {m_current['recall']:.4f}\n")
        f.write(f"  F1        : {m_current['f1']:.4f}\n\n")

        f.write("--- Threshold ótimo (sweep em e2_inu_abs) ---\n")
        f.write(f"  Threshold  : e2_inu_abs > {best['threshold']:.3f}\n")
        f.write(f"  Precision  : {best['precision']:.4f}\n")
        f.write(f"  Recall     : {best['recall']:.4f}\n")
        f.write(f"  F1         : {best['f1']:.4f}\n")
        f.write(f"  TP/FP/FN/TN: {best['tp']}/{best['fp']}/{best['fn']}/{best['tn']}\n\n")

        if improvement > 0.02:
            f.write(f"  ⚑ RECOMENDAÇÃO: ajustar threshold para {best['threshold']:.3f} "
                    f"(melhora F1 em {improvement:+.3f})\n\n")
        else:
            f.write(f"  ✓ Threshold atual adequado — melhora marginal ({improvement:+.3f})\n\n")

        if not macro_fn.empty:
            fn_total = df["sgb_ref"].sum()
            fn_zero_pct = (df[df["sgb_ref"] & (df[E2_ABS_COL] == 0)]).__len__() / max(fn_total, 1)
            f.write("--- Análise de falsos negativos ---\n")
            f.write(f"  Total FN: {best['fn']:,} ({best['fn']/max(fn_total,1):.1%} dos casos SGB+)\n")
            f.write(f"  FN com flood_score=0 (sem sinal HAND+JRC): {fn_zero_pct:.1%}\n\n")
            f.write("  Por macrorregião:\n")
            f.write(macro_fn.to_string(index=False))
            f.write("\n\n")

            # Interpretação dos FN
            if not macro_fn.empty:
                worst_macro = macro_fn.loc[macro_fn["fn_rate"].idxmax()]
                high_zero   = macro_fn[macro_fn["e2_zeros"] > 0.7]
                if not high_zero.empty:
                    regions = ", ".join(high_zero["macro"].tolist())
                    f.write(f"  ⚑ Regiões {regions}: >70% dos FN têm flood_score=0 "
                            f"→ áreas não cobertas pelo JRC ou HAND>6m.\n"
                            f"    Avaliar ampliar teto HAND em h3_e2_inundacoes_hand_gee_v1.js\n\n")

        f.write("--- F1 por macrorregião ---\n")
        f.write(regional.to_string(index=False))
        f.write("\n\n")

        if regional["f1"].min() < best["f1"] - 0.15:
            worst = regional.loc[regional["f1"].idxmin()]
            f.write(f"  ⚑ Macrorregião {worst['macro']} tem F1 substancialmente "
                    f"menor ({worst['f1']:.3f}). Investigar cobertura JRC na região.\n\n")
        else:
            f.write("  ✓ F1 relativamente uniforme entre macrorregiões\n\n")

        f.write("--- Sweep completo (top 15 por F1) ---\n")
        f.write(sweep.nlargest(15, "f1").to_string(index=False))
        f.write("\n")

    sweep.to_csv(csv_path, index=False, float_format="%.4f", encoding="utf-8")

    print(f"\n   ✓ Diagnóstico TXT: {txt_path.name}")
    print(f"   ✓ Sweep CSV:       {csv_path.name}")

    if improvement > 0.02:
        print(f"\n   ⚑ AÇÃO: ajustar threshold de E2 para e2_inu_abs > {best['threshold']:.3f} "
              f"em e2_inundacoes_hand.py")
        if best["threshold"] > 0.1:
            print("    → threshold >0.1 pode indicar necessidade de ampliar teto HAND no GEE")
    else:
        print("\n   ✓ Threshold atual adequado — sem ação necessária no pipeline")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SGB — Validação threshold E2 vs SGB inundações"
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
    print("SGB — Validação Threshold E2 (flood_score vs SGB Inundações)")
    print("=" * 70)

    df           = load_data(args.sgb_ref, args.min_coverage)
    sweep, best  = analyse_primary(df)
    macro_fn     = analyse_false_negatives(df, best["threshold"])
    regional     = analyse_regional(df, best["threshold"])

    DIAGNOSE_DIR.mkdir(parents=True, exist_ok=True)
    write_diagnostic(df, sweep, best, macro_fn, regional,
                     args.sgb_ref, args.min_coverage, ts)
    print("\nConcluído.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERROMPIDO] Ctrl+C — diagnóstico não salvo.")
        sys.exit(1)
