#!/usr/bin/env python3
"""
SGB — Análise de HAND nos Falsos Negativos do E2
=================================================
Consome os CSVs gerados pelo script GEE h3_e2_fn_hand_diagnostic_gee.js
(um por macrorregião) e responde:

  1. Qual a distribuição real de HAND nos hexágonos onde o SGB diz alta
     suscetibilidade a inundações mas o E2 atual não detecta?
  2. Quantos desses falsos negativos seriam recuperados se o teto HAND
     subisse para 8, 10, 12, 15 ou 20 m?
  3. Quantos FN têm jrc_rp100_mean = 0 (problema é a cobertura JRC, não
     o teto HAND — esses não seriam recuperados ampliando só o HAND)?

Inputs:
  Diretório com fn_hand_diag_macro_<S|SE|CO|NE|N>.csv baixados do GEE.
  Padrão: <data_dir>/inputs/raw/gee/fn_e2_hand_diagnostic/

Outputs em cfg.DIAGNOSE_DIR:
  diagnostic_07_fn_hand_<ts>.txt              — relatório legível
  diagnostic_07_fn_hand_candidates_<ts>.csv   — tabela de tetos candidatos
                                                  por macro × classe SGB
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from src import config as cfg


def _load_data_dir() -> Path:
    config_path = PROJECT_ROOT / "config" / "config.local.json"
    with open(config_path, encoding="utf-8") as f:
        return Path(json.load(f)["data_dir"])


_DATA_DIR    = _load_data_dir()
DEFAULT_DIR  = _DATA_DIR / "inputs/raw/gee/e2_fn_hand_diagnostic"
DIAGNOSE_DIR = cfg.DIAGNOSE_DIR

# Tetos candidatos a testar (em metros). O atual é 6 m.
CANDIDATE_CEILINGS = [8, 10, 12, 15, 20]

# Classes SGB: 4=Alta, 5=Muito Alta (são as que entram em sgb_alta_mta_frac)
SGB_CLASS_LABELS = {4: "Alta", 5: "Muito Alta"}

MACROS = ["N", "NE", "CO", "SE", "S"]


# ══════════════════════════════════════════════════════════════════════════════
# CARREGAMENTO
# ══════════════════════════════════════════════════════════════════════════════

def load_fn_hand(input_dir: Path) -> pd.DataFrame:
    csvs = sorted(input_dir.glob("fn_hand_diag_macro_*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {input_dir}\n"
            "Baixe primeiro os CSVs do GEE (folder GEE_fn_e2_hand_diagnostic)."
        )

    parts = []
    for p in csvs:
        df = pd.read_csv(p)
        parts.append(df)
        print(f"   [+] {p.name}  ({len(df):,} hexágonos)")

    df = pd.concat(parts, ignore_index=True)

    # Tipagem
    numeric_cols = ["sgb_max_class", "sgb_alta_mta_frac", "sgb_coverage_frac",
                    "e2_inu_abs", "hand_mean", "hand_min", "hand_max",
                    "hand_p25", "hand_p50", "hand_p75", "hand_p90", "jrc_rp100_mean"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Remove hexágonos sem estatísticas válidas (HAND mascarado integralmente)
    n_before = len(df)
    df = df.dropna(subset=["hand_mean"])
    n_after  = len(df)
    if n_before > n_after:
        print(f"   - {n_before - n_after:,} hexágonos sem HAND válido removidos")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISES
# ══════════════════════════════════════════════════════════════════════════════

def describe_hand_by_macro(df: pd.DataFrame) -> pd.DataFrame:
    """Distribuição de HAND (mean e p50) por macrorregião."""
    print("\n2/4 — Distribuição de HAND nos FN por macrorregião")
    rows = []
    for macro in MACROS:
        sub = df[df["macro"] == macro]
        if sub.empty:
            continue
        rows.append({
            "macro":          macro,
            "n_fn":           len(sub),
            "hand_mean_p50":  sub["hand_mean"].median(),
            "hand_mean_p75":  sub["hand_mean"].quantile(0.75),
            "hand_mean_p90":  sub["hand_mean"].quantile(0.90),
            "hand_mean_max":  sub["hand_mean"].max(),
            "pct_jrc_zero":   (sub["jrc_rp100_mean"] == 0).mean(),
        })
        r = rows[-1]
        print(f"   {macro:2}  n={r['n_fn']:>5,}  "
              f"HAND p50={r['hand_mean_p50']:>5.1f}m  "
              f"p75={r['hand_mean_p75']:>5.1f}m  "
              f"p90={r['hand_mean_p90']:>5.1f}m  "
              f"max={r['hand_mean_max']:>5.1f}m  "
              f"JRC=0: {r['pct_jrc_zero']:.0%}")
    return pd.DataFrame(rows)


def describe_by_class(df: pd.DataFrame) -> pd.DataFrame:
    """HAND por classe SGB máxima (4=Alta, 5=Muito Alta)."""
    print("\n3/4 — Distribuição de HAND por classe SGB máxima")
    rows = []
    for cls, label in SGB_CLASS_LABELS.items():
        sub = df[df["sgb_max_class"] == cls]
        if sub.empty:
            continue
        rows.append({
            "sgb_max_class": cls,
            "label":         label,
            "n":             len(sub),
            "hand_p50":      sub["hand_mean"].median(),
            "hand_p75":      sub["hand_mean"].quantile(0.75),
            "hand_p90":      sub["hand_mean"].quantile(0.90),
        })
        r = rows[-1]
        print(f"   classe {cls} ({label:10}): n={r['n']:>5,}  "
              f"HAND p50={r['hand_p50']:>5.1f}m  "
              f"p75={r['hand_p75']:>5.1f}m  "
              f"p90={r['hand_p90']:>5.1f}m")
    return pd.DataFrame(rows)


def candidate_ceilings_table(df: pd.DataFrame, ceilings: list[int]) -> pd.DataFrame:
    """
    Para cada teto candidato, calcula quantos FN seriam recuperados
    (hand_mean <= teto E jrc_rp100_mean > 0), por macro e por classe SGB.
    """
    print(f"\n4/4 — Recuperação de FN por teto HAND candidato (atual: 6 m)")

    rows = []
    for macro in MACROS:
        sub = df[df["macro"] == macro]
        if sub.empty:
            continue
        n_total = len(sub)
        for ceiling in ceilings:
            recovered_jrc = ((sub["hand_mean"] <= ceiling) &
                             (sub["jrc_rp100_mean"] > 0)).sum()
            recovered_nojrc = (sub["hand_mean"] <= ceiling).sum()
            rows.append({
                "macro":              macro,
                "teto_m":             ceiling,
                "n_fn_total":         n_total,
                "n_recuperados_jrc":  int(recovered_jrc),
                "pct_recuperados_jrc": recovered_jrc / n_total,
                "n_recuperados_sem_jrc": int(recovered_nojrc),
                "pct_recuperados_sem_jrc": recovered_nojrc / n_total,
            })

    out = pd.DataFrame(rows)

    # Print conciso: matriz macro × teto (% recuperados COM JRC)
    print("\n   % de FN recuperados (hand_mean ≤ teto E jrc > 0):")
    pivot = out.pivot(index="macro", columns="teto_m",
                      values="pct_recuperados_jrc").reindex(MACROS).dropna(how="all")
    print(pivot.applymap(lambda v: f"{v:>5.0%}" if pd.notna(v) else "    -").to_string())

    print("\n   % de FN recuperados (só hand_mean ≤ teto, ignorando JRC):")
    pivot2 = out.pivot(index="macro", columns="teto_m",
                       values="pct_recuperados_sem_jrc").reindex(MACROS).dropna(how="all")
    print(pivot2.applymap(lambda v: f"{v:>5.0%}" if pd.notna(v) else "    -").to_string())

    return out


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNÓSTICO
# ══════════════════════════════════════════════════════════════════════════════

def write_diagnostic(
    df: pd.DataFrame,
    macro_summary: pd.DataFrame,
    class_summary: pd.DataFrame,
    candidates: pd.DataFrame,
    ts: str,
) -> None:
    txt_path = DIAGNOSE_DIR / f"diagnostic_07_fn_hand_{ts}.txt"
    csv_path = DIAGNOSE_DIR / f"diagnostic_07_fn_hand_candidates_{ts}.csv"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("SGB — Diagnóstico de HAND nos Falsos Negativos do E2\n")
        f.write(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("--- Sumário ---\n")
        f.write(f"  Total de FN analisados : {len(df):,}\n")
        f.write(f"  Teto HAND atual        : 6 m\n")
        f.write(f"  Tetos testados         : {CANDIDATE_CEILINGS} m\n\n")

        f.write("--- HAND nos FN por macrorregião ---\n")
        f.write("  (p50/p75/p90 do hand_mean; pct_jrc_zero = FN fora da máscara JRC)\n")
        f.write(macro_summary.to_string(index=False, float_format="%.2f"))
        f.write("\n\n")

        f.write("--- HAND por classe SGB máxima ---\n")
        f.write(class_summary.to_string(index=False, float_format="%.2f"))
        f.write("\n\n")

        f.write("--- Tetos candidatos: recuperação por macrorregião ---\n")
        f.write("  n_recuperados_jrc    : FN com hand_mean ≤ teto E jrc_rp100 > 0\n")
        f.write("  n_recuperados_sem_jrc: FN com hand_mean ≤ teto (independente do JRC)\n\n")
        f.write(candidates.to_string(index=False, float_format="%.2f"))
        f.write("\n\n")

        # Insights automáticos
        f.write("--- Leitura ---\n")
        jrc_problems = macro_summary[macro_summary["pct_jrc_zero"] > 0.5]
        if not jrc_problems.empty:
            regs = ", ".join(jrc_problems["macro"].tolist())
            f.write(f"  ⚑ Em {regs}, >50% dos FN têm jrc_rp100=0 → o gargalo NÃO é\n"
                    f"    o teto HAND, é a cobertura JRC. Ampliar HAND não recupera\n"
                    f"    esses FN. Considerar fonte alternativa de hazard (ex: HAND\n"
                    f"    puro sem máscara JRC para regiões com baixa cobertura JRC).\n\n")

        if not candidates.empty:
            best_per_macro = candidates.loc[
                candidates.groupby("macro")["pct_recuperados_jrc"].idxmax()
            ][["macro", "teto_m", "pct_recuperados_jrc"]]
            f.write("  Teto que maximiza recuperação (com JRC) por macro:\n")
            for _, row in best_per_macro.iterrows():
                f.write(f"    {row['macro']:2}: teto={row['teto_m']:>2}m → "
                        f"{row['pct_recuperados_jrc']:.0%} dos FN recuperados\n")
            f.write("\n  Decisão sugerida: escolher o menor teto cuja recuperação se\n"
                    "  aproxima do máximo (evita inflar o score com HAND alto demais).\n")
            f.write("  Esse teto vira o novo limiar no script GEE v2 de E2 e/ou nova\n"
                    "  classe acima do atual (e.g., 6-10m → 0.20).\n")

    candidates.to_csv(csv_path, index=False, float_format="%.4f", encoding="utf-8")

    print(f"\n   ✓ Diagnóstico TXT: {txt_path.name}")
    print(f"   ✓ Tetos CSV       : {csv_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analisa HAND nos falsos negativos do E2 e propõe novo teto"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_DIR,
        help=f"Diretório com fn_hand_diag_macro_*.csv (padrão: {DEFAULT_DIR})"
    )
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print("SGB — Análise de HAND nos Falsos Negativos do E2")
    print("=" * 70)
    print(f"\n1/4 — Carregando CSVs de {args.input_dir}")

    df             = load_fn_hand(args.input_dir)
    macro_summary  = describe_hand_by_macro(df)
    class_summary  = describe_by_class(df)
    candidates     = candidate_ceilings_table(df, CANDIDATE_CEILINGS)

    DIAGNOSE_DIR.mkdir(parents=True, exist_ok=True)
    write_diagnostic(df, macro_summary, class_summary, candidates, ts)
    print("\nConcluído.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERROMPIDO] Ctrl+C — diagnóstico não salvo.")
        sys.exit(1)
