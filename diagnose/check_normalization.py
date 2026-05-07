"""
Checks whether winsorisation (P1-P99) would collapse the normalisation of any indicator.
Reads existing parquets and prints a distribution table for each indicator.

Indicators with an _abs column: checks P99 for winsorisation collapse risk.
Indicators without _abs (e.g. municipal binary variables): analyses the _norm column directly.
"""

import pandas as pd
import sys
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfg

print(f"\n{'Ind':<5} {'Column':<22} {'Total':>8} {'N>0':>8} {'%>0':>6} "
      f"{'P01':>8} {'P99':>8} {'Max':>8}  {'Status'}")
print("─" * 95)

for key, path in cfg.FILES_H3.items():
    if key == "base_metadata":
        continue
    if not path.exists():
        print(f"{key:<5} {'—':<22} {'file not found'}")
        continue

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"{key:<5} {'—':<22} ERROR: {e}")
        continue

    col_norm = cfg.COLUMN_MAP.get(key)
    if not col_norm:
        continue
    col_abs = col_norm.replace("_norm", "_abs")

    # --- Indicadores sem coluna _abs (ex: binários por município) ---
    if col_abs not in df.columns:
        if col_norm not in df.columns:
            print(f"{key:<5} {'—':<22} {'abs and norm columns missing'}")
            continue

        s = df[col_norm].dropna()
        total  = len(s)
        nz     = (s > 0).sum()
        pct_nz = nz / total * 100 if total else 0
        smax   = s.max()
        unique = set(s.unique())

        if unique <= {0, 1, 0.0, 1.0}:
            status = "✓ binário"
        else:
            status = "✓ sem _abs"

        print(f"{key:<5} {col_norm:<22} {total:>8,} {nz:>8,} {pct_nz:>5.1f}% "
              f"{'—':>8} {'—':>8} {smax:>8.4f}  {status}")
        continue

    # --- Indicadores com coluna _abs ---
    s = df[col_abs].dropna()
    total  = len(s)
    nz     = (s > 0).sum()
    pct_nz = nz / total * 100 if total else 0
    p01    = s.quantile(0.01)
    p99    = s.quantile(0.99)
    smax   = s.max()

    if p99 == 0 and pct_nz < 1.0:
        status = "~ SPARSE   — P99=0 but geographic indicator; use winsorize=False"
    elif p99 == 0:
        status = "⚠ COLLAPSE — P99=0, normalised column will be all zeros"
    elif pct_nz < 1.0:
        status = "⚠ RISK     — <1% non-zero, P99 may be 0 across all of Brazil"
    elif pct_nz < 5.0:
        status = "~ SPARSE   — check that P99>0 in the full dataset"
    else:
        status = "✓"

    print(f"{key:<5} {col_abs:<22} {total:>8,} {nz:>8,} {pct_nz:>5.1f}% "
          f"{p01:>8.4f} {p99:>8.4f} {smax:>8.4f}  {status}")

print()
