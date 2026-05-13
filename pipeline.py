"""
Full pipeline orchestrator for the Climate Injustice Index.

Runs all stages in order, stopping on the first failure.

Usage:
    python pipeline.py                  # run all stages
    python pipeline.py --from calc      # start from a specific stage
    python pipeline.py --only export    # run a single stage
    python pipeline.py --skip-tests     # skip the test stage

Stages (in order):
    test        Unit tests                       (pytest tests/)
    calc        Calculate the index              (run_index.py)
    cluster     Cluster analysis                 (explore/analysis/cluster_municipios.py)
    multicol    Multicollinearity check          (explore/checks/check_multicollinearity.py)
    norm        Normalization check              (explore/checks/check_normalization.py)
    export      Export parquet → GeoPackage      (explore/export/parquet_to_gpkg.py)
    scatter     Scatter plots                    (explore/plots/plot_scatter.py)
    report      Generate HTML report             (report/generate_report.py)
"""

import subprocess
import sys
import argparse
from pathlib import Path

VENV_PYTHON = Path(__file__).parent / "venv" / "Scripts" / "python.exe"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

# Each stage is (name, command).
# command is either a script path (str) or a list of args passed directly to subprocess.
STAGES = [
    ("test",     [PYTHON, "-m", "pytest", "tests/", "-v"]),
    ("calc",     "run_index.py"),
    ("cluster",  "explore/analysis/cluster_municipios.py"),
    ("multicol", "explore/checks/check_multicollinearity.py"),
    ("norm",     "explore/checks/check_normalization.py"),
    ("export",   "explore/export/parquet_to_gpkg.py"),
    ("scatter",  "explore/plots/plot_scatter.py"),
    ("report",   "report/generate_report.py"),
]

STAGE_NAMES = [name for name, _ in STAGES]


def run_stage(name: str, command) -> bool:
    print(f"\n{'='*60}")
    print(f"  STAGE: {name}")
    print(f"{'='*60}")
    cmd = command if isinstance(command, list) else [PYTHON, command]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] Stage '{name}' failed (exit code {result.returncode}). Stopping.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Climate Injustice Index — full pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--from", dest="from_stage", metavar="STAGE",
                       choices=STAGE_NAMES, help="Start from this stage (inclusive)")
    group.add_argument("--only", dest="only_stage", metavar="STAGE",
                       choices=STAGE_NAMES, help="Run only this stage")
    parser.add_argument("--skip-tests", action="store_true",
                        help="Skip the test stage")
    args = parser.parse_args()

    if args.only_stage:
        stages_to_run = [(n, s) for n, s in STAGES if n == args.only_stage]
    elif args.from_stage:
        start = STAGE_NAMES.index(args.from_stage)
        stages_to_run = STAGES[start:]
    else:
        stages_to_run = STAGES

    if args.skip_tests:
        stages_to_run = [(n, s) for n, s in stages_to_run if n != "test"]

    print(f"Stages to run: {[n for n, _ in stages_to_run]}")

    for name, command in stages_to_run:
        if not run_stage(name, command):
            sys.exit(1)

    print(f"\n{'='*60}")
    print("  ALL STAGES COMPLETED SUCCESSFULLY")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
