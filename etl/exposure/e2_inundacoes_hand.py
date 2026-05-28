"""
ETL: HAND + JRC Flood Hazard / GEE → Indicator e2 (flood susceptibility).

Metodologia (MapBiomas Risco Climático, sem máscara urbana + overlay SGB):

  Camada base (cobertura nacional, via GEE):
    Pixel-level score = HAND_class × JRC_mask
    • HAND 0–2 m → 1.00 (muito alta)
    • HAND 2–4 m → 0.66 (alta)
    • HAND 4–6 m → 0.33 (média)
    • HAND > 6 m → 0.00 (sem)
    • JRC > 0    → pixel mantido; senão zerado

  Overlay SGB (~600 municípios com cartografia disponível):
    Se sgb_alta_mta_frac > 0.3 AND sgb_coverage_frac >= 0.5:
        flood_score = 1.00  (SGB autoridade local — score máximo)
        sgb_override = True

Toda a lógica da camada base está no script GEE.
Este ETL consolida os CSVs por UF, aplica o overlay SGB e salva.

Input:  cfg.RAW_DIR/gee/.../h3_susc_inund_hand_jrc_v1_uf_*.csv
        cfg.CLEAN_DIR/br_h3_sgb_inundacoes.parquet (output do 04_sgb_h3_intersect.py)
Output: cfg.FILES_H3["e2"] parquet  (inclui coluna sgb_override)

Referência: ADR-0039
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pipeline.py").exists())
sys.path.insert(0, str(_ROOT))
from src import config as cfg
from src import utils

# ==============================================================================
# 1. PATHS
# ==============================================================================
_e2_src             = cfg.INDICATORS["e2"]["source"]
GEE_DIR             = cfg.RAW_DIR / _e2_src["dir"]
SGB_ALTA_FRAC_MIN   = _e2_src["sgb_alta_frac_min"]
SGB_COVERAGE_MIN    = _e2_src["sgb_coverage_min"]
SGB_OVERRIDE_SCORE  = _e2_src["sgb_override_score"]

now = datetime.now().strftime("%Y%m%d_%H%M%S")
DIAGNOSTIC_TXT = cfg.DIAGNOSE_DIR / f"diagnostic_h3_e2_inundacoes_hand_{now}.txt"

col_e2_norm = cfg.COLUMN_MAP["e2"]
col_e2_abs  = col_e2_norm.replace("_norm", "_abs")


# ==============================================================================
# 2. MAIN
# ==============================================================================
def main():
    print("=" * 60)
    print("ETL: HAND + JRC Flood Hazard — E2 (Flood Susceptibility)")
    print(f"Source: {GEE_DIR}")
    print("=" * 60)

    # Load all UF CSVs
    print("\n1/3 - Loading GEE CSVs...")
    csv_files = sorted(GEE_DIR.glob("h3_susc_inund_hand_jrc_v1_uf_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in: {GEE_DIR}")
    print(f"   Files found: {len(csv_files)}")

    parts = []
    for path in csv_files:
        df = utils.read_csv_columns(path, ["h3_id", "flood_score"])
        parts.append(df)
        print(f"   [+] {path.name}  ({len(df):,} hexagons)")

    df_all = pd.concat(parts, ignore_index=True)
    print(f"\n   Total: {len(df_all):,} hexagons loaded")

    # Border hexagons may appear in multiple UF-level GEE exports — deduplicate.
    dupes = df_all["h3_id"].duplicated().sum()
    if dupes > 0:
        print(f"   NOTE: {dupes:,} border hexagons in multiple UF exports ({100*dupes/len(df_all):.1f}%) — keeping first")
        df_all = df_all.drop_duplicates(subset="h3_id", keep="first")

    # ==============================================================================
    # 3. SGB OVERLAY
    # ==============================================================================
    print("\n2/4 - Applying SGB overlay...")
    SGB_H3_PATH = cfg.CLEAN_DIR / "br_h3_sgb_inundacoes.parquet"
    if SGB_H3_PATH.exists():
        sgb = pd.read_parquet(SGB_H3_PATH, columns=["h3_id", "sgb_alta_mta_frac", "sgb_coverage_frac"])
        df_all = df_all.merge(sgb, on="h3_id", how="left")
        sgb_mask = (df_all["sgb_alta_mta_frac"] > SGB_ALTA_FRAC_MIN) & (df_all["sgb_coverage_frac"] >= SGB_COVERAGE_MIN)
        df_all.loc[sgb_mask, "flood_score"] = SGB_OVERRIDE_SCORE
        df_all["sgb_override"] = sgb_mask.fillna(False)
        df_all = df_all.drop(columns=["sgb_alta_mta_frac", "sgb_coverage_frac"])
        n_override = int(sgb_mask.sum())
        print(f"   SGB override applied: {n_override:,} hexagons set to flood_score={SGB_OVERRIDE_SCORE}")
        print(f"   (sgb_alta_mta_frac > {SGB_ALTA_FRAC_MIN} AND sgb_coverage_frac >= {SGB_COVERAGE_MIN})")
    else:
        df_all["sgb_override"] = False
        print(f"   WARNING: SGB parquet not found at {SGB_H3_PATH} — overlay skipped")

    # ==============================================================================
    # 4. INDICATOR CALCULATION
    # ==============================================================================
    print("\n3/4 - Calculating indicator...")

    df_all["flood_score"] = pd.to_numeric(df_all["flood_score"], errors="coerce").fillna(0)

    # e2_abs = score contínuo direto do GEE (0-1)
    # Score já reflete: HAND class × JRC mask, mean por hexágono
    df_all[col_e2_abs] = df_all["flood_score"]

    # Normalização min-max (mantém [0,1], só estica se max < 1)
    # winsorize=False: indicador concentrado geograficamente (planícies de inundação)
    df_all[col_e2_norm] = utils.normalize_minmax(df_all[col_e2_abs], winsorize=False)

    n_risk = (df_all[col_e2_abs] > 0).sum()
    print(f"   Hexagons with flood risk > 0: {n_risk:,}")
    print(f"   Mean e2_abs: {df_all[col_e2_abs].mean():.4f}")
    print(f"   Max e2_abs:  {df_all[col_e2_abs].max():.4f}")

    # ==============================================================================
    # 5. MERGE WITH H3 BASE AND SAVE
    # ==============================================================================
    print("\n4/4 - Merging with H3 base grid and saving...")
    df_h3 = pd.read_parquet(cfg.FILES_H3["base_metadata"], columns=["h3_id"])

    df_final = df_h3.merge(
        df_all[["h3_id", col_e2_abs, col_e2_norm, "sgb_override"]],
        on="h3_id", how="left"
    )
    df_final["sgb_override"] = df_final["sgb_override"].fillna(False)

    print(f"   Hexagons with risk: {(df_final[col_e2_abs] > 0).sum():,}")
    print(f"   Hexagons with sgb_override: {df_final['sgb_override'].sum():,}")
    print(f"   Hexagons without e2 data (NaN): {df_final[col_e2_abs].isna().sum():,}")

    utils.save_parquet(df_final, cfg.FILES_H3["e2"])
    print(f"   [OK] Saved: {cfg.FILES_H3['e2'].name}")

    _write_diagnostic(df_all, df_final, csv_files)
    print(f"\nDiagnostic: {DIAGNOSTIC_TXT}")
    print("Done!")


def _write_diagnostic(df_all, df_final, csv_files):
    with open(DIAGNOSTIC_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("HAND + JRC — E2 Flood Susceptibility ETL Diagnostic\n")
        f.write(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"GEE directory : {GEE_DIR}\n")
        f.write(f"Files read    : {len(csv_files)}\n")
        f.write(f"Methodology   : HAND classificado x JRC Flood Hazard mask\n")
        f.write(f"  HAND 0-2 m  -> score 1.00 (muito alta)\n")
        f.write(f"  HAND 2-4 m  -> score 0.66 (alta)\n")
        f.write(f"  HAND 4-6 m  -> score 0.33 (media)\n")
        f.write(f"  HAND > 6 m  -> score 0.00\n")
        f.write(f"  JRC > 0     -> pixel mantido (return period 100 anos)\n")
        f.write(f"Sources:\n")
        f.write(f"  HAND : projects/sat-io/open-datasets/HAND/MERIT_HAND (MERIT DEM ~90m)\n")
        f.write(f"  JRC  : JRC/CEMS_GLOFAS/FloodHazard/V1\n\n")

        f.write("--- flood_score (raw from GEE, mean per hexagon) ---\n")
        s_raw = df_all["flood_score"].dropna()
        f.write(f"  mean   = {s_raw.mean():.6f}\n")
        f.write(f"  median = {s_raw.median():.6f}\n")
        f.write(f"  min    = {s_raw.min():.6f}\n")
        f.write(f"  max    = {s_raw.max():.6f}\n")
        f.write(f"  = 0    = {(s_raw == 0).sum():,} hexagons\n")
        f.write(f"  > 0    = {(s_raw > 0).sum():,} hexagons\n\n")

        for col in [col_e2_abs, col_e2_norm]:
            s = df_final[col]
            f.write(f"--- {col} (full grid) ---\n")
            f.write(f"  mean   = {s.mean():.6f}\n")
            f.write(f"  median = {s.median():.6f}\n")
            f.write(f"  min    = {s.min():.6f}\n")
            f.write(f"  max    = {s.max():.6f}\n")
            f.write(f"  zeros  = {(s == 0).sum():,}\n")
            f.write(f"  nulls  = {s.isna().sum():,}\n\n")


if __name__ == "__main__":
    main()
