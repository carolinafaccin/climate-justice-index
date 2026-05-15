"""
ETL: HAND + JRC Flood Hazard / GEE → Indicator e2 (flood susceptibility).

Metodologia (adaptada do MapBiomas Risco Climático, sem máscara urbana):
  Pixel-level score = HAND_class × JRC_mask, onde:
    • HAND 0–2 m → 1.00 (muito alta)
    • HAND 2–4 m → 0.66 (alta)
    • HAND 4–6 m → 0.33 (média)
    • HAND > 6 m → 0.00 (sem)
    • JRC > 0    → pixel mantido; senão zerado

Toda a lógica de classificação e máscara está no script GEE.
Este ETL apenas consolida os CSVs por UF, normaliza, e salva.

Input:  cfg.RAW_DIR/gee/.../h3_susc_inund_hand_jrc_v1_uf_*.csv
        Colunas: h3_id, cd_uf, cd_setor, flood_score
Output: cfg.FILES_H3["e2"] parquet
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(PROJECT_ROOT)
from src import config as cfg
from src import utils

# ==============================================================================
# 1. PATHS
# ==============================================================================
GEE_DIR = cfg.RAW_DIR / cfg.INDICATORS["e2"]["source"]["dir"]

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
        try:
            df = pd.read_csv(path, usecols=["h3_id", "flood_score"])
        except ValueError:
            df = pd.read_csv(path)
            df.columns = df.columns.str.lower()
            df = df[["h3_id", "flood_score"]]
        parts.append(df)
        print(f"   [+] {path.name}  ({len(df):,} hexagons)")

    df_all = pd.concat(parts, ignore_index=True)
    print(f"\n   Total: {len(df_all):,} hexagons loaded")

    # Deduplicate (border hexagons may appear in multiple UF exports)
    dupes = df_all["h3_id"].duplicated().sum()
    if dupes > 0:
        print(f"   WARNING: {dupes:,} duplicate h3_ids - keeping first value")
        df_all = df_all.drop_duplicates(subset="h3_id", keep="first")

    # ==============================================================================
    # 3. INDICATOR CALCULATION
    # ==============================================================================
    print("\n2/3 - Calculating indicator...")

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
    # 4. MERGE WITH H3 BASE AND SAVE
    # ==============================================================================
    print("\n3/3 - Merging with H3 base grid and saving...")
    df_h3 = pd.read_parquet(cfg.FILES_H3["base_metadata"], columns=["h3_id"])

    df_final = df_h3.merge(
        df_all[["h3_id", col_e2_abs, col_e2_norm]],
        on="h3_id", how="left"
    )

    print(f"   Hexagons with risk: {(df_final[col_e2_abs] > 0).sum():,}")
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
