import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

# Internal imports
from . import config as cfg
from . import utils
from . import calculations as calc

def validate_inputs(files_dict: dict) -> list:
    """Check all indicator files exist. Returns list of missing indicator keys."""
    missing = [
        key for key, path in files_dict.items()
        if key != "base_metadata" and not Path(path).is_file()
    ]
    total = len(files_dict) - 1  # exclude base_metadata
    found = total - len(missing)

    if missing:
        logging.warning(f"Input validation: {found}/{total} indicator files found.")
        for key in missing:
            logging.warning(f"  Missing [{key}]: {files_dict[key]}")
    else:
        logging.info(f"Input validation: all {total}/{total} indicator files found.")

    return missing


def _load_base_metadata(path: Path, join_key: str) -> pd.DataFrame:
    """Load the base H3 parquet, select metadata columns, deduplicate to one row per hexagon."""
    METADATA_COLS = ['cd_setor', 'cd_mun', 'nm_mun', 'cd_uf', 'nm_uf', 'sigla_uf', 'area_km2', 'peso_dom', 'qtd_dom']
    logging.info("Loading metadata base...")
    df = pd.read_parquet(path)
    existing_cols = [join_key] + [c for c in METADATA_COLS if c in df.columns]
    df = df[existing_cols]
    n_before = len(df)
    df = df.drop_duplicates(subset=[join_key])
    logging.info(f"Base deduplicated: {n_before:,} rows → {len(df):,} unique hexagons.")
    return df


def _merge_indicators(df_master: pd.DataFrame, files_dict: dict, join_key: str) -> pd.DataFrame:
    """Left-join each indicator parquet onto df_master, deduplicating before each merge."""
    for indicator_key, path in files_dict.items():
        if indicator_key == "base_metadata":
            continue

        if not Path(path).is_file():
            logging.warning(f"File not found for {indicator_key}: {path}")
            continue

        df_temp = pd.read_parquet(path)
        logging.info(f"Integrating {indicator_key} | Original shape: {df_temp.shape}")

        actual_column_name = cfg.COLUMN_MAP.get(indicator_key)

        if actual_column_name is None:
            logging.warning(f"Indicator '{indicator_key}' has no entry in COLUMN_MAP — skipped.")
            continue

        if actual_column_name in df_temp.columns:
            df_temp = df_temp[[join_key, actual_column_name]].rename(columns={actual_column_name: indicator_key})

            # Deduplicate by h3_id before merging to prevent row fan-out.
            # Some source files retain duplicate h3_ids from the (h3_id, cd_setor) base;
            # each join with such a file doubles those rows exponentially.
            n_before = len(df_temp)
            df_temp = df_temp.groupby(join_key, as_index=False)[indicator_key].mean()
            if len(df_temp) < n_before:
                logging.warning(
                    f"  [{indicator_key}] Deduplicated {n_before - len(df_temp):,} rows "
                    f"before merge ({n_before:,} → {len(df_temp):,} unique h3_ids)."
                )

            df_master = pd.merge(df_master, df_temp, on=join_key, how='left')
            logging.debug(f"Shape after merging {indicator_key}: {df_master.shape}")
        else:
            logging.warning(f"Column {actual_column_name} not found in {path.name}")

    return df_master


def consolidate_inputs(files_dict: dict, join_key: str) -> pd.DataFrame:
    """Load base H3 metadata and left-join all indicator files. Returns one row per hexagon."""
    path_base = files_dict.get("base_metadata")
    if not path_base or not path_base.is_file():
        logging.error("base_metadata file not found! Merge will fail.")
        return None
    df_master = _load_base_metadata(path_base, join_key)
    return _merge_indicators(df_master, files_dict, join_key)

def run_h3():
    """Run the full H3 pipeline: validate → consolidate → calculate → save outputs."""
    logging.info("=== STARTING PIPELINE: H3 GRID (SIMPLIFIED) ===")

    # 0. Validate inputs
    validate_inputs(cfg.FILES['h3'])

    # 1. Consolidate data
    df_data = consolidate_inputs(cfg.FILES['h3'], cfg.COL_ID_H3)
    
    if df_data is None or df_data.empty:
        logging.error("No data found for H3.")
        return

    # 2. Calculate the Index
    df_calculated = calc.calculate_simple_iic(df_data)

    # =========================================================================
    # 3. DIAGNOSTICS FOR LOGS
    # =========================================================================
    logging.info("--- FINAL FILE DIAGNOSTICS ---")
    
    # A) Logging the columns
    columns = df_calculated.columns.tolist()
    logging.info(f"Total columns generated: {len(columns)}")
    logging.info(f"List of columns: {columns}")
    
    # B) Per-column statistics (column-by-column to avoid large matrix allocation)
    for col in df_calculated.select_dtypes(include='number').columns:
        s = df_calculated[col].dropna()
        logging.info(f"  {col}: n={len(s):,}  mean={s.mean():.4f}  std={s.std():.4f}  min={s.min():.4f}  max={s.max():.4f}")
    # =========================================================================

    # 4. Save full results file (timestamped — never overwrites previous runs)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_output = cfg.FILES['output']['results_dir'] / f"{cfg.IIC_FILE_PREFIX}_{ts}.parquet"
    utils.save_parquet(df_calculated, path_output)
    logging.info(f"Full results saved: {path_output.name}")

    # 5. Save slim dashboard file (timestamped, in repo folder for GitHub commit)
    dashboard_cols = [cfg.COL_ID_H3] + [
        c for c in ['nm_mun', 'nm_uf', 'sigla_uf', 'cd_mun', 'iic_final', 'ip', 'iv', 'ie', 'ig']
        if c in df_calculated.columns
    ]
    df_slim = df_calculated[dashboard_cols].copy()

    # float32 (vs float64) halves numeric column size without affecting dashboard precision
    for col in ['iic_final', 'ip', 'iv', 'ie', 'ig']:
        if col in df_slim.columns:
            df_slim[col] = df_slim[col].astype('float32')

    # category dtype → parquet dictionary-encodes low-cardinality string columns
    for col in ['nm_uf', 'sigla_uf', 'nm_mun', 'cd_mun']:
        if col in df_slim.columns:
            df_slim[col] = df_slim[col].astype('category')

    path_dashboard = cfg.FILES['output']['repo_results_dir'] / f"{cfg.DASHBOARD_FILE_PREFIX}_{ts}.parquet"
    df_slim.to_parquet(path_dashboard, index=False, compression='gzip')
    logging.info(f"Dashboard parquet saved: {path_dashboard.name}  ({path_dashboard.stat().st_size / 1e6:.1f} MB)")

    # 6. Save per-dimension indicator files for hexagon-level indicator maps
    _save_dimension_parquets(df_calculated, ts)

    logging.info("Process completed successfully!")


def _save_dimension_parquets(df: pd.DataFrame, ts: str) -> None:
    """Save one parquet per dimension with h3_id + normalized indicator values.

    Files are split into chunks of at most MAX_COLS_PER_FILE indicators so that
    no single file exceeds GitHub's 100 MB limit (ig has 8 indicators).
    """
    MAX_COLS_PER_FILE = 6
    out_dir = cfg.FILES['output']['repo_results_dir']

    # Build {dimension: [indicator_keys_present_in_df]}
    dim_indicators: dict[str, list[str]] = {}
    for key, meta in cfg.INDICATORS.items():
        if key in df.columns:
            dim_indicators.setdefault(meta['dimension'], []).append(key)

    for dim, indicators in dim_indicators.items():
        abbr = cfg.DIMENSION_META[dim]['abbr'].lower()  # e.g. 'ip', 'iv', 'ie', 'ig'

        # Chunk into groups so each file stays under the GitHub limit
        chunks = [indicators[i:i + MAX_COLS_PER_FILE] for i in range(0, len(indicators), MAX_COLS_PER_FILE)]

        for chunk_idx, chunk_cols in enumerate(chunks, start=1):
            file_abbr = abbr if len(chunks) == 1 else f"{abbr}_{chunk_idx:02d}"
            prefix = cfg.DASHBOARD_DIM_FILE_PREFIX.format(dim_abbr=file_abbr)
            path = out_dir / f"{prefix}_{ts}.parquet"

            df_dim = df[[cfg.COL_ID_H3] + chunk_cols].copy()
            for col in chunk_cols:
                df_dim[col] = df_dim[col].astype('float32')

            df_dim.to_parquet(path, index=False, compression='gzip')
            size_mb = path.stat().st_size / 1e6
            logging.info(f"Dimension parquet [{file_abbr}] saved: {path.name}  ({size_mb:.1f} MB)")
            if size_mb > 95:
                logging.warning(
                    f"  [{file_abbr}] {size_mb:.1f} MB — close to GitHub's 100 MB limit. "
                    "Reduce MAX_COLS_PER_FILE or use Git LFS."
                )

def run():
    """Entry point called by run.py; wraps run_h3 with top-level error handling."""
    try:
        run_h3()
    except Exception as e:
        logging.error(f"Critical failure in H3 pipeline: {e}")

if __name__ == "__main__":
    run()