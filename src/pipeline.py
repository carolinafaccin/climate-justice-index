import pandas as pd
import logging
from pathlib import Path

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


def consolidate_inputs(files_dict: dict, join_key: str) -> pd.DataFrame:
    df_master = None
    # Columns we want to bring ONLY from the base_metadata file
    # Note: We keep these column names in Portuguese because they reflect the actual data columns
    METADATA_COLS = ['cd_setor', 'cd_mun', 'nm_mun', 'cd_uf', 'nm_uf', 'sigla_uf', 'area_km2', 'peso_dom', 'qtd_dom']

    # 1. First, we load the metadata base to ensure it acts as df_master
    path_base = files_dict.get("base_metadata")
    if path_base and path_base.is_file():
        logging.info("Loading metadata base...")
        df_master = pd.read_parquet(path_base)
        # Keep only join_key + existing metadata columns
        existing_cols = [join_key] + [c for c in METADATA_COLS if c in df_master.columns]
        df_master = df_master[existing_cols]
    else:
        logging.error("base_metadata file not found! Merge will fail.")
        return None

    # 2. Now we loop through the indicators (e1, v1, etc.)
    for indicator_key, path in files_dict.items():
        if indicator_key == "base_metadata": 
            continue # Skip since we already loaded it
        
        if not Path(path).is_file():
            logging.warning(f"File not found for {indicator_key}: {path}")
            continue
            
        # READ THE FILE FIRST
        df_temp = pd.read_parquet(path)
        # THEN LOG THE SHAPE
        logging.info(f"Integrating {indicator_key} | Original shape: {df_temp.shape}")
        
        # Gets the actual column name from config (e.g., 'g1' -> 'g1_inv_norm')
        actual_column_name = cfg.COLUMN_MAP.get(indicator_key)
        
        if actual_column_name in df_temp.columns:
            # Select only the ID and the data column, renaming it to the indicator_key (e1, v1...)
            df_temp = df_temp[[join_key, actual_column_name]].rename(columns={actual_column_name: indicator_key})
            
            # Merge left: keep all H3 from the base and bring the data if it exists
            df_master = pd.merge(df_master, df_temp, on=join_key, how='left')
            
            # Log: Show how the main file looks after merging the new column
            logging.debug(f"Shape after merging {indicator_key}: {df_master.shape}")
        else:
            logging.warning(f"Column {actual_column_name} not found in {path.name}")
            
    return df_master

def run_h3():
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
    
    # B) Logging statistics (only numeric columns to avoid breaking)
    # .to_string() is essential here so Pandas formats it as text in the log
    statistics = df_calculated.describe().to_string()
    logging.info(f"Statistical Summary of Final DataFrame:\n{statistics}")
    # =========================================================================

    # 4. Save the file
    path_output = cfg.FILES['output']['h3_final']
    utils.save_parquet(df_calculated, path_output)
    logging.info("Process completed successfully!")

def run():
    try:
        run_h3()
    except Exception as e:
        logging.error(f"Critical failure in H3 pipeline: {e}")

if __name__ == "__main__":
    run()