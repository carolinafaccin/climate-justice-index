import pandas as pd
import numpy as np
import logging
import re
from pathlib import Path
from datetime import datetime
from . import config as cfg  # We need to import config to know where the logs folder is

# Function to configure logs (Remove any old logging.basicConfig from your project)
def setup_logging() -> None:
    # Generates a unique filename with the current date/time (e.g., pipeline_20260226_093000.log)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = cfg.LOGS_DIR / f"pipeline_{timestamp}.log"

    # Create a root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG) # The master logger captures everything

    # Clear old configurations in case you run it more than once in the same session
    if logger.hasHandlers():
        logger.handlers.clear()

    # 1. Text file configuration (Saves everything, even DEBUG level)
    file_format = logging.Formatter('%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d - %(message)s')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_format)

    # 2. Terminal configuration (Shows only INFO and above, to avoid cluttering the screen)
    console_format = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_format)

    # Add both handlers to the project
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info(f"Logs configured. Detailed log file at: {log_file}")

def get_next_version_path(path: Path) -> Path:
    """
    Checks if the file exists. If it does, increments the version (v1 -> v2 -> v3).
    Example: 'result.parquet' -> 'result_v1.parquet' -> 'result_v2.parquet'
    """
    path = Path(path)
    
    if not path.exists():
        if not re.search(r'_v\d+$', path.stem):
            return path.with_name(f"{path.stem}_v1{path.suffix}")
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    match = re.search(r'_v(\d+)$', stem)
    
    if match:
        current_version = int(match.group(1))
        base_name = stem[:match.start()]
        next_version = current_version + 1
    else:
        base_name = stem
        next_version = 1

    while True:
        new_path = parent / f"{base_name}_v{next_version}{suffix}"
        if not new_path.exists():
            return new_path
        next_version += 1

def normalize_minmax(series: pd.Series, winsorize: bool = False, limits: tuple = (0.01, 0.99)) -> pd.Series:
    """
    Normalizes a pandas series between 0 and 1 (Min-Max Scaling).
    """
    s = pd.to_numeric(series, errors='coerce')
    
    if winsorize:
        lower_bound = s.quantile(limits[0])
        upper_bound = s.quantile(limits[1])
        s = s.clip(lower=lower_bound, upper=upper_bound)
    
    min_val = s.min()
    max_val = s.max()
    
    if max_val == min_val:
        return pd.Series(0.0, index=s.index)
        
    return (s - min_val) / (max_val - min_val)

def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Saves the Parquet file, overwriting the previous version. Creates the destination folder if it doesn't exist."""
    path = Path(path)
    # Ensure destination folder exists (e.g. data/inputs/clean)
    path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Saving Parquet to: {path.name}...")
    df.to_parquet(path)
    logging.info("Saved successfully.")


def read_csv_columns(path: Path, cols: list[str]) -> pd.DataFrame:
    """Read CSV keeping only `cols`; falls back to full read + column lowercasing if usecols raises ValueError."""
    try:
        return pd.read_csv(path, usecols=cols)
    except ValueError:
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower()
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"{Path(path).name}: columns {missing} not found after lowercasing. "
                f"Available: {sorted(df.columns.tolist())}"
            )
        print(f"   NOTE: {Path(path).name} — usecols fallback triggered (column names required lowercasing)")
        return df[cols]


def h3_latlng_to_cell(lat: float, lon: float, res: int) -> str:
    """Convert lat/lon to H3 cell, compatible with h3-py v3 and v4."""
    import h3
    return h3.latlng_to_cell(lat, lon, res) if hasattr(h3, "latlng_to_cell") else h3.geo_to_h3(lat, lon, res)


def h3_grid_disk(cell: str, k: int) -> set:
    """Return grid disk (k-ring) of H3 cells, compatible with h3-py v3 and v4."""
    import h3
    return h3.grid_disk(cell, k) if hasattr(h3, "grid_disk") else h3.k_ring(cell, k)


def h3_cell_to_latlng(cell: str) -> tuple:
    """Return (lat, lng) centroid of an H3 cell, compatible with h3-py v3 and v4."""
    import h3
    return h3.cell_to_latlng(cell) if hasattr(h3, "cell_to_latlng") else h3.h3_to_geo(cell)


def h3_cell_to_boundary(cell: str) -> list:
    """Return boundary (lat, lng) coords of an H3 cell, compatible with h3-py v3 and v4."""
    import h3
    return h3.cell_to_boundary(cell) if hasattr(h3, "cell_to_boundary") else h3.h3_to_geo_boundary(cell)