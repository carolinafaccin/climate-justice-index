"""
Convert the final Climate Injustice Index parquet to GeoPackage (.gpkg).

By default, picks the most recent final IIC parquet from the results directory
(files matching the IIC_FILE_PREFIX pattern, excluding dashboard variants).
Pass an explicit path as the first CLI argument to target a specific file.

Usage:
    python diagnose/parquet_to_gpkg.py
    python diagnose/parquet_to_gpkg.py /path/to/br_h3_iic_v2_0_<timestamp>.parquet

Output:
    <results_dir>/gpkg/<parquet_stem>.gpkg
"""

import logging
import multiprocessing
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import h3
from shapely.geometry import Polygon
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src import config as cfg

GPKG_DIR = cfg.RESULTS_DIR / "gpkg"

# Number of parallel workers for H3 → polygon conversion.
# Defaults to half the available CPUs to avoid overloading the machine.
N_WORKERS = max(1, multiprocessing.cpu_count() // 2)


# ==============================================================================
# LOGGING SETUP
# ==============================================================================

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ==============================================================================
# HELPERS
# ==============================================================================

def find_latest_parquet(results_dir: Path) -> Path:
    """Return the most recently modified final IIC parquet, excluding dashboard files."""
    prefix = cfg.IIC_FILE_PREFIX  # e.g. "br_h3_iic_v2_0"
    candidates = [
        p for p in results_dir.glob(f"{prefix}_*.parquet")
        if "dashboard" not in p.name
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No final IIC parquet found in {results_dir}.\n"
            f"Expected pattern: {prefix}_<timestamp>.parquet"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def h3_to_polygon(h3_id: str) -> Polygon:
    """Convert an H3 cell ID string to a Shapely Polygon (longitude, latitude order)."""
    boundary = h3.cell_to_boundary(h3_id)
    return Polygon([(lng, lat) for lat, lng in boundary])


def _convert_batch(h3_ids: list[str]) -> list[Polygon]:
    """Convert a list of H3 IDs to Shapely Polygons (used by worker processes)."""
    return [h3_to_polygon(hid) for hid in h3_ids]


def build_gdf(df: pd.DataFrame, n_workers: int = N_WORKERS) -> gpd.GeoDataFrame:
    """Add H3 hex polygons to a DataFrame and return a GeoDataFrame (EPSG:4326).

    Uses a multiprocessing pool to parallelize the H3 → polygon conversion,
    which is the bottleneck for large datasets (~4.5 M rows takes ~5 min
    single-threaded vs ~1–2 min with 4 workers).
    """
    h3_col = cfg.COL_ID_H3
    if h3_col not in df.columns:
        raise ValueError(f"Column '{h3_col}' not found in the parquet file.")

    n = len(df)
    logging.info(
        f"Converting {n:,} H3 IDs to polygons using {n_workers} worker(s) "
        "— this may take several minutes..."
    )

    h3_ids = df[h3_col].tolist()

    # Split into chunks, one per worker, then process with progress bar.
    chunk_size = max(1, n // (n_workers * 10))  # ~10 tasks per worker for smoother progress
    chunks = [h3_ids[i : i + chunk_size] for i in range(0, n, chunk_size)]

    geometries: list[Polygon] = []
    with multiprocessing.Pool(processes=n_workers) as pool:
        for result in tqdm(
            pool.imap(_convert_batch, chunks),
            total=len(chunks),
            desc="H3 → polygons",
            unit="chunk",
        ):
            geometries.extend(result)

    df = df.copy()
    df["geometry"] = geometries
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    logging.info(f"GeoDataFrame ready: {len(gdf):,} features | CRS: {gdf.crs}")
    return gdf


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    setup_logging()
    log = logging.getLogger(__name__)

    results_dir: Path = cfg.RESULTS_DIR
    log.info(f"data_dir  : {cfg.DATA_DIR}")
    log.info(f"results   : {results_dir}")
    log.info(f"gpkg out  : {GPKG_DIR}")

    # Resolve which parquet to convert
    if len(sys.argv) > 1:
        parquet_path = Path(sys.argv[1]).expanduser().resolve()
        if not parquet_path.is_file():
            log.error(f"File not found: {parquet_path}")
            sys.exit(1)
        log.info(f"Using specified file: {parquet_path.name}")
    else:
        log.info(f"Searching for the most recent final IIC parquet in:\n  {results_dir}")
        parquet_path = find_latest_parquet(results_dir)
        log.info(f"Selected: {parquet_path.name}")

    # Load parquet
    log.info(f"Loading: {parquet_path.name}")
    df = pd.read_parquet(parquet_path)
    log.info(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")
    log.info(f"  Column list: {df.columns.tolist()}")

    # Convert to GeoDataFrame
    gdf = build_gdf(df)

    # Save as GeoPackage
    GPKG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GPKG_DIR / f"{parquet_path.stem}.gpkg"

    log.info(f"Writing GeoPackage: {out_path.name}")
    gdf.to_file(out_path, driver="GPKG", layer="iic")

    size_mb = out_path.stat().st_size / 1e6
    log.info(f"Saved: {out_path}")
    log.info(f"File size: {size_mb:.1f} MB")
    log.info("Done.")


if __name__ == "__main__":
    main()
