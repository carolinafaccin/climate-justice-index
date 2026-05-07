import os
import json
from pathlib import Path

# ==============================================================================
# 1. LOCAL PATHS CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.local.json"

if CONFIG_PATH.exists():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config_local = json.load(f)
    DATA_DIR = Path(config_local["data_dir"])
else:
    DATA_DIR = BASE_DIR / "data"

# Main Folders
INPUTS_DIR = DATA_DIR / "inputs"
OUTPUTS_DIR = DATA_DIR / "outputs"

# Input Folders
CLEAN_DIR = INPUTS_DIR / "clean"
RAW_DIR = INPUTS_DIR / "raw"

# Output Folders
DIAGNOSE_DIR = OUTPUTS_DIR / "diagnose"
FIGURES_DIR = OUTPUTS_DIR / "figures"
RESULTS_DIR = OUTPUTS_DIR / "results"

for d in [OUTPUTS_DIR, RESULTS_DIR, FIGURES_DIR, DIAGNOSE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. GLOBAL PROJECT DEFINITIONS
# ==============================================================================
# Grid Resolution
H3_RES = 9
COL_ID_H3 = 'h3_id'

# Versioning
INDEX_VERSION = "v2.0"

# Formats the version for the filename (e.g., 'v1.0' becomes 'v1_0')
_formatted_version = INDEX_VERSION.replace('.', '_')

# Main file names
FILE_BASE_H3 = "br_h3_res9.parquet"
BASE_H3_DIR = RAW_DIR / "h3" / FILE_BASE_H3
FILE_FINAL_INDEX = f"br_h3_res9_iic_{_formatted_version}.parquet"

# ==============================================================================
# 3. METADATA LOADING (INDICATORS)
# ==============================================================================
INDICATORS_PATH = BASE_DIR / "config" / "indicators.json"

if INDICATORS_PATH.exists():
    with open(INDICATORS_PATH, 'r', encoding='utf-8') as f:
        _raw = json.load(f)
else:
    print(f"Warning: File {INDICATORS_PATH} not found.")
    _raw = {"file_prefix": "br_h3", "dimensions": {}}

# ==============================================================================
# 4. AUTOMATIC DICTIONARY GENERATION
# ==============================================================================
FILE_PREFIX = _raw.get("file_prefix", "br_h3")

# Flatten nested structure: {key: {dimension, name, abbr, ...}}
INDICATORS = {}
DIMENSIONS = {}
DIMENSION_META = {}
for dim, dim_data in _raw.get("dimensions", {}).items():
    DIMENSION_META[dim] = {
        "abbr":   dim_data.get("abbr", dim),
        "name":   dim_data.get("name", dim),
        "invert": dim_data.get("invert", False),
    }
    ind_map = dim_data.get("indicators", {})
    DIMENSIONS[dim] = list(ind_map.keys())
    for key, meta in ind_map.items():
        INDICATORS[key] = {"dimension": dim, **meta}

# col  = "{key}_{abbr}_norm"   e.g. "p1_mul_norm"
# file = "{prefix}_{key}_{name}.parquet"  e.g. "br_h3_p1_mulheres.parquet"
COLUMN_MAP = {k: f"{k}_{v['abbr']}_norm" for k, v in INDICATORS.items()}
FILES_H3   = {k: CLEAN_DIR / f"{FILE_PREFIX}_{k}_{v['name']}.parquet" for k, v in INDICATORS.items()}

FILES_H3["base_metadata"] = BASE_H3_DIR

# Dashboard parquet always lives in the repo root's data/ folder,
# regardless of config.local.json, so it can be committed to GitHub.
REPO_RESULTS_DIR = BASE_DIR / "data" / "outputs" / "results"
REPO_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Prefixes used by pipeline to generate timestamped filenames
IIC_FILE_PREFIX          = f"{FILE_PREFIX}_iic_{_formatted_version}"
DASHBOARD_FILE_PREFIX    = f"{FILE_PREFIX}_iic_{_formatted_version}_dashboard"
# Use .format(dim_abbr=...) to get the prefix for a specific dimension chunk,
# e.g. DASHBOARD_DIM_FILE_PREFIX.format(dim_abbr='ip') → 'br_h3_iic_v2_0_dashboard_dim_ip'
DASHBOARD_DIM_FILE_PREFIX = f"{FILE_PREFIX}_iic_{_formatted_version}_dashboard_dim_{{dim_abbr}}"

FILES = {
    "h3": FILES_H3,
    "output": {
        "results_dir":      RESULTS_DIR,
        "repo_results_dir": REPO_RESULTS_DIR,
    }
}