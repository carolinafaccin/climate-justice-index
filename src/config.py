import os
import json
from pathlib import Path
from datetime import datetime

# ==============================================================================
# 1. LOCAL PATHS CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.local.json"

if CONFIG_PATH.exists():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config_local = json.load(f)
    DATA_DIR = Path(config_local["data_dir"])
    RAW_DIR  = Path(config_local["raw_dir"]) if "raw_dir" in config_local else DATA_DIR / "inputs" / "raw"
else:
    DATA_DIR = BASE_DIR / "data"
    RAW_DIR  = DATA_DIR / "inputs" / "raw"

# Main Folders
INPUTS_DIR = DATA_DIR / "inputs"
OUTPUTS_DIR = DATA_DIR / "outputs"

# Input Folders
CLEAN_DIR = INPUTS_DIR / "clean"

# Output Folders
DIAGNOSE_DIR = OUTPUTS_DIR / "diagnose"
FIGURES_DIR  = OUTPUTS_DIR / "figures"
RESULTS_DIR  = OUTPUTS_DIR / "results"

RESULTS_COMPLETE_DIR  = RESULTS_DIR / "complete"
RESULTS_DASHBOARD_DIR = RESULTS_DIR / "dashboard"
RESULTS_GPKG_DIR      = RESULTS_DIR / "complete_gpkg"

LOGS_DIR = BASE_DIR / "logs"

# IBGE geographic mesh — single vintage constant used by all scripts
IBGE_MALHA_VINTAGE  = "2024"
MALHA_MUNICIPAL_DIR = RAW_DIR / "ibge" / "malha_municipal" / IBGE_MALHA_VINTAGE


def ensure_output_dirs() -> None:
    """Create project output directories. Called at module load; also callable explicitly."""
    for d in [OUTPUTS_DIR, RESULTS_DIR, RESULTS_COMPLETE_DIR, RESULTS_DASHBOARD_DIR,
              RESULTS_GPKG_DIR, FIGURES_DIR, DIAGNOSE_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


ensure_output_dirs()


def diagnostic_path(name: str) -> Path:
    """Return a timestamped diagnostic txt path: DIAGNOSE_DIR/diagnostic_{name}_{ts}.txt"""
    ts = datetime.now().strftime(TS_FORMAT_FILE)
    return DIAGNOSE_DIR / f"diagnostic_{name}_{ts}.txt"


# ==============================================================================
# 2. GLOBAL PROJECT DEFINITIONS
# ==============================================================================
# Grid Resolution
H3_RES = 9
COL_ID_H3 = 'h3_id'

# Brazilian standard CRS (used across ETL scripts)
CRS_LATLON = "EPSG:4674"   # SIRGAS 2000 — geographic (degrees)
CRS_METRIC = "EPSG:5880"   # Brazil Polyconic — metric projection for area/distance calculations
CRS_WGS84  = "EPSG:4326"   # WGS84 — native CRS of H3 and most third-party geodata

# Timestamp formats
TS_FORMAT_FILE = "%Y%m%d_%H%M%S"    # compact, safe for filenames
TS_FORMAT_LOG  = "%Y-%m-%d %H:%M:%S"  # human-readable, for log/diagnostic text

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
# 4. SCHEMA VALIDATION + AUTOMATIC DICTIONARY GENERATION
# ==============================================================================
def _validate_indicators(raw: dict) -> None:
    """Raise ValueError immediately if any indicator is missing a required field."""
    for dim, dim_data in raw.get("dimensions", {}).items():
        for key, meta in dim_data.get("indicators", {}).items():
            for field in ("name", "abbr", "display_name", "source"):
                if field not in meta:
                    raise ValueError(
                        f"indicators.json: [{dim}][{key}] is missing required field '{field}'"
                    )

_validate_indicators(_raw)

FILE_PREFIX        = _raw.get("file_prefix", "br_h3")
SALARIO_MINIMO_REF = _raw.get("salario_minimo_ref", 1212)

# Geographic reference data (region/porte ordering for analysis scripts)
_GEO_PATH = BASE_DIR / "config" / "geo_config.json"
if _GEO_PATH.exists():
    with open(_GEO_PATH, 'r', encoding='utf-8') as _gf:
        _geo = json.load(_gf)
    REGIOES_ORDER = _geo.get("regioes_order", ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"])
    PORTES_ORDER  = _geo.get("portes_order",  [
        "Até 5.000 hab.", "De 5.001 a 10.000 hab.", "De 10.001 a 20.000 hab.",
        "De 20.001 a 50.000 hab.", "De 50.001 a 100.000 hab.",
        "De 100.001 a 500.000 hab.", "Mais de 500.000 hab.",
    ])
else:
    REGIOES_ORDER = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
    PORTES_ORDER  = [
        "Até 5.000 hab.", "De 5.001 a 10.000 hab.", "De 10.001 a 20.000 hab.",
        "De 20.001 a 50.000 hab.", "De 50.001 a 100.000 hab.",
        "De 100.001 a 500.000 hab.", "Mais de 500.000 hab.",
    ]

# SGB pipeline parameters (shared across sgb/ scripts)
_SGB_CFG_PATH = BASE_DIR / "config" / "sgb_config.json"
if _SGB_CFG_PATH.exists():
    with open(_SGB_CFG_PATH, 'r', encoding='utf-8') as _sf:
        SGB_CFG = json.load(_sf)
else:
    SGB_CFG = {
        "simplify_geom_m":         5.0,
        "simplify_h3_intersect_m": 20.0,
        "risk_class_min":          4,
        "sgb_ref_threshold":       0.3,
        "sgb_coverage_min":        0.5,
        "sweep_step_coarse":       0.05,
        "sweep_step_fine":         0.01,
    }

# Cities reference list (used by report and visualization scripts)
_CITIES_PATH = BASE_DIR / "config" / "cities.json"
if _CITIES_PATH.exists():
    with open(_CITIES_PATH, 'r', encoding='utf-8') as _cf:
        CITIES_DATA: list[dict] = json.load(_cf).get("cities", [])
else:
    CITIES_DATA = []

# Flatten nested structure: {key: {dimension, name, abbr, ...}}
INDICATORS = {}
DIMENSIONS = {}
DIMENSION_META = {}
for dim, dim_data in _raw.get("dimensions", {}).items():
    DIMENSION_META[dim] = {
        "abbr":         dim_data.get("abbr", dim),
        "name":         dim_data.get("name", dim),
        "display_name": dim_data.get("display_name", dim_data.get("abbr", dim)),
        "color":        dim_data.get("color", "#999999"),
        "invert":       dim_data.get("invert", False),
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

# Prefixes used by calculation to generate timestamped filenames
IIC_FILE_PREFIX           = f"{FILE_PREFIX}_iic_{_formatted_version}"
DASHBOARD_FILE_PREFIX     = f"{FILE_PREFIX}_iic_{_formatted_version}_dashboard"
# Use .format(dim_abbr=...) to get the prefix for a specific dimension chunk,
# e.g. DASHBOARD_DIM_FILE_PREFIX.format(dim_abbr='ip') → 'br_h3_iic_v2_0_dashboard_dim_ip'
DASHBOARD_DIM_FILE_PREFIX = f"{FILE_PREFIX}_iic_{_formatted_version}_dashboard_dim_{{dim_abbr}}"

FILES = {
    "h3": FILES_H3,
    "output": {
        "results_dir":   RESULTS_COMPLETE_DIR,
        "dashboard_dir": RESULTS_DASHBOARD_DIR,
        "gpkg_dir":      RESULTS_GPKG_DIR,
    }
}