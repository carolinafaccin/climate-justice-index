"""Shared utilities for all scripts in diagnose/.

Centralises three patterns that were copy-pasted across check_highlights,
plot_maps, and plot_scatter:
  - ALL_INDICATOR_KEYS  — ordered list of indicator keys derived from cfg
  - DIMS                — per-dimension display config (colors, Portuguese labels)
  - build_gdf()         — converts an h3_id column to a GeoDataFrame
"""

import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import h3
from pathlib import Path
from shapely.geometry import Polygon

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src import config as cfg

# ---------------------------------------------------------------------------
# Indicator key list — single source of truth, ordered by dimension
# ---------------------------------------------------------------------------
ALL_INDICATOR_KEYS: list[str] = [k for dim in cfg.DIMENSIONS for k in cfg.DIMENSIONS[dim]]

# ---------------------------------------------------------------------------
# Dimension display config
# Labels and ind_labels are derived from display_name fields in indicators.json
# via cfg.  Only the chart colours are render-specific and hardcoded here.
# ---------------------------------------------------------------------------
_ABBR_TO_DIM = {meta["abbr"].lower(): dim for dim, meta in cfg.DIMENSION_META.items()}

_DIM_COLORS: dict[str, str] = {
    "ip": "#C0392B",
    "iv": "#E67E22",
    "ie": "#27AE60",
    "ig": "#2980B9",
}

DIMS: dict = {
    abbr: {
        "label": (
            f"{cfg.DIMENSION_META[dim]['abbr']} - {cfg.DIMENSION_META[dim]['display_name']}"
            + (" (inverted in IIC)" if cfg.DIMENSION_META[dim]["invert"] else "")
        ),
        "color": _DIM_COLORS[abbr],
        "indicators": cfg.DIMENSIONS[dim],
        "ind_labels": {
            k: f"{k} - {cfg.INDICATORS[k]['display_name']}"
            for k in cfg.DIMENSIONS[dim]
        },
    }
    for abbr, dim in _ABBR_TO_DIM.items()
}

# ---------------------------------------------------------------------------
# H3 → GeoDataFrame helpers
# ---------------------------------------------------------------------------

def h3_to_polygon(h3_id: str) -> Polygon:
    coords = h3.cell_to_boundary(h3_id)
    return Polygon([(lng, lat) for lat, lng in coords])


def build_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert a DataFrame with an h3_id column into a GeoDataFrame (EPSG:4326)."""
    df = df.copy()
    df["geometry"] = df["h3_id"].apply(h3_to_polygon)
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
