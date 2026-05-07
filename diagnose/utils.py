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
# Indicator lists come from cfg.DIMENSIONS so they stay in sync with
# indicators.json.  Colors and Portuguese labels are render-specific and
# live here.
# ---------------------------------------------------------------------------
_ABBR_TO_DIM = {meta["abbr"].lower(): dim for dim, meta in cfg.DIMENSION_META.items()}

DIMS: dict = {
    "ip": {
        "label": "IP - Grupos Prioritários",
        "color": "#C0392B",
        "indicators": cfg.DIMENSIONS[_ABBR_TO_DIM["ip"]],
        "ind_labels": {
            "p1": "p1 - Mulheres negras chefes de domicílio",
            "p2": "p2 - População negra",
            "p3": "p3 - Indígenas e quilombolas",
            "p4": "p4 - Idosos (60 anos ou mais)",
            "p5": "p5 - Crianças (9 anos ou menos)",
        },
    },
    "iv": {
        "label": "IV - Vulnerabilidade Socioeconômica",
        "color": "#E67E22",
        "indicators": cfg.DIMENSIONS[_ABBR_TO_DIM["iv"]],
        "ind_labels": {
            "v1": "v1 - Baixa renda",
            "v2": "v2 - Moradia precária",
            "v3": "v3 - Educação",
            "v4": "v4 - Acesso à saúde",
            "v5": "v5 - Infraestrutura",
        },
    },
    "ie": {
        "label": "IE - Exposição a Riscos Climáticos",
        "color": "#27AE60",
        "indicators": cfg.DIMENSIONS[_ABBR_TO_DIM["ie"]],
        "ind_labels": {
            "e1": "e1 - Deslizamentos de terra",
            "e2": "e2 - Inundações, alagamentos e enxurradas",
            "e3": "e3 - Elevação do nível do mar",
            "e4": "e4 - Calor extremo",
            "e5": "e5 - Queimadas",
        },
    },
    "ig": {
        "label": "IG - Gestão Municipal (invertida no IIC)",
        "color": "#2980B9",
        "indicators": cfg.DIMENSIONS[_ABBR_TO_DIM["ig"]],
        "ind_labels": {
            "g1": "g1 - Investimento ambiental",
            "g2": "g2 - Plano de contingência",
            "g3": "g3 - Participação em NUPDECs",
            "g4": "g4 - Conselhos municipais",
            "g5": "g5 - Sistemas de alerta",
            "g6": "g6 - Mapeamento e zoneamento de risco",
            "g7": "g7 - Cadastro de famílias em risco",
            "g8": "g8 - Políticas de direitos humanos",
        },
    },
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
