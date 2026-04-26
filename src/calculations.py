import numpy as np
import pandas as pd
import logging
from . import config as cfg
from . import utils


def _nanmean_cols(df: pd.DataFrame, cols: list) -> pd.Series:
    """Row-wise mean across cols, one column at a time to avoid allocating a full (n_rows × n_cols) matrix."""
    total = np.zeros(len(df), dtype='float64')
    count = np.zeros(len(df), dtype='int32')
    for c in cols:
        vals = df[c].to_numpy(dtype='float64', na_value=np.nan)
        valid = ~np.isnan(vals)
        total[valid] += vals[valid]
        count[valid] += 1
    with np.errstate(invalid='ignore'):
        result = np.where(count > 0, total / count, np.nan)
    return pd.Series(result, index=df.index, dtype='float64')


def calculate_simple_iic(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Calculando Índice de Injustiça Climática via média simples...")

    # 1. Índice por dimensão (com inversão de IG)
    dim_cols = []
    for dim_name, dim_meta in cfg.DIMENSION_META.items():
        indicator_keys = cfg.DIMENSIONS[dim_name]
        existing = [k for k in indicator_keys if k in df.columns]
        if not existing:
            logging.warning(f"Dimensão '{dim_name}': nenhum indicador encontrado.")
            continue

        abbr = dim_meta['abbr'].lower()   # "ip", "iv", "ie", "ig"
        dim_avg = _nanmean_cols(df, existing)

        # Normalize each dimension to [0,1] with winsorization so all dimensions
        # have the same effective weight in the final IIC regardless of their natural spread
        dim_avg = utils.normalize_minmax(dim_avg, winsorize=True)

        if dim_meta['invert']:
            dim_avg = 1.0 - dim_avg

        df[abbr] = dim_avg
        dim_cols.append(abbr)
        logging.info(f"Dimensão '{dim_name}' → {dim_meta['abbr']} calculada (invertida={dim_meta['invert']}).")

    # 2. IIC final: média simples dos índices de dimensão (já em [0,1] — sem renormalizar)
    df['iic_final'] = _nanmean_cols(df, dim_cols)

    return df
