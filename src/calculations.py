import pandas as pd
import logging
from . import config as cfg

def calculate_simple_iic(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Calculando Índice de Injustiça Climática via média simples...")
    df = df.copy()

    # 1. Índice por dimensão (com inversão de IG)
    dim_cols = []
    for dim_name, dim_meta in cfg.DIMENSION_META.items():
        indicator_keys = cfg.DIMENSIONS[dim_name]
        existing = [k for k in indicator_keys if k in df.columns]
        if not existing:
            logging.warning(f"Dimensão '{dim_name}': nenhum indicador encontrado.")
            continue

        abbr = dim_meta['abbr'].lower()   # "ip", "iv", "ie", "ig"
        dim_avg = df[existing].mean(axis=1)

        if dim_meta['invert']:
            dim_avg = 1.0 - dim_avg

        df[abbr] = dim_avg
        dim_cols.append(abbr)
        logging.info(f"Dimensão '{dim_name}' → {dim_meta['abbr']} calculada (invertida={dim_meta['invert']}).")

    # 2. IIC final: média simples dos índices de dimensão
    df['iic_final'] = df[dim_cols].mean(axis=1)

    return df