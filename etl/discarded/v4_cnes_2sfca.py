import pandas as pd
import numpy as np
import h3
import geopandas as gpd
from scipy.spatial import cKDTree
import sys
from pathlib import Path
from datetime import datetime

"""
2SFCA (Two-Step Floating Catchment Area) — Acessibilidade a Saúde (v4 — CNES).

Diferença em relação ao modelo gravitacional simples (v4_cnes.py):
    O modelo gravitacional simples mede apenas a oferta de serviços ponderada
    pela distância, sem considerar quantas pessoas competem por cada
    estabelecimento. Isso gera um artefato: cidades pequenas com poucos
    estabelecimentos concentrados aparecem mais acessíveis do que grandes
    cidades com muitos estabelecimentos — porque a mesma oferta serve menos
    pessoas num espaço menor.

    O 2SFCA corrige isso dividindo a capacidade de cada estabelecimento pela
    demanda da população ao seu redor antes de somar as contribuições.

Fórmula em dois passos:

    Passo 1 — Razão oferta/demanda por estabelecimento j:
        R_j = capacity_j / Σ [ P_i × W(d_ij) ]  para todo hexágono i
              dentro do raio de influência (catchment_m)

    Passo 2 — Acessibilidade acumulada por hexágono i:
        A_i = Σ [ R_j × W(d_ij) ]  para todo estabelecimento j
              dentro do raio de influência

    W(d) = 1 / (d + DISTANCE_BUFFER_M)  — decaimento gravitacional

Dados de população:
    v01006 (IBGE Censo 2022) — total de pessoas por setor censitário,
    agregado ao H3 via interpolação dassimétrica (peso_dom), replicando
    a mesma lógica do ETL de indicadores censitários.

Parâmetros configuráveis em indicators.json → v4 → source:
    catchment_m     : raio de influência em metros (padrão: 5000)
    pop_col         : variável censitária de população (padrão: v01006)
    pop_source_dir  : pasta dos CSVs do IBGE (relativa a RAW_DIR)

Saída:
    Arquivo separado (_2sfca) para comparação com o modelo gravitacional.
    Usa a mesma coluna col_norm (v4_sau_norm) para facilitar substituição.

Direção do indicador:
    A_i alto → boa acessibilidade per capita → MENOS vulnerável.
    col_norm = 1 − rank_percentil(A_i): alto = mais vulnerável.
"""

# ==============================================================================
# 1. ENVIRONMENT CONFIGURATION
# ==============================================================================
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.append(PROJECT_ROOT)

from src import config as cfg
from src import utils

# ==============================================================================
# 2. PATHS AND COLUMN NAMES
# ==============================================================================
h3_path   = cfg.FILES_H3["base_metadata"]
cnes_path = cfg.RAW_DIR / cfg.INDICATORS["v4"]["source"]["file"]

col_norm = cfg.COLUMN_MAP["v4"]
col_abs  = col_norm.replace("_norm", "_abs")

output_path = cfg.CLEAN_DIR / f"{cfg.FILE_PREFIX}_v4_saude_2sfca.parquet"

now = datetime.now().strftime("%Y%m%d_%H%M%S")
DIAGNOSTIC_TXT = cfg.DIAGNOSE_DIR / f"diagnostic_h3_v4_cnes_2sfca_{now}.txt"

# ==============================================================================
# 3. PARAMETERS (from indicators.json → v4 → source)
# ==============================================================================
_src = cfg.INDICATORS["v4"]["source"]
CATCHMENT_M       = _src.get("catchment_m", 5000)
DISTANCE_BUFFER_M = 100
POP_COL           = _src.get("pop_col", "v01006")
POP_SOURCE_DIR    = cfg.RAW_DIR / _src.get("pop_source_dir", "ibge/censo/2022/agregados_por_setores/t0")

CNES_SERVICES = [
    "st_centro_cirurgico", "st_centro_obstetrico", "st_centro_neonatal",
    "st_atend_hospitalar", "st_servico_apoio", "st_atend_ambulatorial",
]

# ==============================================================================
# 4. LOAD H3 BASE
# ==============================================================================
print("Starting ETL Pipeline - V4 CNES 2SFCA...")
print("1/6 - Loading H3 base...")

df_h3 = pd.read_parquet(h3_path)
df_h3.columns = df_h3.columns.str.lower()

# ==============================================================================
# 5. LOAD POPULATION (v01006) AND AGGREGATE TO H3
# ==============================================================================
print(f"2/6 - Loading population ({POP_COL}) from census sectors and aggregating to H3...")

pop_csvs = list(POP_SOURCE_DIR.glob("*.csv"))
df_pop_sector = None

for f in pop_csvs:
    try:
        with open(f, "r", encoding="utf-8", errors="ignore") as tf:
            sep = ";" if ";" in tf.readline() else ","
        df_tmp = pd.read_csv(f, sep=sep, dtype={"cd_setor": str}, encoding="utf-8")
        df_tmp.columns = df_tmp.columns.str.lower()
        if "cd_setor" in df_tmp.columns and POP_COL in df_tmp.columns:
            df_pop_sector = df_tmp[["cd_setor", POP_COL]].copy()
            df_pop_sector[POP_COL] = pd.to_numeric(df_pop_sector[POP_COL], errors="coerce").fillna(0)
            print(f"  Found {POP_COL} in {f.name}")
            break
    except Exception as e:
        print(f"  Warning: could not read {f.name}: {e}")

if df_pop_sector is None:
    raise FileNotFoundError(
        f"Column '{POP_COL}' not found in any CSV under {POP_SOURCE_DIR}. "
        "Check pop_col and pop_source_dir in indicators.json."
    )

# Dasymetric aggregation: same weighting (peso_dom) used by the census ETL
df_h3["cd_setor"] = df_h3["cd_setor"].astype(str)
df_pop_sector["cd_setor"] = df_pop_sector["cd_setor"].astype(str)

df_merged = df_h3[["h3_id", "cd_setor", "peso_dom"]].merge(df_pop_sector, on="cd_setor", how="left")
df_merged[POP_COL] = df_merged[POP_COL].fillna(0) * df_merged["peso_dom"].fillna(0)
df_pop_h3 = df_merged.groupby("h3_id")[POP_COL].sum().reset_index()
df_pop_h3.columns = ["h3_id", "population"]

df_h3 = df_h3.merge(df_pop_h3, on="h3_id", how="left")
df_h3["population"] = df_h3["population"].fillna(0)

print(f"  Total population covered: {df_h3['population'].sum():,.0f}")

# ==============================================================================
# 6. LOAD AND PREPARE CNES DATA
# ==============================================================================
print("3/6 - Loading CNES and calculating capacity scores...")

if not cnes_path.exists():
    raise FileNotFoundError(f"CNES file not found at: {cnes_path}")

df_cnes = pd.read_csv(cnes_path, sep=";", low_memory=False, encoding="latin1")
df_cnes.columns = df_cnes.columns.str.lower()
df_cnes["latitude"]  = pd.to_numeric(df_cnes["latitude"],  errors="coerce")
df_cnes["longitude"] = pd.to_numeric(df_cnes["longitude"], errors="coerce")
df_cnes = df_cnes.dropna(subset=["latitude", "longitude"])

for col in CNES_SERVICES:
    if col in df_cnes.columns:
        df_cnes[col] = pd.to_numeric(df_cnes[col], errors="coerce").fillna(0)

df_cnes["capacity_score"] = df_cnes[CNES_SERVICES].sum(axis=1) + 1

# ==============================================================================
# 7. CONVERT COORDINATES TO METRIC CRS
# ==============================================================================
print("4/6 - Converting coordinates to EPSG:5880...")

gdf_cnes = gpd.GeoDataFrame(
    df_cnes,
    geometry=gpd.points_from_xy(df_cnes.longitude, df_cnes.latitude),
    crs="EPSG:4326",
).to_crs("EPSG:5880")

def get_h3_centroid(h3_id):
    try:
        return h3.cell_to_latlng(h3_id) if hasattr(h3, "cell_to_latlng") else h3.h3_to_geo(h3_id)
    except Exception:
        return (np.nan, np.nan)

centroids    = df_h3["h3_id"].apply(get_h3_centroid)
df_h3["lat"] = [c[0] for c in centroids]
df_h3["lng"] = [c[1] for c in centroids]

gdf_h3 = gpd.GeoDataFrame(
    df_h3,
    geometry=gpd.points_from_xy(df_h3.lng, df_h3.lat),
    crs="EPSG:4326",
).to_crs("EPSG:5880")

coords_cnes = np.array(list(zip(gdf_cnes.geometry.x, gdf_cnes.geometry.y)))
coords_h3   = np.array(list(zip(gdf_h3.geometry.x,   gdf_h3.geometry.y)))
capacities  = gdf_cnes["capacity_score"].values
population  = df_h3["population"].values

# ==============================================================================
# 8. 2SFCA — PASSO 1: RAZÃO OFERTA/DEMANDA POR ESTABELECIMENTO
# ==============================================================================
print(f"5/6 - 2SFCA Passo 1: razão oferta/demanda por estabelecimento (catchment={CATCHMENT_M}m)...")
print("  Consultando hexágonos na área de influência de cada estabelecimento — pode levar alguns minutos.")

tree_h3 = cKDTree(coords_h3)
# Para cada estabelecimento, retorna os índices dos hexágonos dentro do raio
cnes_catchments = tree_h3.query_ball_point(coords_cnes, r=CATCHMENT_M, workers=-1)

R = np.zeros(len(coords_cnes))

for j, neighbor_idxs in enumerate(cnes_catchments):
    if j % 50_000 == 0:
        print(f"    Estabelecimento {j:,} / {len(coords_cnes):,}...")
    if not neighbor_idxs:
        # Nenhum hexágono no raio: estabelecimento não contribui para nenhum score
        continue
    neighbor_idxs = np.asarray(neighbor_idxs)
    dists   = np.linalg.norm(coords_h3[neighbor_idxs] - coords_cnes[j], axis=1)
    weights = 1.0 / (dists + DISTANCE_BUFFER_M)
    demand  = np.sum(population[neighbor_idxs] * weights)
    if demand > 0:
        R[j] = capacities[j] / demand

# ==============================================================================
# 9. 2SFCA — PASSO 2: SCORE DE ACESSIBILIDADE POR HEXÁGONO
# ==============================================================================
print("5/6 - 2SFCA Passo 2: acumulando razões de estabelecimentos por hexágono...")

tree_cnes = cKDTree(coords_cnes)
h3_catchments = tree_cnes.query_ball_point(coords_h3, r=CATCHMENT_M, workers=-1)

A = np.zeros(len(coords_h3))

for i, neighbor_idxs in enumerate(h3_catchments):
    if not neighbor_idxs:
        continue
    neighbor_idxs = np.asarray(neighbor_idxs)
    dists   = np.linalg.norm(coords_cnes[neighbor_idxs] - coords_h3[i], axis=1)
    weights = 1.0 / (dists + DISTANCE_BUFFER_M)
    A[i]    = np.sum(R[neighbor_idxs] * weights)

df_h3[col_abs] = A
df_h3 = df_h3.drop(columns=["lat", "lng", "geometry", "population"], errors="ignore")

# ==============================================================================
# 10. NORMALIZAÇÃO (RANK PERCENTIL) E EXPORTAÇÃO
# ==============================================================================
print("6/6 - Normalizando e salvando...")

# Rank percentil: robusto à distribuição power-law dos scores 2SFCA.
# col_norm = 1 → menos acessível (mais vulnerável); 0 → mais acessível.
df_h3[col_norm] = 1.0 - df_h3[col_abs].rank(pct=True)

df_export = df_h3[["h3_id", col_abs, col_norm]]
df_export.to_parquet(output_path, index=False)
print(f"  ✓ Salvo em: {output_path.name}")

# ==============================================================================
# 11. DIAGNÓSTICO
# ==============================================================================
print("Gerando arquivo de diagnóstico...")

with open(DIAGNOSTIC_TXT, "w", encoding="utf-8") as f:
    f.write("=" * 50 + "\n")
    f.write("DIAGNÓSTICO 2SFCA — ACESSIBILIDADE À SAÚDE (V4 - CNES)\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Total de estabelecimentos processados : {len(df_cnes):,}\n")
    f.write(f"Total de hexágonos processados        : {len(df_h3):,}\n")
    f.write(f"Raio de influência (catchment_m)      : {CATCHMENT_M} m\n")
    f.write(f"Buffer de distância (m)               : {DISTANCE_BUFFER_M}\n")
    f.write(f"Coluna de população                   : {POP_COL}\n\n")

    facilities_no_demand = (R == 0).sum()
    hexagons_no_access   = (A == 0).sum()
    f.write(f"Estabelecimentos sem população no raio : {facilities_no_demand:,}\n")
    f.write(f"Hexágonos sem estabelecimento no raio  : {hexagons_no_access:,}\n\n")

    f.write(f"--- ESTATÍSTICAS DA VARIÁVEL NORMALIZADA ({col_norm}) ---\n")
    f.write(df_h3[col_norm].describe().to_string() + "\n\n")

    cols_show = ["h3_id", col_abs, col_norm]
    for c in ["nm_mun", "nm_uf"]:
        if c in df_h3.columns:
            cols_show.insert(1, c)

    f.write("--- TOP 5 MAIS VULNERÁVEIS (Menor Acessibilidade) ---\n")
    f.write(df_h3.sort_values(by=col_norm, ascending=False)[cols_show].head().to_string() + "\n\n")

    f.write("--- TOP 5 MENOS VULNERÁVEIS (Maior Acessibilidade) ---\n")
    f.write(df_h3.sort_values(by=col_norm, ascending=True)[cols_show].head().to_string() + "\n")

print(f"✅ Diagnóstico 2SFCA salvo em: {DIAGNOSTIC_TXT}")

# ==============================================================================
# 12. VISUALIZAÇÃO (MAPAS DE DEBUG + DISTRIBUIÇÃO)
# ==============================================================================
print("Gerando mapas de debug e gráficos de distribuição...")

try:
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon as ShapelyPolygon
except ImportError as e:
    print(f"  Visualizações puladas — dependência ausente: {e}")
else:
    DEBUG_DIR = cfg.FIGURES_DIR / "debug_saude"
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    import json as _json

    CMAP = "Reds"
    _cities_path = cfg.BASE_DIR / "diagnose" / "cities.json"
    if _cities_path.exists():
        with open(_cities_path, "r", encoding="utf-8") as _f:
            CITIES = _json.load(_f)["cities"]
    else:
        CITIES = [
            {"nm_mun": "Porto Alegre",   "nm_uf": "Rio Grande do Sul"},
            {"nm_mun": "São Paulo",      "nm_uf": "São Paulo"},
            {"nm_mun": "Rio de Janeiro", "nm_uf": "Rio de Janeiro"},
        ]
    _PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

    def _h3_to_polygon(h3_id):
        try:
            boundary = (
                h3.cell_to_boundary(h3_id)
                if hasattr(h3, "cell_to_boundary")
                else h3.h3_to_geo_boundary(h3_id)
            )
            return ShapelyPolygon([(lng, lat) for lat, lng in boundary])
        except Exception:
            return None

    def _city_map(city_info):
        nm_mun = city_info["nm_mun"]
        nm_uf  = city_info.get("nm_uf")
        label  = nm_mun + (f" ({nm_uf})" if nm_uf else "")
        mask = df_h3["nm_mun"].str.lower() == nm_mun.lower()
        if nm_uf and "nm_uf" in df_h3.columns:
            mask &= df_h3["nm_uf"].str.lower() == nm_uf.lower()
        df_city = df_h3[mask].copy()
        if len(df_city) == 0:
            print(f"  {label} não encontrado — pulado.")
            return
        df_city["geometry"] = df_city["h3_id"].map(_h3_to_polygon)
        df_city = df_city.dropna(subset=["geometry"])
        if len(df_city) == 0:
            print(f"  {label}: todas as conversões de polígono falharam — pulado.")
            return
        gdf = gpd.GeoDataFrame(df_city, geometry="geometry", crs="EPSG:4326")

        safe = (
            label.lower()
            .replace(" ", "_").replace("(", "").replace(")", "")
            .replace("ã", "a").replace("â", "a")
            .replace("ê", "e").replace("é", "e")
            .replace("ó", "o").replace("ú", "u")
        )
        fig, ax = plt.subplots(figsize=(12, 12))
        gdf.plot(
            column=col_norm, ax=ax, cmap=CMAP,
            vmin=0, vmax=1, legend=True,
            legend_kwds={"label": "Vulnerabilidade (0 = menor, 1 = maior)", "shrink": 0.5},
        )
        minx, miny, maxx, maxy = gdf.total_bounds
        pad_x = (maxx - minx) * 0.05
        pad_y = (maxy - miny) * 0.05
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
        ax.set_title(
            f"Indicador v4 — Acessibilidade à Saúde (2SFCA)\n"
            f"{col_norm}  |  {label}  |  catchment={int(CATCHMENT_M/1000)}km",
            fontsize=12,
        )
        ax.axis("off")
        out = DEBUG_DIR / f"v4_sau_2sfca_{safe}_c{int(CATCHMENT_M)}m_{now}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {label}: {out.name}")

    # ── Mapas por cidade ──────────────────────────────────────────────────
    if "nm_mun" in df_h3.columns:
        for city in CITIES:
            _city_map(city)
    else:
        print("  nm_mun ausente em df_h3 — mapas de cidade pulados.")

    # ── Distribuição: histograma nacional + cidades em destaque ──────────
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(
        df_h3[col_norm], bins=100,
        alpha=0.25, color="gray", density=True, label=f"Brasil (n={len(df_h3):,})",
    )
    if "nm_mun" in df_h3.columns:
        for idx, city_info in enumerate(CITIES):
            nm_mun = city_info["nm_mun"]
            nm_uf  = city_info.get("nm_uf")
            mask = df_h3["nm_mun"].str.lower() == nm_mun.lower()
            if nm_uf and "nm_uf" in df_h3.columns:
                mask &= df_h3["nm_uf"].str.lower() == nm_uf.lower()
            df_city = df_h3[mask]
            label = nm_mun + (f" ({nm_uf})" if nm_uf else "")
            color = _PALETTE[idx % len(_PALETTE)]
            if len(df_city) > 0:
                ax.hist(
                    df_city[col_norm], bins=60,
                    alpha=0.55, color=color, density=True,
                    label=f"{label} (n={len(df_city):,})",
                )
    ax.axvline(0.5, color="black", linestyle=":", linewidth=1.2, label="limiar 0.5")
    ax.set_xlabel(f"{col_norm}  (0 = menor vulnerabilidade, 1 = maior)")
    ax.set_ylabel("Densidade")
    ax.set_title(f"Distribuição de {col_norm} — Modelo 2SFCA  (catchment={int(CATCHMENT_M/1000)}km)")
    ax.legend()
    out_dist = DEBUG_DIR / f"v4_sau_2sfca_distribuicao_c{int(CATCHMENT_M)}m_{now}.png"
    fig.savefig(out_dist, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Distribuição: {out_dist.name}")

print("✅ Pipeline 2SFCA completo.")
