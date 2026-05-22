#!/usr/bin/env python3
"""
SGB — Interseção com Grade H3 res9
====================================
Carrega os GeoPackages harmonizados (02_sgb_mass_br.gpkg e 02_sgb_floods_br.gpkg),
intersecta com a grade H3 res9 por estado, e calcula a fração de área em classes
Alta/Muito Alta (4–5) por hexágono.

Processa um estado por vez para manter uso de memória baixo. Hexágonos em fronteira
de estado recebem a contribuição de ambos os estados na agregação final.

Outputs em data/inputs/clean/:
  br_h3_sgb_massa.parquet
  br_h3_sgb_inundacoes.parquet

Colunas de saída:
  h3_id              — índice H3 res9 (string)
  cd_estado          — UF com maior área SGB mapeada no hexágono
  sgb_alta_mta_frac  — área em classes 4–5 / área SGB total na célula
  sgb_max_class      — classe máxima presente na célula (0–5)
  sgb_coverage_frac  — área SGB / área total do hexágono (útil para filtrar bordas)
  n_records          — número de feições SGB que intersectam o hexágono

USO:
  python 03_sgb_h3_intersect.py                     # ambos os tipos
  python 03_sgb_h3_intersect.py --tipo massa        # só movimentos de massa
  python 03_sgb_h3_intersect.py --tipo inundacao    # só inundações
  python 03_sgb_h3_intersect.py --state SP,RJ       # filtra estados (teste)
  python 03_sgb_h3_intersect.py --dry-run           # não escreve saída
"""

import json
import sys
import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import h3
from shapely.geometry import Polygon

# ── Paths via config ───────────────────────────────────────────────────────────
def _load_data_dir() -> Path:
    project_root = Path(__file__).resolve().parents[3]
    config_path = project_root / "config" / "config.local.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config não encontrado: {config_path}\n"
            "Crie config/config.local.json com {\"data_dir\": \"/caminho/para/data/\"}"
        )
    with open(config_path) as f:
        return Path(json.load(f)["data_dir"])


_DATA_DIR      = _load_data_dir()
HARMONIZED_DIR = _DATA_DIR / "inputs/raw/sgb/harmonized"
INVENTORY_PATH = _DATA_DIR / "inputs/raw/sgb/01_sgb_inventory.csv"
OUTPUT_DIR     = _DATA_DIR / "inputs/clean"

GPKG_FILES = {
    "massa":     HARMONIZED_DIR / "02_sgb_mass_br.gpkg",
    "inundacao": HARMONIZED_DIR / "02_sgb_floods_br.gpkg",
}
OUTPUT_FILES = {
    "massa":     OUTPUT_DIR / "br_h3_sgb_massa.parquet",
    "inundacao": OUTPUT_DIR / "br_h3_sgb_inundacoes.parquet",
}

CRS_GEO  = "EPSG:4674"   # SIRGAS 2000 geográfico — sistema nativo do H3
CRS_PROJ = "EPSG:5880"   # SIRGAS 2000 / Brazil Polyconic — para cálculo de área em m²
H3_RES   = 9

# Área média de um hexágono H3 res9 em m²  (todos os hexágonos têm ~a mesma área)
H3_CELL_AREA_M2: float = h3.average_hexagon_area(H3_RES, unit="m^2")


# ══════════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS H3
# ══════════════════════════════════════════════════════════════════════════════

def _h3_cell_to_polygon(cell: str) -> Polygon:
    """Converte um H3 cell ID para Shapely Polygon em CRS_GEO (lng, lat)."""
    # h3.cell_to_boundary retorna [(lat, lng), ...] — inverte para (lng, lat)
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])


def cells_to_gdf(cells: set) -> gpd.GeoDataFrame:
    """Converte set de H3 cell IDs para GeoDataFrame de polígonos em CRS_GEO."""
    return gpd.GeoDataFrame(
        [{"h3_id": c, "geometry": _h3_cell_to_polygon(c)} for c in cells],
        crs=CRS_GEO,
    )


def get_h3_cells_for_geom(geom) -> set:
    """
    Retorna H3 res9 cells cujo centroide cai dentro da geometria.
    Trata MultiPolygon e GeometryCollection recursivamente.
    """
    if geom is None or geom.is_empty:
        return set()
    try:
        return set(h3.geo_to_cells(geom.__geo_interface__, H3_RES))
    except Exception:
        # Fallback para geometrias compostas que o h3 não aceita diretamente
        if hasattr(geom, "geoms"):
            cells: set = set()
            for sub in geom.geoms:
                cells |= get_h3_cells_for_geom(sub)
            return cells
        return set()


def get_h3_cells_for_gdf(gdf: gpd.GeoDataFrame) -> set:
    """Retorna o conjunto de H3 cells cobrindo todos os polígonos do GDF."""
    gdf_geo = gdf.to_crs(CRS_GEO)
    cells: set = set()
    for geom in gdf_geo.geometry:
        cells |= get_h3_cells_for_geom(geom)
    return cells


# ══════════════════════════════════════════════════════════════════════════════
# INTERSEÇÃO POR ESTADO
# ══════════════════════════════════════════════════════════════════════════════

def intersect_state(
    state_gdf: gpd.GeoDataFrame,
    state_code: str,
) -> pd.DataFrame:
    """
    Para os polígonos SGB de um estado, calcula por hexágono H3 a área
    em cada faixa de classe. Retorna DataFrame com colunas intermediárias
    (sgb_area_m2, alta_area_m2) para agregação posterior entre estados.

    Retorna DataFrame vazio se não houver interseção.
    """
    # Polígonos válidos com classe mapeada (classe_num >= 0)
    valid = state_gdf[state_gdf["classe_num"] >= 0].copy()
    if valid.empty:
        return pd.DataFrame()

    # Corrige geometrias inválidas antes da interseção
    valid.geometry = valid.geometry.make_valid()
    valid = valid[~valid.geometry.is_empty]
    if valid.empty:
        return pd.DataFrame()

    # 1. Obtém células H3 candidatas via polyfill
    cells = get_h3_cells_for_gdf(valid)
    if not cells:
        return pd.DataFrame()

    # 2. Converte células para GeoDataFrame projetado
    h3_gdf_proj = cells_to_gdf(cells).to_crs(CRS_PROJ)

    # 3. Projeta SGB para cálculo de área
    sgb_proj = valid[["classe_num", "geometry"]].to_crs(CRS_PROJ)

    # 4. Interseção exata
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*index_parts.*")
        inter = gpd.overlay(sgb_proj, h3_gdf_proj, how="intersection", keep_geom_type=False)

    if inter.empty:
        return pd.DataFrame()

    inter["area_m2"] = inter.geometry.area
    inter = inter[inter["area_m2"] > 0]
    if inter.empty:
        return pd.DataFrame()

    # 5. Agrega por hexágono
    grp       = inter.groupby("h3_id")
    total     = grp["area_m2"].sum().rename("sgb_area_m2")
    alta      = (
        inter[inter["classe_num"] >= 4]
        .groupby("h3_id")["area_m2"]
        .sum()
        .rename("alta_area_m2")
    )
    max_class = grp["classe_num"].max().rename("sgb_max_class")
    n_recs    = grp.size().rename("n_records")

    result = pd.concat([total, alta, max_class, n_recs], axis=1).reset_index()
    result["alta_area_m2"] = result["alta_area_m2"].fillna(0.0)
    result["cd_estado"]    = state_code
    return result


# ══════════════════════════════════════════════════════════════════════════════
# AGREGAÇÃO ENTRE ESTADOS
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_results(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatena resultados por estado e reconcilia hexágonos de fronteira
    (que aparecem em mais de um estado). cd_estado fica com o estado que
    tem maior área SGB mapeada no hexágono.
    """
    df = pd.concat(parts, ignore_index=True)

    # Para células de fronteira: soma áreas de ambos os estados
    agg = (
        df.groupby("h3_id")
        .agg(
            sgb_area_m2   = ("sgb_area_m2",  "sum"),
            alta_area_m2  = ("alta_area_m2", "sum"),
            sgb_max_class = ("sgb_max_class", "max"),
            n_records     = ("n_records",     "sum"),
        )
        .reset_index()
    )

    # Estado dominante = maior área SGB mapeada
    dominant = (
        df.sort_values("sgb_area_m2", ascending=False)
        .drop_duplicates("h3_id")[["h3_id", "cd_estado"]]
    )
    agg = agg.merge(dominant, on="h3_id")

    # Frações finais
    agg["sgb_alta_mta_frac"] = (agg["alta_area_m2"] / agg["sgb_area_m2"]).clip(0.0, 1.0)
    agg["sgb_coverage_frac"] = (agg["sgb_area_m2"] / H3_CELL_AREA_M2).clip(0.0, 1.0)

    return agg[[
        "h3_id", "cd_estado",
        "sgb_alta_mta_frac", "sgb_max_class",
        "sgb_coverage_frac", "n_records",
    ]]


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTRAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def get_states_from_inventory(tipo: str) -> list[str]:
    """Lê estados disponíveis no inventário sem abrir o GeoPackage."""
    if not INVENTORY_PATH.exists():
        return []
    df = pd.read_csv(INVENTORY_PATH, usecols=["tipo", "cd_estado"])
    return sorted(df[df["tipo"] == tipo]["cd_estado"].dropna().unique())


def process_tipo(
    tipo: str,
    state_filter: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    gpkg_path = GPKG_FILES[tipo]
    if not gpkg_path.exists():
        print(f"\n[ERRO] GeoPackage não encontrado: {gpkg_path}")
        print("       Execute 02_sgb_harmonize.py primeiro.")
        return

    print(f"\n{'═' * 60}")
    print(f"TIPO: {tipo.upper()}")
    print(f"{'═' * 60}")

    # Lista de estados a processar
    states = get_states_from_inventory(tipo)
    if state_filter:
        states = [s for s in states if s in state_filter]
    if not states:
        # Fallback: lê estados diretamente do GeoPackage (carrega geometria mas descarta imediatamente)
        print("  Inventário não encontrado — lendo estados do GeoPackage...")
        meta_gdf = gpd.read_file(gpkg_path, layer="suscetibilidade", where="cd_estado IS NOT NULL")
        states = sorted(meta_gdf["cd_estado"].dropna().unique())
        del meta_gdf
        if state_filter:
            states = [s for s in states if s in state_filter]

    print(f"  {len(states)} estado(s) a processar: {', '.join(states)}")
    if dry_run:
        print("  [DRY RUN] — nenhum arquivo será escrito.")

    parts: list[pd.DataFrame] = []

    for i, state in enumerate(states, 1):
        print(f"  [{i:>2}/{len(states)}] {state}", end="  ", flush=True)

        # Carrega apenas o estado atual (reduz uso de memória)
        try:
            state_gdf = gpd.read_file(
                gpkg_path,
                layer="suscetibilidade",
                where=f"cd_estado = '{state}'",
            )
        except Exception as e:
            print(f"✗ erro ao carregar: {e}")
            continue

        if state_gdf.empty:
            print("sem feições")
            continue

        n_feats = len(state_gdf)
        print(f"{n_feats:,} feições →", end=" ", flush=True)

        result = intersect_state(state_gdf, state)
        del state_gdf  # libera memória imediatamente

        if result.empty:
            print("sem interseções H3")
            continue

        n_cells = len(result)
        alta_count = (result["alta_area_m2"] > 0).sum()
        print(f"{n_cells:,} hexágonos ({alta_count:,} com classe ≥ 4)")
        parts.append(result)

    if not parts:
        print("\n  Nenhum resultado — verifique se os GeoPackages têm dados.")
        return

    print("\n  Agregando resultados entre estados...", end=" ", flush=True)
    final = aggregate_results(parts)
    print(f"{len(final):,} hexágonos únicos")
    print(f"  sgb_alta_mta_frac > 0.3 : {(final['sgb_alta_mta_frac'] > 0.3).sum():,} hexágonos")
    print(f"  sgb_max_class >= 4      : {(final['sgb_max_class'] >= 4).sum():,} hexágonos")

    if not dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_FILES[tipo]
        final.to_parquet(out_path, index=False)
        print(f"  ✓ Salvo: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SGB — Interseção com grade H3 res9",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tipo",
        choices=["massa", "inundacao", "ambos"],
        default="ambos",
        help="Tipo de processo a processar (padrão: ambos)",
    )
    parser.add_argument(
        "--state",
        type=str,
        default=None,
        help="Filtra por estado(s), ex: SP,RJ (para testes)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simula o processamento sem escrever arquivos",
    )

    args = parser.parse_args()
    state_filter = (
        [s.strip().upper() for s in args.state.split(",")]
        if args.state
        else None
    )
    tipos = ["massa", "inundacao"] if args.tipo == "ambos" else [args.tipo]

    print(f"H3 res{H3_RES} | célula média: {H3_CELL_AREA_M2:,.0f} m²")
    for tipo in tipos:
        process_tipo(tipo, state_filter=state_filter, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
