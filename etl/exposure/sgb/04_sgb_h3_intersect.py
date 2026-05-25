#!/usr/bin/env python3
"""
SGB — Interseção com Grade H3 res9
====================================
Carrega os GeoPackages harmonizados (03_sgb_mass_br.gpkg e 03_sgb_floods_br.gpkg),
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
  python 04_sgb_h3_intersect.py                     # ambos os tipos
  python 04_sgb_h3_intersect.py --tipo massa        # só movimentos de massa
  python 04_sgb_h3_intersect.py --tipo inundacao    # só inundações
  python 04_sgb_h3_intersect.py --state SP,RJ       # filtra estados (teste)
  python 04_sgb_h3_intersect.py --dry-run           # não escreve saída
  python 04_sgb_h3_intersect.py --resume            # continua de onde parou (parquets por estado)
"""

import io
import json
import shutil
import sys
import argparse
import math
import warnings
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import pandas as pd
import h3
import shapely
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
    with open(config_path, encoding="utf-8") as f:
        return Path(json.load(f)["data_dir"])


_DATA_DIR      = _load_data_dir()
HARMONIZED_DIR = _DATA_DIR / "inputs/raw/sgb/harmonized"
INVENTORY_PATH = _DATA_DIR / "inputs/raw/sgb/01_sgb_inventory.csv"
OUTPUT_DIR     = _DATA_DIR / "inputs/clean"

GPKG_FILES = {
    "massa":     HARMONIZED_DIR / "03_sgb_mass_br.gpkg",
    "inundacao": HARMONIZED_DIR / "03_sgb_floods_br.gpkg",
}
OUTPUT_FILES = {
    "massa":     OUTPUT_DIR / "br_h3_sgb_massa.parquet",
    "inundacao": OUTPUT_DIR / "br_h3_sgb_inundacoes.parquet",
}
PARTIAL_DIR = HARMONIZED_DIR / "04_partial"

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
    chunk_size: int = 500,
) -> pd.DataFrame:
    """
    Para os polígonos SGB de um estado, calcula por hexágono H3 a área
    em cada faixa de classe. Usa sjoin (R-tree) + shapely.intersection
    vetorizado — muito mais rápido que gpd.overlay para polígonos densos.

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

    n_feats  = len(valid)
    n_chunks = math.ceil(n_feats / chunk_size)
    sgb_proj = valid[["classe_num", "geometry"]].to_crs(CRS_PROJ).reset_index(drop=True)
    # Simplifica vértices antes da interseção: 20 m ≈ 11 % da aresta H3 res9 (174 m),
    # dentro da precisão nominal do mapeamento 1:25.000. Ver ADR-0033.
    sgb_proj.geometry = sgb_proj.geometry.simplify(20.0, preserve_topology=True)
    sgb_proj.geometry = sgb_proj.geometry.make_valid()  # simplify pode criar topologia inválida
    sgb_geo  = valid[["classe_num", "geometry"]].to_crs(CRS_GEO).reset_index(drop=True)

    all_inter: list[pd.DataFrame] = []

    for ci in range(n_chunks):
        lo, hi     = ci * chunk_size, min((ci + 1) * chunk_size, n_feats)
        chunk_proj = sgb_proj.iloc[lo:hi]
        chunk_geo  = sgb_geo.iloc[lo:hi]

        cells = get_h3_cells_for_gdf(chunk_geo)
        if cells:
            h3_chunk = cells_to_gdf(cells).to_crs(CRS_PROJ)

            # sjoin: acha pares (SGB feature, H3 cell) via índice R-tree — rápido
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*index_parts.*")
                joined = gpd.sjoin(
                    chunk_proj, h3_chunk,
                    how="inner", predicate="intersects",
                )

            if not joined.empty:
                # Mapeia geometria H3 para cada par
                h3_geom_map = h3_chunk.set_index("h3_id")["geometry"]
                # .to_numpy(object) garante array numpy puro — evita pandas
                # interceptar o ufunc via __array_ufunc__ e causar erros GEOS
                sgb_geoms = joined.geometry.to_numpy(dtype=object)
                h3_geoms  = joined["h3_id"].map(h3_geom_map).to_numpy(dtype=object)

                # Interseção vetorizada (shapely 2.x ufunc — sem Python loop)
                clips = shapely.intersection(sgb_geoms, h3_geoms)
                areas = shapely.area(clips)

                rows = pd.DataFrame({
                    "h3_id":     joined["h3_id"].values,
                    "classe_num": joined["classe_num"].values,
                    "area_m2":   areas,
                })
                rows = rows[rows["area_m2"] > 0]
                if not rows.empty:
                    all_inter.append(rows)

        pct = hi * 100 // n_feats
        print(
            f"\r    lote {ci + 1}/{n_chunks} ({pct:3d}%)  {hi:,}/{n_feats:,} feições",
            end="", flush=True,
        )

    if not all_inter:
        return pd.DataFrame()

    inter = pd.concat(all_inter, ignore_index=True)

    # Agrega por hexágono
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

def _intersect_state_worker(task: dict) -> dict:
    """Carrega um estado e calcula interseção H3. Chamado por ProcessPoolExecutor."""
    gpkg_path  = Path(task["gpkg_path"])
    state      = task["state"]
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        state_gdf = gpd.read_file(gpkg_path, layer="suscetibilidade", where=f"sigla_uf = '{state}'")
        if state_gdf.empty:
            sys.stdout = old_stdout
            return {"state": state, "result": None, "n_feats": 0,
                    "output": buf.getvalue(), "error": "sem feições"}
        n_feats = len(state_gdf)
        t0 = time.perf_counter()
        result_df = intersect_state(state_gdf, state)
        elapsed = time.perf_counter() - t0
        del state_gdf
    except Exception as exc:
        sys.stdout = old_stdout
        return {"state": state, "result": None, "n_feats": 0,
                "output": buf.getvalue(), "error": str(exc)}
    finally:
        sys.stdout = old_stdout

    if result_df.empty:
        return {"state": state, "result": None, "n_feats": n_feats,
                "output": buf.getvalue(), "elapsed": elapsed, "error": "sem interseções H3"}
    return {
        "state":   state,
        "result":  result_df,
        "n_feats": n_feats,
        "n_cells": len(result_df),
        "elapsed": elapsed,
        "output":  buf.getvalue(),
        "error":   None,
    }


def get_states_from_inventory(tipo: str) -> list[str]:
    """Lê estados disponíveis no inventário sem abrir o GeoPackage."""
    if not INVENTORY_PATH.exists():
        return []
    df = pd.read_csv(INVENTORY_PATH, usecols=["tipo", "cd_estado"], encoding="utf-8")
    return sorted(df[df["tipo"] == tipo]["cd_estado"].dropna().unique())


def process_tipo(
    tipo: str,
    state_filter: list[str] | None = None,
    dry_run: bool = False,
    resume: bool = False,
    workers: int = 1,
) -> None:
    gpkg_path = GPKG_FILES[tipo]
    if not gpkg_path.exists():
        print(f"\n[ERRO] GeoPackage não encontrado: {gpkg_path}")
        print("       Execute 02_sgb_extract.py e 03_sgb_harmonize.py primeiro.")
        return

    print(f"\n{'═' * 60}")
    print(f"TIPO: {tipo.upper()} | workers={workers}")
    print(f"{'═' * 60}")

    states = get_states_from_inventory(tipo)
    if state_filter:
        states = [s for s in states if s in state_filter]
    if not states:
        print("  Inventário não encontrado — lendo estados do GeoPackage...")
        meta_gdf = gpd.read_file(gpkg_path, layer="suscetibilidade", where="sigla_uf IS NOT NULL")
        states = sorted(meta_gdf["sigla_uf"].dropna().unique())
        del meta_gdf
        if state_filter:
            states = [s for s in states if s in state_filter]

    n_states   = len(states)
    partial_dir = PARTIAL_DIR / tipo

    # Resume: carrega estados já processados de parquets intermediários
    done_states: set[str] = set()
    if resume and not dry_run:
        if partial_dir.exists():
            done_states = {p.stem for p in partial_dir.glob("*.parquet")}
        if done_states:
            print(f"  [RESUME] {len(done_states)} estado(s) já processados: "
                  f"{', '.join(sorted(done_states))}")
    elif not dry_run and partial_dir.exists():
        shutil.rmtree(partial_dir)

    tasks  = [{"gpkg_path": str(gpkg_path), "state": s} for s in states if s not in done_states]
    n_tasks = len(tasks)
    n_skip  = n_states - n_tasks
    print(f"  {n_states} estado(s) total | {n_tasks} a processar, {n_skip} já concluídos")
    if dry_run:
        print("  [DRY RUN] — nenhum arquivo será escrito.")
        return

    # Carrega resultados parciais existentes para agregação final
    parts: list[pd.DataFrame] = []
    if resume and done_states:
        for state in sorted(done_states):
            p = partial_dir / f"{state}.parquet"
            if p.exists():
                parts.append(pd.read_parquet(p))

    interrupted = False
    t_total = time.perf_counter()

    def _handle_state_result(res: dict, idx: int) -> None:
        state_prefix = f"  [{idx:>2}/{n_tasks}] {res['state']}"
        if res.get("error"):
            print(f"{state_prefix}  ✗ {res['error']}")
            return
        n_cells    = res["n_cells"]
        alta_count = (res["result"]["alta_area_m2"] > 0).sum()
        elapsed    = res.get("elapsed", 0)
        print(f"{state_prefix}  {res['n_feats']:,} feições → {n_cells:,} hexágonos "
              f"({alta_count:,} com classe ≥ 4)  [{elapsed:.0f}s]")
        partial_dir.mkdir(parents=True, exist_ok=True)
        res["result"].to_parquet(partial_dir / f"{res['state']}.parquet", index=False)
        parts.append(res["result"])

    try:
        if workers == 1:
            for i, task in enumerate(tasks, 1):
                res = _intersect_state_worker(task)
                _handle_state_result(res, i)
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_intersect_state_worker, t): t for t in tasks}
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    try:
                        res = future.result()
                    except Exception as exc:
                        task = futures[future]
                        print(f"  [{completed:>2}/{n_tasks}] {task['state']}  ✗ ERRO: {exc}")
                        continue
                    _handle_state_result(res, completed)

    except KeyboardInterrupt:
        interrupted = True
        elapsed_so_far = time.perf_counter() - t_total
        n_done = len(done_states) + len(parts) - len(done_states)
        print(f"\n[INTERROMPIDO] Ctrl+C — {len(parts)}/{n_states} estado(s) concluídos "
              f"[{elapsed_so_far:.0f}s]")
        if not parts:
            print("  Nenhum estado completo — nada a salvar.")
            return
        print("  Rode com --resume para continuar de onde parou.")

    if not parts:
        print("\n  Nenhum resultado — verifique se os GeoPackages têm dados.")
        return

    total_elapsed = time.perf_counter() - t_total
    print(f"\n  Agregando resultados entre estados...", end=" ", flush=True)
    final = aggregate_results(parts)
    print(f"{len(final):,} hexágonos únicos")
    print(f"  sgb_alta_mta_frac > 0.3 : {(final['sgb_alta_mta_frac'] > 0.3).sum():,} hexágonos")
    print(f"  sgb_max_class >= 4      : {(final['sgb_max_class'] >= 4).sum():,} hexágonos")
    if interrupted:
        print(f"  [PARCIAL] {len(parts)} estado(s) de {n_states} — rode com --resume para completar.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_FILES[tipo]
    final.to_parquet(out_path, index=False)
    label = " (parcial)" if interrupted else ""
    print(f"  ✓ Salvo{label}: {out_path}  [{total_elapsed:.0f}s total]")

    if not interrupted and partial_dir.exists():
        shutil.rmtree(partial_dir)


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continua do ponto onde parou (lê parquets intermediários por estado)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Estados processados em paralelo (padrão: 1, recomendado: 4)",
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
        process_tipo(tipo, state_filter=state_filter, dry_run=args.dry_run,
                     resume=args.resume, workers=args.workers)


if __name__ == "__main__":
    main()
