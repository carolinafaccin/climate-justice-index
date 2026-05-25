#!/usr/bin/env python3
"""
SGB — Consolidação e Simplificação Nacional
============================================
Lê os GPKGs harmonizados por município gerados pelo 02_sgb_extract.py,
aplica simplificação de 5 m (EPSG:5880) e consolida em dois GeoPackages
nacionais.

Requer (executar antes):
  - 02_sgb_extract.py → por_municipio/{UF}/{cd_mun}_{nm}/

Outputs (em data/inputs/raw/sgb/harmonized/):
  03_sgb_floods_br.gpkg  — Inundação de todos os municípios processados
  03_sgb_mass_br.gpkg    — Movimento de Massa de todos os municípios processados

Colunas de saída: idênticas ao 02_sgb_extract.py (ver docstring lá).

Simplificação: 5 m em EPSG:5880 (SIRGAS 2000 / Brazil Polyconic).
  Dentro da precisão nominal do mapeamento 1:25.000 (~10 m).
  O script 04_sgb_h3_intersect.py aplica simplificação adicional de 20 m
  antes da interseção H3.

USO:
  python 03_sgb_harmonize.py               # processa tudo
  python 03_sgb_harmonize.py --resume      # continua do ponto onde parou
  python 03_sgb_harmonize.py --state SE,BA # filtra por estado
  python 03_sgb_harmonize.py --limit 5     # testa com 5 municípios
  python 03_sgb_harmonize.py --dry-run     # simula sem escrever arquivos
  python 03_sgb_harmonize.py --rebuild-progress  # reconstrói progresso lendo os GPKGs
"""

import json
import re
import sys
import argparse
import sqlite3
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon

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

_DATA_DIR     = _load_data_dir()
POR_MUN_DIR   = _DATA_DIR / "inputs/raw/sgb/por_municipio"
OUTPUT_DIR    = _DATA_DIR / "inputs/raw/sgb/harmonized"
PROGRESS_FILE = OUTPUT_DIR / "03_progress.json"
FAILURES_PATH = _DATA_DIR / "inputs/raw/sgb/03_failures.csv"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_log import log_failure, reset_failures  # noqa: E402

TARGET_CRS  = "EPSG:4674"   # SIRGAS 2000 geográfico
CRS_PROJ    = "EPSG:5880"   # SIRGAS 2000 / Brazil Polyconic — para simplificação em metros
SIMPLIFY_M  = 5.0           # tolerância de simplificação em metros

TIPO_TO_FILE = {
    "inundacao": "03_sgb_floods_br.gpkg",
    "massa":     "03_sgb_mass_br.gpkg",
}

TIPOS = tuple(TIPO_TO_FILE.keys())


# ══════════════════════════════════════════════════════════════════════════════
# PROGRESSO / RESUME
# ══════════════════════════════════════════════════════════════════════════════

def _load_progress() -> dict[str, set[str]]:
    """Retorna dict {tipo: set de cd_mun já gravados}."""
    if not PROGRESS_FILE.exists():
        return {t: set() for t in TIPOS}
    with open(PROGRESS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return {t: set(data.get(t, [])) for t in TIPOS}


def _save_progress(progress: dict[str, set[str]]) -> None:
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({t: sorted(v) for t, v in progress.items()}, f, indent=2)


def _slug(text: str) -> str:
    return re.sub(r"[^\w]", "_", text.strip(), flags=re.ASCII).strip("_")


def rebuild_progress() -> dict[str, set[str]]:
    """Reconstrói progresso lendo os GPKGs consolidados via SQLite."""
    progress: dict[str, set[str]] = {t: set() for t in TIPOS}
    for tipo, fname in TIPO_TO_FILE.items():
        out_path = OUTPUT_DIR / fname
        if not out_path.exists():
            print(f"  {fname}: não encontrado, ignorado.")
            continue
        try:
            conn = sqlite3.connect(str(out_path))
            rows = conn.execute(
                "SELECT DISTINCT sigla_uf, nm_municipio FROM suscetibilidade "
                "WHERE sigla_uf IS NOT NULL AND nm_municipio IS NOT NULL"
            ).fetchall()
            conn.close()
            progress[tipo] = {f"{r[0]}_{_slug(r[1])}" for r in rows}
            print(f"  {fname}: {len(progress[tipo])} municípios já processados.")
        except Exception as e:
            print(f"  [ERRO] ao ler {fname}: {e}")
    return progress


# ══════════════════════════════════════════════════════════════════════════════
# COLETA DOS GPKGS POR MUNICÍPIO
# ══════════════════════════════════════════════════════════════════════════════

def _find_mun_gpkgs(tipo: str, state_filter: list[str] | None) -> list[Path]:
    """Retorna todos os GPKGs de um tipo dentro de por_municipio/."""
    paths = sorted(POR_MUN_DIR.glob(f"**/*_{tipo}.gpkg"))
    if state_filter:
        uf_set = {s.upper() for s in state_filter}
        # Com a estrutura atual, o arquivo fica direto em por_municipio/{UF}/
        paths  = [p for p in paths if p.parent.name.upper() in uf_set]
    return paths


def _mun_id_from_stem(stem: str) -> str:
    """Extrai identificador do município a partir do stem do arquivo (sem o sufixo _tipo)."""
    return stem.rsplit("_", 1)[0]


# ══════════════════════════════════════════════════════════════════════════════
# SIMPLIFICAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def _simplify_worker(task: dict) -> dict:
    """Lê e simplifica um GPKG municipal. Chamado por ProcessPoolExecutor."""
    warnings.filterwarnings("ignore")  # suprime RuntimeWarnings do pyogrio/GDAL neste processo
    gpkg_path = Path(task["gpkg_path"])
    mun_id    = task["mun_id"]
    display   = task["display"]
    t0 = time.perf_counter()
    try:
        gdf = gpd.read_file(gpkg_path, layer="suscetibilidade")
        if gdf is None or gdf.empty:
            return {"mun_id": mun_id, "display": display, "gdf": None, "error": "vazio",
                    "elapsed": time.perf_counter() - t0}
        gdf = _simplify(gdf)
        if gdf.empty:
            return {"mun_id": mun_id, "display": display, "gdf": None,
                    "error": "vazio após simplificação", "elapsed": time.perf_counter() - t0}
        gdf.geometry = gdf.geometry.apply(
            lambda g: MultiPolygon([g]) if isinstance(g, Polygon) else g
        )
        return {"mun_id": mun_id, "display": display, "gdf": gdf, "error": None,
                "elapsed": time.perf_counter() - t0}
    except Exception as e:
        return {"mun_id": mun_id, "display": display, "gdf": None, "error": str(e),
                "elapsed": time.perf_counter() - t0}


def _simplify(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Simplifica geometrias a SIMPLIFY_M metros, reprojetando para CRS_PROJ."""
    if gdf.crs is None:
        # arquivos SGB sem CRS definido assumem SIRGAS 2000 (padrão do SGB)
        gdf = gdf.set_crs(TARGET_CRS)
    gdf = gdf.to_crs(CRS_PROJ)
    gdf.geometry = gdf.geometry.simplify(SIMPLIFY_M, preserve_topology=True)
    gdf.geometry = gdf.geometry.make_valid()
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf.to_crs(TARGET_CRS)


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTRAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def harmonize(
    state_filter: list[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    resume: bool = False,
    workers: int = 1,
) -> None:
    if not POR_MUN_DIR.exists():
        print(f"[ERRO] {POR_MUN_DIR} não encontrado.")
        print("       Execute 02_sgb_extract.py primeiro.")
        sys.exit(1)

    progress: dict[str, set[str]] = {t: set() for t in TIPOS}
    if resume:
        progress = _load_progress()
        n_done = sum(len(v) for v in progress.values())
        print(f"  [RESUME] {n_done} combinações (tipo, município) já processadas.")
    elif not dry_run:
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
        reset_failures(FAILURES_PATH)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for fname in TIPO_TO_FILE.values():
            stale = OUTPUT_DIR / fname
            if stale.exists():
                print(f"  Removendo GPKG anterior para evitar duplicação: {fname}")
                stale.unlink()

    written: set[str] = set()
    if resume and not dry_run:
        for tipo, fname in TIPO_TO_FILE.items():
            if (OUTPUT_DIR / fname).exists() and progress[tipo]:
                written.add(fname)

    counts = {tipo: {"ok": 0, "skip": 0, "err": 0} for tipo in TIPOS}
    interrupted = False
    t_total = time.perf_counter()

    try:
        for tipo in TIPOS:
            gpkg_paths = _find_mun_gpkgs(tipo, state_filter)
            if limit:
                gpkg_paths = gpkg_paths[:limit]

            total     = len(gpkg_paths)
            out_fname = TIPO_TO_FILE[tipo]
            out_path  = OUTPUT_DIR / out_fname
            print(f"\n[{tipo.upper()}] {total} municípios encontrados | workers={workers}")

            # Filtra já processados e monta tasks
            tasks = []
            for p in gpkg_paths:
                mun_id = _mun_id_from_stem(p.stem)
                if resume and mun_id in progress[tipo]:
                    counts[tipo]["skip"] += 1
                    continue
                tasks.append({
                    "gpkg_path": str(p),
                    "mun_id":    mun_id,
                    "display":   p.parent.name,
                })

            n_tasks = len(tasks)
            print(f"  {n_tasks} municípios a processar, {counts[tipo]['skip']} já concluídos.")

            if dry_run:
                counts[tipo]["ok"] += n_tasks
                continue
            if not tasks:
                continue

            def _write_result(res: dict, idx: int) -> None:
                elapsed = res.get("elapsed", 0)
                status_prefix = f"  [{idx:>4}/{n_tasks}] {res['display']}"
                if res["error"]:
                    print(f"{status_prefix}  ✗ {res['error']}  [{elapsed:.1f}s]")
                    counts[tipo]["err"] += 1
                    log_failure(FAILURES_PATH, stage="harmonize", tipo=tipo,
                                reason=res["error"], mun_id=res["mun_id"],
                                sigla_uf=res["display"])
                    return
                mode = "w" if out_fname not in written else "a"
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res["gdf"].to_file(out_path, driver="GPKG", layer="suscetibilidade", mode=mode)
                    written.add(out_fname)
                    progress[tipo].add(res["mun_id"])
                    _save_progress(progress)
                    print(f"{status_prefix}  ✓ {len(res['gdf'])} feições  [{elapsed:.1f}s]")
                    counts[tipo]["ok"] += 1
                except Exception as e:
                    print(f"{status_prefix}  ✗ escrita: {e}  [{elapsed:.1f}s]")
                    counts[tipo]["err"] += 1
                    log_failure(FAILURES_PATH, stage="harmonize", tipo=tipo,
                                reason=f"escrita: {e}", mun_id=res["mun_id"],
                                sigla_uf=res["display"])

            if workers == 1:
                for i, task in enumerate(tasks, 1):
                    _write_result(_simplify_worker(task), i)
            else:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = {executor.submit(_simplify_worker, t): t for t in tasks}
                    completed = 0
                    for future in as_completed(futures):
                        completed += 1
                        try:
                            res = future.result()
                        except Exception as exc:
                            task = futures[future]
                            print(f"  [{completed:>4}/{n_tasks}] {task['display']}  ✗ ERRO: {exc}")
                            counts[tipo]["err"] += 1
                            continue
                        _write_result(res, completed)

    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[INTERROMPIDO] Ctrl+C")
        print("  Rode com --resume para continuar.")
        print("  Se suspeitar de duplicação, use --rebuild-progress para reconciliar antes do resume.")

    total_elapsed = time.perf_counter() - t_total
    print(f"\n{'═'*60}")
    status = "INTERROMPIDO" if interrupted else "HARMONIZAÇÃO CONCLUÍDA"
    print(f"{status}{' (DRY RUN)' if dry_run else ''}  [{total_elapsed:.0f}s total]")
    for tipo, c in counts.items():
        out = OUTPUT_DIR / TIPO_TO_FILE[tipo] if not dry_run else "(não escrito)"
        print(f"  {tipo:12}  ok: {c['ok']:4}  skip: {c['skip']:4}  erros: {c['err']:4}  → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SGB — Consolida GPKGs por município em arquivos nacionais (5 m simplificado)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--state",   type=str,  default=None,
                        help="Filtra por estado, ex: SE,BA")
    parser.add_argument("--limit",   type=int,  default=None,
                        help="Limita a N municípios por tipo (para teste)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simula sem escrever arquivos")
    parser.add_argument("--resume",  action="store_true",
                        help="Continua do ponto onde parou (lê 03_progress.json)")
    parser.add_argument("--rebuild-progress", action="store_true",
                        help="Reconstrói 03_progress.json lendo os GPKGs existentes, depois sai")
    parser.add_argument("--workers", type=int, default=1,
                        help="Processos paralelos para leitura+simplificação (padrão: 1, recomendado: 4)")

    args = parser.parse_args()

    if args.rebuild_progress:
        print("\n[REBUILD-PROGRESS] Lendo GPKGs para reconstruir progresso...")
        prog = rebuild_progress()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _save_progress(prog)
        print(f"\n  Progresso salvo em {PROGRESS_FILE}")
        print("  Agora rode: python 03_sgb_harmonize.py --resume")
        return

    state_filter = [s.strip().upper() for s in args.state.split(",")] if args.state else None

    harmonize(
        state_filter=state_filter,
        limit=args.limit,
        dry_run=args.dry_run,
        resume=args.resume,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
