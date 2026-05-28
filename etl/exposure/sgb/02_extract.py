#!/usr/bin/env python3
"""
SGB — Extração e Harmonização por Município
============================================
Lê os ZIPs de raw_zips/, extrai os arquivos brutos e gera um GeoPackage
harmonizado por município, sem simplificação de geometria.

Requer (executar antes):
  - 00_sgb_scraper.py   → 00_sgb_manifest.csv
  - 01_sgb_explore.py   → 01_sgb_inventory.csv + 01_sgb_mapping.json

Outputs (em data/inputs/raw/sgb/por_municipio/{UF}/):
  {sigla_uf}_{nm_mun}_inundacao.gpkg  — dados de inundação harmonizados, sem simplificação
  {sigla_uf}_{nm_mun}_massa.gpkg      — dados de massa harmonizados, sem simplificação

Os arquivos brutos permanecem nos ZIPs originais (raw_zips/).

Colunas dos GPKGs de saída:
  cd_mun        — geocódigo IBGE de 7 dígitos (ex: 3550308)
  nm_municipio  — nome do município (IBGE; fallback: manifest SGB)
  sigla_uf      — sigla da UF (ex: SP)
  cd_uf         — código numérico da UF (ex: 35)
  nm_uf         — nome da UF (ex: São Paulo)
  classe_orig   — valor textual original de suscetibilidade
  classe_num    — inteiro 0-5 (do class_mapping.json; -1 = não mapeado)
  processo      — tipo de processo (do shapefile, quando disponível)
  fonte         — fonte (do shapefile, quando disponível)
  zip_filename  — nome do ZIP de origem (rastreabilidade)

USO:
  python 02_sgb_extract.py               # processa todos os ZIPs
  python 02_sgb_extract.py --resume      # continua do ponto onde parou
  python 02_sgb_extract.py --state SE,BA # filtra por estado
  python 02_sgb_extract.py --limit 5     # testa com 5 ZIPs
  python 02_sgb_extract.py --dry-run     # simula sem escrever arquivos
"""

import json
import csv
import re
import zipfile
import tempfile
import sys
import argparse
import io
import warnings
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon

# ── Paths via config ───────────────────────────────────────────────────────────
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pipeline.py").exists())
sys.path.insert(0, str(_ROOT))
from src import config as cfg  # noqa: E402

DOWNLOAD_DIR    = cfg.RAW_DIR / "sgb/raw_zips"
MANIFEST_PATH   = cfg.RAW_DIR / "sgb/00_sgb_manifest.csv"
INVENTORY_PATH  = cfg.RAW_DIR / "sgb/01_sgb_inventory.csv"
MAPPING_PATH    = cfg.RAW_DIR / "sgb/01_sgb_mapping.json"
POR_MUN_DIR     = cfg.RAW_DIR / "sgb/por_municipio"
MUNICIPIOS_PATH = cfg.MALHA_MUNICIPAL_DIR / "municipios.gpkg"
PROGRESS_FILE   = POR_MUN_DIR / "02_progress.json"
FAILURES_PATH   = cfg.RAW_DIR / "sgb/02_failures.csv"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_log import log_failure, reset_failures  # noqa: E402

TARGET_CRS = cfg.CRS_LATLON  # SIRGAS 2000 geográfico

ENCODINGS_TO_TRY = ["utf-8", "latin-1", "cp1252"]

CLASS_COL_CANDIDATES = [
    "CLASSE", "CLASSE_SU", "Classe", "classe",
    "CLASS", "CLASSIFIC", "SUSCET", "SUSCETIBIL",
]

TIPOS = ("inundacao", "massa")

OUTPUT_COLS = [
    "cd_mun", "nm_municipio", "sigla_uf", "cd_uf", "nm_uf",
    "classe_orig", "classe_num", "processo", "fonte",
    "zip_filename", "geometry",
]

_W: dict = {}  # worker-process globals — preenchido por _init_worker


def _init_worker(mapping: dict, ibge_lookup: dict) -> None:
    _W["mapping"] = mapping
    _W["ibge_lookup"] = ibge_lookup


# ══════════════════════════════════════════════════════════════════════════════
# PROGRESSO / RESUME
# ══════════════════════════════════════════════════════════════════════════════

def _load_progress() -> dict[str, set[str]]:
    """Retorna dict tipo → set de zip_filenames já extraídos para aquele tipo."""
    if not PROGRESS_FILE.exists():
        return {t: set() for t in TIPOS}
    with open(PROGRESS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        # formato antigo: assume ambos os tipos done para os zips listados
        old_done = set(data)
        return {t: set(old_done) for t in TIPOS}
    return {t: set(data.get(t, [])) for t in TIPOS}


def _save_progress(done: dict[str, set[str]]) -> None:
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({t: sorted(v) for t, v in done.items()}, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# CARREGAMENTO
# ══════════════════════════════════════════════════════════════════════════════

def load_mapping() -> dict[str, int]:
    if not MAPPING_PATH.exists():
        print(f"[ERRO] 01_sgb_mapping.json não encontrado: {MAPPING_PATH}")
        print("       Execute 01_sgb_explore.py e revise o mapping antes de continuar.")
        sys.exit(1)
    with open(MAPPING_PATH, encoding="utf-8") as f:
        data = json.load(f)
    mapping = data.get("mapping", data)
    normalized = {
        k: (-1 if v is None else int(v))
        for k, v in mapping.items()
        if not k.startswith("_")
    }
    unmapped = [k for k, v in normalized.items() if v == -1]
    if unmapped:
        print(f"[AVISO] {len(unmapped)} classe(s) com valor -1 em 01_sgb_mapping.json "
              f"(features incluídas com classe_num=-1):")
        for cls in unmapped[:10]:
            print(f"         '{cls}'")
        if len(unmapped) > 10:
            print(f"         … e mais {len(unmapped)-10}")
    return normalized


def load_inventory() -> pd.DataFrame:
    if not INVENTORY_PATH.exists():
        print(f"[ERRO] 01_sgb_inventory.csv não encontrado: {INVENTORY_PATH}")
        print("       Execute 01_sgb_explore.py primeiro.")
        sys.exit(1)
    df = pd.read_csv(INVENTORY_PATH, encoding="utf-8")
    df = df[df["tipo"].isin(TIPOS)].copy()
    skip = df["revisar"].isin(["leitura_erro"]) if "revisar" in df.columns else pd.Series(False, index=df.index)
    if skip.any():
        print(f"[AVISO] {skip.sum()} arquivo(s) com revisar=leitura_erro ignorados.")
        df = df[~skip]
    return df


def load_manifest() -> dict[str, dict]:
    if not MANIFEST_PATH.exists():
        print("[AVISO] Manifest não encontrado — metadados de município não serão adicionados.")
        return {}
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return {row["filename"]: row for row in csv.DictReader(f)}


def load_ibge_municipios() -> dict[str, dict]:
    if not MUNICIPIOS_PATH.exists():
        print(f"[AVISO] municipios.gpkg não encontrado: {MUNICIPIOS_PATH}")
        return {}
    gdf = gpd.read_file(MUNICIPIOS_PATH)[["cd_mun", "nm_mun", "cd_uf", "nm_uf", "sigla_uf"]]
    print(f"  IBGE: {len(gdf):,} municípios carregados.")
    return {
        str(row["cd_mun"]).strip(): {
            "nm_mun":   str(row["nm_mun"]),
            "cd_uf":    str(row["cd_uf"]),
            "nm_uf":    str(row["nm_uf"]),
            "sigla_uf": str(row["sigla_uf"]),
        }
        for _, row in gdf.iterrows()
    }


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS DE GEOMETRIA E METADADOS
# ══════════════════════════════════════════════════════════════════════════════

def _to_multipolygon(geom):
    """Normaliza qualquer geometria para MultiPolygon, mantendo só partes poligonais."""
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return MultiPolygon([geom])
    if geom.geom_type == "MultiPolygon":
        return geom
    if hasattr(geom, "geoms"):
        polys = [g for g in geom.geoms if g.geom_type == "Polygon"]
        return MultiPolygon(polys) if polys else None
    return None


def _apply_multipolygon(geom_series: gpd.GeoSeries) -> gpd.GeoSeries:
    """Aplica _to_multipolygon via list comprehension (mais rápido que .apply())."""
    result = [_to_multipolygon(g) for g in geom_series.values]
    return gpd.GeoSeries(result, index=geom_series.index, crs=geom_series.crs)


def _get_col(gdf: gpd.GeoDataFrame, *candidates: str) -> pd.Series:
    for c in candidates:
        if c in gdf.columns:
            return gdf[c].astype(str).fillna("")
    return pd.Series("", index=gdf.index)


def _apply_class_mapping(
    gdf: gpd.GeoDataFrame,
    classe_col: str,
    mapping: dict[str, int],
) -> gpd.GeoDataFrame:
    gdf["classe_orig"] = gdf[classe_col].astype(str).fillna("")
    gdf["classe_num"]  = gdf["classe_orig"].map(mapping)
    unmapped = gdf[gdf["classe_num"].isna()]["classe_orig"].unique()
    if len(unmapped) > 0:
        print(f"\n    [AVISO] Valores sem mapeamento: {list(unmapped)}")
        print("             Adicione ao 01_sgb_mapping.json e re-execute.")
    gdf["classe_num"] = gdf["classe_num"].fillna(-1).astype(int)
    return gdf


def _add_metadata(
    gdf: gpd.GeoDataFrame,
    zip_path: Path,
    mun_meta: dict,
    ibge_lookup: dict[str, dict],
) -> gpd.GeoDataFrame:
    cd_mun = str(mun_meta.get("cd_mun_ibge", "")).strip()
    ibge   = ibge_lookup.get(cd_mun, {})
    gdf["cd_mun"]       = cd_mun
    gdf["nm_municipio"] = ibge.get("nm_mun") or mun_meta.get("nm_municipio", "")
    gdf["sigla_uf"]     = ibge.get("sigla_uf") or mun_meta.get("cd_estado", "")
    gdf["cd_uf"]        = ibge.get("cd_uf", "")
    gdf["nm_uf"]        = ibge.get("nm_uf", "")
    gdf["zip_filename"] = zip_path.name
    gdf["processo"]     = _get_col(gdf, "PROCESSO", "processo")
    gdf["fonte"]        = _get_col(gdf, "FONTE",    "fonte")
    return gdf


def _clean_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame | None:
    """make_valid + conversão para MultiPolygon + filtragem de vazios."""
    gdf.geometry = gdf.geometry.make_valid()
    gdf.geometry = _apply_multipolygon(gdf.geometry)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf if not gdf.empty else None


# ══════════════════════════════════════════════════════════════════════════════
# PROCESSAMENTO: SHAPEFILE / GPKG
# ══════════════════════════════════════════════════════════════════════════════

def process_shapefile(
    zip_path: Path,
    shp_zip_path: str,
    classe_col_hint: str,
    mapping: dict[str, int],
    mun_meta: dict,
    ibge_lookup: dict[str, dict],
) -> gpd.GeoDataFrame | None:
    is_gpkg = "::" in shp_zip_path
    if is_gpkg:
        gpkg_zip_path, layer_name = shp_zip_path.split("::", 1)
    else:
        stem     = Path(shp_zip_path).stem
        shp_exts = {".shp", ".dbf", ".prj", ".cpg", ".shx"}

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp_path = Path(tmp)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                if is_gpkg:
                    zf.extract(gpkg_zip_path, tmp)
                else:
                    for name in zf.namelist():
                        p = Path(name)
                        if p.suffix.lower() in shp_exts and p.stem == stem:
                            zf.extract(name, tmp)

            if is_gpkg:
                data_files  = list(tmp_path.rglob("*.gpkg"))
                read_kwargs: dict = {"layer": layer_name}
            else:
                data_files  = list(tmp_path.rglob("*.shp"))
                read_kwargs = {}

            if not data_files:
                print("✗ não encontrado após extração")
                return None

            gdf = None
            for enc in ENCODINGS_TO_TRY:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        gdf = gpd.read_file(data_files[0], encoding=enc, **read_kwargs)
                    break
                except Exception:
                    continue

            if gdf is None or gdf.empty:
                print("✗ vazio ou erro de leitura")
                return None

        except Exception as e:
            print(f"✗ {e}")
            return None

    if gdf.crs is None:
        gdf = gdf.set_crs(TARGET_CRS)
    elif gdf.crs.to_epsg() != 4674:
        gdf = gdf.to_crs(TARGET_CRS)

    gdf = _clean_geometry(gdf)
    if gdf is None:
        print("✗ sem geometrias válidas após limpeza")
        return None

    # Detecta coluna de classe
    classe_col = classe_col_hint if classe_col_hint in gdf.columns else None
    if not classe_col:
        lowered = {c.lower(): c for c in gdf.columns}
        for cand in CLASS_COL_CANDIDATES:
            if cand.lower() in lowered:
                classe_col = lowered[cand.lower()]
                break

    if not classe_col:
        # Fallback: infere classe pelo nome do arquivo
        name_lower = Path(shp_zip_path.split("::")[0]).stem.lower().replace("_", "")
        if "corrida" in name_lower:
            inferred_num, inferred_label = 5, "inferido:corrida(MuitoAlta)"
        elif any(kw in name_lower for kw in ["massa", "suscet", "movimento", "desliz", "escorrega", "queda", "fluxo"]):
            inferred_num, inferred_label = 4, "inferido:massa/suscet(Alta)"
        elif any(kw in name_lower for kw in ["enxurr", "inunda"]):
            inferred_num, inferred_label = 4, "inferido:inundacao(Alta)"
        else:
            print("✗ coluna de classe não encontrada e nome não permite inferência")
            return None
        gdf["classe_orig"] = inferred_label
        gdf["classe_num"]  = inferred_num
        print(f"\n    [INFERIDO nome] classe_num={inferred_num}", end="")
        gdf = _add_metadata(gdf, zip_path, mun_meta, ibge_lookup)
        return gdf[[c for c in OUTPUT_COLS if c in gdf.columns]]

    gdf = _apply_class_mapping(gdf, classe_col, mapping)
    gdf = _add_metadata(gdf, zip_path, mun_meta, ibge_lookup)
    return gdf[[c for c in OUTPUT_COLS if c in gdf.columns]]


# ══════════════════════════════════════════════════════════════════════════════
# PROCESSAMENTO: GEOTIFF
# ══════════════════════════════════════════════════════════════════════════════

def process_tif(
    zip_path: Path,
    tif_zip_path: str,
    mapping: dict[str, int],
    mun_meta: dict,
    ibge_lookup: dict[str, dict],
) -> gpd.GeoDataFrame | None:
    try:
        import rasterio
        from rasterio.features import shapes as rasterio_shapes
        import numpy as np
        from shapely.geometry import shape as shapely_shape
    except ImportError:
        print("✗ rasterio não instalado — instale com: pip install rasterio")
        return None

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp_path = Path(tmp)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extract(tif_zip_path, tmp)

            tif_files = list(tmp_path.rglob("*.tif")) + list(tmp_path.rglob("*.tiff"))
            if not tif_files:
                print("✗ tif não encontrado após extração")
                return None

            with rasterio.open(tif_files[0]) as src:
                data      = src.read(1)
                transform = src.transform
                crs       = src.crs
                nodata    = src.nodata

            mask = (data != nodata).astype(np.uint8) if nodata is not None else None
            geoms = [
                {"geometry": shapely_shape(geom), "pixel_value": int(val)}
                for geom, val in rasterio_shapes(data, mask=mask, transform=transform)
                if nodata is None or int(val) != int(nodata)
            ]

            if not geoms:
                print("✗ raster sem dados válidos após poligonização")
                return None

            gdf = gpd.GeoDataFrame(geoms, crs=crs)

        except Exception as e:
            print(f"✗ {e}")
            return None

    if gdf.crs is None:
        gdf = gdf.set_crs(TARGET_CRS)
    elif gdf.crs.to_epsg() != 4674:
        gdf = gdf.to_crs(TARGET_CRS)

    gdf = _clean_geometry(gdf)
    if gdf is None:
        print("✗ sem geometrias válidas após limpeza")
        return None

    gdf["classe_orig"] = gdf["pixel_value"].astype(str)
    gdf["classe_num"]  = gdf["classe_orig"].map(mapping)
    still_unmapped     = gdf["classe_num"].isna()
    gdf.loc[still_unmapped, "classe_num"] = gdf.loc[still_unmapped, "pixel_value"].apply(
        lambda v: v if 0 <= v <= 5 else -1
    )
    gdf["classe_num"] = gdf["classe_num"].fillna(-1).astype(int)

    gdf = _add_metadata(gdf, zip_path, mun_meta, ibge_lookup)
    return gdf[[c for c in OUTPUT_COLS if c in gdf.columns]]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS DE CAMINHO
# ══════════════════════════════════════════════════════════════════════════════

def _mun_slug(nm_mun: str) -> str:
    return re.sub(r"[^\w]", "_", nm_mun.strip(), flags=re.ASCII).strip("_")


# ══════════════════════════════════════════════════════════════════════════════
# WORKER
# ══════════════════════════════════════════════════════════════════════════════

def _do_process_zip(task: dict) -> dict:
    """Processa os tipos de um ZIP: lê SHP/GPKG/TIF e escreve GPKGs de saída."""
    zip_path = Path(task["zip_path"])
    uf_dir   = Path(task["uf_dir"])
    mun_meta = task["mun_meta"]
    nm_mun   = task["nm_mun"]
    sigla_uf = task["sigla_uf"]
    mapping     = _W["mapping"]
    ibge_lookup = _W["ibge_lookup"]

    tipos_ok: list[str] = []
    tipos_err: list[str] = []
    failures:  list[dict] = []   # [{tipo, reason}] — alimenta 02_failures.csv

    for tipo, rows_list in task["tipo_rows"].items():
        gdfs = []
        for row in rows_list:
            file_in_zip  = row["shp_path_in_zip"]
            display_name = file_in_zip if "::" in file_in_zip else Path(file_in_zip).name
            print(f"    → {tipo}: {display_name}", end=" ", flush=True)
            ext = Path(file_in_zip.split("::")[0]).suffix.lower()
            if ext in {".tif", ".tiff"}:
                gdf = process_tif(zip_path, file_in_zip, mapping, mun_meta, ibge_lookup)
            else:
                gdf = process_shapefile(
                    zip_path, file_in_zip,
                    str(row.get("classe_col") or ""),
                    mapping, mun_meta, ibge_lookup,
                )
            if gdf is not None and not gdf.empty:
                gdfs.append(gdf)

        if not gdfs:
            print(f"    ✗ {tipo}: sem geometrias válidas")
            tipos_err.append(tipo)
            failures.append({"tipo": tipo, "reason": "sem geometrias válidas"})
            continue

        combined = pd.concat(gdfs, ignore_index=True)
        combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=TARGET_CRS)

        slug     = _mun_slug(nm_mun)
        out_path = uf_dir / f"{sigla_uf}_{slug}_{tipo}.gpkg"
        uf_dir.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            out_path.unlink()
        try:
            combined.to_file(out_path, driver="GPKG", layer="suscetibilidade")
            print(f"✓ {len(combined)} feições")
            tipos_ok.append(tipo)
        except Exception as e:
            print(f"✗ erro ao escrever: {e}")
            tipos_err.append(tipo)
            failures.append({"tipo": tipo, "reason": f"erro ao escrever: {e}"})

    return {"zip_name": task["zip_name"], "tipos_ok": tipos_ok,
            "tipos_err": tipos_err, "failures": failures}


def _process_zip_worker(task: dict) -> dict:
    """Wrapper que captura stdout para evitar output interleaved em modo paralelo."""
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    t0 = time.perf_counter()
    try:
        result = _do_process_zip(task)
    except Exception as exc:
        result = {
            "zip_name": task["zip_name"],
            "tipos_ok": [],
            "tipos_err": list(task.get("tipo_rows", {}).keys()),
            "fatal": str(exc),
        }
    finally:
        sys.stdout = old_stdout
    result["output"]  = buf.getvalue()
    result["elapsed"] = time.perf_counter() - t0
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTRAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def extract(
    state_filter: list[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    resume: bool = False,
    workers: int = 1,
) -> None:
    mapping     = load_mapping()
    inventory   = load_inventory()
    manifest    = load_manifest()
    ibge_lookup = load_ibge_municipios()

    all_zips = sorted(inventory["zip_filename"].unique())
    zips_with_tipo = {t: set(inventory[inventory["tipo"] == t]["zip_filename"]) for t in TIPOS}

    if state_filter:
        tags     = [f"_{s.lower()}_" for s in state_filter]
        all_zips = [z for z in all_zips if any(t in z.lower() for t in tags)]
    if limit:
        all_zips = all_zips[:limit]

    print(f"\n[EXTRACT] {len(all_zips)} ZIPs | dry_run={dry_run} | resume={resume} | workers={workers}")

    done: dict[str, set[str]] = {t: set() for t in TIPOS}
    if resume:
        done = _load_progress()
        n_done = sum(len(v) for v in done.values())
        print(f"  [RESUME] {n_done} entradas (zip+tipo) já processadas — serão puladas.")
    elif not dry_run:
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
        reset_failures(FAILURES_PATH)

    counts = {"ok": 0, "skip": 0, "err": 0}
    interrupted = False

    # ── Prepara lista de tasks ─────────────────────────────────────────────────
    tasks: list[dict] = []
    for zip_name in all_zips:
        zip_path = DOWNLOAD_DIR / zip_name
        mun_meta_pre = manifest.get(zip_name, {})
        cd_mun_pre   = str(mun_meta_pre.get("cd_mun_ibge", "")).strip()
        ibge_pre     = ibge_lookup.get(cd_mun_pre, {})
        if not zip_path.exists():
            print(f"  SKIP arquivo não encontrado: {zip_name}")
            counts["skip"] += 1
            if not dry_run:
                for tipo in TIPOS:
                    if zip_name in zips_with_tipo[tipo]:
                        log_failure(
                            FAILURES_PATH, stage="extract", tipo=tipo,
                            reason="ZIP não encontrado em raw_zips/",
                            cd_mun=cd_mun_pre,
                            sigla_uf=ibge_pre.get("sigla_uf") or mun_meta_pre.get("cd_estado", ""),
                            nm_municipio=ibge_pre.get("nm_mun") or mun_meta_pre.get("nm_municipio", ""),
                        )
            continue

        mun_meta = mun_meta_pre
        cd_mun   = cd_mun_pre
        ibge     = ibge_pre
        sigla_uf = ibge.get("sigla_uf") or mun_meta.get("cd_estado", "SEM_UF")
        nm_mun   = ibge.get("nm_mun") or mun_meta.get("nm_municipio", cd_mun or zip_name)

        if resume and all(zip_name in done[t] for t in TIPOS if zip_name in zips_with_tipo[t]):
            counts["skip"] += 1
            continue

        rows = inventory[inventory["zip_filename"] == zip_name].copy()
        for tipo_grp in TIPOS:
            mask_tipo = rows["tipo"] == tipo_grp
            has_gpkg  = rows[mask_tipo]["shp_path_in_zip"].str.contains("::", na=False).any()
            if has_gpkg:
                drop = mask_tipo & ~rows["shp_path_in_zip"].str.contains("::", na=False)
                if drop.any():
                    rows = rows[~drop]

        tipo_rows_dict: dict[str, list] = {}
        for tipo in TIPOS:
            t_rows = rows[rows["tipo"] == tipo]
            if t_rows.empty or (resume and zip_name in done[tipo]):
                continue
            tipo_rows_dict[tipo] = (
                t_rows[["shp_path_in_zip", "classe_col"]].fillna("").to_dict("records")
            )

        if not tipo_rows_dict:
            counts["skip"] += 1
            continue

        tasks.append({
            "zip_name": zip_name,
            "zip_path": str(zip_path),
            "uf_dir":   str(POR_MUN_DIR / sigla_uf),
            "mun_meta": mun_meta,
            "nm_mun":   nm_mun,
            "sigla_uf": sigla_uf,
            "tipo_rows": tipo_rows_dict,
            "label":    f"{nm_mun} ({sigla_uf})",
        })

    n_tasks = len(tasks)
    print(f"  {n_tasks} ZIPs a processar, {counts['skip']} já concluídos ou ausentes.")

    if dry_run:
        for task in tasks:
            print(f"  {task['label']} [DRY RUN]")
        counts["ok"] += n_tasks
        print(f"\nEXTRAÇÃO CONCLUÍDA (DRY RUN)\n  ok: {n_tasks:4}  skip: {counts['skip']:4}")
        return

    if not tasks:
        print(f"\n{'═'*60}\nEXTRAÇÃO CONCLUÍDA — nada a processar.")
        return

    def _handle_result(result: dict, task: dict) -> None:
        for tipo in result.get("tipos_ok", []):
            done[tipo].add(result["zip_name"])
        _save_progress(done)
        if result.get("fatal") or (result.get("tipos_err") and not result.get("tipos_ok")):
            counts["err"] += 1
        else:
            counts["ok"] += 1
        # Registra falhas no CSV de rastreamento
        cd_mun_t      = str(task.get("mun_meta", {}).get("cd_mun_ibge", "")).strip()
        sigla_uf_t    = task.get("sigla_uf", "")
        nm_mun_t      = task.get("nm_mun", "")
        if result.get("fatal"):
            for tipo in result.get("tipos_err", TIPOS):
                log_failure(FAILURES_PATH, stage="extract", tipo=tipo,
                            reason=f"fatal: {result['fatal']}",
                            cd_mun=cd_mun_t, sigla_uf=sigla_uf_t,
                            nm_municipio=nm_mun_t)
        else:
            for f in result.get("failures", []):
                log_failure(FAILURES_PATH, stage="extract", tipo=f["tipo"],
                            reason=f["reason"], cd_mun=cd_mun_t,
                            sigla_uf=sigla_uf_t, nm_municipio=nm_mun_t)

    def _print_result(result: dict, idx: int, label: str) -> None:
        elapsed   = result.get("elapsed", 0)
        tipos_ok  = result.get("tipos_ok", [])
        tipos_err = result.get("tipos_err", [])
        ok_str    = "+".join(tipos_ok)  if tipos_ok  else "—"
        err_str   = "  ✗ " + "+".join(tipos_err) if tipos_err else ""
        print(f"  [{idx:>3}/{n_tasks}] {label}  {ok_str}{err_str}  [{elapsed:.1f}s]")
        print(result.get("output", ""), end="")
        if "fatal" in result:
            print(f"    [ERRO FATAL] {result['fatal']}")

    # ── Execução: serial (workers=1) ou paralelo ───────────────────────────────
    t_total = time.perf_counter()
    try:
        if workers == 1:
            _init_worker(mapping, ibge_lookup)
            for i, task in enumerate(tasks, 1):
                result = _process_zip_worker(task)
                _print_result(result, i, task["label"])
                _handle_result(result, task)
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(mapping, ibge_lookup),
            ) as executor:
                futures = {executor.submit(_process_zip_worker, task): task for task in tasks}
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    task = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        print(f"  [{completed:>3}/{n_tasks}] {task['label']} — ERRO: {exc}")
                        counts["err"] += 1
                        cd_mun_t = str(task.get("mun_meta", {}).get("cd_mun_ibge", "")).strip()
                        for tipo in task.get("tipo_rows", {}).keys():
                            log_failure(FAILURES_PATH, stage="extract", tipo=tipo,
                                        reason=f"executor fatal: {exc}",
                                        cd_mun=cd_mun_t, sigla_uf=task.get("sigla_uf", ""),
                                        nm_municipio=task.get("nm_mun", ""))
                        continue
                    _print_result(result, completed, task["label"])
                    _handle_result(result, task)

    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[INTERROMPIDO] Ctrl+C — rode com --resume para continuar.")

    total_elapsed = time.perf_counter() - t_total
    print(f"\n{'═'*60}")
    status = "INTERROMPIDO" if interrupted else "EXTRAÇÃO CONCLUÍDA"
    print(f"{status}  [{total_elapsed:.0f}s total]")
    print(f"  ok: {counts['ok']:4}  skip: {counts['skip']:4}  erros: {counts['err']:4}")
    print(f"  Saída: {POR_MUN_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SGB — Extração e harmonização por município (sem simplificação)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--state",   type=str,  default=None,
                        help="Filtra por estado, ex: SE,BA")
    parser.add_argument("--limit",   type=int,  default=None,
                        help="Limita a N ZIPs (para teste)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simula sem escrever arquivos")
    parser.add_argument("--resume",  action="store_true",
                        help="Continua do ponto onde parou (lê 02_progress.json)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Processos paralelos (padrão: 1, recomendado: 4)")

    args = parser.parse_args()
    state_filter = [s.strip().upper() for s in args.state.split(",")] if args.state else None

    extract(
        state_filter=state_filter,
        limit=args.limit,
        dry_run=args.dry_run,
        resume=args.resume,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
