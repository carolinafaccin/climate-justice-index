#!/usr/bin/env python3
"""
SGB — Exploração e Inventário dos ZIPs
=======================================
Varre todos os ZIPs em raw_zips/, cataloga a estrutura interna de cada arquivo,
detecta shapefiles de Inundação e Movimento de Massa, e coleta:
  - Colunas disponíveis em cada shapefile
  - Coluna de classe detectada
  - Valores únicos da coluna de classe
  - Número de feições e CRS

A classificação por tipo usa tanto o nome do arquivo quanto os nomes das
pastas pai dentro do ZIP, o que aumenta a cobertura para ZIPs com estruturas
não padronizadas.

Outputs (em data/inputs/raw/sgb/):
  01_sgb_inventario.csv  — um registro por shapefile por ZIP; coluna 'revisar' sinaliza o que
                           precisa de atenção manual (filtrar por revisar != '' no Excel)
  01_sgb_cobertura.csv   — uma linha por ZIP: status_zip, has_inundacao, has_massa, classes
  01_sgb_mapeamento.json — mapeamento classe → 0-5 (editar manualmente antes do harmonize)

Coluna 'revisar' do inventário:
  sem_classe   — tipo=inundacao/massa sem coluna de classe detectada (preencher classe_col)
  leitura_erro — arquivo abre mas tem erro de leitura (CRC, magic number)
  zip_erro     — ZIP corrompido ou incompleto (re-baixar com 00_sgb_scraper.py redownload)
  zip_vazio    — ZIP sem arquivos vetoriais/raster
  (vazio)      — nenhuma ação necessária

USO:
  python 01_sgb_explore.py               # explora ZIPs ainda não processados (retoma de onde parou)
  python 01_sgb_explore.py --limit 5     # testa com os próximos 5 ZIPs não processados
  python 01_sgb_explore.py --state SE,BA # filtra por estado (substring no nome do ZIP)
  python 01_sgb_explore.py --redo        # ignora inventário existente e reprocessa tudo

O inventário (sgb_inventory.csv) é gravado a cada ZIP processado. Ctrl+C interrompe
sem perder o progresso — na próxima execução, os ZIPs já inventariados são pulados.
"""

import json
import csv
import zipfile
import tempfile
import sys
import argparse
import warnings
from collections import defaultdict
from pathlib import Path
import geopandas as gpd

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

_DATA_DIR             = _load_data_dir()
DOWNLOAD_DIR          = _DATA_DIR / "inputs/raw/sgb/raw_zips"
INVENTORY_PATH = _DATA_DIR / "inputs/raw/sgb/01_sgb_inventory.csv"
COVERAGE_PATH  = _DATA_DIR / "inputs/raw/sgb/01_sgb_coverage.csv"
MAPPING_PATH   = _DATA_DIR / "inputs/raw/sgb/01_sgb_mapping.json"
MANIFEST_PATH  = _DATA_DIR / "inputs/raw/sgb/00_sgb_manifest.csv"

# ── Classificação por tipo ─────────────────────────────────────────────────────
# Aplicado tanto ao nome do arquivo quanto às pastas pai dentro do ZIP
TIPO_KEYWORDS = {
    "inundacao": ["inundac", "inund", "inuda", "enxurr"],  # inuda = typo, enxurr = enxurrada
    "massa":     ["massa", "movimento", "desliz", "escorrega", "queda", "corrida", "fluxo"],
}

# Palavras-chave que indicam camadas NÃO são de suscetibilidade, mesmo que contenham
# "massa" no nome (ex: corpos d'água). Têm prioridade sobre TIPO_KEYWORDS.
SKIP_AS_OUTROS_KEYWORDS = [
    "massadagua", "massa_dagua", "hid_massa",
    "hidrografia", "corpos_dagua",
]

# Candidatos para coluna de classe (ordem de prioridade)
CLASS_COL_CANDIDATES = [
    "CLASSE", "CLASSE_SU", "Classe", "classe",
    "CLASS", "CLASSIFIC", "SUSCET", "SUSCETIBIL",
]

ENCODINGS_TO_TRY = ["utf-8", "latin-1", "cp1252"]

# Mapeamento padrão: classe textual → valor 0-5
DEFAULT_MAPPING: dict[str, int] = {
    "Muito Alta":  5, "MUITO ALTA":  5, "Muito alta":  5,
    "Alta":        4, "ALTA":        4, "alta":        4,
    "Média":       3, "Media":       3, "MEDIA":       3, "média": 3,
    "M?dia":       3,  # latin-1 lido como ASCII
    "Moderada":    3, "MODERADA":    3, "moderada":    3,
    "Baixa":       2, "BAIXA":       2, "baixa":       2,
    "Muito Baixa": 1, "MUITO BAIXA": 1, "Muito baixa": 1,
    "Sem Suscetibilidade": 0, "Sem suscetibilidade": 0, "Sem suscet": 0,
    "Área Urbana": 0, "Area Urbana": 0, "area urbana": 0,
    "Sem Dado": 0, "sem dado": 0,
}

INVENTORY_COLS = [
    "zip_filename", "cd_estado", "nm_estado", "nm_municipio",
    "shp_path_in_zip", "tipo",
    "n_features", "colunas", "classe_col", "unique_classes", "crs", "notes", "revisar",
]

_LEITURA_ERRO_KWS = ["crc", "magic", "bad", "erro", "falha", "error", "decompres"]

# ZIPs menores que isso quase certamente são downloads incompletos
MIN_ZIP_SIZE_KB = 50


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICAÇÃO E DETECÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def classify_shp(shp_path: str, layer_name: str = "") -> str:
    """
    Classifica um arquivo como 'inundacao', 'massa' ou 'outros'.
    Verifica o nome do arquivo, as pastas pai (mais interno primeiro) e,
    opcionalmente, o nome de uma camada (para arquivos GPKG com múltiplas camadas).
    Exemplos:
      Deslizamento/Suscetibilidade_A.shp       → massa
      Inundacao/Apt_A.shp                       → inundacao
      dados.gpkg  (layer_name="Inundacao_A")   → inundacao
      base.gpkg   (layer_name="HID_Massa_Dagua_A") → outros  (camada hidrográfica)
    """
    parts = list(Path(shp_path).parts)
    if layer_name:
        parts.append(layer_name)

    # Camadas não-susceptibilidade têm prioridade (ex: HID_Massa_Dagua contém "massa")
    all_text = " ".join(parts).lower().replace("_", "")
    if any(kw.replace("_", "") in all_text for kw in SKIP_AS_OUTROS_KEYWORDS):
        return "outros"

    for part in reversed(parts):
        part_lower = part.lower()
        for tipo, keywords in TIPO_KEYWORDS.items():
            if any(kw in part_lower for kw in keywords):
                return tipo

    return "outros"


def detect_class_col(columns: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for cand in CLASS_COL_CANDIDATES:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


# ══════════════════════════════════════════════════════════════════════════════
# LEITURA DE METADADOS
# ══════════════════════════════════════════════════════════════════════════════

VECTOR_RASTER_EXTS = {".shp", ".gpkg", ".tif", ".tiff"}

def list_zip_shp_structure(zip_path: Path) -> dict[str, list[str]]:
    """
    Retorna {pasta_dentro_do_zip: [arquivos]} para todos os SHPs, GPKGs e TIFs do ZIP.
    Usado para mostrar estrutura de ZIPs problemáticos.
    """
    folders: dict[str, list[str]] = {}
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                p = Path(name)
                if p.suffix.lower() in VECTOR_RASTER_EXTS:
                    folder = str(p.parent) if str(p.parent) != "." else "(raiz)"
                    folders.setdefault(folder, []).append(p.name)
    except Exception:
        pass
    return folders


def read_shp_meta(zip_path: Path, shp_zip_path: str) -> dict:
    """Extrai shapefile para diretório temporário e lê metadados."""
    stem = Path(shp_zip_path).stem
    exts = {".shp", ".dbf", ".prj", ".cpg", ".shx"}

    result = {
        "n_features": "", "colunas": "", "classe_col": "",
        "unique_classes": "", "crs": "", "notes": "",
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for name in zf.namelist():
                    p = Path(name)
                    if p.suffix.lower() in exts and p.stem == stem:
                        zf.extract(name, tmp)

            shp_files = list(tmp_path.rglob("*.shp"))
            if not shp_files:
                result["notes"] = "não encontrado após extração"
                return result

            gdf = None
            for enc in ENCODINGS_TO_TRY:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        gdf = gpd.read_file(shp_files[0], encoding=enc)
                    break
                except Exception:
                    continue

            if gdf is None:
                result["notes"] = "falha ao ler com todas as encodings"
                return result

            cols = [c for c in gdf.columns if c != "geometry"]
            classe_col = detect_class_col(cols)

            result["n_features"]     = len(gdf)
            result["colunas"]        = "|".join(cols)
            result["classe_col"]     = classe_col or ""
            result["crs"]            = str(gdf.crs.to_epsg()) if gdf.crs else "desconhecido"
            result["unique_classes"] = (
                "|".join(sorted(str(v) for v in gdf[classe_col].dropna().unique()))
                if classe_col else ""
            )
            if not classe_col:
                result["notes"] = "coluna de classe não detectada"

        except Exception as e:
            result["notes"] = f"erro: {e}"

    return result


def _read_vector_meta(gdf: gpd.GeoDataFrame) -> dict:
    """Extrai metadados comuns de um GeoDataFrame já carregado."""
    cols = [c for c in gdf.columns if c != "geometry"]
    classe_col = detect_class_col(cols)
    return {
        "n_features":     len(gdf),
        "colunas":        "|".join(cols),
        "classe_col":     classe_col or "",
        "crs":            str(gdf.crs.to_epsg()) if gdf.crs else "desconhecido",
        "unique_classes": (
            "|".join(sorted(str(v) for v in gdf[classe_col].dropna().unique()))
            if classe_col else ""
        ),
        "notes": "" if classe_col else "coluna de classe não detectada",
    }


def read_gpkg_meta(zip_path: Path, gpkg_zip_path: str) -> list[tuple[str, str, dict]]:
    """
    Extrai GPKG do ZIP e lê metadados de cada camada.
    Retorna lista de (shp_path_in_zip, tipo, meta_dict).
    O shp_path_in_zip usa o formato "caminho/arquivo.gpkg::NomeCamada".
    """
    import fiona

    empty_meta = {"n_features": "", "colunas": "", "classe_col": "",
                  "unique_classes": "", "crs": "", "notes": ""}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extract(gpkg_zip_path, tmp)

            gpkg_files = list(tmp_path.rglob("*.gpkg"))
            if not gpkg_files:
                tipo = classify_shp(gpkg_zip_path)
                return [(gpkg_zip_path, tipo,
                         {**empty_meta, "notes": "gpkg não encontrado após extração"})]

            gpkg_file = gpkg_files[0]

            try:
                layers = fiona.listlayers(str(gpkg_file))
            except Exception as e:
                tipo = classify_shp(gpkg_zip_path)
                return [(gpkg_zip_path, tipo,
                         {**empty_meta, "notes": f"erro ao listar camadas: {e}"})]

            results = []
            for layer in layers:
                layer_path = f"{gpkg_zip_path}::{layer}"
                # _L: geometria de linha topológica — sem dado de suscetibilidade
                if layer.upper().endswith("_L"):
                    results.append((layer_path, "outros",
                                    {**empty_meta, "notes": "geometria de linha (_L) — sem dado de suscetibilidade"}))
                    continue
                tipo = classify_shp(gpkg_zip_path, layer_name=layer)

                if tipo in ("inundacao", "massa"):
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=RuntimeWarning)
                            gdf = gpd.read_file(str(gpkg_file), layer=layer)
                        meta = _read_vector_meta(gdf)
                    except Exception as e:
                        meta = {**empty_meta, "notes": f"erro: {e}"}
                else:
                    meta = {**empty_meta}

                results.append((layer_path, tipo, meta))

            return results

        except Exception as e:
            tipo = classify_shp(gpkg_zip_path)
            return [(gpkg_zip_path, tipo,
                     {**empty_meta, "notes": f"erro ao ler gpkg: {e}"})]


def read_tif_meta(zip_path: Path, tif_zip_path: str) -> dict:
    """
    Extrai TIF do ZIP e lê metadados básicos (CRS, dimensões, valores únicos se possível).
    Requer rasterio; se não estiver instalado, registra o arquivo sem metadados.
    """
    meta = {"n_features": "raster", "colunas": "", "classe_col": "",
            "unique_classes": "", "crs": "", "notes": ""}

    try:
        import rasterio
    except ImportError:
        meta["notes"] = "raster GeoTIFF — instale rasterio para metadados completos"
        return meta

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extract(tif_zip_path, tmp)

            tif_files = list(tmp_path.rglob("*.tif")) + list(tmp_path.rglob("*.tiff"))
            if not tif_files:
                meta["notes"] = "tif não encontrado após extração"
                return meta

            with rasterio.open(tif_files[0]) as src:
                epsg = src.crs.to_epsg() if src.crs else None
                meta["crs"]     = str(epsg) if epsg else "desconhecido"
                meta["colunas"] = (f"{src.count} banda(s) | "
                                   f"{src.width}×{src.height}px | {src.dtypes[0]}")
                # Tenta ler valores únicos se o raster for pequeno o suficiente
                if src.count == 1 and src.width * src.height < 5_000_000:
                    try:
                        data = src.read(1, masked=True)
                        unique_vals = sorted({int(v) for v in data.compressed()})
                        meta["unique_classes"] = "|".join(str(v) for v in unique_vals[:30])
                    except Exception:
                        pass
                meta["notes"] = f"raster {src.width}×{src.height}px"

        except Exception as e:
            meta["notes"] = f"erro ao ler tif: {e}"

    return meta


def scan_zip(zip_path: Path) -> list[dict]:
    """
    Lista todos os SHPs, GPKGs e TIFs de um ZIP.
    Lê metadados completos apenas para arquivos classificados como 'inundacao'
    ou 'massa' — os 'outros' são registrados só pelo caminho.
    ZIPs corrompidos ou incompletos retornam um registro com tipo='erro'.
    """
    def err_record(note: str) -> list[dict]:
        return [{"zip_filename": zip_path.name, "shp_path_in_zip": "", "tipo": "erro",
                 "n_features": "", "colunas": "", "classe_col": "",
                 "unique_classes": "", "crs": "", "notes": note, "revisar": "zip_erro"}]

    # ── Verifica tamanho antes de tentar abrir ────────────────────────────
    size_kb = zip_path.stat().st_size / 1024
    if size_kb < MIN_ZIP_SIZE_KB:
        return err_record(f"arquivo suspeito: {size_kb:.0f} KB — download incompleto?")

    try:
        with zipfile.ZipFile(zip_path) as zf:
            all_names = zf.namelist()
    except zipfile.BadZipFile as e:
        return err_record(f"ZIP corrompido: {e}")
    except Exception as e:
        return err_record(f"erro ao abrir ZIP: {e}")

    shp_paths  = [n for n in all_names if n.lower().endswith(".shp")]
    gpkg_paths = [n for n in all_names if n.lower().endswith(".gpkg")]
    tif_paths  = [n for n in all_names
                  if n.lower().endswith(".tif") or n.lower().endswith(".tiff")]

    if not shp_paths and not gpkg_paths and not tif_paths:
        return [{
            "zip_filename": zip_path.name, "shp_path_in_zip": "", "tipo": "vazio",
            "n_features": "", "colunas": "", "classe_col": "",
            "unique_classes": "", "crs": "",
            "notes": "nenhum .shp, .gpkg ou .tif encontrado", "revisar": "zip_vazio",
        }]

    empty_meta = {"n_features": "", "colunas": "", "classe_col": "",
                  "unique_classes": "", "crs": "", "notes": ""}
    records = []

    for shp_path in shp_paths:
        # _L: geometria de linha topológica — sem dado de suscetibilidade, ignorar
        if Path(shp_path).stem.upper().endswith("_L"):
            records.append({"zip_filename": zip_path.name, "shp_path_in_zip": shp_path,
                             "tipo": "outros", **empty_meta,
                             "notes": "geometria de linha (_L) — sem dado de suscetibilidade",
                             "revisar": ""})
            continue
        tipo = classify_shp(shp_path)
        if tipo == "outros":
            shp_norm = shp_path.lower().replace("_", "")
            note = ("camada hidrográfica — excluída da suscetibilidade"
                    if any(kw.replace("_", "") in shp_norm for kw in SKIP_AS_OUTROS_KEYWORDS)
                    else "")
            records.append({"zip_filename": zip_path.name, "shp_path_in_zip": shp_path,
                             "tipo": "outros", **empty_meta, "notes": note, "revisar": ""})
        else:
            meta = read_shp_meta(zip_path, shp_path)
            notes_lower = meta.get("notes", "").lower()
            if any(kw in notes_lower for kw in _LEITURA_ERRO_KWS):
                revisar = "leitura_erro"
            elif not meta.get("classe_col"):
                revisar = "sem_classe"
            else:
                revisar = ""
            records.append({"zip_filename": zip_path.name, "shp_path_in_zip": shp_path,
                             "tipo": tipo, **meta, "revisar": revisar})

    for gpkg_path in gpkg_paths:
        for layer_path, tipo, meta in read_gpkg_meta(zip_path, gpkg_path):
            if tipo in ("inundacao", "massa"):
                notes_lower = meta.get("notes", "").lower()
                if any(kw in notes_lower for kw in _LEITURA_ERRO_KWS):
                    revisar = "leitura_erro"
                elif not meta.get("classe_col"):
                    revisar = "sem_classe"
                else:
                    revisar = ""
            else:
                revisar = ""
            records.append({"zip_filename": zip_path.name, "shp_path_in_zip": layer_path,
                             "tipo": tipo, **meta, "revisar": revisar})

    for tif_path in tif_paths:
        tipo = classify_shp(tif_path)
        meta = read_tif_meta(zip_path, tif_path) if tipo in ("inundacao", "massa") else {
            **empty_meta, "n_features": "raster",
            "notes": "raster GeoTIFF (tipo=outros, não lido)",
        }
        records.append({"zip_filename": zip_path.name, "shp_path_in_zip": tif_path,
                         "tipo": tipo, **meta, "revisar": ""})

    return records


# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISE E RELATÓRIO
# ══════════════════════════════════════════════════════════════════════════════

def build_zip_status(records: list[dict]) -> dict[str, dict]:
    """
    Constrói um dict {zip_filename: status} onde status tem:
      has_inundacao: bool
      has_massa:     bool
      outros:        list[shp_path]   — SHPs não classificados
      sem_classe:    list[(shp_path, tipo)] — SHPs sem coluna de classe
      errors:        list[(shp_path, note)]
    """
    status: dict[str, dict] = {}
    for r in records:
        z = r["zip_filename"]
        if z not in status:
            status[z] = {
                "has_inundacao": False,
                "has_massa":     False,
                "outros":        [],
                "sem_classe":    [],
                "errors":        [],
            }
        s = status[z]
        tipo = r.get("tipo", "")
        shp  = r.get("shp_path_in_zip", "")
        note = r.get("notes", "")

        if tipo == "erro":
            s["errors"].append((shp, note))
        elif tipo == "inundacao":
            s["has_inundacao"] = True
            if note and ("erro" in note.lower() or "falha" in note.lower()):
                s["errors"].append((shp, note))
            elif not r.get("classe_col"):
                s["sem_classe"].append((shp, tipo))
        elif tipo == "massa":
            s["has_massa"] = True
            if note and ("erro" in note.lower() or "falha" in note.lower()):
                s["errors"].append((shp, note))
            elif not r.get("classe_col"):
                s["sem_classe"].append((shp, tipo))
        elif tipo == "outros" and shp:
            s["outros"].append(shp)

    return status


def update_class_mapping(records: list[dict]) -> None:
    """
    Atualiza class_mapping.json com classes encontradas nos dados.
    - Adiciona novas chaves com -1 (ou valor do DEFAULT_MAPPING se conhecido)
    - Nunca sobrescreve valores que o usuário já preencheu
    - Normaliza null → -1 (caso o usuário tenha usado null no JSON)
    Salva direto em MAPPING_PATH — não há arquivo template separado.
    """
    # Coleta todas as classes únicas dos registros
    all_classes: set[str] = set()
    for r in records:
        if r.get("unique_classes"):
            all_classes.update(v.strip() for v in r["unique_classes"].split("|") if v.strip())

    # Carrega mapeamento existente (se houver)
    existing_mapping: dict[str, int] = {}
    if MAPPING_PATH.exists():
        try:
            with open(MAPPING_PATH, encoding="utf-8") as f:
                data = json.load(f)
            raw = data.get("mapping", data)
            # Normaliza null → -1 (JSON null vira None em Python)
            existing_mapping = {
                k: (-1 if v is None else int(v))
                for k, v in raw.items()
                if not k.startswith("_")
            }
        except Exception:
            pass

    # Adiciona chaves novas sem tocar nas existentes
    new_classes = [c for c in sorted(all_classes) if c not in existing_mapping]
    for cls in new_classes:
        existing_mapping[cls] = DEFAULT_MAPPING.get(cls, -1)

    merged = dict(sorted(existing_mapping.items()))

    output = {
        "_instrucoes": (
            "Preencha o valor inteiro (0-5) para cada classe. "
            "Valores -1 são incluídos no output com classe_num=-1 (filtráveis downstream)."
        ),
        "_escala": {
            "5": "Muito Alta", "4": "Alta", "3": "Média / Moderada",
            "2": "Baixa", "1": "Muito Baixa", "0": "Sem suscetibilidade / Área urbana",
        },
        "mapping": merged,
    }

    with open(MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    n_new      = len(new_classes)
    n_unmapped = sum(1 for v in merged.values() if v == -1)
    suffix = f"  ({n_new} novas classes adicionadas)" if n_new else ""
    print(f"Mapping atualizado: {MAPPING_PATH}{suffix}")
    if n_unmapped:
        print(f"                    {n_unmapped} classe(s) com valor -1 → revise antes de harmonizar")


def print_summary(records: list[dict], zip_status: dict) -> None:
    """Imprime resumo geral e seção de problemas com instruções de correção."""
    total_zips = len(zip_status)
    ok_zips    = sum(
        1 for s in zip_status.values()
        if s["has_inundacao"] and s["has_massa"] and not s["sem_classe"] and not s["errors"]
    )
    problem_zips = total_zips - ok_zips

    # ── Contagens por tipo ────────────────────────────────────────────────────
    by_tipo: dict = defaultdict(lambda: {"shps": 0, "sem_classe": 0, "classes": set()})
    for r in records:
        t = r.get("tipo", "")
        by_tipo[t]["shps"] += 1
        if not r.get("classe_col") and t in ("inundacao", "massa"):
            by_tipo[t]["sem_classe"] += 1
        if r.get("unique_classes"):
            by_tipo[t]["classes"].update(v for v in r["unique_classes"].split("|") if v)

    print(f"\n{'═'*70}")
    print("  RESUMO DA EXPLORAÇÃO")
    print(f"{'─'*70}")
    print(f"  ZIPs processados:  {total_zips}")
    print(f"  ZIPs completos:    {ok_zips}  ✓")
    print(f"  ZIPs com problemas:{problem_zips}  {'⚠' if problem_zips else ''}")
    print(f"{'─'*70}")
    for tipo in ("inundacao", "massa", "outros", "vazio"):
        if tipo not in by_tipo:
            continue
        d = by_tipo[tipo]
        sem = f"  sem_classe: {d['sem_classe']}" if d["sem_classe"] else ""
        print(f"  {tipo.upper():12}  SHPs: {d['shps']:4}{sem}")
        if d["classes"] and tipo in ("inundacao", "massa"):
            vals = sorted(d["classes"])
            preview = ", ".join(f'"{v}"' for v in vals[:10])
            suffix  = f"  ...+{len(vals)-10}" if len(vals) > 10 else ""
            print(f"               classes: {preview}{suffix}")

    # ── Frequência de colunas por tipo ───────────────────────────────────────
    col_freq: dict[str, dict[str, int]] = {"inundacao": defaultdict(int), "massa": defaultdict(int)}
    shp_count: dict[str, int] = {"inundacao": 0, "massa": 0}
    for r in records:
        t = r.get("tipo", "")
        if t not in col_freq:
            continue
        shp_count[t] += 1
        for col in (r.get("colunas") or "").split("|"):
            col = col.strip()
            if col:
                col_freq[t][col] += 1

    print(f"\n{'─'*70}")
    print("  COLUNAS MAIS FREQUENTES NOS SHAPEFILES")
    for tipo in ("inundacao", "massa"):
        n_shps = shp_count[tipo]
        if n_shps == 0:
            continue
        top = sorted(col_freq[tipo].items(), key=lambda x: -x[1])[:12]
        print(f"\n  {tipo.upper()} (em {n_shps} SHPs):")
        for col, cnt in top:
            pct = 100 * cnt // n_shps
            bar = "█" * (pct // 10)
            print(f"    {col:<20} {cnt:>4}/{n_shps}  {bar}")

    # ── Seção de problemas ─────────────────────────────────────────────────────
    error_zips   = {z for z, s in zip_status.items() if s["errors"]}
    problems = {
        z: s for z, s in zip_status.items()
        if not s["has_inundacao"] or not s["has_massa"] or s["sem_classe"] or s["errors"]
    }

    if not problems:
        print(f"\n  ✓ Todos os ZIPs têm inundação e massa identificados com coluna de classe.")
        print(f"{'═'*70}\n")
        return

    n_corrupt    = len(error_zips)
    n_no_inund   = sum(1 for z, s in zip_status.items()
                       if not s["has_inundacao"] and z not in error_zips)
    n_no_massa   = sum(1 for z, s in zip_status.items()
                       if not s["has_massa"] and z not in error_zips)
    n_sem_classe = sum(1 for s in zip_status.values() if s["sem_classe"])

    print(f"\n{'─'*70}")
    print(f"  ⚠   {len(problems)} ZIP(s) PRECISAM DE REVISÃO")
    print(f"     Filtre 01_sgb_inventario.csv onde  revisar != ''  para ver o que fazer:")
    print(f"     • sem_classe   → preencha coluna 'classe_col'")
    print(f"     • zip_erro     → re-baixe com: python 00_sgb_scraper.py redownload")
    print(f"     • leitura_erro → verifique o arquivo; pode precisar ser re-baixado")
    print(f"     Ou filtre 01_sgb_cobertura.csv onde  status_zip != ok")
    print(f"{'─'*70}")
    if n_corrupt:
        print(f"  Corrompidos (re-baixar):          {n_corrupt:>4}"
              f"  →  python 00_sgb_scraper.py redownload")
    if n_no_inund:
        print(f"  Sem inundação (valid, sem tipo):  {n_no_inund:>4}"
              f"  →  edite coluna 'tipo' em sgb_inventory.csv")
    if n_no_massa:
        print(f"  Sem massa (válido, sem tipo):     {n_no_massa:>4}"
              f"  →  edite coluna 'tipo' em sgb_inventory.csv")
    if n_sem_classe:
        print(f"  Sem coluna de classe detectada:   {n_sem_classe:>4}"
              f"  →  edite coluna 'classe_col' em sgb_inventory.csv")
    print(f"\n{'═'*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# INVENTÁRIO INCREMENTAL
# ══════════════════════════════════════════════════════════════════════════════

def load_manifest() -> dict[str, dict]:
    """Carrega manifest do scraper (opcional) para enriquecer a cobertura com metadados."""
    if not MANIFEST_PATH.exists():
        return {}
    try:
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return {row["filename"]: row for row in csv.DictReader(f)}
    except Exception:
        return {}


def load_existing_inventory() -> tuple[set[str], list[dict], set[str]]:
    """
    Carrega o inventário existente (se houver).
    Retorna (ZIPs OK, registros OK, ZIPs com erro).
    ZIPs com tipo='erro' são excluídos do conjunto de processados para que sejam
    re-escaneados automaticamente após re-download, sem refazer tudo.
    """
    if not INVENTORY_PATH.exists():
        return set(), [], set()
    all_records: list[dict] = []
    errored_zips: set[str] = set()
    try:
        with open(INVENTORY_PATH, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                all_records.append(row)
                if row.get("tipo") == "erro":
                    errored_zips.add(row["zip_filename"])
    except Exception as e:
        print(f"[AVISO] Não foi possível carregar inventário existente: {e}")
        return set(), [], set()
    processed = {r["zip_filename"] for r in all_records if r["zip_filename"] not in errored_zips}
    clean_records = [r for r in all_records if r["zip_filename"] not in errored_zips]
    return processed, clean_records, errored_zips


COVERAGE_COLS = [
    "zip_filename", "cd_estado", "nm_estado", "nm_municipio", "cd_mun_ibge",
    "status_zip", "has_inundacao", "has_massa", "n_inund_shps", "n_massa_shps",
    "inund_classes", "massa_classes", "notes",
]


def build_coverage(records: list[dict], zip_status: dict) -> list[dict]:
    """Uma linha por ZIP com cobertura de tipos e metadados do município."""
    manifest = load_manifest()

    # Conta SHPs e coleta classes por tipo para cada ZIP
    zip_type_info: dict[str, dict] = {}
    for r in records:
        z = r["zip_filename"]
        if z not in zip_type_info:
            zip_type_info[z] = {
                "inundacao": {"count": 0, "classes": set()},
                "massa":     {"count": 0, "classes": set()},
                "errors":    [],
                "has_zip_erro":  False,
                "has_zip_vazio": False,
            }
        t = r.get("tipo", "")
        if t == "erro":
            zip_type_info[z]["has_zip_erro"] = True
        elif t == "vazio":
            zip_type_info[z]["has_zip_vazio"] = True
        elif t in ("inundacao", "massa"):
            zip_type_info[z][t]["count"] += 1
            if r.get("unique_classes"):
                zip_type_info[z][t]["classes"].update(
                    v.strip() for v in r["unique_classes"].split("|") if v.strip()
                )
        if r.get("notes") and ("erro" in r["notes"].lower() or "falha" in r["notes"].lower()):
            zip_type_info[z]["errors"].append(r["notes"])

    rows = []
    for zip_name, info in sorted(zip_type_info.items()):
        meta = manifest.get(zip_name, {})
        if info["has_zip_erro"]:
            status_zip = "zip_erro"
        elif info["has_zip_vazio"]:
            status_zip = "zip_vazio"
        elif info["inundacao"]["count"] == 0 and info["massa"]["count"] == 0:
            status_zip = "sem_cobertura"
        else:
            status_zip = "ok"
        rows.append({
            "zip_filename":  zip_name,
            "cd_estado":     meta.get("cd_estado",    ""),
            "nm_estado":     meta.get("nm_estado",    ""),
            "nm_municipio":  meta.get("nm_municipio", ""),
            "cd_mun_ibge":   meta.get("cd_mun_ibge",  ""),
            "status_zip":    status_zip,
            "has_inundacao": "sim" if info["inundacao"]["count"] > 0 else "não",
            "has_massa":     "sim" if info["massa"]["count"] > 0 else "não",
            "n_inund_shps":  info["inundacao"]["count"],
            "n_massa_shps":  info["massa"]["count"],
            "inund_classes": "|".join(sorted(info["inundacao"]["classes"])),
            "massa_classes":  "|".join(sorted(info["massa"]["classes"])),
            "notes":         "; ".join(info["errors"]),
        })
    return rows


def save_derived_files(all_records: list[dict]) -> dict:
    """Gera (ou atualiza) 01_sgb_cobertura.csv e 01_sgb_mapeamento.json."""
    zip_status = build_zip_status(all_records)

    coverage = build_coverage(all_records, zip_status)
    with open(COVERAGE_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COVERAGE_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(coverage)
    n_ok          = sum(1 for r in coverage if r["status_zip"] == "ok")
    n_sem_cob     = sum(1 for r in coverage if r["status_zip"] == "sem_cobertura")
    n_erro        = sum(1 for r in coverage if r["status_zip"] == "zip_erro")
    n_vazio       = sum(1 for r in coverage if r["status_zip"] == "zip_vazio")
    print(f"Cobertura salva:   {COVERAGE_PATH}  "
          f"({len(coverage)} ZIPs | ok: {n_ok}  sem_cobertura: {n_sem_cob}"
          f"  zip_erro: {n_erro}  zip_vazio: {n_vazio})")

    update_class_mapping(all_records)

    return zip_status


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICAÇÃO DE INTEGRIDADE
# ══════════════════════════════════════════════════════════════════════════════

def verify_zips(zips: list[Path]) -> None:
    """
    Verifica integridade CRC de todos os ZIPs usando zipfile.testzip().
    Lento (lê e descomprime cada arquivo), mas detecta corrupção de dados
    que só aparece na extração. Use com --verify-zips antes de explorar.
    """
    print(f"[VERIFICAÇÃO] {len(zips)} ZIPs — isso pode demorar...\n")
    bad: list[tuple[str, str]] = []

    for i, zip_path in enumerate(zips, 1):
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"  [{i:>3}/{len(zips)}] {zip_path.name} ({size_mb:.1f} MB)", end="  ", flush=True)

        size_kb = zip_path.stat().st_size / 1024
        if size_kb < MIN_ZIP_SIZE_KB:
            msg = f"suspeito: {size_kb:.0f} KB (download incompleto?)"
            print(f"⚠ {msg}")
            bad.append((zip_path.name, msg))
            continue

        try:
            with zipfile.ZipFile(zip_path) as zf:
                first_bad = zf.testzip()
            if first_bad:
                msg = f"CRC inválido em: {first_bad}"
                print(f"✗ {msg}")
                bad.append((zip_path.name, msg))
            else:
                print("✓")
        except zipfile.BadZipFile as e:
            msg = f"ZIP corrompido: {e}"
            print(f"✗ {msg}")
            bad.append((zip_path.name, msg))
        except Exception as e:
            msg = f"erro: {e}"
            print(f"✗ {msg}")
            bad.append((zip_path.name, msg))

    print(f"\n{'═'*70}")
    if bad:
        print(f"  ✗ {len(bad)} ZIP(s) com problema — precisam ser baixados novamente:")
        for name, note in bad:
            print(f"      {name}: {note}")
    else:
        print(f"  ✓ Todos os {len(zips)} ZIPs estão íntegros.")
    print(f"{'═'*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="SGB — Exploração dos ZIPs")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limita a N ZIPs não processados (para teste rápido)")
    parser.add_argument("--state", type=str, default=None,
                        help="Filtra ZIPs por estado, ex: SE,BA (substring no nome do arquivo)")
    parser.add_argument("--redo", action="store_true",
                        help="Ignora inventário existente e reprocessa tudo do zero")
    parser.add_argument("--verify-zips", action="store_true",
                        help="Verifica integridade CRC de todos os ZIPs e sai (lento)")
    args = parser.parse_args()

    if not DOWNLOAD_DIR.exists():
        print(f"[ERRO] Diretório não encontrado: {DOWNLOAD_DIR}")
        sys.exit(1)

    zips = sorted(DOWNLOAD_DIR.glob("*.zip"))

    if args.state:
        states = [f"_{s.strip().lower()}_" for s in args.state.split(",")]
        zips = [z for z in zips if any(s in z.name.lower() for s in states)]

    if not zips:
        print("[AVISO] Nenhum ZIP encontrado. Execute 00_sgb_scraper.py download primeiro.")
        sys.exit(0)

    if args.verify_zips:
        verify_zips(zips)
        sys.exit(0)

    # ── Carrega progresso anterior ────────────────────────────────────────────
    if args.redo:
        already_done: set[str] = set()
        all_records:  list[dict] = []
        errored_zips: set[str] = set()
    else:
        already_done, all_records, errored_zips = load_existing_inventory()
        if errored_zips:
            print(f"[NOTA] {len(errored_zips)} ZIP(s) com erro anterior serão re-escaneados "
                  f"(execute 'python 00_sgb_scraper.py redownload' se ainda não baixou)")
            # Reescreve o inventário sem os registros de erro para que o append
            # posterior não deixe entradas obsoletas duplicadas
            with open(INVENTORY_PATH, "w", encoding="utf-8", newline="") as _fw:
                _w = csv.DictWriter(_fw, fieldnames=INVENTORY_COLS, extrasaction="ignore")
                _w.writeheader()
                _w.writerows(all_records)

    zips_to_process = [z for z in zips if z.name not in already_done]

    n_done  = sum(1 for z in zips if z.name in already_done)
    n_total = n_done + len(zips_to_process)
    n_rescan = sum(1 for z in zips_to_process if z.name in errored_zips)

    if already_done and not args.redo:
        rescan_note = f"  ({n_rescan} re-escaneados)" if n_rescan else ""
        print(f"[RETOMADA] {n_done}/{n_total} ZIPs OK — "
              f"{len(zips_to_process)} restantes{rescan_note}")
    else:
        print(f"[EXPLORAÇÃO] {n_total} ZIPs | destino: {INVENTORY_PATH.parent}")

    if args.limit:
        zips_to_process = zips_to_process[:args.limit]
        print(f"             limitando a {args.limit} ZIPs nesta execução")

    print()

    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Escrita incremental: abre CSV em modo append (ou write se novo/redo) ──
    # Após reescrita dos erros acima, ou se é execução nova/redo, o arquivo
    # já está correto — usamos append para não perder o cabeçalho.
    needs_new = args.redo or not INVENTORY_PATH.exists()
    csv_mode   = "w" if needs_new else "a"
    write_header = needs_new

    mun_index = load_manifest()  # {filename: {cd_estado, nm_estado, nm_municipio, ...}}

    interrupted = False
    try:
        with open(INVENTORY_PATH, csv_mode, encoding="utf-8", newline="") as csv_out:
            writer = csv.DictWriter(csv_out, fieldnames=INVENTORY_COLS, extrasaction="ignore")
            if write_header:
                writer.writeheader()

            for i, zip_path in enumerate(zips_to_process, 1):
                global_i = n_done + i
                name = zip_path.name
                if len(name) > 52:
                    name = name[:49] + "…"
                print(f"  [{global_i:>3}/{n_total}]  {name:<53}", end="", flush=True)

                recs = scan_zip(zip_path)

                # Enriquece cada registro com metadados do município (do manifest)
                mun = mun_index.get(zip_path.name, {})
                for r in recs:
                    r.setdefault("cd_estado",    mun.get("cd_estado",    ""))
                    r.setdefault("nm_estado",    mun.get("nm_estado",    ""))
                    r.setdefault("nm_municipio", mun.get("nm_municipio", ""))

                counts: dict[str, int] = defaultdict(int)
                for r in recs:
                    counts[r["tipo"]] += 1

                if counts.get("erro"):
                    err_note = next(r["notes"] for r in recs if r["tipo"] == "erro")
                    print(f"  ERRO  {err_note}")
                else:
                    parts = []
                    for tipo in ("inundacao", "massa"):
                        n = counts.get(tipo, 0)
                        parts.append(f"{'✓' if n else '✗'}{tipo[:5]}{'×'+str(n) if n > 1 else ''}")
                    if counts.get("outros"):
                        parts.append(f"+{counts['outros']} outros")
                    if counts.get("vazio"):
                        parts.append("vazio")
                    warn = "  ⚠" if not counts.get("inundacao") or not counts.get("massa") else ""
                    print("  " + "  ".join(parts) + warn)

                # Grava imediatamente e força flush para não perder em caso de Ctrl+C
                writer.writerows(recs)
                csv_out.flush()
                all_records.extend(recs)

    except KeyboardInterrupt:
        interrupted = True
        print("\n[INTERROMPIDO] Progresso salvo. Rode novamente para continuar.")

    # ── Arquivos derivados (sempre gerados com o que foi coletado até agora) ──
    print(f"\nInventário salvo:  {INVENTORY_PATH}  ({len(all_records)} registros)")
    zip_status = save_derived_files(all_records)

    if not interrupted:
        print_summary(all_records, zip_status)
        # O resumo de classes com -1 já é impresso por update_class_mapping() acima
    else:
        n_processed = n_done + len([z for z in zips_to_process if z.name in
                                    {r["zip_filename"] for r in all_records}])
        print(f"\nProgresso: {n_processed}/{n_total} ZIPs processados.")
        print(f"Rode o script novamente para continuar de onde parou.")


if __name__ == "__main__":
    main()
