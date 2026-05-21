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
  sgb_inventory.csv            — um registro por shapefile por ZIP
  sgb_review.csv               — apenas os registros que precisam revisão manual
  class_mapping_template.json  — rascunho do mapeamento classe → 0-5
  class_mapping.json           — cópia inicial do template (editar manualmente)

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
import shutil
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
INVENTORY_PATH        = _DATA_DIR / "inputs/raw/sgb/sgb_inventory.csv"
REVIEW_PATH           = _DATA_DIR / "inputs/raw/sgb/sgb_review.csv"
MAPPING_TEMPLATE_PATH = _DATA_DIR / "inputs/raw/sgb/class_mapping_template.json"
MAPPING_PATH          = _DATA_DIR / "inputs/raw/sgb/class_mapping.json"

# ── Classificação por tipo ─────────────────────────────────────────────────────
# Aplicado tanto ao nome do arquivo quanto às pastas pai dentro do ZIP
TIPO_KEYWORDS = {
    "inundacao": ["inundac", "inund"],
    "massa":     ["massa", "movimento", "desliz", "escorrega", "queda", "corrida", "fluxo"],
}

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
    "zip_filename", "shp_path_in_zip", "tipo",
    "n_features", "colunas", "classe_col", "unique_classes", "crs", "notes",
]


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICAÇÃO E DETECÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def classify_shp(shp_path: str) -> str:
    """
    Classifica um SHP como 'inundacao', 'massa' ou 'outros'.
    Verifica primeiro o nome do arquivo, depois os nomes das pastas pai
    (do mais interno para o mais externo), para capturar estruturas como:
      Deslizamento/Suscetibilidade_A.shp  → massa
      Inundacao/Apt_A.shp                 → inundacao
    """
    parts = Path(shp_path).parts  # (pasta_raiz, ..., subpasta, nome.shp)

    # Verifica do nome do arquivo às pastas mais externas
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

def list_zip_shp_structure(zip_path: Path) -> dict[str, list[str]]:
    """
    Retorna {pasta_dentro_do_zip: [arquivos.shp]} para todos os SHPs do ZIP.
    Usado para mostrar estrutura de ZIPs problemáticos.
    """
    folders: dict[str, list[str]] = {}
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                p = Path(name)
                if p.suffix.lower() == ".shp":
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


def scan_zip(zip_path: Path) -> list[dict]:
    """
    Lista todos os SHPs de um ZIP. Lê metadados completos apenas para os
    SHPs classificados como 'inundacao' ou 'massa' — os 'outros' são registrados
    só pelo caminho, sem abrir com geopandas, para evitar warnings e economizar tempo.
    """
    try:
        with zipfile.ZipFile(zip_path) as zf:
            shp_paths = [n for n in zf.namelist() if n.lower().endswith(".shp")]
    except Exception as e:
        return [{
            "zip_filename": zip_path.name, "shp_path_in_zip": "", "tipo": "",
            "n_features": "", "colunas": "", "classe_col": "",
            "unique_classes": "", "crs": "", "notes": f"erro ao abrir ZIP: {e}",
        }]

    if not shp_paths:
        return [{
            "zip_filename": zip_path.name, "shp_path_in_zip": "", "tipo": "vazio",
            "n_features": "", "colunas": "", "classe_col": "",
            "unique_classes": "", "crs": "", "notes": "nenhum .shp encontrado",
        }]

    records = []
    for shp_path in shp_paths:
        tipo = classify_shp(shp_path)
        if tipo in ("inundacao", "massa"):
            meta = read_shp_meta(zip_path, shp_path)
        else:
            meta = {
                "n_features": "", "colunas": "", "classe_col": "",
                "unique_classes": "", "crs": "", "notes": "",
            }
        records.append({
            "zip_filename":    zip_path.name,
            "shp_path_in_zip": shp_path,
            "tipo":            tipo,
            **meta,
        })
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

        if tipo == "inundacao":
            s["has_inundacao"] = True
        elif tipo == "massa":
            s["has_massa"] = True
        elif tipo == "outros" and shp:
            s["outros"].append(shp)

        if note and ("erro" in note.lower() or "falha" in note.lower()):
            s["errors"].append((shp, note))
        elif not r.get("classe_col") and tipo in ("inundacao", "massa"):
            s["sem_classe"].append((shp, tipo))

    return status


def build_mapping_template(records: list[dict]) -> dict:
    """Constrói rascunho de class_mapping.json a partir de todos os valores únicos."""
    all_classes: set[str] = set()
    for r in records:
        if r.get("unique_classes"):
            all_classes.update(v.strip() for v in r["unique_classes"].split("|") if v.strip())

    mapping = {cls: DEFAULT_MAPPING.get(cls, -1) for cls in sorted(all_classes)}

    return {
        "_instrucoes": (
            "Preencha o valor inteiro (0-5) para cada classe. "
            "Valores -1 precisam de revisão manual. "
            "Salve como class_mapping.json no mesmo diretório."
        ),
        "_escala": {
            "5": "Muito Alta", "4": "Alta", "3": "Média / Moderada",
            "2": "Baixa", "1": "Muito Baixa", "0": "Sem suscetibilidade / Área urbana",
        },
        "mapping": mapping,
    }


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

    # ── Seção de problemas ─────────────────────────────────────────────────────
    problems = {
        z: s for z, s in zip_status.items()
        if not s["has_inundacao"] or not s["has_massa"] or s["sem_classe"] or s["errors"]
    }

    if not problems:
        print(f"\n  ✓ Todos os ZIPs têm inundação e massa identificados com coluna de classe.")
        print(f"{'═'*70}\n")
        return

    print(f"\n{'─'*70}")
    print(f"  ⚠  {len(problems)} ZIP(s) PRECISAM DE REVISÃO  →  edite sgb_inventory.csv")
    print(f"     • Coluna 'tipo': mude 'outros' para 'inundacao' ou 'massa'")
    print(f"     • Coluna 'classe_col': preencha o nome da coluna de classe")
    print(f"     • Detalhes também em: sgb_review.csv")
    print(f"{'─'*70}")

    for zip_name, s in sorted(problems.items()):
        issues = []
        if not s["has_inundacao"]: issues.append("sem inundação")
        if not s["has_massa"]:     issues.append("sem massa")
        if s["sem_classe"]:        issues.append(f"{len(s['sem_classe'])} sem coluna de classe")
        if s["errors"]:            issues.append(f"{len(s['errors'])} erro(s)")

        print(f"\n  {zip_name}")
        print(f"  ↳ {', '.join(issues)}")

        if s["outros"]:
            print(f"    SHPs classificados como 'outros' (não identificados):")
            for shp in s["outros"]:
                print(f"      {Path(shp).name}  ({shp})")

        if s["sem_classe"]:
            print(f"    SHPs sem coluna de classe detectada:")
            for shp, tipo in s["sem_classe"]:
                print(f"      [{tipo}]  {Path(shp).name}")

        if s["errors"]:
            print(f"    Erros ao ler:")
            for shp, note in s["errors"]:
                name = Path(shp).name if shp else "(desconhecido)"
                print(f"      {name}: {note}")

        # Mostra estrutura de pastas do ZIP para ajudar na identificação manual
        shp_structure = list_zip_shp_structure(DOWNLOAD_DIR / zip_name)
        if shp_structure:
            print(f"    Estrutura de SHPs no ZIP:")
            for folder in sorted(shp_structure):
                for fname in sorted(shp_structure[folder]):
                    print(f"      {folder}/{fname}")

    print(f"\n{'═'*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# INVENTÁRIO INCREMENTAL
# ══════════════════════════════════════════════════════════════════════════════

def load_existing_inventory() -> tuple[set[str], list[dict]]:
    """
    Carrega o inventário existente (se houver).
    Retorna (conjunto de ZIPs já processados, lista de todos os registros).
    """
    if not INVENTORY_PATH.exists():
        return set(), []
    processed: set[str] = set()
    records: list[dict] = []
    try:
        with open(INVENTORY_PATH, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                processed.add(row["zip_filename"])
                records.append(row)
    except Exception as e:
        print(f"[AVISO] Não foi possível carregar inventário existente: {e}")
        return set(), []
    return processed, records


def save_derived_files(all_records: list[dict]) -> None:
    """Gera (ou atualiza) sgb_review.csv, class_mapping_template.json e class_mapping.json."""
    zip_status = build_zip_status(all_records)

    problem_zips = {
        z for z, s in zip_status.items()
        if not s["has_inundacao"] or not s["has_massa"] or s["sem_classe"] or s["errors"]
    }
    review_records = [r for r in all_records if r["zip_filename"] in problem_zips]
    with open(REVIEW_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INVENTORY_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(review_records)
    print(f"Revisão salva:     {REVIEW_PATH}  ({len(problem_zips)} ZIPs problemáticos)")

    template = build_mapping_template(all_records)
    with open(MAPPING_TEMPLATE_PATH, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    print(f"Template salvo:    {MAPPING_TEMPLATE_PATH}")

    if not MAPPING_PATH.exists():
        shutil.copy(MAPPING_TEMPLATE_PATH, MAPPING_PATH)
        print(f"class_mapping.json criado: {MAPPING_PATH}")
        print("→ Revise class_mapping.json antes de rodar 02_sgb_harmonize.py")

    return zip_status


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

    # ── Carrega progresso anterior ────────────────────────────────────────────
    if args.redo:
        already_done: set[str] = set()
        all_records:  list[dict] = []
    else:
        already_done, all_records = load_existing_inventory()

    zips_to_process = [z for z in zips if z.name not in already_done]

    n_done  = sum(1 for z in zips if z.name in already_done)
    n_total = n_done + len(zips_to_process)

    if already_done and not args.redo:
        print(f"[RETOMADA] {n_done}/{n_total} ZIPs já processados — "
              f"{len(zips_to_process)} restantes")
    else:
        print(f"[EXPLORAÇÃO] {n_total} ZIPs | destino: {INVENTORY_PATH.parent}")

    if args.limit:
        zips_to_process = zips_to_process[:args.limit]
        print(f"             limitando a {args.limit} ZIPs nesta execução")

    print()

    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Escrita incremental: abre CSV em modo append (ou write se novo/redo) ──
    csv_mode = "w" if (args.redo or not INVENTORY_PATH.exists()) else "a"
    write_header = csv_mode == "w"

    interrupted = False
    try:
        with open(INVENTORY_PATH, csv_mode, encoding="utf-8", newline="") as csv_out:
            writer = csv.DictWriter(csv_out, fieldnames=INVENTORY_COLS, extrasaction="ignore")
            if write_header:
                writer.writeheader()

            for i, zip_path in enumerate(zips_to_process, 1):
                global_i = n_done + i
                print(f"  [{global_i:>3}/{n_total}] {zip_path.name}", end="  ", flush=True)

                recs = scan_zip(zip_path)

                counts: dict[str, int] = defaultdict(int)
                for r in recs:
                    counts[r["tipo"]] += 1

                parts = []
                for tipo in ("inundacao", "massa"):
                    n = counts.get(tipo, 0)
                    parts.append(f"{'✓' if n else '✗'}{tipo[:5]}{'×'+str(n) if n > 1 else ''}")
                if counts.get("outros"):
                    parts.append(f"+{counts['outros']} outros")
                if counts.get("vazio"):
                    parts.append("vazio")
                warn = "  ⚠" if not counts.get("inundacao") or not counts.get("massa") else ""
                print("  ".join(parts) + warn)

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

        needs_review = [
            cls for cls, val in build_mapping_template(all_records)["mapping"].items()
            if val == -1
        ]
        if needs_review:
            print(f"⚠  {len(needs_review)} classe(s) com valor -1 no mapping (revisão manual):")
            for cls in needs_review:
                print(f"    '{cls}'")
        else:
            print("✓  Todas as classes mapeadas automaticamente.")
    else:
        n_processed = n_done + len([z for z in zips_to_process if z.name in
                                    {r["zip_filename"] for r in all_records}])
        print(f"\nProgresso: {n_processed}/{n_total} ZIPs processados.")
        print(f"Rode o script novamente para continuar de onde parou.")


if __name__ == "__main__":
    main()
