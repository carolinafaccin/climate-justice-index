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

Outputs (em data/inputs/raw/sgb/):
  sgb_inventory.csv            — um registro por shapefile por ZIP
  class_mapping_template.json  — rascunho do mapeamento classe → 0-5
  class_mapping.json           — cópia inicial do template (editar manualmente)

USO:
  python 01_sgb_explore.py               # explora todos os ZIPs baixados
  python 01_sgb_explore.py --limit 5     # testa com os primeiros 5 ZIPs
  python 01_sgb_explore.py --state SE,BA # filtra por estado (substring no nome do ZIP)
"""

import json
import csv
import zipfile
import tempfile
import shutil
import sys
import argparse
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
MAPPING_TEMPLATE_PATH = _DATA_DIR / "inputs/raw/sgb/class_mapping_template.json"
MAPPING_PATH          = _DATA_DIR / "inputs/raw/sgb/class_mapping.json"

# ── Classificação de shapefiles por tipo ──────────────────────────────────────
TIPO_KEYWORDS = {
    "inundacao": ["inundac", "inund"],
    "massa":     ["massa", "movimento", "desliz", "escorrega", "queda", "corrida"],
}

# Candidatos para a coluna de classe (ordem de prioridade)
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
    "M?dia":       3,  # latin-1 lido incorretamente como ASCII
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
# FUNÇÕES AUXILIARES
# ══════════════════════════════════════════════════════════════════════════════

def classify_shp(shp_path: str) -> str:
    name = Path(shp_path).name.lower()
    for tipo, keywords in TIPO_KEYWORDS.items():
        if any(kw in name for kw in keywords):
            return tipo
    return "outros"


def detect_class_col(columns: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for cand in CLASS_COL_CANDIDATES:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def read_shp_meta(zip_path: Path, shp_zip_path: str) -> dict:
    """Extrai shapefile para diretório temporário e lê metadados sem reter em memória."""
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
    """Lista todos os shapefiles de um ZIP e lê os metadados de cada um."""
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
        meta = read_shp_meta(zip_path, shp_path)
        records.append({
            "zip_filename":    zip_path.name,
            "shp_path_in_zip": shp_path,
            "tipo":            tipo,
            **meta,
        })
    return records


def build_mapping_template(records: list[dict]) -> dict:
    """Constrói um rascunho de class_mapping.json a partir de todos os valores únicos."""
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
            "5": "Muito Alta",
            "4": "Alta",
            "3": "Média / Moderada",
            "2": "Baixa",
            "1": "Muito Baixa",
            "0": "Sem suscetibilidade / Área urbana",
        },
        "mapping": mapping,
    }


def print_summary(records: list[dict]) -> None:
    from collections import defaultdict

    by_tipo: dict = defaultdict(lambda: {"zips": set(), "shps": 0, "sem_classe": 0, "classes": set()})
    for r in records:
        t = r["tipo"]
        by_tipo[t]["zips"].add(r["zip_filename"])
        by_tipo[t]["shps"] += 1
        if not r.get("classe_col"):
            by_tipo[t]["sem_classe"] += 1
        if r.get("unique_classes"):
            by_tipo[t]["classes"].update(
                v for v in r["unique_classes"].split("|") if v
            )

    print(f"\n{'═'*65}")
    print("  RESUMO DA EXPLORAÇÃO")
    print(f"{'─'*65}")
    for tipo in sorted(by_tipo):
        d = by_tipo[tipo]
        print(f"  {tipo.upper():12}  ZIPs: {len(d['zips']):4}  "
              f"SHPs: {d['shps']:4}  sem_classe: {d['sem_classe']:4}")
        if d["classes"]:
            vals = sorted(d["classes"])
            preview = ", ".join(f'"{v}"' for v in vals[:12])
            suffix = f"  ... +{len(vals)-12}" if len(vals) > 12 else ""
            print(f"               classes: {preview}{suffix}")
    print(f"{'═'*65}\n")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="SGB — Exploração dos ZIPs")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limita a N ZIPs (para teste rápido)")
    parser.add_argument("--state", type=str, default=None,
                        help="Filtra ZIPs por estado, ex: SE,BA (substring no nome do arquivo)")
    args = parser.parse_args()

    if not DOWNLOAD_DIR.exists():
        print(f"[ERRO] Diretório não encontrado: {DOWNLOAD_DIR}")
        sys.exit(1)

    zips = sorted(DOWNLOAD_DIR.glob("*.zip"))

    if args.state:
        states = [f"_{s.strip().lower()}_" for s in args.state.split(",")]
        zips = [z for z in zips if any(s in z.name.lower() for s in states)]

    if args.limit:
        zips = zips[:args.limit]

    if not zips:
        print("[AVISO] Nenhum ZIP encontrado. Execute 00_sgb_scraper.py download primeiro.")
        sys.exit(0)

    print(f"[EXPLORAÇÃO] {len(zips)} ZIPs | destino: {INVENTORY_PATH.parent}\n")

    all_records: list[dict] = []
    for i, zip_path in enumerate(zips, 1):
        print(f"  [{i:>3}/{len(zips)}] {zip_path.name}", end=" ... ", flush=True)
        recs = scan_zip(zip_path)
        tipos = [r["tipo"] for r in recs]
        print(f"{len(recs)} SHP(s): {tipos}")
        all_records.extend(recs)

    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(INVENTORY_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INVENTORY_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\nInventário salvo:  {INVENTORY_PATH}")

    template = build_mapping_template(all_records)
    with open(MAPPING_TEMPLATE_PATH, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    print(f"Template salvo:    {MAPPING_TEMPLATE_PATH}")

    if not MAPPING_PATH.exists():
        shutil.copy(MAPPING_TEMPLATE_PATH, MAPPING_PATH)
        print(f"class_mapping.json criado: {MAPPING_PATH}")
        print("→ Revise e edite class_mapping.json antes de rodar 02_sgb_harmonize.py")

    print_summary(all_records)

    needs_review = [cls for cls, val in template["mapping"].items() if val == -1]
    if needs_review:
        print(f"⚠  {len(needs_review)} classe(s) com valor -1 (precisam revisão manual):")
        for cls in needs_review:
            print(f"    '{cls}'")
    else:
        print("✓  Todas as classes mapeadas automaticamente.")


if __name__ == "__main__":
    main()
