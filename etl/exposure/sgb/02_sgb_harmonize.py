#!/usr/bin/env python3
"""
SGB — Harmonização dos Shapefiles de Suscetibilidade
=====================================================
Lê os ZIPs de raw_zips/, extrai os shapefiles de Inundação e
Movimento de Massa, mapeia os valores de CLASSE para a escala 0-5,
e consolida tudo em dois GeoPackages.

Requer (executar antes):
  - 00_sgb_scraper.py      → sgb_download_manifest.csv
  - 01_sgb_explore.py      → sgb_inventory.csv + class_mapping.json (editar se necessário)

Outputs (em data/inputs/raw/sgb/harmonized/):
  sgb_inundacoes_br.gpkg  — Inundação de todos os municípios processados
  sgb_massa_br.gpkg       — Movimento de Massa de todos os municípios processados

Colunas de saída:
  nm_municipio, cd_estado, cd_mun_ibge   — do manifest
  classe_orig                             — valor textual original
  classe_num                              — inteiro 0-5 (do class_mapping.json)
  processo, fonte                         — do shapefile
  zip_filename                            — rastreabilidade

USO:
  python 02_sgb_harmonize.py               # processa todos os ZIPs
  python 02_sgb_harmonize.py --state SE,BA # filtra por estado
  python 02_sgb_harmonize.py --limit 5     # testa com 5 ZIPs
  python 02_sgb_harmonize.py --dry-run     # simula sem escrever arquivos
"""

import json
import csv
import zipfile
import tempfile
import sys
import argparse
from pathlib import Path
import geopandas as gpd
import pandas as pd

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
DOWNLOAD_DIR   = _DATA_DIR / "inputs/raw/sgb/raw_zips"
MANIFEST_PATH  = _DATA_DIR / "inputs/raw/sgb/sgb_download_manifest.csv"
INVENTORY_PATH = _DATA_DIR / "inputs/raw/sgb/sgb_inventory.csv"
MAPPING_PATH   = _DATA_DIR / "inputs/raw/sgb/class_mapping.json"
OUTPUT_DIR     = _DATA_DIR / "inputs/raw/sgb/harmonized"

TARGET_CRS = "EPSG:4674"  # SIRGAS 2000 geográfico

ENCODINGS_TO_TRY = ["utf-8", "latin-1", "cp1252"]

CLASS_COL_CANDIDATES = [
    "CLASSE", "CLASSE_SU", "Classe", "classe",
    "CLASS", "CLASSIFIC", "SUSCET", "SUSCETIBIL",
]

OUTPUT_COLS = [
    "nm_municipio", "cd_estado", "cd_mun_ibge",
    "classe_orig", "classe_num", "processo", "fonte",
    "zip_filename", "geometry",
]

TIPO_TO_FILE = {
    "inundacao": "sgb_inundacoes_br.gpkg",
    "massa":     "sgb_massa_br.gpkg",
}


# ══════════════════════════════════════════════════════════════════════════════
# CARREGAMENTO
# ══════════════════════════════════════════════════════════════════════════════

def load_mapping() -> dict[str, int]:
    if not MAPPING_PATH.exists():
        print(f"[ERRO] class_mapping.json não encontrado: {MAPPING_PATH}")
        print("       Execute 01_sgb_explore.py e revise o mapping antes de continuar.")
        sys.exit(1)
    with open(MAPPING_PATH) as f:
        data = json.load(f)
    # Suporta tanto {"mapping": {...}} quanto {"chave": valor} direto
    mapping = data.get("mapping", data)
    unmapped = [k for k, v in mapping.items() if v == -1]
    if unmapped:
        print(f"[AVISO] {len(unmapped)} classe(s) com valor -1 em class_mapping.json:")
        for cls in unmapped:
            print(f"         '{cls}' → -1  (feature será incluída com classe_num=-1)")
    return {k: int(v) for k, v in mapping.items() if not k.startswith("_")}


def load_inventory() -> pd.DataFrame:
    if not INVENTORY_PATH.exists():
        print(f"[ERRO] sgb_inventory.csv não encontrado: {INVENTORY_PATH}")
        print("       Execute 01_sgb_explore.py primeiro.")
        sys.exit(1)
    df = pd.read_csv(INVENTORY_PATH)
    return df[df["tipo"].isin(TIPO_TO_FILE.keys())].copy()


def load_manifest() -> dict[str, dict]:
    """Retorna dict {filename_zip: {nm_municipio, cd_estado, cd_mun_ibge, ...}}"""
    if not MANIFEST_PATH.exists():
        print("[AVISO] Manifest não encontrado — metadados de município não serão adicionados.")
        return {}
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return {row["filename"]: row for row in csv.DictReader(f)}


# ══════════════════════════════════════════════════════════════════════════════
# PROCESSAMENTO DE UM SHAPEFILE
# ══════════════════════════════════════════════════════════════════════════════

def _get_col(gdf: gpd.GeoDataFrame, *candidates: str) -> pd.Series:
    """Retorna a primeira coluna encontrada dentre os candidatos, ou série vazia."""
    for c in candidates:
        if c in gdf.columns:
            return gdf[c].astype(str).fillna("")
    return pd.Series("", index=gdf.index)


def process_shapefile(
    zip_path: Path,
    shp_zip_path: str,
    classe_col_hint: str,
    mapping: dict[str, int],
    mun_meta: dict,
) -> gpd.GeoDataFrame | None:
    """
    Extrai um shapefile do ZIP, aplica mapeamento de classe e padroniza colunas.
    Retorna None em caso de erro ou shapefile vazio.
    """
    stem = Path(shp_zip_path).stem
    exts = {".shp", ".dbf", ".prj", ".cpg", ".shx"}

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
                print("✗ não encontrado após extração")
                return None

            gdf = None
            for enc in ENCODINGS_TO_TRY:
                try:
                    gdf = gpd.read_file(shp_files[0], encoding=enc)
                    break
                except Exception:
                    continue

            if gdf is None or gdf.empty:
                print("✗ vazio ou erro de leitura")
                return None

        except Exception as e:
            print(f"✗ {e}")
            return None

    # ── CRS ───────────────────────────────────────────────────────────────
    if gdf.crs is None:
        gdf = gdf.set_crs(TARGET_CRS)
    elif gdf.crs.to_epsg() != 4674:
        gdf = gdf.to_crs(TARGET_CRS)

    # ── Coluna de classe ──────────────────────────────────────────────────
    classe_col = classe_col_hint if classe_col_hint in gdf.columns else None
    if not classe_col:
        lowered = {c.lower(): c for c in gdf.columns}
        for cand in CLASS_COL_CANDIDATES:
            if cand.lower() in lowered:
                classe_col = lowered[cand.lower()]
                break

    if not classe_col:
        print("✗ coluna de classe não encontrada")
        return None

    # ── Mapeamento ────────────────────────────────────────────────────────
    gdf["classe_orig"] = gdf[classe_col].astype(str).fillna("")
    gdf["classe_num"]  = gdf["classe_orig"].map(mapping)

    unmapped_vals = gdf[gdf["classe_num"].isna()]["classe_orig"].unique()
    if len(unmapped_vals) > 0:
        print(f"\n    [AVISO] Valores sem mapeamento: {list(unmapped_vals)}")
        print("             Adicione ao class_mapping.json e re-execute.")

    gdf["classe_num"] = gdf["classe_num"].fillna(-1).astype(int)

    # ── Metadados ─────────────────────────────────────────────────────────
    gdf["nm_municipio"] = mun_meta.get("nm_municipio", "")
    gdf["cd_estado"]    = mun_meta.get("cd_estado",    "")
    gdf["cd_mun_ibge"]  = mun_meta.get("cd_mun_ibge",  "")
    gdf["zip_filename"] = zip_path.name

    gdf["processo"] = _get_col(gdf, "PROCESSO", "processo")
    gdf["fonte"]    = _get_col(gdf, "FONTE",    "fonte")

    return gdf[[c for c in OUTPUT_COLS if c in gdf.columns]]


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTRAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def harmonize(
    state_filter: list[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> None:
    mapping   = load_mapping()
    inventory = load_inventory()
    manifest  = load_manifest()

    # Lista de ZIPs a processar (derivada do inventory)
    all_zips = sorted(inventory["zip_filename"].unique())

    if state_filter:
        tags = [f"_{s.lower()}_" for s in state_filter]
        all_zips = [z for z in all_zips if any(t in z.lower() for t in tags)]

    if limit:
        all_zips = all_zips[:limit]

    total = len(all_zips)
    print(f"\n[HARMONIZE] {total} ZIPs | dry_run={dry_run}")
    if not dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Rastreia se já escrevemos cada arquivo (para alternar mode w/a)
    written: set[str] = set()
    counts = {tipo: {"ok": 0, "skip": 0, "err": 0} for tipo in TIPO_TO_FILE}

    for i, zip_name in enumerate(all_zips, 1):
        zip_path = DOWNLOAD_DIR / zip_name
        if not zip_path.exists():
            print(f"  [{i:>3}/{total}] SKIP arquivo não encontrado: {zip_name}")
            continue

        mun_meta = manifest.get(zip_name, {})
        label = (f"{mun_meta.get('nm_municipio', '?')} "
                 f"({mun_meta.get('cd_estado', '?')})")
        print(f"  [{i:>3}/{total}] {label}")

        rows = inventory[inventory["zip_filename"] == zip_name]
        for _, row in rows.iterrows():
            tipo = row["tipo"]
            print(f"    → {tipo}: {Path(row['shp_path_in_zip']).name}", end=" ", flush=True)

            if dry_run:
                print("[DRY RUN]")
                continue

            gdf = process_shapefile(
                zip_path,
                row["shp_path_in_zip"],
                str(row.get("classe_col", "") or ""),
                mapping,
                mun_meta,
            )

            if gdf is None or gdf.empty:
                counts[tipo]["err"] += 1
                continue

            out_path = OUTPUT_DIR / TIPO_TO_FILE[tipo]
            mode = "w" if out_path.name not in written else "a"
            try:
                gdf.to_file(out_path, driver="GPKG", layer="suscetibilidade", mode=mode)
                written.add(out_path.name)
                print(f"✓ {len(gdf)} feições")
                counts[tipo]["ok"] += 1
            except Exception as e:
                print(f"✗ erro ao escrever: {e}")
                counts[tipo]["err"] += 1

    # ── Resumo ────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("HARMONIZAÇÃO CONCLUÍDA" + (" (DRY RUN)" if dry_run else ""))
    for tipo, c in counts.items():
        out = OUTPUT_DIR / TIPO_TO_FILE[tipo] if not dry_run else "(não escrito)"
        print(f"  {tipo:12}  ok: {c['ok']:4}  erros: {c['err']:4}  → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SGB — Harmonização dos shapefiles de suscetibilidade",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--state",   type=str,  default=None,
                        help="Filtra por estado, ex: SE,BA (substring no nome do ZIP)")
    parser.add_argument("--limit",   type=int,  default=None,
                        help="Limita a N ZIPs (para teste)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simula o processamento sem escrever arquivos")

    args = parser.parse_args()
    state_filter = [s.strip().upper() for s in args.state.split(",")] if args.state else None

    harmonize(
        state_filter=state_filter,
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
