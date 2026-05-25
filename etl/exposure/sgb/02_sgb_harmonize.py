#!/usr/bin/env python3
"""
SGB — Harmonização dos Dados de Suscetibilidade
================================================
Lê os ZIPs de raw_zips/, extrai os dados de Inundação e Movimento
de Massa (SHP, GPKG ou GeoTIFF), mapeia os valores de CLASSE para
a escala 0-5, e consolida tudo em dois GeoPackages.

Requer (executar antes):
  - 00_sgb_scraper.py      → 00_sgb_manifest.csv
  - 01_sgb_explore.py      → 01_sgb_inventory.csv + 01_sgb_mapping.json (editar se necessário)

Outputs (em data/inputs/raw/sgb/harmonized/):
  02_sgb_floods_br.gpkg  — Inundação de todos os municípios processados
  02_sgb_mass_br.gpkg    — Movimento de Massa de todos os municípios processados

Colunas de saída:
  cd_mun        — geocódigo IBGE de 7 dígitos (ex: 3550308)
  nm_municipio  — nome do município (IBGE; fallback: manifest SGB)
  sigla_uf      — sigla da UF (ex: SP)  — padrão IBGE
  cd_uf         — código numérico da UF (ex: 35)
  nm_uf         — nome da UF (ex: São Paulo)
  classe_orig   — valor textual original de suscetibilidade
  classe_num    — inteiro 0-5 (do class_mapping.json; -1 = não mapeado)
  processo      — tipo de processo (do shapefile, quando disponível)
  fonte         — fonte (do shapefile, quando disponível)
  zip_filename  — nome do ZIP de origem (rastreabilidade)

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
import warnings
from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon

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
MANIFEST_PATH  = _DATA_DIR / "inputs/raw/sgb/00_sgb_manifest.csv"
INVENTORY_PATH = _DATA_DIR / "inputs/raw/sgb/01_sgb_inventory.csv"
MAPPING_PATH   = _DATA_DIR / "inputs/raw/sgb/01_sgb_mapping.json"
OUTPUT_DIR      = _DATA_DIR / "inputs/raw/sgb/harmonized"
MUNICIPIOS_PATH = _DATA_DIR / "inputs/raw/ibge/malha_municipal/2024/municipios.gpkg"

TARGET_CRS = "EPSG:4674"  # SIRGAS 2000 geográfico

ENCODINGS_TO_TRY = ["utf-8", "latin-1", "cp1252"]

CLASS_COL_CANDIDATES = [
    "CLASSE", "CLASSE_SU", "Classe", "classe",
    "CLASS", "CLASSIFIC", "SUSCET", "SUSCETIBIL",
]

OUTPUT_COLS = [
    "cd_mun", "nm_municipio", "sigla_uf", "cd_uf", "nm_uf",
    "classe_orig", "classe_num", "processo", "fonte",
    "zip_filename", "geometry",
]

TIPO_TO_FILE = {
    "inundacao": "02_sgb_floods_br.gpkg",
    "massa":     "02_sgb_mass_br.gpkg",
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
    # null no JSON vira None em Python; normaliza para -1
    normalized = {
        k: (-1 if v is None else int(v))
        for k, v in mapping.items()
        if not k.startswith("_")
    }
    unmapped = [k for k, v in normalized.items() if v == -1]
    if unmapped:
        print(f"[AVISO] {len(unmapped)} classe(s) com valor -1 em class_mapping.json "
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
    df = pd.read_csv(INVENTORY_PATH)
    df = df[df["tipo"].isin(TIPO_TO_FILE.keys())].copy()
    skip = df["revisar"].isin(["leitura_erro"]) if "revisar" in df.columns else pd.Series(False, index=df.index)
    if skip.any():
        print(f"[AVISO] {skip.sum()} arquivo(s) com revisar=leitura_erro ignorados (re-baixe os ZIPs).")
        df = df[~skip]
    return df


def load_manifest() -> dict[str, dict]:
    """Retorna dict {filename_zip: {nm_municipio, cd_estado, cd_mun_ibge, ...}}"""
    if not MANIFEST_PATH.exists():
        print("[AVISO] Manifest não encontrado — metadados de município não serão adicionados.")
        return {}
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return {row["filename"]: row for row in csv.DictReader(f)}


def load_ibge_municipios() -> dict[str, dict]:
    """Carrega lookup IBGE: cd_mun (7 dígitos) → {nm_mun, cd_uf, nm_uf, sigla_uf}."""
    if not MUNICIPIOS_PATH.exists():
        print(f"[AVISO] municipios.gpkg não encontrado: {MUNICIPIOS_PATH}")
        print("        Metadados IBGE (nm_uf, cd_uf, sigla_uf) não serão adicionados.")
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
# PROCESSAMENTO DE UM SHAPEFILE
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
        polys = []
        for g in geom.geoms:
            if g.geom_type == "Polygon":
                polys.append(g)
            elif g.geom_type == "MultiPolygon":
                polys.extend(g.geoms)
        return MultiPolygon(polys) if polys else None
    return None


def _get_col(gdf: gpd.GeoDataFrame, *candidates: str) -> pd.Series:
    """Retorna a primeira coluna encontrada dentre os candidatos, ou série vazia."""
    for c in candidates:
        if c in gdf.columns:
            return gdf[c].astype(str).fillna("")
    return pd.Series("", index=gdf.index)


def _apply_class_mapping(
    gdf: gpd.GeoDataFrame,
    classe_col: str,
    mapping: dict[str, int],
) -> gpd.GeoDataFrame:
    """Aplica mapeamento de classe ao GeoDataFrame; avisa sobre valores não mapeados."""
    gdf["classe_orig"] = gdf[classe_col].astype(str).fillna("")
    gdf["classe_num"]  = gdf["classe_orig"].map(mapping)
    unmapped = gdf[gdf["classe_num"].isna()]["classe_orig"].unique()
    if len(unmapped) > 0:
        print(f"\n    [AVISO] Valores sem mapeamento: {list(unmapped)}")
        print("             Adicione ao class_mapping.json e re-execute.")
    gdf["classe_num"] = gdf["classe_num"].fillna(-1).astype(int)
    return gdf


def _add_metadata(
    gdf: gpd.GeoDataFrame,
    zip_path: Path,
    mun_meta: dict,
    ibge_lookup: dict[str, dict],
) -> gpd.GeoDataFrame:
    """Adiciona metadados padronizados via IBGE; usa manifest SGB como fallback."""
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


def process_shapefile(
    zip_path: Path,
    shp_zip_path: str,
    classe_col_hint: str,
    mapping: dict[str, int],
    mun_meta: dict,
    ibge_lookup: dict[str, dict],
) -> gpd.GeoDataFrame | None:
    """
    Extrai SHP ou GPKG do ZIP, aplica mapeamento de classe e padroniza colunas.
    Para GPKGs, shp_zip_path usa o formato "caminho/arquivo.gpkg::NomeCamada"
    (gerado pelo 01_sgb_explore.py). Retorna None em caso de erro.
    """
    # ── Detecta tipo de arquivo ───────────────────────────────────────────
    is_gpkg = "::" in shp_zip_path
    if is_gpkg:
        gpkg_zip_path, layer_name = shp_zip_path.split("::", 1)
    else:
        stem = Path(shp_zip_path).stem
        shp_exts = {".shp", ".dbf", ".prj", ".cpg", ".shx"}

    with tempfile.TemporaryDirectory() as tmp:
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
                data_files = list(tmp_path.rglob("*.gpkg"))
                read_kwargs: dict = {"layer": layer_name}
            else:
                data_files = list(tmp_path.rglob("*.shp"))
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

    # ── CRS ───────────────────────────────────────────────────────────────
    if gdf.crs is None:
        gdf = gdf.set_crs(TARGET_CRS)
    elif gdf.crs.to_epsg() != 4674:
        gdf = gdf.to_crs(TARGET_CRS)

    # ── Limpeza e normalização de geometria ──────────────────────────────
    gdf.geometry = gdf.geometry.make_valid()
    gdf.geometry = gdf.geometry.apply(_to_multipolygon)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        print("✗ sem geometrias válidas após limpeza")
        return None

    # ── Coluna de classe ──────────────────────────────────────────────────
    classe_col = classe_col_hint if classe_col_hint in gdf.columns else None
    if not classe_col:
        lowered = {c.lower(): c for c in gdf.columns}
        for cand in CLASS_COL_CANDIDATES:
            if cand.lower() in lowered:
                classe_col = lowered[cand.lower()]
                break

    if not classe_col:
        # Fallback: infere classe_num pelo nome do arquivo quando não há coluna de classe.
        # Útil para arquivos como "SuscetibilidadeCorrida_A.shp" onde todas as feições
        # representam implicitamente uma única classe de susceptibilidade.
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


def process_tif(
    zip_path: Path,
    tif_zip_path: str,
    mapping: dict[str, int],
    mun_meta: dict,
    ibge_lookup: dict[str, dict],
) -> gpd.GeoDataFrame | None:
    """
    Extrai GeoTIFF do ZIP, poligoniza e padroniza colunas.
    Requer rasterio. Valores de pixel no intervalo 0-5 são usados diretamente
    como classe_num; valores fora do intervalo são mapeados via class_mapping.json.
    """
    try:
        import rasterio
        from rasterio.features import shapes as rasterio_shapes
        import numpy as np
        from shapely.geometry import shape as shapely_shape
    except ImportError:
        print("✗ rasterio não instalado — instale com: pip install rasterio")
        return None

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extract(tif_zip_path, tmp)

            tif_files = (list(tmp_path.rglob("*.tif"))
                         + list(tmp_path.rglob("*.tiff")))
            if not tif_files:
                print("✗ tif não encontrado após extração")
                return None

            with rasterio.open(tif_files[0]) as src:
                data      = src.read(1)
                transform = src.transform
                crs       = src.crs
                nodata    = src.nodata

            # Exclui nodata e poligoniza
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

    # ── CRS ───────────────────────────────────────────────────────────────
    if gdf.crs is None:
        gdf = gdf.set_crs(TARGET_CRS)
    elif gdf.crs.to_epsg() != 4674:
        gdf = gdf.to_crs(TARGET_CRS)

    # ── Limpeza e normalização de geometria ──────────────────────────────
    gdf.geometry = gdf.geometry.make_valid()
    gdf.geometry = gdf.geometry.apply(_to_multipolygon)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        print("✗ sem geometrias válidas após limpeza")
        return None

    # ── Classe: tenta mapping textual; usa pixel value direto se 0-5 ──────
    gdf["classe_orig"] = gdf["pixel_value"].astype(str)
    gdf["classe_num"]  = gdf["classe_orig"].map(mapping)
    still_unmapped     = gdf["classe_num"].isna()
    # Pixel values já numéricos 0-5 são usados diretamente sem necessitar do mapping
    gdf.loc[still_unmapped, "classe_num"] = gdf.loc[still_unmapped, "pixel_value"].apply(
        lambda v: v if 0 <= v <= 5 else -1
    )
    gdf["classe_num"] = gdf["classe_num"].fillna(-1).astype(int)

    gdf = _add_metadata(gdf, zip_path, mun_meta, ibge_lookup)
    return gdf[[c for c in OUTPUT_COLS if c in gdf.columns]]


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTRAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def harmonize(
    state_filter: list[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> None:
    mapping     = load_mapping()
    inventory   = load_inventory()
    manifest    = load_manifest()
    ibge_lookup = load_ibge_municipios()

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
    interrupted = False
    last_i = 0

    try:
        for i, zip_name in enumerate(all_zips, 1):
            last_i = i
            zip_path = DOWNLOAD_DIR / zip_name
            if not zip_path.exists():
                print(f"  [{i:>3}/{total}] SKIP arquivo não encontrado: {zip_name}")
                continue

            mun_meta = manifest.get(zip_name, {})
            label = (f"{mun_meta.get('nm_municipio', '?')} "
                     f"({mun_meta.get('cd_estado', '?')})")
            print(f"  [{i:>3}/{total}] {label}")

            rows = inventory[inventory["zip_filename"] == zip_name].copy()
            # Dedup: quando existem SHP e GPKG para o mesmo tipo, prefere o GPKG
            for tipo_grp in ("inundacao", "massa"):
                mask_tipo = rows["tipo"] == tipo_grp
                tipo_rows = rows[mask_tipo]
                has_gpkg = tipo_rows["shp_path_in_zip"].str.contains("::", na=False).any()
                has_shp  = (~tipo_rows["shp_path_in_zip"].str.contains("::", na=False)).any()
                if has_gpkg and has_shp:
                    drop = mask_tipo & ~rows["shp_path_in_zip"].str.contains("::", na=False)
                    if drop.any():
                        print(f"    [DEDUP] {tipo_grp}: {drop.sum()} SHP(s) ignorado(s) em favor de GPKG")
                        rows = rows[~drop]

            for _, row in rows.iterrows():
                tipo           = row["tipo"]
                file_in_zip    = str(row["shp_path_in_zip"])
                # Para GPKG mostra arquivo::camada; para outros mostra só o nome do arquivo
                display_name   = file_in_zip if "::" in file_in_zip else Path(file_in_zip).name
                print(f"    → {tipo}: {display_name}", end=" ", flush=True)

                if dry_run:
                    print("[DRY RUN]")
                    continue

                # Roteia pelo tipo de arquivo
                base_path = file_in_zip.split("::")[0]
                file_ext  = Path(base_path).suffix.lower()

                if file_ext in {".tif", ".tiff"}:
                    gdf = process_tif(zip_path, file_in_zip, mapping, mun_meta, ibge_lookup)
                else:
                    gdf = process_shapefile(
                        zip_path,
                        file_in_zip,
                        str(row.get("classe_col", "") or ""),
                        mapping,
                        mun_meta,
                        ibge_lookup,
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

    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[INTERROMPIDO] Ctrl+C — {last_i}/{total} ZIPs processados.")
        print("  O GeoPackage contém os municípios gravados até o momento.")
        print("  Rode novamente para reprocessar do zero (o script sempre recria o arquivo).")

    # ── Resumo ────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    status = "INTERROMPIDO" if interrupted else "HARMONIZAÇÃO CONCLUÍDA"
    print(status + (" (DRY RUN)" if dry_run else ""))
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
