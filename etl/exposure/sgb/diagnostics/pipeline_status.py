#!/usr/bin/env python3
"""
SGB — Reconciliação do Status do Pipeline por Município
========================================================
Lê todos os artefatos gerados pelos scripts 00–04 e produz uma tabela
única com o status de cada município em cada etapa do pipeline, separado
por tipo (massa vs inundação).

Inputs (todos inferidos a partir de config/config.local.json via cfg):
  00_sgb_manifest.csv      — universo de URLs coletadas + status de download
  01_sgb_cobertura.csv     — status do ZIP por município (zip_erro, sem_cobertura, ok)
  02_progress.json         — ZIPs extraídos com sucesso por tipo
  02_failures.csv          — falhas de extração com razão (gerado pelo 02 patchado)
  03_progress.json         — municípios harmonizados com sucesso por tipo
  03_failures.csv          — falhas de harmonização com razão (gerado pelo 03 patchado)

Output em <data_dir>/inputs/raw/sgb/:
  sgb_pipeline_status.csv  — uma linha por município, colunas separadas por tipo

STATUS possíveis em cada coluna de etapa:
  ok              — etapa concluída com sucesso
  failed          — etapa falhou (razão em last_failure_reason_*)
  sem_dado        — scraper não encontrou arquivo no portal SGB
  not_processed   — não chegou a esta etapa (bloqueado em etapa anterior)
  n_a             — não aplicável para este tipo neste município

USO:
  python 08_sgb_pipeline_status.py               # gera sgb_pipeline_status.csv
  python 08_sgb_pipeline_status.py --summary     # só imprime sumário, sem CSV
"""

import re
import sys
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

# ── Paths via config ───────────────────────────────────────────────────────────
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pipeline.py").exists())
sys.path.insert(0, str(_ROOT))
from src import config as cfg  # noqa: E402

SGB_DIR     = cfg.RAW_DIR / "sgb"
OUTPUT_PATH = SGB_DIR / "sgb_pipeline_status.csv"

MANIFEST_PATH  = SGB_DIR / "00_sgb_manifest.csv"
COBERTURA_PATH = SGB_DIR / "01_sgb_cobertura.csv"
PROGRESS_02    = SGB_DIR / "por_municipio/02_progress.json"
FAILURES_02    = SGB_DIR / "02_failures.csv"
PROGRESS_03    = SGB_DIR / "harmonized/03_progress.json"
FAILURES_03    = SGB_DIR / "03_failures.csv"

TIPOS = ("massa", "inundacao")

# Status definidos (ordem de prioridade crescente para last_failure)
STATUS_OK             = "ok"
STATUS_FAILED         = "failed"
STATUS_SEM_DADO       = "sem_dado"
STATUS_NOT_PROCESSED  = "not_processed"
STATUS_NA             = "n_a"


def _mun_slug(nm: str) -> str:
    """Mesmo regex que 02_sgb_extract.py usa para nomear diretórios."""
    return re.sub(r"[^\w]", "_", nm.strip(), flags=re.ASCII).strip("_")


# ══════════════════════════════════════════════════════════════════════════════
# CARREGAMENTO DOS ARTEFATOS
# ══════════════════════════════════════════════════════════════════════════════

def load_manifest() -> pd.DataFrame:
    """Universo: uma linha por (zip_filename, município). Agrega por cd_mun_ibge."""
    if not MANIFEST_PATH.exists():
        print(f"  [AVISO] {MANIFEST_PATH.name} não encontrado — 00 não foi rodado.")
        return pd.DataFrame(columns=["cd_mun", "sigla_uf", "nm_mun",
                                     "status_download"])
    df = pd.read_csv(MANIFEST_PATH, dtype=str).fillna("")
    # cd_mun_ibge pode estar ausente se scraper não resolveu o IBGE
    df["cd_mun"]   = df.get("cd_mun_ibge", pd.Series("", index=df.index)).str.strip()
    df["sigla_uf"] = df.get("cd_estado",   pd.Series("", index=df.index)).str.strip()
    df["nm_mun"]   = df.get("nm_municipio",pd.Series("", index=df.index)).str.strip()

    # status no manifest: ok | error | sem_dado
    df["status_download_raw"] = df.get("status", pd.Series("", index=df.index)).str.strip()

    # Agrega por cd_mun: worst-case status
    def _agg_status(g):
        s = g["status_download_raw"].value_counts()
        if "error"    in s: return "failed"
        if "sem_dado" in s: return STATUS_SEM_DADO
        if "ok"       in s: return STATUS_OK
        return ""

    agg = (
        df.groupby("cd_mun")
        .apply(_agg_status)
        .reset_index(name="status_download")
    )
    meta = (
        df.dropna(subset=["cd_mun"])
        .drop_duplicates("cd_mun")[["cd_mun", "sigla_uf", "nm_mun"]]
    )
    return agg.merge(meta, on="cd_mun", how="left")


def load_cobertura() -> pd.DataFrame:
    """01_sgb_cobertura.csv: status do ZIP, has_inundacao, has_massa por município."""
    if not COBERTURA_PATH.exists():
        print(f"  [AVISO] {COBERTURA_PATH.name} não encontrado — 01 não foi rodado.")
        return pd.DataFrame(columns=["cd_mun", "status_zip",
                                     "has_massa", "has_inundacao"])
    df = pd.read_csv(COBERTURA_PATH, dtype=str).fillna("")
    df["cd_mun"] = df.get("cd_mun_ibge", pd.Series("", index=df.index)).str.strip()

    # Agrega por cd_mun (um município pode ter mais de um ZIP)
    def _zip_status(g):
        s = g["status_zip"].value_counts()
        if "zip_erro"      in s: return "failed"
        if "sem_cobertura" in s: return "failed"
        if "zip_vazio"     in s: return "failed"
        if "ok"            in s: return STATUS_OK
        return "failed"

    agg = (
        df.groupby("cd_mun")
        .apply(_zip_status)
        .reset_index(name="status_zip")
    )
    has = (
        df.groupby("cd_mun")
        .agg(
            has_massa=("has_massa",     lambda x: (x.str.lower() == "true").any()),
            has_inundacao=("has_inundacao", lambda x: (x.str.lower() == "true").any()),
            status_zip_raw=("status_zip", lambda x: x.iloc[0]),
        )
        .reset_index()
    )
    return agg.merge(has[["cd_mun", "has_massa", "has_inundacao"]], on="cd_mun", how="left")


def load_progress_02() -> dict[str, set[str]]:
    """02_progress.json: {tipo: [zip_filenames]} — extraídos com sucesso."""
    if not PROGRESS_02.exists():
        return {t: set() for t in TIPOS}
    with open(PROGRESS_02, encoding="utf-8") as f:
        data = json.load(f)
    return {t: set(data.get(t, [])) for t in TIPOS}


def load_failures_02() -> pd.DataFrame:
    """02_failures.csv: (cd_mun, tipo, stage, reason) para falhas de extração."""
    if not FAILURES_02.exists():
        return pd.DataFrame(columns=["cd_mun", "tipo", "reason"])
    df = pd.read_csv(FAILURES_02, dtype=str).fillna("")
    # Mantém só a última falha por (cd_mun, tipo) — mais recente em caso de reruns
    return (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["cd_mun", "tipo"], keep="last")
        [["cd_mun", "tipo", "reason"]]
    )


def load_progress_03() -> dict[str, set[str]]:
    """03_progress.json: {tipo: [mun_ids]} — harmonizados com sucesso."""
    if not PROGRESS_03.exists():
        return {t: set() for t in TIPOS}
    with open(PROGRESS_03, encoding="utf-8") as f:
        data = json.load(f)
    return {t: set(data.get(t, [])) for t in TIPOS}


def load_failures_03() -> pd.DataFrame:
    """03_failures.csv: (mun_id, tipo, reason) para falhas de harmonização."""
    if not FAILURES_03.exists():
        return pd.DataFrame(columns=["mun_id", "tipo", "reason"])
    df = pd.read_csv(FAILURES_03, dtype=str).fillna("")
    return (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["mun_id", "tipo"], keep="last")
        [["mun_id", "tipo", "reason"]]
    )


# ══════════════════════════════════════════════════════════════════════════════
# RECONCILIAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def build_status(
    manifest:    pd.DataFrame,
    cobertura:   pd.DataFrame,
    prog02:      dict[str, set[str]],
    fail02:      pd.DataFrame,
    prog03:      dict[str, set[str]],
    fail03:      pd.DataFrame,
    manifest_df: pd.DataFrame,  # original com zip_filename p/ join com prog02
) -> pd.DataFrame:
    """Constrói a tabela wide com status por (município, tipo)."""

    # Mapeamento zip_filename → cd_mun (para resolver prog02)
    zip_to_cd = {}
    if not manifest_df.empty and "cd_mun_ibge" in manifest_df.columns:
        for _, row in manifest_df.iterrows():
            z = str(row.get("filename", "")).strip()
            c = str(row.get("cd_mun_ibge", "")).strip()
            if z and c:
                zip_to_cd[z] = c

    # Mapeamento cd_mun → mun_id (slug) para resolver prog03
    cd_to_mun_id = {}
    if not manifest_df.empty:
        for _, row in manifest_df.iterrows():
            c  = str(row.get("cd_mun_ibge", "")).strip()
            uf = str(row.get("cd_estado",   "")).strip()
            nm = str(row.get("nm_municipio","")).strip()
            if c and uf and nm:
                cd_to_mun_id[c] = f"{uf}_{_mun_slug(nm)}"

    # fail02 indexado por (cd_mun, tipo)
    f02 = {}
    for _, row in fail02.iterrows():
        f02[(row["cd_mun"], row["tipo"])] = row["reason"]

    # prog03 success: mun_id sets → cd_mun sets via cd_to_mun_id reverse
    mun_id_to_cd = {v: k for k, v in cd_to_mun_id.items()}
    prog03_cd: dict[str, set[str]] = {}
    for tipo, mun_ids in prog03.items():
        prog03_cd[tipo] = {mun_id_to_cd[m] for m in mun_ids if m in mun_id_to_cd}

    # fail03 indexado por (cd_mun, tipo) via mun_id → cd
    f03 = {}
    for _, row in fail03.iterrows():
        mid = row.get("mun_id", "")
        cd  = mun_id_to_cd.get(mid, "")
        if cd:
            f03[(cd, row["tipo"])] = row["reason"]

    # prog02 por cd_mun: {tipo: set of cd_mun com sucesso}
    prog02_cd: dict[str, set[str]] = {}
    for tipo, zips in prog02.items():
        prog02_cd[tipo] = {zip_to_cd[z] for z in zips if z in zip_to_cd}

    rows = []
    all_muns = manifest[["cd_mun", "sigla_uf", "nm_mun"]].drop_duplicates("cd_mun")
    cob = cobertura.set_index("cd_mun") if not cobertura.empty else pd.DataFrame()

    for _, mrow in all_muns.iterrows():
        cd  = mrow["cd_mun"]
        uf  = mrow["sigla_uf"]
        nm  = mrow["nm_mun"]

        # Coleta IBGE code numérico de UF (não disponível aqui, deixa vazio)
        row: dict = {"cd_mun": cd, "sigla_uf": uf, "nm_mun": nm}

        # Status download (único por município)
        dl_status = manifest.set_index("cd_mun")["status_download"].get(cd, "")
        row["status_download"] = dl_status

        # Por tipo
        for tipo in TIPOS:
            suf = f"_{tipo}"

            # Cobertura explore
            has_tipo  = False
            expl_stat = STATUS_NOT_PROCESSED
            if cd in cob.index:
                cob_row   = cob.loc[cd]
                has_tipo  = bool(cob_row.get(f"has_{tipo}", False))
                zip_stat  = cob_row.get("status_zip", "")
                if zip_stat == STATUS_OK and has_tipo:
                    expl_stat = STATUS_OK
                elif zip_stat == STATUS_OK and not has_tipo:
                    expl_stat = STATUS_NA  # ZIP ok mas sem layer deste tipo
                else:
                    expl_stat = STATUS_FAILED
            elif dl_status == STATUS_SEM_DADO:
                expl_stat = STATUS_NA

            row[f"status_explore{suf}"] = expl_stat

            # Extract
            if expl_stat not in (STATUS_OK,):
                ext_stat   = STATUS_NOT_PROCESSED if expl_stat != STATUS_NA else STATUS_NA
                ext_reason = ""
            elif cd in prog02_cd.get(tipo, set()):
                ext_stat   = STATUS_OK
                ext_reason = ""
            elif (cd, tipo) in f02:
                ext_stat   = STATUS_FAILED
                ext_reason = f02[(cd, tipo)]
            else:
                # Sem evidência: pode não ter sido rodado ainda ou falha não registrada
                ext_stat   = STATUS_NOT_PROCESSED
                ext_reason = ""

            row[f"status_extract{suf}"] = ext_stat
            row[f"reason_extract{suf}"] = ext_reason

            # Harmonize
            if ext_stat != STATUS_OK:
                har_stat   = STATUS_NOT_PROCESSED if ext_stat not in (STATUS_NA,) else STATUS_NA
                har_reason = ""
            elif cd in prog03_cd.get(tipo, set()):
                har_stat   = STATUS_OK
                har_reason = ""
            elif (cd, tipo) in f03:
                har_stat   = STATUS_FAILED
                har_reason = f03[(cd, tipo)]
            else:
                har_stat   = STATUS_NOT_PROCESSED
                har_reason = ""

            row[f"status_harmonize{suf}"] = har_stat
            row[f"reason_harmonize{suf}"] = har_reason

            # in_pipeline = harmonize ok (proxy para "chegou ao 04+")
            row[f"in_pipeline{suf}"] = har_stat == STATUS_OK

            # Última falha registrada
            if har_stat == STATUS_FAILED:
                row[f"last_failure_stage{suf}"]  = "harmonize"
                row[f"last_failure_reason{suf}"] = har_reason
            elif ext_stat == STATUS_FAILED:
                row[f"last_failure_stage{suf}"]  = "extract"
                row[f"last_failure_reason{suf}"] = ext_reason
            elif expl_stat == STATUS_FAILED:
                row[f"last_failure_stage{suf}"]  = "explore"
                row[f"last_failure_reason{suf}"] = "zip_erro ou sem_cobertura"
            elif dl_status == "failed":
                row[f"last_failure_stage{suf}"]  = "download"
                row[f"last_failure_reason{suf}"] = "download falhou"
            else:
                row[f"last_failure_stage{suf}"]  = ""
                row[f"last_failure_reason{suf}"] = ""

        rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# SUMÁRIO
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame) -> None:
    n_total = len(df)
    print(f"\n{'═'*60}")
    print(f"SGB PIPELINE STATUS — {n_total} municípios no manifest")
    print(f"{'═'*60}")

    for tipo in TIPOS:
        suf = f"_{tipo}"
        print(f"\n[{tipo.upper()}]")
        dl_ok   = (df["status_download"] == STATUS_OK).sum()
        ex_ok   = (df[f"status_explore{suf}"] == STATUS_OK).sum()
        ex_na   = (df[f"status_explore{suf}"] == STATUS_NA).sum()
        ext_ok  = (df[f"status_extract{suf}"] == STATUS_OK).sum()
        har_ok  = (df[f"status_harmonize{suf}"] == STATUS_OK).sum()
        in_pipe = df[f"in_pipeline{suf}"].sum()

        print(f"  Download OK          : {dl_ok:>5,}")
        print(f"  Explore OK (tipo)    : {ex_ok:>5,}   (sem layer: {ex_na:,})")
        print(f"  Extract OK           : {ext_ok:>5,}")
        print(f"  Harmonize OK         : {har_ok:>5,}")
        print(f"  → in_pipeline_{tipo[:4]:4}: {in_pipe:>5,}  ({in_pipe/n_total:.0%} do total)")

        # Dropout por etapa
        print(f"\n  Dropout por etapa:")
        prev = dl_ok
        for stage, col in [("explore", f"status_explore{suf}"),
                            ("extract", f"status_extract{suf}"),
                            ("harmonize", f"status_harmonize{suf}")]:
            n_ok   = (df[col] == STATUS_OK).sum()
            n_fail = (df[col] == STATUS_FAILED).sum()
            if n_fail:
                print(f"    {stage:12}: {n_fail:>4,} falhas  ({prev - n_ok:>4,} saíram)")
            prev = n_ok

    print(f"\n  Municípios em AMBOS (massa E inundação): "
          f"{(df['in_pipeline_massa'] & df['in_pipeline_inundacao']).sum():,}")
    print(f"  Municípios em nenhum : "
          f"{(~df['in_pipeline_massa'] & ~df['in_pipeline_inundacao']).sum():,}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SGB — Reconcilia status do pipeline por município"
    )
    parser.add_argument("--summary", action="store_true",
                        help="Imprime sumário sem escrever CSV")
    args = parser.parse_args()

    print("=" * 60)
    print("SGB — Reconciliação do Status do Pipeline")
    print(f"Run: {datetime.now().strftime(cfg.TS_FORMAT_LOG)}")
    print("=" * 60)

    print("\nCarregando artefatos...")
    manifest_df_raw = pd.read_csv(MANIFEST_PATH, dtype=str).fillna("") \
        if MANIFEST_PATH.exists() else pd.DataFrame()

    manifest  = load_manifest()
    cobertura = load_cobertura()
    prog02    = load_progress_02()
    fail02    = load_failures_02()
    prog03    = load_progress_03()
    fail03    = load_failures_03()

    print(f"  manifest  : {len(manifest):,} municípios")
    print(f"  cobertura : {len(cobertura):,} municípios")
    print(f"  prog02    : {sum(len(v) for v in prog02.values()):,} entradas")
    print(f"  fail02    : {len(fail02):,} falhas registradas")
    print(f"  prog03    : {sum(len(v) for v in prog03.values()):,} entradas")
    print(f"  fail03    : {len(fail03):,} falhas registradas")

    status_df = build_status(manifest, cobertura, prog02, fail02,
                             prog03, fail03, manifest_df_raw)
    print_summary(status_df)

    if not args.summary:
        status_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
        print(f"✓ Salvo em: {OUTPUT_PATH}")
        print(f"  ({len(status_df):,} municípios × {len(status_df.columns)} colunas)")


if __name__ == "__main__":
    main()
