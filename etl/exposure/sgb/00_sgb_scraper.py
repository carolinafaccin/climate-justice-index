#!/usr/bin/env python3
"""
SGB Cartografia de Suscetibilidade — Coletor de Links e Downloader
===================================================================
Coleta metadados e links de download de todos os municípios disponíveis
no portal do Serviço Geológico do Brasil (SGB/CPRM), usando a API REST
do repositório DSpace (rigeo.sgb.gov.br).

USO:
  python sgb_suscetibilidade.py collect              # Fase 1: raspa links e metadados
  python sgb_suscetibilidade.py download             # Fase 2: baixa arquivos do manifest
  python sgb_suscetibilidade.py all                  # Ambas em sequência
  python sgb_suscetibilidade.py report               # Resumo do manifest atual

OPÇÕES:
  --batch-size N      Arquivos por lote antes de pausar (default: 15)
  --batch-delay N     Segundos de pausa entre lotes (default: 90)
  --download-delay N  Segundos entre downloads individuais (default: 10)
  --page-delay N      Segundos entre requisições de scraping/API (default: 0.8)
  --state AC,SP,...   Processa apenas os estados listados
  --no-resume         Ignora manifest existente e recoleta tudo

FLUXO TÉCNICO:
  1. Página principal SGB → lista de estados com URLs
  2. Página de cada estado (SGB) → lista de municípios com handles do rigeo
  3. API DSpace: GET /server/api/pid/find?id=doc/{ID}  → UUID, handle, data, autores
  4. API DSpace: GET /server/api/core/items/{UUID}/bundles?embed=bitstreams → URL do ZIP
"""

import requests
from bs4 import BeautifulSoup
import csv
import os
import time
import re
import sys
import argparse
from datetime import datetime
from pathlib import Path

# ── Configurações de caminho ───────────────────────────────────────────────────
DOWNLOAD_DIR = Path(
    "/Users/lina/Library/CloudStorage/"
    "GoogleDrive-faccincarolina@gmail.com/"
    "Meu Drive/Workspace/code/data-outputs/"
    "climate-injustice-index/data/inputs/raw/sgb/raw_zips"
)
MANIFEST_PATH = Path(
    "/Users/lina/Library/CloudStorage/"
    "GoogleDrive-faccincarolina@gmail.com/"
    "Meu Drive/Workspace/code/data-outputs/"
    "climate-injustice-index/data/inputs/raw/sgb/sgb_download_manifest.csv"
)

# ── URLs base ──────────────────────────────────────────────────────────────────
SGB_MAIN_URL = "https://www.sgb.gov.br/produtos-por-estado-cartografia-de-suscetibilidade"
SGB_BASE     = "https://www.sgb.gov.br"
RIGEO_BASE   = "https://rigeo.sgb.gov.br"

# ── Colunas do manifest ────────────────────────────────────────────────────────
MANIFEST_COLS = [
    "cd_estado", "nm_estado", "nm_municipio", "cd_mun_ibge",
    "url_download", "uri", "filename",
    "downloaded_at", "status", "autores", "data",
]

# ── Delays padrão (segundos) ───────────────────────────────────────────────────
DEFAULT_PAGE_DELAY     = 0.8   # entre requisições de scraping/API
DEFAULT_DOWNLOAD_DELAY = 10    # entre downloads individuais
DEFAULT_BATCH_SIZE     = 15    # arquivos por lote
DEFAULT_BATCH_DELAY    = 90    # pausa entre lotes (tempo para Drive sincronizar)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "pt-BR,pt;q=0.9",
}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ══════════════════════════════════════════════════════════════════════════════

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def safe_get(session: requests.Session, url: str, timeout: int = 45,
             as_json: bool = False) -> requests.Response:
    """GET com retry simples em caso de timeout ou erro 5xx."""
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=timeout, allow_redirects=True)
            resp.raise_for_status()
            return resp
        except requests.exceptions.Timeout:
            print(f"    [TIMEOUT] tentativa {attempt+1}/3 — {url}")
            time.sleep(5 * (attempt + 1))
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else 0
            if status == 404:
                raise  # Não vale retry para 404
            if status < 500:
                raise
            print(f"    [HTTP {status}] tentativa {attempt+1}/3")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Falhou após 3 tentativas: {url}")


def load_manifest() -> dict:
    """Carrega manifest existente indexado pela URI (handle)."""
    existing = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = row.get("uri") or row.get("url_download", "")
                if key:
                    existing[key] = row
        print(f"[INFO] Manifest existente: {len(existing)} registros carregados")
    return existing


def save_manifest(records: list):
    """Salva manifest CSV, criando diretório se necessário."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1A — SCRAPING DAS PÁGINAS SGB (listas de municípios por estado)
# ══════════════════════════════════════════════════════════════════════════════

def get_state_links(session: requests.Session) -> list[dict]:
    """Coleta todos os links de estado da página principal do SGB."""
    print(f"\n[1/3] Coletando estados em:\n      {SGB_MAIN_URL}")
    resp = safe_get(session, SGB_MAIN_URL)
    soup = BeautifulSoup(resp.text, "html.parser")

    states = []
    for row in soup.select("tr"):
        cells = row.select("td")
        if len(cells) < 2:
            continue
        link = cells[0].find("a")
        if not link:
            continue
        state_text = cells[0].get_text(strip=True)
        m = re.match(r"(.+?)\s*\((\w{2})\)", state_text)
        if not m:
            continue
        nm_estado = m.group(1).strip()
        cd_estado = m.group(2)
        href = link.get("href", "")
        if not href.startswith("http"):
            href = SGB_BASE + href
        count = cells[1].get_text(strip=True)
        states.append({"nm_estado": nm_estado, "cd_estado": cd_estado,
                       "url": href, "count": count})

    print(f"      → {len(states)} estados encontrados")
    return states


def get_municipality_links(session: requests.Session, state: dict,
                           delay: float) -> list[dict]:
    """Coleta lista de municípios e seus handles do rigeo para um estado."""
    time.sleep(delay)
    resp = safe_get(session, state["url"])
    soup = BeautifulSoup(resp.text, "html.parser")

    municipalities = []
    for row in soup.select("tr"):
        cells = row.select("td")
        if len(cells) < 2:
            continue
        nm_municipio = cells[0].get_text(strip=True)
        if not nm_municipio:
            continue

        # Procura link para o rigeo (handle URL)
        rigeo_url = None
        for a in cells[1].select("a"):
            href = a.get("href", "")
            if "rigeo.sgb.gov.br" in href:
                rigeo_url = href
                break

        municipalities.append({
            "nm_municipio": nm_municipio,
            "rigeo_url": rigeo_url,
        })

    return municipalities


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1B — API DSPACE (metadados + URLs de download)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_handle_id(rigeo_url: str) -> str | None:
    """
    Extrai o handle ID de uma URL do rigeo.
    Ex: https://rigeo.sgb.gov.br/handle/doc/22898 → 'doc/22898'
    """
    m = re.search(r"/handle/(\w+/\d+)", rigeo_url)
    return m.group(1) if m else None


def get_item_metadata_api(session: requests.Session, rigeo_url: str,
                          delay: float) -> dict:
    """
    Busca metadados e URL de download de um item via API REST do DSpace.

    Fluxo:
      1. GET /server/api/pid/find?id={handle_id}
         → retorna JSON com uuid, handle, metadata (autores, data)
      2. GET /server/api/core/items/{uuid}/bundles?embed=bitstreams
         → retorna bundles com bitstreams; filtra bundle ORIGINAL e arquivo .zip
    """
    time.sleep(delay)

    handle_id = _extract_handle_id(rigeo_url)
    uri = rigeo_url  # fallback; será atualizado com o handle canônico

    if not handle_id:
        return {
            "uri": rigeo_url, "data": "", "autores": "",
            "url_download": "", "filename": "",
            "status": "error", "_error": f"Não foi possível extrair handle de: {rigeo_url}",
        }

    try:
        # ── Passo 1: metadados do item ─────────────────────────────────────
        pid_url = f"{RIGEO_BASE}/server/api/pid/find?id={handle_id}"
        resp = safe_get(session, pid_url)
        item = resp.json()

        uuid   = item.get("uuid", "")
        handle = item.get("handle", handle_id)
        uri    = f"{RIGEO_BASE}/handle/{handle}"

        metadata = item.get("metadata", {})

        # Data de publicação
        date_vals = metadata.get("dc.date.issued", [])
        data = date_vals[0]["value"] if date_vals else ""
        # Normaliza para YYYY-MM se vier como YYYY-MM-DD
        if data and len(data) > 7:
            data = data[:7]

        # Autores
        author_vals = metadata.get("dc.contributor.author", [])
        autores = "; ".join(a["value"] for a in author_vals)

        if not uuid:
            return {
                "uri": uri, "data": data, "autores": autores,
                "url_download": "", "filename": "",
                "status": "error", "_error": "UUID não encontrado na resposta da API",
            }

        # ── Passo 2: bundles/bitstreams ────────────────────────────────────
        time.sleep(delay)
        bundles_url = f"{RIGEO_BASE}/server/api/core/items/{uuid}/bundles?embed=bitstreams"
        bundles_resp = safe_get(session, bundles_url)
        bundles_data = bundles_resp.json()

        zip_url  = ""
        filename = ""

        bundles = bundles_data.get("_embedded", {}).get("bundles", [])
        for bundle in bundles:
            if bundle.get("name") != "ORIGINAL":
                continue
            bitstreams = (
                bundle.get("_embedded", {})
                      .get("bitstreams", {})
                      .get("_embedded", {})
                      .get("bitstreams", [])
            )
            for bs in bitstreams:
                if bs.get("name", "").lower().endswith(".zip"):
                    filename = bs["name"]
                    # Prefere o link direto de content
                    zip_url = (
                        bs.get("_links", {}).get("content", {}).get("href", "")
                        or f"{RIGEO_BASE}/bitstreams/{bs['uuid']}/download"
                    )
                    break
            if zip_url:
                break

        status = "" if zip_url else "sem_dado"
        return {
            "uri": uri, "data": data, "autores": autores,
            "url_download": zip_url, "filename": filename,
            "status": status,
        }

    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response else 0
        return {
            "uri": uri, "data": "", "autores": "",
            "url_download": "", "filename": "",
            "status": "error", "_error": f"HTTP {code}: {rigeo_url}",
        }
    except Exception as e:
        return {
            "uri": uri, "data": "", "autores": "",
            "url_download": "", "filename": "",
            "status": "error", "_error": str(e),
        }


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — ORQUESTRAÇÃO DA COLETA
# ══════════════════════════════════════════════════════════════════════════════

def collect_all_links(state_filter: list[str] | None = None,
                      resume: bool = True,
                      page_delay: float = DEFAULT_PAGE_DELAY) -> list[dict]:
    """
    Fase 1: Raspa todos os estados/municípios e popula o manifest
    com metadados + URLs de download. Não faz nenhum download de arquivo.
    """
    session = make_session()
    existing = load_manifest() if resume else {}

    states = get_state_links(session)
    if state_filter:
        states = [s for s in states if s["cd_estado"] in state_filter]
        print(f"      → Filtrando: {[s['cd_estado'] for s in states]}")

    total_states = len(states)
    all_records: list[dict] = []

    print(f"\n[2/3] Coletando municípios + metadados ({total_states} estados)...")

    for si, state in enumerate(states, 1):
        print(f"\n  [{si}/{total_states}] {state['nm_estado']} ({state['cd_estado']}) "
              f"— {state.get('count','')} municípios esperados")

        try:
            municipalities = get_municipality_links(session, state, page_delay)
        except Exception as e:
            print(f"    [ERRO] Não foi possível carregar página do estado: {e}")
            continue

        print(f"    → {len(municipalities)} municípios encontrados")

        for mun in municipalities:
            rigeo_url = mun["rigeo_url"]

            record: dict = {
                "cd_estado":     state["cd_estado"],
                "nm_estado":     state["nm_estado"],
                "nm_municipio":  mun["nm_municipio"],
                "cd_mun_ibge":   "",
                "url_download":  "",
                "uri":           rigeo_url or "",
                "filename":      "",
                "downloaded_at": "",
                "status":        "",
                "autores":       "",
                "data":          "",
            }

            if not rigeo_url:
                record["status"] = "sem_dado"
                all_records.append(record)
                continue

            # Se já está no manifest e tem URL de download, pula
            if resume and rigeo_url in existing:
                cached = existing[rigeo_url]
                if cached.get("url_download") or cached.get("status") in ("sem_dado", "ok"):
                    print(f"    [SKIP] Já coletado: {mun['nm_municipio']}")
                    all_records.append({**record, **cached})
                    continue

            print(f"    → API: {mun['nm_municipio']}", end=" ", flush=True)
            meta = get_item_metadata_api(session, rigeo_url, page_delay)

            if meta.get("_error"):
                print(f"✗ ERRO: {meta['_error']}")
            elif meta.get("status") == "sem_dado":
                print(f"⚠ sem ZIP")
            else:
                print(f"✓ {meta.get('filename','?')}")

            record.update({k: v for k, v in meta.items() if not k.startswith("_")})
            all_records.append(record)

            # Salva manifest incrementalmente a cada município
            save_manifest(all_records)

    # ── Estatísticas finais ────────────────────────────────────────────────
    total    = len(all_records)
    with_zip = sum(1 for r in all_records if r.get("url_download"))
    sem_dado = sum(1 for r in all_records if r.get("status") == "sem_dado")
    errors   = sum(1 for r in all_records if r.get("status") == "error")

    print(f"\n[3/3] Fase 1 concluída!")
    print(f"      Total municípios:      {total}")
    print(f"      Com arquivo ZIP:       {with_zip}")
    print(f"      Sem dados:             {sem_dado}")
    print(f"      Erros de coleta:       {errors}")
    print(f"      Manifest salvo em:\n      {MANIFEST_PATH}")

    return all_records


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — DOWNLOAD DOS ARQUIVOS
# ══════════════════════════════════════════════════════════════════════════════

def download_files(batch_size:     int = DEFAULT_BATCH_SIZE,
                   batch_delay:    int = DEFAULT_BATCH_DELAY,
                   download_delay: int = DEFAULT_DOWNLOAD_DELAY,
                   state_filter:   list[str] | None = None) -> None:
    """
    Fase 2: Baixa os arquivos ZIP listados no manifest para DOWNLOAD_DIR.
    - Pula arquivos já existentes no disco (status = 'ok').
    - Salva o manifest após cada arquivo processado.
    - Pausa entre lotes para dar tempo ao Google Drive de sincronizar.
    - Usa arquivo .part durante o download; renomeia ao concluir.
    """
    if not MANIFEST_PATH.exists():
        print("[ERRO] Manifest não encontrado. Execute 'collect' primeiro.")
        sys.exit(1)

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    to_download = [
        r for r in records
        if r.get("url_download")
        and r.get("filename")
        and r.get("status") not in ("ok",)
        and (not state_filter or r.get("cd_estado") in state_filter)
    ]

    total_to_dl = len(to_download)
    print(f"\n[FASE 2] {total_to_dl} arquivos a baixar")
    print(f"         Lotes de {batch_size} | pausa entre lotes: {batch_delay}s")
    print(f"         Delay entre downloads: {download_delay}s")
    print(f"         Destino: {DOWNLOAD_DIR}\n")

    downloaded_ok = 0
    errors        = 0

    for i, record in enumerate(to_download, 1):
        filename = record["filename"]
        url      = record["url_download"]
        dest     = DOWNLOAD_DIR / filename
        label    = f"{record['nm_municipio']} ({record['cd_estado']})"

        # ── Já existe no disco? ──────────────────────────────────────────
        if dest.exists():
            size_mb = dest.stat().st_size / 1_048_576
            print(f"  [{i:>3}/{total_to_dl}] SKIP ({size_mb:.0f} MB já existe): {filename}")
            record["status"]        = "ok"
            record["downloaded_at"] = record.get("downloaded_at") or \
                                      datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_manifest(records)
            continue

        print(f"  [{i:>3}/{total_to_dl}] {label}")
        print(f"           {filename}")

        tmp = dest.with_suffix(".part")
        try:
            resp = session.get(url, timeout=600, stream=True)
            resp.raise_for_status()

            total_bytes = int(resp.headers.get("content-length", 0))
            received    = 0

            with open(tmp, "wb") as fout:
                for chunk in resp.iter_content(chunk_size=2 * 1024 * 1024):
                    if chunk:
                        fout.write(chunk)
                        received += len(chunk)
                        if total_bytes:
                            pct = received / total_bytes * 100
                            print(f"\r           {pct:5.1f}%  "
                                  f"{received/1_048_576:7.1f} / "
                                  f"{total_bytes/1_048_576:.1f} MB ",
                                  end="", flush=True)

            tmp.rename(dest)
            size_mb = dest.stat().st_size / 1_048_576
            print(f"\r           ✓ {size_mb:.1f} MB salvos{' '*20}")

            record["status"]        = "ok"
            record["downloaded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            downloaded_ok += 1

        except Exception as e:
            print(f"\r           ✗ ERRO: {e}{' '*20}")
            record["status"] = "error"
            errors += 1
            if tmp.exists():
                tmp.unlink()

        save_manifest(records)

        # ── Pausa ────────────────────────────────────────────────────────
        if i < total_to_dl:
            if i % batch_size == 0:
                batch_num = i // batch_size
                print(f"\n  ── Lote {batch_num} concluído. "
                      f"Pausando {batch_delay}s para o Drive sincronizar... ──\n")
                time.sleep(batch_delay)
            else:
                time.sleep(download_delay)

    # ── Resumo ────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"FASE 2 CONCLUÍDA")
    print(f"  Baixados com sucesso:  {downloaded_ok}")
    print(f"  Erros:                 {errors}")
    print(f"  Manifest atualizado:   {MANIFEST_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# RELATÓRIO
# ══════════════════════════════════════════════════════════════════════════════

def print_report() -> None:
    """Mostra resumo tabular do estado atual do manifest."""
    if not MANIFEST_PATH.exists():
        print("Manifest não encontrado.")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    total    = len(records)
    ok       = sum(1 for r in records if r.get("status") == "ok")
    errors   = sum(1 for r in records if r.get("status") == "error")
    sem_dado = sum(1 for r in records if r.get("status") == "sem_dado")
    pending  = sum(1 for r in records
                   if r.get("url_download") and r.get("status") not in ("ok", "error"))

    by_state: dict[str, dict] = {}
    for r in records:
        s = r.get("cd_estado", "??")
        by_state.setdefault(s, {"total": 0, "ok": 0, "error": 0, "sem_dado": 0, "pending": 0})
        by_state[s]["total"] += 1
        st = r.get("status", "")
        if st == "ok":
            by_state[s]["ok"] += 1
        elif st == "error":
            by_state[s]["error"] += 1
        elif st == "sem_dado":
            by_state[s]["sem_dado"] += 1
        elif r.get("url_download"):
            by_state[s]["pending"] += 1

    print(f"\n{'═'*65}")
    print(f"  MANIFEST: {MANIFEST_PATH.name}")
    print(f"{'─'*65}")
    print(f"  Total registros:        {total:>5}")
    print(f"  Download OK:            {ok:>5}")
    print(f"  Erros:                  {errors:>5}")
    print(f"  Sem dado (ZIP ausente): {sem_dado:>5}")
    print(f"  Pendentes:              {pending:>5}")
    print(f"{'─'*65}")
    print(f"  {'ESTADO':<8} {'TOTAL':>6} {'OK':>6} {'ERRO':>6} {'SEM_DAD':>8} {'PENDING':>8}")
    print(f"{'─'*65}")
    for s in sorted(by_state):
        d = by_state[s]
        print(f"  {s:<8} {d['total']:>6} {d['ok']:>6} {d['error']:>6} "
              f"{d['sem_dado']:>8} {d['pending']:>8}")
    print(f"{'═'*65}\n")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SGB Suscetibilidade — Coletor e Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "phase",
        choices=["collect", "download", "all", "report"],
        help="collect|download|all|report",
    )
    parser.add_argument("--batch-size",     type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--batch-delay",    type=int,   default=DEFAULT_BATCH_DELAY)
    parser.add_argument("--download-delay", type=int,   default=DEFAULT_DOWNLOAD_DELAY)
    parser.add_argument("--page-delay",     type=float, default=DEFAULT_PAGE_DELAY)
    parser.add_argument("--state",          type=str,   default=None,
                        help="Siglas separadas por vírgula, ex: SP,RJ")
    parser.add_argument("--no-resume",      action="store_true")

    args = parser.parse_args()
    state_filter = [s.strip().upper() for s in args.state.split(",")] if args.state else None
    resume       = not args.no_resume

    if args.phase == "report":
        print_report()
        return

    if args.phase in ("collect", "all"):
        collect_all_links(
            state_filter=state_filter,
            resume=resume,
            page_delay=args.page_delay,
        )

    if args.phase in ("download", "all"):
        download_files(
            batch_size=args.batch_size,
            batch_delay=args.batch_delay,
            download_delay=args.download_delay,
            state_filter=state_filter,
        )


if __name__ == "__main__":
    main()