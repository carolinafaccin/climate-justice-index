#!/usr/bin/env python3
"""
SGB Cartografia de Suscetibilidade — Coletor de Links e Downloader
===================================================================
Coleta metadados e links de download de todos os municípios disponíveis
no portal do Serviço Geológico do Brasil (SGB/CPRM), usando a API REST
do repositório DSpace (rigeo.sgb.gov.br).

Os caminhos de saída (DOWNLOAD_DIR e MANIFEST_PATH) são derivados de
data_dir em config/config.local.json — não há paths hardcoded.

USO:
  python 00_sgb_scraper.py collect      # Fase 1: raspa links e metadados
  python 00_sgb_scraper.py download     # Fase 2: baixa arquivos do manifest
  python 00_sgb_scraper.py all          # Ambas em sequência
  python 00_sgb_scraper.py report       # Resumo do manifest atual
  python 00_sgb_scraper.py redownload   # Re-baixa ZIPs corrompidos (via 01_sgb_inventario.csv)

OPÇÕES:
  --workers N       Downloads paralelos (default: 6; SGB permite no máximo 6 simultâneos)
  --page-delay N    Segundos entre requisições de scraping/API (default: 0.8)
  --state AC,SP,... Processa apenas os estados listados
  --no-resume       Ignora manifest existente e recoleta tudo

COMPORTAMENTO DE DOWNLOAD:
  - Arquivos já presentes em DOWNLOAD_DIR são pulados automaticamente,
    inclusive os baixados manualmente antes de rodar o script.
  - Cada worker usa sua própria sessão HTTP (necessário pois o SGB limita
    a velocidade por conexão, não por IP).
  - Usa arquivo .part durante o download; renomeia ao concluir.

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
import threading
import concurrent.futures
from datetime import datetime
from pathlib import Path
from rich.progress import (
    Progress, BarColumn, DownloadColumn, TransferSpeedColumn,
    TimeRemainingColumn, TextColumn, TaskProgressColumn,
)
from rich.console import Console

_console = Console()

# ── Configurações de caminho ───────────────────────────────────────────────────
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pipeline.py").exists())
sys.path.insert(0, str(_ROOT))
from src import config as cfg  # noqa: E402

DOWNLOAD_DIR   = cfg.RAW_DIR / "sgb/raw_zips"
MANIFEST_PATH  = cfg.RAW_DIR / "sgb/00_sgb_manifest.csv"
INVENTORY_PATH = cfg.RAW_DIR / "sgb/01_sgb_inventory.csv"

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
DEFAULT_PAGE_DELAY = 0.8  # entre requisições de scraping/API
DEFAULT_WORKERS    = 6     # downloads paralelos (SGB limita a 6 simultâneos)

# ── Seleção do ZIP correto dentro de cada item do DSpace ──────────────────────
# Cada item pode ter vários ZIPs (vetorial, ortofoto, MDE, base cartográfica…).
# Ordem de prioridade para ZIPs de suscetibilidade:
#   1. sig_*, suscet*, suscept*  — contêm explicitamente dados de suscetibilidade
#   2. bc_* (base cartográfica)  — em vários municípios é onde estão os SHPs de suscetibilidade
#   3. arquivos_vetoriais_*      — vetorial genérico; frequentemente sem suscetibilidade
#   4. qualquer outro não-excluído
# Excluídos: mde_ (modelo de elevação), imagens_, ortofoto (imagens)
_ZIP_PREFERRED = re.compile(r"(^sig_|suscet|suscept)", re.IGNORECASE)
_ZIP_EXCLUDED  = re.compile(r"(^mde_|^imagens_|ortofoto)", re.IGNORECASE)
_ZIP_BC        = re.compile(r"^bc_",                re.IGNORECASE)
_ZIP_VETORIAL  = re.compile(r"^arquivos_vetoriais_", re.IGNORECASE)

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


def _select_best_zip(bitstreams: list[dict]) -> dict | None:
    """
    Dentre os bitstreams ZIP de uma bundle, escolhe o de dados de suscetibilidade.
    Prioridade (ver constantes _ZIP_* acima):
      1. sig_*, suscet*, suscept*  — suscetibilidade explícita no nome
      2. bc_* (base cartográfica)  — em vários municípios contém os SHPs de suscetibilidade
      3. arquivos_vetoriais_*      — vetorial genérico, frequentemente sem suscetibilidade
      4. qualquer outro não-excluído
      5. último recurso: qualquer ZIP (incluindo mde_, imagens_ etc.)
    Retorna None se não houver nenhum ZIP.
    """
    # Deduplica por nome (DSpace às vezes registra o mesmo arquivo duas vezes)
    seen: set[str] = set()
    zips = []
    for bs in bitstreams:
        name = bs.get("name", "")
        if name.lower().endswith(".zip") and name not in seen:
            seen.add(name)
            zips.append(bs)
    if not zips:
        return None

    preferred = [bs for bs in zips
                 if _ZIP_PREFERRED.search(bs["name"]) and not _ZIP_EXCLUDED.search(bs["name"])]
    if preferred:
        if len(preferred) > 1:
            names = [bs["name"] for bs in preferred]
            print(f"\n    [AVISO] Múltiplos ZIPs de suscetibilidade: {names} → usando {names[0]}")
        return preferred[0]

    bc_zips  = [bs for bs in zips if _ZIP_BC.search(bs["name"])]
    vet_zips = [bs for bs in zips if _ZIP_VETORIAL.search(bs["name"])]

    if bc_zips or vet_zips:
        if bc_zips and vet_zips:
            all_names = [bs["name"] for bs in bc_zips + vet_zips]
            print(f"\n    [AVISO] Sem sig_/suscet; encontrados {all_names}"
                  f" → preferindo bc_ sobre arquivos_vetoriais_")
            return bc_zips[0]
        return (bc_zips or vet_zips)[0]

    # Qualquer ZIP que não seja mde_, imagens_, ortofoto
    other = [bs for bs in zips if not _ZIP_EXCLUDED.search(bs["name"])]
    if other:
        all_names = [bs["name"] for bs in zips]
        print(f"\n    [AVISO] Sem padrão reconhecido em {all_names} → usando {other[0]['name']}")
        return other[0]

    # Último recurso: usa o primeiro mesmo que seja excluído
    names = [bs["name"] for bs in zips]
    print(f"\n    [AVISO] Apenas ZIPs excluídos disponíveis: {names} → usando {names[0]}")
    return zips[0]


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
            chosen = _select_best_zip(bitstreams)
            if chosen:
                filename = chosen["name"]
                zip_url  = (
                    chosen.get("_links", {}).get("content", {}).get("href", "")
                    or f"{RIGEO_BASE}/bitstreams/{chosen['uuid']}/download"
                )
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

            # Se já está no manifest e tem URL de download, pula —
            # mas re-busca se o arquivo gravado for claramente não-vetorial
            if resume and rigeo_url in existing:
                cached = existing[rigeo_url]
                cached_file = cached.get("filename", "")
                if cached_file and _ZIP_EXCLUDED.search(cached_file):
                    print(f"    [RE-FETCH] Arquivo errado em cache"
                          f" ({cached_file}): {mun['nm_municipio']}")
                    # não faz skip — cai no bloco de API abaixo
                elif cached.get("url_download") or cached.get("status") in ("sem_dado", "ok"):
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

_manifest_lock = threading.Lock()
_cancel_event  = threading.Event()

# ── Detecção e tratamento da página de confirmação do Google Drive ─────────────

def _resolve_gdrive_confirmation(session: requests.Session,
                                  original_url: str,
                                  html_resp: requests.Response) -> requests.Response:
    """
    O Google Drive retorna uma página HTML de aviso de vírus para arquivos grandes
    (geralmente > 100 MB) em vez do arquivo diretamente. Esta função extrai a URL
    de download real da página e refaz a requisição.

    Padrões suportados:
      - Link direto drive.usercontent.google.com (Google Drive moderno)
      - Form action com inputs hidden (fallback)
      - Parâmetro confirm= na URL original (fallback final)
    """
    html = html_resp.text

    # Padrão 1: link direto no botão "baixar assim mesmo" (Drive moderno)
    m = re.search(r'href="(https://drive\.usercontent\.google\.com/download[^"]+)"', html)
    if m:
        return session.get(m.group(1).replace("&amp;", "&"), timeout=600, stream=True)

    # Padrão 2: form action com hidden inputs
    action = re.search(r'<form[^>]+action="([^"]+)"', html)
    if action:
        base = action.group(1).replace("&amp;", "&")
        hidden = re.findall(
            r'<input[^>]+type=["\']hidden["\'][^>]+name=["\']([^"\']+)["\'][^>]+value=["\']([^"\']*)["\']',
            html,
        )
        hidden += re.findall(
            r'<input[^>]+name=["\']([^"\']+)["\'][^>]+type=["\']hidden["\'][^>]+value=["\']([^"\']*)["\']',
            html,
        )
        if hidden:
            params = "&".join(f"{k}={v}" for k, v in hidden)
            return session.get(f"{base}?{params}", timeout=600, stream=True)

    # Padrão 3: extrai token confirm= e adiciona à URL original
    confirm = re.search(r'[?&]confirm=([0-9A-Za-z_\-]+)', html)
    if confirm:
        sep = "&" if "?" in original_url else "?"
        return session.get(f"{original_url}{sep}confirm={confirm.group(1)}",
                           timeout=600, stream=True)

    # Fallback: confirm=t funciona na maioria dos casos modernos do Drive
    sep = "&" if "?" in original_url else "?"
    return session.get(f"{original_url}{sep}confirm=t", timeout=600, stream=True)


def _download_one(record: dict, i: int, total: int, records: list,
                  dest_dir: Path, progress: Progress) -> str:
    """
    Baixa um único arquivo com barra de progresso individual.
    Retorna 'skip', 'ok', 'error' ou 'cancelled'.
    Interrompível via _cancel_event (Ctrl+C no processo principal).
    """
    filename = record["filename"]
    url      = record["url_download"]
    dest     = dest_dir / filename
    label    = f"{record['nm_municipio']} ({record['cd_estado']})"

    # ── Arquivo já existe no disco? (inclui downloads manuais) ────────────────
    if dest.exists():
        size_mb = dest.stat().st_size / 1_048_576
        progress.console.print(
            f"  [{i:>3}/{total}] ↷  {label} — já existe ({size_mb:.0f} MB)"
        )
        record["status"]        = "ok"
        record["downloaded_at"] = (record.get("downloaded_at")
                                   or datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        with _manifest_lock:
            save_manifest(records)
        return "skip"

    # ── Ctrl+C foi acionado antes de iniciar este download? ───────────────────
    if _cancel_event.is_set():
        return "cancelled"

    tmp    = dest.with_suffix(".part")
    offset = tmp.stat().st_size if tmp.exists() else 0

    task_id = progress.add_task(f"{label}", total=None, completed=offset)

    try:
        session = make_session()
        req_headers = {"Range": f"bytes={offset}-"} if offset > 0 else {}
        resp = session.get(url, timeout=600, stream=True, headers=req_headers)

        # Google Drive retorna HTML (página de confirmação de vírus) para arquivos > ~100 MB.
        # Detecta pelo Content-Type e resolve antes de começar a gravar.
        if "text/html" in resp.headers.get("content-type", ""):
            progress.console.print(
                f"  [{i:>3}/{total}] ↻  {label} — página de confirmação do Google Drive, resolvendo..."
            )
            resp = _resolve_gdrive_confirmation(session, url, resp)
            # Descarta .part de tentativas anteriores: não dá para retomar com nova URL
            offset = 0
            if tmp.exists():
                tmp.unlink()

        if resp.status_code == 206:
            # Servidor aceita range: retoma do offset
            content_len   = int(resp.headers.get("content-length", 0)) or None
            total_bytes   = (offset + content_len) if content_len else None
            file_mode     = "ab"
            if offset > 0:
                progress.console.print(
                    f"  [{i:>3}/{total}] ↩  {label} — retomando de {offset/1_048_576:.1f} MB"
                )
        else:
            # Servidor retornou 200 (não suporta range) ou outro código
            resp.raise_for_status()
            total_bytes = int(resp.headers.get("content-length", 0)) or None
            file_mode   = "wb"
            offset      = 0
            if tmp.exists():
                tmp.unlink()

        progress.update(task_id, total=total_bytes, completed=offset)

        with open(tmp, file_mode) as fout:
            for chunk in resp.iter_content(chunk_size=512 * 1024):
                if _cancel_event.is_set():
                    raise KeyboardInterrupt
                if chunk:
                    fout.write(chunk)
                    progress.advance(task_id, len(chunk))

        tmp.rename(dest)

        # Valida magic bytes: todo ZIP válido começa com b"PK".
        # Se o conteúdo for HTML (confirmação do Drive não resolvida), detecta aqui.
        with open(dest, "rb") as _fv:
            magic = _fv.read(2)
        if magic != b"PK":
            dest.unlink()
            raise RuntimeError(
                f"Arquivo baixado não é ZIP válido (magic: {magic!r}). "
                "Provável página HTML do Google Drive não resolvida."
            )

        size_mb = dest.stat().st_size / 1_048_576
        progress.remove_task(task_id)
        progress.console.print(
            f"  [{i:>3}/{total}] ✓  {label} — {filename} ({size_mb:.1f} MB)"
        )

        record["status"]        = "ok"
        record["downloaded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with _manifest_lock:
            save_manifest(records)
        return "ok"

    except KeyboardInterrupt:
        progress.remove_task(task_id)
        # Mantém o .part para retomar na próxima execução
        return "cancelled"

    except Exception as e:
        progress.remove_task(task_id)
        progress.console.print(f"  [{i:>3}/{total}] ✗  {label}: {e}")
        record["status"] = "error"
        # Mantém o .part — pode ser erro transitório, retoma na próxima execução
        with _manifest_lock:
            save_manifest(records)
        return "error"


def download_files(workers:      int = DEFAULT_WORKERS,
                   state_filter: list[str] | None = None) -> None:
    """
    Fase 2: Baixa os arquivos ZIP do manifest em paralelo para DOWNLOAD_DIR.
    - Mostra barra de progresso individual por download (speed + ETA).
    - Pula arquivos já presentes no disco (inclusive downloads manuais).
    - Cada worker usa sua própria sessão HTTP (SGB limita 500 KB/s por conexão).
    - Salva o manifest de forma thread-safe após cada arquivo.
    - Usa arquivo .part durante o download; renomeia ao concluir.
    - Ctrl+C termina os downloads em andamento limpos antes de sair.
    """
    if not MANIFEST_PATH.exists():
        print("[ERRO] Manifest não encontrado. Execute 'collect' primeiro.")
        sys.exit(1)

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _cancel_event.clear()

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
    _console.print(f"\n[FASE 2] {total_to_dl} arquivos a baixar  |  {workers} workers paralelos")
    _console.print(f"         Destino: {DOWNLOAD_DIR}\n")

    counts: dict[str, int] = {"ok": 0, "skip": 0, "error": 0, "cancelled": 0}

    with Progress(
        TextColumn("{task.description:<28}"),
        BarColumn(bar_width=35),
        TaskProgressColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=_console,
    ) as progress:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _download_one, record, i, total_to_dl, records, DOWNLOAD_DIR, progress
                ): record
                for i, record in enumerate(to_download, 1)
            }
            try:
                for fut in concurrent.futures.as_completed(futures):
                    result = fut.result()
                    counts[result] = counts.get(result, 0) + 1
            except KeyboardInterrupt:
                _cancel_event.set()
                progress.console.print(
                    "\n[bold yellow][INTERROMPIDO][/bold yellow] "
                    "Aguardando downloads em andamento finalizarem..."
                )

    interrupted = _cancel_event.is_set()
    _console.print(f"\n{'═'*60}")
    _console.print(f"FASE 2 {'INTERROMPIDA' if interrupted else 'CONCLUÍDA'}")
    _console.print(f"  Baixados com sucesso:  {counts['ok']}")
    _console.print(f"  Já existiam no disco:  {counts['skip']}")
    _console.print(f"  Erros:                 {counts['error']}")
    if counts["cancelled"]:
        _console.print(f"  Cancelados (Ctrl+C):   {counts['cancelled']}")
    _console.print(f"  Manifest atualizado:   {MANIFEST_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# RE-DOWNLOAD DE CORROMPIDOS
# ══════════════════════════════════════════════════════════════════════════════

_MIN_ZIP_SIZE_KB = 50  # mesmo limiar do script de exploração

def redownload_corrupt(workers: int = DEFAULT_WORKERS) -> None:
    """
    Detecta ZIPs corrompidos DIRETAMENTE em raw_zips/ (abre cada arquivo com
    zipfile.ZipFile) e re-baixa apenas esses arquivos. Não depende do estado
    do sgb_inventory.csv — é seguro rodar mesmo após re-executar o explore.

    Fluxo recomendado:
      python 00_sgb_scraper.py redownload   # detecta e re-baixa corrompidos
      python 01_sgb_explore.py              # re-escaneia automaticamente
    """
    import zipfile as _zipfile

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    zip_files = sorted(DOWNLOAD_DIR.glob("*.zip"))
    if not zip_files:
        print("[AVISO] Nenhum ZIP em raw_zips/. Execute 'download' primeiro.")
        return

    print(f"\n[REDOWNLOAD] Verificando {len(zip_files)} ZIPs em raw_zips/...", flush=True)

    corrupt_names: set[str] = set()
    for idx, zp in enumerate(zip_files, 1):
        print(f"\r  verificando {idx}/{len(zip_files)}…", end="", flush=True)
        if zp.stat().st_size / 1024 < _MIN_ZIP_SIZE_KB:
            corrupt_names.add(zp.name)
            continue
        try:
            with _zipfile.ZipFile(zp) as zf:
                zf.namelist()   # verifica estrutura central do ZIP
        except Exception:
            corrupt_names.add(zp.name)
    print()

    if not corrupt_names:
        print(f"  ✓ Todos os {len(zip_files)} ZIPs estão íntegros.")
        return

    print(f"  {len(corrupt_names)} ZIPs corrompidos encontrados\n")

    if not MANIFEST_PATH.exists():
        print("[ERRO] Manifest não encontrado. Execute 'collect' primeiro.")
        sys.exit(1)

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    manifest_by_filename = {r["filename"]: r for r in records if r.get("filename")}

    reset_count = 0
    not_in_manifest: list[str] = []

    for name in sorted(corrupt_names):
        dest = DOWNLOAD_DIR / name
        if dest.exists():
            size_mb = dest.stat().st_size / 1_048_576
            dest.unlink()
            print(f"  ✗ Removido: {name}  ({size_mb:.1f} MB)")
        part = dest.with_suffix(".part")
        if part.exists():
            part.unlink()

        if name in manifest_by_filename:
            manifest_by_filename[name]["status"] = ""
            manifest_by_filename[name]["downloaded_at"] = ""
            reset_count += 1
        else:
            not_in_manifest.append(name)

    if not_in_manifest:
        print(f"\n  [AVISO] {len(not_in_manifest)} arquivo(s) sem URL no manifest "
              f"(não podem ser re-baixados automaticamente):")
        for n in not_in_manifest[:10]:
            print(f"      {n}")
        if len(not_in_manifest) > 10:
            print(f"      … e mais {len(not_in_manifest) - 10}")
        print("  Execute 'collect' para re-buscar as URLs.")

    if reset_count == 0:
        print("\n  [AVISO] Nenhum dos ZIPs corrompidos tem URL no manifest.")
        print("  Execute 'collect' para re-buscar as URLs.")
        return

    save_manifest(records)
    print(f"\n  {reset_count} entrada(s) resetadas no manifest.")
    print(f"  Iniciando re-download com {workers} workers...\n")
    download_files(workers=workers)


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
        choices=["collect", "download", "all", "report", "redownload"],
        help="collect|download|all|report|redownload",
    )
    parser.add_argument("--page-delay", type=float, default=DEFAULT_PAGE_DELAY,
                        help="Segundos entre requisições de scraping/API (default: 0.8)")
    parser.add_argument("--workers",    type=int,   default=DEFAULT_WORKERS,
                        help=f"Downloads paralelos (default: {DEFAULT_WORKERS}). "
                             "SGB permite no máximo 6 simultâneos.")
    parser.add_argument("--state",     type=str,   default=None,
                        help="Siglas separadas por vírgula, ex: SP,RJ")
    parser.add_argument("--no-resume", action="store_true")

    args = parser.parse_args()
    state_filter = [s.strip().upper() for s in args.state.split(",")] if args.state else None
    resume       = not args.no_resume

    if args.phase == "report":
        print_report()
        return

    if args.phase == "redownload":
        redownload_corrupt(workers=args.workers)
        return

    if args.phase in ("collect", "all"):
        collect_all_links(
            state_filter=state_filter,
            resume=resume,
            page_delay=args.page_delay,
        )

    if args.phase in ("download", "all"):
        download_files(
            workers=args.workers,
            state_filter=state_filter,
        )


if __name__ == "__main__":
    main()