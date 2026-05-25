"""
SGB Pipeline — Helper de logging de falhas por município
=========================================================
Módulo compartilhado pelos scripts 02_sgb_extract.py e 03_sgb_harmonize.py
para registrar de forma estruturada as falhas por (município, tipo) que
impedem o município de chegar até as análises 05/06.

O reconciliador 08_sgb_pipeline_status.py lê esses CSVs para montar a
tabela única de status do pipeline.

Schema: ver FAILURE_COLS abaixo.

Campos preenchidos por etapa:
  02_failures.csv  → cd_mun, sigla_uf, nm_municipio, tipo, stage, reason
                     (todos preenchidos pois 02 conhece o IBGE via manifest)
  03_failures.csv  → mun_id, sigla_uf, tipo, stage, reason
                     (cd_mun pode vir vazio; 08 reconcilia via slug)

Reexecutabilidade:
  - Use reset_failures() no início de um run completo (sem --resume).
  - Em --resume, NÃO chame reset_failures — o CSV é append-only.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

FAILURE_COLS = [
    "timestamp", "cd_mun", "sigla_uf", "nm_municipio", "mun_id",
    "tipo", "stage", "reason",
]


def log_failure(
    path: Path,
    *,
    stage: str,
    tipo: str,
    reason: str,
    cd_mun: str = "",
    sigla_uf: str = "",
    nm_municipio: str = "",
    mun_id: str = "",
) -> None:
    """Anexa uma linha de falha ao CSV. Cria o arquivo + header se necessário."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FAILURE_COLS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp":   datetime.now().isoformat(timespec="seconds"),
            "cd_mun":      str(cd_mun) if cd_mun else "",
            "sigla_uf":    sigla_uf or "",
            "nm_municipio": nm_municipio or "",
            "mun_id":      mun_id or "",
            "tipo":        tipo,
            "stage":       stage,
            "reason":      str(reason)[:500],
        })


def reset_failures(path: Path) -> None:
    """Remove o CSV de falhas — chamar apenas em runs NÃO-resume."""
    if path.exists():
        path.unlink()
