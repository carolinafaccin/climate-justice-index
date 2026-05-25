# ADR-0037: Rastrear dropout de municípios no pipeline SGB com failures.csv e reconciliador

## Status
Accepted — 2026-05-25

## Contexto

O pipeline SGB vai de ~820 municípios coletados pelo scraper a ~600 municípios
que efetivamente entram nas análises 05 (calibração E1) e 06 (validação E2).
Os ~220 municípios perdidos caem em etapas diferentes e por razões diferentes:

- **Download** falha: HTTP error, timeout, ZIP corrompido no servidor
- **Explore (01)**: ZIP corrompido localmente, sem layer de suscetibilidade, sem classes válidas
- **Extract (02)**: SHP/GPKG ilegível, geometrias inválidas, falha ao escrever GPKG de saída
- **Harmonize (03)**: vazio após simplificação 5 m, falha ao escrever no GPKG consolidado

Antes desta ADR, as falhas ficavam apenas no stdout das execuções — sem
persistência estruturada, sem diferenciação por tipo (massa vs inundação),
e sem forma de verificar quais municípios caíram em qual etapa.

Adicionalmente, um município pode ter dados apenas de um tipo (só massa ou só
inundação), portanto a contagem de exclusões deve ser por `(município, tipo)`,
não apenas por município.

## Decisão

Implementar um sistema de rastreamento em três peças:

### 1. Módulo helper [`_pipeline_log.py`](../etl/exposure/sgb/_pipeline_log.py)
Função `log_failure(path, *, stage, tipo, reason, cd_mun, sigla_uf, nm_municipio, mun_id)`
que anexa uma linha a um CSV de falhas. Função `reset_failures(path)` que
apaga o CSV no início de runs não-resume. Módulo compartilhado por 02 e 03.

### 2. Patches mínimos em [`02_sgb_extract.py`](../etl/exposure/sgb/02_sgb_extract.py) e [`03_sgb_harmonize.py`](../etl/exposure/sgb/03_sgb_harmonize.py)
- `02_failures.csv`: registra falhas de extração com `cd_mun`, `sigla_uf`,
  `nm_municipio`, `tipo` e `reason` no momento em que ocorrem. Apagado e
  recriado em runs completos; preservado em `--resume`.
- `03_failures.csv`: análogo, com `mun_id` (slug) em vez de `cd_mun` (03 não
  carrega o inventário IBGE). O reconciliador 08 converte via slug reverso.
- Reexecutabilidade: `reset_failures()` é chamado automaticamente nos
  runs sem `--resume`; em `--resume` o CSV é append-only (deduplicado por
  timestamp no 08).

### 3. Script reconciliador [`08_sgb_pipeline_status.py`](../etl/exposure/sgb/08_sgb_pipeline_status.py)
Lê todos os artefatos (manifest, cobertura, progress.json, failures.csv) e
produz `sgb_pipeline_status.csv` com **uma linha por município** e colunas
separadas por tipo:

| Coluna | Descrição |
|---|---|
| `status_download` | ok / failed / sem_dado |
| `status_explore_{tipo}` | ok / failed / n_a (sem layer) / not_processed |
| `status_extract_{tipo}` | ok / failed / not_processed / n_a |
| `status_harmonize_{tipo}` | ok / failed / not_processed / n_a |
| `in_pipeline_{tipo}` | bool — proxy para "chegou ao 04+" |
| `last_failure_stage_{tipo}` | etapa onde o município caiu |
| `last_failure_reason_{tipo}` | razão da falha |

O 08 também imprime um sumário do funil de dropout (quantos caem em cada
etapa por tipo) e uma matriz de municípios em ambos, só massa, só inundação
e em nenhum.

## Alternativas consideradas

- **Logging inline + arquivo de log unificado (long format)**: mais flexível
  mas requer mais parsing no 08 e dificulta leitura direta do CSV.
  Rejeitada em favor do formato wide pedido pela usuária.

- **Modificar 00 e 01 para também escrever failures.csv**: 00 e 01 já
  produzem artefatos ricos (`00_sgb_manifest.csv` com coluna `status` e
  `01_sgb_cobertura.csv` com `status_zip`, `has_inundacao`, `has_massa`).
  Não há necessidade de duplicar — o 08 lê esses arquivos diretamente.

- **Inferência pós-hoc pura sem modificar 02 e 03**: possível, mas as razões
  de falha em 02/03 ficam apenas no stdout. O patch mínimo (~10 linhas cada)
  preserva a razão exata e torna o diagnóstico muito mais útil.

- **Tracking em banco de dados ou SQLite**: excessivo para o volume (~820
  municípios) e cria dependência desnecessária. CSV é suficiente.

## Consequências

- **Positivas**:
  - Visibilidade completa do funil: possível identificar em qual etapa cada
    município cai e por quê, sem reler logs de execução.
  - Separação por tipo: municípios que têm só massa (E1) ou só inundação (E2)
    são tratados corretamente, sem inflar/deflate as contagens.
  - Reexecutável: 08 pode ser rodado a qualquer momento para refletir o
    estado atual do pipeline.
  - Rastreável: `sgb_pipeline_status.csv` pode ser commitado como evidência
    da cobertura real usada nas análises.

- **Negativas / trade-offs**:
  - 04 (H3 intersect) e 05/06 não são rastreados ao nível de município por
    limitação dos artefatos (o parquet H3 não tem `cd_mun`). O `in_pipeline`
    é proxy conservador: indica que o município chegou ao 03, mas não confirma
    que gerou hexágonos com `coverage_frac >= 0.5` para as análises 05/06.
  - `03_failures.csv` usa `mun_id` (slug) em vez de `cd_mun` — o 08 resolve
    via slug reverso, mas slugs não são garantidamente únicos (nomes muito
    similares poderiam colidir). Na prática, improvável no Brasil.
  - Patches em 02 e 03 adicionam ~20 linhas de código de rastreamento.
    Se os scripts forem refatorados, o rastreamento precisa migrar junto.

- **Confiança**: Alta — o sistema é simples, baseado em append de CSV, e
  falha silenciosamente (se o log não for escrito, o pipeline continua).

## Referências

- [`_pipeline_log.py`](../etl/exposure/sgb/_pipeline_log.py) — helper de logging
- [`02_sgb_extract.py`](../etl/exposure/sgb/02_sgb_extract.py) — patches de failures
- [`03_sgb_harmonize.py`](../etl/exposure/sgb/03_sgb_harmonize.py) — patches de failures
- [`08_sgb_pipeline_status.py`](../etl/exposure/sgb/08_sgb_pipeline_status.py) — reconciliador
- [ADR-0035](0035-pipeline-sgb-reestruturacao-e-simplificacao-5m.md) — estrutura do pipeline SGB
