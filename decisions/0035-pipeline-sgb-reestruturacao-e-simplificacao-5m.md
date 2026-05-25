# ADR-0035: Reestruturar pipeline SGB em 6 scripts e adicionar simplificação de 5 m

## Status
Accepted — 2026-05-25

## Contexto

O pipeline SGB anterior tinha um único script `02_sgb_harmonize.py` responsável por:
1. Ler cada ZIP de `raw_zips/` e extrair SHP/GPKG/TIF
2. Normalizar CRS, corrigir geometrias e mapear classes
3. Consolidar tudo em dois GeoPackages nacionais (`02_sgb_floods_br.gpkg`, `02_sgb_mass_br.gpkg`)

Dois problemas surgiram em produção:

**Problema 1 — Desempenho e tamanho de arquivo:** O script era lento e produzia arquivos difíceis de trabalhar (abrindo mal no QGIS). Causas identificadas:
- Geometrias em resolução completa (1:25.000, sem qualquer simplificação) escritas no GPKG consolidado.
- Uso de `.apply()` Python para converter geometrias para MultiPolygon — loop puro, lento para feições com muitos vértices.
- Em Windows, `tempfile.TemporaryDirectory()` travava ao tentar limpar o diretório temporário enquanto o fiona ainda mantinha o handle do shapefile aberto (comportamento específico do filesystem Windows).

**Problema 2 — Ausência de acesso por município:** Não havia forma de inspecionar ou validar os dados brutos de um município individual sem reprocessar o ZIP. Todo o dado estava consolidado em dois arquivos nacionais pesados.

## Decisão

### 1. Dividir em dois scripts separados

| Script antigo | Scripts novos |
|---|---|
| `02_sgb_harmonize.py` | `02_sgb_extract.py` + `03_sgb_harmonize.py` |

**`02_sgb_extract.py`** — extração e harmonização por município:
- Lê cada ZIP e grava em `por_municipio/{UF}/{cd_mun}_{nm}/`
- Subfolder `raw/` com os arquivos originais intactos do ZIP
- Um GPKG por tipo (`{cd_mun}_inundacao.gpkg`, `{cd_mun}_massa.gpkg`) com geometria em resolução completa e colunas padronizadas
- **Sem simplificação**

**`03_sgb_harmonize.py`** — consolidação com simplificação:
- Lê os GPKGs por município gerados pelo 02
- Aplica simplificação de **5 m** em EPSG:5880 antes de escrever
- Consolida em `03_sgb_floods_br.gpkg` e `03_sgb_mass_br.gpkg`

Os scripts seguintes foram renumerados: `03→04`, `04→05`, `05→06`.

### 2. Simplificação de 5 m no script 03 (consolidação)

A simplificação de 5 m é aplicada apenas na etapa de consolidação nacional, mantendo os GPKGs por município em resolução completa para referência.

O pipeline passa a ter dois estágios de simplificação:
- **5 m** no `03_sgb_harmonize.py` — para tornar os arquivos consolidados manejáveis
- **20 m** no `04_sgb_h3_intersect.py` — para performance na interseção com H3 (ADR-0034)

### 3. Correções de compatibilidade e desempenho no 02

- `tempfile.TemporaryDirectory(ignore_cleanup_errors=True)` — evita travamento no Windows quando o fiona mantém handle aberto ao fim do bloco `with`.
- `_apply_multipolygon()` via list comprehension sobre `.values` no lugar de `.apply()` — evita overhead do pandas por linha.
- `encoding="utf-8"` explícito em todos os `open()` — comportamento padrão difere entre Mac (UTF-8) e Windows (cp1252).

## Justificativa para 5 m

| Parâmetro | Valor |
|---|---|
| Precisão nominal 1:25.000 | ~5–10 m no campo |
| Simplificação escolhida | 5 m |
| Simplificação na interseção H3 (ADR-0034) | 20 m |

- **Dentro da precisão nominal do dado:** mapeamentos 1:25.000 têm precisão posicional de ~5–10 m. Simplificar a 5 m não piora a qualidade real dos polígonos.
- **Redução de tamanho significativa:** geometrias com milhares de vértices perdem os micro-detalhes sub-métricos sem alterar a forma visual dos polígonos. Estimativa de redução de arquivo: 3–8×.
- **Preserva dados por município intactos:** `por_municipio/` mantém resolução completa, permitindo inspeção visual sem necessidade de re-extrair ZIPs.
- **Compatível com a simplificação de 20 m em 04:** simplificar a 5 m e depois a 20 m dá resultado equivalente a simplificar diretamente a 20 m, sem perda adicional.

### Alternativas consideradas

- **Manter simplificação só no 04 (20 m):** os GPKGs consolidados continuariam pesados e lentos no QGIS.
- **Mover a simplificação de 20 m para o 03 e remover do 04:** possível, mas o ADR-0034 documenta que a simplificação no 04 serve especificamente para evitar hang no GEOS durante a interseção. Manter ambas as etapas preserva a rastreabilidade de cada decisão.
- **Simplificação por região em vez de arquivo nacional:** descartado — adiciona complexidade desnecessária nos scripts downstream sem ganho proporcional dado que a simplificação já resolve o problema de tamanho.

## Consequências

**Positivas:**
- GPKGs consolidados (03) são 3–8× menores — abrem normalmente no QGIS.
- Script 04 (interseção H3) fica mais rápido — lê geometrias já simplificadas.
- Cada município tem seus dados acessíveis em `por_municipio/` sem re-extração.
- Pipeline compatível com Mac e Windows.

**Negativas / trade-offs:**
- O script 03 agora depende do 02 ter rodado primeiro (antes eram independentes; o antigo 02 lia diretamente os ZIPs).
- Tempo total de processamento aumenta ligeiramente porque o 02 escreve GPKGs individuais que o 03 lê novamente — trade-off aceito pelo ganho em organização e rastreabilidade.
- `por_municipio/` ocupa espaço em disco adicional (mesma informação do que está nos GPKGs consolidados, mas em resolução completa).

## Referências
- ADR-0034: simplificação de 20 m antes da interseção H3.
- [etl/exposure/sgb/02_sgb_extract.py](../etl/exposure/sgb/02_sgb_extract.py)
- [etl/exposure/sgb/03_sgb_harmonize.py](../etl/exposure/sgb/03_sgb_harmonize.py)
