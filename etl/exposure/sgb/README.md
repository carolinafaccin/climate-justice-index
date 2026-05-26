# SGB — Como executar o pipeline

Pipeline para baixar, inventariar, harmonizar e usar a Cartografia de Suscetibilidade
do SGB/CPRM para calibrar os indicadores E1 (deslizamentos) e E2 (inundações) do IIC.
Execute os scripts **na ordem numérica**. Cada script depende dos outputs do anterior.

## Pré-requisitos

```bash
# Ativa o virtualenv do projeto (a partir da raiz do repositório)
source venv/bin/activate

# Garante que config/config.local.json existe com o caminho dos dados
# Exemplo: {"data_dir": "/Users/lina/data/climate-injustice-index/"}
```

Dependências: `geopandas`, `fiona`, `requests`, `beautifulsoup4`, `rich`, `h3`.
Espaço em disco estimado: ~50–100 GB para os ~814 ZIPs do SGB.

______________________________________________________________________

## Script 00 — Download

**O que faz:** Raspa o site do SGB, coleta metadados e links de download de todos os
municípios, e baixa os ZIPs em paralelo para `raw_zips/`.

**Output principal:** `00_sgb_manifest.csv`

```bash
# Coleta links e metadados (raspa o site SGB + API DSpace)
python etl/exposure/sgb/00_sgb_scraper.py collect

# Baixa os ZIPs (retoma automaticamente se interrompido)
python etl/exposure/sgb/00_sgb_scraper.py download

# Ou os dois em sequência:
python etl/exposure/sgb/00_sgb_scraper.py all

# Resumo do que foi baixado:
python etl/exposure/sgb/00_sgb_scraper.py report

# Re-baixa ZIPs corrompidos ou incompletos (rodar depois do 01 se houver erros):
python etl/exposure/sgb/00_sgb_scraper.py redownload
```

**Opções úteis:**

```bash
--workers 10        # downloads paralelos (padrão: 6)
--state SP,RJ       # filtra por estado (para testes)
--no-resume         # ignora manifest existente e recoleta tudo do zero
```

**Checklist antes de avançar:**

- `report` mostra status `ok` para a maioria dos municípios
- Pasta `raw_zips/` tem os ZIPs (pode verificar com `ls raw_zips/ | wc -l`)

______________________________________________________________________

## Script 01 — Exploração e Inventário

**O que faz:** Abre cada ZIP, lista os shapefiles/GPKGs/rasters internos, detecta qual é
inundação vs massa vs outros, lê metadados (colunas, valores de classe, nº de feições) e
gera o inventário. É incremental: Ctrl+C salva o progresso.

**Outputs:**

- `01_sgb_inventory.csv` — um registro por arquivo por ZIP, com coluna `revisar`
- `01_sgb_coverage.csv` — uma linha por ZIP com `status_zip`, `has_inundacao`, `has_massa`
- `01_sgb_mapping.json` — rascunho do mapeamento textual → 0-5 **(editar antes do script 02)**

```bash
# Processa ZIPs ainda não inventariados (retoma de onde parou)
python etl/exposure/sgb/01_sgb_explore.py

# Testa com os próximos 5 ZIPs não processados
python etl/exposure/sgb/01_sgb_explore.py --limit 5

# Filtra por estado
python etl/exposure/sgb/01_sgb_explore.py --state SE,BA

# Reprocessa tudo do zero (descarta inventário existente)
python etl/exposure/sgb/01_sgb_explore.py --redo
```

### O que verificar nos outputs

**`01_sgb_inventory.csv`** — filtre pela coluna `revisar != ''`:

- `sem_classe` → preencha `classe_col` com o nome da coluna de classe do shapefile
- `zip_erro` → ZIP corrompido, rodar `00_sgb_scraper.py redownload`
- `leitura_erro` → arquivo com CRC ruim, verificar se precisa re-baixar o ZIP

**`01_sgb_coverage.csv`** — filtre por `status_zip`:

- `sem_cobertura` → ZIP teve arquivos mas nenhum era inundação ou massa (pode ser legítimo)
- `zip_erro` / `zip_vazio` → mesmos casos acima, visão por ZIP inteiro

**`01_sgb_mapping.json`** — **obrigatório revisar antes do script 02**:

- Abra o arquivo e verifique os valores `-1` na seção `"mapping"`
- Cada `-1` significa "ainda não mapeado" — preencha com o inteiro correto (0–5)
- Escala: `5=Muito Alta`, `4=Alta`, `3=Média/Moderada`, `2=Baixa`, `1=Muito Baixa`, `0=Sem`
- Salve como JSON válido (sem vírgula no último item)

### Como editar `01_sgb_inventory.csv` manualmente

Abra no Excel, Numbers ou Google Sheets (UTF-8). Colunas que você pode editar:

| Coluna | Quando editar |
| --- | --- |
| `tipo` | Mude `outros` para `inundacao` ou `massa` se o arquivo foi mal classificado |
| `classe_col` | Preencha com o nome exato da coluna de classe (ex: `CLASSE`, `CLASSE_SU`) |
| `notes` | Adicione observações; não apague notas de erro existentes |

> **Não altere** `zip_filename` nem `shp_path_in_zip` — esses caminhos são usados para
> localizar os arquivos no script 02.

Após editar, rode o script 01 novamente (sem `--redo`) para regenerar os arquivos derivados
com as correções aplicadas.

______________________________________________________________________

## Script 02 — Extração por Município

**O que faz:** Para cada ZIP, extrai os arquivos brutos e gera um GeoPackage harmonizado
por município (CRS padronizado, make_valid, classe mapeada). Sem simplificação de geometria.

**Pré-requisito:** `01_sgb_mapping.json` revisado (sem valores `-1` não intencionais).

**Outputs em `por_municipio/{UF}/`:**

- `{sigla_uf}_{nm_mun}_inundacao.gpkg` — harmonizado, sem simplificação
- `{sigla_uf}_{nm_mun}_massa.gpkg` — harmonizado, sem simplificação

Os arquivos brutos permanecem nos ZIPs originais (`raw_zips/`).

```bash
# Teste sem escrever nada (recomendado na primeira vez)
python etl/exposure/sgb/02_sgb_extract.py --dry-run

# Processa tudo
python etl/exposure/sgb/02_sgb_extract.py

# Filtra por estado (para testes)
python etl/exposure/sgb/02_sgb_extract.py --state SE,BA

# Limita a N ZIPs (para teste rápido)
python etl/exposure/sgb/02_sgb_extract.py --limit 10
```

**Se aparecerem avisos de classe não mapeada:**

1. Anote os valores listados no aviso
1. Adicione-os em `01_sgb_mapping.json` com o inteiro correto
1. Re-execute o script 02

______________________________________________________________________

## Script 03 — Harmonização e Consolidação Nacional

**O que faz:** Lê os GPKGs por município gerados pelo 02, aplica simplificação de 5 m
(EPSG:5880) e consolida em dois GeoPackages nacionais.

**Pré-requisito:** `por_municipio/` populado pelo script 02.

**Outputs em `harmonized/`:**

- `03_sgb_floods_br.gpkg`
- `03_sgb_mass_br.gpkg`

```bash
# Processa tudo
python etl/exposure/sgb/03_sgb_harmonize.py

# Filtra por estado (para testes)
python etl/exposure/sgb/03_sgb_harmonize.py --state SE,BA

# Continua do ponto onde parou
python etl/exposure/sgb/03_sgb_harmonize.py --resume
```

______________________________________________________________________

## Script 04 — Interseção H3

**O que faz:** Para cada GeoPackage harmonizado, intersecta os polígonos SGB com a grade H3
res9, calculando a fração de área em classes Alta/Muito Alta (4–5) por hexágono. Processa
um estado por vez para manter uso de memória baixo.

**Pré-requisito:** `03_sgb_floods_br.gpkg` e `03_sgb_mass_br.gpkg` em `harmonized/`.

**Outputs em `data/inputs/clean/`:**

- `br_h3_sgb_massa.parquet`
- `br_h3_sgb_inundacoes.parquet`

```bash
# Processa ambos os tipos
python etl/exposure/sgb/04_sgb_h3_intersect.py

# Testa com um estado
python etl/exposure/sgb/04_sgb_h3_intersect.py --state SP

# Só movimentos de massa
python etl/exposure/sgb/04_sgb_h3_intersect.py --tipo massa

# Simula sem escrever saída
python etl/exposure/sgb/04_sgb_h3_intersect.py --dry-run
```

**Colunas principais de saída:**

- `sgb_alta_mta_frac` — fração da área SGB mapeada em classes 4–5 (usado em 05 e 06)
- `sgb_coverage_frac` — fração do hexágono coberta por dados SGB (filtrar `>= 0.5` para análise)
- `sgb_max_class` — classe máxima no hexágono

______________________________________________________________________

## Script 05 — Calibração E1

**O que faz:** Varre thresholds de `e1_des_abs` (= lhasa_high_frac) de 0.0 a 1.0 e
compara contra a referência SGB `sgb_alta_mta_frac > 0.3`. Calcula precision/recall/F1
por threshold e recomenda o ótimo. Também testa `lhasa_mean >= t` como variante e
analisa F1 por macrorregião.

**Pré-requisito:** parquet E1 (`br_h3_e1_deslizamentos.parquet`) + `br_h3_sgb_massa.parquet`

```bash
python etl/exposure/sgb/05_sgb_calibrate_e1.py

# Ajusta thresholds de referência (padrão: sgb-ref=0.3, min-coverage=0.5)
python etl/exposure/sgb/05_sgb_calibrate_e1.py --sgb-ref 0.2 --min-coverage 0.3
```

**O que observar no diagnóstico:**

- Se threshold ótimo >> 0: ajustar em `e1_deslizamentos_lhasa.py`, atualizar ADR-0020
- Se `lhasa_mean` supera `lhasa_high_frac` em F1: re-exportar `lhasa_med_high_frac` do GEE
- Se F1 varia muito por macrorregião: avaliar threshold regional

______________________________________________________________________

## Script 06 — Validação E2

**O que faz:** Varre thresholds de `e2_inu_abs` (flood_score), encontra o ótimo, e
analisa os falsos negativos: onde o SGB aponta alta suscetibilidade a inundação mas
a camada HAND×JRC não detecta. Exclui automaticamente hexágonos com `sgb_override=True`
antes de calcular métricas — nesses hexágonos o SGB já entrou como input do E2, então
validar contra SGB seria circular. A validação mede apenas a camada HAND×JRC base.

**Pré-requisito:** parquet E2 (`br_h3_e2_inundacoes.parquet`) + `br_h3_sgb_inundacoes.parquet`

```bash
python etl/exposure/sgb/06_sgb_validate_e2.py

python etl/exposure/sgb/06_sgb_validate_e2.py --sgb-ref 0.2 --min-coverage 0.3
```

**Outputs em `cfg.DIAGNOSE_DIR`:**

- `diagnostic_e2_validation_<ts>.txt` — relatório principal
- `diagnostic_e2_validation_<ts>.csv` — sweep completo por threshold
- `diagnostic_e2_fn_hexagons_<ts>.csv` — hexágonos falsos negativos da camada HAND×JRC

**O que observar no diagnóstico:**

- Métricas refletem apenas a camada HAND×JRC base (hexágonos override excluídos)
- Se FN com `flood_score=0` persistirem em volume relevante: avaliar cobertura JRC/SGB na região

______________________________________________________________________

## Script 08 — Status do Pipeline por Município

**O que faz:** Reconcilia todos os artefatos do pipeline (manifest, cobertura,
progress.json, failures.csv) e gera uma tabela mostrando em qual etapa cada
município "caiu" e por quê, separado por tipo (massa vs inundação).

**Pré-requisito:** Ter rodado ao menos os scripts 00 e 01. Quanto mais scripts
tiverem sido executados, mais completo o diagnóstico.

```bash
python etl/exposure/sgb/08_sgb_pipeline_status.py

# Só imprime o sumário sem escrever o CSV
python etl/exposure/sgb/08_sgb_pipeline_status.py --summary
```

**Output:** `<data_dir>/inputs/raw/sgb/sgb_pipeline_status.csv`

Uma linha por município com colunas de status por tipo:
`status_download`, `status_explore_{massa|inundacao}`, `status_extract_{massa|inundacao}`,
`status_harmonize_{massa|inundacao}`, `in_pipeline_{massa|inundacao}`,
`last_failure_stage_{massa|inundacao}`, `last_failure_reason_{massa|inundacao}`

**Sumário no terminal:**

```bash
[MASSA]
  Download OK          :   814
  Explore OK (tipo)    :   750   (sem layer: 30)
  Extract OK           :   720
  Harmonize OK         :   710
  → in_pipeline_mass:   710  (87% do total)

  Dropout por etapa:
    explore     :   34 falhas
    extract     :   30 falhas
    harmonize   :   10 falhas
```

> **Nota:** O tracking de município por etapa está disponível até o script 03
> (harmonize). O 04 (H3 intersect) opera por estado e não rastreia por município.
> `in_pipeline = True` indica que o município chegou ao 03 com sucesso.

______________________________________________________________________

## Fluxo resumido

```bash
00 collect + download
  ↓
  raw_zips/ + 00_sgb_manifest.csv

01 explore
  ↓
  01_sgb_inventory.csv
  01_sgb_coverage.csv
  01_sgb_mapping.json   ← EDITAR antes de continuar

  se ZIPs com erro → 00 redownload → 01 (retoma automaticamente)
  se classificação errada → editar 01_sgb_inventory.csv → rodar 01 novamente

02 extract --dry-run → extract
  ↓
  por_municipio/{UF}/{sigla_uf}_{nm_mun}_inundacao.gpkg
  por_municipio/{UF}/{sigla_uf}_{nm_mun}_massa.gpkg

03 harmonize --dry-run → harmonize
  ↓
  harmonized/03_sgb_floods_br.gpkg
  harmonized/03_sgb_mass_br.gpkg

04 h3_intersect
  ↓
  clean/br_h3_sgb_massa.parquet
  clean/br_h3_sgb_inundacoes.parquet

05 calibrate_e1 + 06 validate_e2   (independentes entre si, requerem 04)
  ↓
  diagnósticos TXT/CSV → ajustes em e1/e2 conforme ADRs
  (06 exclui hexágonos sgb_override=True — valida só a camada HAND×JRC)
```

______________________________________________________________________

## Estrutura de arquivos

```bash
data/inputs/raw/sgb/
├── raw_zips/                        # ZIPs baixados (um por município)
├── por_municipio/                   # output do 02
│   └── {UF}/
│       ├── {sigla_uf}_{nm_mun}_inundacao.gpkg
│       └── {sigla_uf}_{nm_mun}_massa.gpkg
├── harmonized/                      # output do 03
│   ├── 03_sgb_floods_br.gpkg
│   └── 03_sgb_mass_br.gpkg
├── 00_sgb_manifest.csv              # criado pelo 00
├── 01_sgb_inventory.csv             # criado pelo 01 — um registro por arquivo; col. revisar
├── 01_sgb_coverage.csv              # criado pelo 01 — status por ZIP (status_zip)
└── 01_sgb_mapping.json              # criado pelo 01 — editar manualmente

data/inputs/clean/
├── br_h3_sgb_massa.parquet          # output do 04
└── br_h3_sgb_inundacoes.parquet     # output do 04
```
