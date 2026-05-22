# SGB — Instruções de Execução

Pipeline para baixar, inventariar e harmonizar a Cartografia de Suscetibilidade do SGB/CPRM.
Execute os scripts **na ordem numérica**. Cada script depende dos outputs do anterior.

## Pré-requisitos

```bash
# Ativa o virtualenv do projeto (a partir da raiz do repositório)
source venv/bin/activate

# Garante que config/config.local.json existe com o caminho dos dados
# Exemplo: {"data_dir": "/Users/lina/data/climate-injustice-index/"}
```

Dependências: `geopandas`, `fiona`, `requests`, `beautifulsoup4`, `rich`.
Espaço em disco estimado: ~50–100 GB para os ~814 ZIPs do SGB.

---

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
--workers 10        # downloads paralelos (padrão: 6; SGB suporta até ~10)
--state SP,RJ       # filtra por estado (para testes)
--no-resume         # ignora manifest existente e recoleta tudo do zero
```

**Checklist antes de avançar:**
- `report` mostra status `ok` para a maioria dos municípios
- Pasta `raw_zips/` tem os ZIPs (pode verificar com `ls raw_zips/ | wc -l`)

---

## Script 01 — Exploração e Inventário

**O que faz:** Abre cada ZIP, lista os shapefiles/GPKGs/rasters internos, detecta qual é
inundação vs massa vs outros, lê metadados (colunas, valores de classe, nº de feições) e
gera o inventário. É incremental: Ctrl+C salva o progresso.

**Outputs:**
- `01_sgb_inventario.csv` — um registro por arquivo por ZIP (escrito incrementalmente)
- `01_sgb_revisao.csv` — ZIPs que precisam atenção manual
- `01_sgb_excluidos.csv` — arquivos excluídos automaticamente com categoria de exclusão
- `01_sgb_cobertura.csv` — uma linha por município com has_inundacao / has_massa
- `01_sgb_mapeamento.json` — rascunho do mapeamento textual → 0-5 **(editar antes do script 02)**

```bash
# Processa ZIPs ainda não inventariados (retoma de onde parou)
python etl/exposure/sgb/01_sgb_explore.py

# Testa com os próximos 5 ZIPs não processados
python etl/exposure/sgb/01_sgb_explore.py --limit 5

# Filtra por estado
python etl/exposure/sgb/01_sgb_explore.py --state SE,BA

# Reprocessa tudo do zero (descarta inventário existente)
python etl/exposure/sgb/01_sgb_explore.py --redo

# Verificação CRC completa (lenta, detecta corrupção de dados internos)
python etl/exposure/sgb/01_sgb_explore.py --verify-zips
```

### O que verificar nos outputs

**`01_sgb_revisao.csv`** — filtre por coluna para entender os problemas:
- `tipo=outros` com nome que parece inundação/massa → mude `tipo` no inventário
- `classe_col` vazio → preencha com o nome da coluna de classe do shapefile
- `notes` com "erro" ou "CRC" → ZIP provavelmente corrompido, rodar `redownload`

**`01_sgb_mapeamento.json`** — **obrigatório revisar antes do script 02**:
- Abra o arquivo e verifique os valores `-1` na seção `"mapping"`
- Cada `-1` significa "ainda não mapeado" — preencha com o inteiro correto (0–5)
- Escala: `5=Muito Alta`, `4=Alta`, `3=Média/Moderada`, `2=Baixa`, `1=Muito Baixa`, `0=Sem`
- Salve como JSON válido (sem vírgula no último item)

### Como editar `01_sgb_inventario.csv` manualmente

Abra no Excel, Numbers ou Google Sheets (UTF-8). Colunas que você pode editar:

| Coluna | Quando editar |
|---|---|
| `tipo` | Mude `outros` para `inundacao` ou `massa` se o arquivo foi mal classificado |
| `classe_col` | Preencha com o nome exato da coluna de classe (ex: `CLASSE`, `CLASSE_SU`) |
| `notes` | Adicione observações; não apague notas de erro existentes |

> **Não altere** `zip_filename` nem `shp_path_in_zip` — esses caminhos são usados para
> localizar os arquivos no script 02.

Após editar, rode o script 01 novamente (sem `--redo`) para regenerar os arquivos derivados
com as correções aplicadas.

---

## Script 02 — Harmonização

**O que faz:** Lê o inventário, extrai cada shapefile/GPKG/TIF do ZIP, aplica o mapeamento
de classe, e consolida tudo em dois GeoPackages nacionais.

**Pré-requisito:** `01_sgb_mapeamento.json` revisado (sem valores `-1` não intencionais).

**Outputs em `harmonized/`:**
- `02_sgb_inundacoes_br.gpkg`
- `02_sgb_massa_br.gpkg`

```bash
# Teste sem escrever nada (recomendado na primeira vez)
python etl/exposure/sgb/02_sgb_harmonize.py --dry-run

# Processa tudo
python etl/exposure/sgb/02_sgb_harmonize.py

# Filtra por estado (para testes)
python etl/exposure/sgb/02_sgb_harmonize.py --state SE,BA

# Limita a N ZIPs (para teste rápido)
python etl/exposure/sgb/02_sgb_harmonize.py --limit 10
```

**Se aparecerem avisos de classe não mapeada:**
1. Anote os valores listados no aviso
2. Adicione-os em `01_sgb_mapeamento.json` com o inteiro correto
3. Re-execute o script 02

---

## Scripts 03–06 (a criar)

| Script | Objetivo |
|---|---|
| `03_sgb_h3_intersect.py` | Overlay com grade H3 res9 → parquet por hexágono |
| `04_sgb_rasterize.py` | Rasteriza para GeoTIFF 30m (archival) |
| `05_sgb_calibrate_e1.py` | Calibra threshold LHASA vs SGB massa |
| `06_sgb_validate_e2.py` | Valida threshold HAND vs SGB inundação |

---

## Fluxo resumido

```
00 collect + download
  ↓
  raw_zips/ + 00_sgb_manifest.csv

01 explore
  ↓
  01_sgb_inventario.csv
  01_sgb_revisao.csv
  01_sgb_excluidos.csv
  01_sgb_cobertura.csv
  01_sgb_mapeamento.json   ← EDITAR antes de continuar

  se ZIPs com erro → 00 redownload → 01 (retoma automaticamente)
  se classificação errada → editar 01_sgb_inventario.csv → rodar 01 novamente

02 harmonize --dry-run → harmonize
  ↓
  harmonized/02_sgb_inundacoes_br.gpkg
  harmonized/02_sgb_massa_br.gpkg

03 h3_intersect → 04 rasterize → 05 calibrate_e1 → 06 validate_e2
```

---

## Estrutura de arquivos

```
data/inputs/raw/sgb/
├── raw_zips/                        # ZIPs baixados (um por município)
├── harmonized/
│   ├── 02_sgb_inundacoes_br.gpkg    # output do 02
│   └── 02_sgb_massa_br.gpkg         # output do 02
├── 00_sgb_manifest.csv              # criado pelo 00
├── 01_sgb_inventario.csv            # criado pelo 01 (principal, incremental)
├── 01_sgb_revisao.csv               # criado pelo 01 (ZIPs problemáticos)
├── 01_sgb_excluidos.csv             # criado pelo 01 (exclusões categorizadas)
├── 01_sgb_cobertura.csv             # criado pelo 01 (cobertura por município)
└── 01_sgb_mapeamento.json           # criado pelo 01 — editar manualmente

data/inputs/clean/
├── br_h3_sgb_massa.parquet          # output do 03
└── br_h3_sgb_inundacoes.parquet     # output do 03
```
