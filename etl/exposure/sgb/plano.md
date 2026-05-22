# Pipeline SGB: Calibração de E1 e Validação de E2

## Objetivo

Usar a Cartografia de Suscetibilidade do SGB/CPRM (~800 municípios) como referência empírica para:

- **E1 (deslizamentos):** calibrar o threshold do NASA LHASA (atualmente `lhasa_high_frac >= 4`)
- **E2 (inundações):** validar o teto de 6m do HAND e recomendar novo threshold

## Scripts (nesta pasta)

| Script | Status | O que faz | Inputs | Outputs |
|---|---|---|---|---|
| `00_sgb_scraper.py` | Pronto | Raspa links e metadados do SGB; baixa ZIPs em paralelo com retomada | URL do SGB | `raw_zips/` + `00_sgb_manifest.csv` |
| `01_sgb_explore.py` | Pronto | Inventaria ZIPs: classifica SHPs, detecta colunas de classe, lê metadados; incremental | `raw_zips/` | `01_sgb_inventario.csv`, `01_sgb_revisao.csv`, `01_sgb_excluidos.csv`, `01_sgb_cobertura.csv`, `01_sgb_mapeamento.json` |
| `02_sgb_harmonize.py` | Pronto | Extrai SHPs/GPKGs/TIFs, mapeia classe → 0-5, consolida em GeoPackages nacionais | `raw_zips/` + `01_sgb_inventario.csv` + `01_sgb_mapeamento.json` | `02_sgb_inundacoes_br.gpkg`, `02_sgb_massa_br.gpkg` |
| `03_sgb_h3_intersect.py` | A criar | Overlay com grade H3 res9, calcula fração de área por classe por hexágono | GeoPackages + H3 base | `br_h3_sgb_massa.parquet`, `br_h3_sgb_inundacoes.parquet` |
| `04_sgb_rasterize.py` | A criar | Rasteriza GeoPackages para GeoTIFF 30m (archival + GEE futuro) | GeoPackages | `sgb_massa_br_30m.tif`, `sgb_inundacoes_br_30m.tif` |
| `05_sgb_calibrate_e1.py` | A criar | Sweep de threshold LHASA vs SGB massa; calcula precision/recall/F1 | e1 parquet + sgb_massa parquet | diagnóstico TXT |
| `06_sgb_validate_e2.py` | A criar | HAND+JRC vs SGB inundações, análise de falsos negativos | e2 parquet + sgb_inundacoes parquet | diagnóstico TXT |

Para instruções de execução passo a passo, ver [INSTRUCOES.md](./INSTRUCOES.md).

## Estrutura de dados

```
data/inputs/raw/sgb/
├── raw_zips/                        # ZIPs baixados do SGB, um por município
├── harmonized/
│   ├── 02_sgb_inundacoes_br.gpkg    # output do 02
│   └── 02_sgb_massa_br.gpkg         # output do 02
├── rasters/
│   ├── sgb_massa_br_30m.tif         # output do 04
│   └── sgb_inundacoes_br_30m.tif    # output do 04
├── 00_sgb_manifest.csv              # criado pelo 00 — metadados + links de download
├── 01_sgb_inventario.csv            # criado pelo 01 — um registro por arquivo por ZIP
├── 01_sgb_revisao.csv               # criado pelo 01 — ZIPs problemáticos para revisão manual
├── 01_sgb_excluidos.csv             # criado pelo 01 — exclusões categorizadas automaticamente
├── 01_sgb_cobertura.csv             # criado pelo 01 — cobertura por município
└── 01_sgb_mapeamento.json           # criado/atualizado pelo 01 — editar antes do 02

data/inputs/clean/
├── br_h3_sgb_massa.parquet          # output do 03
└── br_h3_sgb_inundacoes.parquet     # output do 03
```

## Escala de classes harmonizada

| Valor | Classe SGB |
|---|---|
| 5 | Muito Alta |
| 4 | Alta |
| 3 | Média / Moderada |
| 2 | Baixa |
| 1 | Muito Baixa |
| 0 | Sem suscetibilidade / Área urbana |
| -1 | Não mapeado (filtrar downstream) |

## Calibração E1 (script 05)

O script testa `lhasa_high_frac > t` para t de 0.0 a 1.0 em passos de 0.05 contra a referência
SGB `sgb_alta_mta_frac > 0.3` (≥ 30% da área do hexágono em classe Alta ou Muito Alta).

Calcula precision, recall, F1 para cada t → recomenda o t com F1 máximo.

Se `lhasa_mean >= 3` tiver performance melhor que `lhasa_high_frac >= 4`: indica necessidade de
novo GEE export adicionando banda `lhasa_med_high_frac` (fração com LHASA >= 3).

## Validação E2 (script 06)

Falso negativo = hexágono onde SGB indica alta suscetibilidade a inundação mas `flood_score < 0.1`.
Análise identifica onde estão concentrados e qual distribuição de HAND sugerem para novo teto.

## O que fazer com os resultados

| Resultado | Ação |
|---|---|
| 05: threshold ≠ atual → ajustar em `e1_deslizamentos_lhasa.py`, documentar em ADR-0020 |
| 05: lhasa_mean >= 3 melhor → novo GEE script + re-exportar + atualizar ADR-0020 |
| 06: falsos negativos concentrados em HAND 6–Xm → editar `h3_e2_inundacoes_hand_gee_v1.js`, re-exportar, atualizar ADR-0021 |

## Referências

- ADR-0020: decisão de usar NASA LHASA para E1
- ADR-0021: decisão de usar HAND+JRC para E2
- ADR-0032: metodologia da calibração SGB — ver `decisions/0032-sgb-como-referencia-calibracao.md`
- `etl/discarded/e1_validacao_slope_lhasa.py`: padrão de script de validação/correlação a seguir
- Site SGB: https://www.sgb.gov.br/produtos-por-estado-cartografia-de-suscetibilidade
