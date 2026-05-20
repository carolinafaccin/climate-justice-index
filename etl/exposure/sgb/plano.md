# Pipeline SGB: Calibração de E1 e Validação de E2

## Objetivo

Usar a Cartografia de Suscetibilidade do SGB/CPRM (~800 municípios) como ground truth para:

- **E1 (deslizamentos):** calibrar o threshold do NASA LHASA (atualmente `lhasa_high_frac >= 4`)
- **E2 (inundações):** validar o teto de 6m do HAND e recomendar novo threshold

## Scripts a criar (nesta pasta)

| Script | O que faz | Inputs | Outputs |
|---|---|---|---|
| `00_sgb_scraper.py` | Download automatizado do site SGB | URL do SGB | `data/inputs/raw/sgb/raw_zips/` + manifest CSV |
| `01_sgb_harmonize.py` | Lê ZIPs, detecta campo de classe, mapeia → escala 1-5 | ZIPs + `class_mapping.json` | `sgb_massa_br.gpkg`, `sgb_inundacoes_br.gpkg` |
| `02_sgb_h3_intersect.py` | Overlay com grade H3, calcula fração de área por classe | GeoPackages + H3 base | `br_h3_sgb_massa.parquet`, `br_h3_sgb_inundacoes.parquet` |
| `03_sgb_rasterize.py` | Rasteriza para GeoTIFF 30m (archival + GEE futuro) | GeoPackages | `sgb_massa_br_30m.tif`, `sgb_inundacoes_br_30m.tif` |
| `04_sgb_calibrate_e1.py` | Sweep de threshold LHASA vs SGB massa | e1 parquet + sgb_massa parquet | diagnóstico TXT |
| `05_sgb_validate_e2.py` | HAND+JRC vs SGB inundações, análise de falsos negativos | e2 parquet + sgb_inundacoes parquet | diagnóstico TXT |

## Estrutura de dados esperada

```
data/inputs/raw/sgb/
├── raw_zips/                   # ZIPs baixados do SGB, um por município
├── harmonized/
│   ├── sgb_massa_br.gpkg       # output do 01
│   └── sgb_inundacoes_br.gpkg  # output do 01
├── rasters/
│   ├── sgb_massa_br_30m.tif    # output do 03
│   └── sgb_inundacoes_br_30m.tif
├── sgb_download_manifest.csv   # criado pelo 00 (ou manualmente após Claude Cowork)
└── class_mapping.json          # mapeamento string → int 1-5 (iterativo/manual)

data/inputs/clean/
├── br_h3_sgb_massa.parquet       # output do 02
└── br_h3_sgb_inundacoes.parquet  # output do 02
```

## Fluxo de execução

```
Download (Claude Cowork / 00_scraper)
  → raw_zips/ + sgb_download_manifest.csv
  → python etl/exposure/sgb/01_sgb_harmonize.py   (iterar até class_mapping.json completo)
  → python etl/exposure/sgb/02_sgb_h3_intersect.py (~5–30 min)
  → python etl/exposure/sgb/03_sgb_rasterize.py   (opcional)
  → python etl/exposure/sgb/04_sgb_calibrate_e1.py
  → python etl/exposure/sgb/05_sgb_validate_e2.py
```

## Formato do sgb_download_manifest.csv

Colunas esperadas (preencher após download, se feito manualmente):

```
cd_estado, nm_estado, nm_municipio, cd_mun_ibge, url_download, filename, downloaded_at, status
```

- `status`: `ok` / `error` / `sem_dado`
- `cd_mun_ibge`: código de 7 dígitos do IBGE (preencher quando disponível — necessário para join em 02)

## Detalhes metodológicos

### Escala de classes SGB (harmonizada)

| Valor | Classe |
|---|---|
| 5 | Muito Alta |
| 4 | Alta |
| 3 | Média / Moderada |
| 2 | Baixa |
| 1 | Muito Baixa |
| 0 | Sem suscetibilidade / Área urbana não mapeada |

### Calibração E1 (script 04)

O script testa `lhasa_high_frac > t` para t de 0.0 a 1.0 em passos de 0.05 contra a referência
SGB `sgb_alta_mta_frac > 0.3` (≥ 30% da área do hexágono em classe Alta ou Muito Alta).

Calcula precision, recall, F1 para cada t → recomenda o t com F1 máximo.

Se `lhasa_mean >= 3` tiver performance melhor que `lhasa_high_frac >= 4`: indica necessidade de
novo GEE export adicionando banda `lhasa_med_high_frac` (fração com LHASA >= 3).

### Validação E2 (script 05)

Falso negativo = hexágono onde SGB diz alta suscetibilidade a inundação mas `flood_score < 0.1`.
Análise identifica onde estão concentrados e qual distribuição de HAND sugerem para novo teto.

## O que fazer com os resultados

| Resultado | Ação |
|---|---|
| 04: threshold ≠ atual → ajustar em `e1_deslizamentos_lhasa.py`, documentar em ADR-0020 |
| 04: lhasa_mean >= 3 melhor → novo GEE script + re-exportar + atualizar ETL |
| 05: falsos negativos concentrados em HAND 6–Xm → editar `h3_e2_inundacoes_hand_gee_v1.js`, re-exportar, atualizar ADR-0021 |

## Referências

- ADR-0020: decisão de usar NASA LHASA para E1
- ADR-0021: decisão de usar HAND+JRC para E2
- ADR-0032: (a criar) documenta metodologia de calibração SGB
- `archive/e1_validacao_slope_lhasa.py`: padrão de script de validação/correlação a seguir
- Site SGB: https://www.sgb.gov.br/produtos-por-estado-cartografia-de-suscetibilidade
