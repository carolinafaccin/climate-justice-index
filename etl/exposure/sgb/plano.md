# Pipeline SGB: Calibração de E1 e Validação de E2

## Objetivo

Usar a Cartografia de Suscetibilidade do SGB/CPRM (~814 municípios) como referência empírica para:

- **E1 (deslizamentos):** calibrar o threshold do NASA LHASA (atualmente `lhasa_high_frac >= 4`)
- **E2 (inundações):** validar o threshold do `flood_score` (combinação HAND+JRC) e recomendar ajuste se necessário

## Scripts (nesta pasta)

| Script | Status | O que faz | Inputs | Outputs |
|---|---|---|---|---|
| `00_sgb_scraper.py` | Pronto | Raspa links e metadados do SGB; baixa ZIPs em paralelo com retomada | URL do SGB | `raw_zips/` + `00_sgb_manifest.csv` |
| `01_sgb_explore.py` | Pronto | Inventaria ZIPs: classifica SHPs, detecta colunas de classe, lê metadados; incremental | `raw_zips/` | `01_sgb_inventory.csv` (col. `revisar`), `01_sgb_coverage.csv` (col. `status_zip`), `01_sgb_mapping.json` |
| `02_sgb_harmonize.py` | Pronto | Extrai SHPs/GPKGs/TIFs, mapeia classe → 0-5, consolida em GeoPackages nacionais | `raw_zips/` + `01_sgb_inventory.csv` + `01_sgb_mapping.json` | `02_sgb_floods_br.gpkg`, `02_sgb_mass_br.gpkg` |
| `03_sgb_h3_intersect.py` | Pronto | Interseção exata SGB × grade H3 res9 por estado; calcula fração de área em classes 4–5 por hexágono | GeoPackages | `br_h3_sgb_massa.parquet`, `br_h3_sgb_inundacoes.parquet` |
| `04_sgb_rasterize.py` | A criar (baixa prioridade) | Rasteriza GeoPackages para GeoTIFF 30m por estado (archival); não bloqueia 05 e 06 | GeoPackages | `sgb_massa_br_30m.tif`, `sgb_inundacoes_br_30m.tif` |
| `05_sgb_calibrate_e1.py` | A criar | Sweep de threshold LHASA vs SGB massa; calcula precision/recall/F1 por threshold e por macrorregião | parquet E1 + `br_h3_sgb_massa.parquet` | diagnóstico TXT/CSV |
| `06_sgb_validate_e2.py` | A criar | flood_score vs SGB inundações; análise de falsos negativos e distribuição de HAND | parquet E2 + `br_h3_sgb_inundacoes.parquet` | diagnóstico TXT/CSV |

Para instruções de execução passo a passo, ver [INSTRUCOES.md](./INSTRUCOES.md).

## Estrutura de dados

```
data/inputs/raw/sgb/
├── raw_zips/                        # ZIPs baixados do SGB, um por município
├── harmonized/
│   ├── 02_sgb_floods_br.gpkg        # output do 02
│   └── 02_sgb_mass_br.gpkg          # output do 02
├── rasters/
│   ├── sgb_massa_br_30m.tif         # output do 04 (opcional)
│   └── sgb_inundacoes_br_30m.tif    # output do 04 (opcional)
├── 00_sgb_manifest.csv              # criado pelo 00 — metadados + links de download
├── 01_sgb_inventory.csv             # criado pelo 01 — um registro por arquivo; col. revisar
├── 01_sgb_coverage.csv              # criado pelo 01 — status por ZIP (status_zip, has_*)
└── 01_sgb_mapping.json              # criado/atualizado pelo 01 — editar antes do 02

data/inputs/clean/
├── br_h3_sgb_massa.parquet          # output do 03 — colunas: h3_id, sgb_alta_mta_frac, sgb_max_class, sgb_coverage_frac
└── br_h3_sgb_inundacoes.parquet     # output do 03 — mesmas colunas
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
| -1 | Não mapeado (excluído no script 03) |

## Script 03 — Interseção H3

Por limitação de memória, o script processa um tipo (massa ou inundação) por vez e, dentro de cada tipo, um estado por vez via filtro SQL no GPKG. Para cada estado:

1. Polyfill H3 res9 nos polígonos SGB para obter células candidatas
2. Interseção exata (`gpd.overlay`) entre polígonos SGB e polígonos H3, em EPSG:5880 (projetado, para área em m²)
3. Agrega por hexágono: `sgb_alta_mta_frac` = área em classes 4–5 / área SGB total na célula

Hexágonos em fronteiras de estado são reconciliados na agregação final (soma de áreas de ambos os estados).

Colunas de saída:
- `h3_id` — índice H3 res9 (string)
- `cd_estado` — UF com maior área mapeada no hexágono
- `sgb_alta_mta_frac` — fração da área SGB mapeada em classes 4–5
- `sgb_max_class` — classe máxima presente no hexágono (0–5)
- `sgb_coverage_frac` — fração da área do hexágono coberta por dados SGB (útil para filtrar células de borda)
- `n_records` — número de feições SGB que intersectam o hexágono

## Calibração E1 (script 05)

O script testa `lhasa_high_frac > t` para t de 0.0 a 1.0 em passos de 0.05 contra a referência
SGB `sgb_alta_mta_frac > 0.3` (≥ 30% da área SGB mapeada do hexágono em classe Alta ou Muito Alta),
restrito a hexágonos com `sgb_coverage_frac > 0.5`.

Calcula precision, recall, F1 para cada t → recomenda o t com F1 máximo.

Diagnóstico secundário: F1 por macrorregião (N, NE, SE, S, CO) — se variar muito, reporta para avaliar threshold regional. Ver ADR-0033.

Se `lhasa_mean >= 3` tiver performance melhor que `lhasa_high_frac >= 4`: indica que a variável `lhasa_med_high_frac` (fração com LHASA ≥ 3) deveria ser exportada do GEE (ver seção pós-calibração).

## Validação E2 (script 06)

Falso negativo = hexágono com `sgb_alta_mta_frac > 0.3` (SGB indica alta suscetibilidade a inundação)
mas `flood_score < 0.1` (E2 não captura). Análise identifica a distribuição de `flood_score` nos
falsos negativos e em que macrorregiões se concentram, para orientar ajuste de threshold.

Diferença de E1: E2 usa `flood_score` (produto HAND × JRC), não HAND bruto. Portanto a análise
verifica diretamente o score final — se o threshold de `flood_score` precisa mudar, o GEE não
precisa ser re-executado; ajusta-se apenas `e2_inundacoes_hand.py`.

## Pós-calibração: o que muda no pipeline

### E1 — Deslizamentos (script 05)

| Resultado do 05 | Ação | Precisa re-exportar do GEE? |
|---|---|---|
| Threshold ótimo ≠ atual (mas variável é `lhasa_high_frac`) | Ajustar em `e1_deslizamentos_lhasa.py`; atualizar ADR-0020 | **Não** — variável já exportada |
| `lhasa_mean >= 3` supera `lhasa_high_frac >= 4` | Editar `h3_e1_deslizamentos_lhasa_gee.js` para exportar `lhasa_med_high_frac`; re-exportar por UF; atualizar ETL e ADR-0020 | **Sim** — nova banda no GEE |
| F1 varia muito por macrorregião | Avaliar threshold regional; documentar em ADR-0020 | **Não** |

Re-executar o GEE é um processo de ~2h (exportação por UF em paralelo no GEE). O ETL `e1_deslizamentos_lhasa.py` lê os CSVs exportados e regera o parquet em minutos.

### E2 — Inundações (script 06)

| Resultado do 06 | Ação | Precisa re-exportar do GEE? |
|---|---|---|
| Threshold ótimo de `flood_score` ≠ 0.1 | Ajustar corte em `e2_inundacoes_hand.py`; atualizar ADR-0021 | **Não** — score já calculado no GEE |
| Falsos negativos concentrados em HAND > 6m (teto atual) | Editar `h3_e2_inundacoes_hand_gee_v1.js` para ampliar faixa HAND; re-exportar por UF; re-rodar ETL; atualizar ADR-0021 | **Sim** |
| Falsos negativos concentrados em regiões sem cobertura JRC | Avaliar fonte alternativa de perigo fluvial para essas regiões; documentar limitação em ADR-0021 | Depende |

## Referências

- ADR-0020: decisão de usar NASA LHASA para E1
- ADR-0021: decisão de usar HAND+JRC para E2
- ADR-0032: metodologia da calibração SGB — `decisions/0032-sgb-como-referencia-calibracao.md`
- ADR-0033: calibração de threshold vs. ML — `decisions/0033-calibracao-threshold-vs-ml-sgb.md`
- `etl/discarded/e1_validacao_slope_lhasa.py`: padrão de script de validação/correlação
- Site SGB: https://www.sgb.gov.br/produtos-por-estado-cartografia-de-suscetibilidade
