# Índice de Injustiça Climática — v2.0

Versão 2.0 (beta) — Projeto de dados para mensurar desigualdades territoriais e injustiças climáticas nos municípios brasileiros, produzido pelo WRI Brasil.

O índice combina exposição a riscos climáticos, vulnerabilidade socioeconômica, grupos populacionais prioritários e capacidade de gestão municipal em uma grade hexagonal H3 (resolução 9, ~0,1 km²).

---

## Dimensões e Indicadores

### Grupos Prioritários

| Código | Nome | Descrição | Fonte |
|--------|------|-----------|-------|
| p1 | mulheres | Percentual de domicílios chefiados por mulheres pretas e pardas | Censo 2022 |
| p2 | populacao_negra | Percentual de pessoas pretas e pardas | Censo 2022 |
| p3 | indigenas_quilombolas | Percentual de pessoas indígenas e quilombolas | Censo 2022 |
| p4 | idosos | Percentual de pessoas acima de 60 anos | Censo 2022 |
| p5 | criancas | Percentual de pessoas abaixo dos 14 anos | Censo 2022 |

### Vulnerabilidade

| Código | Nome | Descrição | Fonte |
|--------|------|-----------|-------|
| v1 | renda | Percentual de domicílios com renda do responsável até meio salário-mínimo | Censo 2022 |
| v2 | moradia | Percentual de domicílios improvisados, sem banheiro, em casa de cômodos/cortiço ou com estrutura degradada | Censo 2022 |
| v3 | educacao | Percentual de pessoas acima de 15 anos que não sabem ler e escrever | Censo 2022 |
| v4 | saude | Inacessibilidade gravitacional a estabelecimentos de saúde (invertido: alto = menos acesso) | CNES |
| v5 | infraestrutura | Percentual de domicílios sem coleta de esgoto, sem abastecimento de água e/ou sem coleta de lixo | Censo 2022 |

### Exposição

| Código | Nome | Descrição | Fonte |
|--------|------|-----------|-------|
| e1 | deslizamentos | Percentual de domicílios em áreas com suscetibilidade a deslizamentos de terra | MapBiomas 2024 + CNEFE 2022 |
| e2 | inundacoes | Percentual de domicílios em áreas com suscetibilidade a inundações, alagamentos e enxurradas | MapBiomas 2024 + CNEFE 2022 |
| e3 | mar | Quantidade de domicílios em áreas de suscetibilidade a elevação do nível do mar | Copernicus DEM / GEE |
| e4 | calor | Anomalia de temperatura superficial média (2015–2024 vs. 1985–2010) | Landsat 5/7/8/9 / GEE |
| e5 | queimadas | Percentual de domicílios em áreas com até 1 km de proximidade de focos de queimadas | INPE 2016–2025 |

### Capacidade de Gestão Municipal

| Código | Nome | Descrição | Fonte |
|--------|------|-----------|-------|
| g1 | mun_investimento_despesas | Despesas municipais médias anuais per capita em gestão ambiental (2015–2024) | Siconfi 2015–2024 |
| g2 | mun_planejamento_contingencia | Existência de Plano de Gerenciamento de Contingência | MUNIC/IBGE 2023 |
| g3 | mun_participacao_nupdec | Existência de Núcleos Comunitários de Proteção e Defesa Civil (Nupdec) | MUNIC/IBGE 2020 |
| g4 | mun_governanca_conselhos | Existência de Conselho Municipal de Meio Ambiente ou de Cidade/Desenvolvimento Urbano | MUNIC/IBGE 2023 |
| g5 | mun_resposta_alerta | Existência de sistemas de alerta de riscos | MUNIC/IBGE 2023 |
| g6 | mun_informacao_mapeamento | Existência de mapeamento e zoneamento de áreas de risco | MUNIC/IBGE 2023 |
| g7 | mun_reconhecimento_cadastro | Existência de cadastro ou identificação de famílias em áreas de risco | — |
| g8 | mun_reparacao_direitos | Quantidade de políticas ou programas na área dos direitos humanos | MUNIC/IBGE 2023 |

> Indicadores sem fonte definida estão em desenvolvimento.

---

## Metodologia

- **Grade espacial:** H3 resolução 9 (~105 m de lado por hexágono)
- **Normalização:** Min-max de 0 a 1 com winsorização (percentis 1–99) por indicador
- **Transformação logarítmica:** aplicada antes da normalização nos indicadores com distribuição muito assimétrica (v4, g1)
- **Interpolação censitária:** Ponderação dasyimétrica por domicílios (`peso_dom`) para distribuir dados de setor censitário nos hexágonos
- **Indicadores e1/e2:** calculados a partir dos domicílios do CNEFE 2022 georreferenciados e cruzados com rasters MapBiomas de suscetibilidade
- **Indicador v4:** mede inacessibilidade (o score gravitacional bruto é invertido — `1 − norm` — para que alto = menos acesso = mais vulnerável, consistente com os demais indicadores de vulnerabilidade)
- **Índices intermediários:** cada dimensão gera seu próprio índice — IP (Grupos Prioritários), IV (Vulnerabilidade), IE (Exposição), IG (Gestão Municipal)
- **Inversão de IG:** como o índice mede *injustiça*, maior gestão = menor injustiça; IG é invertido (`1 − IG`) antes de compor o IIC
- **Índice final (IIC):** média simples de IP, IV, IE e IG invertido

---

## Como Executar

### Pré-requisitos

```bash
pip install -r requirements.txt
```

### ETL (por indicador)

```bash
python etl/v1235_p12345_censo2022.py   # Grupos prioritários + vulnerabilidade (Censo 2022)
python etl/v4_cnes.py                  # Saúde — modelo gravitacional CNES
python etl/g1_siconfi.py               # Investimento municipal (Siconfi)
python etl/g234568_munic.py            # Gestão municipal (MUNIC)
python etl/e12_mapbiomas.py            # Exposição: deslizamentos e inundações (MapBiomas + CNEFE)
python etl/e3_mar.py                   # Exposição: elevação do nível do mar (Copernicus DEM / GEE)
python etl/e4_calor.py                 # Exposição: calor extremo (Landsat / GEE)
python etl/e5_inpe.py                  # Exposição: queimadas (INPE)
```

### Pipeline Final

```bash
python -m src.pipeline
```

### Visualização

```bash
streamlit run streamlit.py
```

---

## Estrutura de Pastas

```
data/
  inputs/
    clean/          # Parquets intermediários por indicador (gerados pelo ETL)
    raw/            # Dados brutos originais (não versionados)
  outputs/
    diagnose/       # Logs diagnósticos de cada ETL
    figures/        # Figuras geradas
    results/        # Arquivo final do índice

etl/                # Scripts de extração e transformação por fonte de dados
etl/gee_scripts/    # Scripts JavaScript para o Google Earth Engine
src/                # Pipeline, cálculos, configurações e utilitários
indicators.json     # Fonte única de verdade para metadados dos indicadores
```

---

## Configuração de Caminhos

Por padrão, os dados são lidos em `data/` na raiz do projeto. Para usar outro diretório (ex: disco externo), crie um arquivo `config.local.json` na raiz:

```json
{ "data_dir": "/caminho/para/seus/dados" }
```
