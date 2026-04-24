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
| v4 | saude | Acessibilidade gravitacional a estabelecimentos de saúde | CNES |
| v5 | infraestrutura | Percentual de domicílios sem coleta de esgoto, sem abastecimento de água e/ou sem coleta de lixo | Censo 2022 |

### Exposição a Riscos Climáticos

| Código | Nome | Descrição | Fonte |
|--------|------|-----------|-------|
| e1 | deslizamentos | Percentual de áreas com ao menos 1 domicílio com suscetibilidade a deslizamentos de terra | MapBiomas 2024 |
| e2 | inundacoes | Percentual de domicílios em áreas com suscetibilidade a inundações, alagamentos e enxurradas | MapBiomas 2024 |
| e3 | mar | Quantidade de domicílios em áreas de suscetibilidade a elevação do nível do mar | Landsat |
| e4 | calor | Quantidade de domicílios em áreas com alta temperatura superficial média | Copernicus DEM |
| e5 | queimadas | Percentual de domicílios em áreas com até 1 km de proximidade de focos de queimadas | Inpe 2016-2025 |

### Gestão Municipal

| Código | Nome | Descrição | Fonte |
|--------|------|-----------|-------|
| g1 | mun_investimento_despesas | Investimento municipal em gestão ambiental per capita | Siconfi |
| g2 | mun_planejamento_contingencia | Existência de Planos de Contingência | — |
| g3 | mun_participacao_nupdec | Existência de Núcleos Comunitários de Proteção e Defesa Civil (Nupdec) | MUNIC 2020 |
| g4 | mun_governanca_conselhos | Existência de Conselho Municipal de Meio Ambiente, Política Urbana/Desenvolvimento Urbano e/ou Defesa Civil ativo | — |
| g5 | mun_resposta_alerta | Existência de sistemas de alerta de riscos | MUNIC 2023 |
| g6 | mun_informacao_mapeamento | Existência de mapeamento e zoneamento de áreas de risco | MUNIC 2023 |
| g7 | mun_reconhecimento_cadastro | Existência de cadastro ou identificação de famílias em áreas de risco | ICM/MIDR |
| g8 | mun_reparacao_direitos | Quantidade de políticas ou programas na área dos direitos humanos | MUNIC 2023 |

> Indicadores sem fonte definida estão em desenvolvimento.

---

## Metodologia

- **Grade espacial:** H3 resolução 9 (~105 m de lado por hexágono)
- **Normalização:** Min-max de 0 a 1 com winsorização (percentis 1–99) por indicador
- **Interpolação censitária:** Ponderação dasyimétrica por domicílios (`peso_dom`) para distribuir dados de setor censitário nos hexágonos
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
python etl/v124_p12345_censo2022.py   # Grupos prioritários + vulnerabilidade (Censo 2022)
python etl/v5_cnes.py                  # Saúde — modelo gravitacional CNES
python etl/g1_siconfi.py               # Investimento municipal (Siconfi)
python etl/g2345_munic.py              # Gestão municipal (MUNIC)
python etl/e12_mapbiomas.py            # Exposição climática (MapBiomas)
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
src/                # Pipeline, cálculos, configurações e utilitários
indicators.json     # Fonte única de verdade para metadados dos indicadores
```

---

## Configuração de Caminhos

Por padrão, os dados são lidos em `data/` na raiz do projeto. Para usar outro diretório (ex: disco externo), crie um arquivo `config.local.json` na raiz:

```json
{ "data_dir": "/caminho/para/seus/dados" }
```
