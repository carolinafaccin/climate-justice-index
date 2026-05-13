# Índice de Injustiça Climática

[![Tests](https://github.com/carolinafaccin/climate-injustice-index/actions/workflows/tests.yml/badge.svg)](https://github.com/carolinafaccin/climate-injustice-index/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/badge/docs-online-blue?logo=github)](https://carolinafaccin.github.io/climate-injustice-index/)
[![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)

Projeto de dados para mensurar desigualdades territoriais e injustiças climáticas nos municípios brasileiros, produzido pelo WRI Brasil.

O índice combina exposição a riscos climáticos, vulnerabilidade socioeconômica, grupos populacionais prioritários e capacidade de gestão municipal em uma grade hexagonal H3 (resolução 9, ~0,1 km²).

---

## Metodologia

- **Grade espacial:** H3 resolução 9 (~174 m de aresta em cada hexágono)
- **Normalização:** Min-max de 0 a 1
- **Interpolação censitária:** Ponderação dasimétrica por domicílios para distribuir dados de setor censitário nos hexágonos
- **Índices intermediários:** cada dimensão gera seu próprio índice — IP (Grupos Prioritários), IV (Vulnerabilidade), IE (Exposição), IG (Capacidade de Gestão Municipal)
- **Índice final (IIC):** média simples de IP, IV, IE e IG invertido

---

## Como executar

```bash
# Pipeline completo (todas as etapas)
python pipeline.py

# Só o cálculo do índice
python run_index.py

# A partir de uma etapa específica
python pipeline.py --from cluster

# Apenas uma etapa
python pipeline.py --only scatter
```

Etapas disponíveis: `calc` → `cluster` → `multicol` → `norm` → `export` → `scatter` → `report`

---

## Estrutura de pastas

```
pipeline.py                   # Orquestrador completo (substitui o .bat)
run_index.py                  # Executa só o cálculo do índice

config/
├── indicators.json           # Fonte única de verdade para metadados dos indicadores
├── config.example.json       # Modelo de configuração local
└── config.local.json         # Configuração local (gitignored)

src/
├── config.py                 # Carregamento de caminhos e metadados
├── calculation.py            # Orquestra os passos do cálculo
├── formulas.py               # Fórmulas matemáticas (IP, IV, IE, IG → IIC)
└── utils.py                  # Utilitários: logging, normalização, I/O

etl/
├── census/                   # IP + IV — dados censitários (IBGE 2022)
├── vulnerability/            # IV — fontes adicionais (ex: CNES)
├── exposure/                 # IE — exposição climática (MapBiomas, INPE, Landsat)
├── governance/               # IG — gestão municipal (SICONFI, MUNIC, MIDR)
├── geo/                      # Pré-processamento espacial (interpolação dasimétrica)
└── gee_scripts/              # Scripts JavaScript para Google Earth Engine

explore/
├── analysis/                 # Análises científicas (ex: cluster de municípios)
├── checks/                   # Validação de qualidade (multicolinearidade, normalização)
├── plots/                    # Visualizações (mapas, scatter plots)
├── export/                   # Conversão de formatos (parquet → GeoPackage)
└── utils.py                  # Utilitários compartilhados pelos scripts de explore/

report/                       # Geração do relatório HTML
logs/                         # Logs de execução (gitignored)
```

### Saídas em `data_dir/outputs/results/`

```
results/
├── complete/                 # Parquets completos do cálculo do índice
├── dashboard/                # Parquets slim para o dashboard
└── complete_gpkg/            # Arquivos GeoPackage
```

---

## Configuração de caminhos

Por padrão, os dados são lidos em `data/` na raiz do projeto. Para usar outro diretório (ex: OneDrive, disco externo), crie um arquivo `config/config.local.json`:

```json
{ "data_dir": "/caminho/para/seus/dados" }
```
