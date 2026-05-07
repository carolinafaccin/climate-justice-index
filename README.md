# Índice de Injustiça Climática — v2.0

Versão 2.0 (beta) — Projeto de dados para mensurar desigualdades territoriais e injustiças climáticas nos municípios brasileiros, produzido pelo WRI Brasil.

O índice combina exposição a riscos climáticos, vulnerabilidade socioeconômica, grupos populacionais prioritários e capacidade de gestão municipal em uma grade hexagonal H3 (resolução 9, ~0,1 km²).

---

## Metodologia

- **Grade espacial:** H3 resolução 9 (~105 m de lado por hexágono)
- **Normalização:** Min-max de 0 a 1
- **Interpolação censitária:** Ponderação dasimétrica por domicílios para distribuir dados de setor censitário nos hexágonos
- **Índices intermediários:** cada dimensão gera seu próprio índice — IP (Grupos Prioritários), IV (Vulnerabilidade), IE (Exposição), IG (Gestão Municipal)
- **Índice final (IIC):** média simples de IP, IV, IE e IG invertido

---

## Estrutura de pastas

```
run.py                        # Ponto de entrada: executa o pipeline completo
run_pipeline.bat              # Windows: encadeia pipeline + diagnósticos

config/
├── indicators.json           # Fonte única de verdade para metadados dos indicadores
├── config.example.json       # Modelo de configuração local
└── config.local.json         # Configuração local (gitignored)

src/
├── config.py                 # Carregamento de caminhos e metadados
├── pipeline.py               # Orquestrador principal
├── calculations.py           # Cálculo dos índices (IP, IV, IE, IG → IIC)
└── utils.py                  # Utilitários: logging, normalização, I/O

etl/
├── census/                   # IP + IV — dados censitários (IBGE 2022)
├── vulnerability/            # IV — fontes adicionais (ex: CNES)
├── exposure/                 # IE — exposição climática (MapBiomas, INPE, Landsat)
├── governance/               # IG — gestão municipal (SICONFI, MUNIC, MIDR)
├── geo/                      # Pré-processamento espacial (interpolação dasimétrica)
└── gee_scripts/              # Scripts JavaScript para Google Earth Engine

diagnose/                     # Scripts de análise e validação de qualidade
archive/                      # Código experimental e descontinuado
data/                         # Dados de entrada e saída (gitignored)
logs/                         # Logs de execução (gitignored)
```

---

## Configuração de caminhos

Por padrão, os dados são lidos em `data/` na raiz do projeto. Para usar outro diretório (ex: disco externo), crie um arquivo `config/config.local.json`:

```json
{ "data_dir": "/caminho/para/seus/dados" }
```
