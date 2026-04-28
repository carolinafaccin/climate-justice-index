# Índice de Injustiça Climática — v2.0

Versão 2.0 (beta) — Projeto de dados para mensurar desigualdades territoriais e injustiças climáticas nos municípios brasileiros, produzido pelo WRI Brasil.

O índice combina exposição a riscos climáticos, vulnerabilidade socioeconômica, grupos populacionais prioritários e capacidade de gestão municipal em uma grade hexagonal H3 (resolução 9, ~0,1 km²).

---

## Metodologia

- **Grade espacial:** H3 resolução 9 (~105 m de lado por hexágono)
- **Normalização:** Min-max de 0 a 1
- **Interpolação censitária:** Ponderação dasyimétrica por domicílios para distribuir dados de setor censitário nos hexágonos
- **Índices intermediários:** cada dimensão gera seu próprio índice — IP (Grupos Prioritários), IV (Vulnerabilidade), IE (Exposição), IG (Gestão Municipal)
- **Inversão de IG:** como o índice mede *injustiça*, maior gestão = menor injustiça; IG é invertido antes de compor o IIC
- **Índice final (IIC):** média simples de IP, IV, IE e IG invertido

---

## Estrutura de pastas

```
etl/                 # Scripts de extração e transformação por fonte de dados
src/                 # Pipeline, cálculos, configurações e utilitários
indicators.json      # Fonte única de verdade para metadados dos indicadores
config.local.json
```

---

## Configuração de caminhos

Por padrão, os dados são lidos em `data/` na raiz do projeto. Para usar outro diretório (ex: disco externo), crie um arquivo `config.local.json` na raiz:

```json
{ "data_dir": "/caminho/para/seus/dados" }
```
