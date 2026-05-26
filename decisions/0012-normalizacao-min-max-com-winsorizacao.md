# ADR-0012: Normalizar indicadores com min-max 0–1 e winsorização p1–p99

## Status

Accepted — 2026-05-19

## Contexto

Cada indicador do IIC tem unidade e escala diferente (percentuais, contagens, índices contínuos, binários). Para agregar em sub-índices e no IIC final é preciso normalizar em uma escala comum. A escolha do método de normalização afeta a sensibilidade do índice a outliers e a interpretabilidade dos sub-índices. A premissa do projeto pede leitura intuitiva por gestores e comparabilidade interna.

## Decisão

Normalizar cada indicador para a escala **0–1 via min-max**. A winsorização nos percentis 1 e 99 (valores abaixo de p1 tratados como p1; valores acima de p99 como p99) é **aplicada por padrão**, exceto para indicadores espacialmente concentrados onde colapsaria a normalização — esses casos estão registrados em [ADR-0016](0016-indicadores-sem-winsorizacao.md). A winsorização é calculada nacionalmente. Indicadores com lógica inversa (ex: renda em IV, IG inteiro) recebem inversão `1 - x` na etapa de normalização.

A função `normalize_minmax` em [src/utils.py](../src/utils.py) recebe o parâmetro `winsorize` (default `False`), e cada ETL invoca explicitamente o modo desejado.

## Alternativas consideradas

- **Z-score (padronização)**: estatisticamente robusto, mas gera valores negativos e fora da escala 0–1, dificultando leitura por gestores.
- **Quebras de Jenks ou quantis**: bom para visualização, mas perde a comparabilidade contínua entre hexágonos e municípios.
- **Min-max sem winsorização**: simples e legível, mas outliers extremos (poucos hexágonos com valores muito altos) achatam toda a distribuição perto de zero.
- **Min-max com winsorização p1–p99 (escolhida)**: combina legibilidade (escala 0–1) com robustez a outliers; classes padronizadas facilitam comunicação; alinhado com a recomendação consolidada da validação de "usar 0 a 1 com classes padronizadas, sem quebras de Jenks".

## Consequências

- Positivas: sub-índices ficam comparáveis entre si; resultado em [0,1] é imediatamente legível ("0 é o mínimo, 1 é o máximo de injustiça observado"); outliers extremos não distorcem a distribuição.
- Negativas / trade-offs: ~1% dos hexágonos nos extremos perde diferenciação por causa da winsorização; o resultado de um hexágono depende dos limites nacionais p1/p99 — se um indicador for revisado, os valores normalizados mudam para todos os hexágonos.
- Confiança: Alta — escolha validada nas duas rodadas.

## Referências

- `src/utils.py` ou `src/formulas.py` — implementação da normalização.
- `report/methodological_notes.md` — registro técnico.
- Feedback consolidado das duas rodadas de validação, seção "Sobre o cálculo final".
