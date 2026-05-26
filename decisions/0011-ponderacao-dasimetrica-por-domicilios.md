# ADR-0011: Interpolar setor censitário → hexágono H3 por ponderação dasimétrica de domicílios

## Status

Accepted — 2026-05-19

## Contexto

Os dados do Censo IBGE 2022 são publicados em setor censitário, com formato e tamanho variáveis. Os hexágonos H3 resolução 9 cruzam fronteiras de setores. Para usar variáveis do Censo nos hexágonos é preciso interpolar, e o método de interpolação afeta materialmente o resultado: distribuir uma variável socioeconômica supondo densidade populacional uniforme dentro do setor introduz viés (setores com áreas vazias ou industriais distorcem a leitura).

## Decisão

Usar **ponderação dasimétrica por contagem de domicílios** para interpolar variáveis de setor censitário para hexágonos H3. A contagem de domicílios por hexágono é obtida a partir do CNEFE (Cadastro Nacional de Endereços para Fins Estatísticos do IBGE). Cada hexágono recebe o valor do setor proporcionalmente ao número de domicílios efetivamente localizados dentro dele.

## Alternativas consideradas

- **Interpolação areal simples** (densidade uniforme): atribui ao hexágono o valor do setor proporcionalmente à área de sobreposição. Simples, mas distorce em setores com áreas vazias, parques, indústrias ou hidrografia.
- **Interpolação centroidal** (valor do setor cujo centroide cai dentro do hexágono): rápida, mas perde precisão em setores grandes ou de forma irregular.
- **Ponderação por população do CNEFE** (variante da dasimétrica): equivalente em precisão, mas o CNEFE entrega contagem de domicílios diretamente — operacionalmente é o que está disponível.
- **Ponderação dasimétrica por domicílios (escolhida)**: precisão melhor que areal simples; alinhada com a premissa de que o IIC é sobre pessoas/domicílios, não sobre superfície; usa o CNEFE como base auxiliar disponível e oficial.

## Consequências

- Positivas: indicadores socioeconômicos refletem a distribuição real da população dentro do setor; evita viés em setores com áreas vazias; coerente com a premissa de "trazer o dado físico para a geometria social" (ADR-0010).
- Negativas / trade-offs: depende da qualidade e atualização do CNEFE; hexágonos sem domicílios CNEFE recebem zero peso (correto, mas exige tratamento explícito); aumenta complexidade do pipeline de ETL.
- Confiança: Alta — método consolidado na literatura de geografia quantitativa; alinhamento natural com a estrutura do IIC.

## Referências

- ADR-0009 (grade H3), ADR-0010 (não raster 30m).
- `etl/geo/` — implementação da interpolação dasimétrica.
- `report/methodological_notes.md` — registro técnico.
