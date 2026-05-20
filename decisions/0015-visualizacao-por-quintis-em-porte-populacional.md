# ADR-0015: Visualizar o IIC por quintis dentro de grupos de porte populacional

## Status
Accepted — 2026-05-19

## Contexto
O cálculo do IIC produz um valor contínuo em [0,1] por hexágono (ADR-0012). Para visualização em mapa e leitura por gestores, valores contínuos exigem classificação. A classificação compartilhada nacional misturaria municípios de portes muito diferentes (uma capital com milhões de habitantes e uma cidade rural com 5 mil habitantes), distorcendo a leitura local. A premissa do projeto é uso intramunicipal (ADR-0007), então a classificação deve preservar diferenciação dentro do município e em municípios similares.

## Decisão
Para visualização final do IIC, calcular **quintis (cinco classes de 20%)** dentro de **grupos de municípios definidos por porte populacional** (classificação CNAS/MDS ou IBGE). Cada grupo de porte tem sua própria escala de quintis. O valor contínuo do IIC continua disponível para análises quantitativas, mas o mapa interativo do dashboard usa essa classificação por quintis.

## Alternativas consideradas
- **Classificação nacional única**: comparabilidade direta entre cidades, mas mistura municípios incomparáveis e suprime a variação intramunicipal nos municípios menores.
- **Quebras de Jenks por município**: maximiza variância entre classes dentro do município, mas as classes variam muito entre cidades, dificultando comunicação consistente.
- **Quintis por município individualmente**: preserva variação local, mas perde qualquer comparabilidade entre cidades; municípios com poucos hexágonos podem ter quintis instáveis.
- **Quintis por grupos de porte populacional (escolhida)**: equilibra leitura local (diferenciação intramunicipal preservada) com comparabilidade limitada entre cidades de porte similar; estabiliza a classificação contra municípios pequenos.

## Consequências
- Positivas: visualização adaptada ao público-alvo principal (gestor municipal vê variação dentro da sua cidade e em cidades similares); estabilidade da classificação em municípios pequenos; alinhado com a recomendação consolidada da validação.
- Negativas / trade-offs: o mesmo valor numérico de IIC pode cair em quintis diferentes em municípios de portes diferentes; exige explicar essa lógica na legenda do mapa para evitar confusão.
- Confiança: Alta — escolha validada nas duas rodadas.

## Referências
- ADR-0007 (foco intramunicipal), ADR-0012 (normalização min-max).
- Tipologias IBGE de porte populacional em `{data_dir}/inputs/raw/ibge/tipologias/tipologias_municipios_brasil.csv`.
- Feedback consolidado das duas rodadas de validação, seção "Sobre o cálculo final".
