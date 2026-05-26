# ADR-0010: Não calcular o IIC em formato raster 30x30m

## Status

Accepted — 2026-05-19

## Contexto

Na primeira rodada de validação (SP), especialistas levantaram se seria viável calcular o IIC em formato raster com pixels de 30x30 m, alinhado à resolução nativa dos dados Landsat e MapBiomas. A pergunta exige resposta explícita porque o público técnico do índice tende a esperar essa resolução para dados climáticos.

## Decisão

**Não usar raster 30x30 m como unidade espacial do IIC.** A grade adotada continua sendo H3 resolução 9 (ver ADR-0009). Dados raster (Landsat LST, MapBiomas, DEM, etc.) são reamostrados/agregados para os hexágonos H3 durante o processamento.

## Alternativas consideradas

- **Raster 30x30 m**: alinhamento natural com dados Landsat/MapBiomas, mas inviável por dois motivos: (i) **imprecisão social** — dados sociais do Censo são coletados por setor; quebrá-los em pixels de 30 m exige atribuir o mesmo valor a todos os pixels do setor ou desagregação estatística arriscada, sem garantia metodológica; (ii) **custo computacional** — Brasil tem ~8,5 milhões de km², equivalentes a bilhões de pixels de 30 m; cruzar 23 indicadores em quatro dimensões em volume desse porte exige supercomputador ou GEE pesado, comprometendo a leveza da plataforma web.
- **Raster com resolução mais grosseira (100 m, 250 m)**: reduz custo, mas mantém o problema da imprecisão social e perde o alinhamento natural com Landsat.
- **H3 resolução 9 (escolhida, ver ADR-0009)**: hexágono de ~0,1 km² captura diferenciação intramunicipal, traz o dado físico (raster) para a geometria social sem forçar o dado social a virar pixel.

## Consequências

- Positivas: o índice permanece sobre pessoas (geometria social), não sobre superfície contínua; viável de calcular em hardware comum e servir em plataforma web; consistente com a premissa de que justiça climática é sobre pessoas, e pessoas são contadas pelo Censo em polígonos.
- Negativas / trade-offs: revisores familiarizados com climatologia podem esperar resolução de 30 m; exige justificar a escolha no artigo.
- Confiança: Alta — decisão validada e fundamentada em dois argumentos técnicos sólidos.

## Referências

- ADR-0009 (grade H3 resolução 9).
- Feedback da primeira rodada de validação (SP), seção "É viável calcular o índice em formato raster (30x30m)?".
