# ADR-0009: Adotar grade H3 resolução 9 como unidade espacial do IIC

## Status
Accepted — 2026-05-19

## Contexto
O IIC precisa de uma unidade espacial única para agregar dados de fontes heterogêneas (Censo IBGE em setor censitário, rasters de satélite em pixels de 30m, dados municipais). A premissa conceitual original apontava o setor censitário como unidade ideal, por ser a base territorial primária do IBGE. Porém setores variam muito em tamanho e formato, dificultam visualização cartográfica, não suportam agregação hierárquica e tornam comparação intramunicipal entre cidades inconsistente.

## Decisão
Adotar a grade hexagonal **H3 do Uber, resolução 9** (~0,1 km², arestas de ~174 m) como unidade espacial única do IIC. Todos os indicadores são interpolados ou agregados nessa grade antes do cálculo final.

## Alternativas consideradas
- **Setor censitário**: alinhado com a base do IBGE, mas tamanho e forma irregulares, dificulta visualização e agregação hierárquica.
- **Grade regular quadrada (raster 30m, 100m, etc.)**: simples, mas pixels quadrados têm vizinhança ambígua (4 ou 8 vizinhos com distâncias diferentes) e não agregam hierarquicamente em escalas maiores sem reamostragem. Caso específico do raster 30m tratado no ADR-0010.
- **Grade Voronoi ou hexágonos irregulares**: melhor para densidade variável, mas perde a vantagem da indexação global e da reprodução por terceiros.
- **H3 resolução 9 (escolhida)**: hexágonos têm vizinhança uniforme (6 vizinhos equidistantes), indexação global hierárquica permite agregação fácil, tamanho na resolução 9 (~0,1 km²) captura diferenciação intramunicipal sem inflar volume de dados, biblioteca aberta e amplamente adotada.

## Consequências
- Positivas: vizinhança uniforme melhora análise espacial; agregação hierárquica permite zoom-out para escalas maiores; indexação global facilita compartilhamento; tamanho fixo torna comparação intramunicipal consistente entre municípios.
- Negativas / trade-offs: hexágonos cruzam fronteiras de setores censitários, exigindo interpolação dasimétrica (ver ADR-0011); resolução 9 pode ser fina demais em zonas rurais de baixa densidade e grossa demais em centros urbanos densos.
- Confiança: Alta — escolha validada nas duas rodadas; documentada em `report/methodological_notes.md`.

## Referências
- ADR-0010 (não adotar raster 30m), ADR-0011 (ponderação dasimétrica setor → hexágono).
- `report/methodological_notes.md` — registro técnico da grade.
- Feedback consolidado das duas rodadas de validação, seção "Sobre o cálculo final" e "É viável calcular o índice em formato raster (30x30m)?".
