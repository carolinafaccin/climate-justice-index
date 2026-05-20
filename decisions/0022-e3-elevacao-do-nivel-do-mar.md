# ADR-0022: Calcular e3 (Elevação do nível do mar) com DEM Copernicus binarizado em ≤1 m

## Status
Accepted — 2026-05-19

## Contexto
Elevação do nível do mar é uma ameaça climática de longo prazo, mas com impactos já observáveis em zonas costeiras brasileiras (erosão, ressacas, intrusão salina). Na primeira rodada de validação (SP), especialistas indicaram que o indicador deve ser mantido apesar do horizonte temporal longo, e sugeriram considerar também ressacas e erosão via Modelo InVEST (vulnerabilidade costeira). Victor (Cemaden) sugeriu padronizar a linguagem de "vulnerabilidade costeira" para "suscetibilidade costeira", coerente com a dimensão IE.

Calcular o Modelo InVEST para todos os 5.570 municípios é inviável — exige dados de habitat, geomorfologia costeira, exposição a ondas e maré que não estão disponíveis com cobertura nacional uniforme.

## Decisão
Calcular e3 como **suscetibilidade à elevação do nível do mar** usando o **Copernicus GLO-30 DEM** com critério binário no GEE:

1. Preencher pixels oceânicos (NoData no DEM original) com valor −10.
2. Calcular distância euclidiana de cada pixel terrestre ao oceano via `fastDistanceTransform`.
3. Marcar pixel **em risco** se elevação ≤ 1 m **AND** distância ao oceano ≤ 10 km — limiar que inclui estuários e planícies costeiras legítimas, excluindo baixadas fluviais interiores.
4. Fração de área em risco por hexágono via `Reducer.mean()` com buffer de 174 m (circunraio H3 res9).
5. Indicador final: produto `qtd_dom × risco_slr` (quantidade de domicílios multiplicada pela fração de risco).

Hexágonos do interior do país (sem componente costeiro) recebem valor **ausente** (não aplicável) e **são excluídos do cálculo do IE** — e3 só entra na média do IE para municípios costeiros.

Normalização min-max **sem winsorização** (ADR-0016) — indicador costeiro com <8% de hexágonos não-zero.

## Alternativas consideradas
- **Modelo InVEST de vulnerabilidade costeira**: tecnicamente mais completo (combina exposição, geomorfologia, habitat, maré, ondas), mas inviável nacionalmente por ausência de dados uniformes; demanda calibração regional.
- **Apenas elevação ≤1 m (sem critério de distância ao oceano)**: inclui baixadas fluviais interiores sem conexão hidrológica com o mar — falsos positivos.
- **Apenas distância ao oceano ≤10 km (sem critério de elevação)**: inclui escarpas costeiras altas, sem risco real de submersão; falsos positivos.
- **Limiar mais conservador (≤2 m de elevação)**: aumenta cobertura, mas atual literatura de SLR-2050/2100 sugere ≤1 m como limiar plausível para impactos significativos em zonas habitadas no horizonte do índice.
- **Critério binário ≤1 m AND ≤10 km com Copernicus GLO-30 (escolhido)**: simples, replicável, transparente; consistente globalmente; melhor resolução (~30 m) e qualidade que SRTM em zonas costeiras.

## Consequências
- Positivas: cálculo viável nacionalmente; metodologia transparente e replicável; multiplicar pela quantidade de domicílios traz a dimensão social diretamente para o indicador, alinhado à premissa de que justiça climática é sobre pessoas; municípios costeiros são identificados corretamente.
- Negativas / trade-offs: tratamento binário perde gradação (um hexágono com elevação 0.5 m e um com 1.0 m valem igual no critério); ignora dinâmica de ressacas e ondas (questão levantada na validação SP); hexágonos interiores recebem valor ausente, mas o cálculo do IE precisa lidar com isso explicitamente (não confundir ausente com zero); o limiar de 1 m é uma escolha normativa defensável mas debatível.
- Confiança: Média — simplifica a complexidade real do fenômeno; revisitar se uma versão futura conseguir dados de geomorfologia costeira nacional, ou se a metodologia InVEST tornar-se aplicável nacionalmente.

## Referências
- ADR-0009 (grade H3), ADR-0012 (normalização), ADR-0016 (e3 sem winsorização).
- [etl/exposure/e3_mar.py](../etl/exposure/e3_mar.py) — ETL oficial.
- [etl/discarded/h3_e3_elevacao_mar_copernicus-dem_gee.js](../etl/discarded/h3_e3_elevacao_mar_copernicus-dem_gee.js) — versão anterior do script GEE (asset atualizado).
- [report/methodological_notes.md](../report/methodological_notes.md) — seção e3.
- Feedback da primeira rodada de validação (SP), seção sobre elevação do nível do mar.
