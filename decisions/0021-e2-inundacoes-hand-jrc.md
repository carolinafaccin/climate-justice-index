# ADR-0021: Calcular e2 (Inundações) combinando HAND e JRC Global River Flood Hazard

## Status

Accepted — 2026-05-19

## Contexto

Inundações, alagamentos e enxurradas são a categoria de desastre climático mais frequente no Brasil. Precisamos de um indicador de suscetibilidade que: (i) tenha cobertura nacional uniforme; (ii) combine informação topográfica (proximidade a canais de drenagem) com modelagem de perigo hidrológico; (iii) cubra tanto áreas urbanas quanto rurais — a Base Territorial Estatística de Áreas de Risco (BATER) IBGE/Cemaden não atende esses requisitos por restrição de cobertura.

A Coleção 1 do MapBiomas Risco Climático (2024) propõe uma metodologia baseada em HAND + JRC, mas aplica uma **máscara de áreas urbanas do Open Buildings** que limita o produto a perímetros urbanos consolidados. Para o IIC, essa máscara é inadequada — populações ribeirinhas e periurbanas em áreas de risco ficariam invisíveis.

## Decisão

Calcular o indicador e2 no GEE combinando dois rasters:

1. **HAND (Height Above Nearest Drainage)** — `users/gena/global-hand/hand-100`, derivado do SRTM, resolução ~30 m. Classificação por faixas:

| HAND | Score | Interpretação |
|------|-------|---------------|
| 0–2 m | 1.00 | Muito alta suscetibilidade |
| 2–4 m | 0.66 | Alta |
| 4–6 m | 0.33 | Média |
| > 6 m | 0.00 | Sem suscetibilidade |

2. **JRC Global River Flood Hazard Maps v2.1** — `JRC/CEMS_GLOFAS/FloodHazard/v2_1`, resolução ~1 km, banda `RP100_depth` (perigo de inundação modelado para período de retorno de 100 anos).

O score por pixel é o produto: `score_HAND × (RP100_depth > 0)` (máscara binária). Apenas pixels simultaneamente em planície de inundação **e** em zona de perigo modelado recebem score positivo. Score por hexágono = média dos pixels via `Reducer.mean()` com buffer de 174 m (circunraio do H3 res9), restrita a hexágonos habitados.

A **máscara de áreas urbanas do MapBiomas Risco Climático é removida**, permitindo cobertura nacional independente de uso da terra.

Normalização min-max **sem winsorização** (ADR-0016) — inundações são fenômenos geograficamente concentrados em vales e margens fluviais.

## Alternativas consideradas

- **MapBiomas Risco Climático com máscara urbana original**: produto nacional pronto, mas exclui populações ribeirinhas e periurbanas — viola a premissa de cobertura nacional.
- **Apenas HAND (sem JRC)**: simplifica e tem resolução melhor (~30 m), mas perde a calibração hidrológica do JRC; pixels com HAND baixo em regiões secas (sem rios significativos) gerariam falsos positivos.
- **Apenas JRC RP100_depth**: tem calibração hidrológica direta, mas resolução ~1 km perde detalhe topográfico relevante para diferenciação intramunicipal.
- **BATER (IBGE/Cemaden)**: oficial brasileiro, mas cobertura parcial; preserva-se como referência cruzável, não como fonte primária (tratado no ADR-0028).
- **HAND + JRC sem máscara urbana (escolhido)**: combina detalhe topográfico (HAND ~30 m) com calibração hidrológica (JRC RP100); cobertura nacional total; adaptação documentada da metodologia MapBiomas Risco Climático 2024.

## Consequências

- Positivas: cobertura nacional uniforme incluindo áreas rurais e periurbanas; combinação de fontes mitiga falsos positivos puramente topográficos; classificação por faixas HAND é interpretável; alinhada a metodologia internacional consolidada (JRC GloFAS).
- Negativas / trade-offs: depende de dois assets GEE de terceiros; resolução final é limitada pela escala mais grosseira do JRC (~1 km) para a máscara de perigo; pixels em planícies hidrologicamente ativas mas não cobertas pelo modelo JRC ficam fora — o JRC modela rios significativos mas pode subestimar inundações urbanas tipo alagamento sem componente fluvial direto.
- Confiança: Alta — base metodológica internacional reconhecida; adaptação (remoção da máscara urbana) é clara e justificada.

## Referências

- ADR-0009 (grade H3), ADR-0012 (normalização), ADR-0016 (e2 sem winsorização).
- [etl/exposure/e2_inundacoes_hand.py](../etl/exposure/e2_inundacoes_hand.py) — ETL oficial.
- [report/methodological_notes.md](../report/methodological_notes.md) — seção e2 com tabela de classificação HAND.
- Donchyts et al. (2016) — HAND-100 asset.
- JRC/CEMS GloFAS Global River Flood Hazard Maps v2.1.
- MapBiomas Risco Climático Coleção 1 (2024) — base metodológica adaptada.
