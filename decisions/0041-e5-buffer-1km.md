# ADR-0041: Ampliar buffer do E5 (Queimadas) de ~600 m para ~1 km

## Status

Accepted — 2026-05-27

## Contexto

O indicador E5 foi originalmente implementado com `grid_disk(k=2)`, produzindo um buffer de ~600 m ao redor de cada foco de calor. A justificativa inicial era calibrar o buffer à resolução do sensor VIIRS (375 m), de modo que cada foco fosse atribuído ao hexágono correto sem extrapolação excessiva. Na revisão dos slides de apresentação, a equipe questionou se o argumento do sensor é a justificativa mais relevante para o buffer, dado que o objetivo do indicador é capturar exposição humana à fumaça e ao PM2,5 — não precisão de localização do foco.

## Decisão

Ampliar o buffer para `grid_disk(k=3)`, equivalente a ~1 km no H3 resolução 9 (distância máxima ao centroide ≈ 906 m). A justificativa passa a ser o **raio de impacto sanitário documentado na literatura epidemiológica para inalação de material particulado fino (PM2,5)**, que aponta 1–5 km como faixa de efeito significativo. O valor de 1 km representa o limite inferior dessa faixa, preservando diferenciação espacial intramunicipal adequada.

## Alternativas consideradas

- **Manter ~600 m (k=2)**: justificativa baseada na resolução do sensor VIIRS é tecnicamente defensável, mas conceitualmente desalinhada ao objetivo do indicador — que é exposição humana, não precisão de localização do foco.
- **~1 km (k=3, escolhido)**: limite inferior do raio de impacto de PM2,5; preserva diferenciação intramunicipal sem cobrir municípios inteiros.
- **1–5 km (k=5 a k=17)**: faixa completa do impacto epidemiológico, mas elimina diferenciação intramunicipal em cidades de médio e pequeno porte, contradizendo o foco do IIC na análise intramunicipal.

## Consequências

- Positivas: justificativa metodológica mais coerente com o objetivo do indicador; alinhamento com literatura de saúde ambiental; cobertura ligeiramente maior captura hexágonos na borda da área afetada pela fumaça.
- Negativas / trade-offs: resultados mudam marginalmente — hexágonos na faixa 600–1000 m de um foco passam a ser incluídos; requer reprocessamento do parquet E5.
- Confiança: Média — o limiar de 1 km continua sendo heurístico; o raio real de impacto depende de condições atmosféricas, topografia e densidade de focos simultâneos, que um buffer fixo não captura.

## Referências

- ADR-0024 (decisão original do buffer ~600 m e critério de recorrência).
- ADR-0009 (grade H3), ADR-0012 (normalização).
- [etl/exposure/e5_inpe.py](../etl/exposure/e5_inpe.py) — ETL oficial.
- [report/methodological_notes.md](../report/methodological_notes.md) — seção E5.

### Literatura de suporte

Nenhum estudo define formalmente 1 km como limiar único de risco — o que a literatura documenta são buffers operacionais (5–160 km) para separar expostos de não-expostos e gradientes espaciais de PM2,5 que mostram concentrações máximas na pluma próxima à fonte. O argumento mais defensável é: **1 km representa o limite inferior dentro do qual a exposição é máxima**, antes da diluição meteorológica significativa.

- **Aguilera, R.; Corringham, T.; Gershunov, A.; Benmarhnia, T. (2021).** Wildfire smoke impacts respiratory health more than fine particles from other sources: observational evidence from Southern California. *Nature Communications*, 12(1):1493. DOI: 10.1038/s41467-021-21708-0. — Utilizou buffer de 160 km (plumas NOAA HMS) para identificar populações expostas; populações dentro de 1–5 km estão sob as concentrações mais altas do penacho, antes da diluição — sustenta 1 km como limite inferior conservador.

- **Cascio, W.E. (2018).** Wildland fire smoke and human health. *Science of The Total Environment*, 624:586–595. DOI: 10.1016/j.scitotenv.2017.12.086. — Revisão amplamente citada sobre dispersão de PM2,5 de queimadas: as maiores concentrações ocorrem na pluma próxima ao foco, onde a diluição ainda não atenuou significativamente as partículas; populações a menos de poucos quilômetros estão sujeitas às doses mais elevadas.

- **Silva, A.M.C.; Mattos, I.E.; Freitas, S.R.; Longo, K.M.; Hacon, S.S. (2010).** Material particulado (PM2.5) de queima de biomassa e doenças respiratórias no sul da Amazônia brasileira. *Revista Brasileira de Epidemiologia*, 13(2):337–351. DOI: 10.1590/S1415-790X2010000200015. — Estudo ecológico espacial em municípios do Mato Grosso com dados INPE/DATASUS; demonstrou associações significativas entre horas críticas de PM2,5 (>80 µg/m³) e internações respiratórias em crianças de 1–4 anos e idosos ≥65 anos; padrão espacial diretamente relacionado à intensidade dos focos de queimada.
