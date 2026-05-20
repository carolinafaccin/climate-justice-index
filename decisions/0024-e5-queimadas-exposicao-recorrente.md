# ADR-0024: Incluir e5 (Queimadas) com buffer ~600 m e critério de exposição recorrente (≥2 anos)

## Status
Accepted — 2026-05-19

## Contexto
Queimadas não constavam da proposta inicial da dimensão IE. Na validação, **Zuleica (Cidades Sustentáveis)** apontou que a população percebe fumaça e fogo como ameaças tangíveis — mais imediatamente reconhecíveis do que abstrações como "seca" — e que o maior impacto sanitário das queimadas não é o fogo em si, mas a **inalação de material particulado fino e gases tóxicos** (literatura de epidemiologia ambiental indica raio de impacto entre 1 e 5 km do foco).

Isso levanta três decisões metodológicas: (i) qual fonte usar; (ii) qual buffer aplicar ao redor de cada foco; (iii) como tratar exposição pontual (1 ano) vs crônica.

## Decisão
Incluir e5 — **Focos de queimadas** — usando dados do **INPE (2016–2025, 10 anos)**, predominantemente do sensor VIIRS (375 m, satélites NOAA-20, NOAA-21 e Suomi NPP). Implementação no ETL Python:

1. Para cada ano, converter coordenadas dos focos em hexágonos H3 resolução 9.
2. Expandir para **buffer ~600 m** via `grid_disk(k=2)` — vizinhança hexagonal de raio 2.
3. Computar **fração de anos** (2016–2025) em que o hexágono ou sua vizinhança teve ao menos um foco.
4. **Critério de exposição recorrente**: hexágonos com foco em apenas 1 ano dos 10 recebem zero (eventos isolados não caracterizam risco crônico). Apenas fração ≥ 0.2 (≥ 2 anos) entra no cálculo.

Cobertura: hexágonos habitados. Normalização min-max com winsorização p1-p99 (ADR-0012). Sem ponderação pela quantidade de domicílios — escala populacional é capturada em IP e IV.

## Alternativas consideradas
- **Não incluir queimadas**: viola o feedback explícito de validação (Zuleica) e omite uma das ameaças climáticas mais visíveis no Brasil (Amazônia, Cerrado, Pantanal).
- **Buffer de 1 km** (padrão para sensor MODIS): adequado se a fonte fosse MODIS, mas dados INPE recentes vêm predominantemente do VIIRS (375 m de resolução). Buffer de 1 km seria desproporcional ao tamanho do pixel sensor.
- **Buffer maior (1–5 km, faixa de impacto epidemiológico)**: capturaria toda a área de impacto sanitário, mas em cidades pequenas o buffer cobriria o município inteiro, perdendo diferenciação intramunicipal. ~600 m via `grid_disk(k=2)` é compromisso entre captura do impacto local e preservação da diferenciação espacial.
- **Sem critério de recorrência** (qualquer foco em qualquer ano marca o hexágono): inflaria o indicador com eventos pontuais (incêndio acidental, queima controlada isolada) e diluiria o sinal de risco crônico. Eventos pontuais não caracterizam injustiça climática.
- **Critério de recorrência mais rigoroso** (≥3 ou ≥4 anos dos 10): reduziria cobertura demais; ≥2 anos é o mínimo defensável para "recorrente".
- **Buffer ~600 m + recorrência ≥2 anos (escolhido)**: equilibra captura espacial proporcional ao sensor com filtro contra falsos positivos pontuais; alinhada à recomendação consolidada de validação.

## Consequências
- Positivas: indicador captura exposição crônica visível e tangível para a população; calibrado para a resolução real do sensor VIIRS; filtro de recorrência elimina ruído de eventos isolados; cobertura nacional uniforme via INPE; fonte aberta e atualizada anualmente.
- Negativas / trade-offs: 10 anos pode ser período curto para capturar tendências de longo prazo; INPE registra apenas focos detectados por satélite (subestima queimadas pequenas ou sob nuvens persistentes); o critério ≥2 anos é heurístico — uma análise futura pode justificar revisão; buffer de 600 m subestima impacto sanitário real (1–5 km na literatura), mas captura adequadamente a diferenciação intramunicipal.
- Confiança: Média — todos os parâmetros (período, buffer, recorrência) são defensáveis mas heurísticos; reavaliar com mais dados de calibração local se aparecerem.

## Referências
- ADR-0009 (grade H3), ADR-0012 (normalização).
- [etl/exposure/e5_inpe.py](../etl/exposure/e5_inpe.py) — ETL oficial.
- [report/methodological_notes.md](../report/methodological_notes.md) — seção e5.
- Feedback consolidado das duas rodadas de validação, seção sobre exposição (Zuleica/Cidades Sustentáveis).
