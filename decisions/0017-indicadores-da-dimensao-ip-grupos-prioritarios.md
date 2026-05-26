# ADR-0017: Definir os cinco indicadores da dimensão IP (Grupos Prioritários)

## Status

Accepted — 2026-05-19

## Contexto

A dimensão IP captura marcadores sociais que denotam desvantagem histórica e justificam atenção prioritária em políticas de adaptação climática (ADR-0004). A escolha dos cinco indicadores precisa equilibrar: (i) cobertura dos grupos consensualmente apontados pela literatura de justiça climática brasileira; (ii) disponibilidade de dado aberto na escala intramunicipal; (iii) replicabilidade nacional para todos os 5.570 municípios. O Censo Demográfico 2022 do IBGE é a única base que atende aos três critérios simultaneamente para os marcadores selecionados.

## Decisão

A dimensão IP é composta por cinco indicadores, todos derivados de agregados por setor censitário do Censo 2022 e interpolados para hexágonos H3 via `peso_dom` (ponderação dasimétrica por domicílios CNEFE — ADR-0011):

- **p1 — Mulheres negras chefes de domicílio**: percentual de responsáveis do domicílio do sexo feminino e cor preta ou parda. `(v01340 + v01344) / v01042`.
- **p2 — População negra**: percentual de pessoas pretas e pardas residentes. `(v01318 + v01320) / v01006`.
- **p3 — Indígenas e quilombolas**: percentual de pessoas indígenas e quilombolas. `(v01690 + v03196) / v01006`.
- **p4 — Idosos de baixa renda**: percentual de pessoas com 60+ anos ponderado pela proxy de baixa renda do setor. `((v01040 + v01041) × peso_renda) / v01006`.
- **p5 — Crianças de baixa renda**: percentual de crianças de 0-9 anos ponderado pela proxy de baixa renda do setor. `((v01031 + v01032) × peso_renda) / v01006`.

O fator **`peso_renda = min(1, 1212 / renda_média_setor)`** aproxima a probabilidade do setor concentrar domicílios com renda do responsável até 1 salário mínimo (R$ 1.212, referência 2022). Não há variável censitária de "idosos pobres" ou "crianças pobres" diretamente disponível por setor — o `peso_renda` é a aproximação estatística mais defensável.

Todos os indicadores são normalizados por min-max com winsorização p1-p99 (ADR-0012); todos têm direção positiva (maior valor = mais injustiça).

## Alternativas consideradas

- **Apenas variáveis demográficas puras (sem `peso_renda` em p4 e p5)**: simples, mas perde a interseccionalidade renda-idade essencial para diferenciar "idosos ricos de idosos pobres" — pergunta levantada explicitamente na validação de Brasília. Diluiria o sinal de desvantagem.
- **Incluir variáveis adicionais** sugeridas em validação (LGBTQIA+, mães solo, refugiados, PCDs, gestantes, mortalidade materna/infantil, idosos com comorbidades, povos e comunidades tradicionais IBGE): inviáveis por ausência de granularidade intramunicipal ou ausência de fonte oficial nacional. Tratados no ADR-0026 (não-inclusão em IP).
- **Restringir os marcadores às áreas de exposição** (cruzar IP × IE no cálculo): rejeitado em Brasília — criaria filtro geográfico que oculta vulnerabilidade sistêmica fora das manchas de risco.
- **Cinco indicadores como definidos (escolhida)**: cobre os marcadores consensuais; usa fonte única e replicável; `peso_renda` introduz interseccionalidade nas faixas etárias sem inflar a dimensão.

## Consequências

- Positivas: cobertura nacional uniforme; replicabilidade garantida; interseccionalidade idade × renda contemplada sem multiplicar indicadores; alinha com a recomendação de "nomear" grupos (Brasília).
- Negativas / trade-offs: marcadores ausentes (LGBTQIA+, mães solo, etc.) ficam de fora por limitação de dado, não por escolha conceitual — exige explicação no artigo; `peso_renda` é proxy, não medição direta.
- Confiança: Alta para p1-p3; Média para p4 e p5 (dependentes da qualidade do `peso_renda` como aproximação).

## Referências

- ADR-0003 (4 dimensões), ADR-0004 (IP separado de IV), ADR-0011 (ponderação dasimétrica), ADR-0012 (normalização), ADR-0026 (variáveis IP não-incluídas).
- [config/indicators.json](../config/indicators.json) — definições operacionais.
- [report/methodological_notes.md](../report/methodological_notes.md) — registro técnico detalhado.
- [etl/census/v1235_p12345_censo2022.py](../etl/census/v1235_p12345_censo2022.py) — implementação.
- Feedback consolidado das duas rodadas de validação, seções sobre IP/grupos prioritários.
