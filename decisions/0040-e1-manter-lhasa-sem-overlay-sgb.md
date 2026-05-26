# ADR-0040: Manter NASA LHASA como fonte única do E1 sem overlay SGB

## Status

Accepted — 2026-05-26

## Contexto

A calibração do E1 (deslizamentos, via NASA LHASA) contra a Cartografia de
Suscetibilidade a Desastres Geológicos do SGB/CPRM (movimentos de massa) foi
executada pelo [05_sgb_calibrate_e1.py](../etl/exposure/sgb/05_sgb_calibrate_e1.py)
sobre ~173 mil hexágonos com cobertura SGB ≥ 50% nas regiões S e SE.

Resultado do sweep de threshold em `lhasa_high_frac`:

| Threshold | Precision | Recall | F1 |
| --- | --- | --- | --- |
| > 0,00 (atual) | 0,282 | 0,364 | 0,318 |
| > 0,50 | 0,295 | 0,323 | 0,308 |
| > 0,95 | 0,306 | 0,285 | 0,295 |

O F1 não apresenta ponto de inflexão — aumentar o threshold melhora ~2 pp de
precision mas perde ~8 pp de recall, sem ganho líquido. O teste secundário com
`lhasa_mean >= 2.75` produz F1 = 0,326 (+0,008), ganho insuficiente para
justificar mudança de variável.

Para E2 (inundações), o mesmo pipeline identificou falsos negativos
concentrados num gap específico (JRC sub-representando inundações urbano-pluviais)
e resolveu com overlay SGB. Para E1, o gap é estrutural: o LHASA é um modelo
global a ~1 km, calibrado contra inventário mundial que sub-representa encostas
tropicais úmidas brasileiras. O problema não é resolvível por threshold.

## Decisão

Manter o NASA LHASA (`lhasa_high_frac`) como fonte única do E1, **sem** integrar
o SGB como overlay. Documentar F1 = 0,32 e a causa da discrepância nas notas
metodológicas como limitação conhecida.

A lógica central: um overlay SGB para E1 resolveria os falsos negativos nos
~600 municípios com cartografia, mas o gap de F1 é distribuído — não é um
problema pontual como era em E2 (JRC vs. inundações pluviais). Incorporar o
SGB melhoraria a cobertura local sem atacar a fraqueza sistemática do LHASA
no restante do país, criando heterogeneidade metodológica sem o benefício
proporcional que o overlay trouxe para E2.

## Alternativas consideradas

- **Overlay SGB (análogo ao E2)**: \`sgb_alta_mta_frac > 0,3 AND sgb_coverage_frac

  > = 0,5 → e1_abs = 1,00\`. Incorporaria dados de campo 1:25.000 nos ~600
  > municípios cobertos. Rejeitada: o gap de F1 é estrutural e distribuído, não
  > um gap específico resolvível por camada adicional. Criaria heterogeneidade
  > metodológica semelhante à do E2 sem o benefício proporcional, pois o problema
  > do E1 não se concentra nos municípios com cartografia SGB.

- **Ajustar threshold de `lhasa_high_frac`**: o sweep não mostra ponto ótimo;
  qualquer threshold acima de 0 piora F1. Rejeitada.

- **Substituir por `lhasa_mean` com threshold ≥ 2,75\`**: F1 = 0,326 (+0,008
  em relação ao atual). Ganho marginal que não justifica trocar a métrica
  principal e re-rodar o pipeline. Rejeitada.

- **Substituir por modelo MapBiomas Risco Climático**: incorpora LHASA +
  SGB + índices topográficos + filtros geomorfológicos (Camarinha et al. 2020).
  Cobertura restrita a perímetros urbanos do Open Buildings — viola a premissa
  de cobertura nacional do IIC (tratada no ADR-0020). Rejeitada.

- **Manter LHASA com limitações documentadas (escolhida)**: metodologia
  reproduzível, cobertura nacional uniforme, base metodológica internacional
  reconhecida. O F1 = 0,32 é registrado explicitamente nas notas metodológicas
  com causa identificada.

## Consequências

- **Positivas**: sem mudanças no pipeline; cobertura nacional inalterada;
  consistência metodológica com a escolha do ADR-0020; limitação documentada
  com rastreabilidade (arquivo CSV de calibração).

- **Negativas / trade-offs**: F1 = 0,32 é baixo — o indicador captura risco
  relativo em escala nacional mas subestima suscetibilidade em encostas
  tropicais urbanas onde o SGB confirma risco elevado. Em análises
  municipais ou sub-regionais, a precisão do E1 é insuficiente para uso sem
  ressalvas. A integração do SGB como overlay permanece como melhoria futura
  identificada se o argumento de heterogeneidade metodológica for reavaliado.

- **Confiança**: Média — a decisão de não incorporar o SGB é defensável, mas
  o F1 = 0,32 deixa uma fragilidade conhecida no indicador. Revisão
  recomendada se o SGB ampliar cobertura ou se uma nova fonte nacional de
  suscetibilidade a deslizamentos (ex.: produto MapBiomas sem máscara urbana)
  for publicada.

## Referências

- [05_sgb_calibrate_e1.py](../etl/exposure/sgb/05_sgb_calibrate_e1.py) — script de calibração
- [etl/exposure/e1_deslizamentos_lhasa.py](../etl/exposure/e1_deslizamentos_lhasa.py) — ETL oficial
- [ADR-0020](0020-e1-deslizamentos-nasa-lhasa.md) — decisão de usar NASA LHASA para E1
- [ADR-0032](0032-sgb-como-referencia-calibracao.md) — SGB como referência de calibração
- [ADR-0039](0039-e2-mapbiomas-com-overlay-sgb.md) — overlay SGB em E2 (decisão análoga não replicada)
- [report/methodological_notes.md](../report/methodological_notes.md) — seção E1 com limitação documentada
- Stanley, T. A. & Kirschbaum, D. B. (2017). A heuristic approach to global landslide susceptibility mapping. *Natural Hazards*, 87(1), 145–164.
