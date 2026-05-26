# ADR-0039: E2 — retorno à metodologia MapBiomas com overlay SGB

## Status

Accepted — 2026-05-26
Supersedes: [ADR-0038](0038-score-e2-hibrido-hand-jrc-tiers.md)

## Contexto

Esta ADR documenta a jornada completa de investigação e as decisões que levaram
à metodologia final do E2 (inundações). Três rodadas de análise foram realizadas.

### Rodada 1 — Diagnóstico dos falsos negativos no E2 v1

O [06_sgb_validate_e2.py](../etl/exposure/sgb/06_sgb_validate_e2.py) validou o
E2 v1 ([h3_e2_inundacoes_hand_gee_v1.js](../etl/gee_scripts/h3_e2_inundacoes_hand_gee_v1.js))
contra a cartografia SGB. Resultado: boa F1 (≈0.85), mas com taxa relevante de
falsos negativos — hexágonos onde o SGB indica alta suscetibilidade mas o E2
retorna `flood_score = 0`.

O [ADR-0036](0036-diagnostico-hand-falsos-negativos-e2.md) documentou a decisão
de investigar esses FN com um script GEE adicional
(`h3_e2_fn_hand_diagnostic_gee.js` → `07_sgb_analyse_fn_hand.py`).

O diagnóstico revelou que **99% dos FN tinham `jrc_rp100 = 0`**. O teto HAND
não era o gargalo — a máscara JRC era o problema. O JRC global (~1 km) não
representa inundações urbano-pluviais, que o SGB mapeia em detalhe.

### Rodada 2 — Score híbrido em dois tiers (v2 — descartado)

Seguindo o [ADR-0038](0038-score-e2-hibrido-hand-jrc-tiers.md), implementou-se
um score híbrido:

- **Tier 1** (JRC > 0): scores idênticos ao v1 (0–2m=1.00, 2–4m=0.66, 4–6m=0.33).
- **Tier 2** (JRC = 0): scores reduzidos com ceiling até 15m
  (0–2m=0.50, 2–4m=0.33, 4–6m=0.20, 6–10m=0.10, 10–15m=0.05).

Após rodar o GEE v2, o ETL e o script 06, o resultado foi:

| Métrica | v1 | v2 |
| --- | --- | --- |
| F1 | ≈0.85 | 0.8462 |
| TN | ≈32 k | **34** |
| Hexágonos score > 0 | ~8% | **91–97%** |

O problema do tier 2 é estrutural: HAND < 15m é muito comum no Brasil.
Com 91–97% do território recebendo score > 0, o E2 perdeu poder discriminante.
TN = 34 (de ~32.000 hexágonos SGB-negativos) significa que o indicador classifica
praticamente todo hexágono com cobertura SGB como "risco positivo". A F1 caiu
levemente, mas o indicador deixou de distinguir risco real de background
topográfico.

### Rodada 3 — MapBiomas com overlay SGB (decisão final)

A metodologia de referência (MapBiomas Risco Climático) faz exatamente o que
o v2 tentava: usa HAND × JRC e **adiciona o SGB como camada de reforço** onde
há cartografia disponível. A diferença é que o MapBiomas restringe o SGB a
áreas urbanizadas — o IIC não faz essa restrição, aplicando o SGB a todos os
hexágonos cobertos.

## Decisão

Adotar **MapBiomas v1 (HAND 0–6m × JRC) com overlay SGB** como metodologia
definitiva do E2:

### Base: HAND × JRC (metodologia MapBiomas, inalterada)

| HAND | Score |
| --- | --- |
| 0–2 m | 1.00 |
| 2–4 m | 0.66 |
| 4–6 m | 0.33 |
| > 6 m | 0.00 |

Pixels com JRC = 0 são zerados (máscara multiplicativa).

### Overlay SGB (onde há cartografia disponível)

Onde o SGB confirma alta suscetibilidade com cobertura suficiente, o score é
elevado para 1.00 independentemente do HAND ou do JRC:

```bash
se sgb_alta_mta_frac > 0.3 AND sgb_coverage_frac >= 0.5:
    flood_score = 1.00   # score máximo — SGB é autoridade local
    sgb_override = True
```

Implementado em
[e2_inundacoes_hand.py](../etl/exposure/e2_inundacoes_hand.py):
após consolidar os CSVs GEE, faz left-join com
`br_h3_sgb_inundacoes.parquet` (output do 04) e aplica o override.

O campo `sgb_override` (booleano) é salvo no parquet de output para rastreabilidade.

### Cobertura geográfica

- **HAND × JRC**: cobertura nacional (todos os hexágonos do Brasil).
- **SGB overlay**: ~600 municípios onde a cartografia está disponível.
  Onde o SGB não tem dados, o HAND × JRC é a fonte definitiva. Sem buracos.

### Validação (script 06)

O [06_sgb_validate_e2.py](../etl/exposure/sgb/06_sgb_validate_e2.py) exclui os
hexágonos com `sgb_override = True` antes de calcular métricas — para evitar
validação circular (o E2 nesses hexágonos já usa o SGB como input).
A validação mede exclusivamente a performance da camada HAND × JRC onde a
cartografia SGB existe mas não foi aplicada como override.

## Alternativas consideradas

- **Manter v1 sem overlay SGB**: 99% dos FN continuariam sem captura. Deixaria
  de seguir a metodologia MapBiomas que usa SGB como reforço.

- **Score híbrido v2 (dois tiers)**: descartado — TN = 34, 91–97% do território
  com score > 0. O E2 perdeu poder discriminante.

- **Usar SGB somente em áreas urbanas (como o MapBiomas)**: rejeitado. O IIC
  não tem máscara urbana e a cartografia SGB cobre tanto áreas rurais quanto
  urbanas. Recortar para urbano excluiria FN rurais confirmados pelo SGB.
  Esta é a única diferença metodológica intencional em relação ao MapBiomas.

## Consequências

- **Positivas**:

  - FN urbano-pluviais cobertos pelo SGB são capturados sem inflar o score
    nacional (somente ~600 municípios afetados, com cobertura ≥ 50%).
  - A metodologia HAND × JRC (MapBiomas) é preservada integralmente onde
    não há dados SGB — sem divergência no núcleo metodológico.
  - O indicador mantém poder discriminante: ~8% do território com score > 0,
    não 91–97%.
  - `sgb_override` permite auditoria de quais hexágonos foram afetados pelo SGB.
  - O script 06 continua sendo uma validação independente da camada base
    (exclui os hexágonos de override antes do cálculo).

- **Negativas / trade-offs**:

  - Hexágonos com `sgb_override = True` recebem score 1.00
    independentemente do HAND × JRC. Se um hexágono tem HAND de 20m mas está
    em área mapeada pelo SGB como alta suscetibilidade, o E2 vai indicar risco
    alto. Essa é uma escolha consciente — o SGB é autoridade local e tem
    precedência.
  - A cobertura SGB é parcial e heterogênea: municípios com e sem cartografia
    têm metodologias diferentes por trás do mesmo indicador. Isso é aceitável
    porque o HAND × JRC é a base universal e o SGB só adiciona (nunca remove) risco.
  - Os scripts `07_sgb_analyse_fn_hand.py` e `h3_e2_fn_hand_diagnostic_gee.js`
    foram movidos para `etl/discarded/` — não são mais necessários no pipeline
    produtivo.

- **Confiança**: Alta. A decisão replica fielmente a metodologia MapBiomas,
  adicionando apenas o que ela já prevê (SGB como reforço) com a única
  diferença intencional de não recortar por área urbana.

## Referências

- [e2_inundacoes_hand.py](../etl/exposure/e2_inundacoes_hand.py) — implementação (overlay SGB)
- [h3_e2_inundacoes_hand_gee_v1.js](../etl/gee_scripts/h3_e2_inundacoes_hand_gee_v1.js) — script GEE (inalterado)
- [06_sgb_validate_e2.py](../etl/exposure/sgb/06_sgb_validate_e2.py) — validação (filtra override)
- [ADR-0021](0021-e2-inundacoes-hand-jrc.md) — metodologia original HAND × JRC
- [ADR-0036](0036-diagnostico-hand-falsos-negativos-e2.md) — diagnóstico de FN que iniciou esta investigação
- [ADR-0038](0038-score-e2-hibrido-hand-jrc-tiers.md) — abordagem v2 (descartada)
- MapBiomas Risco Climático — documentação metodológica de referência
