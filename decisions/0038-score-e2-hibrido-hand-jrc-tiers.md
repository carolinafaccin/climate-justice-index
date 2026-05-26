# ADR-0038: Score E2 híbrido em dois tiers (HAND × JRC alta confiança + HAND-puro confiança topográfica)

## Status

Superseded — 2026-05-26 → ver [ADR-0039](0039-e2-mapbiomas-com-overlay-sgb.md)

## Contexto

A análise de falsos negativos produzida pelo
[07_sgb_analyse_fn_hand.py](../etl/exposure/sgb/07_sgb_analyse_fn_hand.py) sobre
o E2 v1 ([h3_e2_inundacoes_hand_gee_v1.js](../etl/gee_scripts/h3_e2_inundacoes_hand_gee_v1.js))
mostrou um achado dominante:

- **99% dos FN têm `jrc_rp100 = 0`** (S e SE). A máscara JRC multiplicativa do
  v1 é a principal responsável pelos FN, não o teto HAND.
- Subir só o teto HAND mantendo a exigência `jrc > 0` recupera ~1% dos FN.
- Sem a máscara JRC, um teto de 15 m recupera 48% (SE) e 57% (S) dos FN.
- Os FN concentram-se em macrorregiões S/SE (CO/NE/N caem fora do filtro de
  cobertura SGB ≥ 50% do 06; não há dado para diagnosticar lá).

Causa raiz: o JRC RP100 é modelado a ~1km globalmente, calibrado para grandes
cheias **fluviais**. Áreas urbano-pluviais e de flash flood — que o SGB mapeia
em detalhe e onde se concentram os FN — não estão no JRC. O JRC sub-representa
o risco pluvial urbano brasileiro.

A metodologia de referência (MapBiomas Risco Climático) usa HAND × JRC e
**adiciona o SGB como camada complementar de reforço**. No nosso pipeline isso
não é uma opção operacional: o SGB tem cobertura incompleta (cartografia
disponível em ~600 municípios), e o E2 precisa cobertura nacional — não dá
para integrar SGB diretamente no E2 sem deixar buracos.

Restrições adicionais:

1. **Não podemos usar SGB no E2 sem perder o validador.** Se o SGB virasse
   componente do E2, o script 06 não poderia mais ser usado para calibração
   independente (vazamento metodológico).
1. **Não podemos remover o JRC completamente.** Isso reduziria o E2 a um
   proxy puramente topográfico (HAND), inflaria FP em áreas pristinas longe
   de qualquer histórico de inundação e abandonaria a corroboração
   hidrológica que MapBiomas considera essencial.

## Decisão

Adotar um **score E2 híbrido em dois tiers** no
[h3_e2_inundacoes_hand_gee_v2.js](../etl/gee_scripts/h3_e2_inundacoes_hand_gee_v2.js).
Cada pixel é classificado em exatamente um dos dois tiers, em função do JRC
naquele pixel:

### Tier 1 — JRC > 0 (alta confiança hidrológica)

Score idêntico ao v1. Onde o JRC corrobora, a metodologia MapBiomas é
preservada na íntegra.

| HAND | Score |
| --- | --- |
| 0–2 m | 1.00 |
| 2–4 m | 0.66 |
| 4–6 m | 0.33 |
| > 6 m | 0.00 |

### Tier 2 — JRC = 0 (confiança topográfica apenas)

Score reduzido (teto = 0.50) + ceiling HAND estendido até 15 m. Sinaliza
"topografia favorável a inundação, sem corroboração do modelo hidrológico
global". Captura FN urbano-pluviais sem que o E2 vire HAND puro.

| HAND | Score |
| --- | --- |
| 0–2 m | 0.50 |
| 2–4 m | 0.33 |
| 4–6 m | 0.20 |
| 6–10 m | 0.10 |
| 10–15 m | 0.05 |
| > 15 m | 0.00 |

### Fórmula

```bash
flood_score = tier1_score · 1{jrc > 0} + tier2_score · 1{jrc = 0}
```

Os tiers são mutuamente exclusivos por construção. Pixels com `hand = 0`
(canal de drenagem) e `hand` inválido (oceano/NoData) são mascarados.

### Validação pós-implementação

Após rodar o v2 e o ETL Python, re-rodar
[06_sgb_validate_e2.py](../etl/exposure/sgb/06_sgb_validate_e2.py) para
comparar TP/FP/FN/F1 entre v1 e v2 nos mesmos hexágonos de referência SGB.
Esperado: F1 maior no v2 (mais TP recuperados, mantendo FP controlado pelo
score baixo do tier 2).

## Alternativas consideradas

- **Manter v1 (HAND × JRC clássico)**: rejeitada. Documentaria o gap
  metodológico mas deixaria 99% dos FN urbanos sem captura, mesmo sabendo
  que o gargalo é a sub-representação do JRC em escala global.

- **Remover JRC, usar HAND puro com ceiling estendido (v2 inicial proposto)**:
  rejeitada porque (a) diverge frontalmente do MapBiomas sem corroboração
  hidrológica nas áreas onde o JRC funciona bem (grandes rios), (b) infla FP
  em áreas pristinas perto de drenagem natural sem histórico de inundação,
  (c) o E2 deixa de ser "risco modelado de inundação" e vira "proximidade
  vertical da rede de drenagem".

- **Integrar SGB diretamente no E2 (estilo MapBiomas)**: rejeitada por dois
  motivos. (a) Cobertura SGB é parcial (~600 municípios) — usar SGB no E2
  deixaria buracos onde não há cartografia, criando um indicador
  heterogêneo. (b) Perda do validador independente: o 06 não poderia mais
  calibrar o E2 contra o SGB.

- **Tiers contínuos (score multiplicado por confiança)**: ex.
  `flood_score = hand_score · (0.5 + 0.5 · jrc_normalized)`. Mais elegante
  matematicamente mas menos interpretável. Os thresholds discretos por
  classe são a convenção do MapBiomas; mantê-los facilita comparação.
  Rejeitada por simplicidade.

- **Teto por macrorregião**: o diagnóstico permitiria, mas a diferença entre
  S (p50 = 9.6 m) e SE (p50 = 16.3 m) é parcialmente artefato do filtro de
  cobertura SGB e tamanho amostral diferente. Adicionaria complexidade sem
  ganho claro. Adiada — pode ser revisitada em v3 se necessário.

## Consequências

- **Positivas**:

  - O E2 deixa de zerar em áreas urbano-pluviais que o SGB confirma como de
    alta suscetibilidade, sem virar proxy topográfico puro.
  - A metodologia MapBiomas é preservada nos pixels onde JRC corrobora
    (~grandes rios, áreas fluviais clássicas) — score máximo continua sendo
    1.00 onde antes era 1.00.
  - O score do tier 2 é baixo o suficiente (≤ 0.50) para que o ranking final
    nunca confunda "alta confiança hidrológica" com "topografia sugere mas
    sem confirmação".
  - SGB continua disponível como validador independente — o pipeline 06
    permanece útil para mensurar TP/FP/FN do v2 vs v1.

- **Negativas / trade-offs**:

  - Diverge ligeiramente do MapBiomas (que não usa tier 2). É uma divergência
    justificada por evidência empírica nacional, documentada neste ADR.
  - Requer re-execução completa do pipeline E2 (GEE v2 + ETL Python +
    H3 intersect) — não é só ajuste de parâmetro.
  - Os scores do tier 2 (0.50, 0.33, 0.20, 0.10, 0.05) são uma escolha
    informada mas não derivada de calibração formal. Valores podem ser
    refinados após validação 06 v1-vs-v2 mostrar onde o tier 2 acerta/erra.
  - Comparações com produtos MapBiomas requerem cuidado: o E2 IIC não é
    mais um clone do componente urbano-flood deles.

- **Confiança**: Média-alta. A direção da decisão (relaxar o JRC binário) é
  fortemente apoiada pelo diagnóstico do 07. Os valores específicos dos
  scores do tier 2 são uma calibração heurística que pode ser ajustada
  após a primeira rodada de validação.

## Referências

- [h3_e2_inundacoes_hand_gee_v2.js](../etl/gee_scripts/h3_e2_inundacoes_hand_gee_v2.js) — implementação
- [h3_e2_inundacoes_hand_gee_v1.js](../etl/gee_scripts/h3_e2_inundacoes_hand_gee_v1.js) — versão anterior
- [07_sgb_analyse_fn_hand.py](../etl/exposure/sgb/07_sgb_analyse_fn_hand.py) — diagnóstico que motivou a decisão
- [06_sgb_validate_e2.py](../etl/exposure/sgb/06_sgb_validate_e2.py) — validador para comparar v1 vs v2
- [ADR-0021](0021-e2-inundacoes-hand-jrc.md) — metodologia original do E2 (v1)
- [ADR-0036](0036-diagnostico-hand-falsos-negativos-e2.md) — diagnóstico HAND nos FN
- MapBiomas Risco Climático — documentação metodológica de referência
