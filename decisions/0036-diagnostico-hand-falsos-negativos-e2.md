# ADR-0036: Diagnosticar teto HAND do E2 via análise dos falsos negativos contra SGB

## Status
Accepted — 2026-05-25

## Contexto

A validação do E2 (inundações HAND+JRC) contra a cartografia SGB rodada pelo
[06_sgb_validate_e2.py](../etl/exposure/sgb/06_sgb_validate_e2.py) mostrou que
uma fração relevante dos hexágonos onde o SGB indica alta suscetibilidade
recebe `flood_score = 0` no E2 — isto é, falsos negativos. O teto HAND atual
está fixado em 6 m no script GEE
([h3_e2_inundacoes_hand_gee_v1.js](../etl/gee_scripts/h3_e2_inundacoes_hand_gee_v1.js)),
o que descarta hexágonos com HAND > 6 m mesmo quando o SGB os classifica como
alta ou muito alta suscetibilidade.

Dois fatos restringem a análise:

1. **O parquet E2 não preserva HAND bruto** — só armazena o score já classificado
   (0/0.33/0.66/1.00). Logo o script 06, sozinho, não pode dizer qual seria o
   teto HAND ideal.
2. **A intuição empírica é heterogênea entre regiões** — S e SE concentram
   serras com chuvas intensas (RS 2024: rios subindo até 19 m no Vale do
   Taquari), enquanto N/NE/CO têm geomorfologia diferente. Um único teto
   nacional pode não ser adequado.

Sem uma medição direta de HAND nos hexágonos FN, qualquer ajuste do teto
seria por chute.

## Decisão

Introduzir um diagnóstico em dois passos que mede HAND bruto nos falsos
negativos e produz uma tabela quantitativa de tetos candidatos:

### Passo GEE — [h3_e2_fn_hand_diagnostic_gee.js](../etl/gee_scripts/h3_e2_fn_hand_diagnostic_gee.js)
- Consome o CSV `diagnostic_e2_fn_hexagons_<ts>.csv` exportado pelo 06,
  uploadado no GEE como Table asset.
- Faz join com o asset existente `br_h3_lat_lon` para recuperar geometria
  (evita exigir lat/lon no CSV de upload).
- Reduz, por hexágono FN: `hand_{mean, min, max, p25, p50, p75, p90}` +
  `jrc_rp100_mean`.
- Exporta 5 CSVs (um por macrorregião) para o Drive.

### Passo local — [07_sgb_analyse_fn_hand.py](../etl/exposure/sgb/07_sgb_analyse_fn_hand.py)
- Consolida os CSVs do GEE.
- Calcula, para cada teto candidato `{8, 10, 12, 15, 20}` m × macrorregião:
  quantos FN seriam recuperados (`hand_mean ≤ teto AND jrc > 0`).
- Distingue FN limitados pelo teto HAND de FN com `jrc_rp100 = 0`
  (gargalo é a cobertura JRC, não o HAND — ampliar o teto não recupera esses).
- Outputs em `cfg.DIAGNOSE_DIR`:
  - `diagnostic_07_fn_hand_<ts>.txt` — relatório legível
  - `diagnostic_07_fn_hand_candidates_<ts>.csv` — tabela macro × teto

O resultado dessa análise vai informar uma futura ADR sobre como ajustar o
script GEE v2 do E2 (teto único vs classes adicionais vs teto por região).

## Alternativas consideradas

- **Aumentar o teto HAND no GEE empiricamente (sem medição)**: simples mas
  arbitrário. Não distingue entre regiões e não dá rastreabilidade da decisão.
  Rejeitada.

- **Armazenar HAND bruto no parquet E2**: permitiria a análise direto no
  script 06, sem GEE adicional. Rejeitada: aumentaria o tamanho do parquet
  para todo o Brasil só para investigar uma fração de FN, e perderia a
  máscara JRC (que também precisa entrar na análise). O custo não compensa
  para uma análise pontual.

- **Análise focada só nos FN via GEE (escolhida)**: mede HAND e JRC exatamente
  onde importa, sem alterar o pipeline produtivo. Custo: um upload manual de
  CSV e 5 tasks GEE.

- **Trocar JRC por outra fonte de hazard (ex: GLOFAS futuro, GPM)**:
  considerada, mas é uma decisão maior que a do teto HAND. Vai ser avaliada
  separadamente *se* o diagnóstico mostrar que o JRC é o gargalo principal
  em N/NE/CO.

## Consequências

- **Positivas**:
  - A decisão sobre o teto HAND passa a ser baseada em dados, não em
    intuição. Fica rastreável via os CSVs de diagnóstico arquivados em
    `cfg.DIAGNOSE_DIR`.
  - O passo separa dois problemas que se confundiam ("E2 não detecta" pode
    ser limitação do teto HAND *ou* falta de cobertura JRC), permitindo
    tratá-los independentemente.
  - O workflow é reusável: se SGB liberar novas UFs ou se mudarmos algum
    parâmetro upstream, basta repetir os passos 06 → upload → GEE → 07.

- **Negativas / trade-offs**:
  - Adiciona um passo manual (upload do CSV como asset GEE). Não há API
    do GEE para upload de tabelas via earthengine-api Python sem GCS.
  - Gera dependência temporária de 5 CSVs em um diretório `raw/gee/` que
    precisa ser limpo após a análise.
  - A análise é pontual — se o teto HAND for ajustado, o pipeline E2 vai
    precisar ser rerodado por inteiro (GEE v2 + ETL).

- **Confiança**: Alta — a metodologia é descritiva (não preditiva) e o que
  ela produz é exatamente o input que faltava para a próxima decisão.

## Referências

- [06_sgb_validate_e2.py](../etl/exposure/sgb/06_sgb_validate_e2.py) — gera o CSV de FN
- [h3_e2_fn_hand_diagnostic_gee.js](../etl/gee_scripts/h3_e2_fn_hand_diagnostic_gee.js) — diagnóstico HAND no GEE
- [07_sgb_analyse_fn_hand.py](../etl/exposure/sgb/07_sgb_analyse_fn_hand.py) — análise local e tabela de candidatos
- [ADR-0021](0021-e2-inundacoes-hand-jrc.md) — metodologia original do E2 (teto 6 m)
- [ADR-0032](0032-sgb-como-referencia-calibracao.md) — SGB como referência de calibração
- [ADR-0033](0033-calibracao-threshold-vs-ml-sgb.md) — escolha do método de calibração
