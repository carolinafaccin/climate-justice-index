# ADR-0032: Usar Cartografia de Suscetibilidade do SGB como referência de calibração de e1 e validação de e2

## Status

Accepted — 2026-05-21. Calibração/validação em andamento; resultados finais atualizarão ADR-0020 (threshold LHASA) e ADR-0021 (teto HAND).

## Contexto

ADR-0020 adotou o NASA LHASA para e1 com threshold `lhasa_high_frac >= 4` (classes High/Very High). Esse threshold seguiu a classificação original do LHASA, sem calibração contra dado brasileiro independente. ADR-0021 adotou o teto de 6 m para HAND em e2, derivado de referências internacionais. Ambas as escolhas carecem de validação contra uma referência de campo nacional.

O **SGB (Serviço Geológico do Brasil / CPRM)** disponibiliza cartografia vetorial de suscetibilidade a movimentos de massa e inundações para ~800 municípios, mapeada por geólogos de campo com metodologia padronizada. É a única base oficial brasileira com cobertura parcial mas tecnicamente sólida para esse fim.

## Decisão

Usar a Cartografia de Suscetibilidade do SGB como **ground truth** em dois usos distintos:

1. **Calibração de e1**: sweep de threshold `lhasa_high_frac > t` (t de 0,0 a 1,0, passo 0,05) contra `sgb_alta_mta_frac > 0,3` (≥ 30% do hexágono em classe Alta ou Muito Alta SGB). Métrica de seleção: F1. Se `lhasa_mean >= 3` superar `lhasa_high_frac >= 4` em F1, indica necessidade de novo GEE export com banda adicional.

1. **Validação de e2**: identificar falsos negativos — hexágonos onde o SGB aponta alta suscetibilidade a inundação mas `flood_score < 0,1` — e analisar a distribuição de HAND nesses casos para avaliar se o teto de 6 m precisa ser revisto.

O pipeline completo está em [`etl/exposure/sgb/`](../etl/exposure/sgb/) (scripts `00` a `06`); detalhes de execução em [`etl/exposure/sgb/plano.md`](../etl/exposure/sgb/plano.md).

## Alternativas consideradas

- **S2iD/Defesa Civil (registros de eventos)**: valida impacto declarado, não suscetibilidade espacial — não permite calibrar threshold geoespacial.
- **BATER (IBGE/Cemaden)**: cobertura restrita a perímetros urbanos, inadequada para calibração nacional (descartada como fonte primária em ADR-0021 pelo mesmo motivo).
- **Validação visual por especialistas**: feita nas duas rodadas de validação para os casos mais visíveis; não é sistemática nem cobrindo os ~800 municípios SGB.
- **Usar o SGB como referência (escolhido)**: cobertura parcial mas geograficamente distribuída; metodologia de campo padronizada; fonte aberta oficial — adequada para calibração de thresholds em escala nacional.

## Consequências

- Positivas: thresholds de e1 e e2 passam a ter fundamento empírico contra dado brasileiro de campo; processo replicável se o SGB expandir a cobertura; pipeline de calibração pode ser reutilizado em revisões futuras do IIC.
- Negativas / trade-offs: SGB cobre ~800 dos 5.570 municípios — calibração representa uma amostra, não o universo; municípios cobertos concentram-se em regiões de maior risco histórico, introduzindo viés de seleção; cobertura pode mudar entre atualizações do SGB.
- Confiança: Média — metodologia de calibração é sólida, mas a amostra parcial limita a generalização dos thresholds derivados.

## Referências

- ADR-0020 (e1 LHASA — será atualizado com threshold calibrado).
- ADR-0021 (e2 HAND+JRC — será atualizado se teto for revisto).
- [etl/exposure/sgb/plano.md](../etl/exposure/sgb/plano.md) — pipeline detalhado.
- [etl/exposure/sgb/](../etl/exposure/sgb/) — scripts `00_sgb_scraper.py` a `06_sgb_validate_e2.py`.
- Site SGB: https://www.sgb.gov.br/produtos-por-estado-cartografia-de-suscetibilidade
