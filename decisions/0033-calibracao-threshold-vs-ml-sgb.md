# ADR-0033: Calibrar thresholds dos indicadores existentes em vez de treinar modelos de ML com os dados SGB

## Status
Accepted — 2026-05-22

## Contexto
O SGB cobre ~814 municípios (~15% do total nacional). Os indicadores E1 (NASA LHASA) e E2 (HAND+JRC) têm cobertura nacional mas usam thresholds heurísticos. A questão metodológica é: os dados SGB devem ser usados para (a) calibrar os thresholds dos indicadores já existentes, ou (b) treinar modelos de aprendizado de máquina que extrapolam suscetibilidade mapeada em campo para os ~85% de municípios sem cobertura SGB?

## Decisão
Usar os dados SGB exclusivamente para **calibrar thresholds dos indicadores nacionais existentes** (E1 e E2), não para treinar modelos de ML.

A justificativa tem quatro componentes independentes:

1. **Viés de seleção insolúvel**: O SGB mapeou municípios prioritários por histórico de desastres. O conjunto de dados não contém exemplos negativos verdadeiros — municípios de baixo risco (Cerrado, sertão, planaltos consolidados) simplesmente não foram mapeados. Um modelo treinado nessa amostra sistematicamente superprediz suscetibilidade ao extrapolar para regiões sem cobertura.

2. **Circularidade com os indicadores existentes**: LHASA e HAND já são proxies dos fatores físicos que um modelo de ML usaria (declividade, geologia, distância à drenagem, cobertura vegetal). Treinar ML a partir de LHASA/HAND é circular — aprende o que já sabemos. Treinar a partir de variáveis brutas (DTM, geologia, solo) exigiria curadoria de bases nacionais em escala compatível, o que está fora do escopo do IIC.

3. **Cobertura nacional já existe**: o propósito dos dados SGB no IIC não é criar mapas de suscetibilidade para municípios não cobertos — é calibrar os thresholds dos indicadores nacionais já disponíveis. Após calibração, LHASA e HAND cobrem os ~5.570 municípios.

4. **Interpretabilidade e auditabilidade**: o threshold resultante ("lhasa_high_frac > 0.X") é comunicável e auditável em uma linha. Um modelo de ML não é.

## Alternativas consideradas

- **ML com features brutas (DTM, geologia, solo, precipitação)**: em teoria mais preciso, mas exige curadoria de múltiplas bases nacionais em escala compatível, treinamento com validação cruzada espacial, e ainda sofre do viés de seleção da amostra SGB. Pode ser tema de pesquisa futura se a cobertura SGB se ampliar substancialmente (>40% dos municípios) e se incluir amostragem intencional de municípios de baixo risco.

- **Thresholds regionais (um threshold por macrorregião)**: complementar, não alternativo à calibração de threshold. Se o script 05 mostrar F1 sistematicamente menor em uma macrorregião (ex.: Amazônia), um threshold regional pode ser adotado. Essa análise já está prevista como diagnóstico secundário no script 05.

- **Status quo (thresholds heurísticos sem calibração)**: simples, mas sem evidência empírica de adequação ao Brasil. Inadequado para um índice com pretensão de rigor metodológico.

## Consequências
- Positivas: calibração simples, reproduzível e interpretável; usa os dados SGB em seu propósito mais direto; não amplifica o viés de seleção da amostra; não exige re-exportação GEE salvo se o threshold recomendar uma variável nova (ver ADR-0032 e plano.md).
- Negativas / trade-offs: o threshold calibrado carrega o viés geográfico da amostra SGB (concentrada no Sul/Sudeste/Nordeste); municípios de outros biomas ficam inferidos pelo threshold empírico, não validados; se o F1 for muito heterogêneo por região, um único threshold nacional pode ser subótimo.
- Confiança: Alta — decisão metodologicamente conservadora, coerente com o escopo do IIC e com as limitações da amostra SGB.

## Referências
- ADR-0020: decisão de usar NASA LHASA para E1.
- ADR-0021: decisão de usar HAND+JRC para E2.
- ADR-0032: uso do SGB como referência de calibração; detalhes da amostra e cobertura.
- [etl/exposure/sgb/plano.md](../etl/exposure/sgb/plano.md) — pipeline completo e ações pós-calibração.
