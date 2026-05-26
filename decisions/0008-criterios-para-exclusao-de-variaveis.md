# ADR-0008: Adotar nove critérios formais para exclusão de variáveis candidatas

## Status

Accepted — 2026-05-19

## Contexto

Durante o desenho do IIC e ao longo das duas rodadas de validação, dezenas de variáveis foram sugeridas por especialistas e descartadas por motivos heterogêneos. Sem uma taxonomia explícita, o descarte parece arbitrário e fica difícil defender no artigo cada caso particular. É preciso uma lista padronizada de razões reutilizável em todos os ADRs de não-inclusão.

## Decisão

Toda exclusão de variável candidata deve invocar **uma ou mais** das nove categorias abaixo:

1. **Restrição de acesso e transparência** — dado fechado, dependente de LAI ou sigilo (impede reprodutibilidade pública).
1. **Inadequação conceitual** — variável não tem aderência teórica com as dimensões do IIC.
1. **Lacunas de cobertura territorial** — base não cobre todos os 5.570 municípios.
1. **Insuficiência de granularidade** — não há dado na escala intramunicipal (apenas municipal ou estadual).
1. **Ausência de fonte confiável** — não há base oficial nem sistematização nacional do fenômeno.
1. **Inviabilidade operacional** — complexidade de cálculo ou esforço de limpeza/geocodificação excede recursos técnicos do projeto.
1. **Baixa qualidade ou confiabilidade** — campos vazios, erros de preenchimento ou ruído alto a ponto de comprometer validade estatística.
1. **Redundância estatística** — variável mensura algo já capturado por outra mais robusta.
1. **Defasagem temporal** — última atualização é antiga e não reflete a realidade atual.

Todo ADR de não-inclusão (ADR-0026 a ADR-0031) cita a(s) categoria(s) aplicável(eis) a cada variável descartada.

## Alternativas consideradas

- **Sem taxonomia formal**: cada descarte justificado caso a caso. Mais flexível, mas dificulta comparação entre decisões e ataca a coerência geral do índice em revisão por pares.
- **Taxonomia com mais categorias** (12-15 critérios): mais granular, mas maioria das subcategorias acabaria sem uso real.
- **Nove critérios (escolhido)**: cobre todos os casos de descarte observados nas rodadas de validação sem inflar a lista; permite invocar múltiplas categorias por variável quando aplicável.

## Consequências

- Positivas: ADRs de não-inclusão ficam mais curtos (basta citar a categoria); coerência entre decisões fica visível; defesa em revisão por pares fica mais sólida.
- Negativas / trade-offs: nem toda exclusão se encaixa perfeitamente — algumas exigirão combinação de duas ou três categorias; lista pode precisar de revisão se surgir caso novo não coberto.
- Confiança: Alta — categorias derivadas da prática real de descarte ao longo de duas rodadas de validação.

## Referências

- ADR-0026 (não-inclusão em IP), ADR-0027 (não-inclusão em IV), ADR-0028 (não-inclusão em IE — lista geral), ADR-0029 (não-inclusão em IG), ADR-0030 (não-inclusão de seca SPI/CHIRPS), ADR-0031 (não-inclusão de NDVI áreas verdes).
- Feedback consolidado das duas rodadas de validação, seção "Categorias para a não inclusão de variáveis".
