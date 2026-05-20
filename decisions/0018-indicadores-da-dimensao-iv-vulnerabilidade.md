# ADR-0018: Definir os indicadores v1, v2, v3 e v5 da dimensão IV (Vulnerabilidade)

## Status
Accepted — 2026-05-19

## Contexto
A dimensão IV captura condições materiais do território e do domicílio que reduzem a capacidade da população de absorver e responder a eventos climáticos extremos. Os indicadores precisam refletir vulnerabilidade socioeconômica estrutural — não emergencial — e estar disponíveis em escala intramunicipal nacional. O Censo 2022 do IBGE atende esses requisitos para quatro dos cinco indicadores da dimensão; **v4 (saúde) usa fonte e metodologia distintas** e está documentado no ADR-0019 separadamente.

## Decisão
Quatro indicadores da dimensão IV derivam de agregados por setor censitário do Censo 2022, interpolados para hexágonos H3 via `peso_dom` (ADR-0011):

- **v1 — Renda**: inverso da renda média mensal do responsável pelo domicílio, com teto de 2 salários mínimos (R$ 2.424, referência 2022). Hexágonos com renda acima do teto recebem zero; abaixo, gradiente normalizado por min-max p1-p99 e **invertido** (`1 − norm`), pois renda maior = menor vulnerabilidade. Fórmula: `v06004 × v06001` no numerador ponderado, `v06001` no denominador ponderado.
- **v2 — Moradia precária**: percentual de domicílios com ao menos uma condição de inadequação habitacional (lógica OR entre `v00002, v00050, v00051, v00052, v00236, v00237, v00238`). Numerador clipado ao denominador `v00001` para evitar dupla contagem.
- **v3 — Analfabetismo (15+ anos)**: percentual de pessoas com 15 anos ou mais não alfabetizadas. `(v00853 + v00855 + v00857) / v01006`.
- **v5 — Infraestrutura**: percentual de domicílios sem coleta de esgoto, sem abastecimento de água e/ou sem coleta de lixo (lógica OR entre 17 variáveis `v00311–v00316`, `v00112–v00118`, `v00399–v00402`). Numerador clipado ao denominador para evitar dupla contagem.

Todos os quatro normalizados por min-max com winsorização p1-p99 (ADR-0012); todos têm direção positiva (maior valor = mais vulnerabilidade), com inversão semântica embutida no caso de v1.

## Alternativas consideradas
- **Variáveis sugeridas em validação e descartadas**: segurança alimentar, acesso a áreas verdes, acesso a informação, mobilidade urbana, doenças respiratórias, cultura, microdados DataSus georreferenciados, CadÚnico — inviáveis por ausência de granularidade intramunicipal, restrição de acesso (LAI) ou ausência de fonte oficial nacional. Tratadas no ADR-0027.
- **Manter v2 e v5 sem clipping do numerador**: introduziria dupla contagem em domicílios com múltiplas condições simultâneas (ex: domicílio improvisado com banheiro coletivo conta como 2 em vez de 1). Rejeitado por inflar artificialmente o indicador.
- **Não aplicar teto em v1**: tornaria o indicador sensível a outliers extremos de renda; o teto de 2 salários mínimos foca o sinal na faixa de vulnerabilidade efetiva.

## Consequências
- Positivas: quatro indicadores cobrem dimensões materiais clássicas de vulnerabilidade socioeconômica (renda, moradia, educação, saneamento); fonte única, replicável, com periodicidade decenal definida; lógica OR + clipping em v2 e v5 garante interpretabilidade.
- Negativas / trade-offs: v1 com teto suprime informação sobre desigualdade entre faixas de renda média e alta; o IIC vê todos os hexágonos acima de R$ 2.424 como igualmente "não vulneráveis"; isso é defensável conceitualmente (vulnerabilidade material começa a ser preocupante abaixo do teto), mas é uma escolha normativa.
- Confiança: Alta — todos os indicadores derivam de variáveis censitárias estabelecidas, com fórmulas testadas e implementação validada.

## Referências
- ADR-0003 (4 dimensões), ADR-0011 (ponderação dasimétrica), ADR-0012 (normalização), ADR-0019 (v4 saúde), ADR-0027 (variáveis IV não-incluídas).
- [config/indicators.json](../config/indicators.json) — definições operacionais.
- [report/methodological_notes.md](../report/methodological_notes.md) — registro técnico detalhado.
- [etl/census/v1235_p12345_censo2022.py](../etl/census/v1235_p12345_censo2022.py) — implementação.
