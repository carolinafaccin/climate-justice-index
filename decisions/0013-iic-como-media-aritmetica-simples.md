# ADR-0013: Calcular o IIC final como média aritmética simples das quatro dimensões

## Status
Accepted — 2026-05-19

## Contexto
O IIC agrega quatro sub-índices (IP, IV, IE, IG invertido) em um único valor por hexágono. A regra de agregação tem peso na interpretabilidade: ponderações desiguais exigem justificativa para cada peso e tornam o resultado mais difícil de comunicar. Esquemas não-lineares (média geométrica, fuzzy, PCA) podem capturar interações entre dimensões, mas reduzem a transparência do cálculo.

## Decisão
**IIC = (IP + IV + IE + (1 − IG)) / 4.** Média aritmética simples das quatro dimensões, com IG entrando invertido (ver ADR-0014). Todos os sub-índices têm peso igual.

## Alternativas consideradas
- **Média ponderada com pesos definidos por especialistas**: poderia refletir prioridades teóricas, mas exigiria justificar cada peso e gera resultado sensível à escolha — qualquer ponderação seria contestável em revisão por pares.
- **Média geométrica ou multiplicativa**: penaliza hexágonos com pelo menos um sub-índice baixo, alinhada a uma leitura de "elo mais fraco". Mas zero em qualquer dimensão zera o índice todo (problemática para IG, que pode ser zero em municípios sem capacidade); leitura menos intuitiva para gestores.
- **PCA ou análise fatorial**: redução estatística baseada em variância, mas obscurece a contribuição de cada dimensão e perde alinhamento com a estrutura conceitual de quatro dimensões (ADR-0003).
- **Média aritmética simples (escolhida)**: fórmula transparente; gestor consegue identificar qual dimensão puxa a nota para baixo e agir; alinhado à recomendação consolidada da validação ("simplificar o cálculo, utilizando uma média aritmética simples de todos os indicadores").

## Consequências
- Positivas: cálculo transparente e replicável; resultado decomposto naturalmente em quatro componentes consultáveis isoladamente; ação política derivada do índice fica clara — sub-índice mais alto aponta o tipo de intervenção prioritária.
- Negativas / trade-offs: trata as quatro dimensões como igualmente importantes, o que é uma escolha normativa em si; não captura interações entre dimensões (ex: alta exposição + baixa gestão é especialmente grave, mas a média soma como qualquer outro caso).
- Confiança: Alta — decisão validada e recomendada explicitamente nas duas rodadas.

## Referências
- ADR-0003 (4 dimensões), ADR-0014 (IG invertido), ADR-0012 (normalização).
- `src/formulas.py` — implementação do cálculo.
- Feedback consolidado das duas rodadas de validação, seções "Sobre o cálculo final" e "Como revisar a dimensão de interseccionalidade".
