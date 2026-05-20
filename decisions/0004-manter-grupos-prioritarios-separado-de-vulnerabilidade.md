# ADR-0004: Manter Grupos Prioritários (IP) como dimensão separada de Vulnerabilidade (IV)

## Status
Accepted — 2026-05-19

## Contexto
Marcadores sociais (raça, gênero, povos tradicionais, faixa etária) frequentemente se sobrepõem a áreas de baixa renda e infraestrutura precária. Surgiu nas rodadas de validação a sugestão de mesclar IP e IV em uma única dimensão de "vulnerabilidade", reduzindo multicolinearidade e simplificando o índice.

## Decisão
Manter **IP (Grupos Prioritários)** e **IV (Vulnerabilidade Socioeconômica)** como dimensões separadas no IIC. IP foca em **pessoas** (marcadores sociais que denotam desvantagem histórica); IV foca em **território e recursos materiais** do domicílio. Cada uma gera seu próprio sub-índice antes da composição final.

## Alternativas consideradas
- **Mesclar IP em IV** (uma dimensão única de vulnerabilidade): reduziria multicolinearidade aparente e simplificaria o cálculo. Mas os marcadores sociais desapareceriam diluídos em variáveis de renda/infraestrutura, e a resposta padrão a "vulnerabilidade alta" seria obra física (asfalto, drenagem) — ocultando a necessidade de protocolos diferenciados para mulheres, crianças, idosos, indígenas e quilombolas.
- **Calcular IV apenas para os marcadores sociais de IP**: perderia informação sobre populações vulneráveis que não estão nos marcadores selecionados.
- **Manter separado (escolhido)**: preserva visibilidade política dos marcadores sociais, força gestores a olhar para iniquidades além da infraestrutura, e mantém alinhamento com a premissa de "transcender a lógica técnica do risco climático".

## Consequências
- Positivas: marcadores sociais ficam visíveis no resultado; gestores conseguem identificar bairros onde a alta concentração de IP exige resposta pública diferenciada (protocolos de evacuação, acolhimento, etc.); preserva a justiça reconhecional como leitura própria.
- Negativas / trade-offs: aumenta a multicolinearidade entre IP e IV; exige explicar ao leitor a diferença conceitual entre as duas dimensões.
- Confiança: Alta — decisão validada na segunda rodada (Brasília) com argumentação explícita sobre o efeito prático na gestão pública.

## Referências
- ADR-0003 (estrutura de 4 dimensões operacionais).
- Feedback da segunda rodada de validação (Brasília), seção "Não mesclar dimensão de 'grupos prioritários' em 'vulnerabilidade'".
