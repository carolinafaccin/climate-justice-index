# ADR-0003: Adotar estrutura de quatro dimensões operacionais (IP, IV, IE, IG)

## Status
Accepted — 2026-05-19

## Contexto
Uma das premissas conceituais do IIC é cobrir as quatro dimensões da justiça climática (distributiva, procedural, reconhecional e restaurativa). Essas dimensões conceituais não são diretamente mensuráveis com dados abertos a partir do nível intramunicipal. Foi preciso traduzir o construto teórico em dimensões operacionais que (i) sejam computáveis a partir de fontes públicas brasileiras, (ii) gerem leitura clara para gestores municipais, e (iii) preservem a separação entre "quem" (pessoas) e "onde/como" (território e gestão).

## Decisão
O IIC é composto por **quatro dimensões operacionais**, cada uma gerando um sub-índice agregado por média simples na composição final:
- **IP — Grupos Prioritários**: marcadores sociais que denotam desvantagens históricas.
- **IV — Vulnerabilidade Socioeconômica**: condições materiais do território e dos domicílios.
- **IE — Exposição**: ameaças climáticas físicas com diferenciação intramunicipal.
- **IG — Capacidade de Gestão Municipal**: resposta institucional do município (entra invertido — maior gestão = menor injustiça).

A relação com as dimensões conceituais é transversal: IP opera principalmente a justiça reconhecional, IV e IE operam a distributiva, IG opera a procedural e restaurativa.

## Alternativas consideradas
- **Mapeamento 1:1 com as 4 dimensões conceituais** (Distributiva, Procedural, Reconhecional, Restaurativa): teoricamente puro, mas inviável — não há dados abertos suficientes para operacionalizar cada dimensão conceitual isoladamente, e a nomenclatura é abstrata para gestores municipais.
- **Estrutura clássica de risco IPCC** (Ameaça × Vulnerabilidade × Exposição): bem estabelecida na literatura, mas reduz a justiça climática a risco técnico — viola a premissa de "transcender a lógica técnica do risco climático".
- **Quatro dimensões operacionais (escolhida)**: equilibra rigor teórico (preserva os fundamentos da justiça climática como leitura transversal) com aplicabilidade prática (cada dimensão é computável, comunicável e acionável para gestão pública).

## Consequências
- Positivas: cada dimensão gera um sub-índice consultável isoladamente; gestores conseguem identificar onde agir; mantém vínculo teórico com justiça climática sem prender-se a nomenclatura abstrata.
- Negativas / trade-offs: as 4 dimensões conceituais não aparecem nomeadas no índice final, exigindo explicação no texto do artigo; risco de leitor confundir "vulnerabilidade" do IIC (uma das 4 dimensões) com "vulnerabilidade climática" do IPCC.
- Confiança: Alta — estrutura validada nas duas rodadas com especialistas.

## Referências
- ADR-0004 (separação IP de IV), ADR-0005 (IG como Capacidade de Gestão), ADR-0006 (ordem das dimensões).
- `config/indicators.json` — definição operacional dos indicadores por dimensão.
- Feedback consolidado das duas rodadas de validação, seções "Premissas conceituais" e "Como revisar a dimensão de governança".
