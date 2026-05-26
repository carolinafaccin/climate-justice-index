# ADR-0006: Apresentar as dimensões na ordem IP → IV → IE → IG

## Status

Accepted — 2026-05-19

## Contexto

A ordem natural para um índice baseado em risco climático seguiria a narrativa do IPCC: Ameaça/Exposição → Suscetibilidade/Vulnerabilidade → Resposta. Aplicada ao IIC, essa ordem seria IE → IV → IP → IG. No entanto, uma das premissas conceituais do índice é "transcender a lógica técnica do risco climático" e tratar a injustiça climática como fenômeno primariamente social, não físico. Começar pela dimensão física (IE) contradiz essa premissa narrativamente.

## Decisão

Apresentar as dimensões na ordem **IP → IV → IE → IG** em toda a comunicação do índice (relatório, dashboard, artigo, materiais de validação). A ordem responde a quatro perguntas em sequência:

1. **IP — Quem?** Pessoas e marcadores de desvantagem histórica.
1. **IV — Como vivem?** Condições materiais e iniquidades estruturais.
1. **IE — Qual o gatilho climático?** Ameaça física como agravante de uma situação preexistente.
1. **IG — Qual a resposta do Estado?** Capacidade institucional do município.

## Alternativas consideradas

- **Ordem IPCC (IE → IV → IP → IG)**: alinhada à literatura de risco; mas trata a injustiça como consequência do evento climático, não como condição estrutural preexistente.
- **Ordem por força de contribuição estatística**: mais defensável metodologicamente, mas tornaria a ordem instável entre versões e perderia a função narrativa.
- **Ordem IP → IV → IE → IG (escolhida)**: estabelece que o índice é sobre pessoas e desvantagens históricas antes de ser sobre eventos físicos; o evento climático aparece como agravante, não como causa primária.

## Consequências

- Positivas: alinha a narrativa do índice à sua premissa conceitual; reforça o caráter social-primeiro ao leitor; ajuda a diferenciar o IIC de índices puramente baseados em risco climático.
- Negativas / trade-offs: leitor familiarizado com a literatura IPCC pode estranhar a inversão; exige uma frase de justificativa toda vez que a estrutura é apresentada.
- Confiança: Média — a ordem não altera o cálculo final (média simples), apenas a leitura. Decisão revisitável se gestores reportarem confusão.

## Referências

- ADR-0003 (estrutura de 4 dimensões operacionais).
- Feedback da segunda rodada de validação (Brasília), seção "Ordenar 'grupos prioritários' em primeiro lugar?".
