# ADR-0001: Adotar Architecture Decision Records (ADRs)

## Status

Accepted — 2026-05-19

## Contexto

O IIC tem dezenas de decisões metodológicas espalhadas em notas, scripts arquivados e conhecimento tácito. Sem registro estruturado, revisores do artigo, terceiros e a própria pesquisadora no futuro têm dificuldade de recuperar o "porquê" de cada escolha.

## Decisão

Adotamos ADRs no formato **MADR light** em português, com um arquivo por decisão em `decisions/NNNN-titulo-kebab-case.md`. Cada ADR tem cinco seções: Status, Contexto, Decisão, Alternativas consideradas, Consequências (com nível de confiança). Ver template em [`decisions/_template.md`](_template.md).

## Alternativas consideradas

- Nada formal (status quo) — barato, mas o problema persiste.
- Documento único de metodologia — vira Mega-ADR; difícil rastrear evolução.
- MADR completo (com deciders, consulted, confirmation) — overhead alto para projeto solo.

## Consequências

- ADRs nunca são editados após `Accepted` — se a decisão muda, cria-se novo ADR e o antigo recebe `Superseded by ADR-NNNN`.
- ADRs retrospectivos entram como `Accepted` com a data de redação.
- Nível de confiança Baixo/Médio sinaliza pontos a revisitar antes da submissão do artigo.
- Custo: ~30+ ADRs retrospectivos a redigir; disciplina contínua para registrar novas decisões.
- Confiança: Alta — prática consolidada, baixo custo de manutenção.

## Referências

- [Nygard, 2011 — Documenting Architecture Decisions](https://www.cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [Zimmermann, 2022 — MADR Template Primer](https://www.ozimmer.ch/practices/2022/11/22/MADRTemplatePrimer.html)
