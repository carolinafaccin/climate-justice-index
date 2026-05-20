# ADR-0026: Registrar variáveis consideradas e não-incluídas na dimensão IP (Grupos Prioritários)

## Status
Accepted — 2026-05-19

## Contexto
Durante o desenho do IIC e nas duas rodadas de validação com especialistas, várias variáveis foram sugeridas para a dimensão IP além das cinco efetivamente incluídas (ADR-0017). Cada descarte tem motivo concreto. Sem registro estruturado, decisões parecem arbitrárias e tornam-se difíceis de defender em revisão por pares.

## Decisão
Registrar as variáveis consideradas e descartadas para a dimensão IP, com a categoria de exclusão correspondente (ADR-0008). A inclusão futura de qualquer dessas variáveis exige novo ADR (não basta editar este).

| Variável sugerida | Categoria(s) de exclusão | Razão concreta |
|---|---|---|
| LGBTQIA+ | (3) cobertura territorial; (4) granularidade | Não há base oficial com cobertura nacional intramunicipal |
| Mães solo | (4) granularidade | Censo permite identificar chefe feminina, mas não "solo" sem cônjuge no domicílio com cobertura intramunicipal confiável |
| Refugiados e imigrantes | (3) cobertura; (4) granularidade | Dados de imigração existem em escala nacional/estadual, não por setor censitário |
| Pessoas com deficiência (PCDs) | (4) granularidade | Variáveis Censo 2022 disponíveis apenas em amostra/agregação municipal, não por setor t0 |
| Mulheres em situação de violência | (1) acesso (LAI); (5) fonte | Microdados de notificação têm sigilo; dados públicos apenas municipais |
| População em situação de rua | (3) cobertura; (7) qualidade | Levantamentos municipais heterogêneos; sem base nacional padronizada |
| População ribeirinha | (4) granularidade; (5) fonte | Não há base oficial com cobertura nacional intramunicipal |
| Gestantes | (4) granularidade; (5) fonte | SINASC/SIH têm georreferenciamento via LAI; dados públicos apenas municipais |
| Mortalidade infantil até 5 anos | (4) granularidade | DataSus/SIM público apenas em escala municipal |
| Mortalidade materna | (4) granularidade | DataSus/SIM público apenas em escala municipal |
| Idosos com comorbidades | (1) acesso (LAI); (4) granularidade | Cruzamento exige microdados restritos |
| 29 povos e comunidades tradicionais mapeados pelo IBGE | (3) cobertura | Mapeamento existe mas não cobre os 5.570 municípios; variável p3 já inclui indígenas e quilombolas, principais grupos do IBGE com cobertura nacional |

Critérios usam a taxonomia do ADR-0008.

## Alternativas consideradas
- **Documentar caso a caso em ADRs separados** (~12 ADRs): granular, mas cada descarte é simples e repetitivo — consolidar em um ADR é mais legível.
- **Não documentar formalmente**: deixa a defesa do artigo dependente de memória da pesquisadora e expõe a metodologia a crítica de "por que não incluíram X".
- **Consolidado por dimensão (escolhido)**: cada exclusão fica visível e rastreável; uso da taxonomia do ADR-0008 torna a justificativa uniforme entre dimensões.

## Consequências
- Positivas: defesa do artigo fica preparada; futuras sugestões de revisores podem ser respondidas com um link direto; categorias do ADR-0008 ficam exercitadas de forma sistemática.
- Negativas / trade-offs: lista pode envelhecer (uma fonte que hoje é restrita pode abrir no futuro) — exige revisão periódica; algumas exclusões dependem de mais de uma categoria, o que adiciona complexidade pequena à leitura.
- Confiança: Alta — cada descarte tem motivo verificável; categorias documentadas no ADR-0008.

## Referências
- ADR-0008 (critérios de exclusão), ADR-0017 (indicadores IP incluídos).
- Feedback consolidado das duas rodadas de validação, seção "Variáveis que não foram incluídas na Dimensão de Grupos Prioritários".
