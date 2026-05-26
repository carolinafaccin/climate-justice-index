# ADR-0029: Registrar variáveis consideradas e não-incluídas na dimensão IG (Capacidade de Gestão Municipal)

## Status

Accepted — 2026-05-19

## Contexto

A dimensão IG inclui oito indicadores (ADR-0025). Várias variáveis foram propostas pelos especialistas nas duas rodadas de validação e descartadas. Para uma dimensão que mede capacidade institucional do estado, a tentação de incluir todos os instrumentos existentes é grande — o filtro precisa ser rigoroso para manter o índice acionável.

## Decisão

Registrar variáveis consideradas e descartadas para a dimensão IG.

| Variável sugerida | Categoria(s) ADR-0008 | Razão concreta |
|---|---|---|
| Análise orçamentária via IA (PPA, investimentos em áreas de risco) | (6) operacional | Mineração de dados/IA sobre PPAs municipais é viável conceitualmente, mas exige curadoria por município e algoritmo de classificação que não escala para 5.570 municípios no escopo atual. |
| Capacidade de pagamento do município (Tesouro Nacional) | (8) redundância | Capacidade fiscal já é parcialmente capturada por g1 (investimento em gestão ambiental per capita); adicionar capacidade de pagamento agregada introduziria correlação alta sem ganho informacional claro. |
| Existência de OSCs (Organizações da Sociedade Civil) | (6) operacional; (7) qualidade | Mapeamento Mapa das OSCs (IPEA) tem cobertura desigual; classificação por tema (clima/desastres) exige curadoria que não escala. |
| S2ID — Reconhecimentos Federais de Situação de Emergência e Calamidade | (4) granularidade | Dado disponível apenas em escala municipal; já é um indicador de incidência de desastres (output), não de capacidade institucional (input). Confundiria a dimensão. |
| Existência de planos e políticas ligados ao clima (Planos de Adaptação, Planos Diretores com capítulo climático) | (6) operacional; (7) qualidade | Disponível em Transparência Brasil, mas exige leitura manual de planos para identificar capítulo climático — não escala. |
| Investimento em restauração ambiental | (8) redundância | Já capturado por g1 (função orçamentária 18 — Gestão Ambiental inclui restauração). |
| Existência de Coordenação Municipal de Proteção e Defesa Civil (COMPDEC) | (7) qualidade | Especialistas questionaram a real efetividade da Defesa Civil em validação. Substituído por g3 (NUPDECs — organização comunitária) e g5 (sistema de alerta — métrica objetiva). |
| Plano Diretor / diretrizes urbanísticas atualizadas (últimos 10 anos) | (8) redundância | Sara (PNUMA) e Angela (SERG) sugeriram em validação. Substituído por g2 (Plano de Contingência) — instrumento mais específico para resposta climática que o Plano Diretor genérico. |
| Existência de plataforma de dados abertos | (2) inadequação conceitual | Plataforma de dados abertos não indica necessariamente dados climáticos; substituído por g6 (mapeamento e zoneamento de áreas de risco) como proxy mais direto de "Informação" climática. |
| Indicadores macro de Capacidade Municipal (ICM/MIDR completo) | (8) redundância | Para evitar dupla contagem e distorções, ICM não foi usado como indicador macro. Apenas a variável v7 (cadastro famílias áreas de risco) foi extraída como g7. |

Critérios usam a taxonomia do ADR-0008.

## Alternativas consideradas

- **Incluir mais indicadores para maior abrangência**: aumentaria a dimensão, mas com indicadores redundantes ou de baixa qualidade — dilui o sinal de capacidade institucional efetiva.
- **Não documentar formalmente os descartes**: deixa a defesa do artigo dependente de memória da pesquisadora.
- **Consolidado por dimensão (escolhido)**: estrutura coerente com os ADRs 0026, 0027, 0028; categorias do ADR-0008 garantem uniformidade da justificativa.

## Consequências

- Positivas: IG fica focada em capacidade institucional efetiva, sem inflar com proxies fracos; defesa do artigo preparada; substituições documentadas (COMPDEC → NUPDECs+alerta; Plano Diretor → Plano de Contingência; plataforma de dados → mapeamento de risco) ficam rastreáveis.
- Negativas / trade-offs: ICM completo poderia oferecer dimensões adicionais (mas com risco de dupla contagem); revisitar quando v8/v9 do ICM tiverem nova metodologia.
- Confiança: Alta — cada descarte justificado por razão verificável; substituições explícitas onde aplicável.

## Referências

- ADR-0008 (critérios), ADR-0025 (indicadores IG incluídos).
- Feedback consolidado das duas rodadas de validação, seção "Decisões na Dimensão de Capacidade de Gestão Municipal".
