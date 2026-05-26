# ADR-0025: Definir os oito indicadores da dimensão IG (Capacidade de Gestão Municipal)

## Status

Accepted — 2026-05-19

## Contexto

A dimensão IG foi reformulada (ADR-0005) para medir capacidade institucional do município de responder a impactos climáticos — investimento, planejamento, participação, governança, resposta, mapeamento, reconhecimento e direitos humanos. Diferentemente das outras três dimensões, a maioria dos dados disponíveis para essa dimensão (MUNIC, ICM, Siconfi) é coletada em **escala municipal**, não intramunicipal. Isso exige uma escolha de premissa: forçar interpolação intramunicipal artificial ou aceitar que IG opera em escala diferente.

## Decisão

A dimensão IG é composta por **oito indicadores**, todos calculados em **escala municipal** e propagados uniformemente para todos os hexágonos do município. Essa é uma exceção consciente à premissa de análise intramunicipal (ADR-0007) — IG mede capacidade institucional, que é atributo do município como um todo, não do território.

| Código | Indicador | Fonte | Tipo | Variável |
|--------|-----------|-------|------|----------|
| g1 | Investimento em gestão ambiental | Siconfi/FINBRA 2015–2024 | Contínuo (log1p) | Função 18 per capita |
| g2 | Plano de contingência | MUNIC 2023 | Binário | smap123 |
| g3 | NUPDECs | MUNIC 2020 | Binário | mgrd213 |
| g4 | Conselhos municipais (Meio Ambiente OR Cidade) | MUNIC 2023 | Binário | sdg353 OR sdg351 |
| g5 | Sistema de alerta de riscos | MUNIC 2023 | Binário | smap126 |
| g6 | Mapeamento de áreas de risco | MUNIC 2023 | Binário | smap122 |
| g7 | Cadastro de famílias em áreas de risco | ICM/MIDR 2026 | Binário | v7 |
| g8 | Políticas e programas de direitos humanos | MUNIC 2023 | Contínuo (0–21) | mdhu571–mdhu5716, mdhu58, mdhu61, mdhu64, mdhu67, mdhu69 |

**Normalização**: g1 com transformação log1p antes do min-max p1-p99 (distribuição com assimetria forte à direita); g2–g6 binários (`Sim`/`Não` → 1/0) sem transformação; g7 binário; g8 normalizado por min-max p1-p99 a partir da contagem de iniciativas ativas.

**Direção semântica**: maior `g_norm` = melhor capacidade de gestão. A inversão para o IIC final ocorre no momento da agregação (ADR-0014: `1 − IG_médio`).

## Alternativas consideradas

- **Forçar interpolação intramunicipal**: dados municipais propagados aos hexágonos via algum gradiente artificial (distância ao centro administrativo, densidade populacional etc.). Rejeitado por inventar variação que não existe na fonte — capacidade institucional é atributo da prefeitura, não de bairros.
- **Não incluir IG por inconsistência de escala**: simplificaria a estrutura, mas removeria a única dimensão que mede resposta institucional, tornando o IIC apenas diagnóstico. Rejeitado pelo ADR-0005.
- **Incluir COMPDEC (Coordenação Municipal de Proteção e Defesa Civil) como indicador**: considerado, mas alguns especialistas questionaram a real efetividade das COMPDECs na validação. Substituído por g3 (NUPDECs) e g5 (sistema de alerta), métricas mais objetivas.
- **Variáveis adicionais sugeridas em validação e descartadas**: análise orçamentária via IA (PPA, investimentos), capacidade de pagamento Tesouro Nacional, existência de OSCs, S2ID reconhecimentos federais, planos diretores, plataformas de dados abertos. Inviabilidades documentadas no ADR-0029.
- **Adotar como informação a existência de plataforma de dados abertos**: opção considerada e rejeitada — substituída pelo **mapeamento de áreas de risco (g6)** como indicador de "Informação" mais conectado ao tema climático.
- **Oito indicadores como definidos (escolhido)**: cobre as categorias-chave de gestão municipal pública brasileira; usa apenas fontes oficiais abertas; mistura instrumentos (planos, conselhos, sistemas) com investimento e proteção social; alinhado à recomendação consolidada de validação.

## Consequências

- Positivas: dimensão usa apenas fontes oficiais e abertas brasileiras; cada indicador é interpretável isoladamente por um gestor municipal ("nosso município tem ou não tem este instrumento?"); inclui g8 contínuo para captar profundidade de política em direitos humanos (não apenas existência).
- Negativas / trade-offs: escala municipal contradiz a premissa intramunicipal do índice — exige explicação no artigo e na interface do dashboard (a leitura "mesmo valor de IG em toda a cidade" pode confundir); MUNIC tem coletas em anos distintos (2020 e 2023), introduzindo defasagem entre indicadores; binários perdem gradação (município com plano de contingência básico vale igual a um com plano completo e atualizado); ICM/MIDR é nova fonte (2026), sem histórico de continuidade.
- Confiança: Média — escolhas operacionais defensáveis, mas a heterogeneidade temporal das fontes MUNIC e a binarização limitam precisão.

## Referências

- ADR-0003 (4 dimensões), ADR-0005 (IG = Capacidade de Gestão Municipal), ADR-0007 (foco intramunicipal — exceção registrada aqui), ADR-0012 (normalização), ADR-0014 (IG invertido), ADR-0016 (g2–g6 binários sem winsorização), ADR-0029 (variáveis IG não-incluídas).
- [config/indicators.json](../config/indicators.json) — definições operacionais.
- [report/methodological_notes.md](../report/methodological_notes.md) — registro técnico detalhado.
- [etl/governance/g1_siconfi.py](../etl/governance/g1_siconfi.py), [etl/governance/g234568_munic.py](../etl/governance/g234568_munic.py) — implementação.
- Feedback consolidado das duas rodadas de validação, seção "Decisões na Dimensão de Capacidade de Gestão Municipal".
