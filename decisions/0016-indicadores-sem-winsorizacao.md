# ADR-0016: Não aplicar winsorização em indicadores binários ou espacialmente concentrados

## Status
Accepted — 2026-05-19

## Contexto
O ADR-0012 estabeleceu a normalização min-max com winsorização p1-p99 como padrão. A função `normalize_minmax` em [src/utils.py](../src/utils.py) recebe `winsorize` como parâmetro opt-in (default `False`), e cada ETL decide explicitamente. Em testes preliminares, alguns indicadores apresentaram comportamento problemático com winsorização ativa: quando o indicador é geograficamente concentrado (>90% dos hexágonos com valor zero), o P99 fica próximo de zero e a normalização colapsa — todos os hexágonos com valor não-zero virariam 1 ou ficariam idênticos.

## Decisão
Não aplicar winsorização em duas classes de indicadores:

**(a) Indicadores binários por município** — valores já no domínio {0, 1}, normalização sem efeito:
- `g2` Plano de contingência (MUNIC)
- `g3` NUPDECs (MUNIC)
- `g4` Conselhos municipais (MUNIC)
- `g5` Sistema de alerta (MUNIC)
- `g6` Mapeamento de áreas de risco (MUNIC)

Esses indicadores não passam pela função `normalize_minmax` — o valor `_norm` é atribuído diretamente do campo binário do MUNIC.

**(b) Indicadores espacialmente concentrados** — passam por `normalize_minmax(winsorize=False)` porque <15% dos hexágonos têm valor não-zero:
- `e1` Deslizamentos (LHASA) — ~8% de hexágonos com suscetibilidade > 0.
- `e2` Inundações (HAND + JRC) — ~11% de hexágonos em planícies de inundação.
- `e3` Elevação do mar (DEM Copernicus) — ~8% de hexágonos em zona costeira ≤1m.

Demais indicadores (p1-p5, v1-v5, v4, e4, e5, g1, g7, g8) usam `winsorize=True` com limites (0.01, 0.99).

## Alternativas consideradas
- **Forçar winsorização em todos os indicadores**: simples e uniforme, mas colapsa a normalização dos indicadores espacialmente concentrados, anulando sua contribuição ao IIC.
- **Winsorização com limites assimétricos** (ex: 0.00, 0.99): aproveitaria parte da robustez sem zerar P99 para indicadores esparsos, mas adiciona uma terceira regra e complica a explicação.
- **Excluir indicadores esparsos do índice**: removeria fenômenos legítimos do IIC (deslizamentos, inundações, elevação do mar são problemas reais mesmo em poucos hexágonos).
- **Não aplicar winsorização nos casos descritos (escolhida)**: preserva a contribuição dos indicadores esparsos ao IIC; o min-max sem winsorização nesses casos ainda resulta em [0,1] porque os valores absolutos têm distribuição menos cauda-pesada (LHASA, HAND-flood-score, altitude binária ≤1m).

## Consequências
- Positivas: indicadores espacialmente concentrados contribuem corretamente ao IIC; binários ficam transparentes; cada ETL declara explicitamente sua escolha de winsorização (auditável).
- Negativas / trade-offs: regra heterogênea entre indicadores exige documentação caso a caso; uma nova versão de indicador com distribuição diferente pode precisar de reavaliação da escolha; nenhum indicador esparso é completamente imune a outliers — a confiança em valores extremos é menor.
- Confiança: Média — a heurística "<15% de hexágonos não-zero" é informal; revisitar se um indicador futuro ficar na faixa cinza (15-30%).

## Referências
- ADR-0012 (normalização min-max), ADR-0026 (e1 LHASA — a ser criado).
- [src/utils.py:73-90](../src/utils.py#L73-L90) — função `normalize_minmax` com parâmetro `winsorize`.
- [explore/checks/check_normalization.py](../explore/checks/check_normalization.py) — script que valida o comportamento da winsorização em cada indicador (status `SPARSE` sinaliza casos que exigem `winsorize=False`).
- [etl/exposure/e1_deslizamentos_lhasa.py:89-91](../etl/exposure/e1_deslizamentos_lhasa.py#L89-L91), [etl/exposure/e2_inundacoes_hand.py:91-92](../etl/exposure/e2_inundacoes_hand.py#L91-L92), [etl/exposure/e3_mar.py:80-81](../etl/exposure/e3_mar.py#L80-L81) — chamadas com `winsorize=False` e comentários explicativos.
