# ADR-0014: Inverter IG (Capacidade de Gestão) na composição do IIC final

## Status

Accepted — 2026-05-19

## Contexto

As dimensões IP, IV e IE têm direção semântica positiva: valor maior do sub-índice significa mais injustiça. A dimensão IG (Capacidade de Gestão Municipal — ADR-0005) tem direção semântica oposta: valor maior do sub-índice significa **melhor** gestão, logo **menor** injustiça. Combinar IG diretamente na média aritmética (ADR-0013) com as outras três distorceria o resultado — um município com alta capacidade de gestão receberia um IIC mais alto, o oposto do desejado.

## Decisão

Na composição do IIC final, **IG entra invertido como (1 − IG)**. Internamente, cada indicador g1–g8 mantém o sentido original (maior valor = melhor gestão); a inversão ocorre apenas no momento da agregação no IIC. No `config/indicators.json`, a dimensão `gestao_municipal` é marcada com `invert: true` para sinalizar esse comportamento.

Fórmula final: **IIC = (IP + IV + IE + (1 − IG)) / 4**.

## Alternativas consideradas

- **Inverter cada indicador g1–g8 individualmente** no momento da normalização: tornaria o IG diretamente comparável aos outros sub-índices, mas geraria sub-índice contraintuitivo ("baixa capacidade de gestão"), confundindo gestores que consultam o IG isoladamente.
- **Não usar IG no cálculo final, apresentando-o apenas em separado**: simplifica a fórmula, mas tira a Capacidade de Gestão do índice consolidado, contradizendo a decisão de incluí-la como quarta dimensão (ADR-0005).
- **Inverter apenas no IIC final (escolhida)**: mantém IG legível isoladamente (maior = melhor gestão), e o IIC final continua coerente (maior = mais injustiça).

## Consequências

- Positivas: IG consultado isoladamente mantém leitura natural; IIC final mantém a coerência "maior = pior"; flag `invert: true` em `config/indicators.json` documenta a regra de forma explícita e auditável.
- Negativas / trade-offs: regra de inversão precisa ser explicada toda vez que o cálculo do IIC é apresentado; risco de erro se algum script futuro esquecer a inversão na agregação.
- Confiança: Alta — implementação testada e documentada.

## Referências

- ADR-0003 (4 dimensões), ADR-0005 (IG = Capacidade de Gestão), ADR-0013 (média aritmética simples).
- `config/indicators.json` — flag `invert: true` na dimensão `gestao_municipal`.
- `src/formulas.py` — implementação da inversão na agregação.
