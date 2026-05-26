# ADR-0007: Focar em análise intramunicipal e não em comparação intermunicipal

## Status

Accepted — 2026-05-19

## Contexto

O IIC pode ser interpretado de duas formas: (a) ferramenta para gestores municipais identificarem desigualdades dentro de seu próprio município, ou (b) ranking nacional comparando municípios entre si. Cada uso exige escolhas metodológicas distintas — comparação intermunicipal exigiria mais variáveis em escala municipal, normalização compartilhada nacional e ponderações que diferenciassem perfis de cidades. A segunda rodada de validação (Brasília) levantou explicitamente essa pergunta.

## Decisão

**O IIC é desenhado para uso intramunicipal**, com visualização do índice apenas dentro do território de cada município. Comparações intermunicipais (rankings nacionais) ficam fora do escopo desta versão do índice. A normalização min-max e a visualização final por quintis são calculadas no agrupamento por porte populacional, preservando a leitura local (ver ADR-0015).

## Alternativas consideradas

- **Suporte a ambos os usos**: maior alcance, mas exigiria duplicar parte da metodologia (normalização local vs nacional) e diluir o foco em gestão municipal.
- **Apenas comparação intermunicipal**: relevante para políticas federais e priorização de recursos, mas inviável de implementar com os dados intramunicipais disponíveis sem perder a granularidade do hexágono.
- **Apenas análise intramunicipal (escolhida)**: alinhado ao público-alvo principal (gestores municipais); permite que cada município seja comparado consigo mesmo, evidenciando desigualdades internas.

## Consequências

- Positivas: foco claro de uso; normalização adequada à escala local (não distorcida por extremos nacionais); leitura imediata para o gestor que precisa priorizar áreas dentro do seu município.
- Negativas / trade-offs: o índice não responde "qual município está em pior situação no Brasil"; comparação entre cidades exigirá uma versão futura do IIC com cálculo separado.
- Confiança: Alta — alinhado ao público-alvo principal e validado nas duas rodadas.

## Referências

- ADR-0015 (visualização por quintis dentro de porte populacional).
- Feedback da segunda rodada de validação (Brasília), seção "Manter a comparabilidade em diferentes escalas?".
