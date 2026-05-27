# ADR-0042: Usar análise de clusters k=4 para selecionar municípios como estudo de caso

## Status

Accepted — 2026-05-27

## Contexto

Para um artigo científico sobre o IIC, é necessário selecionar 4 municípios como estudos de caso que ilustrem perfis distintos do índice. O IIC foi projetado para uso intramunicipal (ADR-0007), mas para seleção de casos o interesse é intermunicipal: identificar municípios com perfis marcadamente diferentes nas quatro dimensões (IP, IV, IE, IG). O foco são municípios de pequeno e médio porte com dados mais limitados, para os quais o índice é mais útil como ferramenta de diagnóstico.

## Decisão

Utilizar **análise de agrupamentos k-means com k=4** nas quatro sub-dimensões do IIC (IP, IV, IE, IG) para identificar quatro perfis distintos. Um município representativo é selecionado de cada cluster, priorizando: (1) pequeno ou médio porte (Pequeno I, II ou Médio); (2) IIC acima do percentil 50 do pool nacional; (3) diversidade regional entre os quatro municípios escolhidos.

O objetivo é exclusivamente ilustrativo — mostrar que o índice captura perfis qualitativamente diferentes. Não se trata de classificar todos os municípios do Brasil nem de criar uma tipologia nacional.

## Alternativas consideradas

- **Análise de quadrantes IPI × IG (cogitada, descartada)**: divide municípios em quatro cenários cruzando um Índice de Potencial de Injustiça (média de IP, IV, IE) com a capacidade de gestão (IG), usando a mediana nacional como corte. Metodologicamente válida como ferramenta exploratória, mas inadequada para o objetivo de selecionar casos: exige justificar um novo índice composto (IPI) e um threshold que classifica todos os municípios do Brasil quando o objetivo é apenas encontrar 4 exemplos representativos. Dimensões com distribuição muito diferente (IE tem média 0.13 vs ~0.31 das demais) também enfraquecem a média simples como IPI.

- **Análise de clusters k=4 (escolhida)**: usa diretamente as quatro dimensões no espaço original, sem criar compostos intermediários. A objeção de "threshold arbitrário" não se aplica aqui: o objetivo não é classificar corretamente cada município, mas encontrar perfis suficientemente distintos para selecionar casos ilustrativos. Municípios selecionados do centro de cada cluster são claramente típicos do perfil, o que é tudo que a análise exige.

- **Seleção por bioma**: garantir que os 4 municípios pertencessem a biomas diferentes. Descartada porque adiciona um critério de seleção que não está diretamente ligado ao perfil do índice e introduz complexidade sem ganho analítico. A diversidade regional entre os clusters já captura implicitamente diversidade de biomas (Norte → Amazônia, Nordeste → Caatinga/Cerrado, Sul/Sudeste → Mata Atlântica).

- **Seleção manual sem método formal**: escolher municípios com base em conhecimento prévio ou conveniência. Descartada pela falta de transparência e reprodutibilidade.

## Consequências

- Positivas: método simples, transparente e reproduzível; os quatro perfis emergem dos dados sem pressuposto teórico sobre quais dimensões separam os grupos; argumento no artigo é direto — "aplicamos k-means nas quatro dimensões e selecionamos um município representativo de cada cluster".
- Negativas / trade-offs: k-means pressupõe distâncias euclidianas e é sensível à escala — as dimensões precisam estar comparáveis, o que é garantido pela normalização min-max (ADR-0012). A escolha de k=4 é orientada pelo objetivo do estudo (4 casos), não por critério estatístico; isso deve ser declarado explicitamente no artigo.
- Confiança: Alta — o método está alinhado com o objetivo declarado e a limitação (k definido a priori) é justificável e comum em estudos com número de casos fixado por design.

## Referências

- ADR-0007 (foco intramunicipal), ADR-0012 (normalização min-max), ADR-0015 (quintis por porte).
- `explore/analysis/cluster_municipios.py` — script de agrupamento e geração de candidatos por cluster, porte e região.
- `explore/analysis/quadrante_municipios.py` — análise de quadrantes mantida como ferramenta exploratória complementar, não como método de seleção.
