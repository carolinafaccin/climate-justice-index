# Análise de Clusters

## Distintos perfis do índice

Além do ranking, agrupamos os municípios em **perfis** que compartilham um
padrão semelhante entre as quatro dimensões do índice (IP, IV, IE e IG). A
análise utiliza *k-means* sobre as dimensões padronizadas, resultando em quatro
perfis distintos de injustiça climática.

A tabela abaixo resume cada perfil — quantos municípios o compõem, o IIC médio
do grupo e a dimensão que mais o caracteriza. O mapa de calor mostra, para cada
perfil, o quanto cada dimensão se afasta da média geral (em desvios padrão):
tons quentes indicam desempenho **pior** e tons frios, **melhor**. Como a
dimensão de gestão (IG) entra invertida no índice, valores altos de IG também
representam maior contribuição à injustiça.

<!-- A tabela, o mapa de calor e o mapa do Brasil abaixo são gerados
     automaticamente pelo generate_report.py a partir dos resultados de
     explore/analysis/cluster_analysis.py. Edite o texto à vontade, mas
     mantenha os marcadores {{cluster_table}}, {{cluster_heatmap}} e
     {{cluster_map}} para que o conteúdo gerado seja inserido. -->

{{cluster_table}}

{{cluster_heatmap}}

{{cluster_map}}

## Estudos de caso

Para ilustrar cada perfil, selecionamos municípios representativos. O texto
abaixo é editável — descreva o contexto de cada município e por que ele
exemplifica o perfil.

### Alta Vulnerabilidade Social — Araioses (MA)

*39 mil habitantes · Rural.*

### Alta Exposição a Riscos Climáticos — Altamira (PA)

*126 mil habitantes · Urbano.*

### Alta Injustiça Climática — Fátima do Sul (MS)

*20 mil habitantes · Urbano.*

### Baixa Injustiça Climática — Jaboticatubas (MG)

*20 mil habitantes · Rural.*

## Critérios de seleção dos estudos de caso

- **Região/bioma** — cobertura de diferentes contextos regionais e biomas.
- **Urbano e rural** — equilíbrio entre municípios predominantemente urbanos e rurais.
- **Porte médio e pequeno** — foco em municípios de menor porte populacional.
