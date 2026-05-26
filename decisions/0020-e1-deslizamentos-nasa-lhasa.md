# ADR-0020: Usar NASA LHASA para o indicador e1 (Deslizamentos de terra)

## Status

Accepted — 2026-05-19. Substitui implementação anterior baseada em slope puro (CPRM/IPT), preservada em `etl/discarded/e1_deslizamentos_slope.py` e `etl/discarded/h3_e1_deslizamentos_slope_gee.js`. A implementação anterior nunca foi formalizada em ADR — supersedure registrada aqui.

## Contexto

Deslizamentos de terra são uma das principais ameaças climáticas no Brasil (encostas tropicais, eventos extremos de chuva). Precisamos de um indicador de suscetibilidade que: (i) tenha cobertura nacional uniforme; (ii) capture o risco real, não apenas a condição geomorfológica; (iii) seja reproduzível a partir de bases abertas.

A primeira implementação seguiu os limiares CPRM/IPT (Vale do Taquari/RS), classificando declividade em % slope sobre o Copernicus GLO-30 DEM em faixas Alta (35–60%), Média (25–35% e 60–75%), Baixa e Sem. Uma validação cruzada nacional contra o NASA LHASA revelou **correlação negativa** entre os dois métodos:

| Par comparado | Pearson r | Spearman r |
|---|---|---|
| slope_alta_media × lhasa_high_frac | −0.194 | −0.251 |
| slope_alta_media × lhasa_mean | −0.308 | −0.300 |

A tabela de slope médio por classe LHASA confirmou a inversão (Very Low: 0.250; Very High: 0.018). Interpretação: no Brasil, slopes muito íngremes correspondem frequentemente a afloramentos rochosos consolidados (baixo risco real), enquanto áreas de risco efetivo (encostas tropicais úmidas com solos profundos, áreas com perda de floresta recente) apresentam slopes moderados. O método CPRM/IPT foi calibrado para uma região específica do Sul e não generaliza nacionalmente.

## Decisão

Usar o **NASA LHASA — Global Landslide Susceptibility Map** como base do indicador e1. LHASA é um modelo multicritério (fuzzy overlay) calibrado contra inventário global de deslizamentos, combinando cinco fatores: slope, litologia, perda de floresta, densidade de estradas e distância a falhas geológicas. Resolução ~1 km (30 arc-seconds). Classes: 1 (Very Low) a 5 (Very High); valor 0 indica NoData/oceano (mascarado).

**Métrica do indicador e1:** `lhasa_high_frac` — fração da área do hexágono com LHASA ≥ 4 (classes High ou Very High), restrita a hexágonos habitados. Normalização min-max **sem winsorização** (ADR-0016) — suscetibilidade é geograficamente concentrada (~8% dos hexágonos).

## Alternativas consideradas

- **Slope puro com limiares CPRM/IPT (descartado)**: simples e usa DEM Copernicus aberto, mas calibração regional não generaliza; correlação negativa nacional contra LHASA é evidência forte de inadequação. Preservado em `etl/discarded/` para referência histórica e auditoria.
- **MapBiomas Risco Climático — Deslizamentos**: produto nacional, mas com máscara restrita a perímetros urbanos do Open Buildings — incompatível com a premissa de cobertura nacional (incluindo áreas rurais e periurbanas).
- **Modelo próprio multicritério**: replicar LHASA localmente exigiria curar fontes equivalentes (litologia, perda de floresta, densidade de estradas) e calibrar contra inventário brasileiro — investimento elevado sem ganho claro sobre LHASA.
- **NASA LHASA (escolhido)**: modelo já calibrado globalmente; aplicável uniformemente ao Brasil; disponível como asset GEE; multicritério com fundamentos físicos defensáveis.

## Consequências

- Positivas: cobertura nacional uniforme; metodologia multicritério captura risco real, não apenas geomorfologia; reproduzível a partir de asset GEE público; consistente com a literatura internacional.
- Negativas / trade-offs: resolução ~1 km é mais grosseira que os 30 m do DEM Copernicus que o método anterior usava — perde detalhe topográfico fino; depende de manutenção do asset pela NASA; revisores brasileiros familiarizados com o método CPRM/IPT podem questionar a escolha (o ADR documenta a validação empírica que fundamenta o descarte).
- Confiança: Alta — substituição motivada por evidência empírica direta (correlação negativa nacional), não por preferência teórica.

## Referências

- ADR-0009 (grade H3), ADR-0012 (normalização), ADR-0016 (e1 sem winsorização).
- [report/methodological_notes_e1_deslizamentos.md](../report/methodological_notes_e1_deslizamentos.md) — registro detalhado da decisão e da validação.
- [etl/exposure/e1_deslizamentos_lhasa.py](../etl/exposure/e1_deslizamentos_lhasa.py) — ETL oficial.
- [etl/discarded/e1_validacao_slope_lhasa.py](../etl/discarded/e1_validacao_slope_lhasa.py) — script da validação cruzada.
- [etl/discarded/e1_deslizamentos_slope.py](../etl/discarded/e1_deslizamentos_slope.py), [etl/discarded/h3_e1_deslizamentos_slope_gee.js](../etl/discarded/h3_e1_deslizamentos_slope_gee.js) — implementação descontinuada.
- Stanley, T. A. & Kirschbaum, D. B. (2017). A heuristic approach to global landslide susceptibility mapping. *Natural Hazards*, 87(1), 145–164.
