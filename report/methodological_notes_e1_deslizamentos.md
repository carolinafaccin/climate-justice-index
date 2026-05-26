# Nota Metodológica — E1: Suscetibilidade a Deslizamentos de Terra

## Decisão: NASA LHASA em vez de Slope Puro

**Data:** 2026-05-14\
**Script oficial:** `h3_suscetibilidade_deslizamentos_lhasa_gee.js`\
**Script descontinuado:** `h3_suscetibilidade_deslizamentos_slope_gee_v1.js`

______________________________________________________________________

### Método descontinuado — Slope puro (CPRM/IPT)

O primeiro método implementado classificava a declividade do terreno em % slope
(a partir do Copernicus GLO-30 DEM) seguindo os limiares adotados no mapeamento
de áreas de risco do Vale do Taquari (CPRM/IPT):

| Classe | Faixa (% slope) |
|--------|----------------|
| Alta | 35 – 60 % |
| Média | 25 – 35 % e 60 – 75 % |
| Baixa | 10 – 25 % e > 75 % |
| Sem | 0 – 10 % |

O indicador e1 seria a fração da área do hexágono nas classes Alta ou Média
(`alta_media`).

### Por que foi descartado

Uma validação cruzada contra o NASA LHASA (ver `e1_validacao_lhasa.py` e
`h3_suscetibilidade_deslizamentos_slope_gee_v1.js` Step 3) revelou **correlação
negativa** entre os dois métodos em escala nacional:

| Par comparado | Pearson r | Spearman r |
|---|---|---|
| slope_alta_media × lhasa_high_frac | -0.194 | -0.251 |
| slope_alta_media × lhasa_mean | -0.308 | -0.300 |

A tabela de slope médio por classe LHASA confirmou a inversão:

| Classe LHASA | Slope médio |
|------------------|-------------|
| Very Low (1) | **0.250** |
| Low (2) | 0.095 |
| Moderate (3) | 0.039 |
| High (4) | 0.024 |
| Very High (5) | **0.018** |

**Interpretação:** No Brasil, slopes muito íngremes (> 35 %) correspondem
frequentemente a afloramentos rochosos consolidados — terrenos com pouco depósito
instável e, portanto, baixo risco real de deslizamento. Já as áreas de maior
risco real (Amazônia com desmatamento, encostas tropicais úmidas com solos
profundos) apresentam slopes moderados. O método CPRM/IPT foi calibrado para
uma região específica do Sul do Brasil e não generaliza para o território nacional.

______________________________________________________________________

### Método oficial — NASA LHASA

**Fonte:** Stanley, T. A. & Kirschbaum, D. B. (2017). A heuristic approach to
global landslide susceptibility mapping. *Natural Hazards*, 87(1), 145–164.

O LHASA é um modelo **multicritério** (fuzzy overlay) que combina:

- Slope (declividade)
- Geology (litologia — estabilidade das formações)
- Forest loss (perda de floresta — proxy de instabilidade de solo)
- Road density (densidade de estradas — cortes e aterros)
- Fault distance (distância a falhas geológicas)

Calibrado contra inventário global de deslizamentos, aplicável uniformemente
para todo o Brasil.

**Valores:** 1 = Very Low, 2 = Low, 3 = Moderate, 4 = High, 5 = Very High\
**Resolução:** ~1 km (30 arc-seconds)\
**Asset GEE:** `projects/ee2-linafaccin/assets/nasa_lhasa_susceptibility`

**Métrica do indicador e1:** `lhasa_high_frac` — fração da área do hexágono
com LHASA ≥ 4 (classes High ou Very High).

______________________________________________________________________

### Scripts de referência

| Arquivo | Descrição |
|---|---|
| `h3_suscetibilidade_deslizamentos_lhasa_gee.js` | **Script oficial** — extrai LHASA por hexágono |
| `h3_suscetibilidade_deslizamentos_slope_gee_v1.js` | Legado — slope puro (descontinuado) |
| `etl/exposure/e1_deslizamentos_lhasa.py` | **ETL oficial** |
| `etl/exposure/e1_deslizamentos_slope.py` | Legado — slope puro (descontinuado) |
| `etl/exposure/e1_validacao_lhasa.py` | Validação cruzada slope × LHASA |
