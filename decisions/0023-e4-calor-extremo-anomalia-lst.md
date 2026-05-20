# ADR-0023: Calcular e4 (Calor extremo) como anomalia de LST entre 2015–2024 e 1985–2010

## Status
Accepted — 2026-05-19

## Contexto
Calor extremo é uma das ameaças climáticas com maior impacto direto sobre saúde, mortalidade e produtividade — particularmente em grupos prioritários (idosos, crianças, gestantes, populações em moradias precárias). Na primeira rodada de validação (SP), três pontos emergiram:

1. **Pedro (MS)** sugeriu não usar Landsat como fonte de dados e considerar outro período do ano para captar dias de temperaturas extremas.
2. **Paulo (MIR)** apontou que calor afeta áreas urbanas e rurais de formas distintas — risco de viés se a metodologia privilegiar uma escala.
3. A ONU-Habitat sugere apresentar a mudança como **anomalia** (diferença em relação à média histórica), capturando onde o calor foge do padrão local — abordagem que isola o sinal de mudança climática do clima naturalmente quente da região.

A pergunta central: usar temperatura absoluta (cidades historicamente quentes ficariam sempre no topo do indicador) ou anomalia (foco no afastamento do padrão histórico local)?

## Decisão
Calcular e4 como **anomalia positiva de temperatura de superfície terrestre (LST)** entre dois períodos:

- **Período recente:** 2015–2024 (10 anos).
- **Referência histórica:** 1985–2010 (26 anos).
- **Anomalia:** `LST_recente − LST_histórico`. Anomalias negativas (resfriamento ou estabilidade térmica) recebem pontuação zero.

Fonte: Landsat 5, 7, 8 e 9 (NASA/USGS Collection 2 Level-2) via GEE. Remoção de nuvens via banda QA_PIXEL (bits 3 e 4). Bandas usadas: ST_B6 (Landsat 5 e 7) e ST_B10 (Landsat 8 e 9), convertidas para graus Celsius. Cálculo restrito a hexágonos habitados; desabitados recebem zero.

Normalização min-max com winsorização p1-p99 (ADR-0012). **Não há ponderação pela quantidade de domicílios** — a escala populacional é capturada pelas dimensões IP e IV, onde conceitualmente pertence (manter aqui geraria dupla contagem).

## Alternativas consideradas
- **Temperatura absoluta média**: simples e intuitivo, mas favorece cidades naturalmente quentes (Manaus sempre vence) — não captura o sinal de mudança climática nem permite comparação justa entre regiões climáticas distintas.
- **Anomalia em desvios-padrão (z-score local)**: identifica os hexágonos mais "anormais" relativos à variabilidade local, mas perde a magnitude absoluta da mudança (um aumento de 0.5°C em zona árida fria pode parecer grande, mas é fisiologicamente menos perigoso que 2°C em zona quente úmida).
- **Decil superior (top 10% de hexágonos mais quentes dentro de cada município)**: identifica ilhas de calor relativas, mas não diferencia municípios entre si — todos teriam um top 10% por construção.
- **Não usar Landsat (sugestão Pedro/MS)**: o sensor LST do Landsat tem resolução ~100 m (reamostrada a 30 m), adequada para análise intramunicipal. Sensor MODIS LST tem resolução ~1 km, grosseira demais. Sentinel-3 SLSTR oferece dados de LST a ~1 km. A combinação Landsat 5–9 cobre 40 anos de série, viabilizando a comparação histórica que outros sensores não suportam.
- **Outro período do ano** (verão estendido): considerado, mas a série multianual já captura efeitos sazonais — calcular anomalia anual mantém o sinal climático sem precisar definir um "verão" que varia por região.
- **Anomalia 2015–2024 vs 1985–2010 (escolhida)**: equilibra magnitude absoluta da mudança com isolamento do sinal climático local; usa série Landsat de 40 anos como única fonte aberta com resolução adequada nacional; alinhada à recomendação da ONU-Habitat.

## Consequências
- Positivas: indicador foca em mudança climática, não em clima naturalmente quente; permite comparação justa entre regiões climáticas distintas (Amazônia, Cerrado, Sul subtropical); usa fonte aberta e estável (Landsat); 40 anos de série fornecem referência histórica robusta.
- Negativas / trade-offs: hexágonos em municípios com pouca cobertura Landsat válida (excesso de nuvens) podem ter anomalia menos confiável; anomalias muito altas em desertos podem refletir aumento real de temperatura sem efeito populacional significativo (mas filtro de hexágonos habitados mitiga isso); revisores que esperam temperatura absoluta podem questionar a escolha — exige explicação no artigo.
- Confiança: Alta — fundamentação na recomendação ONU-Habitat e nas duas décadas de literatura de anomalia térmica.

## Referências
- ADR-0009 (grade H3), ADR-0012 (normalização).
- [etl/exposure/e4_calor.py](../etl/exposure/e4_calor.py) — ETL oficial.
- [report/methodological_notes.md](../report/methodological_notes.md) — seção e4.
- Feedback da primeira rodada de validação (SP), seção sobre calor extremo (Pedro/MS, Paulo/MIR) e recomendação ONU-Habitat.
