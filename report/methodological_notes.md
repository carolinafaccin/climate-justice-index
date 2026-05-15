# Dimensão de Grupos Prioritários (IP)

## Fonte e coleta de dados
Os cinco indicadores desta dimensão utilizam os agregados por setor censitário do Censo Demográfico 2022 (IBGE), disponíveis no portal do IBGE como tabelas da série `Tabela 0` (t0), por Unidade da Federação. Os valores são interpolados assimetricamente para a grade hexagonal H3: cada setor censitário é associado aos hexágonos que o intersectam, e seus valores ponderados pelo fator `peso_dom` — a razão entre o número de domicílios do CNEFE no hexágono e o total de domicílios do setor. Todos os indicadores são normalizados por min-max com winsorização nos percentis 1%–99%.

## Variáveis

### A1 — Mulheres negras chefes de família:
Percentual de responsáveis pelo domicílio do sexo feminino com cor da pele preta ou parda. Numerador: v01340 + v01344. Denominador: v01042.

### P2 — População negra:
Percentual de pessoas pretas e pardas residentes. Numerador: v01318 + v01320. Denominador: v01006.

### P3 — Indígenas e quilombolas:
Percentual de pessoas indígenas ou quilombolas. Numerador: v01690 + v03196. Denominador: v01006.

### P4 — Idosos de baixa renda:
Percentual de pessoas com 60 anos ou mais em domicílios de baixa renda. Numerador: (v01040 + v01041) × `peso_renda`. Denominador: v01006. O fator `peso_renda = min(1, 1.212 / renda_média)` aproxima a probabilidade de o setor concentrar domicílios abaixo de 1 salário mínimo, ponderando o grupo etário antes da agregação hexagonal.

### P5 — Crianças de baixa renda:
Percentual de crianças de 0 a 9 anos em domicílios de baixa renda. Numerador: (v01031 + v01032) × `peso_renda`. Denominador: v01006. Mesmo `peso_renda` de p4.

# Dimensão de Vulnerabilidade

## Fonte e coleta de dados
Censo Demográfico 2022 (IBGE), agregados por setor censitário, com interpolação assimétrica para hexágonos H3 via `peso_dom`. O indicador V4 utiliza fonte distinta (CNES), descrita em seguida.

## Variáveis

### V1 — Renda:
Inverso da renda média mensal do responsável pelo domicílio, com teto de 2 salários mínimos (R$ 2.424, referência 2022). A renda média ponderada do hexágono é calculada como (v06004 × v06001) × peso assimétrico / soma ponderada de v06001. Hexágonos com renda acima de R$ 2.424 recebem escore zero; abaixo, o gradiente é normalizado por min-max (p1%–p99%) e invertido (1 − valor normalizado), pois renda maior indica menor vulnerabilidade.

### V2 — Moradia precária:
Percentual de domicílios com ao menos uma condição de inadequação habitacional (domicílios improvisados, cortiços, habitações sem paredes, banheiro coletivo, sanitário improvisado ou ausência de banheiro). Variáveis: v00002, v00050–v00052, v00236–v00238 (numerador, lógica OR); v00001 (denominador). O numerador é limitado ao denominador antes da divisão para evitar dupla contagem de domicílios com múltiplas condições simultâneas. Normalização min-max (p1%–p99%).

### V3 — Analfabetismo:
Percentual de pessoas acima de 15 anos que não sabem ler e escrever. Numerador: v00853 + v00855 + v00857. Denominador: v01006. Normalização min-max (p1%–p99%).

### V4 — Inacessibilidade à saúde *(fonte: CNES 2026)*:
Inacessibilidade gravitacional a estabelecimentos de saúde, medida pelo inverso da força de atração dos três estabelecimentos mais próximos de cada hexágono, ponderada pela capacidade de serviços e pela distância euclidiana.

Os dados foram obtidos do Cadastro Nacional de Estabelecimentos de Saúde (CNES), Ministério da Saúde, com extração em janeiro de 2026. O `capacity_score` de cada estabelecimento é a soma de seis categorias de serviço (centro cirúrgico, obstétrico, neonatal, atendimento hospitalar, apoio e ambulatorial), acrescida de 1 para garantir pontuação mínima não nula a todos os estabelecimentos.

Para cada hexágono, os três estabelecimentos mais próximos são identificados via árvore KD (cKDTree), com distâncias em metros. A acessibilidade bruta é calculada como `v4_abs = Σ [ capacity_score_j / (distância_j + 100) ]`, onde o buffer de 100 m evita divisão por zero. A pontuação bruta foi normalizada com winsorização mais estreita (p3%–p97%) para suprimir o efeito de grandes clusters hospitalares. O valor final é invertido (1 − normalizado) para que maior inacessibilidade corresponda a maior vulnerabilidade.

### v5 — Infraestrutura:
Percentual de domicílios sem coleta de esgoto, sem abastecimento de água e/ou sem coleta de lixo. Variáveis: v00311–v00316 (esgoto), v00112–v00118 (água), v00399–v00402 (lixo), em lógica OR; denominador v00001. O numerador é limitado ao denominador para evitar dupla contagem. Normalização min-max (p1%–p99%).

# Dimensão de Exposição (e1–e5)

### E1 — Deslizamentos de terra *(fonte: NASA LHASA / Stanley & Kirschbaum 2017, via Google Earth Engine)*:
Fração da área de cada hexágono em zona de Alta ou Muito Alta suscetibilidade a deslizamentos de terra, segundo o NASA Global Landslide Susceptibility Map (LHASA).

O LHASA é um modelo multicritério (fuzzy overlay) que combina cinco variáveis: declividade (slope), litologia, perda de cobertura florestal, densidade de estradas e distância a falhas geológicas. O raster classifica cada pixel (~1 km de resolução) em cinco níveis: 1 (Very Low), 2 (Low), 3 (Moderate), 4 (High) e 5 (Very High). Optou-se por esse modelo em substituição ao método baseado em slope puro porque comparações em escala nacional revelaram correlação negativa entre declividade isolada e ocorrência efetiva de deslizamentos: encostas muito íngremes no Brasil correspondem com frequência a afloramentos rochosos consolidados (baixo risco), enquanto áreas de alto risco real (Amazônia desmatada, encostas tropicais úmidas) têm declividades moderadas.

No GEE, o raster foi processado com `Reducer.mean()` aplicado a um buffer de 174 m (circunraio do H3 res9), gerando duas métricas por hexágono: `lhasa_mean` (valor médio 1–5) e `lhasa_high_frac` (fração da área com LHASA ≥ 4). O indicador E1 utiliza apenas `lhasa_high_frac`. Normalização min-max sem winsorização — a suscetibilidade é geograficamente concentrada e a winsorização nos percentis 1%–99% colapsaria a cauda para zero na maior parte do território.

### E2 — Inundações *(fonte: HAND global + JRC Global River Flood Hazard v2.1, via Google Earth Engine)*:
Score médio de suscetibilidade a inundações por hexágono, combinando o modelo de terreno HAND (Height Above Nearest Drainage) com a máscara de perigo do JRC Global River Flood Hazard Maps.

A metodologia é adaptada da Coleção 1 do MapBiomas Risco Climático (2024), removida a máscara de áreas urbanas que limitava o produto original a perímetros do Open Buildings — assim, o indicador alcança cobertura nacional independente de uso da terra. No GEE, cada pixel recebe um score classificado pelo HAND:

| HAND | Score | Interpretação |
| --- | --- | --- |
| 0–2 m | 1.00 | Muito alta suscetibilidade |
| 2–4 m | 0.66 | Alta |
| 4–6 m | 0.33 | Média |
| > 6 m | 0.00 | Sem suscetibilidade |

O score é então multiplicado pela máscara binária `JRC RP100_depth > 0` (perigo de inundação modelado para período de retorno de 100 anos), de forma que apenas pixels simultaneamente em planície de inundação (HAND baixo) e em zona de perigo modelado (JRC) recebem score positivo. Pixels mascarados (calha de rios, oceano) contribuem com zero. O score por hexágono é a média dos pixels via `Reducer.mean()` com buffer de 174 m. Normalização min-max sem winsorização — inundações são fenômenos geograficamente concentrados em vales e margens fluviais, e a winsorização colapsaria a cauda para zero na maior parte do território.

### E3 — Elevação do nível do mar *(fonte: Copernicus GLO-30 / Google Earth Engine)*: 
Quantidade de domicílios em hexágonos costeiros com elevação ≤ 1 m e distância ao oceano ≤ 10 km, ponderada pela fração de área em risco.

No GEE, pixels oceânicos (NoData no DEM original) foram preenchidos com −10; a distância euclidiana de cada pixel terrestre ao oceano foi calculada com `fastDistanceTransform`. Um pixel foi classificado em risco se elevação ≤ 1 m AND distância ≤ 10 km — limiar que inclui estuários e planícies costeiras legítimas, excluindo baixadas fluviais sem conexão hidrológica com o mar. A fração de área em risco de cada hexágono foi obtida por `Reducer.mean()` com buffer de 174 m (circunraio do H3 res9). O indicador final é o produto `qtd_dom × risco_slr`. Normalização min-max sem winsorização. Hexágonos do interior do país recebem valor ausente (não aplicável) e são excluídos do cálculo do IE — o indicador E3 só entra na média do IE para municípios costeiros.

### E4 — Calor extremo *(fonte: Landsat 5, 7, 8 e 9 / NASA/USGS / Google Earth Engine)*:
Anomalia positiva de temperatura de superfície terrestre (LST) em hexágonos habitados, calculada como a diferença entre a média do período recente (2015–2024) e a média histórica de referência (1985–2010). Hexágonos desabitados recebem pontuação zero.

Os dados foram extraídos no GEE das coleções NASA/USGS Collection 2 Level-2, com remoção de nuvens via banda QA_PIXEL (bits 3 e 4). A temperatura foi obtida das bandas ST_B6 (Landsat 5 e 7) e ST_B10 (Landsat 8 e 9), convertidas para graus Celsius. As médias LST dos dois períodos foram calculadas e a anomalia obtida pela diferença. Anomalias negativas (resfriamento ou estabilidade térmica) recebem pontuação zero. A anomalia positiva é diretamente normalizada por min-max com winsorização (p1%–p99%), sem ponderação pela quantidade de domicílios — a escala populacional é capturada nas dimensões de grupos prioritários e vulnerabilidade, onde ela conceitualmente pertence. A abordagem de anomalia, em vez de temperatura absoluta, permite comparar municípios de climas naturalmente distintos em escala nacional.

### E5 — Focos de queimadas *(fonte: INPE, 2016–2025)*:
Exposição crônica a focos de queimadas em hexágonos habitados, medida pela fração de anos (2016–2025) com ao menos um foco de calor na vizinhança (~600 m) do hexágono em pelo menos 2 dos 10 anos analisados.

Para cada ano, as coordenadas dos focos do INPE (predominantemente sensor VIIRS a 375 m de resolução, satélites NOAA-20, NOAA-21 e Suomi NPP) foram convertidas em hexágonos H3 (resolução 9) e expandidas para um raio de aproximadamente 600 m via operação `grid_disk(k=2)` — buffer calibrado para a resolução do sensor VIIRS, ao contrário do buffer de 1 km que seria adequado apenas para o sensor MODIS (1 km). Hexágonos com foco em apenas 1 ano dos 10 recebem pontuação zero, pois eventos isolados não caracterizam risco climático crônico; apenas exposição recorrente (fração ≥ 0,2, equivalente a 2 ou mais anos) é computada. Normalização min-max com winsorização (p1%–p99%), sem ponderação pela quantidade de domicílios — a escala populacional é capturada nas dimensões de grupos prioritários e vulnerabilidade.

# 4 Dimensão de Capacidade de Gestão Municipal

### G1 — Investimento ambiental *(fonte: Siconfi/FINBRA, 2015–2024)*:
Média anual das despesas municipais liquidadas per capita em gestão ambiental (função orçamentária 18), entre 2015 e 2024.

Os dados foram obtidos dos arquivos anuais do Sistema de Informações Contábeis e Fiscais do Setor Público Brasileiro (Siconfi/STN). Para cada ano, as despesas liquidadas na função 18 foram divididas pela população municipal registrada no mesmo arquivo. O indicador é a média dos valores per capita do período, tornando o cálculo robusto a anos sem declaração. Antes da normalização, foi aplicada transformação logarítmica (log1p) para comprimir a assimetria à direita da distribuição — a maioria dos municípios investe pouco e poucos investem valores excepcionais. Normalização min-max (p1%–p99%). Invertido no nível da dimensão (IG = 1 − média dos g).

### G2 a G6 — Indicadores binários de capacidade de gestão *(fonte: MUNIC/IBGE)*:
Os cinco indicadores abaixo utilizam a Pesquisa de Informações Básicas Municipais (MUNIC), pesquisa censitária que levanta a existência de instrumentos, planos e sistemas em todos os municípios brasileiros. Respostas Sim/Não foram convertidas para 1/0 e propagadas diretamente como valor normalizado, sem etapa adicional de transformação. Para g4, o valor é 1 se ao menos um dos dois conselhos existir no município (lógica OR entre as variáveis).

| Código | Indicador | Variável MUNIC | Edição |
| --- | --- | --- | --- |
| g2 | Plano de Contingência | smap123 | 2023 |
| g3 | NUPDECs | mgrd213 | 2020 |
| g4 | Conselhos municipais (Meio Ambiente OR Cidade/Desenv. Urbano) | sdg353 / sdg351 | 2023 |
| g5 | Sistema de alerta de riscos | smap126 | 2023 |
| g6 | Mapeamento e zoneamento de áreas de risco | smap122 | 2023 |

### G7 — Cadastro de famílias em áreas de risco *(fonte: ICM/MIDR, 2026)*:
Existência de cadastro ou identificação de famílias em áreas de risco no município, segundo o Índice de Capacidade Municipal (ICM) do Ministério da Integração e do Desenvolvimento Regional, edição de 2026. Variável v7 (binária: 1 = possui; 0 = não possui). Para municípios presentes em mais de uma lista do ICM, manteve-se o valor máximo, garantindo que qualquer registro positivo seja preservado. Propagado diretamente sem normalização adicional.

### G8 — Políticas e programas de direitos humanos *(fonte: MUNIC 2023)*:
Quantidade de políticas e programas municipais ativos em direitos humanos, contada entre 21 iniciativas levantadas pela MUNIC 2023. Único indicador contínuo da dimensão de gestão — enquanto g2–g7 são binários, g8 varia de 0 a 21 conforme o número de iniciativas ativas. Variáveis: mdhu571–mdhu5716, mdhu58, mdhu61, mdhu64, mdhu67, mdhu69. Normalização min-max (p1%–p99%). Invertido no nível da dimensão.

---

# Dicionário de variáveis e fontes

Este apêndice registra os campos originais de cada base de dados utilizados no cálculo dos indicadores do IIC, garantindo a replicabilidade da metodologia.

## Variáveis do Censo Demográfico 2022 (IBGE) — Agregados por Setor Censitário

Os arquivos de agregados por setor censitário do Censo 2022 estão disponíveis no portal do IBGE. As variáveis utilizadas pertencem às tabelas da série `Tabela 0` (t0) disponibilizadas por Unidade da Federação.

| Código | Descrição | Indicador(es) |
| --- | --- | --- |
| v01006 | Total de pessoas residentes | p2, p3, p4, p5, v3 (denominador) |
| v01042 | Responsáveis pelo domicílio (total) | p1 (denominador) |
| v01318 | Pessoas pretas | p2 (numerador) |
| v01320 | Pessoas pardas | p2 (numerador) |
| v01340 | Responsáveis do sexo feminino — pretas | p1 (numerador) |
| v01344 | Responsáveis do sexo feminino — pardas | p1 (numerador) |
| v01690 | Pessoas indígenas | p3 (numerador) |
| v03196 | Pessoas quilombolas | p3 (numerador) |
| v01031 | Pessoas de 0 a 4 anos | p5 (numerador, ponderado por peso_renda) |
| v01032 | Pessoas de 5 a 9 anos | p5 (numerador, ponderado por peso_renda) |
| v01040 | Pessoas de 60 a 64 anos | p4 (numerador, ponderado por peso_renda) |
| v01041 | Pessoas de 65 anos ou mais | p4 (numerador, ponderado por peso_renda) |
| v06001 | Responsáveis com rendimento | v1 (denominador da renda média) |
| v06004 | Rendimento médio mensal dos responsáveis com rendimento | v1 (numerador — calculado como v06004 × v06001) |
| v00001 | Total de domicílios particulares permanentes | v2, v5 (denominador) |
| v00002 | Domicílios improvisados | v2 (numerador) |
| v00050 | Domicílios em cortiço/cômodo | v2 (numerador) |
| v00051 | Domicílios em habitação indígena sem paredes | v2 (numerador) |
| v00052 | Domicílios em maloca | v2 (numerador) |
| v00236 | Domicílios com banheiro coletivo | v2 (numerador) |
| v00237 | Domicílios com sanitário improvisado | v2 (numerador) |
| v00238 | Domicílios sem banheiro ou sanitário | v2 (numerador) |
| v00853 | Pessoas de 15 a 19 anos não alfabetizadas | v3 (numerador) |
| v00855 | Pessoas de 20 a 39 anos não alfabetizadas | v3 (numerador) |
| v00857 | Pessoas de 40 anos ou mais não alfabetizadas | v3 (numerador) |
| v00311–v00316 | Domicílios sem coleta de esgoto (categorias) | v5 (numerador) |
| v00112–v00118 | Domicílios sem abastecimento de água (categorias) | v5 (numerador) |
| v00399–v00402 | Domicílios sem coleta de lixo (categorias) | v5 (numerador) |

**Nota sobre variáveis derivadas:** As variáveis com sufixo `_ren` (e.g., `v01040_ren`) não são campos nativos do Censo — são colunas calculadas no ETL pela multiplicação das contagens originais pelo fator `peso_renda = min(1, 1.212 / v06004)`, usado para aproximar a proporção do grupo etário em domicílios de baixa renda.

## CNEFE 2022 — Cadastro Nacional de Endereços para Fins Estatísticos (IBGE)

| Campo | Descrição | Uso no IIC |
| --- | --- | --- |
| Latitude, Longitude | Coordenadas geográficas do endereço | Associação ao hexágono H3; construção da malha base e do peso_dom |
| Espécie | Tipo de endereço (1 = domicílio particular permanente; 2 = domicílio coletivo) | Filtro para contagem de domicílios por hexágono |
| cod_setor | Código do setor censitário | Associação com os agregados por setor |

## CNES — Cadastro Nacional de Estabelecimentos de Saúde (Ministério da Saúde)

Referência: dados extraídos em janeiro de 2026. Arquivo: `cnes_estabelecimentos.csv`.

| Campo | Descrição | Uso no IIC |
| --- | --- | --- |
| Latitude, Longitude | Coordenadas geográficas do estabelecimento | Construção do modelo gravitacional (v4) |
| Centro cirúrgico, obstétrico, neonatal | Indicadores binários de serviço | capacity_score |
| Atendimento hospitalar, apoio, ambulatorial | Indicadores binários de serviço | capacity_score |

O `capacity_score` de cada estabelecimento é a soma das seis categorias de serviço presentes, acrescida de 1 (mínimo = 1 para todo estabelecimento válido).

## NASA LHASA — Global Landslide Susceptibility Map (Google Earth Engine)

Utilizado para o indicador e1. Referência: Stanley, T. & Kirschbaum, D. B. (2017). *A heuristic approach to global landslide susceptibility mapping*. Natural Hazards 87(1): 145–164. Modelo multicritério (fuzzy overlay) que combina slope, litologia, perda de floresta, densidade de estradas e distância a falhas. Resolução nativa: ~1 km (30 arc-seconds). Valores: 1 (Very Low) a 5 (Very High); valor 0 indica NoData/oceano (mascarado no processamento).

## HAND — Height Above Nearest Drainage (Google Earth Engine)

Utilizado para o indicador e2. Modelo de terreno hidrologicamente condicionado que representa a altura vertical de cada pixel em relação ao canal de drenagem mais próximo. Asset: `users/gena/global-hand/hand-100` (Donchyts et al. 2016), derivado do SRTM com resolução ~30 m. Classificação aplicada: 0–2 m, 2–4 m, 4–6 m, > 6 m.

## JRC Global River Flood Hazard Maps v2.1 (Google Earth Engine)

Utilizado para o indicador e2 como máscara de perigo. Asset: `JRC/CEMS_GLOFAS/FloodHazard/v2_1`, mantido pelo Joint Research Centre da Comissão Europeia (CEMS-GLOFAS). Resolução: ~1 km. Banda utilizada: `RP100_depth` — profundidade de inundação em metros para período de retorno de 100 anos.

## INPE — Focos de queimadas (2016–2025)

Registros anuais de focos de calor detectados por satélite, disponíveis em queimadas.dgi.inpe.br. Campos utilizados: latitude e longitude do foco. Período: 2016 a 2025 (10 anos).

## Copernicus GLO-30 — Modelo Digital de Elevação (Google Earth Engine)

Utilizado para o indicador e3. Raster de risco calculado no GEE com critérios: elevação ≤ 1 m AND distância ao oceano ≤ 10 km. Resolução: 30 m.

## Landsat 5, 7, 8, 9 — Temperatura de Superfície (NASA/USGS, Google Earth Engine)

Utilizado para o indicador e4. Coleção NASA/USGS Collection 2 Level-2. Bandas: ST_B6 (Landsat 5 e 7) e ST_B10 (Landsat 8 e 9). Período histórico: 1985–2010. Período recente: 2015–2024.

## Siconfi / FINBRA — Sistema de Informações Contábeis e Fiscais (STN/IBGE)

| Campo | Descrição | Indicador |
| --- | --- | --- |
| cod.ibge | Código IBGE do município | Associação com a grade H3 |
| função | Código da função de despesa (18 = Gestão Ambiental) | Filtro para g1 |
| valor | Valor liquidado anual (R$) | g1 numerador |
| populacao | População municipal | g1 denominador (per capita) |

Período: 2015 a 2024 (arquivo anual por exercício).

## MUNIC — Pesquisa de Informações Básicas Municipais (IBGE)

| Variável | Questão | Indicador | Edição |
| --- | --- | --- | --- |
| smap123 | Plano de Gerenciamento de Contingência | g2 | 2023 |
| mgrd213 | Existência de NUPDECs | g3 | 2020 |
| sdg353 / sdg351 | Conselhos de Meio Ambiente / Cidade e Desenvolvimento Urbano | g4 | 2023 |
| smap126 | Sistema de alerta de riscos | g5 | 2023 |
| smap122 | Mapeamento e zoneamento de áreas de risco | g6 | 2023 |
| mdhu571–mdhu5716, mdhu58, mdhu61, mdhu64, mdhu67, mdhu69 | Políticas e programas de direitos humanos (21 itens) | g8 | 2023 |

## ICM — Índice de Capacidade Municipal (MIDR, 2026)

| Campo | Descrição | Indicador |
| --- | --- | --- |
| cod_mun | Código IBGE do município | Associação com a grade H3 |
| v7 | Existência de cadastro de famílias em áreas de risco (binário) | g7 |
