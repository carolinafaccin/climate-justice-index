# ADR-0030: Não incluir seca meteorológica (SPI/CHIRPS) como indicador da dimensão IE

## Status
Accepted — 2026-05-19. Implementação preservada em `etl/discarded/` (script GEE e ETL Python) por se tratar de metodologia bem-desenvolvida que pode ser revisitada.

## Contexto
Seca foi inicialmente proposta como o sexto indicador da dimensão IE (`e6`). A implementação chegou a estágio avançado: nota conceitual completa documentando os quatro tipos de seca (meteorológica, agrícola, hidrológica, socioeconômica), justificativa da escolha de SPI-12 sobre CHIRPS, calibração da climatologia 1991–2020, métrica final (`spi_severe_freq` — fração de meses com SPI ≤ −1.5 entre 2015–2024), e plano de validação contra Monitor de Secas (ANA/FUNCEME), Atlas Brasileiro de Desastres Naturais (CEPED/UFSC) e S2iD/MIDR.

A escolha do SPI-12 (janela anual) foi deliberada: janelas curtas (SPI-1 a SPI-3) capturam seca agrícola de curto prazo; SPI-12 é a escala em que reservatórios secam e o abastecimento urbano é comprometido — escala validada pela crise do Cantareira 2013–2015 (Coelho et al., 2016) e pelo Nordeste 2012–2017 (Marengo et al., 2018). CHIRPS foi escolhido sobre ERA5 e TRMM por sua resolução de 0,05° (~5,5 km), série desde 1981 e calibração com estações INMET — o mesmo produto usado pelo Monitor de Secas da ANA/FUNCEME, permitindo validação cruzada direta.

Na primeira rodada de validação (SP), o indicador foi questionado: a seca meteorológica tem **abrangência regional/estadual** e não diferencia internamente os municípios na grade H3 res 9; além disso, especialistas apontaram que a **população percebe seca como abstrata**, dificultando o uso comunicacional do indicador.

## Decisão
**Não incluir seca como indicador da dimensão IE.** O código é preservado em `etl/discarded/`: `etl/discarded/h3_e6_seca_spi_chirps_gee_v1.js` (script GEE) e `etl/discarded/e6_seca_spi_chirps.py` (ETL Python).

Categoria de exclusão (ADR-0008): **(4) insuficiência de granularidade** — o sinal de seca meteorológica via CHIRPS (~5,5 km) varia pouco dentro do município típico brasileiro, comprometendo a premissa de diferenciação intramunicipal do IIC. Categoria secundária: relevância comunicacional baixa para gestão municipal (não está na taxonomia formal do ADR-0008 mas é razão concreta do descarte).

## Alternativas consideradas
- **Incluir e6 mesmo com baixa diferenciação intramunicipal**: oferece sinal regional importante (Cantareira 2014–15, Nordeste 2012–17), mas em escala não acionável pelo gestor municipal. A escala adequada para o sinal de seca é o município ou a microrregião, não o hexágono — encaixa-se melhor em IG (capacidade municipal de resposta) ou em um futuro índice complementar.
- **Substituir SPI por outro indicador de seca** (NDVI agrícola, vazão fluvial): NDVI confunde-se com mudança de uso do solo (descartado também — ADR-0031); vazão fluvial não tem cobertura nacional uniforme; PDSI TerraClimate a 4 km com calibração genérica não resolve o problema de granularidade.
- **Não incluir e preservar o material para revisão futura (escolhido)**: o trabalho técnico do indicador foi feito e tem valor metodológico (referência potencial para outros índices, evolução futura do IIC); descarte da inclusão é decisão de escopo, não de qualidade.

## Consequências
- Positivas: dimensão IE fica coerente com a premissa intramunicipal; gestor municipal recebe indicadores comunicáveis (calor, queimadas, inundações, deslizamentos, mar) sem o ruído de um indicador regional; índice mais simples e defensável; e4 (calor) e e5 (queimadas) capturam indiretamente parte do sinal de seca (via amplificação por evapotranspiração e ignição).
- Negativas / trade-offs: o IIC subestima impacto de crises hídricas em regiões fortemente afetadas (Cantareira, Nordeste); revisores podem questionar a ausência de seca — o ADR documenta a justificativa; uma versão futura do IIC com componente municipal/regional pode reincorporar o indicador.
- Confiança: Alta — descarte motivado por incompatibilidade de escala (categoria 4 do ADR-0008), não por crítica técnica ao SPI/CHIRPS. Material preservado permite reativação se a granularidade do sinal melhorar ou se o escopo mudar.

## Referências
- ADR-0007 (foco intramunicipal), ADR-0008 (critérios), ADR-0028 (descartes IE gerais).
- [etl/discarded/h3_e6_seca_spi_chirps_gee_v1.js](../etl/discarded/h3_e6_seca_spi_chirps_gee_v1.js) — script GEE.
- [etl/discarded/e6_seca_spi_chirps.py](../etl/discarded/e6_seca_spi_chirps.py) — ETL Python.
- Feedback da primeira rodada de validação (SP), seção sobre seca.
