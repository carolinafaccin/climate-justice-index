# ADR-0028: Registrar variáveis consideradas e não-incluídas na dimensão IE (Exposição)

## Status

Accepted — 2026-05-19

## Contexto

A dimensão IE inclui cinco ameaças climáticas (e1 deslizamentos, e2 inundações, e3 elevação do mar, e4 calor extremo, e5 queimadas — ADRs 0020-0024). Outras ameaças foram propostas pelos especialistas nas duas rodadas de validação. Duas têm ADRs específicos por terem código preservado em `etl/discarded/`: ADR-0030 (seca SPI/CHIRPS) e ADR-0031 (NDVI áreas verdes — embora seja IV, está intimamente conectado a e4). Este ADR registra os demais descartes.

## Decisão

Registrar variáveis consideradas e descartadas para a dimensão IE.

| Variável sugerida | Categoria(s) ADR-0008 | Razão concreta |
|---|---|---|
| Desmatamento (INPE/PRODES) | (2) inadequação conceitual; (8) redundância | Sugerido por Luma (WWF). Desmatamento é ação humana, não evento climático. Se considerado, atuaria apenas como indicador de falha de gestão ambiental — já capturado por g1 (investimento em gestão ambiental). Redundância com IG. |
| Ventos extremos | (4) granularidade; (5) fonte | Sugerido por Raquel (MCidades). Costumam afetar municípios de forma homogênea ou em escalas macrorregionais. Sem fonte oficial nacional com diferenciação intramunicipal. |
| BATER — Base Territorial Estatística de Áreas de Risco (IBGE/Cemaden) | (3) cobertura territorial | Cruzamento de polígonos de risco geo-hidrológico com setores censitários. Não cobre todos os 5.570 municípios. O IIC usa metodologia alternativa equivalente: MapBiomas + CNEFE em e1 (deslizamentos) e HAND + JRC em e2 (inundações), ambas com cobertura nacional. |

Critérios usam a taxonomia do ADR-0008.

## Alternativas consideradas

- **Incluir desmatamento ainda assim, como proxy adicional de gestão ambiental**: redundante com g1; inflaria IE com um indicador que não é ameaça climática.
- **Incluir ventos extremos com fonte mesoescala** (ex: ERA5 reanalysis): viável tecnicamente, mas a resolução nativa (~25 km) não diferencia hexágonos H3 res 9 dentro de um município — perderia sentido na escala do IIC.
- **Usar BATER mesmo com cobertura parcial**, complementando lacunas com fontes alternativas: solução híbrida possível, mas adiciona heterogeneidade metodológica que enfraquece a consistência do indicador.
- **Documentar como descarte (escolhido)**: mantém a estrutura do IIC consistente e a defesa metodológica clara.

## Consequências

- Positivas: alinha IE com a definição estrita de ameaças climáticas (não confundir com ações humanas como desmatamento); preserva diferenciação intramunicipal em todos os indicadores; cobertura nacional uniforme em todos.
- Negativas / trade-offs: ventos extremos é ameaça real em algumas regiões (Sul, especialmente após Catarina 2004 e ciclone bomba 2020) — o IIC subestima esse fenômeno; revisitar quando houver fonte oficial nacional com resolução adequada.
- Confiança: Alta — descartes justificados por motivos verificáveis e por alternativas equivalentes já em uso (BATER substituído por MapBiomas+CNEFE e HAND+JRC).

## Referências

- ADR-0008 (critérios), ADR-0020 a ADR-0024 (indicadores IE incluídos), ADR-0030 (seca SPI/CHIRPS — ADR próprio), ADR-0031 (NDVI áreas verdes — ADR próprio).
- Feedback consolidado das duas rodadas de validação, seção "Decisões na Dimensão de Exposição".
