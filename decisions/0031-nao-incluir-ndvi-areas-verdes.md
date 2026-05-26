# ADR-0031: Não incluir NDVI de áreas verdes como indicador da dimensão IV

## Status

Accepted — 2026-05-19. Implementação preservada em `etl/discarded/v6_areas_verdes.py` e `etl/discarded/h3_v6_areas_verdes_ndvi_gee.js`.

## Contexto

O acesso a áreas verdes foi sugerido em validação como possível indicador da dimensão IV (vulnerabilidade) — populações em áreas com baixa cobertura vegetal estão mais expostas a ilhas de calor, têm menor qualidade de ar e menor amenidade urbana. A implementação foi feita como **v6 — NDVI invertido** (`v6_ver_norm = 1 − normalize(ndvi_mean)`) usando composição mediana Landsat 8/9 (2020–2024) via GEE: hexágonos com menor cobertura vegetal recebiam pontuação maior de vulnerabilidade.

A análise pós-implementação revelou que o sinal de v6 era **fortemente correlacionado (em direção oposta)** com `e4` (calor extremo — anomalia LST). Isso é fisicamente esperado: vegetação reduz temperatura de superfície (evapotranspiração + sombra + albedo), então hexágonos com NDVI baixo tendem a ter LST alta. Em análise para municípios inteiros, os dois indicadores produzem mapas quase espelhados.

## Decisão

**Não incluir v6 (NDVI áreas verdes) na dimensão IV.** O código é preservado em `etl/discarded/`.

Categoria de exclusão (ADR-0008): **(8) redundância estatística** — o sinal informativo de NDVI invertido é capturado pela face oposta do indicador e4 (calor extremo). Incluir ambos inflaria artificialmente a contribuição do fenômeno "déficit verde / excesso de calor" no IIC final, dobrando o peso de uma única realidade física.

## Alternativas consideradas

- **Incluir v6 mesmo com redundância**: adicionaria a dimensão "amenidade urbana" do ponto de vista do morador (vegetação tem valor próprio além do efeito sobre temperatura — bem-estar, qualidade do ar, biodiversidade urbana). Mas inflaria o peso prático do mesmo fenômeno físico no IIC.
- **Substituir e4 por v6** (manter NDVI, descartar calor extremo): possível, mas e4 mede diretamente a ameaça climática (anomalia de temperatura), enquanto v6 mede um proxy estrutural. Para a dimensão IE, e4 é conceitualmente mais defensável.
- **Combinar e4 e v6 em um único indicador composto**: aumentaria a complexidade do cálculo e perderia a interpretabilidade de ambos isoladamente; a média aritmética simples (ADR-0013) não comporta esse tipo de fusão.
- **Não incluir v6 e manter material no archive (escolhido)**: evita redundância estatística e preserva a interpretabilidade do IIC; trabalho técnico fica documentado para revisão futura.

## Consequências

- Positivas: IIC fica livre de dupla contagem do fenômeno "vegetação ↔ calor"; e4 carrega o sinal de forma mais direta e defensável; estrutura do índice fica mais enxuta.
- Negativas / trade-offs: o aspecto de "amenidade urbana" e "acesso a áreas verdes" não é contemplado no IIC — comunidades em áreas com baixa cobertura vegetal aparecem apenas se também tiverem outros indicadores ruins (calor, infraestrutura); revisores podem questionar a ausência de áreas verdes — o ADR documenta a justificativa empírica (correlação inversa com e4).
- Confiança: Alta — análise empírica direta motivou o descarte; categoria 8 (redundância) do ADR-0008 cobre o caso.

## Referências

- ADR-0008 (critérios), ADR-0018 (indicadores IV incluídos), ADR-0023 (e4 calor extremo).
- [etl/discarded/v6_areas_verdes.py](../etl/discarded/v6_areas_verdes.py) — ETL preservado.
- [etl/discarded/h3_v6_areas_verdes_ndvi_gee.js](../etl/discarded/h3_v6_areas_verdes_ndvi_gee.js) — script GEE preservado.
- Feedback consolidado das duas rodadas de validação, seção sobre acesso a áreas verdes.
