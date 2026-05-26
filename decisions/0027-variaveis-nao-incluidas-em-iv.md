# ADR-0027: Registrar variáveis consideradas e não-incluídas na dimensão IV (Vulnerabilidade)

## Status

Accepted — 2026-05-19

## Contexto

Durante o desenho do IIC e nas duas rodadas de validação, especialistas sugeriram variáveis adicionais para a dimensão IV além das cinco incluídas (v1 renda, v2 moradia, v3 analfabetismo, v4 saúde, v5 infraestrutura — ADRs 0018 e 0019). O ADR-0031 trata especificamente do NDVI de áreas verdes (caso com código preservado em `etl/discarded/`). Este ADR registra os demais descartes da dimensão.

## Decisão

Registrar variáveis consideradas e descartadas para a dimensão IV, com a categoria de exclusão do ADR-0008.

| Variável sugerida | Categoria(s) | Razão concreta |
|---|---|---|
| Segurança alimentar | (4) granularidade | EBIA/PNADC público apenas em escala nacional/regional; CadÚnico restrito |
| Acesso a áreas verdes | (6) operacional; redirecionado para ADR-0031 | NDVI testado e descartado por redundância com e4 (calor) |
| Acesso à informação | (4) granularidade | TIC Domicílios (CGI.br) público apenas em recortes amostrais regionais |
| Mobilidade urbana / acessibilidade viária | (6) operacional | Cálculo de acessibilidade viária para 5.570 municípios exige rede viária OSM + modelagem de tempo de viagem — esforço incompatível com escopo atual |
| Doenças respiratórias | (1) acesso (LAI); (4) granularidade | DataSus georreferenciado via LAI; dados públicos apenas municipais |
| Cultura (proximidade a equipamentos culturais) | (3) cobertura; (6) operacional | Bases SNIIC/IBGE com cobertura parcial e geocodificação livre |
| DataSus georreferenciado (microdados) | (1) acesso (LAI); (7) qualidade | Acesso restrito; endereços em campo livre dificultam limpeza automatizada nacional; em municípios pequenos com poucos CEPs, geocodificação por CEP perde diferenciação intramunicipal |
| CadÚnico microdados georreferenciados | (1) acesso (LAI) | Microdados restritos via LAI; não configura dado aberto |

Critérios usam a taxonomia do ADR-0008.

## Alternativas consideradas

- **ADRs separados por variável**: granular, mas a maioria dos descartes tem o mesmo formato e justificativa — consolidar é mais econômico.
- **Manter informalmente na nota metodológica**: perde a defesa estruturada que o ADR oferece.
- **Consolidado por dimensão (escolhido)**: cada exclusão visível e rastreável.

## Consequências

- Positivas: defesa do artigo preparada; revisores que perguntarem "por que não há indicador de saúde direta?" têm resposta direta (DataSus georreferenciado por LAI; v4 usa CNES como melhor proxy aberto disponível).
- Negativas / trade-offs: lista envelhece com mudanças de fontes — DataSus pode abrir mais dados no futuro; revisitar quando aplicável.
- Confiança: Alta — categorias do ADR-0008 cobrem todos os casos com razões verificáveis.

## Referências

- ADR-0008 (critérios), ADR-0018 (indicadores IV incluídos), ADR-0019 (v4 saúde), ADR-0031 (NDVI áreas verdes descartado em ADR próprio).
- Feedback consolidado das duas rodadas de validação, seção "Variáveis não incluídas e justificativas" da dimensão IV.
