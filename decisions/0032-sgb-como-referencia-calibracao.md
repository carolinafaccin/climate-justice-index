# ADR-0032: Usar Cartografia de Suscetibilidade SGB/CPRM como referência para calibrar E1 e validar E2

## Status
Accepted — 2026-05-22

## Contexto
Os indicadores E1 (deslizamentos, via NASA LHASA) e E2 (inundações, via HAND+JRC) foram definidos com thresholds heurísticos: `lhasa_high_frac >= 4` para E1 e teto de 6 m de HAND para E2. Ambos os thresholds foram adaptados de referências internacionais sem calibração contra dados brasileiros.

O Serviço Geológico do Brasil (SGB/CPRM) publica a Cartografia de Suscetibilidade a Desastres Geológicos para municípios prioritários, com mapeamento de campo em escala 1:25.000. Os mapas classificam a suscetibilidade em cinco níveis (Muito Alta a Muito Baixa) para dois processos: movimentos de massa e inundações/enxurradas. Cobertura atual: ~814 municípios, concentrados nas regiões Sul, Sudeste e Nordeste. O acesso é público via portal rigeo.sgb.gov.br.

## Decisão
Usar a Cartografia de Suscetibilidade SGB como **referência empírica nacional** para calibrar o threshold de E1 e validar o threshold de E2. A metodologia é:

1. **Download e harmonização (scripts 00–02):** baixar todos os ZIPs disponíveis, inventariar os shapefiles internos, mapear os valores textuais de classe para a escala 0–5 e consolidar em dois GeoPackages nacionais (inundações e massa).

2. **Interseção com grade H3 (script 03):** calcular, por hexágono H3 res9, a fração de área em cada classe SGB — produzindo `sgb_alta_mta_frac` (fração em classes 4–5) como referência de "alta suscetibilidade".

3. **Calibração E1 (script 05):** sweep de threshold `lhasa_high_frac > t` comparado contra `sgb_alta_mta_frac > 0.3` nos hexágonos com cobertura SGB de massa. Métrica: F1 (precision × recall).

4. **Validação E2 (script 06):** identificar falsos negativos de E2 (hexágono com alta suscetibilidade SGB a inundação e `flood_score < 0.1`) e analisar a distribuição de HAND nesses hexágonos para recomendar novo teto.

A cobertura parcial (~814 de ~5.570 municípios) é tratada como **amostra intencional** — os municípios mapeados pelo SGB são prioritários por histórico de desastres, o que é adequado para calibração de indicadores de risco. Os thresholds calibrados são então aplicados ao Brasil inteiro.

## Alternativas consideradas
- **Manter thresholds heurísticos sem calibração**: simples, mas sem evidência empírica de adequação ao contexto brasileiro. Inadequado para um índice com pretensão de rigor metodológico.
- **Atlas Nacional de Desastres (S2iD/CENAD)**: dados de ocorrência de desastres registrados. Vantagem: cobertura nacional. Desvantagem: reflete capacidade de registro municipal (viés de notificação), não suscetibilidade real. Útil como validação secundária, não como referência primária de calibração.
- **BATER (IBGE/Cemaden)**: mapeamento de áreas de risco em perímetros urbanos. Cobertura parcial e restrita a áreas urbanas — incompatível com a cobertura nacional do IIC (tratado no ADR-0028).
- **Cartografia SGB/CPRM (escolhida)**: mapeamento de campo em escala grande (1:25.000); classificação de suscetibilidade por processo (massa e inundação separados); acesso público e estruturado; cobertura parcial mas suficiente para calibração empírica.

## Consequências
- Positivas: calibração com dados de campo nacionais e escala adequada; separação por processo (massa ≠ inundação) alinhada com a estrutura de E1 e E2; pipeline automatizado permite replicação quando SGB ampliar cobertura.
- Negativas / trade-offs: ~814 municípios cobertos de ~5.570 totais (~15%); concentração regional (Sul/Sudeste/Nordeste) pode introduzir viés geográfico nos thresholds calibrados; mapeamentos SGB têm anos diferentes (2013–2024), não capturando mudanças de uso do solo recentes; heterogeneidade metodológica entre mapeamentos estaduais.
- Confiança: Média — referência de campo sólida, mas cobertura parcial e viés de seleção dos municípios prioritários limitam a generalização dos thresholds calibrados. Resultados devem ser apresentados com esse limite explícito.

## Referências
- ADR-0020: decisão de usar NASA LHASA para E1.
- ADR-0021: decisão de usar HAND+JRC para E2.
- ADR-0028: variáveis não incluídas em IE (inclui BATER).
- [etl/exposure/sgb/plano.md](../etl/exposure/sgb/plano.md) — detalhes da metodologia de calibração.
- [etl/exposure/sgb/README.md](../etl/exposure/sgb/README.md) — como executar o pipeline.
- Site SGB: https://www.sgb.gov.br/produtos-por-estado-cartografia-de-suscetibilidade
