// ==============================================================================
// E1 — Suscetibilidade a Deslizamentos de Terra (NASA LHASA)
//
// Fonte: Stanley & Kirschbaum (2017) — Global Landslide Susceptibility Map
//        Modelo multicritério (fuzzy overlay) que combina:
//          - Slope (declividade)
//          - Geology (litologia)
//          - Forest loss (perda de floresta)
//          - Road density (densidade de estradas)
//          - Fault distance (distância a falhas geológicas)
//        Resolução nativa: ~1 km (30 arc-seconds)
//        Valores: 1 = Very Low, 2 = Low, 3 = Moderate, 4 = High, 5 = Very High
//        (valor 0 no raster = NoData/oceano — mascarado neste script)
//
// Justificativa metodológica:
//   Comparação com método baseado em slope puro (Vale do Taquari) revelou
//   correlação negativa em escala nacional — slopes muito íngremes no Brasil
//   correspondem frequentemente a afloramentos rochosos consolidados (baixo
//   risco), enquanto áreas de alto risco real (Amazônia + desmatamento,
//   encostas tropicais úmidas) têm slopes moderados. O LHASA captura essa
//   complexidade via modelo multicritério.
//
// Saída por hexágono H3 res9:
//   lhasa_mean       – valor médio de suscetibilidade no hexágono (1–5)
//   lhasa_high_frac  – fração da área com LHASA >= 4 (High ou Very High)
//
// WORKFLOW (passo único):
//   Não há classificação a ser feita — o raster LHASA já é a classificação.
//   Apenas extraímos os valores médios por hexágono.
// ==============================================================================

// --------------------------------------------------------------------------
// CONFIGURAÇÃO — Assets e constantes
// --------------------------------------------------------------------------

// Tabela de hexágonos habitados do Brasil (todos os H3 res9 com ≥ 1 domicílio)
var table = ee.FeatureCollection("projects/ee2-linafaccin/assets/br_h3_lat_lon");

// Asset do NASA LHASA (Stanley & Kirschbaum 2017)
var LHASA_ASSET_ID = "projects/ee2-linafaccin/assets/nasa_lhasa_susceptibility";

// Circunraio do hexágono H3 resolução 9 (centro → vértice), em metros
var H3_RES9_CIRCUMRADIUS_M = 174;

// Lista completa de UFs do Brasil
var ufs = [
  11, 12, 13, 14, 15, 16, 17,          // Norte
  21, 22, 23, 24, 25, 26, 27, 28, 29,  // Nordeste
  31, 32, 33, 35,                       // Sudeste
  41, 42, 43,                           // Sul
  50, 51, 52, 53                        // Centro-Oeste
];

// --------------------------------------------------------------------------
// LHASA — carrega raster e mascara pixels NoData (valor 0 = oceano)
// --------------------------------------------------------------------------
var lhasa_raw = ee.Image(LHASA_ASSET_ID);
var lhasa = lhasa_raw.updateMask(lhasa_raw.gt(0));

// Banda 1: valor médio (1–5)
var lhasa_mean_band = lhasa.rename('lhasa_mean');

// Banda 2: fração de área com LHASA High ou Very High (>= 4)
// É a métrica principal — equivalente a "fração da área em alto risco"
var lhasa_high_band = lhasa.gte(4).rename('lhasa_high_frac');

// Função auxiliar — buffer no centróide do hexágono (~área do H3 res9)
function bufferHexagon(feature) {
  return feature.buffer(H3_RES9_CIRCUMRADIUS_M);
}

// --------------------------------------------------------------------------
// EXPORTAÇÃO — CSV por UF para Google Drive
// --------------------------------------------------------------------------
ufs.forEach(function(uf_code) {

  var uf_collection = table.filter(ee.Filter.eq('cd_uf', uf_code));
  var uf_buffered   = uf_collection.map(bufferHexagon);

  // Reducer.mean() em ambas as bandas:
  //   - lhasa_mean      → média do valor LHASA no hexágono (1–5)
  //   - lhasa_high_frac → fração da área com LHASA >= 4 (0–1)
  // scale: 1000 = resolução nativa do LHASA (~1 km)
  var h3_com_lhasa = lhasa_mean_band.addBands(lhasa_high_band).reduceRegions({
    collection: uf_buffered,
    reducer: ee.Reducer.mean(),
    scale: 1000,
    tileScale: 16
  });

  Export.table.toDrive({
    collection: h3_com_lhasa,
    description: 'h3_susc_desliz_lhasa_v1_uf_' + uf_code,
    folder: 'GEE_suscetibilidade-deslizamentos-lhasa-v1',
    fileFormat: 'CSV',
    selectors: ['h3_id', 'cd_uf', 'cd_setor', 'lhasa_mean', 'lhasa_high_frac']
  });
});

print('=== EXPORTAÇÕES ENVIADAS ===');
print(ufs.length + ' tarefas enviadas → Google Drive: GEE_suscetibilidade-deslizamentos-lhasa-v1');
print('Colunas exportadas: h3_id | cd_uf | cd_setor | lhasa_mean | lhasa_high_frac');
print('');
print('Fonte: Stanley & Kirschbaum (2017) — NASA LHASA Global Landslide Susceptibility');
print('Resolução nativa: ~1 km — reamostrado para 1000 m no reduceRegions');
print('');
print('Pronto para processar com etl/exposure/e1_deslizamentos_lhasa.py');
