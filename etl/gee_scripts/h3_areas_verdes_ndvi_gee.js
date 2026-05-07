// ==============================================================================
// V_VERDE — Cobertura Vegetal por Hexágono H3 (NDVI)
// Fonte: Landsat 8 e 9 — NASA/USGS Collection 2 Level-2 Surface Reflectance
// Método: buffer de 174m no centróide do hexágono H3 res9 + Reducer.mean()
//
// Lógica:
//   1. Imagens Landsat 8/9 do período 2020–2024 são mascaradas para nuvens
//      e sombras (bits 3 e 4 do QA_PIXEL).
//   2. NDVI = (SR_B5 − SR_B4) / (SR_B5 + SR_B4), onde B4=Vermelho e B5=NIR.
//   3. Um composto mediano de todos os pixels válidos do período é gerado.
//      Mediana é preferível à média para NDVI porque é mais robusta a
//      artefatos residuais de nuvens e a picos temporários de vegetação agrícola.
//   4. Cada hexágono recebe o NDVI médio dentro de um buffer de 174m
//      (circunraio do H3 res9) aplicado sobre seu centróide.
//
// Saída compatível com v_areas_verdes.py (colunas: h3_id, cd_setor, qtd_dom, ndvi_mean)
// ==============================================================================

// --- 1. CONFIGURAÇÃO ---
var table = ee.FeatureCollection("projects/ee2-linafaccin/assets/br_h3_lat_lon");
var H3_RES9_CIRCUMRADIUS_M = 174;

var ufs = [
  11, 12, 13, 14, 15, 16, 17,
  21, 22, 23, 24, 25, 26, 27, 28, 29,
  31, 32, 33, 35,
  41, 42, 43,
  50, 51, 52, 53
];

// --- 2. CLOUD MASK + NDVI ---
function prepL8_9(image) {
  var qa   = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0).and(qa.bitwiseAnd(1 << 4).eq(0));

  // Reflectância superficial: escala 0.0000275, offset -0.2
  var red = image.select('SR_B4').multiply(0.0000275).add(-0.2).updateMask(mask);
  var nir = image.select('SR_B5').multiply(0.0000275).add(-0.2).updateMask(mask);

  // Clamp para [0,1] — evita valores negativos de reflectância (artefatos atmosféricos)
  red = red.clamp(0, 1);
  nir = nir.clamp(0, 1);

  var ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI');

  return ndvi.copyProperties(image, ['system:time_start']);
}

// --- 3. COMPOSTO MEDIANO LANDSAT 8/9 (2020–2024) ---
var l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
  .filterDate('2020-01-01', '2024-12-31')
  .map(prepL8_9);

var l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
  .filterDate('2020-01-01', '2024-12-31')
  .map(prepL8_9);

// Mediana temporal: reduz efeito de sazonalidade e artefatos residuais
var ndvi_img = l8.merge(l9).median().rename('ndvi_mean');

// --- 4. FUNÇÃO: buffer no centróide para aproximar a área do hexágono ---
function bufferHexagon(feature) {
  return feature.buffer(H3_RES9_CIRCUMRADIUS_M);
}

// --- 5. PROCESSAMENTO E EXPORTAÇÃO POR UF ---
ufs.forEach(function(uf_code) {

  var uf_collection = table.filter(ee.Filter.eq('cd_uf', uf_code));
  var uf_buffered   = uf_collection.map(bufferHexagon);

  var h3_with_ndvi = ndvi_img.reduceRegions({
    collection: uf_buffered,
    reducer: ee.Reducer.mean().setOutputs(['ndvi_mean']),
    scale: 30,
    tileScale: 16
  });

  Export.table.toDrive({
    collection: h3_with_ndvi,
    description: 'h3_areas_verdes_ndvi_uf_' + uf_code,
    folder: 'GEE_areas_verdes_ndvi',
    fileFormat: 'CSV',
    selectors: ['h3_id', 'cd_setor', 'qtd_dom', 'ndvi_mean']
  });
});

print("Tarefas enviadas — " + ufs.length + " UFs → pasta GEE_areas_verdes_ndvi");
print("Composto: mediana Landsat 8+9, 2020–2024, nuvens mascaradas (QA_PIXEL bits 3+4)");
print("Compatível com v_areas_verdes.py sem alterações.");
