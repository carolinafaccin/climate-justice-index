// ==============================================================================
// E4 — Calor Extremo (Anomalia de Temperatura Superficial)
// Fonte: Landsat 5, 7, 8, 9 — NASA/USGS Collection 2 Level-2
// Método: buffer de 174m no centróide do hexágono H3 res9 + Reducer.mean()
//
// v2: o asset br_h3_lat_lon contém apenas o centróide (ponto) de cada hexágono.
//     Com geometria de ponto, Reducer.mean() amostra 1 único pixel — equivalente
//     a Reducer.first(). Para captar a média real de LST dentro do hexágono,
//     aplica-se um buffer de 174m (circunraio do H3 res9) antes do reduceRegions,
//     transformando o ponto em um círculo que aproxima a área do hexágono.
// ==============================================================================

// --- 1. CONFIGURAÇÃO ---
var table = ee.FeatureCollection("projects/ee2-linafaccin/assets/br_h3_lat_lon");

// Circunraio do hexágono H3 resolução 9 (centro → vértice), em metros
var H3_RES9_CIRCUMRADIUS_M = 174;

// Lista de UFs
var ufs = [
  11, 12, 13, 14, 15, 16, 17,
  21, 22, 23, 24, 25, 26, 27, 28, 29,
  31, 32, 33, 35,
  41, 42, 43,
  50, 51, 52, 53
];

// --- 2. FUNÇÕES DE PROCESSAMENTO (Remoção de Nuvens e Conversão para LST em °C) ---

// Landsat 5 e 7 (banda térmica ST_B6)
function prepL5_7(image) {
  var qa   = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0).and(qa.bitwiseAnd(1 << 4).eq(0));
  return image.select('ST_B6')
    .updateMask(mask)
    .multiply(0.00341802).add(149.0).subtract(273.15)
    .rename('LST')
    .copyProperties(image, ['system:time_start']);
}

// Landsat 8 e 9 (banda térmica ST_B10)
function prepL8_9(image) {
  var qa   = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0).and(qa.bitwiseAnd(1 << 4).eq(0));
  return image.select('ST_B10')
    .updateMask(mask)
    .multiply(0.00341802).add(149.0).subtract(273.15)
    .rename('LST')
    .copyProperties(image, ['system:time_start']);
}

// --- 3. COLEÇÕES LANDSAT ---
var l5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").map(prepL5_7);
var l7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2").map(prepL5_7);
var l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").map(prepL8_9);
var l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").map(prepL8_9);

var all_landsat = l5.merge(l7).merge(l8).merge(l9);

// --- 4. PERÍODOS E CÁLCULO DA ANOMALIA ---
var col_hist = all_landsat.filterDate('1985-01-01', '2010-12-31').mean();
var col_curr = all_landsat.filterDate('2015-01-01', '2024-12-31').mean();

// anomalia_temp = LST média atual − LST média histórica (°C)
var anomalia_img = col_curr.subtract(col_hist).rename('anomalia_temp');

// --- 5. FUNÇÃO: buffer no centróide para aproximar a área do hexágono ---
function bufferHexagon(feature) {
  return feature.buffer(H3_RES9_CIRCUMRADIUS_M);
}

// --- 6. PROCESSAMENTO E EXPORTAÇÃO POR UF ---
ufs.forEach(function(uf_code) {

  var uf_collection = table.filter(ee.Filter.eq('cd_uf', uf_code));

  // Aplica buffer para dar área à geometria antes do reduceRegions
  var uf_buffered = uf_collection.map(bufferHexagon);

  var h3_with_anomaly = anomalia_img.reduceRegions({
    collection: uf_buffered,
    reducer: ee.Reducer.mean().setOutputs(['anomalia_temp']),
    scale: 30,
    tileScale: 16
  });

  Export.table.toDrive({
    collection: h3_with_anomaly,
    description: 'h3_anomalia_calor_1985-2025_v2_uf_' + uf_code,
    folder: 'GEE_anomalia_calor_1985-2025_v2',
    fileFormat: 'CSV',
    selectors: ['h3_id', 'cd_setor', 'qtd_dom', 'anomalia_temp']
  });
});

print("Tarefas v2 enviadas! anomalia_temp agora é a média de LST dentro do buffer de 174m (~hexágono H3 res9).");
