// --- 1. CONFIGURAÇÃO ---
var table = ee.FeatureCollection("projects/ee2-linafaccin/assets/br_h3_lat_lon");

// Lista de UFs
var ufs = [
  11, 12, 13, 14, 15, 16, 17, 
  21, 22, 23, 24, 25, 26, 27, 28, 29, 
  31, 32, 33, 35, 
  41, 42, 43, 
  50, 51, 52, 53 
];

// --- 2. SATÉLITE COM REMOÇÃO DE NUVENS ---
function prepL8(image) {
  var qa = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0).and(qa.bitwiseAnd(1 << 4).eq(0));
  return image.select('ST_B10')
    .updateMask(mask)
    .multiply(0.00341802).add(149.0).subtract(273.15)
    .copyProperties(image, ['system:time_start']);
}

// Definindo os períodos para o cálculo da Anomalia
var hist_start = '2014-01-01';
var hist_end = '2018-12-31';

var curr_start = '2019-01-01';
var curr_end = '2024-12-31';

// Média Histórica
var col_hist = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    .filterDate(hist_start, hist_end)
    .map(prepL8)
    .mean();

// Média Atual
var col_curr = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    .filterDate(curr_start, curr_end)
    .map(prepL8)
    .mean();

// Criando a imagem de Anomalia (Atual - Histórica)
var anomalia_img = col_curr.subtract(col_hist).rename('anomalia_temp');

// --- 3. PROCESSAMENTO E EXPORTAÇÃO ---
ufs.forEach(function(uf_code) {
  
  var uf_collection = table.filter(ee.Filter.eq('cd_uf', uf_code));

  var h3_with_anomaly = anomalia_img.reduceRegions({
    collection: uf_collection,
    reducer: ee.Reducer.mean().setOutputs(['anomalia_temp']), 
    scale: 30,
    tileScale: 16
  });

  // Exporta
  Export.table.toDrive({
    collection: h3_with_anomaly,
    description: 'h3_anomalia_calor_2014-2024_uf_' + uf_code,
    folder: 'GEE_anomalia_calor_2014-2024',
    fileFormat: 'CSV',
    selectors: ['h3_id', 'cd_setor', 'qtd_dom', 'anomalia_temp']
  });
});

print("Tarefas enviadas! O GEE calculará a anomalia térmica por hexágono.");