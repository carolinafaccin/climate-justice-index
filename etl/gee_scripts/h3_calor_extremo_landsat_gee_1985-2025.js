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

// --- 2. FUNÇÕES DE PROCESSAMENTO (Nuvem e Temperatura) ---

// Função para Landsat 5 e 7 (Banda 6)
function prepL5_7(image) {
  var qa = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0).and(qa.bitwiseAnd(1 << 4).eq(0));
  return image.select('ST_B6') // Banda térmica L5 e L7
    .updateMask(mask)
    .multiply(0.00341802).add(149.0).subtract(273.15)
    .rename('LST') // Renomeia para padronizar
    .copyProperties(image, ['system:time_start']);
}

// Função para Landsat 8 e 9 (Banda 10)
function prepL8_9(image) {
  var qa = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0).and(qa.bitwiseAnd(1 << 4).eq(0));
  return image.select('ST_B10') // Banda térmica L8 e L9
    .updateMask(mask)
    .multiply(0.00341802).add(149.0).subtract(273.15)
    .rename('LST') // Renomeia para padronizar
    .copyProperties(image, ['system:time_start']);
}

// Importando e preparando as coleções
var l5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").map(prepL5_7);
var l7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2").map(prepL5_7);
var l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").map(prepL8_9);
var l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").map(prepL8_9);

// Mesclando todas as missões em um único histórico unificado
var all_landsat = l5.merge(l7).merge(l8).merge(l9);

// --- 3. DEFINIÇÃO DE PERÍODOS E CÁLCULO DA ANOMALIA ---

// Período Histórico de Referência (25 anos)
var hist_start = '1985-01-01';
var hist_end = '2010-12-31';

// Período Atual (Alinhado com as outras variáveis do índice)
var curr_start = '2015-01-01';
var curr_end = '2024-12-31';

// Média Histórica
var col_hist = all_landsat.filterDate(hist_start, hist_end).mean();

// Média Atual
var col_curr = all_landsat.filterDate(curr_start, curr_end).mean();

// Criando a imagem de Anomalia (Atual - Histórica)
var anomalia_img = col_curr.subtract(col_hist).rename('anomalia_temp');

// --- 4. PROCESSAMENTO E EXPORTAÇÃO ---
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
    description: 'h3_anomalia_calor_1985-2025_uf_' + uf_code,
    folder: 'GEE_anomalia_calor_1985-2025',
    fileFormat: 'CSV',
    selectors: ['h3_id', 'cd_setor', 'qtd_dom', 'anomalia_temp']
  });
});

print("Tarefas enviadas! O GEE utilizará a série histórica de 40 anos do Landsat.");