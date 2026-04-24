// --- 1. CONFIGURAÇÃO ---
// ATENÇÃO: Suba o seu novo arquivo 'h3_br_lat_lon_mar.csv' 
// como um novo Asset de Tabela (do tipo Point, com Lat/Lon) no GEE.
var table = ee.FeatureCollection("projects/ee2-linafaccin/assets/br_h3_lat_lon_mar");

// Lista apenas de UFs que possuem litoral (economiza muito processamento)
var ufs_costeiras = [
  15, 16,                                        // Norte (PA, AP)
  21, 22, 23, 24, 25, 26, 27, 28, 29,            // Nordeste
  32, 33, 35,                                    // Sudeste (ES, RJ, SP)
  41, 42, 43                                     // Sul (PR, SC, RS)
];

// --- 2. PREPARAÇÃO DO DEM (Copernicus GLO-30) ---
// O Copernicus é uma coleção de imagens, então fazemos um mosaic para unir todas
var dem = ee.ImageCollection("COPERNICUS/DEM/GLO30")
  .select('DEM')
  .mosaic();

// Criando a imagem de risco (Binária):
// .lte(1) significa "Less Than or Equal" a 1 metro.
// O resultado é 1 (Verdadeiro/Risco) ou 0 (Falso/Seguro).
var risco_mar = dem.lte(1).rename('risco_slr');

// --- 3. PROCESSAMENTO E EXPORTAÇÃO ---
ufs_costeiras.forEach(function(uf_code) {
  
  // 1. Filtra pela UF (usando o número que o diagnóstico confirmou)
  var uf_collection = table.filter(ee.Filter.eq('cd_uf', uf_code));
  
  // 2. Filtra municípios costeiros (usando o valor lógico true)
  // Nota: se ainda der zero, tente mudar 'true' para 1 (sem aspas)
  var uf_costeira = uf_collection.filter(ee.Filter.eq('municipios_defrontantes_com_o_mar', true));

  // 3. Extrai o valor do risco
  var h3_com_risco = risco_mar.reduceRegions({
    collection: uf_costeira,
    reducer: ee.Reducer.first().setOutputs(['risco_slr']),
    scale: 30,
    tileScale: 16
  });

  // 4. Exporta
  Export.table.toDrive({
    collection: h3_com_risco,
    description: 'h3_susc_mar_1m_uf_' + uf_code,
    folder: 'GEE_suscetibilidade-elevacao-nivel-mar',
    fileFormat: 'CSV',
    selectors: ['h3_id', 'cd_mun', 'cd_setor', 'qtd_dom', 'risco_slr'] 
  });
});

print("Tarefas de Nível do Mar enviadas! Acesse a aba Tasks.");