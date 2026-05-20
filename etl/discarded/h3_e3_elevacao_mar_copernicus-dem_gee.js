// ==============================================================================
// E3 — Suscetibilidade à Elevação do Nível do Mar
// Fonte: Copernicus GLO-30 DEM (resolução 30m)
// Método: buffer de 174m no centróide do hexágono H3 res9 + Reducer.mean()
//
// v2: o asset original contém apenas o centróide (ponto) de cada hexágono.
//     Com geometria de ponto, Reducer.first() e Reducer.mean() são equivalentes
//     — ambos amostram 1 único pixel. Para captar a fração real da área em risco,
//     é necessário dar área à geometria antes do reduceRegions.
//     Solução: buffer de 174m (circunraio do hexágono H3 res9 = distância do
//     centro ao vértice), que aproxima bem o hexágono real a 30m de resolução.
//     O resultado de risco_slr passa a ser a fração da área circular (~hexágono)
//     com elevação ≤ 1m (ex: 0.4 = ~40% da área está em risco).
// ==============================================================================

// --- 1. CONFIGURAÇÃO ---
var table = ee.FeatureCollection("projects/ee2-linafaccin/assets/br_h3_lat_lon_mar");

// Circunraio do hexágono H3 resolução 9 (centro → vértice), em metros
var H3_RES9_CIRCUMRADIUS_M = 174;

// UFs com litoral
var ufs_costeiras = [
  15, 16,                               // Norte (PA, AP)
  21, 22, 23, 24, 25, 26, 27, 28, 29,  // Nordeste
  32, 33, 35,                           // Sudeste (ES, RJ, SP)
  41, 42, 43                            // Sul (PR, SC, RS)
];

// --- 2. PREPARAÇÃO DO DEM (Copernicus GLO-30) ---
var dem = ee.ImageCollection("COPERNICUS/DEM/GLO30")
  .select('DEM')
  .mosaic();

// Raster binário: 1 = elevação ≤ 1m (risco), 0 = elevação > 1m (seguro)
var risco_mar = dem.lte(1).rename('risco_slr').toFloat();

// --- 3. FUNÇÃO: converte ponto em círculo (aproximação do hexágono) ---
function bufferHexagon(feature) {
  return feature.buffer(H3_RES9_CIRCUMRADIUS_M);
}

// --- 4. PROCESSAMENTO E EXPORTAÇÃO ---
ufs_costeiras.forEach(function(uf_code) {

  var uf_costeira = table
    .filter(ee.Filter.eq('cd_uf', uf_code))
    .filter(ee.Filter.eq('municipios_defrontantes_com_o_mar', true));

  // Aplica o buffer para dar área à geometria antes do reduceRegions
  var uf_buffered = uf_costeira.map(bufferHexagon);

  var h3_com_risco = risco_mar.reduceRegions({
    collection: uf_buffered,
    reducer: ee.Reducer.mean().setOutputs(['risco_slr']),
    scale: 30,
    tileScale: 16
  });

  Export.table.toDrive({
    collection: h3_com_risco,
    description: 'h3_susc_mar_1m_mean_uf_' + uf_code,
    folder: 'GEE_suscetibilidade-elevacao-nivel-mar-v2',
    fileFormat: 'CSV',
    selectors: ['h3_id', 'cd_mun', 'cd_setor', 'qtd_dom', 'risco_slr']
  });
});

print("Tarefas v2 enviadas! risco_slr agora é a fração da área do hexágono (~buffer 174m) abaixo de 1m.");
