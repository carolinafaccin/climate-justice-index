// ==============================================================================
// E2 — Suscetibilidade a Inundações (HAND + JRC Flood Hazard)
//
// METODOLOGIA (adaptada do MapBiomas Risco Climático, sem máscara urbana)
//
//   Componente 1 — HAND classificado em 3 classes:
//     • HAND 0–2 m → score 1.00 (muito alta suscetibilidade)
//     • HAND 2–4 m → score 0.66 (alta)
//     • HAND 4–6 m → score 0.33 (média)
//     • HAND > 6 m → score 0.00 (sem)
//
//   Componente 2 — JRC Global River Flood Hazard (100-year return period):
//     Máscara que retém apenas pixels com perigo de inundação modelado
//     (RP100_depth > 0)
//
//   Score final por pixel = score_HAND × máscara_JRC
//
//   IMPORTANTE: Sem máscara de áreas urbanas (diferença vs MapBiomas oficial)
//   Cobertura nacional independente de uso da terra.
//
// FONTES:
//   • HAND global: users/gena/global-hand/hand-100 (Donchyts et al. 2016, ~30m)
//   • JRC Flood Hazard: JRC/CEMS_GLOFAS/FloodHazard/v2_1 (~1km, banda RP100_depth)
//
// SAÍDA POR HEXÁGONO H3 res9:
//   flood_score – score contínuo médio de suscetibilidade (0–1)
//                 = indicador e2 (sem precisar de inversão)
//
// WORKFLOW (dois passos):
//   PASSO 1 (RUN_STEP = 1): gera raster combinado e exporta como asset
//                           *** AGUARDE TERMINAR antes de rodar o passo 2 ***
//   PASSO 2 (RUN_STEP = 2): exporta CSVs por UF para o Google Drive
// ==============================================================================

var RUN_STEP = 1;  // << ALTERE: 1 → 2

// --------------------------------------------------------------------------
// CONFIGURAÇÃO — Assets e constantes
// --------------------------------------------------------------------------

// Tabela de hexágonos habitados do Brasil
var table = ee.FeatureCollection("projects/ee2-linafaccin/assets/br_h3_lat_lon");

// Asset de destino do raster combinado (Passo 1)
var ASSET_ID = "projects/ee2-linafaccin/assets/br_inund_hand_jrc_score_v1";

// Datasets fonte
var HAND_ASSET_ID = "users/gena/global-hand/hand-100";
var JRC_ASSET_ID  = "JRC/CEMS_GLOFAS/FloodHazard/v2_1";

// Circunraio do hexágono H3 resolução 9, em metros
var H3_RES9_CIRCUMRADIUS_M = 174;

// Bounding box do Brasil
var BR_BBOX = ee.Geometry.BBox(-74, -35, -28, 6);

// Lista completa de UFs do Brasil
var ufs = [
  11, 12, 13, 14, 15, 16, 17,
  21, 22, 23, 24, 25, 26, 27, 28, 29,
  31, 32, 33, 35,
  41, 42, 43,
  50, 51, 52, 53
];

// Buffer no centróide do hexágono → aproxima a área real do hexágono
var bufferHexagon = function(feature) {
  return feature.buffer(H3_RES9_CIRCUMRADIUS_M);
};

// --------------------------------------------------------------------------
// PASSO 1 — Calcula score combinado HAND + JRC e exporta como asset
// --------------------------------------------------------------------------
if (RUN_STEP === 1) {

  // HAND como ImageCollection → mosaico
  var hand = ee.ImageCollection(HAND_ASSET_ID).mosaic().select(0);

  // JRC Global River Flood Hazard — return period 100 anos
  // Banda RP100_depth: profundidade de inundação em metros (T=100 anos)
  // Valores > 0 indicam área com perigo modelado
  var jrc = ee.ImageCollection(JRC_ASSET_ID).mosaic().select('RP100_depth');

  // Máscara JRC: pixels com perigo modelado de inundação
  // unmask(0): pixels fora da JRC viram 0 (não-perigosos)
  var jrc_mask = jrc.unmask(0).gt(0);

  // Score HAND por classe (valores 0–1)
  //   0–2 m → 1.00
  //   2–4 m → 0.66
  //   4–6 m → 0.33
  //   > 6 m → 0.00
  var hand_score = ee.Image(0)
    .where(hand.gte(4).and(hand.lt(6)), 0.33)
    .where(hand.gte(2).and(hand.lt(4)), 0.66)
    .where(hand.gt(0).and(hand.lt(2)),  1.00);

  // Máscara de terra: HAND válido (> 0 exclui oceano/NoData)
  var land_mask = hand.gt(0);

  // Score combinado: HAND classificado × máscara JRC
  // Pixels com HAND baixo MAS sem perigo JRC = 0 (não consideramos)
  // Pixels com perigo JRC MAS HAND alto = 0 (acima de 6m)
  var flood_score = hand_score
    .multiply(jrc_mask)
    .updateMask(land_mask)
    .rename('flood_score')
    .float();

  Export.image.toAsset({
    image: flood_score,
    description: 'br_inund_hand_jrc_score_v1',
    assetId: ASSET_ID,
    scale: 30,
    maxPixels: 1e13,
    region: BR_BBOX,
    pyramidingPolicy: { 'flood_score': 'mean' }
  });

  print('=== PASSO 1 ENVIADO ===');
  print('Metodologia: HAND classificado × JRC Flood Hazard (100-yr)');
  print('Classes HAND: 0-2m=1.00, 2-4m=0.66, 4-6m=0.33, >6m=0.00');
  print('Sem máscara urbana → cobertura nacional');
  print('Fonte HAND: ' + HAND_ASSET_ID);
  print('Fonte JRC : ' + JRC_ASSET_ID);
  print('Asset destino: ' + ASSET_ID);
  print('');
  print('*** AGUARDE o término na aba Tasks antes de alterar RUN_STEP para 2 ***');
}

// --------------------------------------------------------------------------
// PASSO 2 — Score médio por hexágono + exportação CSV por UF
// --------------------------------------------------------------------------
if (RUN_STEP === 2) {

  var flood_asset = ee.Image(ASSET_ID);

  ufs.forEach(function(uf_code) {

    var uf_collection = table.filter(ee.Filter.eq('cd_uf', uf_code));
    var uf_buffered   = uf_collection.map(bufferHexagon);

    // unmask(0): pixels mascarados (HAND=0, calha de rios) contribuem com 0
    // setOutputs(['flood_score']): nomeia a propriedade de saída do reducer
    // (sem isso, ee.Reducer.mean() gera propriedade 'mean', que não bate com
    // o selector 'flood_score' do export → CSV sairia com a coluna vazia)
    var h3_com_score = flood_asset.unmask(0).reduceRegions({
      collection: uf_buffered,
      reducer: ee.Reducer.mean().setOutputs(['flood_score']),
      scale: 30,
      tileScale: 16
    });

    Export.table.toDrive({
      collection: h3_com_score,
      description: 'h3_susc_inund_hand_jrc_v1_uf_' + uf_code,
      folder: 'GEE_suscetibilidade-inundacoes-hand-jrc-v1',
      fileFormat: 'CSV',
      selectors: ['h3_id', 'cd_uf', 'cd_setor', 'flood_score']
    });
  });

  print('=== PASSO 2 ENVIADO ===');
  print(ufs.length + ' tarefas enviadas → Google Drive: GEE_suscetibilidade-inundacoes-hand-jrc-v1');
  print('Colunas exportadas: h3_id | cd_uf | cd_setor | flood_score');
  print('Pronto para processar com etl/exposure/e2_inundacoes_hand.py');
}
