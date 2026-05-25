// ==============================================================================
// E2 — Suscetibilidade a Inundações (HAND + JRC híbrido em dois tiers) — v2
//
// MUDANÇAS vs v1 (calibradas pelo diagnóstico de FN — 07_sgb_analyse_fn_hand.py)
//
//   O v1 usava HAND × JRC como máscara binária multiplicativa, replicando
//   MapBiomas. Diagnóstico do script 07 revelou que 99% dos FN do E2 estão
//   em hexágonos com jrc_rp100=0. Esses FN concentram-se em áreas urbano-
//   pluviais (SE, S), onde o JRC global (~1km) sub-representa o risco
//   pluvial/flash flood que o SGB mapeia em detalhe.
//
//   Como removemos completamente o JRC divergiria da metodologia de
//   referência e inflaria FP em áreas pristinas longe de qualquer registro
//   de inundação, adotamos um SCORE HÍBRIDO EM DOIS TIERS:
//
//     • Tier 1 (JRC > 0): score clássico v1, ALTA CONFIANÇA hidrológica.
//     • Tier 2 (JRC = 0): score reduzido + ceiling estendido,
//                          CONFIANÇA TOPOGRÁFICA apenas.
//
//   Decisão documentada em decisions/0038-score-e2-hibrido-hand-jrc-tiers.md.
//
// METODOLOGIA v2 — TABELAS DE SCORE
//
//   Tier 1 — onde JRC RP100 > 0 (modelado pelo JRC como inundável):
//     • HAND 0–2 m  → 1.00 (muito alta)
//     • HAND 2–4 m  → 0.66 (alta)
//     • HAND 4–6 m  → 0.33 (média)
//     • HAND > 6 m  → 0.00
//
//   Tier 2 — onde JRC RP100 = 0 (não modelado, ou fora da máscara):
//     • HAND 0–2 m   → 0.50 (topografia muito favorável, sem corroboração)
//     • HAND 2–4 m   → 0.33
//     • HAND 4–6 m   → 0.20
//     • HAND 6–10 m  → 0.10  [NOVO — captura FN urbano-pluvial]
//     • HAND 10–15 m → 0.05  [NOVO]
//     • HAND > 15 m  → 0.00
//
//   Sem máscara urbana. Cobertura nacional.
//   Score final por pixel = tier1·(jrc>0) + tier2·(jrc=0)
//
// FONTES:
//   • HAND global: users/gena/global-hand/hand-100 (Donchyts et al. 2016, ~30m)
//   • JRC Flood Hazard: JRC/CEMS_GLOFAS/FloodHazard/v2_1 (~1km, RP100_depth)
//
// SAÍDA POR HEXÁGONO H3 res9:
//   flood_score – score contínuo médio de suscetibilidade (0–1)
//                 = indicador e2 (sem inversão)
//
// WORKFLOW (dois passos)
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

// Asset de destino do raster (Passo 1)
var ASSET_ID = "projects/ee2-linafaccin/assets/br_inund_hand_jrc_score_v2";

// Datasets fonte
var HAND_ASSET_ID = "users/gena/global-hand/hand-100";
var JRC_ASSET_ID  = "JRC/CEMS_GLOFAS/FloodHazard/v2_1";

// Circunraio do hexágono H3 resolução 9, em metros
var H3_RES9_CIRCUMRADIUS_M = 174;

// Bounding box do Brasil
var BR_BBOX = ee.Geometry.BBox(-74, -35, -28, 6);

// Lista completa de UFs do Brasil (códigos IBGE)
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
// PASSO 1 — Calcula score híbrido HAND+JRC e exporta como asset
// --------------------------------------------------------------------------
if (RUN_STEP === 1) {

  // HAND como ImageCollection → mosaico
  var hand = ee.ImageCollection(HAND_ASSET_ID).mosaic().select(0);

  // JRC Global River Flood Hazard — return period 100 anos
  // unmask(0): pixels fora da JRC viram 0 → "sem perigo modelado"
  var jrc = ee.ImageCollection(JRC_ASSET_ID).mosaic().select("RP100_depth")
              .unmask(0);

  // Máscaras dos dois tiers (binárias, mutuamente exclusivas)
  var jrc_has  = jrc.gt(0);          // tier 1
  var jrc_zero = jrc_has.not();      // tier 2

  // ----- Tier 1: JRC > 0 (alta confiança hidrológica) -----
  // Score idêntico ao v1
  var tier1 = ee.Image(0)
    .where(hand.gte(4).and(hand.lt(6)), 0.33)
    .where(hand.gte(2).and(hand.lt(4)), 0.66)
    .where(hand.gt(0).and(hand.lt(2)),  1.00);

  // ----- Tier 2: JRC = 0 (confiança topográfica apenas) -----
  // Scores reduzidos + ceiling estendido para capturar FN urbano-pluvial
  var tier2 = ee.Image(0)
    .where(hand.gte(10).and(hand.lt(15)), 0.05)
    .where(hand.gte(6).and(hand.lt(10)),  0.10)
    .where(hand.gte(4).and(hand.lt(6)),   0.20)
    .where(hand.gte(2).and(hand.lt(4)),   0.33)
    .where(hand.gt(0).and(hand.lt(2)),    0.50);

  // Score combinado: cada pixel pertence a exatamente um tier
  var flood_score_raw = tier1.multiply(jrc_has)
                             .add(tier2.multiply(jrc_zero));

  // Máscara de terra: HAND válido (> 0 exclui oceano/NoData)
  var land_mask = hand.gt(0);

  var flood_score = flood_score_raw
    .updateMask(land_mask)
    .rename("flood_score")
    .float();

  Export.image.toAsset({
    image: flood_score,
    description: "br_inund_hand_jrc_score_v2",
    assetId: ASSET_ID,
    scale: 30,
    maxPixels: 1e13,
    region: BR_BBOX,
    pyramidingPolicy: { "flood_score": "mean" }
  });

  print("=== PASSO 1 ENVIADO (v2) ===");
  print("Metodologia: HAND+JRC híbrido em dois tiers");
  print("Tier 1 (JRC>0): 0-2m=1.00 | 2-4m=0.66 | 4-6m=0.33");
  print("Tier 2 (JRC=0): 0-2m=0.50 | 2-4m=0.33 | 4-6m=0.20 | 6-10m=0.10 | 10-15m=0.05");
  print("Fonte HAND   : " + HAND_ASSET_ID);
  print("Fonte JRC    : " + JRC_ASSET_ID);
  print("Asset destino: " + ASSET_ID);
  print("");
  print("*** AGUARDE o término na aba Tasks antes de alterar RUN_STEP para 2 ***");
}

// --------------------------------------------------------------------------
// PASSO 2 — Score médio por hexágono + exportação CSV por UF
// --------------------------------------------------------------------------
if (RUN_STEP === 2) {

  var flood_asset = ee.Image(ASSET_ID);

  ufs.forEach(function(uf_code) {

    var uf_collection = table.filter(ee.Filter.eq("cd_uf", uf_code));
    var uf_buffered   = uf_collection.map(bufferHexagon);

    // unmask(0): pixels mascarados (HAND=0, calhas de rios) contribuem com 0
    // setOutputs(["flood_score"]): nomeia a propriedade de saída do reducer
    var h3_com_score = flood_asset.unmask(0).reduceRegions({
      collection: uf_buffered,
      reducer:    ee.Reducer.mean().setOutputs(["flood_score"]),
      scale:      30,
      tileScale:  16
    });

    Export.table.toDrive({
      collection:  h3_com_score,
      description: "h3_susc_inund_hand_jrc_v2_uf_" + uf_code,
      folder:      "GEE_suscetibilidade-inundacoes-hand-jrc-v2",
      fileFormat:  "CSV",
      selectors:   ["h3_id", "cd_uf", "cd_setor", "flood_score"]
    });
  });

  print("=== PASSO 2 ENVIADO (v2) ===");
  print(ufs.length + " tarefas enviadas → Drive: GEE_suscetibilidade-inundacoes-hand-jrc-v2");
  print("Colunas exportadas: h3_id | cd_uf | cd_setor | flood_score");
  print("");
  print("Próximo passo após download:");
  print("  python etl/exposure/e2_inundacoes_hand.py (ajustar paths para v2)");
  print("Depois re-rodar 06_sgb_validate_e2.py para comparar TP/FP/FN com v1.");
}
