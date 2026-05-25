// ==============================================================================
// E2 — Diagnóstico de HAND nos Falsos Negativos
//
// OBJETIVO
//   Para os hexágonos identificados como falsos negativos pelo
//   06_sgb_validate_e2.py (SGB diz alta suscetibilidade mas E2=0), extrair
//   estatísticas de HAND bruto e da máscara JRC. Isso permite definir um
//   novo teto HAND informado por dados, em vez do atual fixo em 6 m.
//
// FLUXO
//   1. Rodar 06_sgb_validate_e2.py → gera diagnostic_e2_fn_hexagons_<ts>.csv
//   2. Upload do CSV no GEE como Table asset (Assets → New → CSV upload).
//      Não precisa de geometria — usamos o asset br_h3_lat_lon para isso.
//      Conferir que h3_id, macro e cd_estado são importados como string.
//   3. Atualizar FN_ASSET_ID abaixo com o caminho do asset gerado.
//   4. Executar este script — envia 5 tasks (uma por macrorregião) para o Drive.
//   5. Rodar 07_sgb_analyse_fn_hand.py sobre os CSVs baixados.
//
// FONTES (mesmas do v1)
//   • HAND global: users/gena/global-hand/hand-100  (Donchyts et al. 2016, ~30m)
//   • JRC Flood Hazard: JRC/CEMS_GLOFAS/FloodHazard/v2_1 — banda RP100_depth
//
// SAÍDA POR HEXÁGONO
//   hand_mean, hand_min, hand_p25, hand_p50, hand_p75, hand_p90, hand_max
//   jrc_rp100_mean  — útil para distinguir FN por "HAND > 6m" (teto)
//                     de FN por "JRC=0" (problema de cobertura do JRC)
// ==============================================================================

// --------------------------------------------------------------------------
// CONFIGURAÇÃO
// --------------------------------------------------------------------------

// Asset com os falsos negativos exportados pelo 06_sgb_validate_e2.py
// AJUSTE este caminho após upload do CSV no GEE.
var FN_ASSET_ID = "projects/ee2-linafaccin/assets/fn_e2_hexagons";

// Tabela de hexágonos habitados do Brasil (já contém geometria)
var H3_TABLE_ID = "projects/ee2-linafaccin/assets/br_h3_lat_lon";

// Datasets
var HAND_ASSET_ID = "users/gena/global-hand/hand-100";
var JRC_ASSET_ID  = "JRC/CEMS_GLOFAS/FloodHazard/v2_1";

// Circunraio do hexágono H3 res9, em metros
var H3_RES9_CIRCUMRADIUS_M = 174;

// Pasta de saída no Google Drive
var DRIVE_FOLDER = "GEE_fn_e2_hand_diagnostic";

// Macrorregiões a processar (5 tasks)
var MACROS = ["S", "SE", "CO", "NE", "N"];

// --------------------------------------------------------------------------
// PREPARAÇÃO DOS RASTERS
// --------------------------------------------------------------------------

// HAND bruto (metros acima do canal de drenagem mais próximo)
var hand = ee.ImageCollection(HAND_ASSET_ID).mosaic().select(0).rename("hand");

// JRC RP100 — profundidade de inundação modelada (T=100 anos), em metros
// unmask(0): pixels fora do JRC viram 0 → equivalente a "sem perigo modelado"
var jrc = ee.ImageCollection(JRC_ASSET_ID).mosaic().select("RP100_depth")
            .unmask(0).rename("jrc_rp100");

// Imagem combinada para reduzir em uma única passada
var combined = hand.addBands(jrc);

// --------------------------------------------------------------------------
// CARREGAMENTO DOS FN E JOIN COM H3 BASE
// --------------------------------------------------------------------------

var fn_table = ee.FeatureCollection(FN_ASSET_ID);
var h3_base  = ee.FeatureCollection(H3_TABLE_ID);

// Lista de h3_ids dos FN (server-side)
var fn_ids = fn_table.aggregate_array("h3_id");

// Recupera geometrias dos hexágonos FN a partir da tabela base
// Filtra h3_base por h3_id ∈ FN, mantém centróides + cd_uf
var h3_fn_geom = h3_base.filter(ee.Filter.inList("h3_id", fn_ids));

// Faz join para trazer propriedades macro/cd_estado/sgb_max_class dos FN
// para a FeatureCollection com geometria
var join = ee.Join.saveFirst({matchKey: "fn_props"});
var filter = ee.Filter.equals({leftField: "h3_id", rightField: "h3_id"});

var h3_fn = ee.FeatureCollection(join.apply(h3_fn_geom, fn_table, filter))
  .map(function(f) {
    var props = ee.Feature(f.get("fn_props"));
    return f.copyProperties(props, ["macro", "cd_estado", "sgb_max_class",
                                    "sgb_alta_mta_frac", "sgb_coverage_frac",
                                    "e2_inu_abs"]);
  });

// --------------------------------------------------------------------------
// REDUTOR: estatísticas de HAND + média JRC
// --------------------------------------------------------------------------

// Cobre: mean, min, max, percentis [25, 50, 75, 90]
// Saídas separadas para hand e jrc_rp100 (combinedReducer com sharedInputs=false)
var handReducer = ee.Reducer.mean()
  .combine({reducer2: ee.Reducer.minMax(),                  sharedInputs: true})
  .combine({reducer2: ee.Reducer.percentile([25, 50, 75, 90]), sharedInputs: true});

var jrcReducer  = ee.Reducer.mean();

var reducer = handReducer.combine({reducer2: jrcReducer, sharedInputs: false});

// --------------------------------------------------------------------------
// EXPORT POR MACRORREGIÃO
// --------------------------------------------------------------------------

var bufferHexagon = function(feature) {
  return feature.buffer(H3_RES9_CIRCUMRADIUS_M);
};

MACROS.forEach(function(macro) {
  var sub = h3_fn.filter(ee.Filter.eq("macro", macro)).map(bufferHexagon);

  var stats = combined.reduceRegions({
    collection: sub,
    reducer:    reducer,
    scale:      30,
    tileScale:  16
  });

  Export.table.toDrive({
    collection:  stats,
    description: "fn_hand_diag_macro_" + macro,
    folder:      DRIVE_FOLDER,
    fileFormat:  "CSV",
    selectors: [
      "h3_id", "cd_estado", "macro", "sgb_max_class",
      "sgb_alta_mta_frac", "sgb_coverage_frac", "e2_inu_abs",
      "hand_mean", "hand_min", "hand_max",
      "hand_p25", "hand_p50", "hand_p75", "hand_p90",
      "jrc_rp100_mean"
    ]
  });
});

print("=== TASKS ENVIADAS ===");
print("FN asset    : " + FN_ASSET_ID);
print("Macros      : " + MACROS.join(", "));
print("Drive folder: " + DRIVE_FOLDER);
print("");
print("Após terminar todas as tasks, baixe os CSVs e rode:");
print("  python etl/exposure/sgb/07_sgb_analyse_fn_hand.py");
