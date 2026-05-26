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
//      O CSV já contém colunas latitude e longitude (centróide H3).
//   2. Upload do CSV no GEE:
//      Assets → New → CSV upload → selecionar arquivo
//      Em "Advanced options": X column = longitude, Y column = latitude
//      Isso cria uma FeatureCollection de pontos (um por hexágono FN).
//   3. Atualizar FN_ASSET_ID abaixo com o caminho do asset gerado.
//   4. Executar este script — envia 5 tasks (uma por macrorregião) para o Drive.
//   5. Baixar CSVs do Drive → pasta local configurada em DATA_DIR
//   6. Rodar 07_sgb_analyse_fn_hand.py sobre os CSVs baixados.
//
// FONTES (mesmas do v1)
//   • HAND global: users/gena/global-hand/hand-100  (Donchyts et al. 2016, ~30m)
//   • JRC Flood Hazard: JRC/CEMS_GLOFAS/FloodHazard/v2_1 — banda RP100_depth
//
// SAÍDA POR HEXÁGONO
//   hand_mean, hand_min, hand_p25, hand_p50, hand_p75, hand_p90, hand_max
//   jrc_rp100_mean  — distingue FN por "HAND > 6m" (teto) de FN por "JRC=0"
// ==============================================================================

// --------------------------------------------------------------------------
// CONFIGURAÇÃO
// --------------------------------------------------------------------------

// Asset criado a partir do CSV diagnostic_e2_fn_hexagons_<ts>.csv
// AJUSTE este caminho após o upload no GEE.
var FN_ASSET_ID = "projects/ee2-linafaccin/assets/e2_fn_hexagons";

// Datasets
var HAND_ASSET_ID = "users/gena/global-hand/hand-100";
var JRC_ASSET_ID  = "JRC/CEMS_GLOFAS/FloodHazard/v2_1";

// Circunraio do hexágono H3 res9, em metros (aproxima a área do hexágono)
var H3_RES9_CIRCUMRADIUS_M = 174;

// Pasta de saída no Google Drive
var DRIVE_FOLDER = "GEE_fn_e2_hand_diagnostic";

// Macrorregiões a processar.
// Só S e SE têm FN suficientes — CO/NE/N caem fora do filtro de cobertura
// SGB ≥ 50% no script 06, então o asset chega vazio para esses macros.
// Se rodar o script 06 com cobertura completa no futuro, reincluir os outros.
var MACROS = ["S", "SE"];

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
// CARREGAMENTO DOS FN
// O CSV já tem lat/lon → GEE cria geometria de ponto por hexágono no upload.
// Aplicamos buffer de 174 m para aproximar a área real do hexágono H3 res9.
// --------------------------------------------------------------------------

var bufferHexagon = function(feature) {
  return feature.buffer(H3_RES9_CIRCUMRADIUS_M);
};

var fn_table = ee.FeatureCollection(FN_ASSET_ID).map(bufferHexagon);

// --------------------------------------------------------------------------
// DIAGNÓSTICO — confirme no Console antes de rodar as tasks
// Se algum print mostrar 0, o filtro por macro não está casando.
// Verifique no print de "first feature" qual é o valor real da coluna macro.
// --------------------------------------------------------------------------
print("FN asset total features:", fn_table.size());
print("FN first feature properties:", fn_table.first());
print("Macros encontrados no asset:",
      fn_table.aggregate_histogram("macro"));

// --------------------------------------------------------------------------
// REDUTOR: estatísticas de HAND + média JRC
// --------------------------------------------------------------------------

// Redutor único aplicado a todas as bandas da imagem combinada.
// O GEE prefixa automaticamente os outputs com o nome da banda:
//   hand_mean, hand_min, hand_max, hand_p25 … hand_p90
//   jrc_rp100_mean, jrc_rp100_min, … (descartados pelos selectors do Export)
var reducer = ee.Reducer.mean()
  .combine({reducer2: ee.Reducer.minMax(),                     sharedInputs: true})
  .combine({reducer2: ee.Reducer.percentile([25, 50, 75, 90]), sharedInputs: true});

// --------------------------------------------------------------------------
// EXPORT POR MACRORREGIÃO (5 tasks paralelas)
// --------------------------------------------------------------------------

MACROS.forEach(function(macro) {
  var sub = fn_table.filter(ee.Filter.eq("macro", macro));
  print("Macro " + macro + " — features:", sub.size());

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
