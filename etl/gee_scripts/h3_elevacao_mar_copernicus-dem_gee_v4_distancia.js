// ==============================================================================
// E3 v4 — Suscetibilidade à Elevação do Nível do Mar
//         Método: Distância ao Oceano + Limiar de Elevação
//
// Por que a v3 (cumulativeCost) foi descartada:
//   cumulativeCost é um algoritmo global de caminho mínimo (Dijkstra).
//   Para o Brasil inteiro a 30m, o custo computacional é proibitivo no GEE.
//
// Abordagem v4 (fastDistanceTransform):
//   1. Pixels oceânicos são identificados via DEM: NoData no Copernicus GLO-30
//      (preenchidos com -10 por unmask) → elevação < 0 = oceano
//   2. fastDistanceTransform calcula a distância euclidiana de cada pixel
//      terrestre ao pixel oceânico mais próximo (O(n), muito mais rápido)
//   3. Risco SLR = elevação ≤ ELEV_MAX_M  AND  distância ao oceano ≤ DIST_MAX_KM
//
// Por que isso resolve o problema:
//   Baixadas interiores (várzeas, planícies fluviais) em municípios costeiros
//   grandes (ex: PA, MA) ficam a dezenas de km do oceano → eliminadas pelo
//   filtro de distância. A faixa costeira legítima (0–10 km do mar) cobre
//   praticamente todos os hexágonos em risco real de 1m SLR.
//
// Limitação conhecida:
//   Distância euclidiana ≠ conectividade hidrológica. Hexágonos atrás de uma
//   barreira topográfica mas dentro de DIST_MAX_KM podem ser incluídos.
//   Na prática isso é raro: terraços costeiros > 1m separam fisicamente as
//   baixadas interiores do oceano. Caso necessário, subir linha de costa IBGE
//   como asset e substituir a máscara oceânica do DEM por ela.
//
// WORKFLOW:
//   PASSO 1 (RUN_STEP = 1): exporta raster de risco como asset (aguarde terminar)
//   PASSO 2 (RUN_STEP = 2): exporta CSVs por UF para o Drive
//   Saída compatível com e3_mar.py sem alterações.
// ==============================================================================

var RUN_STEP = 1;  // << ALTERE PARA 1 (primeiro) ou 2 (após asset pronto)

// --------------------------------------------------------------------------
// PARÂMETROS — ajuste aqui se necessário
// --------------------------------------------------------------------------
var ELEV_MAX_M  = 1;    // elevação máxima considerada em risco (m)
var DIST_MAX_KM = 10;   // distância máxima ao oceano (km)
                        // 10 km captura estuários e planícies costeiras legítimas
                        // Aumente para 15-20 km se alguma faixa costeira conhecida
                        // ficar de fora; reduza para 5 km para resultado mais estrito

// --------------------------------------------------------------------------
// CONFIGURAÇÃO
// --------------------------------------------------------------------------
var TABLE    = ee.FeatureCollection("projects/ee2-linafaccin/assets/br_h3_lat_lon_mar");
var ASSET_ID = "projects/ee2-linafaccin/assets/br_coastal_risk_slr_1m_v4";

var H3_RES9_CIRCUMRADIUS_M = 174;

var UFS_COSTEIRAS = [
  15, 16,                               // Norte (PA, AP)
  21, 22, 23, 24, 25, 26, 27, 28, 29,  // Nordeste
  32, 33, 35,                           // Sudeste (ES, RJ, SP)
  41, 42, 43                            // Sul (PR, SC, RS)
];

var BR_COAST_BBOX = ee.Geometry.BBox(-75, -35, -25, 10);

// --------------------------------------------------------------------------
// DEM Copernicus GLO-30
// --------------------------------------------------------------------------
var dem_raw = ee.ImageCollection("COPERNICUS/DEM/GLO30")
  .select('DEM')
  .mosaic();

// No Copernicus GLO-30, pixels oceânicos são NoData. unmask(-10) os preenche
// com -10 para que dem.lt(0) os identifique como oceano.
var dem = dem_raw.unmask(-10);

// --------------------------------------------------------------------------
// PASSO 1 — Gerar e exportar o raster de risco costeiro como asset
// --------------------------------------------------------------------------
if (RUN_STEP === 1) {

  // Máscara oceânica: 1 onde oceano (elevação < 0), 0 onde terra
  var ocean = dem.lt(0);

  // fastDistanceTransform retorna a distância ao quadrado em pixels
  // até o pixel oceânico mais próximo.
  //
  // maxDistance em pixels = DIST_MAX_KM * 1000m / 30m/pixel + margem de 100px
  // A margem garante que pixels além de DIST_MAX_KM recebam distância > DIST_MAX_KM
  // e sejam corretamente excluídos no filtro abaixo.
  var DIST_MAX_PX = Math.ceil(DIST_MAX_KM * 1000 / 30) + 100;

  var dist_sq_px = ocean.fastDistanceTransform(DIST_MAX_PX, 'pixels');

  // Converte distância em metros: sqrt(pixels²) × 30 m/pixel
  var dist_m = dist_sq_px.sqrt().multiply(30);

  // Risco SLR:
  //   terra (≥ 0m, não oceano)  AND  elevação ≤ 1m  AND  distância ≤ DIST_MAX_KM
  var risco = dem.gte(0)
    .and(dem.lte(ELEV_MAX_M))
    .and(dist_m.lte(DIST_MAX_KM * 1000))
    .rename('risco_slr')
    .toByte();

  Export.image.toAsset({
    image: risco,
    description: 'br_coastal_risk_slr_1m_v4_dist' + DIST_MAX_KM + 'km',
    assetId: ASSET_ID,
    scale: 30,
    maxPixels: 1e13,
    region: BR_COAST_BBOX,
    pyramidingPolicy: { risco_slr: 'mode' }
  });

  print('=== PASSO 1 ENVIADO ===');
  print('Critério: elevação ≤ ' + ELEV_MAX_M + 'm  AND  distância ao oceano ≤ ' + DIST_MAX_KM + ' km');
  print('Asset destino: ' + ASSET_ID);
  print('Aguarde o término na aba Tasks antes de alterar RUN_STEP para 2.');
}

// --------------------------------------------------------------------------
// PASSO 2 — Fração de risco por hexágono e exportação CSV por UF
// --------------------------------------------------------------------------
if (RUN_STEP === 2) {

  var risco_coastal = ee.Image(ASSET_ID);

  function bufferHexagon(feature) {
    return feature.buffer(H3_RES9_CIRCUMRADIUS_M);
  }

  UFS_COSTEIRAS.forEach(function(uf_code) {

    var uf_costeira = TABLE
      .filter(ee.Filter.eq('cd_uf', uf_code))
      .filter(ee.Filter.eq('municipios_defrontantes_com_o_mar', true));

    var uf_buffered = uf_costeira.map(bufferHexagon);

    // Reducer.mean() sobre raster binário → fração da área do hexágono em risco
    // (ex: 0.6 = 60% da área do hexágono satisfaz os critérios de risco SLR)
    var h3_com_risco = risco_coastal.reduceRegions({
      collection: uf_buffered,
      reducer: ee.Reducer.mean().setOutputs(['risco_slr']),
      scale: 30,
      tileScale: 16
    });

    Export.table.toDrive({
      collection: h3_com_risco,
      description: 'h3_susc_mar_1m_dist' + DIST_MAX_KM + 'km_uf_' + uf_code,
      folder: 'GEE_suscetibilidade-elevacao-nivel-mar-v4',
      fileFormat: 'CSV',
      selectors: ['h3_id', 'cd_mun', 'cd_setor', 'qtd_dom', 'risco_slr']
    });
  });

  print('=== PASSO 2 ENVIADO ===');
  print(UFS_COSTEIRAS.length + ' tarefas → pasta GEE_suscetibilidade-elevacao-nivel-mar-v4');
  print('Compatível com e3_mar.py sem alterações.');
}
