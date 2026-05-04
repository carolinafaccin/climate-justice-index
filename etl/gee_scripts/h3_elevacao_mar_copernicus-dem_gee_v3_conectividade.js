// ==============================================================================
// E3 v3 — Suscetibilidade à Elevação do Nível do Mar
//         Método: Conectividade Costeira via cumulativeCost
//
// Problema da v2: usava apenas elevação ≤ 1m como critério, o que flagueava
// baixadas interiores (várzeas, planícies fluviais) sem conexão física com o
// oceano — especialmente em municípios grandes do Norte e Nordeste.
//
// Solução v3: o risco só se propaga por pixels fisicamente conectados ao oceano
// através de uma cadeia contínua de elevação ≤ 1m. Baixadas interiores isoladas
// por terreno mais alto (> 1m) ficam de fora, mesmo dentro de municípios costeiros.
//
// Fonte DEM  : Copernicus GLO-30 (resolução ~30m)
// Sementes   : pixels com elevação < 0 no DEM (proxy para oceano/abaixo do mar)
// Propagação : cumulativeCost pelos pixels ≤ 1m — barreiras > 1m bloqueiam
//
// WORKFLOW (dois passos — rodar em sequência):
//
//   PASSO 1 (RUN_STEP = 1):
//     Calcula o raster binário de conectividade costeira e exporta como
//     Earth Engine Image Asset. Aguarde a tarefa terminar na aba "Tasks"
//     antes de prosseguir para o Passo 2.
//
//   PASSO 2 (RUN_STEP = 2):
//     Usa o asset gerado no Passo 1 para calcular a fração de risco por
//     hexágono (Reducer.mean) e exporta os CSVs por UF para o Drive.
//     Saída compatível com e3_mar.py sem necessidade de alterações.
// ==============================================================================

var RUN_STEP = 1;  // << ALTERE PARA 1 (primeira rodada) ou 2 (após asset pronto)

// --------------------------------------------------------------------------
// CONFIGURAÇÃO
// --------------------------------------------------------------------------
var TABLE    = ee.FeatureCollection("projects/ee2-linafaccin/assets/br_h3_lat_lon_mar");
var ASSET_ID = "projects/ee2-linafaccin/assets/br_coastal_risk_slr_1m_v3";

// Circunraio do hexágono H3 resolução 9 (centro → vértice), em metros
var H3_RES9_CIRCUMRADIUS_M = 174;

// UFs com litoral
var UFS_COSTEIRAS = [
  15, 16,                               // Norte (PA, AP)
  21, 22, 23, 24, 25, 26, 27, 28, 29,  // Nordeste
  32, 33, 35,                           // Sudeste (ES, RJ, SP)
  41, 42, 43                            // Sul (PR, SC, RS)
];

// Bounding box da costa brasileira com margem generosa (inclui oceano adjacente)
var BR_COAST_BBOX = ee.Geometry.BBox(-75, -35, -25, 10);

// --------------------------------------------------------------------------
// DEM Copernicus GLO-30
// --------------------------------------------------------------------------
var dem_raw = ee.ImageCollection("COPERNICUS/DEM/GLO30")
  .select('DEM')
  .mosaic();

// No Copernicus GLO-30, pixels sobre o oceano são NoData (sem valor).
// Preenchemos com -10 para que sirvam como sementes negativas na análise.
// Pixels de terra com elevação genuinamente negativa (praticamente inexistentes
// no Brasil) também entrariam como sementes — comportamento desejável.
var dem = dem_raw.unmask(-10);

// --------------------------------------------------------------------------
// PASSO 1 — Gerar e exportar o raster de conectividade costeira como asset
// --------------------------------------------------------------------------
if (RUN_STEP === 1) {

  // Sementes oceânicas: pixels com elevação < 0 após unmask(-10)
  //   → inclui pixels de oceano (NoData → -10) e eventuais áreas sub-zero
  var ocean_seeds = dem.lt(0).selfMask();

  // Imagem de custo para cumulativeCost:
  //   elevação ≤ 1m → custo 1 (corredor passável — risco SLR)
  //   elevação > 1m → custo 1e9 (barreira — bloqueia a propagação)
  var cost_img = dem.gt(1).multiply(1e9).add(1).toFloat();

  // Custo acumulado mínimo a partir das sementes oceânicas.
  //
  // maxDistance = 200 km cobre todos os cenários relevantes, incluindo estuários
  // extensos como a foz do Amazonas, onde influência de maré atinge > 100 km.
  //
  // Interpretação do custo acumulado:
  //   - Pixel conectado ao oceano por N pixels ≤ 1m: custo ≈ N (ex: 100 km / 30m ≈ 3333)
  //   - Pixel separado do oceano por qualquer barreira > 1m: custo ≥ 1e9
  //   - Limiar de 1e6 separa conectados (custo << 1e6) de bloqueados (custo >> 1e6)
  var connected_cost = cost_img.cumulativeCost({
    source: ocean_seeds,
    maxDistance: 200000,    // 200 km (geodeticDistance: true → unidade = metros)
    geodeticDistance: true
  });

  // Risco costeiro binário:
  //   1 = conectado ao oceano por corredor contínuo de elevação ≤ 1m
  //   0 = não conectado (ou além de 200 km do oceano)
  //
  // unmask(1e12): pixels sem custo calculado (NoData, além do maxDistance ou sem
  // caminho possível) recebem valor > 1e6 → corretamente classificados como sem risco
  var risco_coastal = connected_cost
    .unmask(1e12)
    .lt(1e6)
    .rename('risco_slr')
    .toByte();

  Export.image.toAsset({
    image: risco_coastal,
    description: 'br_coastal_risk_slr_1m_v3_asset',
    assetId: ASSET_ID,
    scale: 30,
    maxPixels: 1e13,
    region: BR_COAST_BBOX,
    pyramidingPolicy: { risco_slr: 'mode' }
  });

  print('=== PASSO 1 ENVIADO ===');
  print('Tarefa: exportação do raster de conectividade costeira (30m, binário).');
  print('Aguarde o término na aba "Tasks" antes de alterar RUN_STEP para 2.');
  print('Asset destino: ' + ASSET_ID);
}

// --------------------------------------------------------------------------
// PASSO 2 — Calcular fração de risco por hexágono e exportar CSVs por UF
// --------------------------------------------------------------------------
if (RUN_STEP === 2) {

  // Carrega o raster de conectividade gerado no Passo 1
  var risco_coastal = ee.Image(ASSET_ID);

  // Buffer de 174m converte centróide do hexágono em aproximação da área real
  function bufferHexagon(feature) {
    return feature.buffer(H3_RES9_CIRCUMRADIUS_M);
  }

  UFS_COSTEIRAS.forEach(function(uf_code) {

    var uf_costeira = TABLE
      .filter(ee.Filter.eq('cd_uf', uf_code))
      .filter(ee.Filter.eq('municipios_defrontantes_com_o_mar', true));

    var uf_buffered = uf_costeira.map(bufferHexagon);

    // Reducer.mean() sobre raster binário → fração da área do hexágono em risco
    // (ex: 0.4 = ~40% da área conectada ao risco costeiro de 1m SLR)
    var h3_com_risco = risco_coastal.reduceRegions({
      collection: uf_buffered,
      reducer: ee.Reducer.mean().setOutputs(['risco_slr']),
      scale: 30,
      tileScale: 16
    });

    Export.table.toDrive({
      collection: h3_com_risco,
      description: 'h3_susc_mar_1m_coastal_uf_' + uf_code,
      folder: 'GEE_suscetibilidade-elevacao-nivel-mar-v3',
      fileFormat: 'CSV',
      selectors: ['h3_id', 'cd_mun', 'cd_setor', 'qtd_dom', 'risco_slr']
    });
  });

  print('=== PASSO 2 ENVIADO ===');
  print(UFS_COSTEIRAS.length + ' tarefas de exportação CSV enviadas para o Drive.');
  print('Pasta: GEE_suscetibilidade-elevacao-nivel-mar-v3');
  print('Compatível com e3_mar.py sem alterações (mesma estrutura de colunas).');
}
