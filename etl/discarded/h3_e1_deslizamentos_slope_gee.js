// ==============================================================================
// E1 — Suscetibilidade a Deslizamentos de Terra (Slope-based, Camada 1)
// Fonte: Copernicus GLO-30 DEM (mesma fonte do indicador E3)
//
// Método: Classificação de declividade (% slope) conforme metodologia
//         de mapeamento de áreas de risco (referência: estudos CPRM/IPT,
//         aplicada no Vale do Taquari):
//
//           Alta:  35 – 60 % slope        → suscetibilidade alta
//           Média: 25 – 35 %  e  60 – 75 % → suscetibilidade média
//           Baixa: 10 – 25 %  e  > 75 %    → suscetibilidade baixa
//           Sem:   0  – 10 %              → sem suscetibilidade
//
//         Nota: slopes muito íngremes (> 75 %) tendem a corresponder a
//         afloramentos rochosos com pouco depósito instável — por isso
//         recebem classe "Baixa" e não "Alta".
//
// Saída por hexágono H3 res9:
//   alta       – fração da área do hexágono com declividade Alta  (0–1)
//   media      – fração da área do hexágono com declividade Média (0–1)
//   alta_media – fração da área com declividade Alta OU Média — indicador e1
//
// VALIDAÇÃO (opcional, RUN_STEP = 3):
//   Comparação com NASA LHASA v2 (static susceptibility). Requer upload
//   prévio do raster LHASA como asset no GEE — ver instruções ao final.
//
// WORKFLOW (dois passos obrigatórios + um opcional):
//   PASSO 1 (RUN_STEP = 1): gera raster classificado e exporta como asset
//                            *** AGUARDE TERMINAR antes de rodar o passo 2 ***
//   PASSO 2 (RUN_STEP = 2): exporta CSVs por UF para o Google Drive
//   PASSO 3 (RUN_STEP = 3): exporta tabela de comparação com NASA LHASA (opcional)
// ==============================================================================

var RUN_STEP = 1;  // << ALTERE: 1 → 2 → 3 (opcional)

// --------------------------------------------------------------------------
// PARÂMETROS DE CLASSIFICAÇÃO (limites em % de declividade)
// Ajuste aqui caso queira testar outros limiares
// --------------------------------------------------------------------------
var ALTA_MIN   = 35;  var ALTA_MAX   = 60;  // faixa Alta
var MEDIA_MIN1 = 25;  var MEDIA_MAX1 = 35;  // faixa Média — inferior
var MEDIA_MIN2 = 60;  var MEDIA_MAX2 = 75;  // faixa Média — superior

// --------------------------------------------------------------------------
// CONFIGURAÇÃO — Assets e constantes
// --------------------------------------------------------------------------

// Tabela de hexágonos habitados do Brasil (todos os H3 res9 com ≥ 1 domicílio)
// Mesma tabela usada pelo indicador E4 (calor extremo)
var table = ee.FeatureCollection("projects/ee2-linafaccin/assets/br_h3_lat_lon");

// Asset de destino do raster classificado (Passo 1)
var ASSET_ID = "projects/ee2-linafaccin/assets/br_slope_suscept_desliz_v1";

// Asset do NASA LHASA — preencha após fazer upload (ver instruções abaixo)
var LHASA_ASSET_ID = "projects/ee2-linafaccin/assets/nasa_lhasa_susceptibility";

// Circunraio do hexágono H3 resolução 9 (centro → vértice), em metros
// Mesmo valor usado nos indicadores E3 e E4
var H3_RES9_CIRCUMRADIUS_M = 174;

// Bounding box do Brasil (continental + ilhas oceânicas)
var BR_BBOX = ee.Geometry.BBox(-74, -35, -28, 6);

// Lista completa de UFs do Brasil
var ufs = [
  11, 12, 13, 14, 15, 16, 17,          // Norte
  21, 22, 23, 24, 25, 26, 27, 28, 29,  // Nordeste
  31, 32, 33, 35,                       // Sudeste
  41, 42, 43,                           // Sul
  50, 51, 52, 53                        // Centro-Oeste
];

// --------------------------------------------------------------------------
// DEM Copernicus GLO-30 (30 m de resolução)
// Mesma coleção e mosaico usados pelo indicador E3
// --------------------------------------------------------------------------
var dem = ee.ImageCollection("COPERNICUS/DEM/GLO30")
  .select('DEM')
  .mosaic();

// Máscara de terra: no GLO-30, pixels válidos têm elevação >= -100 m (inclui oceano raso)
// Para excluir oceano profundo: dem >= 0 (acima do nível do mar)
// Para máxima cobertura: usamos dem > -500
var land_mask = dem.gt(-500);

// Função auxiliar — declarada fora dos blocos if
function bufferHexagon(feature) {
  return feature.buffer(H3_RES9_CIRCUMRADIUS_M);
}

// --------------------------------------------------------------------------
// PASSO 1 — Calcular declividade, classificar e exportar raster como asset
// --------------------------------------------------------------------------
if (RUN_STEP === 1) {

  // Cálculo manual de slope usando gradiente (melhor compatibilidade com GLO-30)
  // slope_deg = arctan(sqrt(dx² + dy²))
  var slope_x = dem.derivative().select([0]);
  var slope_y = dem.derivative().select([1]);
  var slope_mag = slope_x.pow(2).add(slope_y.pow(2)).sqrt();
  var slope_deg = slope_mag.atan().multiply(180 / Math.PI);

  // Conversão para % slope: tan(graus × π/180) × 100
  var slope_pct = slope_deg
    .multiply(Math.PI / 180)
    .tan()
    .multiply(100)
    .rename('slope_pct');

  // Bandas binárias (1 = pertence à classe, 0 = não pertence)
  var alta = slope_pct
    .gte(ALTA_MIN).and(slope_pct.lt(ALTA_MAX))
    .rename('alta');

  var media = slope_pct.gte(MEDIA_MIN1).and(slope_pct.lt(MEDIA_MAX1))
    .or(slope_pct.gte(MEDIA_MIN2).and(slope_pct.lt(MEDIA_MAX2)))
    .rename('media');

  // Indicador principal e1: Alta OU Média susceptibilidade
  var alta_media = alta.or(media).rename('alta_media');

  // Imagem multi-banda, mascarada para terra
  var suscept = alta
    .addBands(media)
    .addBands(alta_media)
    .updateMask(land_mask)
    .toByte();

  Export.image.toAsset({
    image: suscept,
    description: 'br_slope_suscept_desliz_v1',
    assetId: ASSET_ID,
    scale: 30,
    maxPixels: 1e13,
    region: BR_BBOX,
    // 'mean' preserva a capacidade de inspecionar frações em escalas mais grosseiras
    pyramidingPolicy: { 'alta': 'mean', 'media': 'mean', 'alta_media': 'mean' }
  });

  print('=== PASSO 1 ENVIADO ===');
  print('Fonte: Copernicus GLO-30, 30 m, slope em % (ee.Terrain.slope → tan → ×100)');
  print('Limiares aplicados:');
  print('  Alta:  ' + ALTA_MIN  + '–' + ALTA_MAX  + ' %');
  print('  Média: ' + MEDIA_MIN1 + '–' + MEDIA_MAX1 + ' %  e  ' + MEDIA_MIN2 + '–' + MEDIA_MAX2 + ' %');
  print('Asset destino: ' + ASSET_ID);
  print('');
  print('*** AGUARDE o término na aba Tasks antes de alterar RUN_STEP para 2 ***');
}

// --------------------------------------------------------------------------
// PASSO 2 — Fração de suscetibilidade por hexágono + exportação CSV por UF
// --------------------------------------------------------------------------
if (RUN_STEP === 2) {

  var suscept_asset = ee.Image(ASSET_ID);

  // Buffer de 174 m no centróide do hexágono → aproxima a área real do hexágono
  // Mesmo padrão dos indicadores E3 e E4

  ufs.forEach(function(uf_code) {

    var uf_collection = table.filter(ee.Filter.eq('cd_uf', uf_code));
    var uf_buffered   = uf_collection.map(bufferHexagon);

    // Reducer.mean() em banda binária → fração da área do hexágono em cada classe
    // Exemplo: alta_media = 0.40 → 40 % da área do hexágono tem declividade Alta ou Média
    var h3_com_suscept = suscept_asset.reduceRegions({
      collection: uf_buffered,
      reducer: ee.Reducer.mean(),
      scale: 30,
      tileScale: 16
    });

    Export.table.toDrive({
      collection: h3_com_suscept,
      description: 'h3_susc_desliz_slope_v1_uf_' + uf_code,
      folder: 'GEE_suscetibilidade-deslizamentos-slope-v1',
      fileFormat: 'CSV',
      selectors: ['h3_id', 'cd_uf', 'cd_setor', 'alta', 'media', 'alta_media']
    });
  });

  print('=== PASSO 2 ENVIADO ===');
  print(ufs.length + ' tarefas enviadas → Google Drive: GEE_suscetibilidade-deslizamentos-slope-v1');
  print('Colunas exportadas: h3_id | cd_uf | cd_setor | alta | media | alta_media');
  print('Pronto para processar com etl/exposure/e1_deslizamentos_slope.py');
}

// --------------------------------------------------------------------------
// PASSO 3 (OPCIONAL) — Validação contra NASA LHASA
// --------------------------------------------------------------------------
//
// SOBRE O ARQUIVO nasa_lhasa_susceptibility.tif:
//   Fonte: Stanley & Kirschbaum (2017) — Global Landslide Susceptibility Map
//   Resolução: ~1 km (30 arc-seconds)
//   Valores da banda única (inteiros 0–5):
//     0 = NoData / oceano (MASCARAR antes de usar)
//     1 = Very Low susceptibility   (fuzzy < 0.11)
//     2 = Low                       (fuzzy 0.11–0.49)
//     3 = Moderate                  (fuzzy 0.49–0.67)
//     4 = High                      (fuzzy 0.67–0.75)
//     5 = Very High                 (fuzzy > 0.75)
//
// PRÉ-REQUISITOS:
//   Asset já carregado como: projects/ee2-linafaccin/assets/nasa_lhasa_susceptibility
//   Atualize LHASA_ASSET_ID (no topo deste script) se necessário.
//
// O output é uma tabela CSV com:
//   slope_alta_media  – fração da área do hexágono em classe Alta ou Média (0–1)
//   lhasa_mean        – valor médio de susceptibilidade LHASA no hexágono (1–5)
//   lhasa_high_frac   – fração da área com LHASA >= 4 (High ou Very High)
// --------------------------------------------------------------------------
if (RUN_STEP === 3) {

  var suscept_slope = ee.Image(ASSET_ID);

  // Carrega LHASA e mascara pixels com valor 0 (oceano / NoData)
  // Sem essa máscara, pixels oceânicos zerariam a média em hexágonos costeiros
  var lhasa_raw = ee.Image(LHASA_ASSET_ID);
  var lhasa = lhasa_raw.updateMask(lhasa_raw.gt(0));

  // Fração de área com LHASA High ou Very High (valores 4 e 5)
  // Equivalente ao que fazemos com slope: limiar de risco relevante
  var lhasa_high = lhasa.gte(4).rename('lhasa_high_frac');

  // Para um teste rápido, mantenha .limit(10000)
  // Remova .limit() para processar todos os hexágonos habitados do Brasil
  var sample_hexagons = table.limit(10000);
  var sample_buffered = sample_hexagons.map(bufferHexagon);

  // Combina as três bandas de interesse
  // scale: 30 — usa a resolução nativa do slope; o GEE reamostrar o LHASA de ~1km para
  // 30m (nearest-neighbor), o que é aceitável para análise de correlação em nível de hexágono
  var compare = suscept_slope.select('alta_media').rename('slope_alta_media')
    .addBands(lhasa.rename('lhasa_mean'))
    .addBands(lhasa_high)
    .reduceRegions({
      collection: sample_buffered,
      reducer: ee.Reducer.mean(),
      scale: 30,
      tileScale: 8
    });

  Export.table.toDrive({
    collection: compare,
    description: 'h3_validacao_lhasa_vs_slope_v1',
    folder: 'GEE_suscetibilidade-deslizamentos-slope-v1',
    fileFormat: 'CSV',
    selectors: ['h3_id', 'cd_uf', 'cd_setor', 'slope_alta_media', 'lhasa_mean', 'lhasa_high_frac']
  });

  print('=== PASSO 3 ENVIADO ===');
  print('Comparação slope × NASA LHASA (Stanley & Kirschbaum 2017)');
  print('Colunas: slope_alta_media | lhasa_mean (1–5) | lhasa_high_frac (frac ≥4)');
  print('Pasta: GEE_suscetibilidade-deslizamentos-slope-v1');
}
