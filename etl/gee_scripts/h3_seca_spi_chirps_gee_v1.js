// ==============================================================================
// E6 — Suscetibilidade a Secas Meteorológicas (SPI-12 via CHIRPS)
// Fonte: CHIRPS Pentad — UCSB-CHG, calibrado com estações in situ (1981–presente)
//
// Método:
//   1. Agregação da precipitação CHIRPS (pentads de 5 dias) para mensal (1991–2024)
//   2. Janela móvel de 12 meses (precipitação acumulada anual rolante)
//   3. Climatologia 1991–2020 (padrão OMM): média e desvio-padrão por mês
//      do ano, calculados pixel a pixel sobre o acumulado de 12 meses
//   4. SPI ≈ (P_acumulada_atual − μ_climatologia_do_mês) / σ_climatologia_do_mês
//      (aproximação por z-score — ver nota metodológica em CONCEITO_seca_SPI.md)
//   5. Indicador final por hexágono: fração de meses entre 2015 e 2024 em que
//      o SPI esteve em categoria de seca severa (SPI ≤ −1.5)
//
// Categorias OMM/McKee et al. (1993):
//   SPI ≥  2.0   extremamente úmido
//   1.5 – 2.0    muito úmido
//   1.0 – 1.5    moderadamente úmido
//  −1.0 – 1.0    normal
//  −1.5 – −1.0   moderadamente seco
//  −2.0 – −1.5   severamente seco       ← gatilho do indicador
//   ≤ −2.0       extremamente seco
//
// Resolução nativa CHIRPS: ~5 km (0.05°). Cada hexágono H3 res9 (~174 m) é
// muito menor que um pixel CHIRPS, então é amostrado com buffer largo
// (3500 m, cobrindo metade da diagonal de um pixel CHIRPS) na escala nativa,
// sem reamostragem — o que torna o reduceRegions tratável para o Brasil
// inteiro. Conceitualmente: o hexágono herda o SPI do pixel CHIRPS sob ele,
// com suavização nas bordas entre pixels.
// ==============================================================================

// --- 1. CONFIGURAÇÃO ---
var table = ee.FeatureCollection("projects/ee2-linafaccin/assets/br_h3_lat_lon");

// Para CHIRPS (5 km), usamos buffer dimensionado pelo dataset, não pelo H3.
// 5000×√2/2 ≈ 3536 m é a distância máxima de qualquer ponto ao centro de
// pixel mais próximo em uma grade de 5 km; 3500 m garante intersecção em
// praticamente 100 % dos casos, sem precisar reamostrar a imagem.
var CHIRPS_SAMPLE_BUFFER_M = 3500;

// Períodos
var CLIM_START_YEAR    = 1991;  // climatologia OMM 1991–2020
var CLIM_END_YEAR      = 2020;
var ASSESS_START_YEAR  = 2015;  // janela de avaliação consistente com e4 (calor)
var ASSESS_END_YEAR    = 2024;

// Parâmetros SPI
var WINDOW_MONTHS         = 12;     // SPI-12 (seca hidrológica anual)
var SPI_SEVERE_THRESHOLD  = -1.5;   // gatilho de seca severa (OMM)

// UFs
var ufs = [
  11, 12, 13, 14, 15, 16, 17,
  21, 22, 23, 24, 25, 26, 27, 28, 29,
  31, 32, 33, 35,
  41, 42, 43,
  50, 51, 52, 53
];

// --- 2. PRECIPITAÇÃO CHIRPS ---
// PENTAD 0.05° (~5 km), calibrado com estações INMET no Brasil.
// PENTAD agrega 5 dias por imagem → ~6 imagens/mês contra ~30 do DAILY,
// reduzindo em ~5× o volume processado. Diferença em SPI-12 é desprezível
// (acumulado anual, erro de borda de pentad <1% da soma mensal).
var chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD").select('precipitation');

// --- 3. AGREGAÇÃO MENSAL ---
// Gera lista de primeiros-dias-do-mês de jan/1991 a dez/2024 (408 meses).
var firstDate = ee.Date.fromYMD(CLIM_START_YEAR, 1, 1);
var nMonths   = (ASSESS_END_YEAR - CLIM_START_YEAR + 1) * 12;

var monthStarts = ee.List.sequence(0, nMonths - 1).map(function(i) {
  return firstDate.advance(ee.Number(i), 'month');
});

// Soma diária CHIRPS dentro de cada mês.
function monthlySum(dateAny) {
  var start = ee.Date(dateAny);
  var end   = start.advance(1, 'month');
  return chirps
    .filterDate(start, end)
    .sum()
    .set('system:time_start', start.millis())
    .set('year',  start.get('year'))
    .set('month', start.get('month'));
}

var monthlyPrecip = ee.ImageCollection(monthStarts.map(monthlySum));

// --- 4. ACUMULADO DE 12 MESES (JANELA MÓVEL) ---
// Para o mês M, soma a precipitação dos 12 meses anteriores (M-11 a M).
// Começa em dez/1991 (primeiro mês com janela completa).
var rollingStarts = monthStarts.slice(WINDOW_MONTHS - 1);

function rollingSum(targetDate) {
  var anchor = ee.Date(targetDate);
  var end    = anchor.advance(1, 'month');
  var start  = end.advance(-WINDOW_MONTHS, 'month');
  return monthlyPrecip
    .filterDate(start, end)
    .sum()
    .set('system:time_start', anchor.millis())
    .set('year',  anchor.get('year'))
    .set('month', anchor.get('month'));
}

var rolling12 = ee.ImageCollection(rollingStarts.map(rollingSum));

// --- 5. CLIMATOLOGIA POR MÊS DO ANO (1991–2020) ---
// Para cada mês do ano (1–12), calcula μ e σ do acumulado de 12 meses
// sobre todos os anos da climatologia. Isso preserva a sazonalidade:
// um SPI de janeiro é comparado a janeiros históricos, fevereiro a
// fevereiros, etc.
var climSet = rolling12.filter(
  ee.Filter.calendarRange(CLIM_START_YEAR, CLIM_END_YEAR, 'year')
);

var climByMonth = ee.List.sequence(1, 12).map(function(m) {
  var subset = climSet.filter(ee.Filter.eq('month', m));
  var meanImg = subset.mean().rename('mean');
  var stdImg  = subset.reduce(ee.Reducer.stdDev()).rename('std');
  return meanImg.addBands(stdImg).set('month', m);
});

// --- 6. SPI APROXIMADO NO PERÍODO DE AVALIAÇÃO (2015–2024) ---
var assessSet = rolling12.filter(
  ee.Filter.calendarRange(ASSESS_START_YEAR, ASSESS_END_YEAR, 'year')
);

var spi = assessSet.map(function(img) {
  var m = ee.Number(img.get('month'));
  var clim = ee.Image(ee.List(climByMonth).get(m.subtract(1)));
  var mean = clim.select('mean');
  var std  = clim.select('std');

  // Máscara contra divisão por zero (pixels sem variabilidade histórica)
  var safeStd = std.where(std.lte(0), 1e6);

  return img.subtract(mean)
    .divide(safeStd)
    .updateMask(std.gt(0))
    .rename('spi')
    .copyProperties(img, ['system:time_start', 'year', 'month']);
});

// --- 7. INDICADOR FINAL POR PIXEL ---
// spi_severe_freq: fração de meses em que o SPI ficou ≤ −1.5 (severo)
// spi_min        : SPI mais negativo do período (pior episódio)
var severeMask = spi.map(function(img) {
  return img.lte(SPI_SEVERE_THRESHOLD).unmask(0).rename('severe');
});

var nAssess = ee.Number(spi.size());
var spi_severe_freq = severeMask.sum().divide(nAssess).rename('spi_severe_freq');
var spi_min         = spi.min().rename('spi_min');

var indicator = spi_severe_freq.addBands(spi_min);

// --- 8. FUNÇÃO: buffer dimensionado pelo CHIRPS ---
function bufferForChirps(feature) {
  return feature.buffer(CHIRPS_SAMPLE_BUFFER_M);
}

// --- 9. AGREGAÇÃO POR HEXÁGONO E EXPORTAÇÃO POR UF ---
ufs.forEach(function(uf_code) {

  var uf_collection = table.filter(ee.Filter.eq('cd_uf', uf_code));
  var uf_buffered   = uf_collection.map(bufferForChirps);

  var h3_with_spi = indicator.reduceRegions({
    collection: uf_buffered,
    reducer: ee.Reducer.mean(),  // multi-banda: GEE usa os nomes das bandas automaticamente
    // scale 5000: resolução nativa do CHIRPS. Não reamostra → mantém
    // performance tratável para UFs grandes. Combinado com buffer de
    // 3500 m, cada hexágono intersecta ≥ 1 pixel CHIRPS.
    scale:    5000,
    tileScale: 16
  });

  Export.table.toDrive({
    collection: h3_with_spi,
    description: 'h3_seca_spi_chirps_v1_uf_' + uf_code,
    folder:     'GEE_seca_spi_chirps_v1',
    fileFormat: 'CSV',
    selectors:  ['h3_id', 'cd_setor', 'qtd_dom', 'spi_severe_freq', 'spi_min']
  });
});

print('✓ E6 SPI-12 / CHIRPS — tarefas enviadas.');
print('  Climatologia : ' + CLIM_START_YEAR + '–' + CLIM_END_YEAR + ' (30 anos, padrão OMM)');
print('  Avaliação    : ' + ASSESS_START_YEAR + '–' + ASSESS_END_YEAR + ' (consistente com e4)');
print('  Janela       : SPI-' + WINDOW_MONTHS + ' (seca hidrológica anual)');
print('  Gatilho      : SPI ≤ ' + SPI_SEVERE_THRESHOLD + ' (categoria "severamente seco")');
print('  Saída        : spi_severe_freq ∈ [0,1] ; spi_min ∈ (~−4, 0]');
