// 09 ACCURACY ASSESSMENT

// SUPERVISED CLASSIFICATION - Geometry Imports
// Buat sampel kelas 'urban', 'vegetation', dan 'water' dalam point
// Buat region dalam polygon

// Muat Landsat 8 surface reflectance data
var l8sr = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR');

// Fungsi untuk cloud mask dari band Fmask data Landsat 8 SR.
function maskL8sr(image) {
    // Bit 3 dan 5 masing-masing adalah cloud shadow dan cloud.
    var cloudShadowBitMask = ee.Number(2).pow(3).int();
    var cloudsBitMask = ee.Number(2).pow(5).int();

    // Dapatkan band pixel QA.
    var qa = image.select('pixel_qa');

    // Kedua 'flag' harus diatur ke 'nol', yang menunjukkan kondisi yang jelas.
    var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
        .and(qa.bitwiseAnd(cloudsBitMask).eq(0));

    // Kembalikan nilai citra yang di-mask, diskalakan ke [0, 1].
    return image.updateMask(mask).divide(10000);
}

// Memetakan fungsi lebih dari satu tahun data dan mengambil median.
var image = l8sr.filterDate('2016-01-01', '2016-12-31')
    .map(maskL8sr)
    .median();

// Tampilkan hasil image median.
Map.addLayer(image, { bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3 }, 'image');

// Membuat feature class untuk training
var newfc = urban.merge(vegetation).merge(water);
//print(newfc); // new feature class
//Map.centerObject(newfc,11);
Map.centerObject(region, 11);

// Membuat feature class untuk test/validasi
var newfc_validation = urban_validation.merge(vegetation_validation).merge(water_validation);

var bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'];

var sample = image.select(bands).sampleRegions({
    collection: newfc,
    properties: ['landcover'],
    scale: 30
});
// Print(sample)

var sample_validation = image.select(bands).sampleRegions({
    collection: newfc_validation,
    properties: ['landcover'],
    scale: 30
});
// Print(sample_validation)

// Membagi sampel untuk training 80% dan validation 20%
sample = sample.randomColumn({ seed: 1 });
var training = sample.filter(ee.Filter.lt('random', 0.8)); // 80%
var validation = sample.filter(ee.Filter.gte('random', 0.8)); // 20%

var classifier = ee.Classifier.minimumDistance().train({ // coba ganti dengan smileCart, libsvm, gmoMaxEnt
    features: training,
    classProperty: 'landcover',
    inputProperties: bands
});
print(classifier.explain());

var classified = image.select(bands).classify(classifier);
Map.addLayer(classified, { min: 0, max: 2, palette: ['red', 'green', 'blue'] }, 'CART');

var trainAccuracy = classifier.confusionMatrix().accuracy();
print('trainAccuracy', trainAccuracy); // 1.0

var testAccuracy = validation
    .classify(classifier)
    .errorMatrix('landcover', 'classification')
    .accuracy();
print('testAccuracy', testAccuracy); // 1.0

var testAccuracy2 = sample_validation // Menggunakan sample baru di luar sample training
    .classify(classifier)
    .errorMatrix('landcover', 'classification')
    .accuracy();
print('testAccuracy2', testAccuracy2); // 1.0

// Hasil CHART:
var options = {
    lineWidth: 1,
    pointSize: 2,
    hAxis: { title: 'Classes' },
    vAxis: { title: 'Area m^2' },
    title: 'Area by class',
    series: {
        0: { color: 'red' },
        1: { color: 'green' },
        2: { color: 'blue' }
    }
};

var areaChart = ui.Chart.image.byClass({
        image: ee.Image.pixelArea().addBands(classified),
        classBand: 'classification',
        region: region,
        scale: 30,
        reducer: ee.Reducer.sum()
    }).setOptions(options)
    .setSeriesNames(['urban', 'vegetation', 'water']);
print(areaChart);