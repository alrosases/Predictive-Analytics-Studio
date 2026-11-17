import * as XLSX from 'xlsx';
import type {
  ConfusionMatrix,
  DataRow,
  DatasetState,
  EvaluationMetrics,
  ModelResult,
} from '../types/analytics';
import { computeConfusionMatrix, computeEvaluationMetrics } from './logisticRegression';

export function downloadCSVRows(rows: DataRow[], columns: string[], filename: string): void {
  const escapeCell = (val: unknown): string => {
    if (val === null || val === undefined) return '';
    const str = String(val);
    if (/[",\n\r;]/.test(str)) {
      return `"${str.replace(/"/g, '""')}"`;
    }
    return str;
  };

  const lines = [
    columns.map(escapeCell).join(','),
    ...rows.map((r) => columns.map((c) => escapeCell(r[c])).join(',')),
  ];

  // UTF-8 BOM for clean Excel display of Spanish accents
  const blob = new Blob(['\uFEFF' + lines.join('\r\n')], {
    type: 'text/csv;charset=utf-8;',
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename.endsWith('.csv') ? filename : `${filename}.csv`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export function exportDatasetCopy(dataset: DatasetState, format: 'csv' | 'xlsx'): void {
  const baseName = dataset.fileName.replace(/\.[^.]+$/, '') + '_copia';
  if (format === 'csv') {
    downloadCSVRows(dataset.workingRows, dataset.columns, `${baseName}.csv`);
    return;
  }
  const ws = XLSX.utils.json_to_sheet(dataset.workingRows, { header: dataset.columns });
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, 'Datos');
  XLSX.writeFile(wb, `${baseName}.xlsx`);
}

export function exportPredictionsCSV(model: ModelResult, split: 'test' | 'train' | 'all'): void {
  const records =
    split === 'train'
      ? model.trainPredictions.map((r) => ({ ...r, particion: 'Entrenamiento' }))
      : split === 'test'
      ? model.testPredictions.map((r) => ({ ...r, particion: 'Prueba' }))
      : [
          ...model.trainPredictions.map((r) => ({ ...r, particion: 'Entrenamiento' })),
          ...model.testPredictions.map((r) => ({ ...r, particion: 'Prueba' })),
        ];

  const rows: DataRow[] = records.map((r) => ({
    fila_original: r.rowIndex,
    particion: r.particion,
    ...r.rawRow,
    valor_real: r.actualLabel,
    valor_real_binario: r.actual,
    valor_Z: Number(r.zValue.toFixed(6)),
    probabilidad_estimada: Number(r.probability.toFixed(6)),
    probabilidad_pct: `${(r.probability * 100).toFixed(2)}%`,
    prediccion_binaria: r.probability >= model.config.threshold ? 1 : 0,
    prediccion_clase:
      r.probability >= model.config.threshold
        ? model.config.positiveClass
        : model.config.negativeClass,
  }));

  const cols = rows.length > 0 ? Object.keys(rows[0]) : ['fila_original', 'probabilidad_estimada'];
  downloadCSVRows(rows, cols, `predicciones_${model.name.replace(/\s+/g, '_').toLowerCase()}.csv`);
}

function formatMetricSheetRow(
  name: string,
  trainVal: number | null,
  testVal: number | null,
  trainReason?: string,
  testReason?: string
) {
  return {
    Métrica: name,
    Entrenamiento_Decimal: trainVal !== null ? Number(trainVal.toFixed(4)) : 'No definida',
    Entrenamiento_Pct: trainVal !== null ? `${(trainVal * 100).toFixed(2)}%` : trainReason || 'No definida',
    Prueba_Decimal: testVal !== null ? Number(testVal.toFixed(4)) : 'No definida',
    Prueba_Pct: testVal !== null ? `${(testVal * 100).toFixed(2)}%` : testReason || 'No definida',
  };
}

export function exportResultsExcel(dataset: DatasetState, model: ModelResult): void {
  const wb = XLSX.utils.book_new();

  // Sheet 1: Descriptive Stats
  const descRows = dataset.profiles.map((p) => ({
    Variable: p.name,
    Tipo: p.effectiveKind,
    Validos: p.validCount,
    Faltantes: p.missingCount,
    Unicos: p.uniqueCount,
    Media: p.numericStats ? Number(p.numericStats.mean.toFixed(4)) : 'N/A',
    Mediana: p.numericStats ? Number(p.numericStats.median.toFixed(4)) : 'N/A',
    Desv_Estandar: p.numericStats ? Number(p.numericStats.stdDev.toFixed(4)) : 'N/A',
    Minimo: p.numericStats ? Number(p.numericStats.min.toFixed(4)) : 'N/A',
    Maximo: p.numericStats ? Number(p.numericStats.max.toFixed(4)) : 'N/A',
    Atipicos_IQR: p.numericStats ? p.numericStats.outlierCount : 'N/A',
  }));
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(descRows), 'Descriptivo');

  // Sheet 2: Coefficients
  const coefRows = model.coefficients.map((c) => ({
    Termino: c.term,
    Coeficiente_Beta: Number(c.beta.toFixed(6)),
    Error_Estandar: c.stdError !== null ? Number(c.stdError.toFixed(6)) : 'N/D',
    Estadistico_Z: c.zValue !== null ? Number(c.zValue.toFixed(4)) : 'N/D',
    P_Valor: c.pValue !== null ? Number(c.pValue.toFixed(6)) : 'N/D',
    Odds_Ratio: Number(c.oddsRatio.toFixed(6)),
    IC_95_Inferior: c.ciLower95 !== null ? Number(c.ciLower95.toFixed(4)) : 'N/D',
    IC_95_Superior: c.ciUpper95 !== null ? Number(c.ciUpper95.toFixed(4)) : 'N/D',
  }));
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(coefRows), 'Coeficientes_GLM');

  // Sheet 3: Metrics Train vs Test
  const cmTrain: ConfusionMatrix = computeConfusionMatrix(
    model.trainPredictions,
    model.config.threshold
  );
  const cmTest: ConfusionMatrix = computeConfusionMatrix(
    model.testPredictions,
    model.config.threshold
  );
  const mTrain: EvaluationMetrics = computeEvaluationMetrics(cmTrain);
  const mTest: EvaluationMetrics = computeEvaluationMetrics(cmTest);

  const metricRows = [
    formatMetricSheetRow('Accuracy', mTrain.accuracy, mTest.accuracy, mTrain.undefinedReasons.accuracy, mTest.undefinedReasons.accuracy),
    formatMetricSheetRow('Error de Clasificación', mTrain.classificationError, mTest.classificationError, mTrain.undefinedReasons.classificationError, mTest.undefinedReasons.classificationError),
    formatMetricSheetRow('Precision (Clase 1)', mTrain.precision1, mTest.precision1, mTrain.undefinedReasons.precision1, mTest.undefinedReasons.precision1),
    formatMetricSheetRow('Precision (Clase 0)', mTrain.precision0, mTest.precision0, mTrain.undefinedReasons.precision0, mTest.undefinedReasons.precision0),
    formatMetricSheetRow('Recall (Clase 1)', mTrain.recall1, mTest.recall1, mTrain.undefinedReasons.recall1, mTest.undefinedReasons.recall1),
    formatMetricSheetRow('Recall (Clase 0)', mTrain.recall0, mTest.recall0, mTrain.undefinedReasons.recall0, mTest.undefinedReasons.recall0),
    formatMetricSheetRow('F1-Score (Clase 1)', mTrain.f1_1, mTest.f1_1, mTrain.undefinedReasons.f1_1, mTest.undefinedReasons.f1_1),
    formatMetricSheetRow('F1-Score (Clase 0)', mTrain.f1_0, mTest.f1_0, mTrain.undefinedReasons.f1_0, mTest.undefinedReasons.f1_0),
    formatMetricSheetRow('Balanced Accuracy', mTrain.balancedAccuracy, mTest.balancedAccuracy, mTrain.undefinedReasons.balancedAccuracy, mTest.undefinedReasons.balancedAccuracy),
  ];
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(metricRows), 'Metricas');

  // Sheet 4: Confusion Matrices
  const cmRows = [
    { Conjunto: 'Entrenamiento', TP: cmTrain.tp, TN: cmTrain.tn, FP: cmTrain.fp, FN: cmTrain.fn, Total: cmTrain.total },
    { Conjunto: 'Prueba', TP: cmTest.tp, TN: cmTest.tn, FP: cmTest.fp, FN: cmTest.fn, Total: cmTest.total },
  ];
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(cmRows), 'Matriz_Confusion');

  // Sheet 5: Test Predictions
  const predRows = model.testPredictions.map((r) => ({
    Fila: r.rowIndex,
    Valor_Real: r.actualLabel,
    Z: Number(r.zValue.toFixed(6)),
    Probabilidad: Number(r.probability.toFixed(6)),
    Clase_Predicha:
      r.probability >= model.config.threshold
        ? model.config.positiveClass
        : model.config.negativeClass,
  }));
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(predRows), 'Predicciones_Prueba');

  XLSX.writeFile(wb, `resultados_${model.name.replace(/\s+/g, '_').toLowerCase()}.xlsx`);
}
