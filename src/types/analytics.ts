export type VariableKind =
  | 'numeric_continuous'
  | 'numeric_discrete'
  | 'categorical'
  | 'binary';

export type CellValue = string | number | boolean | null;
export type DataRow = Record<string, CellValue>;

export type ModuleId =
  | 'dashboard'
  | 'cargar'
  | 'exploracion'
  | 'descriptivo'
  | 'preparacion'
  | 'regresion'
  | 'predicciones'
  | 'matriz'
  | 'metricas'
  | 'balanceo'
  | 'validacion'
  | 'simulador'
  | 'interpretacion'
  | 'informe'
  | 'ayuda';

export interface FrequencyEntry {
  value: string;
  count: number;
  relativeFreq: number;
  percentage: number;
}

export interface NumericStats {
  count: number;
  mean: number;
  median: number;
  mode: string;
  min: number;
  max: number;
  range: number;
  variance: number;
  stdDev: number;
  q1: number;
  q2: number;
  q3: number;
  iqr: number;
  outlierCount: number;
  lowerFence: number;
  upperFence: number;
}

export interface ColumnProfile {
  name: string;
  detectedKind: VariableKind;
  overrideKind?: VariableKind;
  effectiveKind: VariableKind;
  totalCount: number;
  validCount: number;
  missingCount: number;
  uniqueCount: number;
  sampleValues: string[];
  numericStats: NumericStats | null;
  frequencies: FrequencyEntry[];
  interpretation: string;
}

export interface CorrelationMatrix {
  columns: string[];
  matrix: number[][];
}

export interface DatasetState {
  fileName: string;
  sourceType: 'csv' | 'excel' | 'manual' | 'practice';
  originalRows: DataRow[];
  workingRows: DataRow[];
  columns: string[];
  profiles: ColumnProfile[];
  correlation: CorrelationMatrix;
  duplicateCount: number;
  loadedAt: string;
  sheetNames?: string[];
  selectedSheet?: string;
}

export type MissingStrategy = 'drop_rows' | 'impute_mean_mode' | 'impute_median_mode';
export type OutlierStrategy = 'keep' | 'cap_iqr';
export type BalanceMethod = 'none' | 'undersample' | 'oversample' | 'smote';

export interface PrepConfig {
  missingStrategy: MissingStrategy;
  outlierStrategy: OutlierStrategy;
  standardizeNumeric: boolean;
  factorColumns: string[];
}

export interface SplitConfig {
  trainRatio: number;
  seed: number;
  stratified: boolean;
}

export interface LogisticRegressionConfig {
  targetColumn: string;
  positiveClass: string;
  negativeClass: string;
  featureColumns: string[];
  threshold: number;
  balanceMethod: BalanceMethod;
  prepConfig: PrepConfig;
  splitConfig: SplitConfig;
}

export interface CoefficientEstimate {
  term: string;
  sourceColumn: string;
  categoryLevel?: string;
  beta: number;
  stdError: number | null;
  zValue: number | null;
  pValue: number | null;
  oddsRatio: number;
  ciLower95: number | null;
  ciUpper95: number | null;
}

export interface EncodedFeatureMeta {
  term: string;
  sourceColumn: string;
  isCategoricalDummy: boolean;
  categoryLevel?: string;
  referenceLevel?: string;
  mean: number;
  std: number;
  lowerFence?: number;
  upperFence?: number;
}

export interface PredictionRecord {
  rowIndex: number;
  actual: 0 | 1;
  actualLabel: string;
  zValue: number;
  probability: number;
  predicted: 0 | 1;
  predictedLabel: string;
  rawRow: DataRow;
}

export interface ConfusionMatrix {
  tp: number;
  tn: number;
  fp: number;
  fn: number;
  total: number;
}

export interface EvaluationMetrics {
  accuracy: number | null;
  classificationError: number | null;
  precision1: number | null;
  precision0: number | null;
  recall1: number | null;
  recall0: number | null;
  f1_1: number | null;
  f1_0: number | null;
  balancedAccuracy: number | null;
  undefinedReasons: Partial<Record<
    | 'accuracy'
    | 'classificationError'
    | 'precision1'
    | 'precision0'
    | 'recall1'
    | 'recall0'
    | 'f1_1'
    | 'f1_0'
    | 'balancedAccuracy',
    string
  >>;
}

export interface ModelResult {
  id: string;
  name: string;
  createdAt: string;
  config: LogisticRegressionConfig;
  converged: boolean;
  iterations: number;
  warning: string | null;
  error: string | null;
  coefficients: CoefficientEstimate[];
  nullDeviance: number;
  residualDeviance: number;
  aic: number;
  logLikelihood: number;
  mcfaddenR2: number;
  trainCount: number;
  testCount: number;
  balancedTrainCount: number;
  droppedTrainRows: number;
  droppedTestRows: number;
  trainClassDistBefore: { c0: number; c1: number };
  trainClassDistAfter: { c0: number; c1: number };
  testClassDist: { c0: number; c1: number };
  trainPredictions: PredictionRecord[];
  testPredictions: PredictionRecord[];
  encodedFeatures: EncodedFeatureMeta[];
  categoryLevelsByColumn: Record<string, { reference: string; levels: string[] }>;
  imputationValues: Record<string, number | string>;
}

export interface ScenarioItem {
  id: string;
  name: string;
  sector: string;
  inputs: Record<string, string | number>;
  zValue: number;
  probability: number;
  predicted: 0 | 1;
  predictedLabel: string;
  createdAt: string;
}

export interface ReportMetadata {
  projectTitle: string;
  analysisTitle: string;
  studentName: string;
  institution: string;
}
