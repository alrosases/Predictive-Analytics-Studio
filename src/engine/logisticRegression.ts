import type {
  BalanceMethod,
  CoefficientEstimate,
  ColumnProfile,
  ConfusionMatrix,
  DataRow,
  EncodedFeatureMeta,
  EvaluationMetrics,
  LogisticRegressionConfig,
  ModelResult,
  PredictionRecord,
} from '../types/analytics';
import { isMissingValue, tryParseNumeric } from './statistics';

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffleWithSeed<T>(arr: T[], rand: () => number): T[] {
  const copy = [...arr];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = copy[i];
    copy[i] = copy[j];
    copy[j] = tmp;
  }
  return copy;
}

export function sigmoid(z: number): number {
  if (z >= 35) return 1 - 1e-15;
  if (z <= -35) return 1e-15;
  return 1 / (1 + Math.exp(-z));
}

// Complementary error function for accurate normal two-tailed p-values
function erfc(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r =
    t *
    Math.exp(
      -z * z -
        1.26551223 +
        t *
          (1.00002368 +
            t *
              (0.37409196 +
                t *
                  (0.09678418 +
                    t *
                      (-0.18628806 +
                        t *
                          (0.27886807 +
                            t *
                              (-1.13520398 +
                                t *
                                  (1.48851587 +
                                    t * (-0.82215223 + t * 0.17087277))))))))
    );
  return x >= 0 ? r : 2 - r;
}

function twoTailedPValue(z: number): number {
  if (!Number.isFinite(z)) return 1;
  return Math.min(1, Math.max(0, erfc(Math.abs(z) / Math.SQRT2)));
}

// Gauss-Jordan matrix inversion with partial pivoting; returns null if singular
function invertSymmetricMatrix(mat: number[][]): number[][] | null {
  const n = mat.length;
  const aug: number[][] = mat.map((row, i) => {
    const id = Array(n).fill(0);
    id[i] = 1;
    return [...row, ...id];
  });

  for (let col = 0; col < n; col++) {
    let pivotRow = col;
    let maxVal = Math.abs(aug[col][col]);
    for (let r = col + 1; r < n; r++) {
      const val = Math.abs(aug[r][col]);
      if (val > maxVal) {
        maxVal = val;
        pivotRow = r;
      }
    }

    if (maxVal < 1e-11 || !Number.isFinite(maxVal)) {
      return null;
    }

    if (pivotRow !== col) {
      const tmp = aug[col];
      aug[col] = aug[pivotRow];
      aug[pivotRow] = tmp;
    }

    const pivot = aug[col][col];
    for (let c = col; c < 2 * n; c++) {
      aug[col][c] /= pivot;
    }

    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = aug[r][col];
      if (factor === 0) continue;
      for (let c = col; c < 2 * n; c++) {
        aug[r][c] -= factor * aug[col][c];
      }
    }
  }

  return aug.map((row) => row.slice(n));
}

function solveLinearSystem(A: number[][], b: number[]): number[] | null {
  const inv = invertSymmetricMatrix(A);
  if (!inv) return null;
  const n = b.length;
  const x = Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) {
      sum += inv[i][j] * b[j];
    }
    x[i] = sum;
  }
  return x;
}

function quantileSorted(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0];
  const idx = (sorted.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  const frac = idx - lo;
  return sorted[lo] * (1 - frac) + sorted[hi] * frac;
}

function applyClassBalance(
  X: number[][],
  y: (0 | 1)[],
  isDummyCol: boolean[],
  method: BalanceMethod,
  rand: () => number
): { X: number[][]; y: (0 | 1)[] } {
  if (method === 'none') {
    return { X, y };
  }

  const idx0: number[] = [];
  const idx1: number[] = [];
  for (let i = 0; i < y.length; i++) {
    if (y[i] === 0) idx0.push(i);
    else idx1.push(i);
  }

  if (idx0.length === 0 || idx1.length === 0) {
    return { X, y };
  }

  if (idx0.length === idx1.length) {
    return { X, y };
  }

  const minorityIdx = idx0.length < idx1.length ? idx0 : idx1;
  const majorityIdx = idx0.length < idx1.length ? idx1 : idx0;
  const minorityClass: 0 | 1 = idx0.length < idx1.length ? 0 : 1;
  const majorityClass: 0 | 1 = minorityClass === 0 ? 1 : 0;

  if (method === 'undersample') {
    const sampledMajority = shuffleWithSeed(majorityIdx, rand).slice(0, minorityIdx.length);
    const combined = shuffleWithSeed([...minorityIdx, ...sampledMajority], rand);
    return {
      X: combined.map((i) => X[i]),
      y: combined.map((i) => y[i]),
    };
  }

  if (method === 'oversample') {
    const outIndices = [...majorityIdx, ...minorityIdx];
    const needed = majorityIdx.length - minorityIdx.length;
    for (let k = 0; k < needed; k++) {
      const pick = minorityIdx[Math.floor(rand() * minorityIdx.length)];
      outIndices.push(pick);
    }
    const shuffled = shuffleWithSeed(outIndices, rand);
    return {
      X: shuffled.map((i) => X[i]),
      y: shuffled.map((i) => y[i]),
    };
  }

  // SMOTE on minority class in standardized numeric space
  const p = X[0]?.length || 0;
  if (minorityIdx.length < 2 || p === 0) {
    // Fallback to oversampling if minority < 2
    return applyClassBalance(X, y, isDummyCol, 'oversample', rand);
  }

  // Compute scales across training set for Euclidean distance
  const colMeans = Array(p).fill(0);
  const colStds = Array(p).fill(1);
  for (let j = 0; j < p; j++) {
    let s = 0;
    for (let i = 0; i < X.length; i++) s += X[i][j];
    const mu = s / X.length;
    colMeans[j] = mu;
    let sq = 0;
    for (let i = 0; i < X.length; i++) sq += (X[i][j] - mu) ** 2;
    const sd = Math.sqrt(sq / Math.max(1, X.length - 1));
    colStds[j] = sd > 1e-9 ? sd : 1;
  }

  const minorityVectors = minorityIdx.map((idx) => X[idx]);
  const kNeighbors = Math.min(5, minorityVectors.length - 1);
  const syntheticX: number[][] = [];
  const syntheticY: (0 | 1)[] = [];
  const needed = majorityIdx.length - minorityIdx.length;

  for (let s = 0; s < needed; s++) {
    const baseIdx = Math.floor(rand() * minorityVectors.length);
    const baseVec = minorityVectors[baseIdx];

    // Find k nearest neighbors
    const dists: { idx: number; dist: number }[] = [];
    for (let other = 0; other < minorityVectors.length; other++) {
      if (other === baseIdx) continue;
      let d2 = 0;
      for (let j = 0; j < p; j++) {
        const diff = (baseVec[j] - minorityVectors[other][j]) / colStds[j];
        d2 += diff * diff;
      }
      dists.push({ idx: other, dist: d2 });
    }
    dists.sort((a, b) => a.dist - b.dist);
    const nnPick = dists[Math.floor(rand() * kNeighbors)].idx;
    const neighborVec = minorityVectors[nnPick];

    const alpha = rand();
    const synth = Array(p).fill(0);
    for (let j = 0; j < p; j++) {
      const val = baseVec[j] + alpha * (neighborVec[j] - baseVec[j]);
      synth[j] = isDummyCol[j] ? (val >= 0.5 ? 1 : 0) : val;
    }
    syntheticX.push(synth);
    syntheticY.push(minorityClass);
  }

  const allX = [...majorityIdx.map((i) => X[i]), ...minorityVectors, ...syntheticX];
  const allY: (0 | 1)[] = [
    ...majorityIdx.map(() => majorityClass),
    ...minorityVectors.map(() => minorityClass),
    ...syntheticY,
  ];
  const order = shuffleWithSeed(
    Array.from({ length: allX.length }, (_, i) => i),
    rand
  );

  return {
    X: order.map((i) => allX[i]),
    y: order.map((i) => allY[i]),
  };
}

export function trainLogisticRegression(
  rows: DataRow[],
  profiles: ColumnProfile[],
  config: LogisticRegressionConfig,
  modelName = 'Modelo Logístico'
): ModelResult {
  const id = `mdl_${Date.now()}_${Math.floor(Math.random() * 10000)}`;
  const createdAt = new Date().toISOString();

  const makeFailure = (errorMsg: string): ModelResult => ({
    id,
    name: modelName,
    createdAt,
    config,
    converged: false,
    iterations: 0,
    warning: null,
    error: errorMsg,
    coefficients: [],
    nullDeviance: 0,
    residualDeviance: 0,
    aic: 0,
    logLikelihood: 0,
    mcfaddenR2: 0,
    trainCount: 0,
    testCount: 0,
    balancedTrainCount: 0,
    droppedTrainRows: 0,
    droppedTestRows: 0,
    trainClassDistBefore: { c0: 0, c1: 0 },
    trainClassDistAfter: { c0: 0, c1: 0 },
    testClassDist: { c0: 0, c1: 0 },
    trainPredictions: [],
    testPredictions: [],
    encodedFeatures: [],
    categoryLevelsByColumn: {},
    imputationValues: {},
  });

  if (!config.targetColumn) {
    return makeFailure('Debe seleccionar una variable dependiente.');
  }
  if (config.featureColumns.length === 0) {
    return makeFailure('Debe seleccionar al menos una variable independiente predictora.');
  }
  if (config.featureColumns.includes(config.targetColumn)) {
    return makeFailure('La variable dependiente no puede usarse como predictora.');
  }
  if (!config.positiveClass || !config.negativeClass || config.positiveClass === config.negativeClass) {
    return makeFailure('Debe definir dos categorías distintas para el evento positivo (1) y negativo (0).');
  }

  // Filter rows having valid binary target (0 or 1)
  interface IndexedRow {
    rowIndex: number;
    row: DataRow;
    y: 0 | 1;
    yLabel: string;
  }

  const validTargetRows: IndexedRow[] = [];
  for (let i = 0; i < rows.length; i++) {
    const rawTarget = rows[i][config.targetColumn];
    if (isMissingValue(rawTarget)) continue;
    const strTarget = String(rawTarget).trim();
    if (strTarget === config.positiveClass) {
      validTargetRows.push({ rowIndex: i + 1, row: rows[i], y: 1, yLabel: strTarget });
    } else if (strTarget === config.negativeClass) {
      validTargetRows.push({ rowIndex: i + 1, row: rows[i], y: 0, yLabel: strTarget });
    }
  }

  if (validTargetRows.length < 12) {
    return makeFailure(
      `Muestra insuficiente (${validTargetRows.length} registros válidos). Se requieren al menos 12 observaciones con las categorías definidas.`
    );
  }

  const rand = mulberry32(config.splitConfig.seed || 42);
  const ratio = Math.min(0.9, Math.max(0.5, config.splitConfig.trainRatio));

  let rawTrain: IndexedRow[] = [];
  let rawTest: IndexedRow[] = [];

  if (config.splitConfig.stratified) {
    const c0 = shuffleWithSeed(
      validTargetRows.filter((r) => r.y === 0),
      rand
    );
    const c1 = shuffleWithSeed(
      validTargetRows.filter((r) => r.y === 1),
      rand
    );
    if (c0.length < 3 || c1.length < 3) {
      return makeFailure(
        `Una de las categorías tiene muy pocos registros (Clase 0: ${c0.length}, Clase 1: ${c1.length}). Se requieren al menos 3 por categoría para dividir estratificadamente.`
      );
    }
    const split0 = Math.max(1, Math.min(c0.length - 1, Math.round(c0.length * ratio)));
    const split1 = Math.max(1, Math.min(c1.length - 1, Math.round(c1.length * ratio)));
    rawTrain = shuffleWithSeed([...c0.slice(0, split0), ...c1.slice(0, split1)], rand);
    rawTest = shuffleWithSeed([...c0.slice(split0), ...c1.slice(split1)], rand);
  } else {
    const allShuffled = shuffleWithSeed(validTargetRows, rand);
    const splitIdx = Math.max(2, Math.min(allShuffled.length - 2, Math.round(allShuffled.length * ratio)));
    rawTrain = allShuffled.slice(0, splitIdx);
    rawTest = allShuffled.slice(splitIdx);
  }

  // Determine column modes (numeric vs categorical/factor) strictly from profile + prepConfig.factorColumns
  const colIsCategorical = new Map<string, boolean>();
  for (const col of config.featureColumns) {
    const prof = profiles.find((p) => p.name === col);
    const forcedFactor = config.prepConfig.factorColumns.includes(col);
    const isNumeric =
      !forcedFactor &&
      prof !== undefined &&
      (prof.effectiveKind === 'numeric_continuous' ||
        prof.effectiveKind === 'numeric_discrete' ||
        (prof.effectiveKind === 'binary' && prof.numericStats !== null));
    colIsCategorical.set(col, !isNumeric);
  }

  // Fit imputation values, outlier fences, and category levels ONLY on rawTrain (no leakage!)
  const imputationValues: Record<string, number | string> = {};
  const outlierFences: Record<string, { lower: number; upper: number }> = {};
  const categoryLevelsByColumn: Record<string, { reference: string; levels: string[] }> = {};

  for (const col of config.featureColumns) {
    const isCat = colIsCategorical.get(col) ?? false;
    if (isCat) {
      const counts = new Map<string, number>();
      for (const item of rawTrain) {
        const val = item.row[col];
        if (!isMissingValue(val)) {
          const s = String(val).trim();
          counts.set(s, (counts.get(s) || 0) + 1);
        }
      }
      const sortedLevels = Array.from(counts.entries())
        .sort((a, b) => b[1] - a[1])
        .map(([lvl]) => lvl);

      if (sortedLevels.length < 2) {
        return makeFailure(
          `La variable categórica "${col}" no tiene variabilidad suficiente en el conjunto de entrenamiento (${sortedLevels.length} categoría).`
        );
      }
      imputationValues[col] = sortedLevels[0];
      // Alphabetical or most frequent reference: use most frequent as reference level to maximize numerical stability
      categoryLevelsByColumn[col] = {
        reference: sortedLevels[0],
        levels: sortedLevels,
      };
    } else {
      const nums: number[] = [];
      for (const item of rawTrain) {
        const n = tryParseNumeric(item.row[col]);
        if (n !== null) nums.push(n);
      }
      if (nums.length < 3) {
        return makeFailure(`La variable numérica "${col}" no tiene suficientes valores válidos en entrenamiento.`);
      }
      const sorted = [...nums].sort((a, b) => a - b);
      const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
      const median = quantileSorted(sorted, 0.5);
      const q1 = quantileSorted(sorted, 0.25);
      const q3 = quantileSorted(sorted, 0.75);
      const iqr = q3 - q1;

      imputationValues[col] =
        config.prepConfig.missingStrategy === 'impute_mean_mode' ? mean : median;
      outlierFences[col] = {
        lower: q1 - 1.5 * iqr,
        upper: q3 + 1.5 * iqr,
      };
    }
  }

  // Build encoded feature definitions
  const featureSpecs: {
    term: string;
    sourceColumn: string;
    isCategoricalDummy: boolean;
    categoryLevel?: string;
    referenceLevel?: string;
  }[] = [];

  for (const col of config.featureColumns) {
    const isCat = colIsCategorical.get(col) ?? false;
    if (isCat) {
      const { reference, levels } = categoryLevelsByColumn[col];
      for (const lvl of levels) {
        if (lvl === reference) continue;
        featureSpecs.push({
          term: `${col}[${lvl}]`,
          sourceColumn: col,
          isCategoricalDummy: true,
          categoryLevel: lvl,
          referenceLevel: reference,
        });
      }
    } else {
      featureSpecs.push({
        term: col,
        sourceColumn: col,
        isCategoricalDummy: false,
      });
    }
  }

  const encodeSet = (
    dataset: IndexedRow[]
  ): { X: number[][]; y: (0 | 1)[]; kept: IndexedRow[]; dropped: number } => {
    const X: number[][] = [];
    const y: (0 | 1)[] = [];
    const kept: IndexedRow[] = [];
    let dropped = 0;

    for (const item of dataset) {
      let hasMissing = false;
      const cleanValues: Record<string, number | string> = {};

      for (const col of config.featureColumns) {
        const isCat = colIsCategorical.get(col) ?? false;
        const rawVal = item.row[col];
        if (isCat) {
          if (isMissingValue(rawVal)) {
            if (config.prepConfig.missingStrategy === 'drop_rows') {
              hasMissing = true;
              break;
            }
            cleanValues[col] = String(imputationValues[col]);
          } else {
            cleanValues[col] = String(rawVal).trim();
          }
        } else {
          const num = tryParseNumeric(rawVal);
          if (num === null) {
            if (config.prepConfig.missingStrategy === 'drop_rows') {
              hasMissing = true;
              break;
            }
            cleanValues[col] = Number(imputationValues[col]);
          } else {
            let v = num;
            if (config.prepConfig.outlierStrategy === 'cap_iqr') {
              const fence = outlierFences[col];
              if (v < fence.lower) v = fence.lower;
              if (v > fence.upper) v = fence.upper;
            }
            cleanValues[col] = v;
          }
        }
      }

      if (hasMissing) {
        dropped++;
        continue;
      }

      const vec: number[] = [];
      for (const spec of featureSpecs) {
        if (spec.isCategoricalDummy) {
          vec.push(cleanValues[spec.sourceColumn] === spec.categoryLevel ? 1 : 0);
        } else {
          vec.push(Number(cleanValues[spec.sourceColumn]));
        }
      }
      X.push(vec);
      y.push(item.y);
      kept.push(item);
    }

    return { X, y, kept, dropped };
  };

  const trainEncoded = encodeSet(rawTrain);
  const testEncoded = encodeSet(rawTest);

  if (trainEncoded.X.length < featureSpecs.length + 5) {
    return makeFailure(
      `Observaciones insuficientes en entrenamiento (${trainEncoded.X.length}) para estimar ${
        featureSpecs.length + 1
      } parámetros. Reduzca variables o cambie la estrategia de valores faltantes.`
    );
  }
  if (testEncoded.X.length === 0) {
    return makeFailure(
      'El conjunto de prueba quedó sin observaciones válidas tras manejar valores faltantes.'
    );
  }

  const trainClassDistBefore = {
    c0: trainEncoded.y.filter((v) => v === 0).length,
    c1: trainEncoded.y.filter((v) => v === 1).length,
  };
  const testClassDist = {
    c0: testEncoded.y.filter((v) => v === 0).length,
    c1: testEncoded.y.filter((v) => v === 1).length,
  };

  if (trainClassDistBefore.c0 === 0 || trainClassDistBefore.c1 === 0) {
    return makeFailure(
      'El conjunto de entrenamiento solo contiene una categoría. Ajuste el porcentaje de división o verifique los datos.'
    );
  }

  // Check zero variance on predictors in trainEncoded BEFORE balancing
  const p = featureSpecs.length;
  for (let j = 0; j < p; j++) {
    let minV = Infinity;
    let maxV = -Infinity;
    for (let i = 0; i < trainEncoded.X.length; i++) {
      const val = trainEncoded.X[i][j];
      if (val < minV) minV = val;
      if (val > maxV) maxV = val;
    }
    if (Math.abs(maxV - minV) < 1e-12) {
      return makeFailure(
        `El predictor "${featureSpecs[j].term}" no presenta variabilidad (es constante) en el conjunto de entrenamiento.`
      );
    }
  }

  // Apply class balancing strictly on training data
  const isDummyCol = featureSpecs.map((f) => f.isCategoricalDummy);
  const balancedTrain = applyClassBalance(
    trainEncoded.X,
    trainEncoded.y,
    isDummyCol,
    config.balanceMethod,
    rand
  );

  const trainClassDistAfter = {
    c0: balancedTrain.y.filter((v) => v === 0).length,
    c1: balancedTrain.y.filter((v) => v === 1).length,
  };

  // Compute train means and stds on balancedTrain for numerical conditioning
  // If standardizeNumeric is true, these also define the model's standardized feature space!
  const featureMeans = Array(p).fill(0);
  const featureStds = Array(p).fill(1);
  for (let j = 0; j < p; j++) {
    let s = 0;
    for (let i = 0; i < balancedTrain.X.length; i++) {
      s += balancedTrain.X[i][j];
    }
    const mu = s / balancedTrain.X.length;
    let sq = 0;
    for (let i = 0; i < balancedTrain.X.length; i++) {
      sq += (balancedTrain.X[i][j] - mu) ** 2;
    }
    const sd = Math.sqrt(sq / Math.max(1, balancedTrain.X.length - 1));
    if (sd < 1e-11) {
      return makeFailure(
        `El predictor "${featureSpecs[j].term}" perdió variabilidad tras el muestreo de entrenamiento.`
      );
    }
    featureMeans[j] = mu;
    featureStds[j] = sd;
  }

  // Build standardized design matrix Z = [1, (x1-mu1)/sd1, ...] for numerically stable IRLS
  const nTrain = balancedTrain.X.length;
  const k = p + 1;
  const Ztrain: number[][] = balancedTrain.X.map((row) => [
    1,
    ...row.map((val, j) => (val - featureMeans[j]) / featureStds[j]),
  ]);

  // Run IRLS (Newton-Raphson) in standardized coordinates
  let betaStd = Array(k).fill(0);
  const yMean = balancedTrain.y.reduce((a: number, b) => a + b, 0) / nTrain;
  const clampedP0 = Math.min(0.99, Math.max(0.01, yMean));
  betaStd[0] = Math.log(clampedP0 / (1 - clampedP0));

  let converged = false;
  let iterations = 0;
  const maxIter = 60;
  let fisherMatrix: number[][] = [];
  let warning: string | null = null;

  for (let iter = 1; iter <= maxIter; iter++) {
    iterations = iter;
    const grad = Array(k).fill(0);
    const H: number[][] = Array.from({ length: k }, () => Array(k).fill(0));

    for (let i = 0; i < nTrain; i++) {
      const xi = Ztrain[i];
      let eta = 0;
      for (let j = 0; j < k; j++) eta += xi[j] * betaStd[j];
      const mu = sigmoid(eta);
      const w = Math.max(1e-10, mu * (1 - mu));
      const diff = balancedTrain.y[i] - mu;

      for (let a = 0; a < k; a++) {
        grad[a] += xi[a] * diff;
        for (let b = a; b < k; b++) {
          H[a][b] += w * xi[a] * xi[b];
        }
      }
    }

    for (let a = 0; a < k; a++) {
      for (let b = 0; b < a; b++) {
        H[a][b] = H[b][a];
      }
    }

    fisherMatrix = H;
    const step = solveLinearSystem(H, grad);
    if (!step) {
      return makeFailure(
        'No fue posible estimar el modelo: la matriz de diseño es singular (existe multicolinealidad exacta o variables redundantes entre los predictores seleccionados).'
      );
    }

    let maxStep = 0;
    // Step damping if step is huge (prevents overshoot near separation)
    const stepNorm = Math.sqrt(step.reduce((acc, s) => acc + s * s, 0));
    const scale = stepNorm > 5 ? 5 / stepNorm : 1;

    for (let j = 0; j < k; j++) {
      const delta = step[j] * scale;
      betaStd[j] += delta;
      if (Math.abs(delta) > maxStep) maxStep = Math.abs(delta);
    }

    if (maxStep < 1e-8) {
      converged = true;
      break;
    }

    // Check separation explosion
    if (betaStd.some((b) => Math.abs(b) > 25)) {
      warning =
        'Advertencia estadística: posible separación completa o cuasi-completa en los datos de entrenamiento. Algunos errores estándar pueden ser muy elevados.';
      break;
    }
  }

  if (!converged && !warning) {
    warning =
      'El algoritmo de máxima verosimilitud alcanzó el límite de iteraciones sin convergencia estricta; revise posible colinealidad o separación entre predictores.';
  }

  // Recompute final Fisher information at final betaStd
  const finalH: number[][] = Array.from({ length: k }, () => Array(k).fill(0));
  let logLikelihood = 0;
  for (let i = 0; i < nTrain; i++) {
    const xi = Ztrain[i];
    let eta = 0;
    for (let j = 0; j < k; j++) eta += xi[j] * betaStd[j];
    const mu = sigmoid(eta);
    const w = Math.max(1e-10, mu * (1 - mu));
    logLikelihood +=
      balancedTrain.y[i] * Math.log(Math.max(1e-15, mu)) +
      (1 - balancedTrain.y[i]) * Math.log(Math.max(1e-15, 1 - mu));

    for (let a = 0; a < k; a++) {
      for (let b = a; b < k; b++) {
        finalH[a][b] += w * xi[a] * xi[b];
      }
    }
  }
  for (let a = 0; a < k; a++) {
    for (let b = 0; b < a; b++) finalH[a][b] = finalH[b][a];
  }
  fisherMatrix = finalH;

  const covStd = invertSymmetricMatrix(fisherMatrix);

  // Transform betaStd & covStd to target coordinate system (raw scale unless standardizeNumeric === true)
  let betaFinal: number[];
  let covFinal: number[][] | null = null;

  if (config.prepConfig.standardizeNumeric) {
    // Only continuous/discrete non-dummy numeric variables remain standardized, or all standardized:
    // For clarity and exact consistency with prediction, if standardizeNumeric=true we use z-scored numeric predictors and keep dummies 0/1?
    // Even cleaner: if standardizeNumeric=true, standardize ALL encoded columns using featureMeans/featureStds,
    // OR standardize only non-dummy columns. Let's standardize ALL numeric non-dummy columns and transform dummies back to 0/1!
    // Actually transforming ALL back to raw except when standardizeNumeric=true where non-dummies stay standardized:
    // Let's keep Jacobian transformation clean:
    // T[0][0] = 1; for j>=1: if standardizeNumeric && !isDummy[j-1], scale=1, shift=0; else scale=1/sd, shift=-mu/sd.
    const T: number[][] = Array.from({ length: k }, () => Array(k).fill(0));
    T[0][0] = 1;
    betaFinal = Array(k).fill(0);
    betaFinal[0] = betaStd[0];

    for (let j = 0; j < p; j++) {
      const keepStd = !isDummyCol[j]; // standardizeNumeric applies to numeric predictors
      const sd = keepStd ? 1 : featureStds[j];
      const mu = keepStd ? 0 : featureMeans[j];
      if (!keepStd) {
        betaFinal[j + 1] = betaStd[j + 1] / featureStds[j];
        betaFinal[0] -= (betaStd[j + 1] * featureMeans[j]) / featureStds[j];
        T[0][j + 1] = -featureMeans[j] / featureStds[j];
        T[j + 1][j + 1] = 1 / featureStds[j];
      } else {
        // Already standardized in Ztrain
        betaFinal[j + 1] = betaStd[j + 1];
        T[j + 1][j + 1] = 1;
      }
      void sd;
      void mu;
    }

    if (covStd) {
      // covFinal = T * covStd * T^T
      const temp: number[][] = Array.from({ length: k }, () => Array(k).fill(0));
      for (let a = 0; a < k; a++) {
        for (let b = 0; b < k; b++) {
          let sum = 0;
          for (let c = 0; c < k; c++) sum += T[a][c] * covStd[c][b];
          temp[a][b] = sum;
        }
      }
      covFinal = Array.from({ length: k }, () => Array(k).fill(0));
      for (let a = 0; a < k; a++) {
        for (let b = 0; b < k; b++) {
          let sum = 0;
          for (let c = 0; c < k; c++) sum += temp[a][c] * T[b][c];
          covFinal[a][b] = sum;
        }
      }
    }
  } else {
    // Back-transform all parameters to original unstandardized units (identical to R glm)
    const T: number[][] = Array.from({ length: k }, () => Array(k).fill(0));
    T[0][0] = 1;
    betaFinal = Array(k).fill(0);
    betaFinal[0] = betaStd[0];

    for (let j = 0; j < p; j++) {
      const invSd = 1 / featureStds[j];
      betaFinal[j + 1] = betaStd[j + 1] * invSd;
      betaFinal[0] -= betaStd[j + 1] * featureMeans[j] * invSd;
      T[0][j + 1] = -featureMeans[j] * invSd;
      T[j + 1][j + 1] = invSd;
    }

    if (covStd) {
      const temp: number[][] = Array.from({ length: k }, () => Array(k).fill(0));
      for (let a = 0; a < k; a++) {
        for (let b = 0; b < k; b++) {
          let sum = 0;
          for (let c = 0; c < k; c++) sum += T[a][c] * covStd[c][b];
          temp[a][b] = sum;
        }
      }
      covFinal = Array.from({ length: k }, () => Array(k).fill(0));
      for (let a = 0; a < k; a++) {
        for (let b = 0; b < k; b++) {
          let sum = 0;
          for (let c = 0; c < k; c++) sum += temp[a][c] * T[b][c];
          covFinal[a][b] = sum;
        }
      }
    }
  }

  const coefficients: CoefficientEstimate[] = [];
  for (let j = 0; j < k; j++) {
    const beta = betaFinal[j];
    const variance = covFinal ? covFinal[j][j] : null;
    const stdError = variance !== null && variance > 0 ? Math.sqrt(variance) : null;
    const zValue = stdError !== null && stdError > 1e-15 ? beta / stdError : null;
    const pValue = zValue !== null ? twoTailedPValue(zValue) : null;
    const oddsRatio = Math.exp(Math.min(50, Math.max(-50, beta)));
    const ciLower95 = stdError !== null ? Math.exp(Math.min(50, Math.max(-50, beta - 1.96 * stdError))) : null;
    const ciUpper95 = stdError !== null ? Math.exp(Math.min(50, Math.max(-50, beta + 1.96 * stdError))) : null;

    if (j === 0) {
      coefficients.push({
        term: '(Intercepto)',
        sourceColumn: '(Intercepto)',
        beta,
        stdError,
        zValue,
        pValue,
        oddsRatio,
        ciLower95,
        ciUpper95,
      });
    } else {
      const spec = featureSpecs[j - 1];
      coefficients.push({
        term: spec.term,
        sourceColumn: spec.sourceColumn,
        categoryLevel: spec.categoryLevel,
        beta,
        stdError,
        zValue,
        pValue,
        oddsRatio,
        ciLower95,
        ciUpper95,
      });
    }
  }

  const nullProb = Math.min(1 - 1e-15, Math.max(1e-15, yMean));
  const nullLL =
    nTrain * (yMean * Math.log(nullProb) + (1 - yMean) * Math.log(1 - nullProb));
  const nullDeviance = -2 * nullLL;
  const residualDeviance = -2 * logLikelihood;
  const aic = -2 * logLikelihood + 2 * k;
  const mcfaddenR2 = nullLL !== 0 ? 1 - logLikelihood / nullLL : 0;

  const encodedFeatures: EncodedFeatureMeta[] = featureSpecs.map((spec, j) => ({
    ...spec,
    mean: featureMeans[j],
    std: featureStds[j],
    lowerFence: outlierFences[spec.sourceColumn]?.lower,
    upperFence: outlierFences[spec.sourceColumn]?.upper,
  }));

  const scoreVector = (xRow: number[]): { z: number; prob: number } => {
    let z = betaFinal[0];
    for (let j = 0; j < p; j++) {
      const useStd = config.prepConfig.standardizeNumeric && !isDummyCol[j];
      const val = useStd ? (xRow[j] - featureMeans[j]) / featureStds[j] : xRow[j];
      z += betaFinal[j + 1] * val;
    }
    return { z, prob: sigmoid(z) };
  };

  // Predict on ORIGINAL (unbalanced) training rows and TEST rows
  const trainPredictions: PredictionRecord[] = trainEncoded.X.map((xRow, idx) => {
    const { z, prob } = scoreVector(xRow);
    const pred: 0 | 1 = prob >= config.threshold ? 1 : 0;
    const item = trainEncoded.kept[idx];
    return {
      rowIndex: item.rowIndex,
      actual: item.y,
      actualLabel: item.yLabel,
      zValue: z,
      probability: prob,
      predicted: pred,
      predictedLabel: pred === 1 ? config.positiveClass : config.negativeClass,
      rawRow: item.row,
    };
  });

  const testPredictions: PredictionRecord[] = testEncoded.X.map((xRow, idx) => {
    const { z, prob } = scoreVector(xRow);
    const pred: 0 | 1 = prob >= config.threshold ? 1 : 0;
    const item = testEncoded.kept[idx];
    return {
      rowIndex: item.rowIndex,
      actual: item.y,
      actualLabel: item.yLabel,
      zValue: z,
      probability: prob,
      predicted: pred,
      predictedLabel: pred === 1 ? config.positiveClass : config.negativeClass,
      rawRow: item.row,
    };
  });

  return {
    id,
    name: modelName,
    createdAt,
    config,
    converged,
    iterations,
    warning,
    error: null,
    coefficients,
    nullDeviance,
    residualDeviance,
    aic,
    logLikelihood,
    mcfaddenR2,
    trainCount: trainEncoded.X.length,
    testCount: testEncoded.X.length,
    balancedTrainCount: nTrain,
    droppedTrainRows: trainEncoded.dropped,
    droppedTestRows: testEncoded.dropped,
    trainClassDistBefore,
    trainClassDistAfter,
    testClassDist,
    trainPredictions,
    testPredictions,
    encodedFeatures,
    categoryLevelsByColumn,
    imputationValues,
  };
}

export function computeConfusionMatrix(
  records: PredictionRecord[],
  threshold: number
): ConfusionMatrix {
  let tp = 0;
  let tn = 0;
  let fp = 0;
  let fn = 0;
  for (const r of records) {
    const pred = r.probability >= threshold ? 1 : 0;
    if (r.actual === 1 && pred === 1) tp++;
    else if (r.actual === 0 && pred === 0) tn++;
    else if (r.actual === 0 && pred === 1) fp++;
    else if (r.actual === 1 && pred === 0) fn++;
  }
  return { tp, tn, fp, fn, total: records.length };
}

export function computeEvaluationMetrics(cm: ConfusionMatrix): EvaluationMetrics {
  const { tp, tn, fp, fn, total } = cm;
  const undefinedReasons: EvaluationMetrics['undefinedReasons'] = {};

  const accuracy = total > 0 ? (tp + tn) / total : null;
  if (accuracy === null) {
    undefinedReasons.accuracy = 'No definida: el total de observaciones (TP + TN + FP + FN) es 0.';
  }

  const classificationError = accuracy !== null ? 1 - accuracy : null;
  if (classificationError === null) {
    undefinedReasons.classificationError = 'No definida porque Accuracy no está definida.';
  }

  const denomP1 = tp + fp;
  const precision1 = denomP1 > 0 ? tp / denomP1 : null;
  if (precision1 === null) {
    undefinedReasons.precision1 =
      'No definida (división por cero): el modelo no predijo ningún caso en la categoría 1 (TP + FP = 0).';
  }

  const denomP0 = tn + fn;
  const precision0 = denomP0 > 0 ? tn / denomP0 : null;
  if (precision0 === null) {
    undefinedReasons.precision0 =
      'No definida (división por cero): el modelo no predijo ningún caso en la categoría 0 (TN + FN = 0).';
  }

  const denomR1 = tp + fn;
  const recall1 = denomR1 > 0 ? tp / denomR1 : null;
  if (recall1 === null) {
    undefinedReasons.recall1 =
      'No definida (división por cero): no hay observaciones reales de la categoría 1 en este conjunto (TP + FN = 0).';
  }

  const denomR0 = tn + fp;
  const recall0 = denomR0 > 0 ? tn / denomR0 : null;
  if (recall0 === null) {
    undefinedReasons.recall0 =
      'No definida (división por cero): no hay observaciones reales de la categoría 0 en este conjunto (TN + FP = 0).';
  }

  let f1_1: number | null = null;
  if (precision1 === null || recall1 === null) {
    undefinedReasons.f1_1 =
      'No definida: depende de Precision(1) y Recall(1), y al menos una no está definida.';
  } else if (precision1 + recall1 === 0) {
    undefinedReasons.f1_1 =
      'No definida (división por cero): Precision(1) + Recall(1) = 0 porque TP = 0.';
  } else {
    f1_1 = (2 * precision1 * recall1) / (precision1 + recall1);
  }

  let f1_0: number | null = null;
  if (precision0 === null || recall0 === null) {
    undefinedReasons.f1_0 =
      'No definida: depende de Precision(0) y Recall(0), y al menos una no está definida.';
  } else if (precision0 + recall0 === 0) {
    undefinedReasons.f1_0 =
      'No definida (división por cero): Precision(0) + Recall(0) = 0 porque TN = 0.';
  } else {
    f1_0 = (2 * precision0 * recall0) / (precision0 + recall0);
  }

  let balancedAccuracy: number | null = null;
  if (recall1 === null || recall0 === null) {
    undefinedReasons.balancedAccuracy =
      'No definida: requiere que Recall(0) y Recall(1) estén definidos.';
  } else {
    balancedAccuracy = (recall0 + recall1) / 2;
  }

  return {
    accuracy,
    classificationError,
    precision1,
    precision0,
    recall1,
    recall0,
    f1_1,
    f1_0,
    balancedAccuracy,
    undefinedReasons,
  };
}

export interface SinglePredictionStep {
  term: string;
  beta: number;
  xValue: number;
  rawDisplay: string;
  contribution: number;
}

export function predictSingleObservation(
  model: ModelResult,
  inputs: Record<string, string | number>,
  thresholdOverride?: number
): {
  zValue: number;
  probability: number;
  predicted: 0 | 1;
  predictedLabel: string;
  threshold: number;
  steps: SinglePredictionStep[];
} {
  const threshold = thresholdOverride ?? model.config.threshold;
  const intercept = model.coefficients[0];
  let z = intercept ? intercept.beta : 0;
  const steps: SinglePredictionStep[] = [
    {
      term: '(Intercepto)',
      beta: z,
      xValue: 1,
      rawDisplay: '1',
      contribution: z,
    },
  ];

  for (let i = 0; i < model.encodedFeatures.length; i++) {
    const feat = model.encodedFeatures[i];
    const coef = model.coefficients[i + 1];
    const rawIn = inputs[feat.sourceColumn];
    let xVal = 0;
    let rawDisplay = '';

    if (feat.isCategoricalDummy) {
      const strVal = String(rawIn ?? '');
      xVal = strVal === feat.categoryLevel ? 1 : 0;
      rawDisplay = `${strVal} → ${xVal}`;
    } else {
      let numVal =
        typeof rawIn === 'number' && Number.isFinite(rawIn)
          ? rawIn
          : tryParseNumeric(rawIn) ?? feat.mean;
      if (
        model.config.prepConfig.outlierStrategy === 'cap_iqr' &&
        feat.lowerFence !== undefined &&
        feat.upperFence !== undefined
      ) {
        if (numVal < feat.lowerFence) numVal = feat.lowerFence;
        if (numVal > feat.upperFence) numVal = feat.upperFence;
      }
      const useStd = model.config.prepConfig.standardizeNumeric;
      xVal = useStd ? (numVal - feat.mean) / feat.std : numVal;
      rawDisplay = useStd ? `${numVal} (z=${xVal.toFixed(3)})` : String(numVal);
    }

    const contrib = coef.beta * xVal;
    z += contrib;
    steps.push({
      term: feat.term,
      beta: coef.beta,
      xValue: xVal,
      rawDisplay,
      contribution: contrib,
    });
  }

  const probability = sigmoid(z);
  const predicted: 0 | 1 = probability >= threshold ? 1 : 0;
  const predictedLabel =
    predicted === 1 ? model.config.positiveClass : model.config.negativeClass;

  return {
    zValue: z,
    probability,
    predicted,
    predictedLabel,
    threshold,
    steps,
  };
}

export function predictBatchRows(
  model: ModelResult,
  rows: DataRow[],
  thresholdOverride?: number
): {
  rowIndex: number;
  inputs: Record<string, string | number>;
  zValue: number;
  probability: number;
  predicted: 0 | 1;
  predictedLabel: string;
  actual?: string;
}[] {
  const threshold = thresholdOverride ?? model.config.threshold;
  return rows.map((row, idx) => {
    const inputs: Record<string, string | number> = {};
    for (const col of model.config.featureColumns) {
      const val = row[col];
      if (isMissingValue(val)) {
        inputs[col] = model.imputationValues[col] ?? 0;
      } else if (model.categoryLevelsByColumn[col]) {
        inputs[col] = String(val).trim();
      } else {
        inputs[col] = tryParseNumeric(val) ?? Number(model.imputationValues[col] ?? 0);
      }
    }
    const res = predictSingleObservation(model, inputs, threshold);
    const actualRaw = row[model.config.targetColumn];
    return {
      rowIndex: idx + 1,
      inputs,
      zValue: res.zValue,
      probability: res.probability,
      predicted: res.predicted,
      predictedLabel: res.predictedLabel,
      actual: isMissingValue(actualRaw) ? undefined : String(actualRaw).trim(),
    };
  });
}
