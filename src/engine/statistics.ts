import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import type {
  CellValue,
  ColumnProfile,
  CorrelationMatrix,
  DataRow,
  DatasetState,
  FrequencyEntry,
  NumericStats,
  VariableKind,
} from '../types/analytics';

export function isMissingValue(val: unknown): boolean {
  if (val === null || val === undefined) return true;
  if (typeof val === 'number') return Number.isNaN(val) || !Number.isFinite(val);
  if (typeof val === 'string') {
    const t = val.trim().toLowerCase();
    return t === '' || t === 'na' || t === 'n/a' || t === 'null' || t === 'nan' || t === 'undefined';
  }
  return false;
}

export function tryParseNumeric(val: unknown): number | null {
  if (isMissingValue(val)) return null;
  if (typeof val === 'number') return Number.isFinite(val) ? val : null;
  if (typeof val === 'boolean') return val ? 1 : 0;
  if (typeof val === 'string') {
    const cleaned = val.trim().replace(/\s+/g, '');
    const isStandardOrCommaDec = /^[-+]?(\d+([.,]\d+)?|\d{1,3}(,\d{3})+(\.\d+)?)$/.test(cleaned);
    const isEuroThousands = /^[-+]?\d{1,3}(\.\d{3})+(,\d+)?$/.test(cleaned);
    if (!isStandardOrCommaDec && !isEuroThousands) {
      return null;
    }
    let normalized = cleaned;
    if (isEuroThousands) {
      normalized = cleaned.replace(/\./g, '').replace(',', '.');
    } else if (cleaned.includes(',') && !cleaned.includes('.')) {
      const commaCount = (cleaned.match(/,/g) || []).length;
      normalized = commaCount === 1 ? cleaned.replace(',', '.') : cleaned.replace(/,/g, '');
    } else if (cleaned.includes(',') && cleaned.includes('.')) {
      normalized = cleaned.replace(/,/g, '');
    }
    const n = Number(normalized);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function normalizeCell(raw: unknown): CellValue {
  if (isMissingValue(raw)) return null;
  if (typeof raw === 'boolean') return raw ? 1 : 0;
  if (typeof raw === 'number') return Number.isFinite(raw) ? raw : null;
  if (raw instanceof Date) {
    return raw.toISOString().split('T')[0];
  }
  const str = String(raw).trim();
  const num = tryParseNumeric(str);
  if (num !== null && !/^0\d+$/.test(str)) {
    return num;
  }
  return str;
}

export function normalizeRows(rawRows: Record<string, unknown>[]): {
  columns: string[];
  rows: DataRow[];
} {
  if (!rawRows || rawRows.length === 0) {
    return { columns: [], rows: [] };
  }

  const colSet = new Set<string>();
  for (const row of rawRows) {
    Object.keys(row).forEach((k) => {
      const cleanKey = k.trim();
      if (cleanKey !== '' && !cleanKey.startsWith('__EMPTY')) {
        colSet.add(cleanKey);
      }
    });
  }
  const columns = Array.from(colSet);

  const rows: DataRow[] = [];
  for (const raw of rawRows) {
    const obj: DataRow = {};
    let hasAnyValue = false;
    for (const col of columns) {
      // Find matching original key (trimmed)
      const matchingKey = Object.keys(raw).find((k) => k.trim() === col);
      const val = matchingKey !== undefined ? normalizeCell(raw[matchingKey]) : null;
      obj[col] = val;
      if (val !== null) hasAnyValue = true;
    }
    if (hasAnyValue) {
      rows.push(obj);
    }
  }

  return { columns, rows };
}

export async function parseDatasetFile(
  file: File,
  preferredSheet?: string
): Promise<{
  columns: string[];
  rows: DataRow[];
  sourceType: 'csv' | 'excel';
  sheetNames?: string[];
  selectedSheet?: string;
}> {
  const ext = file.name.split('.').pop()?.toLowerCase() || '';

  if (ext === 'csv' || ext === 'txt') {
    const text = await file.text();
    const parsed = Papa.parse<Record<string, unknown>>(text, {
      header: true,
      skipEmptyLines: 'greedy',
      dynamicTyping: false,
    });

    if (parsed.errors.length > 0 && parsed.data.length === 0) {
      throw new Error(`No se pudo leer el archivo CSV: ${parsed.errors[0].message}`);
    }

    const { columns, rows } = normalizeRows(parsed.data);
    if (columns.length === 0 || rows.length === 0) {
      throw new Error('El archivo CSV no contiene columnas ni filas válidas.');
    }

    return { columns, rows, sourceType: 'csv' };
  }

  if (ext === 'xls' || ext === 'xlsx') {
    const buffer = await file.arrayBuffer();
    const workbook = XLSX.read(buffer, { type: 'array', cellDates: true });
    const sheetNames = workbook.SheetNames;
    if (!sheetNames || sheetNames.length === 0) {
      throw new Error('El archivo de Excel no contiene hojas.');
    }

    const selectedSheet =
      preferredSheet && sheetNames.includes(preferredSheet)
        ? preferredSheet
        : sheetNames[0];

    const worksheet = workbook.Sheets[selectedSheet];
    const jsonRows = XLSX.utils.sheet_to_json<Record<string, unknown>>(worksheet, {
      defval: null,
    });

    const { columns, rows } = normalizeRows(jsonRows);
    if (columns.length === 0 || rows.length === 0) {
      throw new Error(`La hoja "${selectedSheet}" no contiene registros válidos.`);
    }

    return {
      columns,
      rows,
      sourceType: 'excel',
      sheetNames,
      selectedSheet,
    };
  }

  throw new Error(
    `Formato no compatible (.${ext}). Cargue archivos en formato CSV, XLS o XLSX.`
  );
}

function quantile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0];
  const idx = (sorted.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  const frac = idx - lo;
  return sorted[lo] * (1 - frac) + sorted[hi] * frac;
}

export function computeNumericStats(values: number[]): NumericStats | null {
  if (values.length === 0) return null;
  const n = values.length;
  const sorted = [...values].sort((a, b) => a - b);
  const min = sorted[0];
  const max = sorted[n - 1];
  const range = max - min;

  let sum = 0;
  for (const v of values) sum += v;
  const mean = sum / n;

  const q1 = quantile(sorted, 0.25);
  const q2 = quantile(sorted, 0.5);
  const q3 = quantile(sorted, 0.75);
  const iqr = q3 - q1;
  const lowerFence = q1 - 1.5 * iqr;
  const upperFence = q3 + 1.5 * iqr;

  let sqDiffSum = 0;
  let outlierCount = 0;
  const freqMap = new Map<string, number>();

  for (const v of values) {
    sqDiffSum += (v - mean) * (v - mean);
    if (v < lowerFence || v > upperFence) outlierCount++;
    const key = String(Math.round(v * 1e6) / 1e6);
    freqMap.set(key, (freqMap.get(key) || 0) + 1);
  }

  const variance = n > 1 ? sqDiffSum / (n - 1) : 0;
  const stdDev = Math.sqrt(variance);

  let modeVal = '';
  let maxFreq = 0;
  for (const [k, cnt] of freqMap.entries()) {
    if (cnt > maxFreq) {
      maxFreq = cnt;
      modeVal = k;
    }
  }
  const mode = maxFreq <= 1 && n > 10 ? 'Múltiple / Sin moda única' : modeVal;

  return {
    count: n,
    mean,
    median: q2,
    mode,
    min,
    max,
    range,
    variance,
    stdDev,
    q1,
    q2,
    q3,
    iqr,
    outlierCount,
    lowerFence,
    upperFence,
  };
}

export function detectVariableKind(
  values: CellValue[],
  numericValues: number[],
  uniqueStrings: string[]
): VariableKind {
  const validCount = values.filter((v) => !isMissingValue(v)).length;
  if (validCount === 0) return 'categorical';

  if (uniqueStrings.length === 2) {
    return 'binary';
  }

  // If >= 85% of non-missing values are numeric
  if (numericValues.length / validCount >= 0.85 && numericValues.length > 0) {
    const allIntegers = numericValues.every((n) => Math.abs(n - Math.round(n)) < 1e-9);
    if (allIntegers && uniqueStrings.length <= 20) {
      return 'numeric_discrete';
    }
    return 'numeric_continuous';
  }

  return 'categorical';
}

export function formatNumberES(value: number, decimals = 2): string {
  if (!Number.isFinite(value)) return '—';
  return value.toLocaleString('es-CO', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function generateVariableInterpretation(
  name: string,
  kind: VariableKind,
  validCount: number,
  missingCount: number,
  numericStats: NumericStats | null,
  frequencies: FrequencyEntry[]
): string {
  if (validCount === 0) {
    return `La variable "${name}" no contiene observaciones válidas (${missingCount} valores perdidos).`;
  }

  if ((kind === 'numeric_continuous' || kind === 'numeric_discrete') && numericStats) {
    const skewHint =
      Math.abs(numericStats.mean - numericStats.median) > 0.25 * (numericStats.stdDev || 1)
        ? numericStats.mean > numericStats.median
          ? 'presenta asimetría positiva (cola hacia valores altos)'
          : 'presenta asimetría negativa (cola hacia valores bajos)'
        : 'presenta una distribución relativamente simétrica alrededor de su centro';

    const outlierText =
      numericStats.outlierCount > 0
        ? ` Se detectaron ${numericStats.outlierCount} valores atípicos potenciales según el criterio de rango intercuartílico (IQR).`
        : ' No se observan valores atípicos extremos según el criterio IQR.';

    return `El análisis muestra que la variable "${name}" (${
      kind === 'numeric_continuous' ? 'numérica continua' : 'numérica discreta'
    }) ${skewHint}, con un promedio de ${formatNumberES(numericStats.mean, 2)}, mediana de ${formatNumberES(
      numericStats.median,
      2
    )} y una desviación estándar de ${formatNumberES(numericStats.stdDev, 2)} (rango entre ${formatNumberES(
      numericStats.min,
      2
    )} y ${formatNumberES(numericStats.max, 2)}).${outlierText}`;
  }

  const top = frequencies[0];
  const second = frequencies[1];
  if (kind === 'binary' && top && second) {
    return `La variable binaria "${name}" registra ${validCount} observaciones válidas distribuidas entre "${top.value}" (${formatNumberES(
      top.percentage,
      1
    )}%, n=${top.count}) y "${second.value}" (${formatNumberES(
      second.percentage,
      1
    )}%, n=${second.count}).`;
  }

  if (top) {
    return `La variable categórica "${name}" posee ${frequencies.length} categorías distintas; la categoría predominante es "${
      top.value
    }" con ${top.count} registros (${formatNumberES(top.percentage, 1)}% del total válido).`;
  }

  return `La variable "${name}" cuenta con ${validCount} registros válidos.`;
}

export function countDuplicateRows(rows: DataRow[], columns: string[]): number {
  const seen = new Set<string>();
  let duplicates = 0;
  for (const row of rows) {
    const sig = columns.map((c) => String(row[c] ?? '<<NULL>>')).join('\u0001');
    if (seen.has(sig)) {
      duplicates++;
    } else {
      seen.add(sig);
    }
  }
  return duplicates;
}

export function removeDuplicateRows(rows: DataRow[], columns: string[]): DataRow[] {
  const seen = new Set<string>();
  const unique: DataRow[] = [];
  for (const row of rows) {
    const sig = columns.map((c) => String(row[c] ?? '<<NULL>>')).join('\u0001');
    if (!seen.has(sig)) {
      seen.add(sig);
      unique.push(row);
    }
  }
  return unique;
}

export function computeCorrelationMatrix(
  rows: DataRow[],
  numericCols: string[]
): CorrelationMatrix {
  const m = numericCols.length;
  const matrix: number[][] = Array.from({ length: m }, () => Array(m).fill(0));

  for (let i = 0; i < m; i++) {
    matrix[i][i] = 1;
    for (let j = i + 1; j < m; j++) {
      const xs: number[] = [];
      const ys: number[] = [];
      for (const row of rows) {
        const vx = tryParseNumeric(row[numericCols[i]]);
        const vy = tryParseNumeric(row[numericCols[j]]);
        if (vx !== null && vy !== null) {
          xs.push(vx);
          ys.push(vy);
        }
      }
      let r = 0;
      if (xs.length >= 3) {
        const n = xs.length;
        const meanX = xs.reduce((a, b) => a + b, 0) / n;
        const meanY = ys.reduce((a, b) => a + b, 0) / n;
        let num = 0;
        let denX = 0;
        let denY = 0;
        for (let k = 0; k < n; k++) {
          const dx = xs[k] - meanX;
          const dy = ys[k] - meanY;
          num += dx * dy;
          denX += dx * dx;
          denY += dy * dy;
        }
        const denom = Math.sqrt(denX * denY);
        r = denom > 1e-12 ? num / denom : 0;
      }
      matrix[i][j] = r;
      matrix[j][i] = r;
    }
  }

  return { columns: numericCols, matrix };
}

export function profileDataset(
  rows: DataRow[],
  columns: string[],
  overrides: Record<string, VariableKind> = {}
): {
  profiles: ColumnProfile[];
  correlation: CorrelationMatrix;
  duplicateCount: number;
} {
  const profiles: ColumnProfile[] = columns.map((col) => {
    const rawValues = rows.map((r) => (r[col] === undefined ? null : r[col]));
    const validValues = rawValues.filter((v) => !isMissingValue(v));
    const missingCount = rawValues.length - validValues.length;

    const numericValues: number[] = [];
    const freqCounter = new Map<string, number>();

    for (const v of validValues) {
      const str = String(v).trim();
      freqCounter.set(str, (freqCounter.get(str) || 0) + 1);
      const num = tryParseNumeric(v);
      if (num !== null) numericValues.push(num);
    }

    const frequencies: FrequencyEntry[] = Array.from(freqCounter.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([value, count]) => ({
        value,
        count,
        relativeFreq: validValues.length > 0 ? count / validValues.length : 0,
        percentage: validValues.length > 0 ? (count / validValues.length) * 100 : 0,
      }));

    const uniqueStrings = Array.from(freqCounter.keys());
    const detectedKind = detectVariableKind(rawValues, numericValues, uniqueStrings);
    const effectiveKind = overrides[col] || detectedKind;

    const showNumeric =
      effectiveKind === 'numeric_continuous' ||
      effectiveKind === 'numeric_discrete' ||
      (effectiveKind === 'binary' && numericValues.length === validValues.length);

    const numericStats = showNumeric ? computeNumericStats(numericValues) : null;

    const interpretation = generateVariableInterpretation(
      col,
      effectiveKind,
      validValues.length,
      missingCount,
      numericStats,
      frequencies
    );

    return {
      name: col,
      detectedKind,
      overrideKind: overrides[col],
      effectiveKind,
      totalCount: rawValues.length,
      validCount: validValues.length,
      missingCount,
      uniqueCount: uniqueStrings.length,
      sampleValues: uniqueStrings.slice(0, 5),
      numericStats,
      frequencies,
      interpretation,
    };
  });

  const numericCols = profiles
    .filter(
      (p) =>
        p.effectiveKind === 'numeric_continuous' ||
        p.effectiveKind === 'numeric_discrete' ||
        (p.effectiveKind === 'binary' && p.numericStats !== null)
    )
    .map((p) => p.name);

  const correlation = computeCorrelationMatrix(rows, numericCols);
  const duplicateCount = countDuplicateRows(rows, columns);

  return { profiles, correlation, duplicateCount };
}

export function buildDatasetState(
  fileName: string,
  sourceType: DatasetState['sourceType'],
  columns: string[],
  originalRows: DataRow[],
  sheetNames?: string[],
  selectedSheet?: string
): DatasetState {
  const { profiles, correlation, duplicateCount } = profileDataset(originalRows, columns);
  return {
    fileName,
    sourceType,
    originalRows,
    workingRows: originalRows.map((r) => ({ ...r })),
    columns,
    profiles,
    correlation,
    duplicateCount,
    loadedAt: new Date().toISOString(),
    sheetNames,
    selectedSheet,
  };
}
