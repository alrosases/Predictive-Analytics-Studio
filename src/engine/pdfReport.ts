import { jsPDF } from 'jspdf';
import autoTable from 'jspdf-autotable';
import type {
  ConfusionMatrix,
  DatasetState,
  EvaluationMetrics,
  ModelResult,
  ReportMetadata,
} from '../types/analytics';
import {
  computeConfusionMatrix,
  computeEvaluationMetrics,
  sigmoid,
} from './logisticRegression';
import { formatNumberES, tryParseNumeric } from './statistics';

function fmtPct(v: number | null): string {
  if (v === null || !Number.isFinite(v)) return 'No definida';
  return `${(v * 100).toFixed(2)}%`;
}

function fmtDec(v: number | null, dec = 4): string {
  if (v === null || !Number.isFinite(v)) return 'N/D';
  return v.toFixed(dec);
}

function renderSigmoidChartToDataUrl(model: ModelResult): string {
  if (typeof document === 'undefined') return '';
  const canvas = document.createElement('canvas');
  canvas.width = 1100;
  canvas.height = 480;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';

  ctx.fillStyle = '#FFFFFF';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const padL = 90;
  const padR = 45;
  const padT = 48;
  const padB = 65;
  const plotW = canvas.width - padL - padR;
  const plotH = canvas.height - padT - padB;

  const zMin = -6;
  const zMax = 6;
  const toX = (z: number) => padL + ((Math.max(zMin, Math.min(zMax, z)) - zMin) / (zMax - zMin)) * plotW;
  const toY = (p: number) => padT + (1 - p) * plotH;

  // Background grid
  ctx.strokeStyle = '#E2E8F0';
  ctx.lineWidth = 1;
  for (let p = 0; p <= 1.001; p += 0.25) {
    const y = toY(p);
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + plotW, y);
    ctx.stroke();

    ctx.fillStyle = '#475569';
    ctx.font = '18px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(`${Math.round(p * 100)}%`, padL - 12, y + 6);
  }

  for (let z = -6; z <= 6; z += 2) {
    const x = toX(z);
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, padT + plotH);
    ctx.stroke();

    ctx.fillStyle = '#475569';
    ctx.font = '18px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(String(z), x, padT + plotH + 26);
  }

  // Threshold line
  const thrY = toY(model.config.threshold);
  ctx.save();
  ctx.setLineDash([8, 6]);
  ctx.strokeStyle = '#D97706';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(padL, thrY);
  ctx.lineTo(padL + plotW, thrY);
  ctx.stroke();
  ctx.restore();

  ctx.fillStyle = '#B45309';
  ctx.font = 'bold 17px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(`Punto de corte: ${(model.config.threshold * 100).toFixed(0)}%`, padL + 12, thrY - 10);

  // Sigmoid curve
  ctx.strokeStyle = '#0D9488';
  ctx.lineWidth = 4;
  ctx.beginPath();
  for (let i = 0; i <= 200; i++) {
    const z = zMin + (i / 200) * (zMax - zMin);
    const p = sigmoid(z);
    const x = toX(z);
    const y = toY(p);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Plot test points (capped to 250 for readability)
  const sample = model.testPredictions.slice(0, 250);
  for (const rec of sample) {
    const x = toX(rec.zValue);
    const y = toY(rec.probability);
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fillStyle = rec.actual === 1 ? 'rgba(13, 148, 136, 0.72)' : 'rgba(15, 23, 42, 0.55)';
    ctx.fill();
  }

  // Titles & axes
  ctx.fillStyle = '#0F172A';
  ctx.font = 'bold 21px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('Curva Sigmoide P(Y=1) = 1 / (1 + e^-Z) y observaciones de prueba', padL, 30);

  ctx.fillStyle = '#334155';
  ctx.font = '18px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Combinación lineal Z', padL + plotW / 2, canvas.height - 14);

  return canvas.toDataURL('image/png');
}

function renderMetricsBarChartToDataUrl(
  mTrain: EvaluationMetrics,
  mTest: EvaluationMetrics
): string {
  if (typeof document === 'undefined') return '';
  const canvas = document.createElement('canvas');
  canvas.width = 1100;
  canvas.height = 460;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';

  ctx.fillStyle = '#FFFFFF';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const items = [
    { label: 'Accuracy', train: mTrain.accuracy, test: mTest.accuracy },
    { label: 'Prec. (1)', train: mTrain.precision1, test: mTest.precision1 },
    { label: 'Recall (1)', train: mTrain.recall1, test: mTest.recall1 },
    { label: 'F1 (1)', train: mTrain.f1_1, test: mTest.f1_1 },
    { label: 'F1 (0)', train: mTrain.f1_0, test: mTest.f1_0 },
    { label: 'Bal. Acc.', train: mTrain.balancedAccuracy, test: mTest.balancedAccuracy },
  ];

  const padL = 80;
  const padR = 40;
  const padT = 65;
  const padB = 75;
  const plotW = canvas.width - padL - padR;
  const plotH = canvas.height - padT - padB;

  ctx.strokeStyle = '#E2E8F0';
  ctx.lineWidth = 1;
  for (let p = 0; p <= 100; p += 25) {
    const y = padT + (1 - p / 100) * plotH;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + plotW, y);
    ctx.stroke();

    ctx.fillStyle = '#475569';
    ctx.font = '18px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(`${p}%`, padL - 10, y + 6);
  }

  const groupW = plotW / items.length;
  const barW = Math.min(44, groupW * 0.28);

  items.forEach((it, idx) => {
    const center = padL + idx * groupW + groupW / 2;
    const tVal = it.train ?? 0;
    const teVal = it.test ?? 0;

    // Train bar
    const h1 = tVal * plotH;
    ctx.fillStyle = '#0F172A';
    ctx.fillRect(center - barW - 4, padT + plotH - h1, barW, h1);
    ctx.fillStyle = '#0F172A';
    ctx.font = '15px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(
      it.train !== null ? `${(it.train * 100).toFixed(1)}%` : 'N/D',
      center - barW / 2 - 4,
      padT + plotH - h1 - 8
    );

    // Test bar
    const h2 = teVal * plotH;
    ctx.fillStyle = '#0D9488';
    ctx.fillRect(center + 4, padT + plotH - h2, barW, h2);
    ctx.fillStyle = '#0F766E';
    ctx.fillText(
      it.test !== null ? `${(it.test * 100).toFixed(1)}%` : 'N/D',
      center + barW / 2 + 4,
      padT + plotH - h2 - 8
    );

    ctx.fillStyle = '#1E293B';
    ctx.font = 'bold 17px sans-serif';
    ctx.fillText(it.label, center, padT + plotH + 30);
  });

  // Title & Legend
  ctx.fillStyle = '#0F172A';
  ctx.font = 'bold 21px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('Comparación de Métricas: Entrenamiento vs. Prueba', padL, 32);

  ctx.fillStyle = '#0F172A';
  ctx.fillRect(canvas.width - 330, 18, 18, 18);
  ctx.fillStyle = '#334155';
  ctx.font = '17px sans-serif';
  ctx.fillText('Entrenamiento', canvas.width - 304, 33);

  ctx.fillStyle = '#0D9488';
  ctx.fillRect(canvas.width - 160, 18, 18, 18);
  ctx.fillStyle = '#334155';
  ctx.fillText('Prueba', canvas.width - 134, 33);

  return canvas.toDataURL('image/png');
}

function renderClassDistChartToDataUrl(model: ModelResult): string {
  if (typeof document === 'undefined') return '';
  const canvas = document.createElement('canvas');
  canvas.width = 1100;
  canvas.height = 380;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';

  ctx.fillStyle = '#FFFFFF';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const groups = [
    {
      label: 'Entrenamiento (Original)',
      c0: model.trainClassDistBefore.c0,
      c1: model.trainClassDistBefore.c1,
    },
    {
      label: `Entrenamiento (${model.config.balanceMethod})`,
      c0: model.trainClassDistAfter.c0,
      c1: model.trainClassDistAfter.c1,
    },
    {
      label: 'Prueba (Sin alterar)',
      c0: model.testClassDist.c0,
      c1: model.testClassDist.c1,
    },
  ];

  const maxCount = Math.max(
    1,
    ...groups.flatMap((g) => [g.c0, g.c1])
  );

  const padL = 80;
  const padR = 40;
  const padT = 60;
  const padB = 65;
  const plotW = canvas.width - padL - padR;
  const plotH = canvas.height - padT - padB;
  const groupW = plotW / groups.length;
  const barW = 54;

  ctx.strokeStyle = '#E2E8F0';
  for (let i = 0; i <= 4; i++) {
    const val = Math.round((maxCount * i) / 4);
    const y = padT + (1 - i / 4) * plotH;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + plotW, y);
    ctx.stroke();
    ctx.fillStyle = '#475569';
    ctx.font = '16px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(String(val), padL - 10, y + 5);
  }

  groups.forEach((g, idx) => {
    const cx = padL + idx * groupW + groupW / 2;
    const h0 = (g.c0 / maxCount) * plotH;
    const h1 = (g.c1 / maxCount) * plotH;

    ctx.fillStyle = '#334155';
    ctx.fillRect(cx - barW - 6, padT + plotH - h0, barW, h0);
    ctx.fillStyle = '#0D9488';
    ctx.fillRect(cx + 6, padT + plotH - h1, barW, h1);

    ctx.fillStyle = '#0F172A';
    ctx.font = '16px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(String(g.c0), cx - barW / 2 - 6, padT + plotH - h0 - 8);
    ctx.fillText(String(g.c1), cx + barW / 2 + 6, padT + plotH - h1 - 8);

    ctx.font = 'bold 16px sans-serif';
    ctx.fillText(g.label, cx, padT + plotH + 28);
  });

  ctx.fillStyle = '#0F172A';
  ctx.font = 'bold 20px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('Distribución de Categorías en Particiones y Balanceo', padL, 30);

  ctx.fillStyle = '#334155';
  ctx.fillRect(canvas.width - 360, 16, 16, 16);
  ctx.font = '16px sans-serif';
  ctx.fillText(`0: ${model.config.negativeClass}`, canvas.width - 336, 30);

  ctx.fillStyle = '#0D9488';
  ctx.fillRect(canvas.width - 185, 16, 16, 16);
  ctx.fillStyle = '#334155';
  ctx.fillText(`1: ${model.config.positiveClass}`, canvas.width - 161, 30);

  return canvas.toDataURL('image/png');
}

function renderDescriptiveHistogramToDataUrl(dataset: DatasetState, colName: string): string {
  if (typeof document === 'undefined') return '';
  const vals: number[] = [];
  for (const r of dataset.workingRows) {
    const n = tryParseNumeric(r[colName]);
    if (n !== null) vals.push(n);
  }
  if (vals.length < 3) return '';

  const min = Math.min(...vals);
  const max = Math.max(...vals);
  if (min === max) return '';

  const numBins = Math.min(12, Math.max(5, Math.ceil(Math.sqrt(vals.length))));
  const width = (max - min) / numBins;
  const counts = Array(numBins).fill(0);
  for (const v of vals) {
    let idx = Math.floor((v - min) / width);
    if (idx >= numBins) idx = numBins - 1;
    if (idx < 0) idx = 0;
    counts[idx]++;
  }

  const canvas = document.createElement('canvas');
  canvas.width = 1100;
  canvas.height = 360;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';

  ctx.fillStyle = '#FFFFFF';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const padL = 75;
  const padR = 35;
  const padT = 52;
  const padB = 56;
  const plotW = canvas.width - padL - padR;
  const plotH = canvas.height - padT - padB;
  const maxC = Math.max(1, ...counts);
  const barW = plotW / numBins;

  ctx.strokeStyle = '#E2E8F0';
  for (let i = 0; i <= 4; i++) {
    const y = padT + (1 - i / 4) * plotH;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + plotW, y);
    ctx.stroke();
    ctx.fillStyle = '#475569';
    ctx.font = '15px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(String(Math.round((maxC * i) / 4)), padL - 10, y + 5);
  }

  counts.forEach((c, i) => {
    const h = (c / maxC) * plotH;
    const x = padL + i * barW + 4;
    const y = padT + plotH - h;
    ctx.fillStyle = '#0D9488';
    ctx.fillRect(x, y, Math.max(4, barW - 8), h);
    ctx.fillStyle = '#0F172A';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    if (c > 0) ctx.fillText(String(c), x + (barW - 8) / 2, y - 6);
  });

  ctx.fillStyle = '#475569';
  ctx.font = '15px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(formatNumberES(min, 1), padL, padT + plotH + 26);
  ctx.textAlign = 'right';
  ctx.fillText(formatNumberES(max, 1), padL + plotW, padT + plotH + 26);

  ctx.fillStyle = '#0F172A';
  ctx.font = 'bold 20px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(`Distribución Descriptiva (Histograma): ${colName}`, padL, 30);

  return canvas.toDataURL('image/png');
}

function renderConfusionMatrixGraphicToDataUrl(
  cmTest: ConfusionMatrix,
  model: ModelResult
): string {
  if (typeof document === 'undefined') return '';
  const canvas = document.createElement('canvas');
  canvas.width = 1100;
  canvas.height = 380;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';

  ctx.fillStyle = '#FFFFFF';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = '#0F172A';
  ctx.font = 'bold 20px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(
    `Matriz de Confusión Visual — Conjunto de Prueba (Corte ${(model.config.threshold * 100).toFixed(0)}%)`,
    40,
    34
  );

  const cells = [
    {
      x: 240,
      y: 95,
      w: 390,
      h: 120,
      bg: '#F1F5F9',
      border: '#94A3B8',
      title: `Verdaderos Negativos (TN)`,
      val: cmTest.tn,
      sub: `Real: ${model.config.negativeClass} · Pred: ${model.config.negativeClass}`,
      color: '#0F172A',
    },
    {
      x: 650,
      y: 95,
      w: 390,
      h: 120,
      bg: '#FEF3C7',
      border: '#F59E0B',
      title: `Falsos Positivos (FP)`,
      val: cmTest.fp,
      sub: `Real: ${model.config.negativeClass} · Pred: ${model.config.positiveClass}`,
      color: '#92400E',
    },
    {
      x: 240,
      y: 230,
      w: 390,
      h: 120,
      bg: '#FFE4E6',
      border: '#F43F5E',
      title: `Falsos Negativos (FN)`,
      val: cmTest.fn,
      sub: `Real: ${model.config.positiveClass} · Pred: ${model.config.negativeClass}`,
      color: '#9F1239',
    },
    {
      x: 650,
      y: 230,
      w: 390,
      h: 120,
      bg: '#CCFBF1',
      border: '#0D9488',
      title: `Verdaderos Positivos (TP)`,
      val: cmTest.tp,
      sub: `Real: ${model.config.positiveClass} · Pred: ${model.config.positiveClass}`,
      color: '#115E59',
    },
  ];

  ctx.fillStyle = '#334155';
  ctx.font = 'bold 16px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(`Predicho 0 (${model.config.negativeClass})`, 435, 76);
  ctx.fillText(`Predicho 1 (${model.config.positiveClass})`, 845, 76);

  ctx.textAlign = 'right';
  ctx.fillText(`Real 0 (${model.config.negativeClass})`, 220, 160);
  ctx.fillText(`Real 1 (${model.config.positiveClass})`, 220, 295);

  for (const c of cells) {
    ctx.fillStyle = c.bg;
    ctx.fillRect(c.x, c.y, c.w, c.h);
    ctx.strokeStyle = c.border;
    ctx.lineWidth = 2;
    ctx.strokeRect(c.x, c.y, c.w, c.h);

    ctx.fillStyle = c.color;
    ctx.textAlign = 'center';
    ctx.font = 'bold 16px sans-serif';
    ctx.fillText(c.title, c.x + c.w / 2, c.y + 30);
    ctx.font = 'bold 34px sans-serif';
    ctx.fillText(String(c.val), c.x + c.w / 2, c.y + 74);
    ctx.font = '14px sans-serif';
    ctx.fillText(c.sub, c.x + c.w / 2, c.y + 102);
  }

  return canvas.toDataURL('image/png');
}

export function generateExecutivePDF(
  dataset: DatasetState,
  model: ModelResult,
  meta: ReportMetadata
): void {
  const doc = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: 'a4',
  });

  const pageW = 210;
  const pageH = 297;
  const margin = 18;
  const contentW = pageW - margin * 2;
  const nowStr = new Date().toLocaleDateString('es-CO', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });

  const cmTrain = computeConfusionMatrix(model.trainPredictions, model.config.threshold);
  const cmTest = computeConfusionMatrix(model.testPredictions, model.config.threshold);
  const mTrain = computeEvaluationMetrics(cmTrain);
  const mTest = computeEvaluationMetrics(cmTest);

  const tocEntries: { title: string; page: number }[] = [];

  // ============ PAGE 1: PORTADA ============
  doc.setFillColor(15, 23, 42); // Dark navy
  doc.rect(0, 0, pageW, 92, 'F');

  doc.setFillColor(13, 148, 136); // Turquoise band
  doc.rect(0, 92, pageW, 3, 'F');

  doc.setTextColor(148, 163, 184);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(10);
  doc.text('PREDICTIVE ANALYTICS STUDIO', margin, 26);

  doc.setTextColor(255, 255, 255);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(22);
  const titleLines = doc.splitTextToSize(meta.projectTitle || 'Informe Ejecutivo de Analítica Predictiva', contentW);
  doc.text(titleLines, margin, 44);

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(12);
  doc.setTextColor(203, 213, 225);
  const subLines = doc.splitTextToSize(
    meta.analysisTitle || `Modelo de Regresión Logística Binaria: ${model.config.targetColumn}`,
    contentW
  );
  doc.text(subLines, margin, 68);

  // Executive Metadata Box on Cover
  doc.setDrawColor(226, 232, 240);
  doc.setFillColor(248, 250, 252);
  doc.roundedRect(margin, 118, contentW, 68, 2, 2, 'FD');

  const coverMeta: [string, string][] = [
    ['Analista / Autor:', meta.studentName || 'No especificado'],
    ['Institución / Org.:', meta.institution || 'No especificada'],
    ['Base de datos:', `${dataset.fileName} (${dataset.workingRows.length} registros · ${dataset.columns.length} variables)`],
    ['Variable objetivo:', `${model.config.targetColumn} (1 = ${model.config.positiveClass} | 0 = ${model.config.negativeClass})`],
    ['Fecha de generación:', nowStr],
  ];

  let cy = 131;
  for (const [k, v] of coverMeta) {
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(10);
    doc.setTextColor(51, 65, 85);
    doc.text(k, margin + 8, cy);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(15, 23, 42);
    const valLines = doc.splitTextToSize(v, contentW - 58);
    doc.text(valLines, margin + 52, cy);
    cy += Math.max(9, valLines.length * 5.5 + 3);
  }

  // Cover KPI summary strip
  const kpiY = 222;
  const kpiW = (contentW - 9) / 4;
  const coverKpis = [
    { label: 'Observaciones', val: String(dataset.workingRows.length) },
    { label: 'Accuracy (Prueba)', val: fmtPct(mTest.accuracy) },
    { label: 'F1-Score (Clase 1)', val: fmtPct(mTest.f1_1) },
    { label: 'Punto de Corte', val: `${(model.config.threshold * 100).toFixed(0)}%` },
  ];

  coverKpis.forEach((k, i) => {
    const kx = margin + i * (kpiW + 3);
    doc.setDrawColor(203, 213, 225);
    doc.setFillColor(255, 255, 255);
    doc.roundedRect(kx, kpiY, kpiW, 24, 1.5, 1.5, 'FD');
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8.5);
    doc.setTextColor(100, 116, 139);
    doc.text(k.label, kx + 4, kpiY + 8);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(13);
    doc.setTextColor(15, 23, 42);
    doc.text(k.val, kx + 4, kpiY + 18);
  });

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9);
  doc.setTextColor(100, 116, 139);
  doc.text(
    'Documento generado automáticamente con cálculos reales de máxima verosimilitud (MLE).',
    margin,
    274
  );

  // ============ PAGE 2: ÍNDICE (RESERVED) ============
  doc.addPage();
  const tocPageNumber = doc.getNumberOfPages();

  // ============ CONTENT PAGES ============
  doc.addPage();
  let y = 26;

  const ensureSpace = (neededMm: number) => {
    if (y + neededMm > pageH - 22) {
      doc.addPage();
      y = 26;
    }
  };

  const addSectionTitle = (numTitle: string) => {
    ensureSpace(18);
    tocEntries.push({
      title: numTitle,
      page: doc.getNumberOfPages(),
    });
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(13.5);
    doc.setTextColor(15, 23, 42);
    doc.text(numTitle, margin, y);
    y += 2.5;
    doc.setDrawColor(13, 148, 136);
    doc.setLineWidth(0.6);
    doc.line(margin, y, margin + contentW, y);
    y += 7;
  };

  const addParagraph = (text: string, fontSize = 10, color: [number, number, number] = [51, 65, 85]) => {
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(fontSize);
    doc.setTextColor(...color);
    const lines = doc.splitTextToSize(text, contentW);
    for (const line of lines) {
      ensureSpace(6);
      doc.text(line, margin, y);
      y += 5.2;
    }
    y += 2.5;
  };

  const addSubheading = (text: string) => {
    ensureSpace(10);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(10.5);
    doc.setTextColor(15, 23, 42);
    doc.text(text, margin, y);
    y += 5.5;
  };

  // --- 1. RESUMEN EJECUTIVO ---
  addSectionTitle('1. Resumen Ejecutivo');
  const posRateTest =
    cmTest.total > 0 ? ((cmTest.tp + cmTest.fn) / cmTest.total) * 100 : 0;
  addParagraph(
    `El presente informe documenta el análisis descriptivo, la preparación de datos y la construcción de un modelo predictivo de regresión logística binaria sobre la base "${dataset.fileName}" (${dataset.workingRows.length} registros y ${dataset.columns.length} variables), con el objetivo de estimar la probabilidad del evento "${model.config.positiveClass}" (codificado como 1) frente a "${model.config.negativeClass}" (codificado como 0) en la variable dependiente "${model.config.targetColumn}".`
  );
  addParagraph(
    `En el conjunto de prueba independiente (${model.testCount} observaciones, donde el evento positivo representa el ${posRateTest.toFixed(
      1
    )}%), el modelo alcanzó un Accuracy de ${fmtPct(mTest.accuracy)}, una Precisión para la clase positiva de ${fmtPct(
      mTest.precision1
    )}, un Recall (sensibilidad) de ${fmtPct(mTest.recall1)} y un F1-Score de ${fmtPct(
      mTest.f1_1
    )} utilizando un punto de corte de ${(model.config.threshold * 100).toFixed(0)}%.`
  );

  // --- 2. DESCRIPCIÓN DE LA BASE DE DATOS ---
  addSectionTitle('2. Descripción de la Base de Datos');
  addParagraph(
    `Archivo fuente: ${dataset.fileName}. Total de observaciones activas: ${dataset.workingRows.length}. Número de columnas: ${dataset.columns.length}. Registros duplicados detectados: ${dataset.duplicateCount}.`
  );

  autoTable(doc, {
    startY: y,
    margin: { left: margin, right: margin },
    head: [['Variable', 'Tipo Detectado', 'Válidos', 'Faltantes', 'Únicos', 'Muestra']],
    body: dataset.profiles.map((p) => [
      p.name,
      p.effectiveKind === 'numeric_continuous'
        ? 'Numérica continua'
        : p.effectiveKind === 'numeric_discrete'
        ? 'Numérica discreta'
        : p.effectiveKind === 'binary'
        ? 'Binaria'
        : 'Categórica',
      String(p.validCount),
      String(p.missingCount),
      String(p.uniqueCount),
      p.sampleValues.slice(0, 3).join(', '),
    ]),
    styles: { fontSize: 8.5, cellPadding: 2.2 },
    headStyles: { fillColor: [15, 23, 42], textColor: 255, fontStyle: 'bold' },
    alternateRowStyles: { fillColor: [248, 250, 252] },
  });
  y = (doc as unknown as { lastAutoTable: { finalY: number } }).lastAutoTable.finalY + 8;

  // --- 3. ANÁLISIS DESCRIPTIVO ---
  addSectionTitle('3. Análisis Descriptivo');
  const numericProfiles = dataset.profiles.filter((p) => p.numericStats !== null);
  if (numericProfiles.length > 0) {
    autoTable(doc, {
      startY: y,
      margin: { left: margin, right: margin },
      head: [['Variable', 'Media', 'Mediana', 'Desv. Est.', 'Mín.', 'Máx.', 'Q1 / Q3', 'Atípicos']],
      body: numericProfiles.map((p) => {
        const s = p.numericStats!;
        return [
          p.name,
          formatNumberES(s.mean, 2),
          formatNumberES(s.median, 2),
          formatNumberES(s.stdDev, 2),
          formatNumberES(s.min, 2),
          formatNumberES(s.max, 2),
          `${formatNumberES(s.q1, 1)} / ${formatNumberES(s.q3, 1)}`,
          String(s.outlierCount),
        ];
      }),
      styles: { fontSize: 8, cellPadding: 2 },
      headStyles: { fillColor: [15, 23, 42], textColor: 255, fontStyle: 'bold' },
      alternateRowStyles: { fillColor: [248, 250, 252] },
    });
    y = (doc as unknown as { lastAutoTable: { finalY: number } }).lastAutoTable.finalY + 6;
  }

  for (const p of dataset.profiles.slice(0, 6)) {
    addParagraph(`• ${p.interpretation}`, 9);
  }

  const primaryNumCol =
    model.config.featureColumns.find((c) =>
      numericProfiles.some((np) => np.name === c)
    ) || numericProfiles[0]?.name;
  if (primaryNumCol) {
    const histImg = renderDescriptiveHistogramToDataUrl(dataset, primaryNumCol);
    if (histImg) {
      ensureSpace(64);
      doc.addImage(histImg, 'PNG', margin, y, contentW, 56);
      y += 62;
    }
  }

  // --- 4. METODOLOGÍA ---
  addSectionTitle('4. Metodología');
  addParagraph(
    `• Variable dependiente: ${model.config.targetColumn} (1 = "${model.config.positiveClass}", 0 = "${model.config.negativeClass}").`
  );
  addParagraph(`• Variables independientes: ${model.config.featureColumns.join(', ')}.`);
  addParagraph(
    `• Partición de datos: ${(model.config.splitConfig.trainRatio * 100).toFixed(0)}% entrenamiento (${
      model.trainCount
    } registros) y ${((1 - model.config.splitConfig.trainRatio) * 100).toFixed(0)}% prueba (${
      model.testCount
    } registros), semilla = ${model.config.splitConfig.seed}, estratificación = ${
      model.config.splitConfig.stratified ? 'Sí' : 'No'
    }.`
  );
  addParagraph(
    `• Preparación sin fuga de información: Manejo de faltantes (${model.config.prepConfig.missingStrategy}), tratamiento de atípicos (${model.config.prepConfig.outlierStrategy}), estandarización numérica (${
      model.config.prepConfig.standardizeNumeric ? 'Sí' : 'Escala original'
    }) y balanceo exclusivo en entrenamiento (${model.config.balanceMethod}).`
  );
  addParagraph(
    `• Estimación: Regresión logística binaria por Máxima Verosimilitud (IRLS / Newton-Raphson). Función logística P(Y=1) = 1 / (1 + e^(-Z)), con umbral de clasificación = ${(
      model.config.threshold * 100
    ).toFixed(0)}%.`
  );

  const distImg = renderClassDistChartToDataUrl(model);
  if (distImg) {
    ensureSpace(66);
    doc.addImage(distImg, 'PNG', margin, y, contentW, 58);
    y += 64;
  }

  // --- 5. RESULTADOS DEL MODELO ---
  addSectionTitle('5. Resultados del Modelo de Regresión Logística');
  const eqTerms = model.coefficients
    .map((c, idx) => {
      const bStr = c.beta.toFixed(4);
      if (idx === 0) return bStr;
      const sign = c.beta >= 0 ? ' + ' : ' - ';
      return `${sign}${Math.abs(c.beta).toFixed(4)} * ${c.term}`;
    })
    .join('');

  doc.setFillColor(248, 250, 252);
  doc.setDrawColor(203, 213, 225);
  const eqLines = doc.splitTextToSize(`Z = ${eqTerms}`, contentW - 8);
  ensureSpace(eqLines.length * 5 + 18);
  doc.roundedRect(margin, y, contentW, eqLines.length * 5 + 12, 1.5, 1.5, 'FD');
  doc.setFont('courier', 'bold');
  doc.setFontSize(9);
  doc.setTextColor(15, 23, 42);
  doc.text(eqLines, margin + 4, y + 6);
  doc.setFont('courier', 'normal');
  doc.text('P(Y = 1) = 1 / (1 + exp(-Z))', margin + 4, y + eqLines.length * 5 + 8);
  y += eqLines.length * 5 + 16;

  autoTable(doc, {
    startY: y,
    margin: { left: margin, right: margin },
    head: [['Término', 'Beta (β)', 'Error Est.', 'Estadístico z', 'p-valor', 'Odds Ratio', 'IC 95% OR']],
    body: model.coefficients.map((c) => [
      c.term,
      c.beta.toFixed(4),
      fmtDec(c.stdError, 4),
      fmtDec(c.zValue, 3),
      c.pValue !== null ? (c.pValue < 0.0001 ? '< 0.0001' : c.pValue.toFixed(4)) : 'N/D',
      c.oddsRatio.toFixed(4),
      c.ciLower95 !== null && c.ciUpper95 !== null
        ? `[${c.ciLower95.toFixed(2)}, ${c.ciUpper95.toFixed(2)}]`
        : 'N/D',
    ]),
    styles: { fontSize: 8, cellPadding: 2 },
    headStyles: { fillColor: [15, 23, 42], textColor: 255, fontStyle: 'bold' },
    alternateRowStyles: { fillColor: [248, 250, 252] },
  });
  y = (doc as unknown as { lastAutoTable: { finalY: number } }).lastAutoTable.finalY + 6;

  addParagraph(
    `Ajuste global: Log-Verosimilitud = ${model.logLikelihood.toFixed(
      2
    )} | Devianza Nula = ${model.nullDeviance.toFixed(
      2
    )} | Devianza Residual = ${model.residualDeviance.toFixed(2)} | AIC = ${model.aic.toFixed(
      2
    )} | Pseudo-R² McFadden = ${(model.mcfaddenR2 * 100).toFixed(2)}% | Convergencia = ${
      model.converged ? `Sí (${model.iterations} iteraciones)` : 'No estricta'
    }.`
  );

  const sigImg = renderSigmoidChartToDataUrl(model);
  if (sigImg) {
    ensureSpace(80);
    doc.addImage(sigImg, 'PNG', margin, y, contentW, 74);
    y += 80;
  }

  // --- 6. MATRIZ DE CONFUSIÓN ---
  addSectionTitle('6. Matriz de Confusión (Entrenamiento y Prueba)');
  autoTable(doc, {
    startY: y,
    margin: { left: margin, right: margin },
    head: [['Conjunto', 'Verdaderos Positivos (TP)', 'Verdaderos Negativos (TN)', 'Falsos Positivos (FP)', 'Falsos Negativos (FN)', 'Total']],
    body: [
      ['Entrenamiento', String(cmTrain.tp), String(cmTrain.tn), String(cmTrain.fp), String(cmTrain.fn), String(cmTrain.total)],
      ['Prueba (Test)', String(cmTest.tp), String(cmTest.tn), String(cmTest.fp), String(cmTest.fn), String(cmTest.total)],
    ],
    styles: { fontSize: 8.5, cellPadding: 2.5 },
    headStyles: { fillColor: [15, 23, 42], textColor: 255, fontStyle: 'bold' },
  });
  y = (doc as unknown as { lastAutoTable: { finalY: number } }).lastAutoTable.finalY + 7;

  const cmImg = renderConfusionMatrixGraphicToDataUrl(cmTest, model);
  if (cmImg) {
    ensureSpace(66);
    doc.addImage(cmImg, 'PNG', margin, y, contentW, 58);
    y += 64;
  }

  // --- 7. MÉTRICAS DE EVALUACIÓN ---
  addSectionTitle('7. Métricas de Evaluación');
  autoTable(doc, {
    startY: y,
    margin: { left: margin, right: margin },
    head: [['Métrica', 'Entrenamiento (%)', 'Entrenamiento (Dec.)', 'Prueba (%)', 'Prueba (Dec.)']],
    body: [
      ['Accuracy', fmtPct(mTrain.accuracy), fmtDec(mTrain.accuracy), fmtPct(mTest.accuracy), fmtDec(mTest.accuracy)],
      ['Error de clasificación', fmtPct(mTrain.classificationError), fmtDec(mTrain.classificationError), fmtPct(mTest.classificationError), fmtDec(mTest.classificationError)],
      ['Precision (Clase 1)', fmtPct(mTrain.precision1), fmtDec(mTrain.precision1), fmtPct(mTest.precision1), fmtDec(mTest.precision1)],
      ['Precision (Clase 0)', fmtPct(mTrain.precision0), fmtDec(mTrain.precision0), fmtPct(mTest.precision0), fmtDec(mTest.precision0)],
      ['Recall (Clase 1)', fmtPct(mTrain.recall1), fmtDec(mTrain.recall1), fmtPct(mTest.recall1), fmtDec(mTest.recall1)],
      ['Recall (Clase 0)', fmtPct(mTrain.recall0), fmtDec(mTrain.recall0), fmtPct(mTest.recall0), fmtDec(mTest.recall0)],
      ['F1-Score (Clase 1)', fmtPct(mTrain.f1_1), fmtDec(mTrain.f1_1), fmtPct(mTest.f1_1), fmtDec(mTest.f1_1)],
      ['F1-Score (Clase 0)', fmtPct(mTrain.f1_0), fmtDec(mTrain.f1_0), fmtPct(mTest.f1_0), fmtDec(mTest.f1_0)],
      ['Balanced Accuracy', fmtPct(mTrain.balancedAccuracy), fmtDec(mTrain.balancedAccuracy), fmtPct(mTest.balancedAccuracy), fmtDec(mTest.balancedAccuracy)],
    ],
    styles: { fontSize: 8.5, cellPadding: 2.2 },
    headStyles: { fillColor: [15, 23, 42], textColor: 255, fontStyle: 'bold' },
    alternateRowStyles: { fillColor: [248, 250, 252] },
  });
  y = (doc as unknown as { lastAutoTable: { finalY: number } }).lastAutoTable.finalY + 6;

  const undefEntries = Object.entries(mTest.undefinedReasons);
  if (undefEntries.length > 0) {
    for (const [, reason] of undefEntries) {
      if (reason) addParagraph(`• Nota de métrica no definida (Prueba): ${reason}`, 8.5, [159, 18, 57]);
    }
  }

  const barImg = renderMetricsBarChartToDataUrl(mTrain, mTest);
  if (barImg) {
    ensureSpace(78);
    doc.addImage(barImg, 'PNG', margin, y, contentW, 72);
    y += 78;
  }

  // --- 8. VALIDACIÓN DEL MODELO ---
  addSectionTitle('8. Validación del Modelo y Estabilidad');
  const accGap =
    mTrain.accuracy !== null && mTest.accuracy !== null
      ? (mTrain.accuracy - mTest.accuracy) * 100
      : null;
  const stabilityText =
    accGap === null
      ? 'No fue posible calcular la brecha entre entrenamiento y prueba.'
      : Math.abs(accGap) <= 5
      ? `La diferencia de Accuracy entre entrenamiento y prueba es de ${accGap.toFixed(
          2
        )} puntos porcentuales, lo que sugiere un comportamiento estable y buena capacidad de generalización sobre datos no vistos.`
      : accGap > 5
      ? `El Accuracy en entrenamiento supera al de prueba por ${accGap.toFixed(
          2
        )} puntos porcentuales, lo cual advierte posible sobreajuste (overfitting); se recomienda evaluar menos variables o revisar la muestra.`
      : `El Accuracy en prueba supera al de entrenamiento por ${Math.abs(accGap).toFixed(
          2
        )} puntos porcentuales, atribuible a la variabilidad muestral de la partición.`;

  addParagraph(
    `El modelo obtuvo un Accuracy de ${fmtPct(mTrain.accuracy)} en entrenamiento y ${fmtPct(
      mTest.accuracy
    )} en prueba. ${stabilityText}`
  );
  addParagraph(
    `Para la categoría positiva de interés ("${model.config.positiveClass}"), el F1-Score en prueba fue ${fmtPct(
      mTest.f1_1
    )} (Precision = ${fmtPct(mTest.precision1)}, Recall = ${fmtPct(
      mTest.recall1
    )}). En escenarios con desbalance o alto costo de falsos negativos, esta métrica es más informativa que el Accuracy global.`
  );

  // --- 9. RESULTADOS EMPRESARIALES ---
  addSectionTitle('9. Resultados Empresariales e Interpretación');
  const sigPredictors = model.coefficients
    .slice(1)
    .filter((c) => c.pValue !== null && c.pValue < 0.05)
    .sort((a, b) => (a.pValue ?? 1) - (b.pValue ?? 1));

  if (sigPredictors.length > 0) {
    addSubheading('Variables con mayor asociación estadística (p < 0,05):');
    for (const c of sigPredictors) {
      const dir = c.beta >= 0 ? 'aumenta' : 'reduce';
      addParagraph(
        `• ${c.term}: β = ${c.beta.toFixed(4)}, Odds Ratio = ${c.oddsRatio.toFixed(
          4
        )} (p = ${c.pValue! < 0.0001 ? '<0,0001' : c.pValue!.toFixed(4)}). Por cada unidad adicional, los odds de "${
          model.config.positiveClass
        }" se multiplican por ${c.oddsRatio.toFixed(3)}, lo que ${dir} la probabilidad estimada manteniendo constantes las demás variables.`,
        9.5
      );
    }
  } else {
    addParagraph(
      'Ningún predictor individual alcanzó significancia estadística estricta al 5% (p < 0,05) en esta especificación; evalúe la combinación de variables o el tamaño de muestra.'
    );
  }

  // Top 5 highest risk records in test
  const topRisk = [...model.testPredictions]
    .sort((a, b) => b.probability - a.probability)
    .slice(0, 5);

  if (topRisk.length > 0) {
    addSubheading('Observaciones de prueba con mayor probabilidad estimada del evento (1):');
    autoTable(doc, {
      startY: y,
      margin: { left: margin, right: margin },
      head: [['Fila', 'Valor Real', 'Valor Z', 'Probabilidad P(Y=1)', 'Clasificación']],
      body: topRisk.map((r) => [
        String(r.rowIndex),
        r.actualLabel,
        r.zValue.toFixed(3),
        `${(r.probability * 100).toFixed(2)}%`,
        r.probability >= model.config.threshold
          ? model.config.positiveClass
          : model.config.negativeClass,
      ]),
      styles: { fontSize: 8.5, cellPadding: 2 },
      headStyles: { fillColor: [13, 148, 136], textColor: 255, fontStyle: 'bold' },
    });
    y = (doc as unknown as { lastAutoTable: { finalY: number } }).lastAutoTable.finalY + 7;
  }

  // --- 10. CONCLUSIONES Y RECOMENDACIONES ---
  addSectionTitle('10. Conclusiones, Recomendaciones y Limitaciones');
  addParagraph(
    `Conclusión general: El modelo de regresión logística "${model.name}" alcanzó un Accuracy de ${fmtPct(
      mTest.accuracy
    )}, Balanced Accuracy de ${fmtPct(mTest.balancedAccuracy)} y F1-Score de ${fmtPct(
      mTest.f1_1
    )} para la categoría "${model.config.positiveClass}" en el conjunto de prueba independiente (${
      model.testCount
    } registros).${
      sigPredictors.length > 0
        ? ` Los factores con mayor significancia estadística (p < 0,05) fueron: ${sigPredictors
            .map((s) => `${s.term} (OR=${s.oddsRatio.toFixed(2)})`)
            .join(', ')}.`
        : ''
    }`
  );
  addParagraph(
    `1. Priorización operativa: Utilizar las probabilidades individuales P(Y=1) generadas por el modelo para segmentar los registros en niveles de riesgo/propensión y concentrar acciones de seguimiento en el decil superior.`
  );
  addParagraph(
    `2. Calibración del punto de corte: El umbral actual es ${(model.config.threshold * 100).toFixed(
      0
    )}%. Si el costo empresarial de un falso negativo (${cmTest.fn} casos en prueba) supera al de un falso positivo (${
      cmTest.fp
    } casos en prueba), considere reducir el umbral para incrementar el Recall.`
  );
  addParagraph(
    `3. Limitación metodológica: Las asociaciones estadísticas identificadas mediante regresión logística son correlacionales dentro de la muestra analizada y no implican causalidad directa sin un diseño experimental o validación adicional.`
  );

  // --- 11. ANEXOS ---
  addSectionTitle('11. Anexos Técnicos');
  autoTable(doc, {
    startY: y,
    margin: { left: margin, right: margin },
    head: [['Parámetro de Configuración', 'Valor Aplicado']],
    body: [
      ['Nombre del modelo', model.name],
      ['Archivo de datos', dataset.fileName],
      ['Proporción Train / Test', `${(model.config.splitConfig.trainRatio * 100).toFixed(0)}% / ${((1 - model.config.splitConfig.trainRatio) * 100).toFixed(0)}%`],
      ['Semilla aleatoria', String(model.config.splitConfig.seed)],
      ['División estratificada', model.config.splitConfig.stratified ? 'Activada' : 'Desactivada'],
      ['Balanceo (solo Train)', model.config.balanceMethod],
      ['Manejo de faltantes', model.config.prepConfig.missingStrategy],
      ['Tratamiento de atípicos', model.config.prepConfig.outlierStrategy],
      ['Estandarización numérica', model.config.prepConfig.standardizeNumeric ? 'Activada (Z-score)' : 'Desactivada (escala original)'],
      ['Registros excluidos por faltantes', `Train: ${model.droppedTrainRows} | Test: ${model.droppedTestRows}`],
    ],
    styles: { fontSize: 8.5, cellPadding: 2.2 },
    headStyles: { fillColor: [15, 23, 42], textColor: 255, fontStyle: 'bold' },
  });

  // ============ FILL PAGE 2: TABLE OF CONTENTS ============
  doc.setPage(tocPageNumber);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(16);
  doc.setTextColor(15, 23, 42);
  doc.text('Índice de Contenidos', margin, 32);
  doc.setDrawColor(13, 148, 136);
  doc.setLineWidth(0.7);
  doc.line(margin, 36, margin + contentW, 36);

  let tocY = 48;
  for (const entry of tocEntries) {
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(10.5);
    doc.setTextColor(30, 41, 59);
    doc.text(entry.title, margin, tocY);
    doc.setFont('helvetica', 'bold');
    doc.text(String(entry.page), pageW - margin, tocY, { align: 'right' });
    doc.setDrawColor(226, 232, 240);
    doc.setLineWidth(0.2);
    doc.line(margin, tocY + 2, pageW - margin, tocY + 2);
    tocY += 10;
  }

  // ============ HEADERS & FOOTERS ON ALL CONTENT PAGES ============
  const totalPages = doc.getNumberOfPages();
  for (let p = 2; p <= totalPages; p++) {
    doc.setPage(p);
    // Header
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8);
    doc.setTextColor(100, 116, 139);
    doc.text('Predictive Analytics Studio · Informe Ejecutivo', margin, 12);
    doc.text(dataset.fileName, pageW - margin, 12, { align: 'right' });
    doc.setDrawColor(226, 232, 240);
    doc.setLineWidth(0.3);
    doc.line(margin, 15, pageW - margin, 15);

    // Footer
    doc.line(margin, pageH - 14, pageW - margin, pageH - 14);
    doc.text(
      `${meta.institution || 'Predictive Analytics Studio'} · ${meta.studentName || ''}`,
      margin,
      pageH - 9
    );
    doc.text(`Página ${p} de ${totalPages}`, pageW - margin, pageH - 9, {
      align: 'right',
    });
  }

  const safeName = (meta.projectTitle || 'informe_ejecutivo')
    .toLowerCase()
    .replace(/[^a-z0-9]+/gi, '_')
    .replace(/^_|_$/g, '');
  doc.save(`${safeName || 'informe_ejecutivo'}.pdf`);
}
