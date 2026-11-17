import React, { useState } from 'react';
import type { EvaluationMetrics, ModelResult, ModuleId } from '../../types/analytics';
import {
  computeConfusionMatrix,
  computeEvaluationMetrics,
} from '../../engine/logisticRegression';

interface MetricsViewProps {
  activeModel: ModelResult | null;
  onNavigate: (mod: ModuleId) => void;
}

function renderValue(v: number | null, reason?: string) {
  if (v === null || !Number.isFinite(v)) {
    return (
      <div>
        <span className="font-mono text-sm font-semibold text-rose-700">No definida</span>
        {reason && <p className="mt-1 text-[11px] text-rose-600">{reason}</p>}
      </div>
    );
  }
  return (
    <div className="flex items-baseline gap-2 tabular-nums">
      <span className="font-mono text-xl font-bold text-slate-900">
        {(v * 100).toFixed(2)}%
      </span>
      <span className="font-mono text-xs text-slate-500">({v.toFixed(4)})</span>
    </div>
  );
}

export const MetricsView: React.FC<MetricsViewProps> = ({
  activeModel,
  onNavigate,
}) => {
  const [split, setSplit] = useState<'test' | 'train'>('test');

  if (!activeModel || activeModel.error) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-10 text-center">
        <h2 className="text-lg font-semibold text-slate-900">Sin modelo para evaluar</h2>
        <p className="mt-1 text-sm text-slate-600">
          Estime primero un modelo de regresión logística para calcular Accuracy, Precision, Recall, F1-Score y Balanced Accuracy.
        </p>
        <button
          type="button"
          onClick={() => onNavigate('regresion')}
          className="mt-5 rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white"
        >
          Ir a Regresión Logística
        </button>
      </div>
    );
  }

  const records = split === 'test' ? activeModel.testPredictions : activeModel.trainPredictions;
  const cm = computeConfusionMatrix(records, activeModel.config.threshold);
  const m: EvaluationMetrics = computeEvaluationMetrics(cm);

  const cards: {
    title: string;
    formula: string;
    val: number | null;
    reason?: string;
    highlight?: boolean;
  }[] = [
    {
      title: 'Accuracy general',
      formula: '(TP + TN) / (TP + TN + FP + FN)',
      val: m.accuracy,
      reason: m.undefinedReasons.accuracy,
    },
    {
      title: 'Error de clasificación',
      formula: '1 - Accuracy',
      val: m.classificationError,
      reason: m.undefinedReasons.classificationError,
    },
    {
      title: 'Balanced Accuracy',
      formula: '(Recall(0) + Recall(1)) / 2',
      val: m.balancedAccuracy,
      reason: m.undefinedReasons.balancedAccuracy,
      highlight: true,
    },
    {
      title: `Precision — Categoría 1 (${activeModel.config.positiveClass})`,
      formula: 'TP / (TP + FP)',
      val: m.precision1,
      reason: m.undefinedReasons.precision1,
    },
    {
      title: `Recall — Categoría 1 (${activeModel.config.positiveClass})`,
      formula: 'TP / (TP + FN)',
      val: m.recall1,
      reason: m.undefinedReasons.recall1,
    },
    {
      title: `F1-Score — Categoría 1 (${activeModel.config.positiveClass})`,
      formula: '2 × (Precision × Recall) / (Precision + Recall)',
      val: m.f1_1,
      reason: m.undefinedReasons.f1_1,
      highlight: true,
    },
    {
      title: `Precision — Categoría 0 (${activeModel.config.negativeClass})`,
      formula: 'TN / (TN + FN)',
      val: m.precision0,
      reason: m.undefinedReasons.precision0,
    },
    {
      title: `Recall — Categoría 0 (${activeModel.config.negativeClass})`,
      formula: 'TN / (TN + FP)',
      val: m.recall0,
      reason: m.undefinedReasons.recall0,
    },
    {
      title: `F1-Score — Categoría 0 (${activeModel.config.negativeClass})`,
      formula: '2 × (Precision₀ × Recall₀) / (Precision₀ + Recall₀)',
      val: m.f1_0,
      reason: m.undefinedReasons.f1_0,
    },
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-900">Métricas de Evaluación del Modelo</h1>
          <p className="text-sm text-slate-600">
            Cálculo exacto en porcentaje y decimal a partir de TP={cm.tp}, TN={cm.tn}, FP={cm.fp}, FN={cm.fn}.
          </p>
        </div>

        <div className="inline-flex rounded-lg bg-slate-100 p-1 text-xs shrink-0">
          <button
            type="button"
            onClick={() => setSplit('test')}
            className={`rounded-md px-3 py-1.5 font-medium ${
              split === 'test' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600'
            }`}
          >
            Conjunto de Prueba ({activeModel.testCount})
          </button>
          <button
            type="button"
            onClick={() => setSplit('train')}
            className={`rounded-md px-3 py-1.5 font-medium ${
              split === 'train' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600'
            }`}
          >
            Entrenamiento ({activeModel.trainCount})
          </button>
        </div>
      </div>

      {/* Grid de métricas */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        {cards.map((c) => (
          <div
            key={c.title}
            className={`rounded-xl border p-5 ${
              c.highlight
                ? 'border-teal-300 bg-teal-50/35'
                : 'border-slate-200 bg-white'
            }`}
          >
            <p className="text-xs font-semibold text-slate-800">{c.title}</p>
            <p className="mt-1 font-mono text-[11px] text-slate-500">{c.formula}</p>
            <div className="mt-3">{renderValue(c.val, c.reason)}</div>
          </div>
        ))}
      </div>

      {/* Explicaciones conceptuales obligatorias de la sección 10 */}
      <section className="rounded-xl border border-slate-200 bg-white p-6 space-y-4">
        <h2 className="text-base font-semibold text-slate-900">
          Análisis Crítico de las Métricas (Conceptos Clave)
        </h2>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 text-xs leading-relaxed text-slate-700">
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 space-y-1.5">
            <h3 className="font-semibold text-slate-900">
              1. ¿Por qué un modelo puede tener 99% de Accuracy y no servir para la categoría de interés?
            </h3>
            <p>
              Cuando existe un desbalanceo severo (por ejemplo, 99% de casos sin retraso y 1% con retraso), un modelo trivial que prediga &quot;0&quot; para todos obtendría 99% de Accuracy, pero <strong>0% de Recall y F1-Score no definido</strong> para la categoría 1 que realmente interesa detectar.
            </p>
          </div>

          <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 space-y-1.5">
            <h3 className="font-semibold text-slate-900">
              2. ¿Por qué el F1-Score evalúa conjuntamente Precision y Recall?
            </h3>
            <p>
              El F1-Score es la <strong>media armónica</strong> entre Precision y Recall. A diferencia del promedio aritmético, penaliza severamente si una de las dos métricas es muy baja, garantizando que el modelo tanto acierte cuando predice 1 como capture los casos reales 1.
            </p>
          </div>

          <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 space-y-1.5">
            <h3 className="font-semibold text-slate-900">
              3. ¿Qué significa tener muchos Falsos Negativos (FN)?
            </h3>
            <p>
              Indica que el modelo deja escapar casos positivos reales clasificándolos como negativos (bajo Recall). En cartera o salud, equivale a no detectar clientes que caerán en mora o pacientes en riesgo.
            </p>
          </div>

          <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 space-y-1.5">
            <h3 className="font-semibold text-slate-900">
              4. ¿Qué significa tener muchos Falsos Positivos (FP)?
            </h3>
            <p>
              Indica exceso de falsas alarmas (baja Precision): el modelo predice la clase positiva en registros que realmente son negativos, generando costos operativos o fricción comercial innecesaria.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};
