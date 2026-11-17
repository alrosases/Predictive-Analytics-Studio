import React, { useState } from 'react';
import type { ModelResult, ModuleId } from '../../types/analytics';
import { computeConfusionMatrix } from '../../engine/logisticRegression';

interface ConfusionMatrixViewProps {
  activeModel: ModelResult | null;
  onThresholdChange: (t: number) => void;
  onNavigate: (mod: ModuleId) => void;
}

const CLASS_EXAMPLES = [
  {
    title: 'Gatos y perros',
    pos: 'Perro (1)',
    neg: 'Gato (0)',
    fp: 'Gato clasificado erróneamente como perro.',
    fn: 'Perro clasificado erróneamente como gato.',
  },
  {
    title: 'Compra y no compra',
    pos: 'Compra (1)',
    neg: 'No compra (0)',
    fp: 'Cliente que no compra pero recibe campaña costosa creyendo que sí comprará.',
    fn: 'Cliente potencial comprador que el modelo descarta y no es contactado.',
  },
  {
    title: 'Pago y retraso (Cartera)',
    pos: 'Presenta retraso (1)',
    neg: 'Paga a tiempo (0)',
    fp: 'Cliente cumplido tratado preventivamente como de alto riesgo.',
    fn: 'Cliente moroso que no fue alertado a tiempo (riesgo financiero directo).',
  },
  {
    title: 'Tiene enfermedad y no tiene enfermedad',
    pos: 'Enfermo (1)',
    neg: 'Sano (0)',
    fp: 'Persona sana diagnosticada erróneamente como enferma (alarma falsa).',
    fn: 'Paciente enfermo dado de alta sin tratamiento (error de alto costo).',
  },
];

export const ConfusionMatrixView: React.FC<ConfusionMatrixViewProps> = ({
  activeModel,
  onThresholdChange,
  onNavigate,
}) => {
  const [split, setSplit] = useState<'test' | 'train'>('test');
  const [perspective, setPerspective] = useState<'counts' | 'rows' | 'cols'>('counts');

  if (!activeModel || activeModel.error) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-10 text-center">
        <h2 className="text-lg font-semibold text-slate-900">No hay modelo activo</h2>
        <p className="mt-1 text-sm text-slate-600">
          Estime primero un modelo de regresión logística para construir su matriz de confusión.
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

  const row1Total = cm.tp + cm.fn;
  const row0Total = cm.tn + cm.fp;
  const col1Total = cm.tp + cm.fp;
  const col0Total = cm.tn + cm.fn;

  const formatCell = (count: number, rowSum: number, colSum: number) => {
    if (perspective === 'counts') return `${count}`;
    if (perspective === 'rows') {
      return rowSum > 0 ? `${((count / rowSum) * 100).toFixed(1)}%` : 'N/D';
    }
    return colSum > 0 ? `${((count / colSum) * 100).toFixed(1)}%` : 'N/D';
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-900">Matriz de Confusión Interactiva</h1>
          <p className="text-sm text-slate-600">
            Comparación directa entre valores reales y valores predichos para comprender aciertos y errores por categoría.
          </p>
        </div>
        <button
          type="button"
          onClick={() => onNavigate('metricas')}
          className="rounded-lg bg-teal-600 px-4 py-2 text-xs font-medium text-white hover:bg-teal-700 whitespace-nowrap shrink-0"
        >
          Ver Métricas de Evaluación →
        </button>
      </div>

      {/* Controles */}
      <section className="rounded-xl border border-slate-200 bg-white p-5">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="inline-flex rounded-lg bg-slate-100 p-1 text-xs">
            <button
              type="button"
              onClick={() => setSplit('test')}
              className={`rounded-md px-3 py-1.5 font-medium ${
                split === 'test' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600'
              }`}
            >
              Prueba ({activeModel.testCount})
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

          <div className="inline-flex rounded-lg bg-slate-100 p-1 text-xs">
            <button
              type="button"
              onClick={() => setPerspective('counts')}
              className={`rounded-md px-3 py-1.5 font-medium ${
                perspective === 'counts' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600'
              }`}
            >
              Conteos absolutos
            </button>
            <button
              type="button"
              onClick={() => setPerspective('rows')}
              className={`rounded-md px-3 py-1.5 font-medium ${
                perspective === 'rows' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600'
              }`}
            >
              Por filas (Reales / Recall)
            </button>
            <button
              type="button"
              onClick={() => setPerspective('cols')}
              className={`rounded-md px-3 py-1.5 font-medium ${
                perspective === 'cols' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600'
              }`}
            >
              Por columnas (Predichos / Precision)
            </button>
          </div>

          <div className="flex items-center gap-2 text-xs">
            <span className="text-slate-600">
              Corte:{' '}
              <strong className="font-mono text-teal-700">
                {(activeModel.config.threshold * 100).toFixed(0)}%
              </strong>
            </span>
            <input
              type="range"
              min={0.05}
              max={0.95}
              step={0.01}
              value={activeModel.config.threshold}
              onChange={(e) => onThresholdChange(Number(e.target.value))}
              className="w-28 accent-teal-600"
            />
          </div>
        </div>

        {/* Matriz visual 2x2 */}
        <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-12">
          <div className="lg:col-span-7">
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-center text-xs tabular-nums">
                <thead>
                  <tr>
                    <th className="p-2 text-left text-slate-400">Real \ Predicho</th>
                    <th className="p-2 font-semibold text-slate-800">
                      Predicho 0 ({activeModel.config.negativeClass})
                    </th>
                    <th className="p-2 font-semibold text-teal-800">
                      Predicho 1 ({activeModel.config.positiveClass})
                    </th>
                    <th className="p-2 text-right text-slate-500">Total Real</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="p-3 text-left font-semibold text-slate-800">
                      Real 0 ({activeModel.config.negativeClass})
                    </td>
                    <td className="p-2">
                      <div className="rounded-lg border border-slate-300 bg-slate-100 p-4">
                        <p className="text-[11px] font-semibold text-slate-700">
                          Verdaderos Negativos (TN)
                        </p>
                        <p className="mt-1 font-mono text-2xl font-bold text-slate-900">
                          {formatCell(cm.tn, row0Total, col0Total)}
                        </p>
                        <p className="mt-1 text-[11px] text-slate-500">n = {cm.tn}</p>
                      </div>
                    </td>
                    <td className="p-2">
                      <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
                        <p className="text-[11px] font-semibold text-amber-800">
                          Falsos Positivos (FP)
                        </p>
                        <p className="mt-1 font-mono text-2xl font-bold text-amber-900">
                          {formatCell(cm.fp, row0Total, col1Total)}
                        </p>
                        <p className="mt-1 text-[11px] text-amber-700">n = {cm.fp}</p>
                      </div>
                    </td>
                    <td className="p-3 text-right font-mono font-semibold text-slate-700">
                      {row0Total}
                    </td>
                  </tr>
                  <tr>
                    <td className="p-3 text-left font-semibold text-teal-800">
                      Real 1 ({activeModel.config.positiveClass})
                    </td>
                    <td className="p-2">
                      <div className="rounded-lg border border-rose-200 bg-rose-50 p-4">
                        <p className="text-[11px] font-semibold text-rose-800">
                          Falsos Negativos (FN)
                        </p>
                        <p className="mt-1 font-mono text-2xl font-bold text-rose-900">
                          {formatCell(cm.fn, row1Total, col0Total)}
                        </p>
                        <p className="mt-1 text-[11px] text-rose-700">n = {cm.fn}</p>
                      </div>
                    </td>
                    <td className="p-2">
                      <div className="rounded-lg border border-teal-300 bg-teal-50 p-4">
                        <p className="text-[11px] font-semibold text-teal-800">
                          Verdaderos Positivos (TP)
                        </p>
                        <p className="mt-1 font-mono text-2xl font-bold text-teal-900">
                          {formatCell(cm.tp, row1Total, col1Total)}
                        </p>
                        <p className="mt-1 text-[11px] text-teal-700">n = {cm.tp}</p>
                      </div>
                    </td>
                    <td className="p-3 text-right font-mono font-semibold text-slate-700">
                      {row1Total}
                    </td>
                  </tr>
                  <tr className="border-t border-slate-200">
                    <td className="p-2 text-left text-slate-500">Total Predicho</td>
                    <td className="p-2 font-mono font-semibold text-slate-700">{col0Total}</td>
                    <td className="p-2 font-mono font-semibold text-slate-700">{col1Total}</td>
                    <td className="p-2 text-right font-mono font-bold text-slate-900">{cm.total}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Interpretación técnica de los 4 cuadrantes */}
          <div className="space-y-3 text-xs lg:col-span-5">
            <h3 className="text-sm font-semibold text-slate-900">
              Interpretación de los Cuadrantes y Perspectivas
            </h3>
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <strong className="text-teal-800">TP ({cm.tp}):</strong> Casos que realmente son{' '}
              <em>{activeModel.config.positiveClass}</em> y el modelo acertó prediciendo 1.
            </div>
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <strong className="text-slate-900">TN ({cm.tn}):</strong> Casos que realmente son{' '}
              <em>{activeModel.config.negativeClass}</em> y el modelo acertó prediciendo 0.
            </div>
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <strong className="text-amber-800">FP ({cm.fp} — Error Tipo I):</strong> Realmente eran 0, pero el modelo predijo 1. Reduce la Precision de la categoría 1.
            </div>
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <strong className="text-rose-800">FN ({cm.fn} — Error Tipo II):</strong> Realmente eran 1, pero el modelo predijo 0. Reduce el Recall (sensibilidad) de la categoría 1.
            </div>
          </div>
        </div>
      </section>

      {/* Casos de negocio por categoría */}
      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <h2 className="text-base font-semibold text-slate-900">
          Casos de Negocio: Impacto Diferencial entre Categoría 0 y Categoría 1
        </h2>
        <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
          {CLASS_EXAMPLES.map((ex) => (
            <div key={ex.title} className="rounded-lg border border-slate-200 bg-slate-50/70 p-4 text-xs space-y-1.5">
              <h3 className="text-sm font-semibold text-slate-900">{ex.title}</h3>
              <p className="text-slate-600">
                Positivo (1): <strong>{ex.pos}</strong> · Negativo (0): <strong>{ex.neg}</strong>
              </p>
              <p className="text-amber-900">
                <strong>Falso Positivo:</strong> {ex.fp}
              </p>
              <p className="text-rose-900">
                <strong>Falso Negativo:</strong> {ex.fn}
              </p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
};
