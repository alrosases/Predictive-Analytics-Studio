import React from 'react';
import { Play, ShieldCheck } from 'lucide-react';
import type {
  DatasetState,
  LogisticRegressionConfig,
  ModelResult,
  ModuleId,
} from '../../types/analytics';
import {
  computeConfusionMatrix,
  computeEvaluationMetrics,
} from '../../engine/logisticRegression';

interface ValidationViewProps {
  dataset: DatasetState | null;
  config: LogisticRegressionConfig;
  activeModel: ModelResult | null;
  onUpdateConfig: (updater: (prev: LogisticRegressionConfig) => LogisticRegressionConfig) => void;
  onRetrain: () => void;
  onNavigate: (mod: ModuleId) => void;
}

function fmtMetric(v: number | null): string {
  return v !== null && Number.isFinite(v) ? `${(v * 100).toFixed(2)}%` : 'No definida';
}

export const ValidationView: React.FC<ValidationViewProps> = ({
  dataset,
  config,
  activeModel,
  onUpdateConfig,
  onRetrain,
  onNavigate,
}) => {
  if (!dataset) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-10 text-center">
        <h2 className="text-lg font-semibold text-slate-900">Sin datos cargados</h2>
        <button
          type="button"
          onClick={() => onNavigate('cargar')}
          className="mt-4 rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white"
        >
          Cargar Base de Datos
        </button>
      </div>
    );
  }

  const cmTrain =
    activeModel && !activeModel.error
      ? computeConfusionMatrix(activeModel.trainPredictions, activeModel.config.threshold)
      : null;
  const cmTest =
    activeModel && !activeModel.error
      ? computeConfusionMatrix(activeModel.testPredictions, activeModel.config.threshold)
      : null;
  const mTrain = cmTrain ? computeEvaluationMetrics(cmTrain) : null;
  const mTest = cmTest ? computeEvaluationMetrics(cmTest) : null;

  const accDiffPP =
    mTrain?.accuracy !== null &&
    mTrain?.accuracy !== undefined &&
    mTest?.accuracy !== null &&
    mTest?.accuracy !== undefined
      ? (mTrain.accuracy - mTest.accuracy) * 100
      : null;

  const f1DiffPP =
    mTrain?.f1_1 !== null &&
    mTrain?.f1_1 !== undefined &&
    mTest?.f1_1 !== null &&
    mTest?.f1_1 !== undefined
      ? (mTrain.f1_1 - mTest.f1_1) * 100
      : null;

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-900">
            División Entrenamiento / Prueba y Validación
          </h1>
          <p className="text-sm text-slate-600">
            Configure la partición sin fuga de información y compare las métricas independientes entre entrenamiento y prueba.
          </p>
        </div>
        <button
          type="button"
          onClick={() => onNavigate('interpretacion')}
          className="rounded-lg bg-teal-600 px-4 py-2 text-xs font-medium text-white hover:bg-teal-700 whitespace-nowrap shrink-0"
        >
          Interpretar Resultados →
        </button>
      </div>

      {/* Configuración de división */}
      <section className="rounded-xl border border-slate-200 bg-white p-6 space-y-4">
        <div className="flex items-center gap-2 border-b border-slate-200 pb-3">
          <ShieldCheck className="h-4 w-4 text-teal-700" />
          <h2 className="text-base font-semibold text-slate-900">
            Parámetros de Partición Aleatoria y Estratificación
          </h2>
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <div>
            <label className="block text-xs font-medium text-slate-700">
              Proporción Entrenamiento / Prueba
            </label>
            <select
              value={config.splitConfig.trainRatio}
              onChange={(e) =>
                onUpdateConfig((prev) => ({
                  ...prev,
                  splitConfig: { ...prev.splitConfig, trainRatio: Number(e.target.value) },
                }))
              }
              className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 font-mono text-xs text-slate-900"
            >
              <option value={0.75}>75% Entrenamiento / 25% Prueba (Estándar)</option>
              <option value={0.7}>70% Entrenamiento / 30% Prueba</option>
              <option value={0.8}>80% Entrenamiento / 20% Prueba</option>
              <option value={0.65}>65% Entrenamiento / 35% Prueba</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-700">
              Semilla aleatoria (reproducibilidad)
            </label>
            <input
              type="number"
              value={config.splitConfig.seed}
              onChange={(e) =>
                onUpdateConfig((prev) => ({
                  ...prev,
                  splitConfig: {
                    ...prev.splitConfig,
                    seed: Math.max(1, Math.round(Number(e.target.value) || 42)),
                  },
                }))
              }
              className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2 font-mono text-xs text-slate-900"
            />
          </div>

          <div className="flex flex-col justify-end">
            <label className="flex items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-medium text-slate-800">
              <input
                type="checkbox"
                checked={config.splitConfig.stratified}
                onChange={(e) =>
                  onUpdateConfig((prev) => ({
                    ...prev,
                    splitConfig: { ...prev.splitConfig, stratified: e.target.checked },
                  }))
                }
                className="h-4 w-4 rounded border-slate-300 text-teal-600"
              />
              <span>División estratificada (conserva % de clases)</span>
            </label>
          </div>
        </div>

        <div className="flex justify-end pt-2">
          <button
            type="button"
            onClick={onRetrain}
            className="inline-flex items-center gap-2 rounded-lg bg-slate-900 px-4 py-2 text-xs font-medium text-white hover:bg-slate-800"
          >
            <Play className="h-3.5 w-3.5" />
            Aplicar Partición y Re-entrenar Modelo
          </button>
        </div>
      </section>

      {activeModel && !activeModel.error && mTrain && mTest && cmTrain && cmTest && (
        <>
          {/* Tamaños y distribución */}
          <section className="grid grid-cols-1 gap-4 md:grid-cols-2 tabular-nums">
            <div className="rounded-xl border border-slate-200 bg-white p-5">
              <h3 className="text-sm font-semibold text-slate-900">
                A. Conjunto de Entrenamiento (Evaluado sin balanceo sintético)
              </h3>
              <p className="mt-1 text-xs text-slate-600">
                {activeModel.trainCount} observaciones reales ({activeModel.trainClassDistBefore.c0} clase 0 ·{' '}
                {activeModel.trainClassDistBefore.c1} clase 1). Balanceo usado al ajustar:{' '}
                <span className="font-mono">{activeModel.config.balanceMethod}</span> ({activeModel.balancedTrainCount} filas).
              </p>
              <div className="mt-3 grid grid-cols-4 gap-2 text-center font-mono text-xs">
                <div className="rounded bg-teal-50 p-2"><span className="block text-[10px] text-teal-700">TP</span><strong>{cmTrain.tp}</strong></div>
                <div className="rounded bg-slate-100 p-2"><span className="block text-[10px] text-slate-600">TN</span><strong>{cmTrain.tn}</strong></div>
                <div className="rounded bg-amber-50 p-2"><span className="block text-[10px] text-amber-700">FP</span><strong>{cmTrain.fp}</strong></div>
                <div className="rounded bg-rose-50 p-2"><span className="block text-[10px] text-rose-700">FN</span><strong>{cmTrain.fn}</strong></div>
              </div>
            </div>

            <div className="rounded-xl border border-slate-200 bg-white p-5">
              <h3 className="text-sm font-semibold text-slate-900">
                B. Conjunto de Prueba (Datos no vistos en entrenamiento)
              </h3>
              <p className="mt-1 text-xs text-slate-600">
                {activeModel.testCount} observaciones independientes ({activeModel.testClassDist.c0} clase 0 ·{' '}
                {activeModel.testClassDist.c1} clase 1).
              </p>
              <div className="mt-3 grid grid-cols-4 gap-2 text-center font-mono text-xs">
                <div className="rounded bg-teal-50 p-2"><span className="block text-[10px] text-teal-700">TP</span><strong>{cmTest.tp}</strong></div>
                <div className="rounded bg-slate-100 p-2"><span className="block text-[10px] text-slate-600">TN</span><strong>{cmTest.tn}</strong></div>
                <div className="rounded bg-amber-50 p-2"><span className="block text-[10px] text-amber-700">FP</span><strong>{cmTest.fp}</strong></div>
                <div className="rounded bg-rose-50 p-2"><span className="block text-[10px] text-rose-700">FN</span><strong>{cmTest.fn}</strong></div>
              </div>
            </div>
          </section>

          {/* Tabla comparativa Train vs Test */}
          <section className="rounded-xl border border-slate-200 bg-white p-6">
            <h2 className="text-base font-semibold text-slate-900">
              Tabla Comparativa: Entrenamiento vs. Prueba
            </h2>

            <div className="mt-4 overflow-x-auto">
              <table className="w-full border-collapse text-left text-xs tabular-nums">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500">
                    <th className="py-2 pr-4 font-medium">Métrica</th>
                    <th className="py-2 pr-4 text-right font-medium">Entrenamiento</th>
                    <th className="py-2 pr-4 text-right font-medium">Prueba</th>
                    <th className="py-2 text-right font-medium">Diferencia (Train - Test)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {[
                    { name: 'Accuracy', a: mTrain.accuracy, b: mTest.accuracy },
                    { name: 'Error de clasificación', a: mTrain.classificationError, b: mTest.classificationError },
                    { name: `Precision (Clase 1: ${activeModel.config.positiveClass})`, a: mTrain.precision1, b: mTest.precision1 },
                    { name: `Recall (Clase 1: ${activeModel.config.positiveClass})`, a: mTrain.recall1, b: mTest.recall1 },
                    { name: `F1-Score (Clase 1: ${activeModel.config.positiveClass})`, a: mTrain.f1_1, b: mTest.f1_1 },
                    { name: `F1-Score (Clase 0: ${activeModel.config.negativeClass})`, a: mTrain.f1_0, b: mTest.f1_0 },
                    { name: 'Balanced Accuracy', a: mTrain.balancedAccuracy, b: mTest.balancedAccuracy },
                  ].map((row) => {
                    const diff =
                      row.a !== null && row.b !== null ? (row.a - row.b) * 100 : null;
                    return (
                      <tr key={row.name} className="hover:bg-slate-50">
                        <td className="py-2.5 pr-4 font-medium text-slate-900">{row.name}</td>
                        <td className="py-2.5 pr-4 text-right font-mono text-slate-800">
                          {fmtMetric(row.a)}
                        </td>
                        <td className="py-2.5 pr-4 text-right font-mono font-semibold text-teal-700">
                          {fmtMetric(row.b)}
                        </td>
                        <td className="py-2.5 text-right font-mono text-slate-600">
                          {diff !== null ? `${diff >= 0 ? '+' : ''}${diff.toFixed(2)} pp` : 'N/D'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Interpretación de generalización */}
            <div className="mt-5 rounded-lg border border-teal-200 bg-teal-50/50 p-4 text-xs leading-relaxed text-slate-800">
              <strong>Diagnóstico de Validación: </strong>
              El modelo obtuvo un Accuracy de <strong>{fmtMetric(mTrain.accuracy)}</strong> en entrenamiento y{' '}
              <strong>{fmtMetric(mTest.accuracy)}</strong> en prueba.{' '}
              {accDiffPP !== null && Math.abs(accDiffPP) <= 5
                ? `La diferencia de ${accDiffPP.toFixed(2)} puntos porcentuales evidencia estabilidad y buena capacidad de generalización sobre registros nuevos.`
                : accDiffPP !== null && accDiffPP > 5
                ? `La brecha de +${accDiffPP.toFixed(2)} puntos porcentuales sugiere posible sobreajuste (overfitting) al conjunto de entrenamiento.`
                : 'Revise la variabilidad muestral entre ambos subconjuntos.'}{' '}
              Para la categoría objetivo (1), el F1-Score en prueba es{' '}
              <strong>{fmtMetric(mTest.f1_1)}</strong>
              {f1DiffPP !== null ? ` (variación de ${f1DiffPP >= 0 ? '+' : ''}${f1DiffPP.toFixed(2)} pp respecto a entrenamiento).` : '.'}
            </div>
          </section>
        </>
      )}
    </div>
  );
};
