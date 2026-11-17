import React from 'react';
import { Play, Scale } from 'lucide-react';
import type {
  BalanceMethod,
  DatasetState,
  LogisticRegressionConfig,
  ModelResult,
  ModuleId,
} from '../../types/analytics';
import {
  computeConfusionMatrix,
  computeEvaluationMetrics,
} from '../../engine/logisticRegression';

interface BalancingViewProps {
  dataset: DatasetState | null;
  config: LogisticRegressionConfig;
  activeModel: ModelResult | null;
  onUpdateConfig: (updater: (prev: LogisticRegressionConfig) => LogisticRegressionConfig) => void;
  onRetrainWithBalance: (method: BalanceMethod) => void;
  onNavigate: (mod: ModuleId) => void;
}

const METHODS: { id: BalanceMethod; title: string; desc: string }[] = [
  {
    id: 'none',
    title: 'Sin balanceo (Datos originales de entrenamiento)',
    desc: 'Conserva las proporciones reales observadas en el conjunto de entrenamiento.',
  },
  {
    id: 'undersample',
    title: 'Submuestreo (Random Undersampling)',
    desc: 'Reduce aleatoriamente la clase mayoritaria en entrenamiento hasta igualar la minoritaria. Útil con bases grandes.',
  },
  {
    id: 'oversample',
    title: 'Sobremuestreo (Random Oversampling)',
    desc: 'Replica aleatoriamente observaciones de la clase minoritaria en entrenamiento hasta equilibrar ambas categorías.',
  },
  {
    id: 'smote',
    title: 'SMOTE (Sobremuestreo Sintético de Minoría)',
    desc: 'Genera ejemplos sintéticos interpolando vecinos cercanos (k-NN) de la clase minoritaria exclusivamente en el espacio numérico de entrenamiento.',
  },
];

export const BalancingView: React.FC<BalancingViewProps> = ({
  dataset,
  config,
  activeModel,
  onUpdateConfig,
  onRetrainWithBalance,
  onNavigate,
}) => {
  if (!dataset || !config.targetColumn) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-10 text-center">
        <h2 className="text-lg font-semibold text-slate-900">Configure primero la variable objetivo</h2>
        <p className="mt-1 text-sm text-slate-600">
          Necesita cargar datos y definir la variable dependiente binaria (0/1) para analizar el balanceo de clases.
        </p>
        <button
          type="button"
          onClick={() => onNavigate('preparacion')}
          className="mt-5 rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white"
        >
          Ir a Preparación de Datos
        </button>
      </div>
    );
  }

  const targetProf = dataset.profiles.find((p) => p.name === config.targetColumn);
  const c0Count =
    targetProf?.frequencies.find((f) => f.value === config.negativeClass)?.count ?? 0;
  const c1Count =
    targetProf?.frequencies.find((f) => f.value === config.positiveClass)?.count ?? 0;
  const totalValid = c0Count + c1Count;
  const pct0 = totalValid > 0 ? (c0Count / totalValid) * 100 : 0;
  const pct1 = totalValid > 0 ? (c1Count / totalValid) * 100 : 0;
  const diffPct = Math.abs(pct0 - pct1);

  const cmTest =
    activeModel && !activeModel.error
      ? computeConfusionMatrix(activeModel.testPredictions, activeModel.config.threshold)
      : null;
  const mTest = cmTest ? computeEvaluationMetrics(cmTest) : null;

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-900">Balanceo de Datos</h1>
          <p className="text-sm text-slate-600">
            Detecte desproporciones entre las categorías 0 y 1 y aplique técnicas de balanceo <strong>exclusivamente sobre el entrenamiento</strong> (nunca sobre el conjunto de prueba).
          </p>
        </div>
        <button
          type="button"
          onClick={() => onNavigate('validacion')}
          className="rounded-lg bg-teal-600 px-4 py-2 text-xs font-medium text-white hover:bg-teal-700 whitespace-nowrap shrink-0"
        >
          Ver Validación Train / Test →
        </button>
      </div>

      {/* Diagnóstico de desbalanceo */}
      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <div className="flex items-center gap-2 border-b border-slate-200 pb-3">
          <Scale className="h-4 w-4 text-teal-700" />
          <h2 className="text-base font-semibold text-slate-900">
            Diagnóstico de Proporciones en la Base Original ({config.targetColumn})
          </h2>
        </div>

        <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-3 tabular-nums">
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
            <p className="text-xs text-slate-500">Categoría 0 ({config.negativeClass})</p>
            <p className="mt-1 font-mono text-2xl font-bold text-slate-900">
              {c0Count} <span className="text-sm font-normal text-slate-600">({pct0.toFixed(1)}%)</span>
            </p>
          </div>
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
            <p className="text-xs text-slate-500">Categoría 1 ({config.positiveClass})</p>
            <p className="mt-1 font-mono text-2xl font-bold text-teal-700">
              {c1Count} <span className="text-sm font-normal text-slate-600">({pct1.toFixed(1)}%)</span>
            </p>
          </div>
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
            <p className="text-xs text-slate-500">Diferencia entre proporciones</p>
            <p className="mt-1 font-mono text-2xl font-bold text-slate-900">
              {diffPct.toFixed(1)} pp
            </p>
            <p className="mt-0.5 text-[11px] text-slate-500">
              {diffPct >= 20 ? 'Desbalanceo relevante detectado' : 'Distribución relativamente equilibrada'}
            </p>
          </div>
        </div>

        {/* Barra visual comparativa */}
        <div className="mt-5 space-y-3">
          <p className="text-xs font-medium text-slate-700">Distribución original vs. Entrenamiento balanceado:</p>
          <div className="space-y-2">
            <div>
              <div className="mb-1 flex justify-between text-xs text-slate-600 tabular-nums">
                <span>Base original ({totalValid} registros)</span>
                <span>0: {pct0.toFixed(1)}% · 1: {pct1.toFixed(1)}%</span>
              </div>
              <div className="flex h-4 w-full overflow-hidden rounded-md bg-slate-100">
                <div className="bg-slate-800" style={{ width: `${pct0}%` }} />
                <div className="bg-teal-600" style={{ width: `${pct1}%` }} />
              </div>
            </div>

            {activeModel && !activeModel.error && (
              <div>
                <div className="mb-1 flex justify-between text-xs text-slate-600 tabular-nums">
                  <span>
                    Entrenamiento tras balanceo &quot;{activeModel.config.balanceMethod}&quot; ({activeModel.balancedTrainCount} registros)
                  </span>
                  <span>
                    0: {activeModel.trainClassDistAfter.c0} · 1: {activeModel.trainClassDistAfter.c1}
                  </span>
                </div>
                <div className="flex h-4 w-full overflow-hidden rounded-md bg-slate-100">
                  <div
                    className="bg-slate-800"
                    style={{
                      width: `${
                        (activeModel.trainClassDistAfter.c0 /
                          Math.max(1, activeModel.balancedTrainCount)) *
                        100
                      }%`,
                    }}
                  />
                  <div
                    className="bg-teal-600"
                    style={{
                      width: `${
                        (activeModel.trainClassDistAfter.c1 /
                          Math.max(1, activeModel.balancedTrainCount)) *
                        100
                      }%`,
                    }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Selector de técnica de balanceo */}
      <section className="rounded-xl border border-slate-200 bg-white p-6 space-y-4">
        <h2 className="text-base font-semibold text-slate-900">
          Seleccionar Técnica de Balanceo y Re-estimar Modelo
        </h2>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          {METHODS.map((m) => {
            const active = config.balanceMethod === m.id;
            return (
              <button
                key={m.id}
                type="button"
                onClick={() =>
                  onUpdateConfig((prev) => ({ ...prev, balanceMethod: m.id }))
                }
                className={`rounded-xl border p-4 text-left transition-colors ${
                  active
                    ? 'border-teal-600 bg-teal-50/40'
                    : 'border-slate-200 bg-white hover:border-slate-300'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-semibold text-slate-900">{m.title}</span>
                  <span className="font-mono text-xs text-teal-700">
                    {active ? 'Seleccionado' : ''}
                  </span>
                </div>
                <p className="mt-1 text-xs leading-relaxed text-slate-600">{m.desc}</p>
              </button>
            );
          })}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-4 border-t border-slate-200 pt-4">
          <p className="text-xs text-slate-600">
            El conjunto de prueba ({activeModel?.testCount ?? '—'} filas) permanece intacto con sus proporciones reales para una evaluación honesta.
          </p>
          <button
            type="button"
            onClick={() => onRetrainWithBalance(config.balanceMethod)}
            className="inline-flex items-center gap-2 rounded-lg bg-slate-900 px-4 py-2 text-xs font-medium text-white hover:bg-slate-800 whitespace-nowrap"
          >
            <Play className="h-3.5 w-3.5" />
            Aplicar Balanceo y Entrenar Modelo
          </button>
        </div>

        {mTest && activeModel && !activeModel.error && (
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 text-xs text-slate-700 tabular-nums">
            <strong>Impacto actual en Prueba ({activeModel.config.balanceMethod}):</strong>{' '}
            Accuracy: <strong>{mTest.accuracy !== null ? `${(mTest.accuracy * 100).toFixed(2)}%` : 'N/D'}</strong> ·{' '}
            Recall(1): <strong>{mTest.recall1 !== null ? `${(mTest.recall1 * 100).toFixed(2)}%` : 'N/D'}</strong> ·{' '}
            Precision(1): <strong>{mTest.precision1 !== null ? `${(mTest.precision1 * 100).toFixed(2)}%` : 'N/D'}</strong> ·{' '}
            F1-Score(1): <strong className="text-teal-700">{mTest.f1_1 !== null ? `${(mTest.f1_1 * 100).toFixed(2)}%` : 'N/D'}</strong>
          </div>
        )}
      </section>
    </div>
  );
};
