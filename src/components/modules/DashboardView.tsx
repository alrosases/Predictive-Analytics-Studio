import React from 'react';
import {
  ArrowRight,
  Database,
  FileSpreadsheet,
  FileText,
  Sliders,
  TrendingUp,
} from 'lucide-react';
import type { DatasetState, ModelResult, ModuleId } from '../../types/analytics';
import { computeConfusionMatrix, computeEvaluationMetrics } from '../../engine/logisticRegression';
import { formatNumberES } from '../../engine/statistics';

interface DashboardViewProps {
  dataset: DatasetState | null;
  activeModel: ModelResult | null;
  savedModels: ModelResult[];
  onNavigate: (mod: ModuleId) => void;
}

const PIPELINE_STEPS: { id: ModuleId; num: string; label: string; desc: string }[] = [
  { id: 'cargar', num: '01', label: 'Cargar datos', desc: 'CSV, XLS o XLSX' },
  { id: 'exploracion', num: '02', label: 'Explorar', desc: 'Filas y tipos' },
  { id: 'descriptivo', num: '03', label: 'Analizar', desc: 'Estadística y gráficos' },
  { id: 'preparacion', num: '04', label: 'Preparar', desc: 'Faltantes y clase 0/1' },
  { id: 'regresion', num: '05', label: 'Entrenar', desc: 'Máxima verosimilitud' },
  { id: 'predicciones', num: '06', label: 'Predecir', desc: 'Z y P(Y=1)' },
  { id: 'metricas', num: '07', label: 'Evaluar', desc: 'Matriz y F1-Score' },
  { id: 'validacion', num: '08', label: 'Validar', desc: 'Train vs. Prueba' },
  { id: 'interpretacion', num: '09', label: 'Interpretar', desc: 'Hallazgos y negocio' },
  { id: 'informe', num: '10', label: 'Informe PDF', desc: 'Reporte ejecutivo' },
];

export const DashboardView: React.FC<DashboardViewProps> = ({
  dataset,
  activeModel,
  savedModels,
  onNavigate,
}) => {
  const cmTest = activeModel
    ? computeConfusionMatrix(activeModel.testPredictions, activeModel.config.threshold)
    : null;
  const metricsTest = cmTest ? computeEvaluationMetrics(cmTest) : null;

  const missingTotal = dataset
    ? dataset.profiles.reduce((acc, p) => acc + p.missingCount, 0)
    : 0;

  return (
    <div className="space-y-8">
      {/* Encabezado ejecutivo */}
      <section className="rounded-xl border border-slate-200 bg-white p-6 md:p-8">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
          <div className="max-w-2xl space-y-2">
            <p className="text-xs font-medium text-teal-700">
              Plataforma de Analítica Predictiva · Regresión Logística y Machine Learning
            </p>
            <h1 className="text-2xl font-bold tracking-tight text-slate-900 md:text-3xl text-balance">
              Predictive Analytics Studio
            </h1>
            <p className="text-sm leading-relaxed text-slate-600">
              Entorno completo para exploración estadística, preparación sin fuga de información,
              estimación de regresión logística binaria por máxima verosimilitud, validación en
              entrenamiento/prueba y generación de informes ejecutivos en PDF.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3 shrink-0">
            <button
              type="button"
              onClick={() => onNavigate(dataset ? 'regresion' : 'cargar')}
              className="inline-flex items-center gap-2 rounded-lg bg-slate-900 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-slate-800 whitespace-nowrap shrink-0"
            >
              {dataset ? 'Configurar Modelo Logístico' : 'Cargar Base de Datos'}
              <ArrowRight className="h-4 w-4" />
            </button>
            {activeModel && !activeModel.error && (
              <button
                type="button"
                onClick={() => onNavigate('informe')}
                className="inline-flex items-center gap-2 rounded-lg border border-teal-600 bg-teal-600 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-teal-700 whitespace-nowrap shrink-0"
              >
                <FileText className="h-4 w-4" />
                Ir a Informe PDF
              </button>
            )}
          </div>
        </div>
      </section>

      {/* KPIs del proyecto activo */}
      <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <div className="flex items-center justify-between text-xs text-slate-500">
            <span>Base de datos activa</span>
            <Database className="h-4 w-4 text-slate-400" />
          </div>
          <p className="mt-2 truncate text-lg font-semibold text-slate-900" title={dataset?.fileName}>
            {dataset ? dataset.fileName : 'Sin base cargada'}
          </p>
          <p className="mt-1 text-xs text-slate-600 tabular-nums">
            {dataset
              ? `${dataset.workingRows.length} registros · ${dataset.columns.length} variables`
              : 'Importe CSV, XLS o XLSX para iniciar'}
          </p>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <div className="flex items-center justify-between text-xs text-slate-500">
            <span>Calidad de registros</span>
            <FileSpreadsheet className="h-4 w-4 text-slate-400" />
          </div>
          <p className="mt-2 font-mono text-2xl font-bold text-slate-900 tabular-nums">
            {dataset ? missingTotal : '—'}
          </p>
          <p className="mt-1 text-xs text-slate-600 tabular-nums">
            {dataset
              ? `Valores faltantes · ${dataset.duplicateCount} duplicados`
              : 'Esperando inspección inicial'}
          </p>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <div className="flex items-center justify-between text-xs text-slate-500">
            <span>Accuracy en Prueba</span>
            <TrendingUp className="h-4 w-4 text-teal-600" />
          </div>
          <p className="mt-2 font-mono text-2xl font-bold text-slate-900 tabular-nums">
            {metricsTest && metricsTest.accuracy !== null
              ? `${(metricsTest.accuracy * 100).toFixed(2)}%`
              : '—'}
          </p>
          <p className="mt-1 text-xs text-slate-600 tabular-nums">
            {activeModel && !activeModel.error
              ? `${activeModel.testCount} filas de prueba · Corte ${(
                  activeModel.config.threshold * 100
                ).toFixed(0)}%`
              : 'Entrene un modelo logístico'}
          </p>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <div className="flex items-center justify-between text-xs text-slate-500">
            <span>F1-Score Clase Positiva (1)</span>
            <Sliders className="h-4 w-4 text-teal-600" />
          </div>
          <p className="mt-2 font-mono text-2xl font-bold text-teal-700 tabular-nums">
            {metricsTest && metricsTest.f1_1 !== null
              ? `${(metricsTest.f1_1 * 100).toFixed(2)}%`
              : '—'}
          </p>
          <p className="mt-1 text-xs text-slate-600 tabular-nums">
            {activeModel && !activeModel.error
              ? `Modelos guardados: ${savedModels.length} · Objetivo: ${activeModel.config.targetColumn}`
              : 'Métrica clave de equilibrio Precision/Recall'}
          </p>
        </div>
      </section>

      {/* Flujo metodológico */}
      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <div className="flex flex-col gap-1 border-b border-slate-200 pb-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Flujo de Trabajo Analítico
            </h2>
            <p className="text-xs text-slate-600">
              Seleccione cualquier etapa para avanzar en el proyecto activo.
            </p>
          </div>
          <span className="text-xs text-slate-500 tabular-nums">
            {dataset ? 'Base cargada' : 'Pendiente de datos'} ·{' '}
            {activeModel && !activeModel.error ? 'Modelo estimado' : 'Modelo pendiente'}
          </span>
        </div>

        <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          {PIPELINE_STEPS.map((step) => {
            const done =
              (step.id === 'cargar' && dataset !== null) ||
              ((step.id === 'exploracion' || step.id === 'descriptivo' || step.id === 'preparacion') &&
                dataset !== null) ||
              ((step.id === 'regresion' ||
                step.id === 'predicciones' ||
                step.id === 'metricas' ||
                step.id === 'validacion' ||
                step.id === 'interpretacion' ||
                step.id === 'informe') &&
                activeModel !== null &&
                !activeModel.error);
            return (
              <button
                key={step.id}
                type="button"
                onClick={() => onNavigate(step.id)}
                className="group flex flex-col items-start rounded-lg border border-slate-200 bg-slate-50/60 p-3.5 text-left transition-colors hover:border-teal-600 hover:bg-white"
              >
                <div className="flex w-full items-center justify-between">
                  <span className="font-mono text-xs font-semibold text-teal-700">
                    {step.num}
                  </span>
                  <span className="text-[11px] text-slate-500">
                    {done ? 'Listo' : 'Pendiente'}
                  </span>
                </div>
                <span className="mt-2 text-sm font-semibold text-slate-900 group-hover:text-teal-800">
                  {step.label}
                </span>
                <span className="mt-0.5 text-xs text-slate-500">{step.desc}</span>
              </button>
            );
          })}
        </div>
      </section>

      {/* Resumen del modelo activo si existe */}
      {activeModel && !activeModel.error && dataset && (
        <section className="rounded-xl border border-slate-200 bg-white p-6">
          <div className="flex flex-col gap-2 border-b border-slate-200 pb-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h2 className="text-base font-semibold text-slate-900">
                Modelo Activo: {activeModel.name}
              </h2>
              <p className="text-xs text-slate-600">
                {activeModel.config.targetColumn} ~ {activeModel.config.featureColumns.join(' + ')}
              </p>
            </div>
            <button
              type="button"
              onClick={() => onNavigate('interpretacion')}
              className="text-xs font-medium text-teal-700 hover:text-teal-900 whitespace-nowrap shrink-0"
            >
              Ver interpretación completa →
            </button>
          </div>

          <div className="mt-4 grid grid-cols-1 gap-6 lg:grid-cols-2">
            <div className="space-y-2 text-sm text-slate-700">
              <p>
                <strong className="text-slate-900">Variable dependiente:</strong>{' '}
                <span className="font-mono">{activeModel.config.targetColumn}</span> (1 ={' '}
                {activeModel.config.positiveClass} · 0 = {activeModel.config.negativeClass})
              </p>
              <p>
                <strong className="text-slate-900">Partición y balanceo:</strong>{' '}
                {formatNumberES(activeModel.config.splitConfig.trainRatio * 100, 0)}% entrenamiento (
                {activeModel.trainCount} filas) ·{' '}
                {formatNumberES((1 - activeModel.config.splitConfig.trainRatio) * 100, 0)}% prueba (
                {activeModel.testCount} filas) · Balanceo:{' '}
                <span className="font-mono">{activeModel.config.balanceMethod}</span>
              </p>
              <p>
                <strong className="text-slate-900">Convergencia MLE:</strong>{' '}
                {activeModel.converged
                  ? `Convergió en ${activeModel.iterations} iteraciones`
                  : `Sin convergencia estricta (${activeModel.iterations} iteraciones)`}{' '}
                · AIC: <span className="font-mono">{activeModel.aic.toFixed(2)}</span>
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 tabular-nums">
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <p className="text-[11px] text-slate-500">Verd. Positivos</p>
                <p className="mt-1 font-mono text-lg font-bold text-teal-700">{cmTest?.tp ?? 0}</p>
              </div>
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <p className="text-[11px] text-slate-500">Verd. Negativos</p>
                <p className="mt-1 font-mono text-lg font-bold text-slate-900">{cmTest?.tn ?? 0}</p>
              </div>
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <p className="text-[11px] text-slate-500">Falsos Positivos</p>
                <p className="mt-1 font-mono text-lg font-bold text-amber-700">{cmTest?.fp ?? 0}</p>
              </div>
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <p className="text-[11px] text-slate-500">Falsos Negativos</p>
                <p className="mt-1 font-mono text-lg font-bold text-rose-700">{cmTest?.fn ?? 0}</p>
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
};
