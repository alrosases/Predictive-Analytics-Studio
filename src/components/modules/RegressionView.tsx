import React, { useState } from 'react';
import { AlertCircle, HelpCircle, Layers, Play } from 'lucide-react';
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
import { SigmoidChart } from '../charts/SigmoidChart';
import { StatExplainModal } from '../common/StatExplainModal';

interface RegressionViewProps {
  dataset: DatasetState | null;
  config: LogisticRegressionConfig;
  activeModel: ModelResult | null;
  savedModels: ModelResult[];
  onUpdateConfig: (updater: (prev: LogisticRegressionConfig) => LogisticRegressionConfig) => void;
  onTrainModel: (modelName: string) => void;
  onSelectSavedModel: (modelId: string) => void;
  onThresholdChange: (threshold: number) => void;
  onNavigate: (mod: ModuleId) => void;
}

function sigStars(p: number | null): string {
  if (p === null) return '';
  if (p < 0.001) return '***';
  if (p < 0.01) return '**';
  if (p < 0.05) return '*';
  if (p < 0.1) return '.';
  return '';
}

export const RegressionView: React.FC<RegressionViewProps> = ({
  dataset,
  config,
  activeModel,
  savedModels,
  onUpdateConfig,
  onTrainModel,
  onSelectSavedModel,
  onThresholdChange,
  onNavigate,
}) => {
  const [modelName, setModelName] = useState('Modelo Logístico Principal');
  const [showExplain, setShowExplain] = useState(false);

  if (!dataset) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-10 text-center">
        <h2 className="text-lg font-semibold text-slate-900">Sin datos cargados</h2>
        <p className="mt-1 text-sm text-slate-600">
          Cargue una base de datos y configure la variable dependiente para estimar el modelo de regresión logística.
        </p>
        <button
          type="button"
          onClick={() => onNavigate('cargar')}
          className="mt-5 rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white"
        >
          Cargar Base de Datos
        </button>
      </div>
    );
  }

  // Presets rápidos para comparación de modelos (Financieros, Comerciales, Combinados, Personalizado)
  const applyPreset = (preset: 'financieras' | 'comerciales' | 'combinado') => {
    const avail = dataset.columns.filter((c) => c !== config.targetColumn);
    const finKeywords = ['saldo', 'venta', 'pasivo', 'relacion', 'cargo', 'ingreso', 'monto'];
    const comKeywords = ['edad', 'antiguedad', 'dueno', 'direccion', 'insumo', 'tipologia', 'segmento', 'ticket', 'uso', 'contrato'];

    let picked: string[] = [];
    if (preset === 'financieras') {
      picked = avail.filter((c) => finKeywords.some((k) => c.toLowerCase().includes(k)));
      setModelName('Modelo 1: Variables Financieras');
    } else if (preset === 'comerciales') {
      picked = avail.filter((c) => comKeywords.some((k) => c.toLowerCase().includes(k)));
      setModelName('Modelo 2: Variables Comerciales');
    } else {
      picked = avail.filter((c) => !c.toLowerCase().startsWith('id'));
      setModelName('Modelo 3: Combinación Completa');
    }
    if (picked.length === 0) picked = avail.slice(0, Math.min(3, avail.length));
    onUpdateConfig((prev) => ({ ...prev, featureColumns: picked }));
  };

  const equationTerms =
    activeModel && !activeModel.error
      ? activeModel.coefficients
          .map((c, idx) => {
            const b = c.beta.toFixed(4);
            if (idx === 0) return b;
            return `${c.beta >= 0 ? ' + ' : ' - '}${Math.abs(c.beta).toFixed(4)}·(${c.term})`;
          })
          .join('')
      : '';

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-900">
            Modelo de Regresión Logística Binaria (MLE / glm)
          </h1>
          <p className="text-sm text-slate-600">
            Estimación real mediante Máxima Verosimilitud (IRLS), equivalente a <code className="font-mono text-xs">glm(formula, family = binomial)</code> en R.
          </p>
        </div>
        <button
          type="button"
          onClick={() => setShowExplain(true)}
          className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3.5 py-2 text-xs font-medium text-slate-800 hover:bg-slate-50 whitespace-nowrap shrink-0"
        >
          <HelpCircle className="h-4 w-4 text-teal-600" />
          Cómo funciona Z y Sigmoide
        </button>
      </div>

      {/* Configuración rápida del modelo */}
      <section className="rounded-xl border border-slate-200 bg-white p-6 space-y-5">
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <div>
            <label className="block text-xs font-medium text-slate-700">Nombre del modelo</label>
            <input
              type="text"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2 text-xs text-slate-900"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-700">
              Variable dependiente (Y = 1: {config.positiveClass || '?'})
            </label>
            <select
              value={config.targetColumn}
              onChange={(e) => {
                const col = e.target.value;
                const prof = dataset.profiles.find((p) => p.name === col);
                const cats = [...(prof?.frequencies.map((f) => f.value) || [])].sort();
                onUpdateConfig((prev) => ({
                  ...prev,
                  targetColumn: col,
                  negativeClass: cats[0] || '',
                  positiveClass: cats[1] || cats[0] || '',
                  featureColumns: prev.featureColumns.filter((f) => f !== col),
                }));
              }}
              className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 font-mono text-xs text-slate-900"
            >
              <option value="">Seleccione objetivo...</option>
              {dataset.columns.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-700">
              Punto de corte (umbral): <span className="font-mono font-bold text-teal-700">{(config.threshold * 100).toFixed(0)}%</span>
            </label>
            <input
              type="range"
              min={0.05}
              max={0.95}
              step={0.01}
              value={config.threshold}
              onChange={(e) => onThresholdChange(Number(e.target.value))}
              className="mt-3 w-full accent-teal-600"
            />
          </div>
        </div>

        {/* Selección de variables + presets de comparación */}
        <div>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-xs font-medium text-slate-700">
              Variables independientes ({config.featureColumns.length} seleccionadas):
            </span>
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={() => applyPreset('financieras')}
                className="rounded-md border border-slate-300 bg-slate-50 px-2.5 py-1 text-[11px] font-medium text-slate-700 hover:bg-slate-100"
              >
                Preset 1: Financieras
              </button>
              <button
                type="button"
                onClick={() => applyPreset('comerciales')}
                className="rounded-md border border-slate-300 bg-slate-50 px-2.5 py-1 text-[11px] font-medium text-slate-700 hover:bg-slate-100"
              >
                Preset 2: Comerciales
              </button>
              <button
                type="button"
                onClick={() => applyPreset('combinado')}
                className="rounded-md border border-slate-300 bg-slate-50 px-2.5 py-1 text-[11px] font-medium text-slate-700 hover:bg-slate-100"
              >
                Preset 3: Combinado
              </button>
            </div>
          </div>

          <div className="mt-2.5 flex flex-wrap gap-2">
            {dataset.columns
              .filter((c) => c !== config.targetColumn)
              .map((col) => {
                const selected = config.featureColumns.includes(col);
                return (
                  <button
                    key={col}
                    type="button"
                    onClick={() =>
                      onUpdateConfig((prev) => ({
                        ...prev,
                        featureColumns: selected
                          ? prev.featureColumns.filter((f) => f !== col)
                          : [...prev.featureColumns, col],
                      }))
                    }
                    className={`rounded-lg border px-3 py-1.5 font-mono text-xs transition-colors ${
                      selected
                        ? 'border-teal-600 bg-teal-600 text-white'
                        : 'border-slate-300 bg-white text-slate-700 hover:border-slate-400'
                    }`}
                  >
                    {col}
                  </button>
                );
              })}
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-4 border-t border-slate-200 pt-4">
          <div className="space-y-1">
            <div className="font-mono text-xs text-slate-600">
              Fórmula: <strong className="text-slate-900">{config.targetColumn || 'Y'}</strong> ~{' '}
              {config.featureColumns.length > 0 ? config.featureColumns.join(' + ') : '...'}
            </div>
            {activeModel &&
              !activeModel.error &&
              (activeModel.config.targetColumn !== config.targetColumn ||
                activeModel.config.positiveClass !== config.positiveClass ||
                activeModel.config.negativeClass !== config.negativeClass ||
                activeModel.config.balanceMethod !== config.balanceMethod ||
                activeModel.config.splitConfig.trainRatio !== config.splitConfig.trainRatio ||
                activeModel.config.splitConfig.seed !== config.splitConfig.seed ||
                activeModel.config.splitConfig.stratified !== config.splitConfig.stratified ||
                JSON.stringify(activeModel.config.prepConfig) !== JSON.stringify(config.prepConfig) ||
                JSON.stringify(activeModel.config.featureColumns) !==
                  JSON.stringify(config.featureColumns)) && (
                <p className="text-xs font-medium text-amber-700">
                  La configuración actual difiere del modelo mostrado abajo. Presione &quot;Estimar Modelo&quot; para actualizar los resultados.
                </p>
              )}
          </div>
          <button
            type="button"
            onClick={() => onTrainModel(modelName || 'Modelo Logístico')}
            className="inline-flex items-center gap-2 rounded-lg bg-slate-900 px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-slate-800 whitespace-nowrap shrink-0"
          >
            <Play className="h-4 w-4" />
            Estimar Modelo por Máxima Verosimilitud
          </button>
        </div>
      </section>

      {/* Error o Advertencia del estimador */}
      {activeModel?.error && (
        <div className="flex items-start gap-3 rounded-xl border border-rose-200 bg-rose-50 p-5 text-sm text-rose-900">
          <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-rose-600" />
          <div>
            <p className="font-semibold">No fue posible estimar el modelo</p>
            <p className="mt-1 text-xs leading-relaxed">{activeModel.error}</p>
          </div>
        </div>
      )}

      {activeModel && !activeModel.error && (
        <>
          {activeModel.warning && (
            <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs text-amber-900">
              {activeModel.warning}
            </div>
          )}

          {/* Estructura matemática paso a paso */}
          <section className="rounded-xl border border-slate-200 bg-white p-6">
            <h2 className="text-base font-semibold text-slate-900">
              Ecuación Logística Estimada (Paso a Paso)
            </h2>
            <div className="mt-3 space-y-3">
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 font-mono text-xs text-slate-900 break-words">
                <p className="text-slate-500">1. Combinación lineal (Logit Z):</p>
                <p className="mt-1 font-semibold">Z = {equationTerms}</p>
              </div>
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 font-mono text-xs text-slate-900">
                <p className="text-slate-500">2. Transformación logística a probabilidad:</p>
                <p className="mt-1 font-semibold">
                  P({activeModel.config.targetColumn} = &quot;{activeModel.config.positiveClass}&quot;) = 1 / (1 + e^(-Z))
                </p>
              </div>
            </div>
          </section>

          {/* Tabla estilo glm() de R */}
          <section className="rounded-xl border border-slate-200 bg-white p-6">
            <div className="flex flex-col gap-2 border-b border-slate-200 pb-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-base font-semibold text-slate-900">
                  Resumen Estadístico del Modelo (Estilo R glm · family = binomial)
                </h2>
                <p className="text-xs text-slate-600">
                  Estimado en entrenamiento ({activeModel.balancedTrainCount} registros tras balanceo &quot;{activeModel.config.balanceMethod}&quot;) · {activeModel.iterations} iteraciones
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-3 font-mono text-xs text-slate-700 tabular-nums">
                <span>AIC: <strong>{activeModel.aic.toFixed(2)}</strong></span>
                <span>·</span>
                <span>Dev. Residual: <strong>{activeModel.residualDeviance.toFixed(2)}</strong></span>
                <span>·</span>
                <span>R² McFadden: <strong>{(activeModel.mcfaddenR2 * 100).toFixed(2)}%</strong></span>
              </div>
            </div>

            <div className="mt-4 overflow-x-auto">
              <table className="w-full border-collapse text-left text-xs tabular-nums">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500">
                    <th className="py-2 pr-4 font-medium">Término</th>
                    <th className="py-2 pr-4 text-right font-medium">Estimado (β)</th>
                    <th className="py-2 pr-4 text-right font-medium">Error Est.</th>
                    <th className="py-2 pr-4 text-right font-medium">Valor z</th>
                    <th className="py-2 pr-4 text-right font-medium">Pr(&gt;|z|)</th>
                    <th className="py-2 pr-4 text-right font-medium">Odds Ratio (e^β)</th>
                    <th className="py-2 text-right font-medium">IC 95% OR</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {activeModel.coefficients.map((c) => {
                    const stars = sigStars(c.pValue);
                    return (
                      <tr key={c.term} className="hover:bg-slate-50">
                        <td className="py-2.5 pr-4 font-mono font-semibold text-slate-900">
                          {c.term}
                        </td>
                        <td className="py-2.5 pr-4 text-right font-mono text-slate-900">
                          {c.beta.toFixed(5)}
                        </td>
                        <td className="py-2.5 pr-4 text-right font-mono text-slate-700">
                          {c.stdError !== null ? c.stdError.toFixed(5) : 'N/D'}
                        </td>
                        <td className="py-2.5 pr-4 text-right font-mono text-slate-700">
                          {c.zValue !== null ? c.zValue.toFixed(3) : 'N/D'}
                        </td>
                        <td className="py-2.5 pr-4 text-right font-mono">
                          <span className={c.pValue !== null && c.pValue < 0.05 ? 'font-semibold text-teal-700' : 'text-slate-700'}>
                            {c.pValue !== null
                              ? c.pValue < 0.0001
                                ? '< 0.0001'
                                : c.pValue.toFixed(4)
                              : 'N/D'}{' '}
                            {stars}
                          </span>
                        </td>
                        <td className="py-2.5 pr-4 text-right font-mono font-semibold text-slate-900">
                          {c.oddsRatio.toFixed(4)}
                        </td>
                        <td className="py-2.5 text-right font-mono text-slate-600">
                          {c.ciLower95 !== null && c.ciUpper95 !== null
                            ? `[${c.ciLower95.toFixed(3)}, ${c.ciUpper95.toFixed(3)}]`
                            : 'N/D'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-[11px] text-slate-500">
              Códigos de significancia: 0 &apos;***&apos; 0.001 &apos;**&apos; 0.01 &apos;*&apos; 0.05 &apos;.&apos; 0.1. Devianza nula: {activeModel.nullDeviance.toFixed(2)} · Log-Verosimilitud: {activeModel.logLikelihood.toFixed(2)}.
            </p>
          </section>

          {/* Curva Sigmoide interactiva */}
          <section className="rounded-xl border border-slate-200 bg-white p-6">
            <div className="flex flex-col gap-2 border-b border-slate-200 pb-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-base font-semibold text-slate-900">
                  Curva Sigmoide y Zonas de Clasificación
                </h2>
                <p className="text-xs text-slate-600">
                  Mueve el punto de corte arriba para observar cómo cambian las zonas 0 y 1 sobre las observaciones de prueba.
                </p>
              </div>
              <button
                type="button"
                onClick={() => onNavigate('predicciones')}
                className="text-xs font-medium text-teal-700 hover:text-teal-900 whitespace-nowrap"
              >
                Probar predicciones individuales →
              </button>
            </div>
            <div className="mt-4">
              <SigmoidChart
                threshold={activeModel.config.threshold}
                records={activeModel.testPredictions}
                positiveLabel={activeModel.config.positiveClass}
                negativeLabel={activeModel.config.negativeClass}
              />
            </div>
          </section>
        </>
      )}

      {/* Comparación de modelos guardados */}
      {savedModels.length > 0 && (
        <section className="rounded-xl border border-slate-200 bg-white p-6">
          <div className="flex items-center gap-2 border-b border-slate-200 pb-3">
            <Layers className="h-4 w-4 text-teal-700" />
            <h2 className="text-base font-semibold text-slate-900">
              Comparación de Modelos Estimados (Mismas Particiones de Prueba)
            </h2>
          </div>
          <p className="mt-1 text-xs text-slate-600">
            No seleccione únicamente por Accuracy: compare Precision, Recall y F1-Score de la categoría de interés. Haga clic en un modelo para activarlo.
          </p>

          <div className="mt-4 overflow-x-auto">
            <table className="w-full border-collapse text-left text-xs tabular-nums">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="py-2 pr-3 font-medium">Modelo</th>
                  <th className="py-2 pr-3 font-medium">Predictores</th>
                  <th className="py-2 pr-3 font-medium">Partición</th>
                  <th className="py-2 pr-3 text-right font-medium">AIC</th>
                  <th className="py-2 pr-3 text-right font-medium">Accuracy (Test)</th>
                  <th className="py-2 pr-3 text-right font-medium">Precision 1</th>
                  <th className="py-2 pr-3 text-right font-medium">Recall 1</th>
                  <th className="py-2 pr-3 text-right font-medium">F1-Score 1</th>
                  <th className="py-2 text-right font-medium">Bal. Acc.</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {savedModels.map((m) => {
                  const cm = computeConfusionMatrix(m.testPredictions, m.config.threshold);
                  const met = computeEvaluationMetrics(cm);
                  const isActive = activeModel?.id === m.id;
                  return (
                    <tr
                      key={m.id}
                      onClick={() => onSelectSavedModel(m.id)}
                      className={`cursor-pointer transition-colors ${
                        isActive ? 'bg-teal-50/70 font-medium' : 'hover:bg-slate-50'
                      }`}
                    >
                      <td className="py-2.5 pr-3 text-slate-900">
                        {m.name} {isActive && <span className="text-teal-700">(Activo)</span>}
                      </td>
                      <td className="py-2.5 pr-3 font-mono text-slate-600 max-w-[180px] truncate" title={m.config.featureColumns.join(', ')}>
                        {m.config.featureColumns.join(', ')}
                      </td>
                      <td className="py-2.5 pr-3 font-mono text-[11px] text-slate-500 whitespace-nowrap">
                        {Math.round(m.config.splitConfig.trainRatio * 100)}/{Math.round((1 - m.config.splitConfig.trainRatio) * 100)} · s={m.config.splitConfig.seed}
                      </td>
                      <td className="py-2.5 pr-3 text-right font-mono">{m.aic.toFixed(1)}</td>
                      <td className="py-2.5 pr-3 text-right font-mono">
                        {met.accuracy !== null ? `${(met.accuracy * 100).toFixed(2)}%` : 'N/D'}
                      </td>
                      <td className="py-2.5 pr-3 text-right font-mono">
                        {met.precision1 !== null ? `${(met.precision1 * 100).toFixed(2)}%` : 'N/D'}
                      </td>
                      <td className="py-2.5 pr-3 text-right font-mono">
                        {met.recall1 !== null ? `${(met.recall1 * 100).toFixed(2)}%` : 'N/D'}
                      </td>
                      <td className="py-2.5 pr-3 text-right font-mono font-semibold text-teal-700">
                        {met.f1_1 !== null ? `${(met.f1_1 * 100).toFixed(2)}%` : 'N/D'}
                      </td>
                      <td className="py-2.5 text-right font-mono">
                        {met.balancedAccuracy !== null ? `${(met.balancedAccuracy * 100).toFixed(2)}%` : 'N/D'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <StatExplainModal
        open={showExplain}
        onClose={() => setShowExplain(false)}
        title="¿Cómo estima la Regresión Logística el valor Z y la Probabilidad?"
        formula="Z = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ   |   P(Y = 1) = 1 / (1 + e⁻ᶻ)"
        summary="A diferencia de la regresión lineal, la regresión logística modela el logaritmo de los odds (razón de probabilidades) para garantizar que la salida siempre esté acotada entre 0 y 1 (0% a 100%)."
        details={[
          'Máxima Verosimilitud (MLE): busca los coeficientes β que maximizan la probabilidad conjunta de observar los datos reales de entrenamiento.',
          'Odds Ratio (e^β): indica cuánto se multiplican los odds del evento positivo por cada aumento de una unidad en la variable.',
          'Punto de corte: convierte la probabilidad continua P(Y=1) en una clasificación binaria (0 o 1).',
        ]}
        example="Si Z = 0, entonces e⁰ = 1 y P(Y=1) = 1 / (1 + 1) = 0.50 (50%). Valores positivos de Z producen probabilidades mayores a 50%."
      />
    </div>
  );
};
