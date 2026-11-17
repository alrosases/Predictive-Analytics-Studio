import React, { useEffect, useMemo, useState } from 'react';
import { Calculator, Download, Upload } from 'lucide-react';
import type { DataRow, DatasetState, ModelResult, ModuleId } from '../../types/analytics';
import {
  predictBatchRows,
  predictSingleObservation,
} from '../../engine/logisticRegression';
import { parseDatasetFile } from '../../engine/statistics';
import { downloadCSVRows, exportPredictionsCSV } from '../../engine/exporters';
import { SigmoidChart } from '../charts/SigmoidChart';

interface PredictionsViewProps {
  dataset: DatasetState | null;
  activeModel: ModelResult | null;
  onThresholdChange: (threshold: number) => void;
  onNavigate: (mod: ModuleId) => void;
}

export const PredictionsView: React.FC<PredictionsViewProps> = ({
  dataset,
  activeModel,
  onThresholdChange,
  onNavigate,
}) => {
  const [inputs, setInputs] = useState<Record<string, string | number>>({});
  const [batchSource, setBatchSource] = useState<'test' | 'all' | 'external'>('test');
  const [externalRows, setExternalRows] = useState<DataRow[]>([]);
  const [externalName, setExternalName] = useState<string>('');
  const [externalError, setExternalError] = useState<string | null>(null);

  // Initialize form inputs using real column means or most frequent categories
  useEffect(() => {
    if (!activeModel || activeModel.error || !dataset) return;
    const defaults: Record<string, string | number> = {};
    for (const col of activeModel.config.featureColumns) {
      const catInfo = activeModel.categoryLevelsByColumn[col];
      if (catInfo) {
        defaults[col] = catInfo.reference;
      } else {
        const prof = dataset.profiles.find((p) => p.name === col);
        defaults[col] = prof?.numericStats ? Number(prof.numericStats.median.toFixed(2)) : 0;
      }
    }
    setInputs(defaults);
  }, [activeModel?.id]);

  const singleResult = useMemo(() => {
    if (!activeModel || activeModel.error) return null;
    return predictSingleObservation(activeModel, inputs, activeModel.config.threshold);
  }, [activeModel, inputs]);

  const batchResults = useMemo(() => {
    if (!activeModel || activeModel.error || !dataset) return [];
    if (batchSource === 'test') {
      return activeModel.testPredictions.map((r) => ({
        rowIndex: r.rowIndex,
        inputs: r.rawRow,
        zValue: r.zValue,
        probability: r.probability,
        predicted: (r.probability >= activeModel.config.threshold ? 1 : 0) as 0 | 1,
        predictedLabel:
          r.probability >= activeModel.config.threshold
            ? activeModel.config.positiveClass
            : activeModel.config.negativeClass,
        actual: r.actualLabel,
      }));
    }
    if (batchSource === 'all') {
      return predictBatchRows(activeModel, dataset.workingRows, activeModel.config.threshold);
    }
    return predictBatchRows(activeModel, externalRows, activeModel.config.threshold);
  }, [activeModel, dataset, batchSource, externalRows]);

  if (!dataset || !activeModel || activeModel.error) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-10 text-center">
        <h2 className="text-lg font-semibold text-slate-900">
          Se requiere un modelo logístico estimado
        </h2>
        <p className="mt-1 text-sm text-slate-600">
          Estime primero un modelo de regresión logística para calcular probabilidades individuales y masivas.
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

  const handleExternalFile = async (file: File) => {
    setExternalError(null);
    try {
      const parsed = await parseDatasetFile(file);
      const missingCols = activeModel.config.featureColumns.filter(
        (c) => !parsed.columns.includes(c)
      );
      if (missingCols.length > 0) {
        throw new Error(
          `El archivo no contiene las variables predictoras requeridas: ${missingCols.join(', ')}`
        );
      }
      setExternalRows(parsed.rows);
      setExternalName(file.name);
      setBatchSource('external');
    } catch (err: unknown) {
      setExternalError(err instanceof Error ? err.message : 'Error al leer el archivo.');
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-900">
            Predicciones Individuales y Masivas
          </h1>
          <p className="text-sm text-slate-600">
            Calcule el valor de Z, la probabilidad estimada P(Y=1) y la clasificación según el umbral de corte.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <label className="text-xs font-medium text-slate-700">
            Umbral:{' '}
            <span className="font-mono font-bold text-teal-700">
              {(activeModel.config.threshold * 100).toFixed(0)}%
            </span>
          </label>
          <input
            type="range"
            min={0.05}
            max={0.95}
            step={0.01}
            value={activeModel.config.threshold}
            onChange={(e) => onThresholdChange(Number(e.target.value))}
            className="w-32 accent-teal-600"
          />
        </div>
      </div>

      {/* Predicción individual */}
      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <div className="flex items-center gap-2 border-b border-slate-200 pb-3">
          <Calculator className="h-4 w-4 text-teal-700" />
          <h2 className="text-base font-semibold text-slate-900">
            Predicción Individual (Sustitución en la Ecuación del Modelo)
          </h2>
        </div>

        <div className="mt-4 grid grid-cols-1 gap-6 lg:grid-cols-12">
          {/* Formulario de variables */}
          <div className="space-y-3 lg:col-span-5">
            {activeModel.config.featureColumns.map((col) => {
              const catInfo = activeModel.categoryLevelsByColumn[col];
              return (
                <div key={col}>
                  <label className="block font-mono text-xs font-medium text-slate-700">
                    {col}
                  </label>
                  {catInfo ? (
                    <select
                      value={String(inputs[col] ?? catInfo.reference)}
                      onChange={(e) =>
                        setInputs((prev) => ({ ...prev, [col]: e.target.value }))
                      }
                      className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs text-slate-900"
                    >
                      {catInfo.levels.map((lvl) => (
                        <option key={lvl} value={lvl}>
                          {lvl}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="number"
                      step="any"
                      value={inputs[col] ?? ''}
                      onChange={(e) =>
                        setInputs((prev) => ({
                          ...prev,
                          [col]: e.target.value === '' ? '' : Number(e.target.value),
                        }))
                      }
                      className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 font-mono text-xs text-slate-900"
                    />
                  )}
                </div>
              );
            })}
          </div>

          {/* Resultado paso a paso */}
          {singleResult && (
            <div className="space-y-4 lg:col-span-7">
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3 tabular-nums">
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3.5">
                  <p className="text-xs text-slate-500">1. Valor de Z</p>
                  <p className="mt-1 font-mono text-xl font-bold text-slate-900">
                    {singleResult.zValue.toFixed(4)}
                  </p>
                </div>
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3.5">
                  <p className="text-xs text-slate-500">2. Probabilidad P(Y=1)</p>
                  <p className="mt-1 font-mono text-xl font-bold text-teal-700">
                    {(singleResult.probability * 100).toFixed(2)}%
                  </p>
                </div>
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3.5">
                  <p className="text-xs text-slate-500">3. Clasificación ({(singleResult.threshold * 100).toFixed(0)}%)</p>
                  <p className="mt-1 font-mono text-lg font-bold text-slate-900">
                    {singleResult.predicted} ({singleResult.predictedLabel})
                  </p>
                </div>
              </div>

              {/* Desglose matemático */}
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-3.5 font-mono text-xs text-slate-800 space-y-1.5">
                <p className="font-sans font-semibold text-slate-900">
                  Cálculo detallado con los valores ingresados:
                </p>
                <div className="max-h-36 overflow-y-auto space-y-1">
                  {singleResult.steps.map((st, idx) => (
                    <div key={idx} className="flex justify-between gap-2 tabular-nums">
                      <span className="truncate">
                        {idx === 0 ? 'β₀ (Intercepto)' : `${st.term} (${st.rawDisplay}) × ${st.beta.toFixed(4)}`}
                      </span>
                      <span className="font-semibold">
                        {st.contribution >= 0 && idx > 0 ? '+' : ''}
                        {st.contribution.toFixed(4)}
                      </span>
                    </div>
                  ))}
                </div>
                <div className="border-t border-slate-300 pt-1.5 flex justify-between font-semibold">
                  <span>P(Y=1) = 1 / (1 + e^(-{singleResult.zValue.toFixed(4)}))</span>
                  <span className="text-teal-700">{(singleResult.probability * 100).toFixed(2)}%</span>
                </div>
              </div>

              {/* Interpretación automática */}
              <div className="rounded-lg border border-teal-200 bg-teal-50/50 p-3.5 text-xs leading-relaxed text-slate-800">
                <strong>Interpretación: </strong>
                El modelo estima una probabilidad de{' '}
                <strong>{(singleResult.probability * 100).toFixed(2)}%</strong> para la categoría{' '}
                <strong>&quot;{activeModel.config.positiveClass}&quot;</strong>. Como este valor{' '}
                {singleResult.probability >= singleResult.threshold
                  ? 'supera o iguala'
                  : 'no alcanza'}{' '}
                el umbral de clasificación de {(singleResult.threshold * 100).toFixed(0)}%, clasifica
                la observación en la categoría{' '}
                <strong>
                  {singleResult.predicted} ({singleResult.predictedLabel})
                </strong>
                .
              </div>
            </div>
          )}
        </div>

        {singleResult && (
          <div className="mt-6 border-t border-slate-200 pt-5">
            <h3 className="mb-2 text-xs font-semibold text-slate-700">
              Ubicación del caso ingresado sobre la Curva Sigmoide (punto rojo)
            </h3>
            <SigmoidChart
              threshold={activeModel.config.threshold}
              records={activeModel.testPredictions}
              highlightZ={singleResult.zValue}
              positiveLabel={activeModel.config.positiveClass}
              negativeLabel={activeModel.config.negativeClass}
            />
          </div>
        )}
      </section>

      {/* Predicciones Masivas */}
      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <div className="flex flex-col gap-3 border-b border-slate-200 pb-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Predicciones Masivas y Exportación
            </h2>
            <p className="text-xs text-slate-600">
              Visualice probabilidades y clasificaciones para todo el conjunto de prueba, la base completa o un nuevo archivo CSV/Excel.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => {
                if (batchSource === 'external' && batchResults.length > 0) {
                  downloadCSVRows(
                    batchResults.map((r) => ({
                      fila: r.rowIndex,
                      ...r.inputs,
                      Z: Number(r.zValue.toFixed(6)),
                      probabilidad: Number(r.probability.toFixed(6)),
                      prediccion: r.predictedLabel,
                    })),
                    ['fila', ...activeModel.config.featureColumns, 'Z', 'probabilidad', 'prediccion'],
                    `predicciones_${externalName.replace(/\.[^.]+$/, '')}.csv`
                  );
                } else {
                  exportPredictionsCSV(activeModel, batchSource === 'all' ? 'all' : 'test');
                }
              }}
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-800 hover:bg-slate-50"
            >
              <Download className="h-3.5 w-3.5" />
              Descargar Predicciones CSV
            </button>
            <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg bg-slate-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-slate-800">
              <Upload className="h-3.5 w-3.5" />
              Predecir Nuevo Archivo
              <input
                type="file"
                accept=".csv,.xls,.xlsx"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleExternalFile(f);
                  e.target.value = '';
                }}
              />
            </label>
          </div>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-2">
          <div className="inline-flex rounded-lg bg-slate-100 p-1 text-xs">
            <button
              type="button"
              onClick={() => setBatchSource('test')}
              className={`rounded-md px-3 py-1.5 font-medium ${
                batchSource === 'test' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600'
              }`}
            >
              Conjunto de Prueba ({activeModel.testCount})
            </button>
            <button
              type="button"
              onClick={() => setBatchSource('all')}
              className={`rounded-md px-3 py-1.5 font-medium ${
                batchSource === 'all' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600'
              }`}
            >
              Base Completa ({dataset.workingRows.length})
            </button>
            {externalRows.length > 0 && (
              <button
                type="button"
                onClick={() => setBatchSource('external')}
                className={`rounded-md px-3 py-1.5 font-medium ${
                  batchSource === 'external' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600'
                }`}
              >
                {externalName} ({externalRows.length})
              </button>
            )}
          </div>

          {batchSource === 'external' && batchResults.length > 0 && (
            <button
              type="button"
              onClick={() =>
                downloadCSVRows(
                  batchResults.map((r) => ({
                    fila: r.rowIndex,
                    ...r.inputs,
                    Z: Number(r.zValue.toFixed(6)),
                    probabilidad: Number(r.probability.toFixed(6)),
                    prediccion: r.predictedLabel,
                  })),
                  ['fila', ...activeModel.config.featureColumns, 'Z', 'probabilidad', 'prediccion'],
                  `predicciones_${externalName.replace(/\.[^.]+$/, '')}.csv`
                )
              }
              className="text-xs font-medium text-teal-700 hover:text-teal-900"
            >
              Exportar archivo predicho
            </button>
          )}
        </div>

        {externalError && (
          <p className="mt-3 text-xs text-rose-700">{externalError}</p>
        )}

        <div className="mt-4 max-h-80 overflow-auto">
          <table className="w-full border-collapse text-left text-xs tabular-nums">
            <thead className="sticky top-0 bg-white">
              <tr className="border-b border-slate-200 text-slate-500">
                <th className="py-2 pr-3 font-mono font-medium">Fila</th>
                {activeModel.config.featureColumns.slice(0, 4).map((c) => (
                  <th key={c} className="py-2 pr-3 font-mono font-medium">{c}</th>
                ))}
                <th className="py-2 pr-3 text-right font-medium">Valor Z</th>
                <th className="py-2 pr-3 text-right font-medium">Probabilidad P(Y=1)</th>
                <th className="py-2 pr-3 font-medium">Predicción</th>
                {batchSource !== 'external' && <th className="py-2 font-medium">Valor Real</th>}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {batchResults.slice(0, 100).map((r) => (
                <tr key={r.rowIndex} className="hover:bg-slate-50">
                  <td className="py-2 pr-3 font-mono text-slate-400">{r.rowIndex}</td>
                  {activeModel.config.featureColumns.slice(0, 4).map((c) => (
                    <td key={c} className="py-2 pr-3 font-mono text-slate-800">
                      {String(r.inputs[c] ?? '—')}
                    </td>
                  ))}
                  <td className="py-2 pr-3 text-right font-mono text-slate-700">
                    {r.zValue.toFixed(3)}
                  </td>
                  <td className="py-2 pr-3 text-right font-mono font-semibold text-teal-700">
                    {(r.probability * 100).toFixed(2)}%
                  </td>
                  <td className="py-2 pr-3 font-medium text-slate-900">
                    {r.predicted} ({r.predictedLabel})
                  </td>
                  {batchSource !== 'external' && (
                    <td className="py-2 font-mono text-slate-600">{r.actual ?? '—'}</td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};
