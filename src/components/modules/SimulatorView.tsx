import React, { useEffect, useMemo, useState } from 'react';
import { BookmarkPlus, Trash2 } from 'lucide-react';
import type {
  DatasetState,
  ModelResult,
  ModuleId,
  ScenarioItem,
} from '../../types/analytics';
import { predictSingleObservation } from '../../engine/logisticRegression';
import { formatNumberES } from '../../engine/statistics';

interface SimulatorViewProps {
  dataset: DatasetState | null;
  activeModel: ModelResult | null;
  scenarios: ScenarioItem[];
  onSaveScenario: (s: ScenarioItem) => void;
  onRemoveScenario: (id: string) => void;
  onNavigate: (mod: ModuleId) => void;
}

const SECTORS = [
  {
    id: 'Financiero',
    desc: 'Evaluación de probabilidad de pago, riesgo de retraso y comportamiento de clientes.',
  },
  {
    id: 'Salud (Preventivo)',
    desc: 'Evaluación estadística de perfiles de riesgo clínico (herramienta analítica de apoyo; no sustituye diagnósticos médicos).',
  },
  {
    id: 'Marketing',
    desc: 'Propensión de compra, respuesta a promociones y retorno de clientes.',
  },
  {
    id: 'Recursos Humanos',
    desc: 'Análisis de probabilidad de rotación, abandono y satisfacción laboral.',
  },
];

export const SimulatorView: React.FC<SimulatorViewProps> = ({
  dataset,
  activeModel,
  scenarios,
  onSaveScenario,
  onRemoveScenario,
  onNavigate,
}) => {
  const [sector, setSector] = useState('Financiero');
  const [scenarioName, setScenarioName] = useState('Escenario Perfil A');
  const [inputs, setInputs] = useState<Record<string, string | number>>({});

  useEffect(() => {
    if (!activeModel || activeModel.error || !dataset) return;
    const init: Record<string, string | number> = {};
    for (const col of activeModel.config.featureColumns) {
      const cat = activeModel.categoryLevelsByColumn[col];
      if (cat) {
        init[col] = cat.reference;
      } else {
        const prof = dataset.profiles.find((p) => p.name === col);
        init[col] = prof?.numericStats ? Number(prof.numericStats.median.toFixed(2)) : 0;
      }
    }
    setInputs(init);
  }, [activeModel?.id]);

  const simResult = useMemo(() => {
    if (!activeModel || activeModel.error) return null;
    return predictSingleObservation(activeModel, inputs, activeModel.config.threshold);
  }, [activeModel, inputs]);

  if (!dataset || !activeModel || activeModel.error) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-10 text-center">
        <h2 className="text-lg font-semibold text-slate-900">
          Estime un modelo para usar el simulador
        </h2>
        <p className="mt-1 text-sm text-slate-600">
          El simulador ajusta las variables reales del modelo activo para evaluar distintos perfiles empresariales.
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

  const handleSave = () => {
    if (!simResult) return;
    onSaveScenario({
      id: `scn_${Date.now()}_${Math.floor(Math.random() * 1000)}`,
      name: scenarioName.trim() || `Escenario ${scenarios.length + 1}`,
      sector,
      inputs: { ...inputs },
      zValue: simResult.zValue,
      probability: simResult.probability,
      predicted: simResult.predicted,
      predictedLabel: simResult.predictedLabel,
      createdAt: new Date().toISOString(),
    });
    setScenarioName(`Escenario ${scenarios.length + 2}`);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-900">Simulador de Escenarios Empresariales</h1>
          <p className="text-sm text-slate-600">
            Modifique las variables predictoras para simular perfiles de clientes y comparar probabilidades estimadas.
          </p>
        </div>
        <button
          type="button"
          onClick={() => onNavigate('interpretacion')}
          className="rounded-lg bg-teal-600 px-4 py-2 text-xs font-medium text-white hover:bg-teal-700 whitespace-nowrap shrink-0"
        >
          Ver Interpretación →
        </button>
      </div>

      {/* Selector de contexto sectorial */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {SECTORS.map((sec) => {
          const active = sector === sec.id;
          return (
            <button
              key={sec.id}
              type="button"
              onClick={() => setSector(sec.id)}
              className={`rounded-xl border p-3.5 text-left transition-colors ${
                active
                  ? 'border-teal-600 bg-teal-50/40'
                  : 'border-slate-200 bg-white hover:border-slate-300'
              }`}
            >
              <p className="text-xs font-semibold text-slate-900">{sec.id}</p>
              <p className="mt-1 text-[11px] leading-relaxed text-slate-600">{sec.desc}</p>
            </button>
          );
        })}
      </div>

      {/* Controles interactivos de simulación */}
      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
          <div className="space-y-4 lg:col-span-7">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
              <div className="flex-1">
                <label className="block text-xs font-medium text-slate-700">
                  Nombre del escenario a guardar
                </label>
                <input
                  type="text"
                  value={scenarioName}
                  onChange={(e) => setScenarioName(e.target.value)}
                  className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-xs text-slate-900"
                />
              </div>
              <div className="pt-5">
                <button
                  type="button"
                  onClick={handleSave}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-slate-900 px-4 py-2 text-xs font-medium text-white hover:bg-slate-800 whitespace-nowrap"
                >
                  <BookmarkPlus className="h-3.5 w-3.5" />
                  Guardar Escenario
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              {activeModel.config.featureColumns.map((col) => {
                const cat = activeModel.categoryLevelsByColumn[col];
                const prof = dataset.profiles.find((p) => p.name === col);
                const stats = prof?.numericStats;

                if (cat) {
                  return (
                    <div key={col} className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
                      <label className="block font-mono text-xs font-medium text-slate-800">
                        {col}
                      </label>
                      <select
                        value={String(inputs[col] ?? cat.reference)}
                        onChange={(e) =>
                          setInputs((prev) => ({ ...prev, [col]: e.target.value }))
                        }
                        className="mt-2 w-full rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-xs text-slate-900"
                      >
                        {cat.levels.map((lvl) => (
                          <option key={lvl} value={lvl}>
                            {lvl}
                          </option>
                        ))}
                      </select>
                    </div>
                  );
                }

                const minVal = stats ? stats.min : 0;
                const maxVal = stats && stats.max > stats.min ? stats.max : minVal + 100;
                const currentNum = Number(inputs[col] ?? stats?.median ?? 0);
                const stepVal = (maxVal - minVal) / 100 || 1;

                return (
                  <div key={col} className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
                    <div className="flex items-center justify-between gap-2">
                      <label className="font-mono text-xs font-medium text-slate-800">{col}</label>
                      <input
                        type="number"
                        step="any"
                        value={currentNum}
                        onChange={(e) =>
                          setInputs((prev) => ({
                            ...prev,
                            [col]: Number(e.target.value),
                          }))
                        }
                        className="w-24 rounded border border-slate-300 bg-white px-2 py-1 text-right font-mono text-xs text-slate-900"
                      />
                    </div>
                    <input
                      type="range"
                      min={minVal}
                      max={maxVal}
                      step={stepVal}
                      value={Math.min(maxVal, Math.max(minVal, currentNum))}
                      onChange={(e) =>
                        setInputs((prev) => ({
                          ...prev,
                          [col]: Number(Number(e.target.value).toFixed(2)),
                        }))
                      }
                      className="mt-2 w-full accent-teal-600"
                    />
                    <div className="flex justify-between font-mono text-[10px] text-slate-400 tabular-nums">
                      <span>{formatNumberES(minVal, 1)}</span>
                      <span>{formatNumberES(maxVal, 1)}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Indicador en tiempo real */}
          {simResult && (
            <div className="flex flex-col justify-between rounded-xl border border-slate-200 bg-slate-50 p-6 lg:col-span-5">
              <div>
                <p className="text-xs font-medium text-slate-500">
                  Simulación Activa · Sector {sector}
                </p>
                <h3 className="mt-1 text-base font-semibold text-slate-900">
                  Probabilidad de {activeModel.config.positiveClass}
                </h3>
                <p className="mt-4 font-mono text-4xl font-bold text-teal-700 tabular-nums">
                  {(simResult.probability * 100).toFixed(2)}%
                </p>

                {/* Barra de probabilidad con marca de umbral */}
                <div className="mt-4">
                  <div className="relative h-3.5 w-full overflow-hidden rounded-full bg-slate-200">
                    <div
                      className={`h-full transition-all ${
                        simResult.predicted === 1 ? 'bg-teal-600' : 'bg-slate-700'
                      }`}
                      style={{ width: `${simResult.probability * 100}%` }}
                    />
                    <div
                      className="absolute top-0 bottom-0 w-0.5 bg-amber-600"
                      style={{ left: `${activeModel.config.threshold * 100}%` }}
                    />
                  </div>
                  <div className="mt-1.5 flex justify-between text-[11px] text-slate-500 tabular-nums">
                    <span>0%</span>
                    <span>Umbral: {(activeModel.config.threshold * 100).toFixed(0)}%</span>
                    <span>100%</span>
                  </div>
                </div>
              </div>

              <div className="mt-6 space-y-2 border-t border-slate-200 pt-4 text-xs tabular-nums">
                <div className="flex justify-between">
                  <span className="text-slate-600">Valor Z calculado:</span>
                  <span className="font-mono font-semibold text-slate-900">
                    {simResult.zValue.toFixed(4)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-600">Clasificación resultante:</span>
                  <span className="font-mono font-bold text-slate-900">
                    {simResult.predicted} ({simResult.predictedLabel})
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Tabla comparativa de escenarios guardados */}
      {scenarios.length > 0 && (
        <section className="rounded-xl border border-slate-200 bg-white p-6">
          <h2 className="text-base font-semibold text-slate-900">
            Comparación de Escenarios Guardados ({scenarios.length})
          </h2>
          <div className="mt-4 overflow-x-auto">
            <table className="w-full border-collapse text-left text-xs tabular-nums">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="py-2 pr-3 font-medium">Escenario</th>
                  <th className="py-2 pr-3 font-medium">Sector</th>
                  <th className="py-2 pr-3 font-medium">Parámetros</th>
                  <th className="py-2 pr-3 text-right font-medium">Z</th>
                  <th className="py-2 pr-3 text-right font-medium">Probabilidad</th>
                  <th className="py-2 pr-3 font-medium">Clasificación</th>
                  <th className="py-2 text-right font-medium">Quitar</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {scenarios.map((sc) => (
                  <tr key={sc.id} className="hover:bg-slate-50">
                    <td className="py-2.5 pr-3 font-semibold text-slate-900">{sc.name}</td>
                    <td className="py-2.5 pr-3 text-slate-600">{sc.sector}</td>
                    <td className="py-2.5 pr-3 font-mono text-slate-600">
                      {Object.entries(sc.inputs)
                        .map(([k, v]) => `${k}=${v}`)
                        .join(' · ')}
                    </td>
                    <td className="py-2.5 pr-3 text-right font-mono">{sc.zValue.toFixed(3)}</td>
                    <td className="py-2.5 pr-3 text-right font-mono font-bold text-teal-700">
                      {(sc.probability * 100).toFixed(2)}%
                    </td>
                    <td className="py-2.5 pr-3 font-medium text-slate-800">
                      {sc.predicted} ({sc.predictedLabel})
                    </td>
                    <td className="py-2.5 text-right">
                      <button
                        type="button"
                        onClick={() => onRemoveScenario(sc.id)}
                        className="text-slate-400 hover:text-rose-600"
                        aria-label="Eliminar escenario"
                      >
                        <Trash2 className="inline h-3.5 w-3.5" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
};
