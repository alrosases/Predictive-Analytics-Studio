import React, { useState } from 'react';
import type { DatasetState, ModuleId } from '../../types/analytics';
import { formatNumberES } from '../../engine/statistics';
import {
  BoxplotChart,
  CategoricalBarChart,
  CorrelationHeatmap,
  HistogramChart,
} from '../charts/DescriptiveCharts';

interface DescriptiveViewProps {
  dataset: DatasetState | null;
  onNavigate: (mod: ModuleId) => void;
}

export const DescriptiveView: React.FC<DescriptiveViewProps> = ({
  dataset,
  onNavigate,
}) => {
  const [selectedCol, setSelectedCol] = useState<string>('');

  if (!dataset) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-10 text-center">
        <h2 className="text-lg font-semibold text-slate-900">Sin datos para analizar</h2>
        <p className="mt-1 text-sm text-slate-600">
          Cargue primero una base de datos para generar medidas descriptivas, distribuciones y matrices de correlación.
        </p>
        <button
          type="button"
          onClick={() => onNavigate('cargar')}
          className="mt-5 rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800"
        >
          Cargar Base de Datos
        </button>
      </div>
    );
  }

  const activeProfile =
    dataset.profiles.find((p) => p.name === selectedCol) || dataset.profiles[0];

  const numericProfiles = dataset.profiles.filter((p) => p.numericStats !== null);

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-900">Análisis Descriptivo Completo</h1>
          <p className="text-sm text-slate-600">
            Estadísticas de tendencia central, dispersión, cuartiles, frecuencias, correlaciones e interpretaciones automáticas sobre {dataset.workingRows.length} registros.
          </p>
        </div>
        <button
          type="button"
          onClick={() => onNavigate('preparacion')}
          className="rounded-lg bg-teal-600 px-4 py-2 text-sm font-medium text-white hover:bg-teal-700 whitespace-nowrap shrink-0"
        >
          Preparar Datos →
        </button>
      </div>

      {/* Selector de variable para análisis visual detallado */}
      {activeProfile && (
        <section className="rounded-xl border border-slate-200 bg-white p-6">
          <div className="flex flex-col gap-3 border-b border-slate-200 pb-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h2 className="text-base font-semibold text-slate-900">
                Inspección Detallada de Variable
              </h2>
              <p className="text-xs text-slate-600">
                Seleccione cualquier variable para visualizar su distribución e interpretación automática.
              </p>
            </div>
            <select
              value={activeProfile.name}
              onChange={(e) => setSelectedCol(e.target.value)}
              className="rounded-lg border border-slate-300 bg-white px-3 py-2 font-mono text-xs font-medium text-slate-900"
            >
              {dataset.profiles.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.name} ({p.effectiveKind})
                </option>
              ))}
            </select>
          </div>

          <div className="mt-5 grid grid-cols-1 gap-6 lg:grid-cols-2">
            {/* Gráfico 1: Histograma o Barras */}
            <div className="space-y-3">
              <h3 className="text-sm font-semibold text-slate-800">
                {activeProfile.numericStats
                  ? `Histograma de frecuencias: ${activeProfile.name}`
                  : `Diagrama de barras: ${activeProfile.name}`}
              </h3>
              {activeProfile.numericStats ? (
                <HistogramChart rows={dataset.workingRows} profile={activeProfile} />
              ) : (
                <CategoricalBarChart profile={activeProfile} />
              )}
            </div>

            {/* Gráfico 2: Boxplot o Tabla de Frecuencias */}
            <div className="space-y-4">
              {activeProfile.numericStats ? (
                <>
                  <h3 className="text-sm font-semibold text-slate-800">
                    Diagrama de Caja (Cuartiles y Atípicos)
                  </h3>
                  <BoxplotChart profile={activeProfile} />
                  <div className="grid grid-cols-3 gap-2.5 text-xs tabular-nums">
                    <div className="rounded-lg border border-slate-200 bg-slate-50 p-2.5">
                      <span className="text-slate-500">Varianza</span>
                      <p className="mt-0.5 font-mono font-semibold text-slate-900">
                        {formatNumberES(activeProfile.numericStats.variance, 2)}
                      </p>
                    </div>
                    <div className="rounded-lg border border-slate-200 bg-slate-50 p-2.5">
                      <span className="text-slate-500">Desv. Estándar</span>
                      <p className="mt-0.5 font-mono font-semibold text-slate-900">
                        {formatNumberES(activeProfile.numericStats.stdDev, 2)}
                      </p>
                    </div>
                    <div className="rounded-lg border border-slate-200 bg-slate-50 p-2.5">
                      <span className="text-slate-500">Moda</span>
                      <p className="mt-0.5 truncate font-mono font-semibold text-slate-900" title={activeProfile.numericStats.mode}>
                        {activeProfile.numericStats.mode}
                      </p>
                    </div>
                  </div>
                </>
              ) : (
                <>
                  <h3 className="text-sm font-semibold text-slate-800">
                    Tabla de Frecuencias (Absoluta, Relativa y Porcentaje)
                  </h3>
                  <div className="max-h-56 overflow-y-auto">
                    <table className="w-full border-collapse text-xs tabular-nums">
                      <thead>
                        <tr className="border-b border-slate-200 text-slate-500">
                          <th className="py-1.5 text-left font-medium">Categoría</th>
                          <th className="py-1.5 text-right font-medium">Frec. Abs.</th>
                          <th className="py-1.5 text-right font-medium">Frec. Rel.</th>
                          <th className="py-1.5 text-right font-medium">%</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {activeProfile.frequencies.map((f) => (
                          <tr key={f.value}>
                            <td className="py-1.5 font-medium text-slate-800">{f.value}</td>
                            <td className="py-1.5 text-right font-mono text-slate-700">{f.count}</td>
                            <td className="py-1.5 text-right font-mono text-slate-700">
                              {f.relativeFreq.toFixed(4)}
                            </td>
                            <td className="py-1.5 text-right font-mono font-semibold text-teal-700">
                              {formatNumberES(f.percentage, 2)}%
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Interpretación automática basada en datos reales */}
          <div className="mt-5 rounded-lg border border-teal-200 bg-teal-50/50 p-4 text-sm text-slate-800">
            <span className="font-semibold text-teal-900">Interpretación estadística automática: </span>
            {activeProfile.interpretation}
          </div>
        </section>
      )}

      {/* Tabla maestra de estadísticas numéricas */}
      {numericProfiles.length > 0 && (
        <section className="rounded-xl border border-slate-200 bg-white p-6">
          <h2 className="text-base font-semibold text-slate-900">
            Resumen Estadístico de Variables Numéricas
          </h2>
          <div className="mt-4 overflow-x-auto">
            <table className="w-full border-collapse text-xs tabular-nums">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="py-2 pr-3 text-left font-medium">Variable</th>
                  <th className="py-2 pr-3 text-right font-medium">N</th>
                  <th className="py-2 pr-3 text-right font-medium">Perdidos</th>
                  <th className="py-2 pr-3 text-right font-medium">Media</th>
                  <th className="py-2 pr-3 text-right font-medium">Mediana</th>
                  <th className="py-2 pr-3 text-right font-medium">Mínimo</th>
                  <th className="py-2 pr-3 text-right font-medium">Máximo</th>
                  <th className="py-2 pr-3 text-right font-medium">Rango</th>
                  <th className="py-2 pr-3 text-right font-medium">Desv. Est.</th>
                  <th className="py-2 pr-3 text-right font-medium">Q1</th>
                  <th className="py-2 text-right font-medium">Q3</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {numericProfiles.map((p) => {
                  const s = p.numericStats!;
                  return (
                    <tr key={p.name} className="hover:bg-slate-50">
                      <td className="py-2 pr-3 font-mono font-semibold text-slate-900">{p.name}</td>
                      <td className="py-2 pr-3 text-right font-mono">{s.count}</td>
                      <td className="py-2 pr-3 text-right font-mono">{p.missingCount}</td>
                      <td className="py-2 pr-3 text-right font-mono">{formatNumberES(s.mean, 2)}</td>
                      <td className="py-2 pr-3 text-right font-mono">{formatNumberES(s.median, 2)}</td>
                      <td className="py-2 pr-3 text-right font-mono">{formatNumberES(s.min, 2)}</td>
                      <td className="py-2 pr-3 text-right font-mono">{formatNumberES(s.max, 2)}</td>
                      <td className="py-2 pr-3 text-right font-mono">{formatNumberES(s.range, 2)}</td>
                      <td className="py-2 pr-3 text-right font-mono">{formatNumberES(s.stdDev, 2)}</td>
                      <td className="py-2 pr-3 text-right font-mono">{formatNumberES(s.q1, 2)}</td>
                      <td className="py-2 text-right font-mono">{formatNumberES(s.q3, 2)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* Matriz de correlación */}
      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <div className="border-b border-slate-200 pb-3">
          <h2 className="text-base font-semibold text-slate-900">
            Matriz de Correlación de Pearson
          </h2>
          <p className="text-xs text-slate-600">
            Valores cercanos a +1 o -1 entre predictores advierten posible colinealidad.
          </p>
        </div>
        <div className="mt-4">
          <CorrelationHeatmap correlation={dataset.correlation} />
        </div>
      </section>
    </div>
  );
};
