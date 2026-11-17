import React, { useMemo, useState } from 'react';
import { AlertTriangle, Check, Plus, RotateCcw, Trash2 } from 'lucide-react';
import type {
  DataRow,
  DatasetState,
  LogisticRegressionConfig,
  ModuleId,
} from '../../types/analytics';
import { buildDatasetState } from '../../engine/statistics';

interface PreparationViewProps {
  dataset: DatasetState | null;
  config: LogisticRegressionConfig;
  onUpdateConfig: (updater: (prev: LogisticRegressionConfig) => LogisticRegressionConfig) => void;
  onRemoveDuplicates: () => void;
  onRestoreOriginal: () => void;
  onDatasetLoaded: (ds: DatasetState) => void;
  onNavigate: (mod: ModuleId) => void;
}

export const PreparationView: React.FC<PreparationViewProps> = ({
  dataset,
  config,
  onUpdateConfig,
  onRemoveDuplicates,
  onRestoreOriginal,
  onDatasetLoaded,
  onNavigate,
}) => {
  const [confirmDup, setConfirmDup] = useState(false);

  // Constructor de DataFrame en memoria
  const [showManualBuilder, setShowManualBuilder] = useState(false);
  const [manualColsInput, setManualColsInput] = useState('edad, saldo, satisfecho');
  const [manualColumns, setManualColumns] = useState<string[]>(['edad', 'saldo', 'satisfecho']);
  const [manualRows, setManualRows] = useState<DataRow[]>([
    { edad: 28, saldo: 850, satisfecho: 0 },
    { edad: 41, saldo: 1000, satisfecho: 1 },
    { edad: 52, saldo: 3200, satisfecho: 1 },
    { edad: 24, saldo: 450, satisfecho: 0 },
    { edad: 36, saldo: 2100, satisfecho: 1 },
    { edad: 47, saldo: 1650, satisfecho: 1 },
    { edad: 22, saldo: 520, satisfecho: 0 },
    { edad: 39, saldo: 1900, satisfecho: 1 },
    { edad: 31, saldo: 780, satisfecho: 0 },
    { edad: 55, saldo: 4100, satisfecho: 1 },
    { edad: 27, saldo: 620, satisfecho: 0 },
    { edad: 44, saldo: 2400, satisfecho: 1 },
  ]);
  const [newRowDraft, setNewRowDraft] = useState<Record<string, string>>({});

  const highCorrelations = useMemo(() => {
    if (!dataset) return [];
    const { columns, matrix } = dataset.correlation;
    const pairs: { a: string; b: string; r: number }[] = [];
    for (let i = 0; i < columns.length; i++) {
      for (let j = i + 1; j < columns.length; j++) {
        if (Math.abs(matrix[i][j]) >= 0.75) {
          pairs.push({ a: columns[i], b: columns[j], r: matrix[i][j] });
        }
      }
    }
    return pairs;
  }, [dataset]);

  const targetProfile = dataset?.profiles.find((p) => p.name === config.targetColumn);
  const targetCategories = targetProfile?.frequencies.map((f) => f.value) || [];
  const isTargetBinaryValid = targetCategories.length === 2;

  const applyManualSchema = () => {
    const cols = manualColsInput
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);
    if (cols.length < 2) return;
    setManualColumns(cols);
    setManualRows([]);
    setNewRowDraft({});
  };

  const addManualRow = () => {
    const row: DataRow = {};
    for (const c of manualColumns) {
      const raw = (newRowDraft[c] ?? '').trim();
      if (raw === '') {
        row[c] = null;
      } else {
        const num = Number(raw);
        row[c] = !Number.isNaN(num) && Number.isFinite(num) ? num : raw;
      }
    }
    setManualRows((prev) => [...prev, row]);
    setNewRowDraft({});
  };

  const commitManualDataFrame = () => {
    if (manualRows.length === 0 || manualColumns.length < 2) return;
    const ds = buildDatasetState('dataframe_manual.csv', 'manual', manualColumns, manualRows);
    onDatasetLoaded(ds);
    setShowManualBuilder(false);
  };

  const toggleFeature = (col: string) => {
    onUpdateConfig((prev) => {
      const exists = prev.featureColumns.includes(col);
      return {
        ...prev,
        featureColumns: exists
          ? prev.featureColumns.filter((c) => c !== col)
          : [...prev.featureColumns, col],
      };
    });
  };

  const toggleFactor = (col: string) => {
    onUpdateConfig((prev) => {
      const exists = prev.prepConfig.factorColumns.includes(col);
      return {
        ...prev,
        prepConfig: {
          ...prev.prepConfig,
          factorColumns: exists
            ? prev.prepConfig.factorColumns.filter((c) => c !== col)
            : [...prev.prepConfig.factorColumns, col],
        },
      };
    });
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-900">Preparación de Datos</h1>
          <p className="text-sm text-slate-600">
            Configure la variable dependiente binaria (0/1), variables predictoras, manejo de faltantes, duplicados, factores y valores atípicos.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 shrink-0">
          <button
            type="button"
            onClick={() => setShowManualBuilder((v) => !v)}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3.5 py-2 text-xs font-medium text-slate-800 hover:bg-slate-50 whitespace-nowrap"
          >
            <Plus className="h-3.5 w-3.5" />
            {showManualBuilder ? 'Ocultar Creador DataFrame' : 'Crear DataFrame Manual'}
          </button>
          {dataset && (
            <button
              type="button"
              onClick={() => onNavigate('regresion')}
              className="rounded-lg bg-teal-600 px-4 py-2 text-xs font-medium text-white hover:bg-teal-700 whitespace-nowrap"
            >
              Ir a Regresión Logística →
            </button>
          )}
        </div>
      </div>

      {/* Creador de DataFrame manual dentro de la aplicación */}
      {showManualBuilder && (
        <section className="rounded-xl border border-teal-200 bg-white p-6">
          <div className="border-b border-slate-200 pb-3">
            <h2 className="text-base font-semibold text-slate-900">
              Constructor de DataFrame en Memoria
            </h2>
            <p className="text-xs text-slate-600">
              Defina columnas separadas por coma e ingrese registros directamente en la plataforma para análisis rápidos o validación de escenarios.
            </p>
          </div>

          <div className="mt-4 flex flex-col gap-2 sm:flex-row">
            <input
              type="text"
              value={manualColsInput}
              onChange={(e) => setManualColsInput(e.target.value)}
              placeholder="Ej: edad, saldo, satisfecho"
              className="flex-1 rounded-lg border border-slate-300 px-3 py-2 font-mono text-xs text-slate-900"
            />
            <button
              type="button"
              onClick={applyManualSchema}
              className="rounded-lg border border-slate-300 bg-slate-50 px-3.5 py-2 text-xs font-medium text-slate-800 hover:bg-slate-100 whitespace-nowrap"
            >
              Definir Columnas
            </button>
          </div>

          <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-4">
            {manualColumns.map((c) => (
              <input
                key={c}
                type="text"
                value={newRowDraft[c] ?? ''}
                onChange={(e) => setNewRowDraft((prev) => ({ ...prev, [c]: e.target.value }))}
                placeholder={`Valor para ${c}`}
                className="rounded-lg border border-slate-300 px-3 py-2 font-mono text-xs text-slate-900"
              />
            ))}
            <button
              type="button"
              onClick={addManualRow}
              className="rounded-lg bg-slate-900 px-3.5 py-2 text-xs font-medium text-white hover:bg-slate-800"
            >
              Agregar Fila
            </button>
          </div>

          <div className="mt-4 max-h-48 overflow-auto rounded-lg border border-slate-200">
            <table className="w-full border-collapse text-left text-xs tabular-nums">
              <thead className="bg-slate-50">
                <tr className="border-b border-slate-200 text-slate-600">
                  <th className="p-2 font-mono">#</th>
                  {manualColumns.map((c) => (
                    <th key={c} className="p-2 font-mono">{c}</th>
                  ))}
                  <th className="p-2 text-right">Acción</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {manualRows.map((r, idx) => (
                  <tr key={idx}>
                    <td className="p-2 font-mono text-slate-400">{idx + 1}</td>
                    {manualColumns.map((c) => (
                      <td key={c} className="p-2 font-mono text-slate-800">
                        {r[c] === null ? 'NA' : String(r[c])}
                      </td>
                    ))}
                    <td className="p-2 text-right">
                      <button
                        type="button"
                        onClick={() => setManualRows((prev) => prev.filter((_, i) => i !== idx))}
                        className="text-rose-600 hover:text-rose-800"
                        aria-label="Eliminar fila"
                      >
                        <Trash2 className="inline h-3.5 w-3.5" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-4 flex items-center justify-between">
            <span className="text-xs text-slate-500 tabular-nums">
              {manualRows.length} registros ingresados (mínimo 12 recomendados para entrenamiento/prueba)
            </span>
            <button
              type="button"
              disabled={manualRows.length < 4}
              onClick={commitManualDataFrame}
              className="rounded-lg bg-teal-600 px-4 py-2 text-xs font-medium text-white hover:bg-teal-700 disabled:opacity-50"
            >
              Activar DataFrame como Base del Proyecto
            </button>
          </div>
        </section>
      )}

      {!dataset ? (
        <div className="rounded-xl border border-slate-200 bg-white p-10 text-center">
          <h2 className="text-base font-semibold text-slate-900">No hay base cargada</h2>
          <p className="mt-1 text-sm text-slate-600">
            Cargue un archivo o utilice el botón &quot;Crear DataFrame Manual&quot; arriba para ingresar sus propios registros.
          </p>
          <button
            type="button"
            onClick={() => onNavigate('cargar')}
            className="mt-4 rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white"
          >
            Ir a Cargar Bases de Datos
          </button>
        </div>
      ) : (
        <>
          {/* Validación de variable dependiente y codificación 0/1 */}
          <section className="rounded-xl border border-slate-200 bg-white p-6">
            <h2 className="text-base font-semibold text-slate-900">
              1. Variable Dependiente (Objetivo) y Codificación Binaria
            </h2>
            <p className="mt-0.5 text-xs text-slate-600">
              Para regresión logística binaria, la variable objetivo debe contar con dos categorías claramente asignadas a 0 y 1.
            </p>

            <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-3">
              <div>
                <label className="block text-xs font-medium text-slate-700">
                  Variable dependiente (Y)
                </label>
                <select
                  value={config.targetColumn}
                  onChange={(e) => {
                    const col = e.target.value;
                    const prof = dataset.profiles.find((p) => p.name === col);
                    const cats = prof?.frequencies.map((f) => f.value) || [];
                    const sorted = [...cats].sort();
                    onUpdateConfig((prev) => ({
                      ...prev,
                      targetColumn: col,
                      negativeClass: sorted[0] || '',
                      positiveClass: sorted[1] || sorted[0] || '',
                      featureColumns: prev.featureColumns.filter((f) => f !== col),
                    }));
                  }}
                  className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 font-mono text-xs text-slate-900"
                >
                  <option value="">Seleccione variable...</option>
                  {dataset.columns.map((c) => {
                    const p = dataset.profiles.find((x) => x.name === c);
                    return (
                      <option key={c} value={c}>
                        {c} ({p?.uniqueCount ?? 0} categorías)
                      </option>
                    );
                  })}
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700">
                  Evento Negativo (Codificado como 0)
                </label>
                <select
                  value={config.negativeClass}
                  onChange={(e) =>
                    onUpdateConfig((prev) => ({ ...prev, negativeClass: e.target.value }))
                  }
                  disabled={targetCategories.length < 2}
                  className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 font-mono text-xs text-slate-900 disabled:bg-slate-100"
                >
                  {targetCategories.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700">
                  Evento Positivo (Codificado como 1)
                </label>
                <select
                  value={config.positiveClass}
                  onChange={(e) =>
                    onUpdateConfig((prev) => ({ ...prev, positiveClass: e.target.value }))
                  }
                  disabled={targetCategories.length < 2}
                  className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 font-mono text-xs text-slate-900 disabled:bg-slate-100"
                >
                  {targetCategories.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {config.targetColumn && (
              <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-3.5 text-xs">
                {!isTargetBinaryValid ? (
                  <p className="text-rose-700">
                    <strong>Validación:</strong> La variable &quot;{config.targetColumn}&quot; tiene{' '}
                    {targetCategories.length} valores distintos. La regresión logística binaria requiere exactamente 2 categorías (o definir cuáles dos se compararán).
                  </p>
                ) : config.positiveClass === config.negativeClass ? (
                  <p className="text-rose-700">
                    La categoría positiva (1) y la negativa (0) no pueden ser iguales.
                  </p>
                ) : (
                  <div className="flex flex-wrap items-center gap-3 text-slate-800">
                    <Check className="h-4 w-4 text-teal-600" />
                    <span>
                      <strong>Codificación aplicada:</strong> 0 = &quot;{config.negativeClass}&quot; · 1 = &quot;{config.positiveClass}&quot;
                    </span>
                  </div>
                )}
              </div>
            )}
          </section>

          {/* Selección de predictores y factores */}
          <section className="rounded-xl border border-slate-200 bg-white p-6">
            <h2 className="text-base font-semibold text-slate-900">
              2. Selección de Variables Predictoras (X) y Conversión a Factor
            </h2>
            <p className="mt-0.5 text-xs text-slate-600">
              Seleccione las variables independientes del modelo. Las variables categóricas o convertidas a factor se codifican automáticamente mediante variables dummy (k - 1 niveles).
            </p>

            {highCorrelations.length > 0 && (
              <div className="mt-3 flex items-start gap-2.5 rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-900">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
                <div>
                  <span className="font-semibold">Alerta de alta correlación (posible multicolinealidad): </span>
                  {highCorrelations
                    .map((p) => `${p.a} y ${p.b} (r = ${p.r.toFixed(2)})`)
                    .join(' · ')}
                  . Evite incluir predictores altamente redundantes simultáneamente.
                </div>
              </div>
            )}

            <div className="mt-4 overflow-x-auto">
              <table className="w-full border-collapse text-left text-xs tabular-nums">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500">
                    <th className="py-2 pr-4 font-medium">Incluir (X)</th>
                    <th className="py-2 pr-4 font-medium">Variable</th>
                    <th className="py-2 pr-4 font-medium">Tipo</th>
                    <th className="py-2 pr-4 text-right font-medium">Faltantes</th>
                    <th className="py-2 pr-4 text-right font-medium">Atípicos IQR</th>
                    <th className="py-2 font-medium">Tratar como Factor</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {dataset.profiles
                    .filter((p) => p.name !== config.targetColumn)
                    .map((p) => {
                      const included = config.featureColumns.includes(p.name);
                      const isFactor = config.prepConfig.factorColumns.includes(p.name);
                      const highCard = p.effectiveKind === 'categorical' && p.uniqueCount > 15;
                      return (
                        <tr key={p.name} className="hover:bg-slate-50">
                          <td className="py-2.5 pr-4">
                            <input
                              type="checkbox"
                              checked={included}
                              disabled={highCard}
                              onChange={() => toggleFeature(p.name)}
                              className="h-4 w-4 rounded border-slate-300 text-teal-600 focus:ring-teal-600"
                            />
                          </td>
                          <td className="py-2.5 pr-4 font-mono font-semibold text-slate-900">
                            {p.name}
                            {highCard && (
                              <span className="ml-2 font-sans text-[11px] font-normal text-slate-400">
                                ({p.uniqueCount} niveles, posible ID)
                              </span>
                            )}
                          </td>
                          <td className="py-2.5 pr-4 text-slate-700">{p.effectiveKind}</td>
                          <td className="py-2.5 pr-4 text-right font-mono text-slate-700">
                            {p.missingCount}
                          </td>
                          <td className="py-2.5 pr-4 text-right font-mono text-slate-700">
                            {p.numericStats ? p.numericStats.outlierCount : '—'}
                          </td>
                          <td className="py-2.5">
                            {p.effectiveKind === 'numeric_discrete' ||
                            p.effectiveKind === 'numeric_continuous' ? (
                              <label className="inline-flex items-center gap-1.5 text-slate-700">
                                <input
                                  type="checkbox"
                                  checked={isFactor}
                                  onChange={() => toggleFactor(p.name)}
                                  className="h-3.5 w-3.5 rounded border-slate-300 text-teal-600"
                                />
                                <span>Convertir a factor</span>
                              </label>
                            ) : (
                              <span className="text-slate-400">Automático</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>
          </section>

          {/* Limpieza de duplicados, faltantes y atípicos */}
          <section className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <div className="rounded-xl border border-slate-200 bg-white p-6 space-y-4">
              <h2 className="text-base font-semibold text-slate-900">
                3. Tratamiento de Faltantes, Atípicos y Escala
              </h2>
              <p className="text-xs text-slate-600">
                Estas transformaciones se ajustan estrictamente sobre los datos de entrenamiento para evitar fuga de información.
              </p>

              <div className="space-y-3 text-xs">
                <div>
                  <label className="block font-medium text-slate-700">
                    Valores faltantes (NA)
                  </label>
                  <select
                    value={config.prepConfig.missingStrategy}
                    onChange={(e) =>
                      onUpdateConfig((prev) => ({
                        ...prev,
                        prepConfig: {
                          ...prev.prepConfig,
                          missingStrategy: e.target.value as typeof prev.prepConfig.missingStrategy,
                        },
                      }))
                    }
                    className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900"
                  >
                    <option value="drop_rows">Eliminar filas con valores faltantes (recomendado si son pocos)</option>
                    <option value="impute_median_mode">Imputar mediana (numéricas) y moda (categóricas)</option>
                    <option value="impute_mean_mode">Imputar media (numéricas) y moda (categóricas)</option>
                  </select>
                </div>

                <div>
                  <label className="block font-medium text-slate-700">
                    Valores atípicos (criterio 1.5 × IQR)
                  </label>
                  <select
                    value={config.prepConfig.outlierStrategy}
                    onChange={(e) =>
                      onUpdateConfig((prev) => ({
                        ...prev,
                        prepConfig: {
                          ...prev.prepConfig,
                          outlierStrategy: e.target.value as typeof prev.prepConfig.outlierStrategy,
                        },
                      }))
                    }
                    className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900"
                  >
                    <option value="keep">Conservar valores originales</option>
                    <option value="cap_iqr">Acotar extremos a los límites Q1 - 1.5×IQR y Q3 + 1.5×IQR</option>
                  </select>
                </div>

                <label className="flex items-center gap-2 pt-1 text-slate-800">
                  <input
                    type="checkbox"
                    checked={config.prepConfig.standardizeNumeric}
                    onChange={(e) =>
                      onUpdateConfig((prev) => ({
                        ...prev,
                        prepConfig: {
                          ...prev.prepConfig,
                          standardizeNumeric: e.target.checked,
                        },
                      }))
                    }
                    className="h-4 w-4 rounded border-slate-300 text-teal-600"
                  />
                  <span>Estandarizar predictores numéricos (Z-score; desactivado conserva escala R glm())</span>
                </label>
              </div>
            </div>

            <div className="rounded-xl border border-slate-200 bg-white p-6 space-y-4">
              <h2 className="text-base font-semibold text-slate-900">
                4. Control de Registros Duplicados
              </h2>
              <p className="text-xs text-slate-600">
                La base activa contiene <strong className="font-mono text-slate-900">{dataset.duplicateCount}</strong> registros duplicados exactos (de {dataset.workingRows.length} actuales; original: {dataset.originalRows.length}).
              </p>

              <div className="flex flex-wrap items-center gap-3 pt-2">
                {dataset.duplicateCount > 0 && !confirmDup && (
                  <button
                    type="button"
                    onClick={() => setConfirmDup(true)}
                    className="rounded-lg border border-amber-400 bg-amber-50 px-3.5 py-2 text-xs font-medium text-amber-900 hover:bg-amber-100"
                  >
                    Solicitar eliminación de {dataset.duplicateCount} duplicados
                  </button>
                )}

                {confirmDup && (
                  <div className="flex flex-wrap items-center gap-2 rounded-lg border border-rose-200 bg-rose-50 p-3 text-xs">
                    <span className="text-rose-900">¿Autoriza eliminar los registros duplicados de la copia de trabajo?</span>
                    <button
                      type="button"
                      onClick={() => {
                        onRemoveDuplicates();
                        setConfirmDup(false);
                      }}
                      className="rounded-md bg-rose-600 px-3 py-1.5 font-medium text-white hover:bg-rose-700"
                    >
                      Confirmar eliminación
                    </button>
                    <button
                      type="button"
                      onClick={() => setConfirmDup(false)}
                      className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-slate-700"
                    >
                      Cancelar
                    </button>
                  </div>
                )}

                {dataset.workingRows.length !== dataset.originalRows.length && (
                  <button
                    type="button"
                    onClick={onRestoreOriginal}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3.5 py-2 text-xs font-medium text-slate-700 hover:bg-slate-50"
                  >
                    <RotateCcw className="h-3.5 w-3.5" />
                    Restaurar copia original ({dataset.originalRows.length} filas)
                  </button>
                )}
              </div>
            </div>
          </section>
        </>
      )}
    </div>
  );
};
