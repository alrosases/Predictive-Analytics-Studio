import React, { useRef, useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  Download,
  FileSpreadsheet,
  Upload,
} from 'lucide-react';
import type { DatasetState, ModuleId } from '../../types/analytics';
import { buildDatasetState, formatNumberES, parseDatasetFile } from '../../engine/statistics';
import { PRACTICE_DATASETS } from '../../engine/practiceDatasets';
import { exportDatasetCopy } from '../../engine/exporters';

interface UploadDataViewProps {
  dataset: DatasetState | null;
  onDatasetLoaded: (ds: DatasetState) => void;
  onNavigate: (mod: ModuleId) => void;
}

const KIND_LABELS: Record<string, string> = {
  numeric_continuous: 'Numérica continua',
  numeric_discrete: 'Numérica discreta',
  categorical: 'Categórica',
  binary: 'Binaria',
};

export const UploadDataView: React.FC<UploadDataViewProps> = ({
  dataset,
  onDatasetLoaded,
  onNavigate,
}) => {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastFile, setLastFile] = useState<File | null>(null);

  const handleFile = async (file: File, sheetName?: string) => {
    setError(null);
    setLoading(true);
    try {
      const parsed = await parseDatasetFile(file, sheetName);
      const state = buildDatasetState(
        file.name,
        parsed.sourceType,
        parsed.columns,
        parsed.rows,
        parsed.sheetNames,
        parsed.selectedSheet
      );
      setLastFile(file);
      onDatasetLoaded(state);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Error al procesar el archivo.');
    } finally {
      setLoading(false);
    }
  };

  const handlePracticeLoad = (id: 'financierosatisfaccion' | 'cartera' | 'services2') => {
    setError(null);
    const found = PRACTICE_DATASETS.find((p) => p.id === id);
    if (!found) return;
    const columns = Object.keys(found.rows[0] || {});
    const state = buildDatasetState(found.fileName, 'practice', columns, found.rows);
    setLastFile(null);
    onDatasetLoaded(state);
  };

  const firstFive = dataset ? dataset.workingRows.slice(0, 5) : [];
  const lastFive = dataset
    ? dataset.workingRows.slice(Math.max(0, dataset.workingRows.length - 5))
    : [];

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-900">Cargar Bases de Datos</h1>
          <p className="text-sm text-slate-600">
            Importe archivos CSV, XLS o XLSX. Los datos se procesan localmente en su navegador sin modificar el archivo original.
          </p>
        </div>
        {dataset && (
          <div className="flex items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={() => exportDatasetCopy(dataset, 'csv')}
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-50 whitespace-nowrap shrink-0"
            >
              <Download className="h-3.5 w-3.5" />
              Copiar CSV
            </button>
            <button
              type="button"
              onClick={() => exportDatasetCopy(dataset, 'xlsx')}
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-50 whitespace-nowrap shrink-0"
            >
              <Download className="h-3.5 w-3.5" />
              Copiar XLSX
            </button>
          </div>
        )}
      </div>

      {/* Zona Drag & Drop */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const file = e.dataTransfer.files?.[0];
          if (file) handleFile(file);
        }}
        className={`rounded-xl border-2 border-dashed p-8 text-center transition-colors ${
          dragging
            ? 'border-teal-600 bg-teal-50/40'
            : 'border-slate-300 bg-white hover:border-slate-400'
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv,.xls,.xlsx"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
            e.target.value = '';
          }}
        />
        <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-slate-100 text-slate-700">
          <Upload className="h-6 w-6" />
        </div>
        <h2 className="mt-3 text-base font-semibold text-slate-900">
          {loading ? 'Procesando archivo y perfilando variables...' : 'Arrastre y suelte su archivo aquí'}
        </h2>
        <p className="mt-1 text-xs text-slate-500">
          Compatible con financierosatisfaccion.xlsx, cartera.csv, services2.xlsx o cualquier base tabular (.csv, .xls, .xlsx)
        </p>
        <div className="mt-4">
          <button
            type="button"
            disabled={loading}
            onClick={() => inputRef.current?.click()}
            className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-slate-800 disabled:opacity-60 whitespace-nowrap shrink-0"
          >
            Seleccionar archivo del computador
          </button>
        </div>
      </div>

      {error && (
        <div className="flex items-start gap-3 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-600" />
          <div>
            <p className="font-semibold">Error al cargar la base de datos</p>
            <p className="mt-0.5 text-xs">{error}</p>
          </div>
        </div>
      )}

      {/* Conjuntos de datos de referencia */}
      <section className="rounded-xl border border-slate-200 bg-white p-5">
        <div className="border-b border-slate-200 pb-3">
          <h2 className="text-sm font-semibold text-slate-900">
            Conjuntos de Datos de Referencia (Demostración)
          </h2>
          <p className="mt-0.5 text-xs text-slate-600">
            Datasets de demostración precargados para validar flujos analíticos de inmediato. Todos los cálculos estadísticos y estimaciones se ejecutan de forma real sobre estos registros.
          </p>
        </div>
        <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-3">
          {PRACTICE_DATASETS.map((p) => (
            <div
              key={p.id}
              className="flex flex-col justify-between rounded-lg border border-slate-200 bg-slate-50/70 p-4"
            >
              <div>
                <div className="flex items-center gap-2 text-xs font-mono text-teal-700">
                  <FileSpreadsheet className="h-3.5 w-3.5" />
                  <span>{p.fileName}</span>
                </div>
                <h3 className="mt-1.5 text-sm font-semibold text-slate-900">{p.title}</h3>
                <p className="mt-1 text-xs leading-relaxed text-slate-600">{p.description}</p>
              </div>
              <button
                type="button"
                onClick={() => handlePracticeLoad(p.id)}
                className="mt-4 w-full rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-800 transition-colors hover:border-teal-600 hover:text-teal-800"
              >
                Cargar dataset de referencia ({p.rows.length} filas)
              </button>
            </div>
          ))}
        </div>
      </section>

      {/* Diagnóstico automático de la base cargada */}
      {dataset && (
        <>
          <section className="rounded-xl border border-slate-200 bg-white p-6">
            <div className="flex flex-col gap-3 border-b border-slate-200 pb-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center gap-2.5">
                <CheckCircle2 className="h-5 w-5 text-teal-600 shrink-0" />
                <div>
                  <h2 className="text-base font-semibold text-slate-900">
                    Resumen del Archivo: {dataset.fileName}
                  </h2>
                  <p className="text-xs text-slate-500">
                    Detección automática de estructura y calidad de datos
                  </p>
                </div>
              </div>

              {dataset.sheetNames && dataset.sheetNames.length > 1 && lastFile && (
                <div className="flex items-center gap-2 text-xs">
                  <span className="text-slate-600">Hoja de Excel:</span>
                  <select
                    value={dataset.selectedSheet}
                    onChange={(e) => handleFile(lastFile, e.target.value)}
                    className="rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-xs text-slate-800"
                  >
                    {dataset.sheetNames.map((s) => (
                      <option key={s} value={s}>
                        {s}
                      </option>
                    ))}
                  </select>
                </div>
              )}
            </div>

            <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4 tabular-nums">
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-3.5">
                <p className="text-xs text-slate-500">Observaciones</p>
                <p className="mt-1 font-mono text-xl font-bold text-slate-900">
                  {dataset.workingRows.length}
                </p>
              </div>
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-3.5">
                <p className="text-xs text-slate-500">Columnas / Variables</p>
                <p className="mt-1 font-mono text-xl font-bold text-slate-900">
                  {dataset.columns.length}
                </p>
              </div>
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-3.5">
                <p className="text-xs text-slate-500">Valores faltantes</p>
                <p className="mt-1 font-mono text-xl font-bold text-slate-900">
                  {dataset.profiles.reduce((a, p) => a + p.missingCount, 0)}
                </p>
              </div>
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-3.5">
                <p className="text-xs text-slate-500">Registros duplicados</p>
                <p className="mt-1 font-mono text-xl font-bold text-slate-900">
                  {dataset.duplicateCount}
                </p>
              </div>
            </div>

            {/* Tabla de variables detectadas */}
            <div className="mt-6 overflow-x-auto">
              <table className="w-full border-collapse text-left text-xs tabular-nums">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500">
                    <th className="py-2.5 pr-4 font-medium">Variable</th>
                    <th className="py-2.5 pr-4 font-medium">Tipo Detectado</th>
                    <th className="py-2.5 pr-4 text-right font-medium">Válidos</th>
                    <th className="py-2.5 pr-4 text-right font-medium">Faltantes</th>
                    <th className="py-2.5 pr-4 text-right font-medium">Únicos</th>
                    <th className="py-2.5 font-medium">Distribución / Valores de muestra</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {dataset.profiles.map((p) => (
                    <tr key={p.name} className="hover:bg-slate-50">
                      <td className="py-2.5 pr-4 font-mono font-semibold text-slate-900">
                        {p.name}
                      </td>
                      <td className="py-2.5 pr-4 text-slate-700">
                        {KIND_LABELS[p.effectiveKind] || p.effectiveKind}
                      </td>
                      <td className="py-2.5 pr-4 text-right font-mono text-slate-700">
                        {p.validCount}
                      </td>
                      <td className="py-2.5 pr-4 text-right font-mono text-slate-700">
                        {p.missingCount}
                      </td>
                      <td className="py-2.5 pr-4 text-right font-mono text-slate-700">
                        {p.uniqueCount}
                      </td>
                      <td className="py-2.5 text-slate-600">
                        {p.numericStats
                          ? `Media: ${formatNumberES(p.numericStats.mean, 2)} · Rango: [${formatNumberES(
                              p.numericStats.min,
                              1
                            )}, ${formatNumberES(p.numericStats.max, 1)}]`
                          : p.frequencies
                              .slice(0, 3)
                              .map((f) => `${f.value} (${formatNumberES(f.percentage, 0)}%)`)
                              .join(' · ')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {/* Cinco primeras y cinco últimas filas */}
          <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
            {[
              { title: 'Cinco primeras filas (head)', rows: firstFive, offset: 0 },
              {
                title: 'Cinco últimas filas (tail)',
                rows: lastFive,
                offset: Math.max(0, dataset.workingRows.length - 5),
              },
            ].map((block) => (
              <section
                key={block.title}
                className="rounded-xl border border-slate-200 bg-white p-5"
              >
                <h3 className="text-sm font-semibold text-slate-900">{block.title}</h3>
                <div className="mt-3 overflow-x-auto">
                  <table className="w-full border-collapse text-left text-xs tabular-nums">
                    <thead>
                      <tr className="border-b border-slate-200 text-slate-500">
                        <th className="py-2 pr-3 font-mono font-medium">#</th>
                        {dataset.columns.map((c) => (
                          <th key={c} className="py-2 pr-3 font-mono font-medium whitespace-nowrap">
                            {c}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {block.rows.map((row, idx) => (
                        <tr key={idx} className="hover:bg-slate-50">
                          <td className="py-2 pr-3 font-mono text-slate-400">
                            {block.offset + idx + 1}
                          </td>
                          {dataset.columns.map((c) => (
                            <td
                              key={c}
                              className="py-2 pr-3 font-mono text-slate-800 whitespace-nowrap"
                            >
                              {row[c] === null || row[c] === undefined ? (
                                <span className="text-rose-500">NA</span>
                              ) : (
                                String(row[c])
                              )}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            ))}
          </div>

          <div className="flex justify-end gap-3">
            <button
              type="button"
              onClick={() => onNavigate('exploracion')}
              className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-800 hover:bg-slate-50"
            >
              Explorar tabla completa
            </button>
            <button
              type="button"
              onClick={() => onNavigate('descriptivo')}
              className="rounded-lg bg-teal-600 px-4 py-2 text-sm font-medium text-white hover:bg-teal-700"
            >
              Continuar a Análisis Descriptivo →
            </button>
          </div>
        </>
      )}
    </div>
  );
};
