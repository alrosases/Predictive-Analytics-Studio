import React, { useState } from 'react';
import { CheckCircle2, Download, FileSpreadsheet, FileText } from 'lucide-react';
import type {
  DatasetState,
  ModelResult,
  ModuleId,
  ReportMetadata,
} from '../../types/analytics';
import { generateExecutivePDF } from '../../engine/pdfReport';
import { exportPredictionsCSV, exportResultsExcel } from '../../engine/exporters';

interface ReportPDFViewProps {
  dataset: DatasetState | null;
  activeModel: ModelResult | null;
  reportMeta: ReportMetadata;
  onUpdateReportMeta: (updater: (prev: ReportMetadata) => ReportMetadata) => void;
  onNavigate: (mod: ModuleId) => void;
}

const PDF_SECTIONS = [
  'Portada ejecutiva (Título, fecha, responsable e institución)',
  'Índice automático con numeración exacta de páginas',
  '1. Resumen Ejecutivo (objetivo, problema empresarial y hallazgos reales)',
  '2. Descripción de la Base de Datos (archivo, filas, columnas y tipos)',
  '3. Análisis Descriptivo (tablas estadísticas e interpretaciones)',
  '4. Metodología (preparación, partición Train/Test, balanceo y umbral)',
  '5. Resultados del Modelo (ecuación Z, coeficientes glm, Odds Ratios y curva sigmoide)',
  '6. Matriz de Confusión (entrenamiento y prueba: TP, TN, FP, FN)',
  '7. Métricas de Evaluación (Accuracy, Precision, Recall, F1, Balanced Accuracy y gráfico)',
  '8. Validación del Modelo (comparación Train vs. Test y estabilidad)',
  '9. Resultados Empresariales (segmentos y registros de mayor riesgo)',
  '10. Recomendaciones y Decisiones de Negocio',
  '11. Anexos Técnicos (semilla, parámetros y configuración completa)',
];

export const ReportPDFView: React.FC<ReportPDFViewProps> = ({
  dataset,
  activeModel,
  reportMeta,
  onUpdateReportMeta,
  onNavigate,
}) => {
  const [generatedNotice, setGeneratedNotice] = useState(false);

  const canGenerate = Boolean(dataset && activeModel && !activeModel.error);

  const handleGenerate = () => {
    if (!dataset || !activeModel || activeModel.error) return;
    generateExecutivePDF(dataset, activeModel, reportMeta);
    setGeneratedNotice(true);
    setTimeout(() => setGeneratedNotice(false), 5000);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-900">
            Informe Ejecutivo PDF y Exportación de Resultados
          </h1>
          <p className="text-sm text-slate-600">
            Construye un documento editorial formato A4 en español con todos los cálculos reales del proyecto activo.
          </p>
        </div>
      </div>

      {!canGenerate && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-5 text-sm text-amber-900">
          <p className="font-semibold">Falta estimar el modelo logístico</p>
          <p className="mt-1 text-xs">
            Para generar el informe ejecutivo completo en PDF y exportar tablas/predicciones, primero cargue una base de datos y estime un modelo en el módulo de Regresión Logística.
          </p>
          <button
            type="button"
            onClick={() => onNavigate(dataset ? 'regresion' : 'cargar')}
            className="mt-3 rounded-lg bg-slate-900 px-3.5 py-2 text-xs font-medium text-white"
          >
            {dataset ? 'Ir a Regresión Logística' : 'Ir a Cargar Base de Datos'}
          </button>
        </div>
      )}

      {/* Personalización de portada */}
      <section className="rounded-xl border border-slate-200 bg-white p-6 space-y-4">
        <h2 className="text-base font-semibold text-slate-900">
          Datos de Portada del Informe (Editables)
        </h2>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div>
            <label className="block text-xs font-medium text-slate-700">
              Nombre del Proyecto
            </label>
            <input
              type="text"
              value={reportMeta.projectTitle}
              onChange={(e) =>
                onUpdateReportMeta((prev) => ({ ...prev, projectTitle: e.target.value }))
              }
              className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-xs text-slate-900"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-700">
              Título del Análisis
            </label>
            <input
              type="text"
              value={reportMeta.analysisTitle}
              onChange={(e) =>
                onUpdateReportMeta((prev) => ({ ...prev, analysisTitle: e.target.value }))
              }
              className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-xs text-slate-900"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-700">
              Responsable / Analista
            </label>
            <input
              type="text"
              value={reportMeta.studentName}
              onChange={(e) =>
                onUpdateReportMeta((prev) => ({ ...prev, studentName: e.target.value }))
              }
              placeholder="Escriba su nombre completo"
              className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-xs text-slate-900"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-700">
              Institución / Empresa
            </label>
            <input
              type="text"
              value={reportMeta.institution}
              onChange={(e) =>
                onUpdateReportMeta((prev) => ({ ...prev, institution: e.target.value }))
              }
              placeholder="Empresa / Organización"
              className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-xs text-slate-900"
            />
          </div>
        </div>

        {/* Botón principal de descarga PDF + exportadores Excel/CSV */}
        <div className="flex flex-wrap items-center gap-3 border-t border-slate-200 pt-5">
          <button
            type="button"
            disabled={!canGenerate}
            onClick={handleGenerate}
            className="inline-flex items-center gap-2 rounded-lg bg-teal-600 px-5 py-3 text-sm font-semibold text-white transition-colors hover:bg-teal-700 disabled:cursor-not-allowed disabled:opacity-45"
          >
            <FileText className="h-4 w-4" />
            GENERAR Y DESCARGAR INFORME PDF
          </button>

          <button
            type="button"
            disabled={!canGenerate}
            onClick={() => dataset && activeModel && exportResultsExcel(dataset, activeModel)}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-3 text-xs font-semibold text-slate-800 hover:bg-slate-50 disabled:opacity-45"
          >
            <FileSpreadsheet className="h-4 w-4 text-teal-700" />
            Descargar Tablas en Excel (.xlsx)
          </button>

          <button
            type="button"
            disabled={!canGenerate}
            onClick={() => activeModel && exportPredictionsCSV(activeModel, 'all')}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-3 text-xs font-semibold text-slate-800 hover:bg-slate-50 disabled:opacity-45"
          >
            <Download className="h-4 w-4 text-slate-700" />
            Descargar Predicciones (.csv)
          </button>
        </div>

        {generatedNotice && (
          <div className="flex items-center gap-2 rounded-lg border border-teal-200 bg-teal-50 p-3 text-xs text-teal-900">
            <CheckCircle2 className="h-4 w-4 text-teal-600" />
            <span>Informe ejecutivo PDF generado y descargado correctamente con los datos reales de su proyecto.</span>
          </div>
        )}
      </section>

      {/* Estructura incluida en el PDF */}
      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <h2 className="text-base font-semibold text-slate-900">
          Contenido del Informe Ejecutivo (Formato A4)
        </h2>
        <div className="mt-4 grid grid-cols-1 gap-2.5 md:grid-cols-2">
          {PDF_SECTIONS.map((sec) => (
            <div
              key={sec}
              className="flex items-center gap-2.5 rounded-lg border border-slate-200 bg-slate-50/70 px-3.5 py-2.5 text-xs text-slate-800"
            >
              <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-teal-600" />
              <span>{sec}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
};
