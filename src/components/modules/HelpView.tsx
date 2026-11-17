import React from 'react';
import { BookOpen, Database, Layers } from 'lucide-react';
import type { ColumnProfile, DatasetState, ModelResult } from '../../types/analytics';
import { formatNumberES } from '../../engine/statistics';

interface HelpViewProps {
  dataset: DatasetState | null;
  activeModel: ModelResult | null;
}

const KIND_LABELS: Record<string, string> = {
  numeric_continuous: 'Numérica continua',
  numeric_discrete: 'Numérica discreta',
  categorical: 'Categórica',
  binary: 'Binaria',
};

function describeVariable(
  p: ColumnProfile,
  activeModel: ModelResult | null
): { role: string; summary: string; definition: string } {
  const cfg = activeModel && !activeModel.error ? activeModel.config : null;
  let role = 'Disponible';
  if (cfg && cfg.targetColumn === p.name) {
    role = `Objetivo Y (1="${cfg.positiveClass}", 0="${cfg.negativeClass}")`;
  } else if (cfg && cfg.featureColumns.includes(p.name)) {
    const isFactor = cfg.prepConfig.factorColumns.includes(p.name);
    role = isFactor ? 'Predictor X (Factor)' : 'Predictor X (Activo)';
  }

  let summary = '—';
  if (p.numericStats) {
    summary = `Rango [${formatNumberES(p.numericStats.min, 2)}, ${formatNumberES(
      p.numericStats.max,
      2
    )}] · Media ${formatNumberES(p.numericStats.mean, 2)} · DE ${formatNumberES(
      p.numericStats.stdDev,
      2
    )}`;
  } else if (p.frequencies.length > 0) {
    summary = p.frequencies
      .slice(0, 3)
      .map((f) => `${f.value} (${formatNumberES(f.percentage, 1)}%)`)
      .join(' · ');
  }

  let definition = '';
  if (p.effectiveKind === 'binary') {
    const cats = p.frequencies.map((f) => `"${f.value}"`).join(' y ');
    definition = `Variable dicotómica con 2 niveles observados (${cats || 'N/D'}); apta como variable objetivo binaria o indicador 0/1.`;
  } else if (p.effectiveKind === 'categorical') {
    definition = `Variable cualitativa con ${p.uniqueCount} categorías distintas; se codifica mediante variables dummy (k - 1 niveles) al modelar.`;
  } else if (p.effectiveKind === 'numeric_discrete') {
    definition = `Variable cuantitativa discreta con ${p.uniqueCount} valores únicos; permite análisis numérico directo o conversión a factor.`;
  } else {
    definition = `Variable cuantitativa continua; aporta información de magnitud y dispersión al predictor lineal Z.`;
  }

  if (p.missingCount > 0) {
    definition += ` Registra ${p.missingCount} valores faltantes (${formatNumberES(
      (p.missingCount / Math.max(1, p.totalCount)) * 100,
      1
    )}%).`;
  }

  return { role, summary, definition };
}

export const HelpView: React.FC<HelpViewProps> = ({ dataset, activeModel }) => {
  const hasModel = Boolean(activeModel && !activeModel.error);

  const dynamicExampleZ = hasModel
    ? `En el modelo activo (${activeModel!.name}), Z combina ${activeModel!.config.featureColumns.join(', ')} con el intercepto ${activeModel!.coefficients[0]?.beta.toFixed(3)}.`
    : 'Ejemplo de especificación lineal: Z = β₀ + β₁·edad + β₂·saldo.';

  const dynamicTarget = hasModel
    ? `En la base activa (${dataset?.fileName}), la variable objetivo es "${activeModel!.config.targetColumn}" (1 = "${activeModel!.config.positiveClass}", 0 = "${activeModel!.config.negativeClass}").`
    : 'Ejemplo operativo: satisfecho (1 = Sí, 0 = No) o retrasos (1 = Presenta retraso, 0 = Paga a tiempo).';

  const concepts = [
    {
      q: '01. Regresión Logística Binaria',
      a: 'Modelo estadístico supervisado de clasificación binaria que estima la probabilidad de que una observación pertenezca a una categoría objetivo (1) frente a la categoría de referencia (0), a partir de múltiples predictores.',
    },
    {
      q: '02. Predictor Lineal (Z / Logit)',
      a: `Z = β₀ + β₁X₁ + ... + βₙXₙ representa el logaritmo natural de los odds (ln(P / (1 - P))). ${dynamicExampleZ}`,
    },
    {
      q: '03. Función Sigmoide y Probabilidad',
      a: 'Transforma Z en una probabilidad acotada mediante P(Y=1) = 1 / (1 + e⁻ᶻ). Garantiza resultados coherentes entre 0 y 1 (0% a 100%) para cualquier combinación de variables.',
    },
    {
      q: '04. Punto de Corte (Umbral de Decisión)',
      a: 'Regla operativa de clasificación: si P(Y=1) ≥ umbral se asigna 1; en caso contrario se asigna 0. Su ajuste (ej. 0.35 o 0.70) calibra el equilibrio entre falsos positivos y falsos negativos según el costo del negocio.',
    },
    {
      q: '05. Matriz de Confusión (2×2)',
      a: 'Tabla de contingencia que cruza las categorías reales observadas contra las categorías predichas, desglosando aciertos (TP y TN) y errores de clasificación (FP y FN).',
    },
    {
      q: '06. Verdaderos Positivos (TP) y Falsos Negativos (FN)',
      a: `TP son casos reales positivos detectados correctamente. FN son casos reales positivos que el modelo omitió al clasificarlos como 0. ${dynamicTarget}`,
    },
    {
      q: '07. Accuracy vs. Precision',
      a: 'Accuracy mide la tasa global de aciertos ((TP+TN)/Total). Precision mide la confiabilidad de las alertas positivas emitidas por el modelo (TP/(TP+FP)).',
    },
    {
      q: '08. Recall (Sensibilidad)',
      a: 'Mide la cobertura del modelo sobre los casos reales positivos: Recall = TP / (TP + FN). Un Recall elevado minimiza los falsos negativos.',
    },
    {
      q: '09. F1-Score y Balanced Accuracy',
      a: 'F1-Score es la media armónica entre Precision y Recall: 2 × (Precision × Recall) / (Precision + Recall). Resume el desempeño sobre cada categoría y resulta clave en escenarios con desbalanceo.',
    },
    {
      q: '10. Balanceo de Categorías en Entrenamiento',
      a: 'Técnicas aplicadas exclusivamente al conjunto de entrenamiento (submuestreo, sobremuestreo o SMOTE) para evitar sesgo hacia la categoría mayoritaria. El conjunto de prueba siempre conserva su distribución real.',
    },
    {
      q: '11. Validación Train / Test (Generalización)',
      a: 'Partición independiente entre entrenamiento y prueba para medir el desempeño sobre observaciones no vistas durante la estimación por máxima verosimilitud, previniendo el sobreajuste.',
    },
    {
      q: '12. Coeficientes (β) y Odds Ratios (e^β)',
      a: 'El signo de β indica si el predictor eleva (+) o reduce (-) la probabilidad estimada. El Odds Ratio cuantifica el factor multiplicativo sobre los odds del evento por cada incremento unitario del predictor.',
    },
  ];

  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-slate-200 bg-white p-6">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-teal-50 text-teal-700">
            <BookOpen className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-slate-900">
              Diccionario de Datos y Metodología Analítica
            </h1>
            <p className="text-sm text-slate-600">
              Especificación técnica de variables del proyecto activo, definiciones estadísticas y fundamentos metodológicos de modelado predictivo.
            </p>
          </div>
        </div>
      </div>

      {/* Diccionario Dinámico de Variables del Proyecto Activo */}
      <section className="rounded-xl border border-slate-200 bg-white p-6 space-y-4">
        <div className="flex flex-col gap-2 border-b border-slate-200 pb-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-2">
            <Database className="h-4 w-4 text-teal-700" />
            <h2 className="text-base font-semibold text-slate-900">
              Diccionario de Datos del Proyecto Activo
            </h2>
          </div>
          {dataset && (
            <span className="font-mono text-xs text-slate-500 tabular-nums">
              {dataset.fileName} · {dataset.workingRows.length} registros · {dataset.columns.length} variables
            </span>
          )}
        </div>

        {!dataset ? (
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-5 text-xs text-slate-600">
            Cargue una base de datos en <strong>Cargar bases de datos</strong> para generar automáticamente el diccionario técnico de sus variables (tipos, roles en el modelo, rangos, valores faltantes y descripción estadística).
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left text-xs tabular-nums">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="py-2.5 pr-4 font-medium">Variable</th>
                  <th className="py-2.5 pr-4 font-medium">Tipo Estadístico</th>
                  <th className="py-2.5 pr-4 font-medium">Rol Analítico</th>
                  <th className="py-2.5 pr-4 text-right font-medium">Válidos / NA / Únicos</th>
                  <th className="py-2.5 pr-4 font-medium">Rango o Niveles</th>
                  <th className="py-2.5 font-medium">Especificación Técnica</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {dataset.profiles.map((p) => {
                  const meta = describeVariable(p, activeModel);
                  return (
                    <tr key={p.name} className="align-top hover:bg-slate-50">
                      <td className="py-3 pr-4 font-mono font-semibold text-slate-900 whitespace-nowrap">
                        {p.name}
                      </td>
                      <td className="py-3 pr-4 text-slate-700 whitespace-nowrap">
                        {KIND_LABELS[p.effectiveKind] || p.effectiveKind}
                      </td>
                      <td className="py-3 pr-4 font-medium text-teal-800">
                        {meta.role}
                      </td>
                      <td className="py-3 pr-4 text-right font-mono text-slate-700 whitespace-nowrap">
                        {p.validCount} / {p.missingCount} / {p.uniqueCount}
                      </td>
                      <td className="py-3 pr-4 font-mono text-[11px] text-slate-600">
                        {meta.summary}
                      </td>
                      <td className="py-3 text-slate-600 leading-relaxed">
                        {meta.definition}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Glosario Metodológico y Estadístico */}
      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <Layers className="h-4 w-4 text-teal-700" />
          <h2 className="text-base font-semibold text-slate-900">
            Fundamentos Metodológicos y Glosario Técnico
          </h2>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {concepts.map((c) => (
            <article
              key={c.q}
              className="rounded-xl border border-slate-200 bg-white p-5 space-y-2"
            >
              <h3 className="text-sm font-semibold text-slate-900">{c.q}</h3>
              <p className="text-xs leading-relaxed text-slate-600">{c.a}</p>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
};
