import React, { useMemo, useState } from 'react';
import { BookOpen, Building2, MessageSquare, Send } from 'lucide-react';
import type { DatasetState, ModelResult, ModuleId } from '../../types/analytics';
import {
  computeConfusionMatrix,
  computeEvaluationMetrics,
  predictBatchRows,
} from '../../engine/logisticRegression';
import { formatNumberES } from '../../engine/statistics';

interface InterpretationViewProps {
  dataset: DatasetState | null;
  activeModel: ModelResult | null;
  onNavigate: (mod: ModuleId) => void;
}

const PRESET_QUESTIONS = [
  '¿Cómo interpreto los coeficientes y Odds Ratios de mi modelo actual?',
  '¿Por qué el punto de corte actual afecta Precision y Recall?',
  '¿El modelo presenta señales de sobreajuste entre entrenamiento y prueba?',
  '¿Qué significa Z en este modelo?',
];

export const InterpretationView: React.FC<InterpretationViewProps> = ({
  dataset,
  activeModel,
  onNavigate,
}) => {
  const [question, setQuestion] = useState('');
  const [assistantMessages, setAssistantMessages] = useState<
    { role: 'user' | 'assistant'; text: string }[]
  >([]);
  const [loadingAI, setLoadingAI] = useState(false);

  const cmTest =
    activeModel && !activeModel.error
      ? computeConfusionMatrix(activeModel.testPredictions, activeModel.config.threshold)
      : null;
  const mTest = cmTest ? computeEvaluationMetrics(cmTest) : null;

  const cmTrain =
    activeModel && !activeModel.error
      ? computeConfusionMatrix(activeModel.trainPredictions, activeModel.config.threshold)
      : null;
  const mTrain = cmTrain ? computeEvaluationMetrics(cmTrain) : null;

  // Caso Empresarial: Cartera de IPS (análisis real sobre los datos cargados)
  const ipsAnalysis = useMemo(() => {
    if (!dataset || !activeModel || activeModel.error) return null;
    const isCartera =
      dataset.fileName.toLowerCase().includes('cartera') ||
      dataset.columns.some((c) => c.toLowerCase().includes('retraso')) ||
      dataset.columns.some((c) => c.toLowerCase().includes('tipologia'));

    if (!isCartera) return null;

    const scoredAll = predictBatchRows(
      activeModel,
      dataset.workingRows,
      activeModel.config.threshold
    );

    const total = dataset.workingRows.length;
    const positiveObs = scoredAll.filter(
      (r) => r.actual === activeModel.config.positiveClass
    ).length;
    const obsRate = total > 0 ? (positiveObs / total) * 100 : 0;

    // Agrupar riesgo por tipología si existe columna tipologia
    const tipCol = dataset.columns.find((c) => c.toLowerCase().includes('tipologia'));
    const riskByTip: { tipo: string; count: number; avgProb: number; obsRate: number }[] = [];
    if (tipCol) {
      const map = new Map<string, { n: number; probSum: number; posSum: number }>();
      dataset.workingRows.forEach((row, idx) => {
        const key = String(row[tipCol] ?? 'Sin dato');
        const cur = map.get(key) || { n: 0, probSum: 0, posSum: 0 };
        cur.n++;
        cur.probSum += scoredAll[idx].probability;
        if (String(row[activeModel.config.targetColumn] ?? '').trim() === activeModel.config.positiveClass) {
          cur.posSum++;
        }
        map.set(key, cur);
      });
      for (const [tipo, v] of map.entries()) {
        riskByTip.push({
          tipo,
          count: v.n,
          avgProb: (v.probSum / v.n) * 100,
          obsRate: (v.posSum / v.n) * 100,
        });
      }
      riskByTip.sort((a, b) => b.avgProb - a.avgProb);
    }

    const highRiskClients = [...scoredAll]
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 8);

    const idCol = dataset.columns.find((c) => c.toLowerCase().includes('id')) || null;

    return {
      total,
      positiveObs,
      obsRate,
      riskByTip,
      highRiskClients,
      idCol,
      scoredAll,
    };
  }, [dataset, activeModel]);

  const buildLocalDeterministicAnswer = (q: string): string => {
    if (!dataset) {
      return 'Aún no hay una base de datos cargada. Cargue un archivo CSV o Excel para analizar los datos y responder con valores reales de su proyecto.';
    }
    if (!activeModel || activeModel.error) {
      return `Actualmente tiene cargada la base "${dataset.fileName}" (${dataset.workingRows.length} filas), pero todavía no ha estimado un modelo logístico válido. Vaya al módulo "Regresión logística" para estimar los coeficientes por máxima verosimilitud.`;
    }

    const eq = activeModel.coefficients
      .map((c, i) => (i === 0 ? c.beta.toFixed(4) : `${c.beta >= 0 ? ' + ' : ' - '}${Math.abs(c.beta).toFixed(4)}·${c.term}`))
      .join('');

    const qLower = q.toLowerCase();
    if (qLower.includes('coeficiente') || qLower.includes('odds')) {
      const lines = activeModel.coefficients.slice(1).map((c) => {
        const dir = c.beta >= 0 ? 'incrementa' : 'reduce';
        return `• ${c.term}: β = ${c.beta.toFixed(4)}, Odds Ratio = ${c.oddsRatio.toFixed(4)} (${dir} los odds de "${activeModel.config.positiveClass}" manteniendo constantes las demás variables; p = ${c.pValue !== null ? c.pValue.toFixed(4) : 'N/D'}).`;
      });
      return `Interpretación basada en los coeficientes reales de "${activeModel.name}":\n${lines.join('\n')}\n\nRecuerde que en estudios observacionales estas relaciones indican asociación estadística, no causalidad directa.`;
    }

    if (qLower.includes('corte') || qLower.includes('umbral') || qLower.includes('precision') || qLower.includes('recall')) {
      return `Con el punto de corte actual de ${(activeModel.config.threshold * 100).toFixed(0)}% en el conjunto de prueba:\n• Precision(1) = ${mTest?.precision1 !== null && mTest?.precision1 !== undefined ? `${(mTest.precision1 * 100).toFixed(2)}%` : 'No definida'}\n• Recall(1) = ${mTest?.recall1 !== null && mTest?.recall1 !== undefined ? `${(mTest.recall1 * 100).toFixed(2)}%` : 'No definida'}\n• F1-Score(1) = ${mTest?.f1_1 !== null && mTest?.f1_1 !== undefined ? `${(mTest.f1_1 * 100).toFixed(2)}%` : 'No definida'}\nSi baja el umbral, el modelo clasificará más casos como "${activeModel.config.positiveClass}" (aumenta Recall pero suelen subir los Falsos Positivos y bajar Precision). Si lo sube, ocurre lo contrario.`;
    }

    if (qLower.includes('sobreajuste') || qLower.includes('valid') || qLower.includes('entrenamiento')) {
      const accTr = mTrain?.accuracy !== null && mTrain?.accuracy !== undefined ? `${(mTrain.accuracy * 100).toFixed(2)}%` : 'N/D';
      const accTe = mTest?.accuracy !== null && mTest?.accuracy !== undefined ? `${(mTest.accuracy * 100).toFixed(2)}%` : 'N/D';
      return `Comparación real de validación:\n• Entrenamiento (${activeModel.trainCount} filas): Accuracy = ${accTr}, F1(1) = ${mTrain?.f1_1 !== null && mTrain?.f1_1 !== undefined ? `${(mTrain.f1_1 * 100).toFixed(2)}%` : 'N/D'}\n• Prueba (${activeModel.testCount} filas): Accuracy = ${accTe}, F1(1) = ${mTest?.f1_1 !== null && mTest?.f1_1 !== undefined ? `${(mTest.f1_1 * 100).toFixed(2)}%` : 'N/D'}\nUna brecha pequeña entre ambos confirma que el modelo generaliza bien sobre datos no vistos.`;
    }

    return `En su modelo "${activeModel.name}" (${activeModel.config.targetColumn} ~ ${activeModel.config.featureColumns.join(' + ')}):\n• Ecuación logit: Z = ${eq}\n• Probabilidad: P(Y=1) = 1 / (1 + e^(-Z))\n• En el conjunto de prueba (${activeModel.testCount} registros), el Accuracy es ${mTest?.accuracy !== null && mTest?.accuracy !== undefined ? `${(mTest.accuracy * 100).toFixed(2)}%` : 'N/D'} y el F1-Score de la clase positiva es ${mTest?.f1_1 !== null && mTest?.f1_1 !== undefined ? `${(mTest.f1_1 * 100).toFixed(2)}%` : 'N/D'}.`;
  };

  const handleAskAssistant = async (textToAsk: string) => {
    const cleanQ = textToAsk.trim();
    if (!cleanQ) return;

    setAssistantMessages((prev) => [...prev, { role: 'user', text: cleanQ }]);
    setQuestion('');
    setLoadingAI(true);

    try {
      const eqText =
        activeModel && !activeModel.error
          ? activeModel.coefficients
              .map((c, idx) =>
                idx === 0
                  ? c.beta.toFixed(4)
                  : `${c.beta >= 0 ? ' + ' : ' - '}${Math.abs(c.beta).toFixed(4)}*${c.term}`
              )
              .join('')
          : '';

      const coefSummary =
        activeModel && !activeModel.error
          ? activeModel.coefficients
              .map(
                (c) =>
                  `${c.term}: beta=${c.beta.toFixed(4)}, OR=${c.oddsRatio.toFixed(4)}, p=${
                    c.pValue !== null ? c.pValue.toFixed(4) : 'N/D'
                  }`
              )
              .join('\n')
          : '';

      const metricsSummary =
        mTest && mTrain
          ? `Train Acc=${mTrain.accuracy !== null ? (mTrain.accuracy * 100).toFixed(2) + '%' : 'N/D'}, Test Acc=${
              mTest.accuracy !== null ? (mTest.accuracy * 100).toFixed(2) + '%' : 'N/D'
            }, Test Precision(1)=${
              mTest.precision1 !== null ? (mTest.precision1 * 100).toFixed(2) + '%' : 'N/D'
            }, Test Recall(1)=${
              mTest.recall1 !== null ? (mTest.recall1 * 100).toFixed(2) + '%' : 'N/D'
            }, Test F1(1)=${mTest.f1_1 !== null ? (mTest.f1_1 * 100).toFixed(2) + '%' : 'N/D'}`
          : '';

      const resp = await fetch('/api/assistant', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: cleanQ,
          contextSummary: {
            datasetName: dataset?.fileName,
            rowCount: dataset?.workingRows.length,
            columnCount: dataset?.columns.length,
            targetColumn: activeModel?.config.targetColumn,
            positiveClass: activeModel?.config.positiveClass,
            negativeClass: activeModel?.config.negativeClass,
            features: activeModel?.config.featureColumns,
            threshold: activeModel?.config.threshold,
            balanceMethod: activeModel?.config.balanceMethod,
            trainRatio: activeModel?.config.splitConfig.trainRatio,
            equationText: eqText ? `Z = ${eqText}` : undefined,
            coefficientsSummary: coefSummary || undefined,
            metricsSummary: metricsSummary || undefined,
          },
        }),
      });

      if (resp.ok) {
        const data = (await resp.json()) as { answer?: string };
        if (data.answer) {
          setAssistantMessages((prev) => [...prev, { role: 'assistant', text: data.answer! }]);
          return;
        }
      }
      // Graceful fallback to deterministic statistical tutor if server AI is unavailable
      const localAnswer = buildLocalDeterministicAnswer(cleanQ);
      setAssistantMessages((prev) => [...prev, { role: 'assistant', text: localAnswer }]);
    } catch {
      const localAnswer = buildLocalDeterministicAnswer(cleanQ);
      setAssistantMessages((prev) => [...prev, { role: 'assistant', text: localAnswer }]);
    } finally {
      setLoadingAI(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-900">
            Interpretación de Resultados y Caso Empresarial
          </h1>
          <p className="text-sm text-slate-600">
            Traducción estadística y gerencial de coeficientes, Odds Ratios, riesgo por segmento y consultor analítico inteligente.
          </p>
        </div>
        <button
          type="button"
          onClick={() => onNavigate('informe')}
          className="rounded-lg bg-teal-600 px-4 py-2 text-xs font-medium text-white hover:bg-teal-700 whitespace-nowrap shrink-0"
        >
          Generar Informe PDF →
        </button>
      </div>

      {/* Interpretación de coeficientes reales */}
      {activeModel && !activeModel.error ? (
        <section className="rounded-xl border border-slate-200 bg-white p-6 space-y-4">
          <div className="flex items-center gap-2 border-b border-slate-200 pb-3">
            <BookOpen className="h-4 w-4 text-teal-700" />
            <h2 className="text-base font-semibold text-slate-900">
              Interpretación Estadística de Coeficientes y Odds Ratios
            </h2>
          </div>

          <div className="space-y-3 text-xs leading-relaxed text-slate-700">
            {activeModel.coefficients.slice(1).map((c) => {
              const isSig = c.pValue !== null && c.pValue < 0.05;
              const pctChangeOdds = (c.oddsRatio - 1) * 100;
              return (
                <div
                  key={c.term}
                  className="rounded-lg border border-slate-200 bg-slate-50 p-4"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-mono text-sm font-semibold text-slate-900">
                      {c.term}
                    </span>
                    <span className="font-mono text-xs text-slate-600 tabular-nums">
                      β = {c.beta.toFixed(4)} · OR = {c.oddsRatio.toFixed(4)} · p ={' '}
                      {c.pValue !== null ? (c.pValue < 0.0001 ? '<0.0001' : c.pValue.toFixed(4)) : 'N/D'}{' '}
                      ({isSig ? 'Significativa al 5%' : 'No significativa al 5%'})
                    </span>
                  </div>
                  <p className="mt-2">
                    Manteniendo constantes las demás variables del modelo, un incremento unitario en{' '}
                    <strong className="font-mono">{c.term}</strong> se asocia con un cambio del{' '}
                    <strong>
                      {pctChangeOdds >= 0 ? '+' : ''}
                      {formatNumberES(pctChangeOdds, 2)}%
                    </strong>{' '}
                    en los <em>odds</em> de pertenecer a la categoría{' '}
                    <strong>&quot;{activeModel.config.positiveClass}&quot;</strong> (Odds Ratio ={' '}
                    {c.oddsRatio.toFixed(3)}).{' '}
                    {c.beta >= 0
                      ? 'La asociación es positiva: valores mayores elevan la probabilidad estimada.'
                      : 'La asociación es negativa: valores mayores reducen la probabilidad estimada.'}
                  </p>
                </div>
              );
            })}
          </div>
          <p className="text-[11px] text-slate-500">
            Nota metodológica: las relaciones observadas representan asociaciones estadísticas dentro de la muestra analizada y no deben interpretarse como relaciones causales directas.
          </p>
        </section>
      ) : (
        <div className="rounded-xl border border-slate-200 bg-white p-6 text-sm text-slate-600">
          Estime un modelo en <strong>Regresión logística</strong> para visualizar la interpretación detallada de sus coeficientes.
        </div>
      )}

      {/* Caso Empresarial Cartera de IPS (cuando aplica) */}
      {ipsAnalysis && activeModel && (
        <section className="rounded-xl border border-teal-200 bg-white p-6 space-y-5">
          <div className="flex items-center gap-2 border-b border-slate-200 pb-3">
            <Building2 className="h-4 w-4 text-teal-700" />
            <div>
              <h2 className="text-base font-semibold text-slate-900">
                Caso Empresarial: Análisis de Cartera de IPS (Insumos Médicos)
              </h2>
              <p className="text-xs text-slate-600">
                Respuestas fundamentadas en el modelo logístico y los datos reales de {ipsAnalysis.total} instituciones.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 text-xs leading-relaxed text-slate-700">
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 space-y-1">
              <p className="font-semibold text-slate-900">
                1. ¿Qué proporción de clientes presenta retrasos?
              </p>
              <p>
                En la base activa, <strong>{ipsAnalysis.positiveObs}</strong> de{' '}
                <strong>{ipsAnalysis.total}</strong> IPS presentan el evento positivo{' '}
                (&quot;{activeModel.config.positiveClass}&quot;), equivalente al{' '}
                <strong className="font-mono text-teal-700">
                  {formatNumberES(ipsAnalysis.obsRate, 2)}%
                </strong>
                .
              </p>
            </div>

            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 space-y-1">
              <p className="font-semibold text-slate-900">
                2–3. Características y variables asociadas al comportamiento
              </p>
              <p>
                Los predictores con mayor peso estadístico en su modelo actual son:{' '}
                <strong>
                  {activeModel.coefficients
                    .slice(1)
                    .sort((a, b) => (a.pValue ?? 1) - (b.pValue ?? 1))
                    .slice(0, 3)
                    .map((c) => `${c.term} (OR=${c.oddsRatio.toFixed(2)})`)
                    .join(', ')}
                </strong>
                . Estas variables ayudan a perfilar el riesgo sin implicar causalidad directa.
              </p>
            </div>

            {ipsAnalysis.riskByTip.length > 0 && (
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 space-y-2 md:col-span-2">
                <p className="font-semibold text-slate-900">
                  5. ¿Qué tipologías de IPS presentan mayor riesgo?
                </p>
                <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4 tabular-nums">
                  {ipsAnalysis.riskByTip.map((t) => (
                    <div key={t.tipo} className="rounded border border-slate-200 bg-white p-2.5">
                      <p className="font-medium text-slate-900 truncate">{t.tipo}</p>
                      <p className="mt-1 font-mono text-teal-700 font-semibold">
                        Prob. media: {formatNumberES(t.avgProb, 1)}%
                      </p>
                      <p className="text-[11px] text-slate-500">
                        Mora observada: {formatNumberES(t.obsRate, 1)}% (n={t.count})
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 space-y-2 md:col-span-2">
              <p className="font-semibold text-slate-900">
                4 y 6. Clientes con mayor probabilidad estimada que requieren seguimiento prioritario
              </p>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse text-left text-xs tabular-nums">
                  <thead>
                    <tr className="border-b border-slate-200 text-slate-500">
                      <th className="py-1.5 pr-3">Fila / ID</th>
                      <th className="py-1.5 pr-3 text-right">Valor Z</th>
                      <th className="py-1.5 pr-3 text-right">Prob. Retraso</th>
                      <th className="py-1.5">Estado real</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200/70">
                    {ipsAnalysis.highRiskClients.map((c) => (
                      <tr key={c.rowIndex}>
                        <td className="py-1.5 pr-3 font-mono font-medium text-slate-900">
                          {ipsAnalysis.idCol
                            ? String(dataset?.workingRows[c.rowIndex - 1]?.[ipsAnalysis.idCol] ?? c.rowIndex)
                            : `Fila ${c.rowIndex}`}
                        </td>
                        <td className="py-1.5 pr-3 text-right font-mono">{c.zValue.toFixed(3)}</td>
                        <td className="py-1.5 pr-3 text-right font-mono font-bold text-rose-700">
                          {(c.probability * 100).toFixed(2)}%
                        </td>
                        <td className="py-1.5 font-mono">{c.actual ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="rounded-lg border border-teal-200 bg-teal-50/50 p-4 space-y-1 md:col-span-2">
              <p className="font-semibold text-teal-900">
                7. Acciones comerciales fundamentadas para reducir el riesgo de cartera
              </p>
              <p>
                Priorizar cobranza preventiva sobre las IPS ubicadas en el segmento de mayor riesgo estimado (probabilidad &ge; {(activeModel.config.threshold * 100).toFixed(0)}%), revisar cupos de crédito y plazos en tipologías con mayor incidencia de retraso, y monitorear periódicamente la relación pasivos/ventas antes de aprobar despachos de alto volumen.
              </p>
            </div>
          </div>
        </section>
      )}

      {/* Consultor Analítico Inteligente */}
      <section className="rounded-xl border border-slate-200 bg-white p-6 space-y-4">
        <div className="flex items-center gap-2 border-b border-slate-200 pb-3">
          <MessageSquare className="h-4 w-4 text-teal-700" />
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Consultor Analítico Inteligente
            </h2>
            <p className="text-xs text-slate-600">
              Resuelve dudas conceptuales e interpreta los resultados reales del modelo activo sin enviar filas confidenciales de sus datos.
            </p>
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          {PRESET_QUESTIONS.map((pq) => (
            <button
              key={pq}
              type="button"
              onClick={() => handleAskAssistant(pq)}
              className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs text-slate-700 hover:border-teal-600 hover:text-teal-800"
            >
              {pq}
            </button>
          ))}
        </div>

        {assistantMessages.length > 0 && (
          <div className="max-h-80 space-y-3 overflow-y-auto rounded-lg border border-slate-200 bg-slate-50 p-4">
            {assistantMessages.map((msg, i) => (
              <div
                key={i}
                className={`rounded-lg p-3 text-xs leading-relaxed whitespace-pre-line ${
                  msg.role === 'user'
                    ? 'ml-8 bg-slate-900 text-white'
                    : 'mr-8 border border-slate-200 bg-white text-slate-800'
                }`}
              >
                <span className="mb-1 block text-[10px] font-semibold opacity-70">
                  {msg.role === 'user' ? 'Analista' : 'Consultor Analítico'}
                </span>
                {msg.text}
              </div>
            ))}
            {loadingAI && (
              <div className="mr-8 rounded-lg border border-slate-200 bg-white p-3 text-xs text-slate-500">
                Analizando contexto estadístico...
              </div>
            )}
          </div>
        )}

        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleAskAssistant(question);
          }}
          className="flex gap-2"
        >
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Escriba su pregunta sobre el modelo, Z, Odds Ratios, matriz de confusión o métricas..."
            className="flex-1 rounded-lg border border-slate-300 px-3.5 py-2 text-xs text-slate-900 placeholder:text-slate-400 focus:border-teal-600 focus:outline-none"
          />
          <button
            type="submit"
            disabled={loadingAI || !question.trim()}
            className="inline-flex items-center gap-1.5 rounded-lg bg-slate-900 px-4 py-2 text-xs font-medium text-white hover:bg-slate-800 disabled:opacity-50 whitespace-nowrap"
          >
            <Send className="h-3.5 w-3.5" />
            Consultar
          </button>
        </form>
      </section>
    </div>
  );
};
