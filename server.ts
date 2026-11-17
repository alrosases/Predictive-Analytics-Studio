import express from 'express';
import type { Request, Response } from 'express';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

app.use(express.json({ limit: '1mb' }));

interface AssistantRequestBody {
  question?: string;
  contextSummary?: {
    datasetName?: string;
    rowCount?: number;
    columnCount?: number;
    targetColumn?: string;
    positiveClass?: string;
    negativeClass?: string;
    features?: string[];
    threshold?: number;
    balanceMethod?: string;
    trainRatio?: number;
    equationText?: string;
    metricsSummary?: string;
    coefficientsSummary?: string;
  };
}

app.post('/api/assistant', async (req: Request, res: Response) => {
  try {
    const body = (req.body || {}) as AssistantRequestBody;
    const question = (body.question || '').trim();

    if (!question) {
      res.status(400).json({ error: 'Debe escribir una pregunta para el asistente.' });
      return;
    }

    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
      res.status(503).json({
        error: 'El servicio de IA no está configurado con una clave activa en el entorno.',
      });
      return;
    }

    const ai = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });

    const ctx = body.contextSummary || {};
    const contextBlock = [
      ctx.datasetName ? `Base activa: ${ctx.datasetName} (${ctx.rowCount ?? '?'} registros, ${ctx.columnCount ?? '?'} variables).` : 'No hay base de datos cargada todavía.',
      ctx.targetColumn ? `Variable dependiente: ${ctx.targetColumn} (Positiva=1: ${ctx.positiveClass}, Negativa=0: ${ctx.negativeClass}).` : '',
      ctx.features?.length ? `Variables predictoras: ${ctx.features.join(', ')}.` : '',
      ctx.threshold !== undefined ? `Punto de corte activo: ${(ctx.threshold * 100).toFixed(0)}%.` : '',
      ctx.balanceMethod ? `Balanceo en entrenamiento: ${ctx.balanceMethod}.` : '',
      ctx.equationText ? `Ecuación estimada: ${ctx.equationText}` : '',
      ctx.coefficientsSummary ? `Coeficientes estimados:\n${ctx.coefficientsSummary}` : '',
      ctx.metricsSummary ? `Métricas reales del proyecto:\n${ctx.metricsSummary}` : '',
    ]
      .filter(Boolean)
      .join('\n');

    const systemInstruction = [
      'Eres el Consultor Analítico de Predictive Analytics Studio, diseñado para apoyar a analistas y equipos de negocio en analítica de datos, estadística aplicada y Machine Learning.',
      'REGLAS ESTRICTAS:',
      '1. Responde siempre en español claro, técnico, ejecutivo y preciso.',
      '2. Utiliza exclusivamente los resultados estadísticos reales proporcionados en el contexto del proyecto. NUNCA inventes cifras, coeficientes ni métricas que no estén en el contexto.',
      '3. Nunca asumas causalidad a partir de asociaciones estadísticas en modelos observacionales.',
      '4. Explica paso a paso cómo se relacionan Z, la función sigmoide P(Y=1), el punto de corte, la matriz de confusión y las métricas (Accuracy, Precision, Recall, F1-Score).',
      '5. Si una métrica no está definida por división por cero o si aún no hay modelo entrenado, indícalo con claridad y explica cómo obtenerla.',
    ].join('\n');

    const prompt = `Contexto estadístico agregado del proyecto activo (sin filas individuales por privacidad):\n${contextBlock}\n\nConsulta del analista:\n${question}`;

    const response = await ai.models.generateContent({
      model: 'gemini-4-hydrogen',
      contents: prompt,
      config: {
        systemInstruction,
      },
    });

    const answer = response.text?.trim();
    if (!answer) {
      res.status(502).json({ error: 'No se recibió respuesta del asistente.' });
      return;
    }

    res.json({ answer });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Error al consultar el asistente.';
    res.status(500).json({ error: message });
  }
});

async function startServer() {
  const isProd = process.env.NODE_ENV === 'production';

  if (!isProd) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req: Request, res: Response) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Predictive Analytics Studio listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
