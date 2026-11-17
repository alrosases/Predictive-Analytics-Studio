import type { DataRow } from '../types/analytics';

export interface PracticeDatasetDefinition {
  id: 'financierosatisfaccion' | 'cartera' | 'services2';
  fileName: string;
  title: string;
  description: string;
  targetHint: string;
  rows: DataRow[];
}

function seededRandom(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function generateFinancieroSatisfaccion(): DataRow[] {
  const rand = seededRandom(202601);
  const rows: DataRow[] = [];
  for (let i = 0; i < 160; i++) {
    const edad = Math.round(21 + rand() * 44);
    const saldo = Math.round((350 + rand() * 4650) * 100) / 100;
    const zTrue = -2.35 + 0.032 * edad + 0.00058 * saldo;
    const p = 1 / (1 + Math.exp(-zTrue));
    const satisfecho = rand() < p ? 1 : 0;
    rows.push({
      edad,
      saldo,
      satisfecho,
    });
  }
  return rows;
}

function generateCarteraIPS(): DataRow[] {
  const rand = seededRandom(202602);
  const tipologias = ['Hospital Público', 'Clínica Privada', 'Centro Ambulatorio', 'IPS Especializada'];
  const rows: DataRow[] = [];
  for (let i = 1; i <= 220; i++) {
    const tipologia_ips = tipologias[Math.floor(rand() * tipologias.length)];
    const igual_dueno = rand() < 0.62 ? 1 : 0;
    const anos_direccion_actual = Math.round(1 + rand() * 18);
    const antiguedad_cliente = Math.round(1 + rand() * 15);
    const ventas_mensuales = Math.round((18 + rand() * 240) * 10) / 10; // millones COP
    const ratioBase = 0.22 + rand() * 0.85 + (tipologia_ips === 'Hospital Público' ? 0.18 : 0);
    const pasivos = Math.round(ventas_mensuales * ratioBase * 10) / 10;
    const relacion_pasivos_ventas = Math.round((pasivos / ventas_mensuales) * 1000) / 1000;
    const insumos_tipo_a = Math.round((ventas_mensuales * (0.3 + rand() * 0.4)) * 10) / 10;
    const insumos_tipo_b = Math.round((ventas_mensuales * (0.15 + rand() * 0.3)) * 10) / 10;

    const tipEffect =
      tipologia_ips === 'Hospital Público'
        ? 0.85
        : tipologia_ips === 'Centro Ambulatorio'
        ? 0.25
        : tipologia_ips === 'Clínica Privada'
        ? -0.2
        : -0.35;

    const zTrue =
      -1.45 +
      1.85 * relacion_pasivos_ventas -
      0.09 * antiguedad_cliente -
      0.04 * anos_direccion_actual -
      0.45 * igual_dueno +
      0.004 * ventas_mensuales +
      tipEffect;

    const p = 1 / (1 + Math.exp(-zTrue));
    const retrasos = rand() < p ? 1 : 0;

    rows.push({
      id_ips: `IPS-${String(i).padStart(3, '0')}`,
      tipologia_ips,
      igual_dueno,
      anos_direccion_actual,
      antiguedad_cliente,
      ventas_mensuales,
      pasivos,
      relacion_pasivos_ventas,
      insumos_tipo_a,
      insumos_tipo_b,
      retrasos,
    });
  }
  return rows;
}

function generateServices2(): DataRow[] {
  const rand = seededRandom(202603);
  const segmentos = ['Corporativo', 'Pyme', 'Profesional'];
  const rows: DataRow[] = [];
  for (let i = 1; i <= 180; i++) {
    const segmento = segmentos[Math.floor(rand() * segmentos.length)];
    const antiguedad_meses = Math.round(2 + rand() * 58);
    const uso_mensual_gb = Math.round((15 + rand() * 185) * 10) / 10;
    const tickets_soporte = Math.floor(rand() * 7);
    const cargo_mensual = Math.round((45 + rand() * 145) * 100) / 100;
    const contrato_anual = rand() < 0.55 ? 1 : 0;

    const zTrue =
      -0.85 -
      0.042 * antiguedad_meses +
      0.52 * tickets_soporte +
      0.015 * cargo_mensual -
      1.15 * contrato_anual -
      0.006 * uso_mensual_gb +
      (segmento === 'Pyme' ? 0.3 : 0);

    const p = 1 / (1 + Math.exp(-zTrue));
    const abandono = rand() < p ? 1 : 0;

    rows.push({
      id_cliente: `SRV-${String(i).padStart(3, '0')}`,
      segmento,
      antiguedad_meses,
      uso_mensual_gb,
      tickets_soporte,
      cargo_mensual,
      contrato_anual,
      abandono,
    });
  }
  return rows;
}

export const PRACTICE_DATASETS: PracticeDatasetDefinition[] = [
  {
    id: 'financierosatisfaccion',
    fileName: 'financierosatisfaccion_demo.xlsx',
    title: 'Satisfacción Financiera (Referencia)',
    description:
      'Dataset de demostración con edad, saldo y satisfecho (0/1) para modelar probabilidad de satisfacción mediante regresión logística.',
    targetHint: 'satisfecho',
    rows: generateFinancieroSatisfaccion(),
  },
  {
    id: 'cartera',
    fileName: 'cartera_demo.csv',
    title: 'Cartera de IPS – Insumos Médicos (Referencia)',
    description:
      'Caso empresarial de cartera B2B con tipología de IPS, ventas mensuales, pasivos, antigüedad, insumos tipo A/B y retrasos (0/1).',
    targetHint: 'retrasos',
    rows: generateCarteraIPS(),
  },
  {
    id: 'services2',
    fileName: 'services2_demo.xlsx',
    title: 'Servicios y Retención – services2 (Referencia)',
    description:
      'Dataset multisegmento de demostración con variables continuas, discretas, categóricas y binarias para segmentación y retención.',
    targetHint: 'abandono',
    rows: generateServices2(),
  },
];
