import React, { useMemo, useState } from 'react';
import type { ColumnProfile, CorrelationMatrix, DataRow } from '../../types/analytics';
import { formatNumberES, tryParseNumeric } from '../../engine/statistics';

interface HistogramProps {
  rows: DataRow[];
  profile: ColumnProfile;
}

export const HistogramChart: React.FC<HistogramProps> = ({ rows, profile }) => {
  const [hoveredBin, setHoveredBin] = useState<number | null>(null);

  const bins = useMemo(() => {
    const vals: number[] = [];
    for (const r of rows) {
      const n = tryParseNumeric(r[profile.name]);
      if (n !== null) vals.push(n);
    }
    if (vals.length === 0) return [];
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    if (min === max) {
      return [{ x0: min, x1: max, count: vals.length }];
    }
    const numBins = Math.min(14, Math.max(5, Math.ceil(Math.sqrt(vals.length))));
    const width = (max - min) / numBins;
    const out = Array.from({ length: numBins }, (_, i) => ({
      x0: min + i * width,
      x1: min + (i + 1) * width,
      count: 0,
    }));
    for (const v of vals) {
      let idx = Math.floor((v - min) / width);
      if (idx >= numBins) idx = numBins - 1;
      if (idx < 0) idx = 0;
      out[idx].count++;
    }
    return out;
  }, [rows, profile.name]);

  if (bins.length === 0) {
    return (
      <div className="flex h-52 items-center justify-center text-xs text-slate-500">
        Sin valores numéricos suficientes para graficar.
      </div>
    );
  }

  const maxCount = Math.max(1, ...bins.map((b) => b.count));
  const W = 560;
  const H = 220;
  const padL = 44;
  const padR = 16;
  const padT = 18;
  const padB = 34;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const barW = plotW / bins.length;

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="h-56 w-full select-none">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const y = padT + (1 - t) * plotH;
          const val = Math.round(maxCount * t);
          return (
            <g key={t}>
              <line x1={padL} y1={y} x2={W - padR} y2={y} stroke="#E2E8F0" strokeWidth="1" />
              <text
                x={padL - 8}
                y={y + 4}
                textAnchor="end"
                className="fill-slate-500 font-mono text-[10px]"
              >
                {val}
              </text>
            </g>
          );
        })}

        {bins.map((b, i) => {
          const h = (b.count / maxCount) * plotH;
          const x = padL + i * barW + 1.5;
          const y = padT + plotH - h;
          const isHover = hoveredBin === i;
          return (
            <g
              key={i}
              onMouseEnter={() => setHoveredBin(i)}
              onMouseLeave={() => setHoveredBin(null)}
              className="cursor-pointer"
            >
              <rect
                x={x}
                y={y}
                width={Math.max(2, barW - 3)}
                height={Math.max(1, h)}
                rx="2"
                fill={isHover ? '#0F172A' : '#0D9488'}
                opacity={isHover ? 1 : 0.85}
              />
            </g>
          );
        })}

        <text x={padL} y={H - 8} className="fill-slate-500 font-mono text-[10px]">
          {formatNumberES(bins[0].x0, 1)}
        </text>
        <text x={W - padR} y={H - 8} textAnchor="end" className="fill-slate-500 font-mono text-[10px]">
          {formatNumberES(bins[bins.length - 1].x1, 1)}
        </text>
      </svg>

      <div className="mt-1 flex min-h-5 items-center justify-between text-xs text-slate-600 tabular-nums">
        {hoveredBin !== null && bins[hoveredBin] ? (
          <span className="font-mono">
            Intervalo [{formatNumberES(bins[hoveredBin].x0, 2)},{' '}
            {formatNumberES(bins[hoveredBin].x1, 2)}]:{' '}
            <strong className="text-slate-900">{bins[hoveredBin].count} observaciones</strong>
          </span>
        ) : (
          <span className="text-slate-400">Pase el cursor sobre las barras para inspeccionar intervalos</span>
        )}
      </div>
    </div>
  );
};

interface BoxplotProps {
  profile: ColumnProfile;
}

export const BoxplotChart: React.FC<BoxplotProps> = ({ profile }) => {
  const s = profile.numericStats;
  if (!s) {
    return (
      <div className="flex h-44 items-center justify-center text-xs text-slate-500">
        El diagrama de caja aplica únicamente a variables numéricas.
      </div>
    );
  }

  const W = 540;
  const H = 150;
  const padL = 36;
  const padR = 36;
  const span = Math.max(1e-9, s.max - s.min);
  const toX = (v: number) => padL + ((v - s.min) / span) * (W - padL - padR);

  const wLow = Math.max(s.min, s.lowerFence);
  const wHigh = Math.min(s.max, s.upperFence);

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="h-36 w-full select-none">
        <line x1={padL} y1={110} x2={W - padR} y2={110} stroke="#CBD5E1" strokeWidth="1" />
        {/* Whiskers */}
        <line x1={toX(wLow)} y1={62} x2={toX(s.q1)} y2={62} stroke="#334155" strokeWidth="2" />
        <line x1={toX(s.q3)} y1={62} x2={toX(wHigh)} y2={62} stroke="#334155" strokeWidth="2" />
        <line x1={toX(wLow)} y1={48} x2={toX(wLow)} y2={76} stroke="#334155" strokeWidth="2" />
        <line x1={toX(wHigh)} y1={48} x2={toX(wHigh)} y2={76} stroke="#334155" strokeWidth="2" />

        {/* IQR Box */}
        <rect
          x={toX(s.q1)}
          y={38}
          width={Math.max(4, toX(s.q3) - toX(s.q1))}
          height={48}
          fill="#0D9488"
          fillOpacity="0.18"
          stroke="#0D9488"
          strokeWidth="2"
          rx="2"
        />

        {/* Median line */}
        <line x1={toX(s.median)} y1={38} x2={toX(s.median)} y2={86} stroke="#0F172A" strokeWidth="2.5" />

        {/* Mean marker */}
        <circle cx={toX(s.mean)} cy={62} r={4} fill="#D97706" />

        {/* Labels */}
        <text x={toX(s.min)} y={128} textAnchor="middle" className="fill-slate-500 font-mono text-[10px]">
          Mín {formatNumberES(s.min, 1)}
        </text>
        <text x={toX(s.q1)} y={26} textAnchor="middle" className="fill-slate-600 font-mono text-[10px]">
          Q1: {formatNumberES(s.q1, 1)}
        </text>
        <text x={toX(s.median)} y={128} textAnchor="middle" className="fill-slate-900 font-mono text-[10px] font-semibold">
          Med: {formatNumberES(s.median, 1)}
        </text>
        <text x={toX(s.q3)} y={26} textAnchor="middle" className="fill-slate-600 font-mono text-[10px]">
          Q3: {formatNumberES(s.q3, 1)}
        </text>
        <text x={toX(s.max)} y={128} textAnchor="middle" className="fill-slate-500 font-mono text-[10px]">
          Máx {formatNumberES(s.max, 1)}
        </text>
      </svg>
      <div className="flex flex-wrap items-center gap-4 text-xs text-slate-600 tabular-nums">
        <span>Línea oscura: Mediana ({formatNumberES(s.median, 2)})</span>
        <span aria-hidden="true">·</span>
        <span>Punto ámbar: Media ({formatNumberES(s.mean, 2)})</span>
        <span aria-hidden="true">·</span>
        <span>IQR: {formatNumberES(s.iqr, 2)}</span>
        <span aria-hidden="true">·</span>
        <span>Atípicos IQR: {s.outlierCount}</span>
      </div>
    </div>
  );
};

interface BarCategoryProps {
  profile: ColumnProfile;
}

export const CategoricalBarChart: React.FC<BarCategoryProps> = ({ profile }) => {
  const topFreqs = profile.frequencies.slice(0, 10);
  if (topFreqs.length === 0) {
    return <div className="py-8 text-center text-xs text-slate-500">Sin categorías válidas.</div>;
  }
  const maxPct = Math.max(1, ...topFreqs.map((f) => f.percentage));

  return (
    <div className="space-y-2.5">
      {topFreqs.map((f) => (
        <div key={f.value} className="space-y-1">
          <div className="flex items-center justify-between text-xs">
            <span className="truncate font-medium text-slate-800 max-w-[60%]" title={f.value}>
              {f.value}
            </span>
            <span className="font-mono text-slate-600 tabular-nums">
              {f.count} ({formatNumberES(f.percentage, 1)}%)
            </span>
          </div>
          <div className="h-2.5 w-full overflow-hidden rounded-sm bg-slate-100">
            <div
              className="h-full rounded-sm bg-teal-600 transition-all"
              style={{ width: `${(f.percentage / maxPct) * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
};

interface CorrelationHeatmapProps {
  correlation: CorrelationMatrix;
}

export const CorrelationHeatmap: React.FC<CorrelationHeatmapProps> = ({ correlation }) => {
  const { columns, matrix } = correlation;
  if (columns.length < 2) {
    return (
      <div className="py-6 text-sm text-slate-500">
        Se requieren al menos dos variables numéricas para calcular la matriz de correlación de Pearson.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-xs tabular-nums">
        <thead>
          <tr>
            <th className="p-2 text-left font-medium text-slate-600">Variable</th>
            {columns.map((c) => (
              <th key={c} className="p-2 text-right font-mono font-medium text-slate-700 max-w-[110px] truncate" title={c}>
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {columns.map((rowName, i) => (
            <tr key={rowName} className="border-t border-slate-200">
              <td className="p-2 font-mono font-medium text-slate-800 max-w-[140px] truncate" title={rowName}>
                {rowName}
              </td>
              {columns.map((_, j) => {
                const r = matrix[i][j];
                const absR = Math.abs(r);
                const bg =
                  i === j
                    ? 'bg-slate-900 text-white'
                    : r >= 0
                    ? `rgba(13, 148, 136, ${(absR * 0.72).toFixed(2)})`
                    : `rgba(220, 38, 38, ${(absR * 0.65).toFixed(2)})`;
                const textLight = i === j || absR > 0.55;
                return (
                  <td
                    key={j}
                    style={i === j ? undefined : { backgroundColor: bg }}
                    className={`p-2 text-right font-mono ${
                      i === j ? 'bg-slate-900 text-white font-semibold' : textLight ? 'text-white font-medium' : 'text-slate-800'
                    }`}
                    title={`${rowName} vs ${columns[j]}: r = ${r.toFixed(4)}`}
                  >
                    {r.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
