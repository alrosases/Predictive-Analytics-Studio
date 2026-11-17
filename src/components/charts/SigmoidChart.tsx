import React, { useState } from 'react';
import type { PredictionRecord } from '../../types/analytics';
import { sigmoid } from '../../engine/logisticRegression';

interface SigmoidChartProps {
  threshold: number;
  records?: PredictionRecord[];
  highlightZ?: number;
  positiveLabel: string;
  negativeLabel: string;
}

export const SigmoidChart: React.FC<SigmoidChartProps> = ({
  threshold,
  records = [],
  highlightZ,
  positiveLabel,
  negativeLabel,
}) => {
  const [hoverZ, setHoverZ] = useState<number | null>(null);

  const W = 720;
  const H = 310;
  const padL = 56;
  const padR = 24;
  const padT = 24;
  const padB = 42;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const zMin = -6;
  const zMax = 6;
  const toX = (z: number) =>
    padL + ((Math.max(zMin, Math.min(zMax, z)) - zMin) / (zMax - zMin)) * plotW;
  const toY = (p: number) => padT + (1 - p) * plotH;

  // Threshold z* where sigmoid(z*) = threshold => z* = ln(t / (1-t))
  const clampedT = Math.min(0.99, Math.max(0.01, threshold));
  const zCutoff = Math.log(clampedT / (1 - clampedT));
  const xCutoff = toX(zCutoff);
  const yCutoff = toY(threshold);

  // Build SVG path for sigmoid curve
  const curvePoints: string[] = [];
  for (let i = 0; i <= 140; i++) {
    const z = zMin + (i / 140) * (zMax - zMin);
    const p = sigmoid(z);
    curvePoints.push(`${i === 0 ? 'M' : 'L'} ${toX(z).toFixed(1)} ${toY(p).toFixed(1)}`);
  }

  const sampleRecords = records.slice(0, 220);

  return (
    <div className="space-y-2">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="h-72 w-full select-none"
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const relX = ((e.clientX - rect.left) / rect.width) * W;
          if (relX >= padL && relX <= W - padR) {
            const z = zMin + ((relX - padL) / plotW) * (zMax - zMin);
            setHoverZ(z);
          } else {
            setHoverZ(null);
          }
        }}
        onMouseLeave={() => setHoverZ(null)}
      >
        {/* Classification zone backgrounds */}
        <rect
          x={padL}
          y={padT}
          width={Math.max(0, xCutoff - padL)}
          height={plotH}
          fill="#F1F5F9"
          opacity="0.7"
        />
        <rect
          x={xCutoff}
          y={padT}
          width={Math.max(0, W - padR - xCutoff)}
          height={plotH}
          fill="#CCFBF1"
          opacity="0.45"
        />

        {/* Zone captions */}
        <text x={padL + 10} y={padT + 18} className="fill-slate-600 text-[11px] font-medium">
          Zona 0 ({negativeLabel})
        </text>
        <text x={W - padR - 10} y={padT + 18} textAnchor="end" className="fill-teal-800 text-[11px] font-medium">
          Zona 1 ({positiveLabel})
        </text>

        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map((p) => {
          const y = toY(p);
          return (
            <g key={p}>
              <line x1={padL} y1={y} x2={W - padR} y2={y} stroke="#E2E8F0" strokeWidth="1" />
              <text x={padL - 8} y={y + 4} textAnchor="end" className="fill-slate-500 font-mono text-[10px]">
                {(p * 100).toFixed(0)}%
              </text>
            </g>
          );
        })}

        {[-6, -4, -2, 0, 2, 4, 6].map((z) => {
          const x = toX(z);
          return (
            <g key={z}>
              <line x1={x} y1={padT} x2={x} y2={padT + plotH} stroke="#E2E8F0" strokeWidth="1" />
              <text x={x} y={padT + plotH + 18} textAnchor="middle" className="fill-slate-500 font-mono text-[10px]">
                {z}
              </text>
            </g>
          );
        })}

        {/* Threshold dashed lines */}
        <line
          x1={padL}
          y1={yCutoff}
          x2={W - padR}
          y2={yCutoff}
          stroke="#D97706"
          strokeWidth="1.75"
          strokeDasharray="6 4"
        />
        <line
          x1={xCutoff}
          y1={padT}
          x2={xCutoff}
          y2={padT + plotH}
          stroke="#D97706"
          strokeWidth="1.25"
          strokeDasharray="4 4"
        />

        {/* Sigmoid curve */}
        <path d={curvePoints.join(' ')} fill="none" stroke="#0D9488" strokeWidth="3" />

        {/* Real observations */}
        {sampleRecords.map((r, idx) => (
          <circle
            key={idx}
            cx={toX(r.zValue)}
            cy={toY(r.probability)}
            r={3.5}
            fill={r.actual === 1 ? '#0F766E' : '#334155'}
            fillOpacity={0.65}
            stroke="#FFFFFF"
            strokeWidth="0.7"
          />
        ))}

        {/* Highlighted observation (from single prediction or scenario) */}
        {highlightZ !== undefined && Number.isFinite(highlightZ) && (
          <g>
            <circle
              cx={toX(highlightZ)}
              cy={toY(sigmoid(highlightZ))}
              r={6.5}
              fill="#DC2626"
              stroke="#FFFFFF"
              strokeWidth="2"
            />
          </g>
        )}

        {/* Interactive hover crosshair */}
        {hoverZ !== null && (
          <g>
            <line
              x1={toX(hoverZ)}
              y1={padT}
              x2={toX(hoverZ)}
              y2={padT + plotH}
              stroke="#0F172A"
              strokeWidth="1"
              strokeDasharray="3 3"
            />
            <circle cx={toX(hoverZ)} cy={toY(sigmoid(hoverZ))} r={5} fill="#0F172A" />
          </g>
        )}

        <text x={padL + plotW / 2} y={H - 6} textAnchor="middle" className="fill-slate-600 text-[11px]">
          Valor de la combinación lineal Z = β₀ + β₁X₁ + ... + βₙXₙ
        </text>
      </svg>

      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-200 pt-2 text-xs text-slate-600 tabular-nums">
        <div className="flex flex-wrap items-center gap-4">
          <span>
            Corte: <strong className="font-mono text-amber-700">{(threshold * 100).toFixed(0)}%</strong> (Z ={' '}
            <span className="font-mono">{zCutoff.toFixed(2)}</span>)
          </span>
          <span aria-hidden="true">·</span>
          <span>Puntos oscuros: Real 0 ({negativeLabel})</span>
          <span aria-hidden="true">·</span>
          <span>Puntos turquesa: Real 1 ({positiveLabel})</span>
        </div>
        {hoverZ !== null && (
          <span className="font-mono font-medium text-slate-900">
            Z = {hoverZ.toFixed(2)} → P(Y=1) = {(sigmoid(hoverZ) * 100).toFixed(2)}% (Clase{' '}
            {sigmoid(hoverZ) >= threshold ? '1' : '0'})
          </span>
        )}
      </div>
    </div>
  );
};
