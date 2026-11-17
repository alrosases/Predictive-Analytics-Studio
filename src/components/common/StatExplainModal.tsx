import React from 'react';
import { X } from 'lucide-react';

interface StatExplainModalProps {
  open: boolean;
  title: string;
  formula?: string;
  summary: string;
  details: string[];
  example?: string;
  onClose: () => void;
}

export const StatExplainModal: React.FC<StatExplainModalProps> = ({
  open,
  title,
  formula,
  summary,
  details,
  example,
  onClose,
}) => {
  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/50 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="explain-modal-title"
    >
      <div className="w-full max-w-xl rounded-xl border border-slate-200 bg-white p-6 shadow-xl">
        <div className="flex items-start justify-between gap-4 border-b border-slate-200 pb-4">
          <div>
            <p className="text-xs font-medium text-teal-700">Especificación estadística</p>
            <h3 id="explain-modal-title" className="mt-1 text-lg font-semibold text-slate-900">
              {title}
            </h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-2 text-slate-500 transition-colors hover:bg-slate-100 hover:text-slate-900 focus-visible:outline-2 focus-visible:outline-teal-600"
            aria-label="Cerrar explicación"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="mt-4 space-y-4 text-sm leading-relaxed text-slate-700">
          <p>{summary}</p>

          {formula && (
            <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 font-mono text-xs text-slate-900">
              {formula}
            </div>
          )}

          {details.length > 0 && (
            <ul className="space-y-2">
              {details.map((item, idx) => (
                <li key={idx} className="flex gap-2.5">
                  <span className="font-mono text-xs font-semibold text-teal-700">
                    {String(idx + 1).padStart(2, '0')}.
                  </span>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          )}

          {example && (
            <div className="rounded-lg border border-teal-200 bg-teal-50/50 p-3.5 text-xs text-slate-800">
              <span className="font-semibold text-teal-900">Aplicación operativa: </span>
              {example}
            </div>
          )}
        </div>

        <div className="mt-6 flex justify-end border-t border-slate-200 pt-4">
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg bg-slate-900 px-4 py-2 text-xs font-medium text-white transition-colors hover:bg-slate-800 whitespace-nowrap shrink-0"
          >
            Entendido
          </button>
        </div>
      </div>
    </div>
  );
};
