import React, { useMemo, useState } from 'react';
import { ArrowUpDown, Search } from 'lucide-react';
import type { DataRow, DatasetState, ModuleId, VariableKind } from '../../types/analytics';
import { formatNumberES, tryParseNumeric } from '../../engine/statistics';

interface ExploreDataViewProps {
  dataset: DatasetState | null;
  onOverrideVariableKind: (column: string, kind: VariableKind) => void;
  onNavigate: (mod: ModuleId) => void;
}

export const ExploreDataView: React.FC<ExploreDataViewProps> = ({
  dataset,
  onOverrideVariableKind,
  onNavigate,
}) => {
  const [search, setSearch] = useState('');
  const [sortCol, setSortCol] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');
  const [page, setPage] = useState(0);
  const pageSize = 25;

  const filteredRows = useMemo(() => {
    if (!dataset) return [];
    const q = search.trim().toLowerCase();
    let items: { __idx: number; row: DataRow }[] = dataset.workingRows.map((row, i) => ({
      __idx: i + 1,
      row,
    }));
    if (q) {
      items = items.filter((item) =>
        dataset.columns.some((c) =>
          String(item.row[c] ?? '')
            .toLowerCase()
            .includes(q)
        )
      );
    }
    if (sortCol) {
      items = [...items].sort((a, b) => {
        const va = a.row[sortCol];
        const vb = b.row[sortCol];
        if (va === null || va === undefined) return 1;
        if (vb === null || vb === undefined) return -1;
        const na = tryParseNumeric(va);
        const nb = tryParseNumeric(vb);
        if (na !== null && nb !== null) {
          return sortDir === 'asc' ? na - nb : nb - na;
        }
        return sortDir === 'asc'
          ? String(va).localeCompare(String(vb), 'es')
          : String(vb).localeCompare(String(va), 'es');
      });
    }
    return items;
  }, [dataset, search, sortCol, sortDir]);

  // Detectar si aplica vista exploratoria del caso Cartera IPS
  const ipsInsights = useMemo(() => {
    if (!dataset) return null;
    const colsLower = dataset.columns.map((c) => c.toLowerCase());
    const hasRetrasos = dataset.columns.find((c) => c.toLowerCase().includes('retraso'));
    const hasTipologia = dataset.columns.find((c) => c.toLowerCase().includes('tipologia'));
    if (!hasRetrasos && !colsLower.some((c) => c.includes('pasivo'))) return null;

    const total = dataset.workingRows.length;
    const delayedRows = hasRetrasos
      ? dataset.workingRows.filter((r) => String(r[hasRetrasos] ?? '').trim() === '1' || String(r[hasRetrasos] ?? '').toLowerCase() === 'si')
      : [];

    const byTipologia: { tipo: string; total: number; retrasos: number; tasa: number }[] = [];
    if (hasTipologia && hasRetrasos) {
      const map = new Map<string, { total: number; retrasos: number }>();
      for (const r of dataset.workingRows) {
        const key = String(r[hasTipologia] ?? 'Sin dato');
        const cur = map.get(key) || { total: 0, retrasos: 0 };
        cur.total++;
        if (String(r[hasRetrasos] ?? '').trim() === '1') cur.retrasos++;
        map.set(key, cur);
      }
      for (const [tipo, v] of map.entries()) {
        byTipologia.push({
          tipo,
          total: v.total,
          retrasos: v.retrasos,
          tasa: v.total > 0 ? (v.retrasos / v.total) * 100 : 0,
        });
      }
      byTipologia.sort((a, b) => b.tasa - a.tasa);
    }

    return {
      targetCol: hasRetrasos,
      total,
      delayedCount: delayedRows.length,
      delayedRate: total > 0 ? (delayedRows.length / total) * 100 : 0,
      byTipologia,
    };
  }, [dataset]);

  if (!dataset) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-10 text-center">
        <h2 className="text-lg font-semibold text-slate-900">No hay base de datos activa</h2>
        <p className="mt-1 text-sm text-slate-600">
          Cargue un archivo CSV o Excel para explorar sus registros, filtrar observaciones y revisar tipos de variables.
        </p>
        <button
          type="button"
          onClick={() => onNavigate('cargar')}
          className="mt-5 rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800"
        >
          Ir a Cargar Bases de Datos
        </button>
      </div>
    );
  }

  const totalPages = Math.max(1, Math.ceil(filteredRows.length / pageSize));
  const safePage = Math.min(page, totalPages - 1);
  const pageRows = filteredRows.slice(safePage * pageSize, (safePage + 1) * pageSize);

  const toggleSort = (col: string) => {
    if (sortCol === col) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortCol(col);
      setSortDir('asc');
    }
    setPage(0);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-900">Exploración de Datos</h1>
          <p className="text-sm text-slate-600">
            Inspeccione los registros, ajuste la clasificación de variables y filtre valores en tiempo real.
          </p>
        </div>
        <button
          type="button"
          onClick={() => onNavigate('descriptivo')}
          className="rounded-lg bg-teal-600 px-4 py-2 text-sm font-medium text-white hover:bg-teal-700 whitespace-nowrap shrink-0"
        >
          Ver Análisis Descriptivo →
        </button>
      </div>

      {/* Exploración rápida Caso Cartera IPS si aplica */}
      {ipsInsights && (
        <section className="rounded-xl border border-slate-200 bg-white p-5">
          <div className="flex flex-col gap-1 border-b border-slate-200 pb-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h2 className="text-sm font-semibold text-slate-900">
                Contexto Detectado: Cartera de IPS (Insumos Médicos)
              </h2>
              <p className="text-xs text-slate-600">
                Resumen exploratorio real sobre la variable objetivo{' '}
                <span className="font-mono font-semibold">{ipsInsights.targetCol || 'retrasos'}</span>
              </p>
            </div>
            <span className="font-mono text-xs font-semibold text-teal-700 tabular-nums">
              Tasa de retraso observada: {formatNumberES(ipsInsights.delayedRate, 1)}% ({ipsInsights.delayedCount} de {ipsInsights.total})
            </span>
          </div>

          {ipsInsights.byTipologia.length > 0 && (
            <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4 tabular-nums">
              {ipsInsights.byTipologia.map((item) => (
                <div key={item.tipo} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <p className="truncate text-xs font-medium text-slate-700" title={item.tipo}>
                    {item.tipo}
                  </p>
                  <p className="mt-1 font-mono text-lg font-bold text-slate-900">
                    {formatNumberES(item.tasa, 1)}%
                  </p>
                  <p className="text-[11px] text-slate-500">
                    {item.retrasos} retrasos de {item.total} IPS
                  </p>
                </div>
              ))}
            </div>
          )}
        </section>
      )}

      {/* Control de tipos de variables */}
      <section className="rounded-xl border border-slate-200 bg-white p-5">
        <h2 className="text-sm font-semibold text-slate-900">
          Clasificación de Variables (Editable)
        </h2>
        <p className="mt-0.5 text-xs text-slate-600">
          Si una variable numérica representa categorías (por ejemplo, un código 1, 2, 3), puede cambiarla a categórica aquí o en Preparación de Datos.
        </p>

        <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {dataset.profiles.map((p) => (
            <div
              key={p.name}
              className="flex items-center justify-between gap-3 rounded-lg border border-slate-200 bg-slate-50/70 px-3.5 py-2.5"
            >
              <div className="min-w-0">
                <p className="truncate font-mono text-xs font-semibold text-slate-900" title={p.name}>
                  {p.name}
                </p>
                <p className="text-[11px] text-slate-500 tabular-nums">
                  {p.uniqueCount} únicos · {p.missingCount} faltantes
                </p>
              </div>
              <select
                value={p.effectiveKind}
                onChange={(e) => onOverrideVariableKind(p.name, e.target.value as VariableKind)}
                className="rounded-md border border-slate-300 bg-white px-2 py-1 text-xs text-slate-800"
              >
                <option value="numeric_continuous">Num. continua</option>
                <option value="numeric_discrete">Num. discreta</option>
                <option value="binary">Binaria</option>
                <option value="categorical">Categórica</option>
              </select>
            </div>
          ))}
        </div>
      </section>

      {/* Tabla interactiva con filtro y ordenamiento */}
      <section className="rounded-xl border border-slate-200 bg-white p-5">
        <div className="flex flex-col gap-3 border-b border-slate-200 pb-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="relative w-full sm:max-w-sm">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setPage(0);
              }}
              placeholder="Buscar en cualquier columna..."
              className="w-full rounded-lg border border-slate-300 bg-white py-2 pl-9 pr-3 text-xs text-slate-900 placeholder:text-slate-400 focus:border-teal-600 focus:outline-none"
            />
          </div>
          <div className="text-xs text-slate-600 tabular-nums">
            Mostrando {filteredRows.length} de {dataset.workingRows.length} registros
          </div>
        </div>

        <div className="mt-3 overflow-x-auto">
          <table className="w-full border-collapse text-left text-xs tabular-nums">
            <thead>
              <tr className="border-b border-slate-200 text-slate-600">
                <th className="py-2.5 pr-3 font-mono font-medium text-slate-400">#</th>
                {dataset.columns.map((col) => (
                  <th key={col} className="py-2.5 pr-4 font-medium whitespace-nowrap">
                    <button
                      type="button"
                      onClick={() => toggleSort(col)}
                      className="inline-flex items-center gap-1 font-mono text-slate-800 hover:text-teal-700"
                    >
                      <span>{col}</span>
                      <ArrowUpDown className="h-3 w-3 text-slate-400" />
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {pageRows.map((item) => (
                <tr key={item.__idx} className="hover:bg-slate-50">
                  <td className="py-2 pr-3 font-mono text-slate-400">{item.__idx}</td>
                  {dataset.columns.map((col) => (
                    <td key={col} className="py-2 pr-4 font-mono text-slate-800 whitespace-nowrap">
                      {item.row[col] === null || item.row[col] === undefined ? (
                        <span className="text-rose-500">NA</span>
                      ) : (
                        String(item.row[col])
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Paginación */}
        <div className="mt-4 flex items-center justify-between border-t border-slate-200 pt-3 text-xs text-slate-600 tabular-nums">
          <span>
            Página {safePage + 1} de {totalPages}
          </span>
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={safePage === 0}
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-40"
            >
              Anterior
            </button>
            <button
              type="button"
              disabled={safePage >= totalPages - 1}
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-40"
            >
              Siguiente
            </button>
          </div>
        </div>
      </section>
    </div>
  );
};
