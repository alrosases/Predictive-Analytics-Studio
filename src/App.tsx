import React, { useMemo, useState } from 'react';
import {
  Activity,
  BarChart3,
  BookOpen,
  Calculator,
  CheckSquare,
  Database,
  FileSpreadsheet,
  FileText,
  Gauge,
  Grid2x2,
  LayoutDashboard,
  Lightbulb,
  Menu,
  Scale,
  Sliders,
  Sparkles,
  Wrench,
  X,
} from 'lucide-react';
import type {
  BalanceMethod,
  DatasetState,
  LogisticRegressionConfig,
  ModelResult,
  ModuleId,
  ReportMetadata,
  ScenarioItem,
  VariableKind,
} from './types/analytics';
import { profileDataset, removeDuplicateRows } from './engine/statistics';
import { trainLogisticRegression } from './engine/logisticRegression';

import { DashboardView } from './components/modules/DashboardView';
import { UploadDataView } from './components/modules/UploadDataView';
import { ExploreDataView } from './components/modules/ExploreDataView';
import { DescriptiveView } from './components/modules/DescriptiveView';
import { PreparationView } from './components/modules/PreparationView';
import { RegressionView } from './components/modules/RegressionView';
import { PredictionsView } from './components/modules/PredictionsView';
import { ConfusionMatrixView } from './components/modules/ConfusionMatrixView';
import { MetricsView } from './components/modules/MetricsView';
import { BalancingView } from './components/modules/BalancingView';
import { ValidationView } from './components/modules/ValidationView';
import { SimulatorView } from './components/modules/SimulatorView';
import { InterpretationView } from './components/modules/InterpretationView';
import { ReportPDFView } from './components/modules/ReportPDFView';
import { HelpView } from './components/modules/HelpView';

const NAV_ITEMS: { id: ModuleId; label: string; icon: React.FC<{ className?: string }> }[] = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'cargar', label: 'Cargar bases de datos', icon: Database },
  { id: 'exploracion', label: 'Exploración de datos', icon: FileSpreadsheet },
  { id: 'descriptivo', label: 'Análisis descriptivo', icon: BarChart3 },
  { id: 'preparacion', label: 'Preparación de datos', icon: Wrench },
  { id: 'regresion', label: 'Regresión logística', icon: Activity },
  { id: 'predicciones', label: 'Predicciones', icon: Calculator },
  { id: 'matriz', label: 'Matriz de confusión', icon: Grid2x2 },
  { id: 'metricas', label: 'Métricas de evaluación', icon: Gauge },
  { id: 'balanceo', label: 'Balanceo de datos', icon: Scale },
  { id: 'validacion', label: 'Validación del modelo', icon: CheckSquare },
  { id: 'simulador', label: 'Simulador de escenarios', icon: Sliders },
  { id: 'interpretacion', label: 'Interpretación de resultados', icon: Lightbulb },
  { id: 'informe', label: 'Informe ejecutivo PDF', icon: FileText },
  { id: 'ayuda', label: 'Diccionario de datos', icon: BookOpen },
];

const INITIAL_CONFIG: LogisticRegressionConfig = {
  targetColumn: '',
  positiveClass: '',
  negativeClass: '',
  featureColumns: [],
  threshold: 0.5,
  balanceMethod: 'none',
  prepConfig: {
    missingStrategy: 'drop_rows',
    outlierStrategy: 'keep',
    standardizeNumeric: false,
    factorColumns: [],
  },
  splitConfig: {
    trainRatio: 0.75,
    seed: 42,
    stratified: true,
  },
};

export default function App() {
  const [activeModule, setActiveModule] = useState<ModuleId>('dashboard');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const [dataset, setDataset] = useState<DatasetState | null>(null);
  const [variableOverrides, setVariableOverrides] = useState<Record<string, VariableKind>>({});
  const [config, setConfig] = useState<LogisticRegressionConfig>(INITIAL_CONFIG);

  const [savedModels, setSavedModels] = useState<ModelResult[]>([]);
  const [activeModelId, setActiveModelId] = useState<string | null>(null);
  const [scenarios, setScenarios] = useState<ScenarioItem[]>([]);
  const [isComputing, setIsComputing] = useState(false);

  const [reportMeta, setReportMeta] = useState<ReportMetadata>({
    projectTitle: 'Análisis Predictivo mediante Regresión Logística',
    analysisTitle: 'Modelado Binario, Validación y Evaluación Empresarial',
    studentName: '',
    institution: '',
  });

  const activeModel = useMemo(() => {
    if (!activeModelId) return savedModels[0] || null;
    return savedModels.find((m) => m.id === activeModelId) || savedModels[0] || null;
  }, [savedModels, activeModelId]);

  const handleDatasetLoaded = (newDs: DatasetState) => {
    setDataset(newDs);
    setVariableOverrides({});
    setSavedModels([]);
    setActiveModelId(null);
    setScenarios([]);

    // Detectar automáticamente variable binaria candidata (priorizando satisfecho, retrasos, abandono)
    const priorityNames = ['satisfecho', 'retrasos', 'retraso', 'abandono', 'target', 'y'];
    const binaryCols = newDs.profiles.filter((p) => p.uniqueCount === 2);
    const preferredTarget =
      binaryCols.find((p) =>
        priorityNames.some((kw) => p.name.toLowerCase().includes(kw))
      ) || binaryCols[0];

    const targetCol = preferredTarget?.name || '';
    const cats = [...(preferredTarget?.frequencies.map((f) => f.value) || [])].sort();
    const negClass = cats.includes('0') ? '0' : cats[0] || '';
    const posClass = cats.includes('1') ? '1' : cats[1] || cats[0] || '';

    const defaultFeatures = newDs.profiles
      .filter(
        (p) =>
          p.name !== targetCol &&
          !p.name.toLowerCase().startsWith('id') &&
          !(p.effectiveKind === 'categorical' && p.uniqueCount > 15)
      )
      .slice(0, 6)
      .map((p) => p.name);

    setConfig({
      ...INITIAL_CONFIG,
      targetColumn: targetCol,
      negativeClass: negClass,
      positiveClass: posClass,
      featureColumns: defaultFeatures,
    });
  };

  const handleOverrideVariableKind = (column: string, kind: VariableKind) => {
    if (!dataset) return;
    const nextOverrides = { ...variableOverrides, [column]: kind };
    setVariableOverrides(nextOverrides);
    const { profiles, correlation, duplicateCount } = profileDataset(
      dataset.workingRows,
      dataset.columns,
      nextOverrides
    );
    setDataset({
      ...dataset,
      profiles,
      correlation,
      duplicateCount,
    });
  };

  const handleRemoveDuplicates = () => {
    if (!dataset) return;
    const uniqueRows = removeDuplicateRows(dataset.workingRows, dataset.columns);
    const { profiles, correlation, duplicateCount } = profileDataset(
      uniqueRows,
      dataset.columns,
      variableOverrides
    );
    setDataset({
      ...dataset,
      workingRows: uniqueRows,
      profiles,
      correlation,
      duplicateCount,
    });
  };

  const handleRestoreOriginal = () => {
    if (!dataset) return;
    const restored = dataset.originalRows.map((r) => ({ ...r }));
    const { profiles, correlation, duplicateCount } = profileDataset(
      restored,
      dataset.columns,
      variableOverrides
    );
    setDataset({
      ...dataset,
      workingRows: restored,
      profiles,
      correlation,
      duplicateCount,
    });
  };

  const runTraining = (modelName: string, overrideConfig?: LogisticRegressionConfig) => {
    if (!dataset) return;
    const cfgToUse = overrideConfig || config;
    setIsComputing(true);
    setTimeout(() => {
      const result = trainLogisticRegression(
        dataset.workingRows,
        dataset.profiles,
        cfgToUse,
        modelName
      );
      setSavedModels((prev) => {
        if (result.error) {
          return [result, ...prev.filter((m) => !m.error)];
        }
        const filtered = prev.filter((m) => !m.error && m.name !== result.name);
        return [result, ...filtered];
      });
      setActiveModelId(result.id);
      setIsComputing(false);
    }, 20);
  };

  const handleThresholdChange = (newThreshold: number) => {
    setConfig((prev) => ({ ...prev, threshold: newThreshold }));
    setSavedModels((prev) =>
      prev.map((m) =>
        m.id === activeModel?.id
          ? {
              ...m,
              config: { ...m.config, threshold: newThreshold },
            }
          : m
      )
    );
  };

  const handleRetrainWithBalance = (method: BalanceMethod) => {
    const nextCfg: LogisticRegressionConfig = { ...config, balanceMethod: method };
    setConfig(nextCfg);
    runTraining(
      activeModel && !activeModel.error ? activeModel.name : 'Modelo Logístico',
      nextCfg
    );
  };

  const currentNav = NAV_ITEMS.find((n) => n.id === activeModule) || NAV_ITEMS[0];

  const selectModule = (id: ModuleId) => {
    setActiveModule(id);
    setMobileMenuOpen(false);
  };

  return (
    <div className="flex min-h-screen bg-slate-50 text-slate-900">
      {/* Sidebar Desktop (260px) */}
      <aside className="hidden w-64 shrink-0 flex-col border-r border-slate-800 bg-slate-900 text-slate-100 lg:flex">
        <div className="border-b border-slate-800 px-5 py-5">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-teal-600 text-white">
              <Sparkles className="h-4 w-4" />
            </div>
            <div className="min-w-0">
              <p className="truncate font-display text-sm font-bold tracking-tight text-white">
                Predictive Studio
              </p>
              <p className="truncate text-[11px] text-slate-400">
                Regresión Logística y ML
              </p>
            </div>
          </div>
        </div>

        <nav className="flex-1 space-y-0.5 overflow-y-auto px-3 py-4" aria-label="Módulos principales">
          {NAV_ITEMS.map((item, index) => {
            const Icon = item.icon;
            const active = activeModule === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => selectModule(item.id)}
                className={`flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-xs font-medium transition-colors ${
                  active
                    ? 'bg-teal-600 text-white'
                    : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                }`}
              >
                <Icon className={`h-4 w-4 shrink-0 ${active ? 'text-white' : 'text-slate-400'}`} />
                <span className="truncate">
                  <span className="mr-1.5 font-mono text-[10px] opacity-60">
                    {String(index + 1).padStart(2, '0')}
                  </span>
                  {item.label}
                </span>
              </button>
            );
          })}
        </nav>

        <div className="border-t border-slate-800 px-4 py-3 text-[11px] text-slate-400">
          <p className="font-medium text-slate-300">Predictive Analytics Studio</p>
          <p className="truncate">Estadística y Machine Learning</p>
        </div>
      </aside>

      {/* Mobile Drawer Sidebar */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 z-50 flex lg:hidden">
          <div
            className="fixed inset-0 bg-slate-950/60"
            onClick={() => setMobileMenuOpen(false)}
          />
          <aside className="relative z-10 flex w-72 flex-col bg-slate-900 text-slate-100">
            <div className="flex items-center justify-between border-b border-slate-800 px-4 py-4">
              <span className="font-display text-sm font-bold text-white">
                Predictive Studio
              </span>
              <button
                type="button"
                onClick={() => setMobileMenuOpen(false)}
                className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white"
                aria-label="Cerrar menú"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <nav className="flex-1 space-y-0.5 overflow-y-auto px-3 py-3">
              {NAV_ITEMS.map((item, index) => {
                const Icon = item.icon;
                const active = activeModule === item.id;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => selectModule(item.id)}
                    className={`flex w-full items-center gap-2.5 rounded-lg px-3 py-2.5 text-left text-xs font-medium ${
                      active
                        ? 'bg-teal-600 text-white'
                        : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                    }`}
                  >
                    <Icon className="h-4 w-4 shrink-0" />
                    <span className="truncate">
                      {String(index + 1).padStart(2, '0')} · {item.label}
                    </span>
                  </button>
                );
              })}
            </nav>
          </aside>
        </div>
      )}

      {/* Main Content Area */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Top Header Contract */}
        <header className="sticky top-0 z-30 flex items-center justify-between gap-4 border-b border-slate-200 bg-white px-4 py-3 sm:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <button
              type="button"
              onClick={() => setMobileMenuOpen(true)}
              className="rounded-lg border border-slate-200 p-2 text-slate-700 hover:bg-slate-50 lg:hidden"
              aria-label="Abrir navegación"
            >
              <Menu className="h-4 w-4" />
            </button>
            <div className="flex min-w-0 items-center gap-2 text-sm">
              <span className="hidden font-semibold text-slate-900 sm:inline whitespace-nowrap">
                Predictive Analytics Studio
              </span>
              <span className="hidden text-slate-300 sm:inline">/</span>
              <span className="truncate font-medium text-teal-800">
                {currentNav.label}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-4 shrink-0">
            <div className="hidden md:flex items-center gap-2 text-xs text-slate-500 tabular-nums">
              <span>{dataset ? dataset.fileName : 'Sin base activa'}</span>
              {dataset && (
                <>
                  <span aria-hidden="true">·</span>
                  <span>{dataset.workingRows.length} filas</span>
                </>
              )}
              {isComputing && (
                <>
                  <span aria-hidden="true">·</span>
                  <span className="font-medium text-teal-700">Calculando MLE...</span>
                </>
              )}
            </div>

            <button
              type="button"
              onClick={() =>
                selectModule(activeModel && !activeModel.error ? 'informe' : 'cargar')
              }
              className="rounded-lg bg-slate-900 px-3.5 py-2 text-xs font-medium text-white transition-colors hover:bg-slate-800 whitespace-nowrap shrink-0"
            >
              {activeModel && !activeModel.error ? 'Informe PDF' : 'Cargar Datos'}
            </button>
          </div>
        </header>

        {/* Central Workspace */}
        <main className="mx-auto w-full max-w-[1280px] flex-1 p-4 sm:p-6 lg:p-8">
          {activeModule === 'dashboard' && (
            <DashboardView
              dataset={dataset}
              activeModel={activeModel}
              savedModels={savedModels.filter((m) => !m.error)}
              onNavigate={selectModule}
            />
          )}

          {activeModule === 'cargar' && (
            <UploadDataView
              dataset={dataset}
              onDatasetLoaded={handleDatasetLoaded}
              onNavigate={selectModule}
            />
          )}

          {activeModule === 'exploracion' && (
            <ExploreDataView
              dataset={dataset}
              onOverrideVariableKind={handleOverrideVariableKind}
              onNavigate={selectModule}
            />
          )}

          {activeModule === 'descriptivo' && (
            <DescriptiveView dataset={dataset} onNavigate={selectModule} />
          )}

          {activeModule === 'preparacion' && (
            <PreparationView
              dataset={dataset}
              config={config}
              onUpdateConfig={setConfig}
              onRemoveDuplicates={handleRemoveDuplicates}
              onRestoreOriginal={handleRestoreOriginal}
              onDatasetLoaded={handleDatasetLoaded}
              onNavigate={selectModule}
            />
          )}

          {activeModule === 'regresion' && (
            <RegressionView
              dataset={dataset}
              config={config}
              activeModel={activeModel}
              savedModels={savedModels.filter((m) => !m.error)}
              onUpdateConfig={setConfig}
              onTrainModel={(name) => runTraining(name)}
              onSelectSavedModel={(id) => {
                setActiveModelId(id);
                const found = savedModels.find((m) => m.id === id);
                if (found) setConfig(found.config);
              }}
              onThresholdChange={handleThresholdChange}
              onNavigate={selectModule}
            />
          )}

          {activeModule === 'predicciones' && (
            <PredictionsView
              dataset={dataset}
              activeModel={activeModel}
              onThresholdChange={handleThresholdChange}
              onNavigate={selectModule}
            />
          )}

          {activeModule === 'matriz' && (
            <ConfusionMatrixView
              activeModel={activeModel}
              onThresholdChange={handleThresholdChange}
              onNavigate={selectModule}
            />
          )}

          {activeModule === 'metricas' && (
            <MetricsView activeModel={activeModel} onNavigate={selectModule} />
          )}

          {activeModule === 'balanceo' && (
            <BalancingView
              dataset={dataset}
              config={config}
              activeModel={activeModel}
              onUpdateConfig={setConfig}
              onRetrainWithBalance={handleRetrainWithBalance}
              onNavigate={selectModule}
            />
          )}

          {activeModule === 'validacion' && (
            <ValidationView
              dataset={dataset}
              config={config}
              activeModel={activeModel}
              onUpdateConfig={setConfig}
              onRetrain={() =>
                runTraining(
                  activeModel && !activeModel.error
                    ? activeModel.name
                    : 'Modelo Logístico'
                )
              }
              onNavigate={selectModule}
            />
          )}

          {activeModule === 'simulador' && (
            <SimulatorView
              dataset={dataset}
              activeModel={activeModel}
              scenarios={scenarios}
              onSaveScenario={(s) => setScenarios((prev) => [s, ...prev])}
              onRemoveScenario={(id) =>
                setScenarios((prev) => prev.filter((s) => s.id !== id))
              }
              onNavigate={selectModule}
            />
          )}

          {activeModule === 'interpretacion' && (
            <InterpretationView
              dataset={dataset}
              activeModel={activeModel}
              onNavigate={selectModule}
            />
          )}

          {activeModule === 'informe' && (
            <ReportPDFView
              dataset={dataset}
              activeModel={activeModel}
              reportMeta={reportMeta}
              onUpdateReportMeta={setReportMeta}
              onNavigate={selectModule}
            />
          )}

          {activeModule === 'ayuda' && (
            <HelpView dataset={dataset} activeModel={activeModel} />
          )}
        </main>
      </div>
    </div>
  );
}
