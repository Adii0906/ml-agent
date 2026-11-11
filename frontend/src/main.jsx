import '@fontsource-variable/geist';
import '@fontsource-variable/jetbrains-mono';
import './styles/tokens.css';
import './styles/app.css';

import { MotionConfig } from 'framer-motion';
import { lazy, StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { AppShell } from './components/shell/Shell';
import { ThemeProvider } from './lib/theme';
import { ToastProvider } from './lib/toast';
import { WorkspaceProvider } from './lib/workspace';

const Overview = lazy(() => import('./pages/Overview'));
const Datasets = lazy(() => import('./pages/Datasets'));
const ExplorerRedirect = lazy(() => import('./pages/explorer/ExplorerRedirect'));
const DataExplorer = lazy(() => import('./pages/explorer/DataExplorer'));
const ModelStudio = lazy(() => import('./pages/studio/ModelStudio'));
const Experiments = lazy(() => import('./pages/Experiments'));
const ExperimentDetail = lazy(() => import('./pages/experiment/ExperimentDetail'));
const Compare = lazy(() => import('./pages/Compare'));
const SavedModels = lazy(() => import('./pages/models/SavedModels'));
const ModelDetail = lazy(() => import('./pages/models/ModelDetail'));
const Research = lazy(() => import('./pages/Research'));
const Reports = lazy(() => import('./pages/Reports'));
const ReportView = lazy(() => import('./pages/ReportView'));
const Settings = lazy(() => import('./pages/Settings'));
const NotFound = lazy(() => import('./pages/NotFound'));

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ThemeProvider>
      <MotionConfig reducedMotion="user">
        <ToastProvider>
          <BrowserRouter>
            <WorkspaceProvider>
              <Routes>
                  <Route element={<AppShell />}>
                    <Route index element={<Overview />} />
                    <Route path="datasets" element={<Datasets />} />
                    <Route path="datasets/:id" element={<DataExplorer />} />
                    <Route path="explorer" element={<ExplorerRedirect />} />
                    <Route path="studio" element={<ModelStudio />} />
                    <Route path="experiments" element={<Experiments />} />
                    <Route path="experiments/:id" element={<ExperimentDetail />} />
                    <Route path="compare" element={<Compare />} />
                    <Route path="models" element={<SavedModels />} />
                    <Route path="models/:id" element={<ModelDetail />} />
                    <Route path="research" element={<Research />} />
                    <Route path="reports" element={<Reports />} />
                    <Route path="reports/:id" element={<ReportView />} />
                    <Route path="settings" element={<Settings />} />
                    <Route path="home" element={<Navigate to="/" replace />} />
                    <Route path="*" element={<NotFound />} />
                  </Route>
              </Routes>
            </WorkspaceProvider>
          </BrowserRouter>
        </ToastProvider>
      </MotionConfig>
    </ThemeProvider>
  </StrictMode>,
);
