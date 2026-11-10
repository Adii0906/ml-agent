import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { api } from './api';
import { useInterval, useLocalStorage } from './hooks';

const WorkspaceCtx = createContext(null);
const ACTIVE = new Set(['queued', 'running', 'initializing', 'training']);

/** Shared lists (datasets, experiments) + the active dataset context shown in the header. */
export function WorkspaceProvider({ children }) {
  const [datasets, setDatasets] = useState(null);
  const [experiments, setExperiments] = useState(null);
  const [error, setError] = useState(null);
  const [system, setSystem] = useState(null);
  const [activeDatasetId, setActiveDatasetId] = useLocalStorage('mlagent.activeDataset', null);

  const refreshDatasets = useCallback(async () => {
    try {
      const d = await api.datasets();
      setDatasets(d.datasets);
      setError(null);
      return d.datasets;
    } catch (e) { setError(e); return null; }
  }, []);

  const refreshExperiments = useCallback(async () => {
    try {
      const d = await api.experiments();
      setExperiments(d.experiments);
      setError(null);
      return d.experiments;
    } catch (e) { setError(e); return null; }
  }, []);

  const refreshSystem = useCallback(() => api.system().then(setSystem).catch(() => {}), []);

  useEffect(() => { refreshDatasets(); refreshExperiments(); refreshSystem(); }, [refreshDatasets, refreshExperiments, refreshSystem]);

  const hasActive = (experiments || []).some((e) => ACTIVE.has(e.status));
  useInterval(refreshExperiments, hasActive ? 2500 : null);

  useEffect(() => {
    if (datasets && activeDatasetId && !datasets.some((d) => d.id === activeDatasetId)) setActiveDatasetId(null);
  }, [datasets, activeDatasetId, setActiveDatasetId]);

  const activeDataset = useMemo(() => (datasets || []).find((d) => d.id === activeDatasetId) || null, [datasets, activeDatasetId]);

  const value = useMemo(() => ({
    datasets, experiments, error, system, activeDataset, activeDatasetId,
    setActiveDatasetId, refreshDatasets, refreshExperiments, refreshSystem,
    activeCount: (experiments || []).filter((e) => ACTIVE.has(e.status)).length,
  }), [datasets, experiments, error, system, activeDataset, activeDatasetId, setActiveDatasetId, refreshDatasets, refreshExperiments, refreshSystem]);

  return <WorkspaceCtx.Provider value={value}>{children}</WorkspaceCtx.Provider>;
}

export const useWorkspace = () => useContext(WorkspaceCtx);
export const isActiveStatus = (s) => ACTIVE.has(s);
