import { motion } from 'framer-motion';
import { Download, Package, Search, Trash2 } from 'lucide-react';
import { useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Badge, Button, ConfirmDialog, EmptyState, ErrorState, PageHeader, Panel, Prov, SkeletonBlock } from '../../components/ui';
import { api, download } from '../../lib/api';
import { fmtBytes, fmtNum, METRICS, timeAgo } from '../../lib/format';
import { useAsync } from '../../lib/hooks';
import { useToast } from '../../lib/toast';

export function ModelStatus({ status }) {
  if (status === 'ready') return <Badge tone="good"><span className="dot" />Ready</Badge>;
  if (status === 'missing') return <Badge tone="bad" title="An artifact file is missing from storage">Artifact missing</Badge>;
  if (status === 'corrupted') return <Badge tone="bad" title="An artifact no longer matches its recorded size/hash">Integrity failed</Badge>;
  return <Badge>{status}</Badge>;
}

export async function downloadModel(model, format, toast) {
  try {
    const res = await download(`/models/${model.id}/download`, { format }, `${model.lineage}-v${model.version}.${format}`);
    toast.success(`Downloaded ${res.filename}`, `SHA-256 ${res.sha256?.slice(0, 16)}… — verify it against the manifest before loading.`);
  } catch (e) {
    toast.error('Download failed', e.message);
  }
}

export default function SavedModels() {
  const list = useAsync(() => api.models(), []);
  const [q, setQ] = useState('');
  const [toDelete, setToDelete] = useState(null);
  const [busy, setBusy] = useState(false);
  const toast = useToast();
  const navigate = useNavigate();
  const models = useMemo(() => (list.data?.models || []).filter((m) =>
    !q || `${m.name} ${m.algorithm_name} ${m.dataset_name} ${m.target}`.toLowerCase().includes(q.toLowerCase())), [list.data, q]);

  const remove = async () => {
    setBusy(true);
    try {
      await api.deleteModel(toDelete.id);
      await list.reload();
      toast.success('Saved model deleted', `${toDelete.name} v${toDelete.version} and its artifacts were removed.`);
      setToDelete(null);
    } catch (e) {
      toast.error('Model not deleted', e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="page">
      <PageHeader eyebrow="Experimentation" title="Saved models"
        lede="Named, versioned copies of trained pipelines (preprocessing + estimator) with their input schema, metrics and reproducibility metadata. Independent of the experiments they came from."
        actions={<Link className="btn" to="/experiments">From an experiment →</Link>} />
      {list.error && <ErrorState error={list.error} onRetry={list.reload} title="Could not load saved models" />}
      <Panel flush actions={
        <div className="input-icon" style={{ width: 260 }}><Search size={14} /><input className="input" placeholder="Search name, algorithm, dataset" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search saved models" /></div>
      }>
        {!list.data ? <div style={{ padding: 16 }}><SkeletonBlock rows={5} /></div> : models.length === 0 ? (
          <EmptyState icon={Package} title={list.data.models.length ? 'No models match' : 'No saved models yet'}
            action={!list.data.models.length && <Link className="btn btn-primary" to="/experiments">Open an experiment</Link>}>
            {list.data.models.length ? 'Try a different search.' : 'Open a completed experiment and choose “Save model” to keep its fitted pipeline here.'}
          </EmptyState>
        ) : (
          <div className="table-wrap" style={{ marginTop: 12 }}>
            <table className="table">
              <thead>
                <tr><th>Model</th><th className="desktop-only">Algorithm</th><th className="desktop-only">Dataset → target</th>
                  <th className="r">Test score <Prov kind="test" /></th><th>Status</th><th className="desktop-only">Trained</th><th className="desktop-only">Saved</th><th /></tr>
              </thead>
              <tbody>
                {models.map((m, i) => (
                  <motion.tr key={m.id} className="clickable" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: Math.min(i * 0.02, 0.2) }}
                    onClick={() => navigate(`/models/${m.id}`)}>
                    <td style={{ maxWidth: 280 }}>
                      <Link to={`/models/${m.id}`} onClick={(ev) => ev.stopPropagation()} className="truncate" style={{ color: 'var(--text)', fontWeight: 550, display: 'block' }}>
                        {m.name} <span className="mono xs subtle">v{m.version}</span>
                      </Link>
                      <span className="xs subtle mono">{m.id}</span>
                    </td>
                    <td className="small desktop-only">{m.algorithm_name}</td>
                    <td className="small desktop-only" style={{ maxWidth: 220 }}><div className="truncate">{m.dataset_name || '—'} → {m.target}</div><div className="xs subtle">{m.task}</div></td>
                    <td className="r num">{m.primary_value !== null && m.primary_value !== undefined ? <>{fmtNum(m.primary_value, 4)} <span className="xs subtle">{METRICS[m.primary_metric]?.label}</span></> : '—'}</td>
                    <td><ModelStatus status={m.status} /></td>
                    <td className="small subtle nowrap desktop-only">{timeAgo(m.trained_at)}</td>
                    <td className="small subtle nowrap desktop-only">{timeAgo(m.created_at)}</td>
                    <td className="r nowrap" onClick={(ev) => ev.stopPropagation()}>
                      <Button variant="ghost" size="sm" icon={Download} disabled={m.status !== 'ready'} title={`Download .joblib (${fmtBytes(m.artifacts?.joblib?.bytes)})`}
                        onClick={() => downloadModel(m, 'joblib', toast)}><span className="desktop-only">.joblib</span></Button>
                      <Button variant="ghost" size="sm" icon={Download} disabled={m.status !== 'ready'} title={`Download .pkl (${fmtBytes(m.artifacts?.pkl?.bytes)})`}
                        onClick={() => downloadModel(m, 'pkl', toast)} className="desktop-only">.pkl</Button>
                      <Button variant="ghost" size="sm" iconOnly icon={Trash2} aria-label={`Delete ${m.name} v${m.version}`} title="Delete saved model" onClick={() => setToDelete(m)} />
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
      <ConfirmDialog open={!!toDelete} onClose={() => !busy && setToDelete(null)} onConfirm={remove} loading={busy} title="Delete saved model?" confirmLabel="Delete model">
        <strong style={{ color: 'var(--text)' }}>{toDelete?.name} v{toDelete?.version}</strong> and its .joblib, .pkl and manifest files will be permanently removed. Other versions and the source experiment are not affected.
      </ConfirmDialog>
    </div>
  );
}
