import { motion } from 'framer-motion';
import { FlaskConical, GitCompare, Layers, Search, Trash2 } from 'lucide-react';
import { useMemo, useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { Button, ConfirmDialog, EmptyState, ErrorState, PageHeader, Panel, Prov, SkeletonBlock, StatusBadge } from '../components/ui';
import { api } from '../lib/api';
import { useToast } from '../lib/toast';
import { fmtDuration, fmtNum, METRICS, primaryValue, timeAgo } from '../lib/format';
import { isActiveStatus, useWorkspace } from '../lib/workspace';

export default function Experiments() {
  const { experiments, datasets, error, refreshExperiments } = useWorkspace();
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const [q, setQ] = useState('');
  const [selected, setSelected] = useState([]);
  const [toDelete, setToDelete] = useState(null);
  const [deleting, setDeleting] = useState(false);
  const toast = useToast();

  const confirmDelete = async () => {
    setDeleting(true);
    try {
      const res = await api.deleteExperiment(toDelete.id);
      await refreshExperiments(); // re-read from the server rather than hiding the row locally
      setSelected((s) => s.filter((x) => x !== toDelete.id));
      toast.success('Experiment deleted', res.saved_models_retained
        ? `${toDelete.name}. ${res.saved_models_retained} saved model${res.saved_models_retained > 1 ? 's' : ''} created from it ${res.saved_models_retained > 1 ? 'were' : 'was'} kept.`
        : toDelete.name);
      setToDelete(null);
    } catch (e) {
      toast.error('Experiment not deleted', e.message);
    } finally {
      setDeleting(false);
    }
  };
  const status = params.get('status') || '';
  const task = params.get('task') || '';
  const dataset = params.get('dataset') || '';
  const setFilter = (k, v) => setParams((p) => { const n = new URLSearchParams(p); if (v) n.set(k, v); else n.delete(k); return n; }, { replace: true });

  const rows = useMemo(() => (experiments || []).filter((e) =>
    (!status || e.status === status) && (!task || e.task_type === task) && (!dataset || e.dataset_id === dataset)
    && (!q || `${e.name} ${e.algorithm} ${e.target}`.toLowerCase().includes(q.toLowerCase()))), [experiments, status, task, dataset, q]);
  const dsName = (id) => datasets?.find((d) => d.id === id)?.name;
  const toggle = (id) => setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));
  const selectedTasks = new Set((experiments || []).filter((e) => selected.includes(e.id)).map((e) => e.task_type));

  return (
    <div className="page">
      <PageHeader eyebrow="Experimentation" title="Experiments"
        lede="Every training run — completed, failed or cancelled — with the configuration and held-out results it produced."
        actions={<>
          <Button icon={GitCompare} disabled={selected.length < 2 || selectedTasks.size > 1} title={selectedTasks.size > 1 ? 'Select experiments of the same task type' : undefined}
            onClick={() => navigate(`/compare?ids=${selected.join(',')}`)}>Compare {selected.length > 1 ? selected.length : ''}</Button>
          <Link className="btn btn-primary" to="/studio"><FlaskConical size={15} />New experiment</Link>
        </>} />
      {error && <ErrorState error={error} onRetry={refreshExperiments} />}
      <Panel flush actions={
        <div className="row wrap" style={{ gap: 8, width: '100%' }}>
          <div className="input-icon" style={{ width: 220 }}><Search size={14} /><input className="input" placeholder="Search name, algorithm, target" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search experiments" /></div>
          <select className="select" style={{ width: 140 }} value={status} onChange={(e) => setFilter('status', e.target.value)} aria-label="Status">
            <option value="">Any status</option>{['completed', 'running', 'queued', 'failed', 'cancelled'].map((s) => <option key={s}>{s}</option>)}
          </select>
          <select className="select" style={{ width: 150 }} value={task} onChange={(e) => setFilter('task', e.target.value)} aria-label="Task">
            <option value="">Any task</option><option value="classification">Classification</option><option value="regression">Regression</option>
          </select>
          <select className="select" style={{ width: 180 }} value={dataset} onChange={(e) => setFilter('dataset', e.target.value)} aria-label="Dataset">
            <option value="">Any dataset</option>{(datasets || []).map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
          </select>
        </div>
      }>
        {experiments === null ? <div style={{ padding: 16 }}><SkeletonBlock rows={6} /></div> : rows.length === 0 ? (
          <EmptyState icon={Layers} title={experiments.length ? 'No experiments match these filters' : 'No experiments yet'}
            action={!experiments.length && <Link className="btn btn-primary" to="/studio">Open Model Studio</Link>}>
            {experiments.length ? 'Clear a filter to see more.' : 'Train your first model in Model Studio. Every run is saved here.'}
          </EmptyState>
        ) : (
          <div className="table-wrap" style={{ marginTop: 12 }}>
            <table className="table">
              <thead>
                <tr>
                  <th style={{ width: 32 }}><span className="sr-only">Select</span></th>
                  <th>Experiment</th><th className="desktop-only">Dataset · target</th><th>Status</th>
                  <th className="r">Test score <Prov kind="test" /></th><th className="r desktop-only">CV</th><th className="r desktop-only">Time</th><th className="desktop-only">Created</th><th style={{ width: 40 }}><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {rows.map((e, i) => {
                  const pv = primaryValue(e);
                  return (
                    <motion.tr key={e.id} className={`clickable ${selected.includes(e.id) ? 'selected' : ''}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: Math.min(i * 0.015, 0.2) }}
                      onClick={() => navigate(`/experiments/${e.id}`)}>
                      <td onClick={(ev) => ev.stopPropagation()}>
                        <input type="checkbox" aria-label={`Select ${e.name}`} checked={selected.includes(e.id)} disabled={e.status !== 'completed'} onChange={() => toggle(e.id)} style={{ accentColor: 'var(--accent)' }} />
                      </td>
                      <td style={{ maxWidth: 300 }}>
                        <Link to={`/experiments/${e.id}`} onClick={(ev) => ev.stopPropagation()} style={{ color: 'var(--text)', fontWeight: 550 }} className="truncate" title={e.name}>{e.name}</Link>
                        <div className="xs subtle">{e.algorithm}{e.group_id ? ' · batch' : ''}</div>
                      </td>
                      <td className="small desktop-only" style={{ maxWidth: 220 }}><div className="truncate">{dsName(e.dataset_id) || <span className="subtle">deleted dataset</span>}</div><div className="xs subtle">{e.target} · {e.task_type}</div></td>
                      <td><StatusBadge status={e.status} /></td>
                      <td className="r num">{pv.value !== null && pv.value !== undefined ? <><span style={{ fontWeight: 600 }}>{fmtNum(pv.value, 4)}</span> <span className="xs subtle">{METRICS[pv.key].label}</span></> : <span className="subtle">—</span>}</td>
                      <td className="r num small desktop-only">{e.metrics?.cv_mean !== undefined ? `${fmtNum(e.metrics.cv_mean, 3)} ± ${fmtNum(e.metrics.cv_std, 3)}` : '—'}</td>
                      <td className="r num small desktop-only">{e.training_time ? fmtDuration(e.training_time) : '—'}</td>
                      <td className="small subtle nowrap desktop-only">{timeAgo(e.created_at)}</td>
                      <td className="r" onClick={(ev) => ev.stopPropagation()}>
                        <Button variant="ghost" size="sm" iconOnly icon={Trash2} aria-label={`Delete ${e.name}`} title="Delete experiment" onClick={() => setToDelete(e)} />
                      </td>
                    </motion.tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
      <ConfirmDialog open={!!toDelete} onClose={() => !deleting && setToDelete(null)} onConfirm={confirmDelete} loading={deleting}
        title="Delete experiment?" confirmLabel="Delete experiment">
        <div className="stack-sm">
          <p><strong style={{ color: 'var(--text)' }}>{toDelete?.name}</strong> will be permanently removed: its record, metrics, logs and working model file.</p>
          {toDelete && isActiveStatus(toDelete.status) && <p>It is still training — the run will be cancelled.</p>}
          <p className="small">Models you saved from it in Saved Models are separate copies and are <b>not</b> deleted.</p>
        </div>
      </ConfirmDialog>
    </div>
  );
}
