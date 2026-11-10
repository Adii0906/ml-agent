import { AnimatePresence, motion } from 'framer-motion';
import { Download, FileText, FlaskConical, GitCompare, Layers, Pencil, RotateCcw, Trash2 } from 'lucide-react';
import { useCallback, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { Button, ConfirmDialog, Disclaimer, EmptyState, ErrorState, Field, Modal, PageHeader, Panel, SkeletonBlock, StatusBadge } from '../../components/ui';
import { api, fileUrl } from '../../lib/api';
import { fmtDate, fmtDuration, fmtNum, METRICS } from '../../lib/format';
import { useAsync } from '../../lib/hooks';
import { useToast } from '../../lib/toast';
import { isActiveStatus, useWorkspace } from '../../lib/workspace';
import Evaluation from './Evaluation';
import TrainingMonitor from './TrainingMonitor';

export default function ExperimentDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const toast = useToast();
  const { refreshExperiments } = useWorkspace();
  const exp = useAsync(() => api.experiment(id).then((d) => d.experiment), [id]);
  const e = exp.data;
  const completed = e?.status === 'completed' && e.results?.metrics;
  const report = useAsync(() => api.report(id).then((d) => d.report), [id, completed], { enabled: !!completed });
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [rename, setRename] = useState(null);
  const [busy, setBusy] = useState(false);

  const onFinished = useCallback((evt) => {
    refreshExperiments();
    exp.reload();
    if (evt.status === 'completed') toast.success('Training completed', 'Evaluation results are ready.');
    else if (evt.status === 'failed') toast.error('Training failed', evt.error || evt.message);
    else if (evt.status === 'cancelled') toast.info('Training cancelled');
  }, [exp, refreshExperiments, toast]);

  const remove = async () => {
    setBusy(true);
    try {
      await api.deleteExperiment(id);
      await refreshExperiments();
      toast.success('Experiment deleted');
      navigate('/experiments');
    } catch (err) { toast.error('Delete failed', err.message); setBusy(false); }
  };
  const doRename = async () => {
    setBusy(true);
    try { await api.renameExperiment(id, rename); setRename(null); exp.reload(); refreshExperiments(); } catch (err) { toast.error('Rename failed', err.message); } finally { setBusy(false); }
  };

  if (exp.error?.status === 404) {
    return <div className="page"><EmptyState icon={Layers} title="Experiment not found" action={<Link className="btn" to="/experiments">All experiments</Link>}>It may have been deleted.</EmptyState></div>;
  }
  if (exp.error) return <div className="page"><ErrorState error={exp.error} onRetry={exp.reload} /></div>;
  if (!e) return <div className="page"><SkeletonBlock rows={6} /></div>;

  const cfg = e.config || {};
  const primary = e.metrics?.primary_metric;
  const primaryVal = primary ? e.results?.metrics?.test?.[primary] : null;
  const studioLink = `/studio?dataset=${e.dataset_id || cfg.dataset_id || ''}&target=${encodeURIComponent(e.target || cfg.target || '')}`;

  return (
    <div className="page">
      <PageHeader
        eyebrow={<span className="row" style={{ gap: 8 }}>Experiment <StatusBadge status={e.status} /></span>}
        title={<span className="row" style={{ gap: 8 }}>{e.name}<Button variant="ghost" size="sm" iconOnly icon={Pencil} aria-label="Rename experiment" onClick={() => setRename(e.name)} /></span>}
        lede={<>
          {e.algorithm} · target <b>{e.target || cfg.target}</b> ({e.task_type || cfg.task_type}) · {cfg.dataset_id ? <Link to={`/datasets/${cfg.dataset_id}`}>{cfg.dataset_name || 'dataset'}</Link> : 'dataset'}
          {' · '}created {fmtDate(e.created_at)}{e.training_time ? ` · ${fmtDuration(e.training_time)}` : ''}
          {primaryVal !== null && primaryVal !== undefined && <> · <b className="num">{METRICS[primary].label} {fmtNum(primaryVal, 4)}</b> (test)</>}
        </>}
        actions={<>
          {completed && <Link className="btn" to={`/compare?ids=${id}`}><GitCompare size={14} />Compare</Link>}
          {completed && <Link className="btn" to={`/reports/${id}`}><FileText size={14} />Report</Link>}
          {completed && <a className="btn" href={fileUrl(`/experiments/${id}/export-model`)} title="Download the fitted scikit-learn pipeline (joblib)"><Download size={14} />Model</a>}
          {!isActiveStatus(e.status) && <Link className="btn" to={studioLink}><RotateCcw size={14} />Iterate</Link>}
          <Button iconOnly icon={Trash2} variant="ghost" aria-label="Delete experiment" title="Delete experiment" onClick={() => setConfirmDelete(true)} />
        </>}
      />

      <AnimatePresence mode="wait">
        <motion.div key={isActiveStatus(e.status) ? 'monitor' : e.status} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.22 }}>
          {isActiveStatus(e.status) && <TrainingMonitor experiment={e} onFinished={onFinished} />}
          {completed && <Evaluation experiment={e} report={report.data} />}
          {(e.status === 'failed' || e.status === 'cancelled') && (
            <div className="grid grid-main">
              <Panel title={e.status === 'failed' ? 'Training failed' : 'Training cancelled'}>
                <div className="stack">
                  <div className={`callout ${e.status === 'failed' ? 'callout-bad' : ''}`} role="alert"><div className="grow"><strong>{e.status === 'failed' ? 'Error' : 'Status'}:</strong> {e.error || e.status}</div></div>
                  <p className="small muted">Failed and cancelled runs stay in your history so nothing disappears silently. Adjust the configuration in Model Studio and try again.</p>
                  <div><Link className="btn btn-primary" to={studioLink}><FlaskConical size={14} />Open in Model Studio</Link></div>
                </div>
              </Panel>
              <Panel title="Event log">
                <div className="log">
                  {(e.logs || []).map((l, i) => <div key={`${l.seq ?? 'n'}-${i}`} className={`log-line ${l.level || ''} ${l.status === 'failed' ? 'error' : ''}`}><span className="t">{l.time ? new Date(l.time).toLocaleTimeString([], { hour12: false }) : ''}</span><span className="s">{l.stage || l.status}</span><span className="m">{l.message}</span></div>)}
                  {!e.logs?.length && <span className="subtle">No events recorded.</span>}
                </div>
              </Panel>
            </div>
          )}
        </motion.div>
      </AnimatePresence>
      <Disclaimer className="no-print" />

      <ConfirmDialog open={confirmDelete} onClose={() => setConfirmDelete(false)} onConfirm={remove} loading={busy} title="Delete experiment?">
        This permanently removes the experiment record, its metrics and the saved model file.
      </ConfirmDialog>
      <Modal open={rename !== null} onClose={() => setRename(null)} title="Rename experiment"
        footer={<><Button onClick={() => setRename(null)}>Cancel</Button><Button variant="primary" loading={busy} disabled={!rename?.trim()} onClick={doRename}>Save</Button></>}>
        <Field label="Name" htmlFor="exp-name"><input id="exp-name" className="input" maxLength={120} value={rename || ''} onChange={(ev) => setRename(ev.target.value)} onKeyDown={(ev) => ev.key === 'Enter' && rename?.trim() && doRename()} /></Field>
      </Modal>
    </div>
  );
}
