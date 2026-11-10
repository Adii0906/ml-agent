import { CheckCircle2, GitBranch, Wand2 } from 'lucide-react';
import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button, EmptyState, Modal, Panel, Prov, SeverityIcon, SkeletonBlock } from '../../components/ui';
import { api } from '../../lib/api';
import { useToast } from '../../lib/toast';
import { useWorkspace } from '../../lib/workspace';

/** Data-quality findings. Suggested actions are only applied after explicit review,
 *  and always produce a NEW dataset version — the original upload is never modified. */
export default function QualityPanel({ datasetId, findings, loading }) {
  const [selected, setSelected] = useState({});
  const [review, setReview] = useState(false);
  const [busy, setBusy] = useState(false);
  const [severity, setSeverity] = useState('all');
  const toast = useToast();
  const navigate = useNavigate();
  const { refreshDatasets } = useWorkspace();

  const visible = findings.filter((f) => severity === 'all' || f.severity === severity);
  const chosen = useMemo(() => findings.filter((f) => selected[f.id] && f.action), [findings, selected]);
  const counts = findings.reduce((acc, f) => ({ ...acc, [f.severity]: (acc[f.severity] || 0) + 1 }), {});

  const apply = async () => {
    setBusy(true);
    try {
      // merge drop_columns actions into one
      const dropCols = chosen.filter((f) => f.action.type === 'drop_columns').flatMap((f) => f.action.params.columns);
      const others = chosen.filter((f) => f.action.type !== 'drop_columns').map((f) => ({ type: f.action.type, params: f.action.params }));
      const actions = [...others, ...(dropCols.length ? [{ type: 'drop_columns', params: { columns: [...new Set(dropCols)] } }] : [])];
      const meta = await api.clean(datasetId, actions);
      await refreshDatasets();
      toast.success('New dataset version created', `${meta.name} · ${meta.n_rows.toLocaleString()} rows. The original is unchanged.`);
      setReview(false);
      setSelected({});
      navigate(`/datasets/${meta.id}?tab=quality`);
    } catch (e) {
      toast.error('Could not apply actions', e.message);
    } finally { setBusy(false); }
  };

  if (loading) return <Panel><SkeletonBlock rows={6} /></Panel>;
  if (!findings.length) {
    return <Panel><EmptyState icon={CheckCircle2} title="No issues found">No duplicates, constant columns, identifiers, heavy missingness, type problems or extreme outliers were detected.</EmptyState></Panel>;
  }
  return (
    <>
      <Panel flush title="Data-quality findings" actions={<>
        <Prov kind="rule" />
        <select className="select" style={{ width: 150, height: 28 }} value={severity} onChange={(e) => setSeverity(e.target.value)} aria-label="Filter by severity">
          <option value="all">All ({findings.length})</option>
          {['critical', 'warning', 'info'].filter((s) => counts[s]).map((s) => <option key={s} value={s}>{s} ({counts[s]})</option>)}
        </select>
        <Button variant="primary" size="sm" icon={Wand2} disabled={!chosen.length} onClick={() => setReview(true)}>Review {chosen.length || ''} action{chosen.length === 1 ? '' : 's'}</Button>
      </>} footer="Findings are computed by fixed rules from your data. Tick a suggested action to queue it; nothing changes until you review and confirm.">
        <div style={{ marginTop: 12 }}>
          {visible.map((f) => (
            <div key={f.id} className="finding">
              <SeverityIcon severity={f.severity} />
              <div style={{ minWidth: 0 }}>
                <div className="ttl">{f.title}</div>
                <div className="det">{f.detail}</div>
              </div>
              {f.action ? (
                <label className="check small nowrap" title="Queue this action for review">
                  <input type="checkbox" checked={!!selected[f.id]} onChange={(e) => setSelected((s) => ({ ...s, [f.id]: e.target.checked }))} />
                  <span className="desktop-only">{f.action.label}</span>
                </label>
              ) : <span />}
            </div>
          ))}
        </div>
      </Panel>
      <Modal open={review} onClose={() => setReview(false)} title="Review changes" width={560}
        footer={<><Button onClick={() => setReview(false)}>Cancel</Button><Button variant="primary" icon={GitBranch} loading={busy} onClick={apply}>Create cleaned version</Button></>}>
        <div className="stack">
          <p>These actions will be applied to a <strong>copy</strong> of the dataset. The original upload stays exactly as it is, and existing experiments keep pointing to it.</p>
          <ol style={{ margin: 0, paddingLeft: 18 }} className="stack-sm">
            {chosen.map((f) => <li key={f.id}><strong style={{ color: 'var(--text)' }}>{f.action.label}</strong> <span className="subtle">— {f.title}</span></li>)}
          </ol>
        </div>
      </Modal>
    </>
  );
}
