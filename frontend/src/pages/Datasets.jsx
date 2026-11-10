import { motion } from 'framer-motion';
import { Database, Download, FlaskConical, GitBranch, Pencil, Sparkles, Table2, Trash2 } from 'lucide-react';
import { useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import Dropzone from '../components/Dropzone';
import { Button, ConfirmDialog, EmptyState, ErrorState, Field, Modal, PageHeader, Panel, SkeletonBlock } from '../components/ui';
import { api, fileUrl } from '../lib/api';
import { useAsync } from '../lib/hooks';
import { fmtBytes, fmtInt, timeAgo } from '../lib/format';
import { useToast } from '../lib/toast';
import { useWorkspace } from '../lib/workspace';

export function SampleDatasets({ compact }) {
  const { data } = useAsync(() => api.samples(), []);
  const [busy, setBusy] = useState(null);
  const toast = useToast();
  const navigate = useNavigate();
  const { refreshDatasets, setActiveDatasetId } = useWorkspace();
  const describe = { iris: 'Classic 3-class flower classification (150 rows)', sample: 'Small customer purchase dataset (29 rows)' };
  const load = async (name) => {
    setBusy(name);
    try {
      const meta = await api.importSample(name);
      await refreshDatasets();
      setActiveDatasetId(meta.id);
      navigate(`/datasets/${meta.id}`);
    } catch (e) {
      toast.error('Could not load sample', e.message);
    } finally { setBusy(null); }
  };
  if (!data?.samples?.length) return null;
  return (
    <div className={compact ? 'stack-sm' : 'grid grid-2'}>
      {data.samples.map((s) => (
        <button key={s.name} className="list-item panel" style={{ borderRadius: 'var(--radius)', borderTop: '1px solid var(--border)' }} onClick={() => load(s.name)} disabled={!!busy}>
          <span className="list-icon"><Sparkles size={14} /></span>
          <span className="grow" style={{ minWidth: 0 }}>
            <span style={{ fontWeight: 600, display: 'block' }}>{s.filename}</span>
            <span className="small subtle truncate" style={{ display: 'block' }}>{describe[s.name] || fmtBytes(s.size_bytes)}</span>
          </span>
          <span className="small" style={{ color: 'var(--accent-text)' }}>{busy === s.name ? 'Loading…' : 'Load'}</span>
        </button>
      ))}
    </div>
  );
}

export default function Datasets() {
  const [params] = useSearchParams();
  const { datasets, error, refreshDatasets, refreshExperiments, setActiveDatasetId } = useWorkspace();
  const [toDelete, setToDelete] = useState(null);
  const [toRename, setToRename] = useState(null);
  const [busy, setBusy] = useState(false);
  const toast = useToast();

  const remove = async () => {
    setBusy(true);
    try {
      await api.deleteDataset(toDelete.id);
      toast.success('Dataset deleted', `${toDelete.name} was removed. Its experiments keep their recorded results.`);
      setToDelete(null);
      refreshDatasets();
      refreshExperiments();
    } catch (e) { toast.error('Delete failed', e.message); } finally { setBusy(false); }
  };
  const rename = async () => {
    setBusy(true);
    try {
      await api.renameDataset(toRename.id, toRename.name);
      setToRename(null);
      refreshDatasets();
    } catch (e) { toast.error('Rename failed', e.message); } finally { setBusy(false); }
  };

  return (
    <div className="page">
      <PageHeader eyebrow="Data" title="Datasets" lede="Upload tabular data once; every profile, experiment and report links back to the exact file version (SHA-256)." />
      <div className="grid grid-main">
        <div className="stack-lg">
          {error && <ErrorState error={error} onRetry={refreshDatasets} />}
          <Panel title="Dataset library" icon={Database} flush
            actions={datasets?.length ? <span className="small subtle">{datasets.length} dataset{datasets.length === 1 ? '' : 's'}</span> : null}>
            {datasets === null ? <div style={{ padding: 16 }}><SkeletonBlock /></div> : datasets.length === 0 ? (
              <EmptyState icon={Database} title="No datasets yet">Upload a CSV on the right, or start with a sample dataset.</EmptyState>
            ) : (
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr><th>Name</th><th className="r">Rows</th><th className="r">Cols</th><th className="r desktop-only">Size</th><th className="desktop-only">Experiments</th><th className="desktop-only">Updated</th><th /></tr>
                  </thead>
                  <tbody>
                    {datasets.map((d, i) => (
                      <motion.tr key={d.id} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: Math.min(i * 0.02, 0.2) }}>
                        <td style={{ maxWidth: 280 }}>
                          <Link to={`/datasets/${d.id}`} onClick={() => setActiveDatasetId(d.id)} style={{ color: 'var(--text)', fontWeight: 550 }} className="truncate" title={d.name}>
                            <span className="row" style={{ gap: 6 }}><Table2 size={13} color="var(--text-3)" /><span className="truncate">{d.name}</span></span>
                          </Link>
                          <div className="xs subtle truncate">
                            {d.parent_id ? <span className="row" style={{ gap: 4, display: 'inline-flex' }}><GitBranch size={10} /> derived · {d.note}</span> : d.filename}
                          </div>
                        </td>
                        <td className="r num">{fmtInt(d.n_rows)}</td>
                        <td className="r num">{d.n_cols}</td>
                        <td className="r num desktop-only">{fmtBytes(d.size_bytes)}</td>
                        <td className="desktop-only">{d.experiment_count ? <Link to={`/experiments?dataset=${d.id}`}>{d.experiment_count}</Link> : <span className="subtle">—</span>}</td>
                        <td className="small subtle nowrap desktop-only">{timeAgo(d.last_opened_at || d.created_at)}</td>
                        <td className="r nowrap">
                          <Link className="btn btn-ghost btn-sm" to={`/studio?dataset=${d.id}`} title="Train a model on this dataset"><FlaskConical size={13} /><span className="desktop-only">Train</span></Link>
                          <a className="btn btn-ghost btn-sm btn-icon" href={fileUrl(`/datasets/${d.id}/download`)} title="Download original file" aria-label={`Download ${d.name}`}><Download size={13} /></a>
                          <Button variant="ghost" size="sm" iconOnly icon={Pencil} title="Rename" aria-label={`Rename ${d.name}`} onClick={() => setToRename({ id: d.id, name: d.name })} />
                          <Button variant="ghost" size="sm" iconOnly icon={Trash2} title="Delete" aria-label={`Delete ${d.name}`} onClick={() => setToDelete(d)} />
                        </td>
                      </motion.tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Panel>
        </div>
        <div className="stack-lg">
          <Panel title="Upload" bodyClass="stack">
            <Dropzone autoFocus={params.get('upload') === '1'} />
            <ul className="small muted" style={{ margin: 0, paddingLeft: 16 }}>
              <li>The first row must contain column names.</li>
              <li>Comma, semicolon or tab delimited; UTF-8 or Latin-1.</li>
              <li>Uploads are never modified. Cleaning creates a new version.</li>
            </ul>
          </Panel>
          <Panel title="Sample datasets"><SampleDatasets compact /></Panel>
        </div>
      </div>

      <ConfirmDialog open={!!toDelete} onClose={() => setToDelete(null)} onConfirm={remove} loading={busy} title="Delete dataset?">
        <strong>{toDelete?.name}</strong> and its file will be removed from the server. Completed experiments keep their recorded metrics and reports, but cannot be re-run on this data.
      </ConfirmDialog>
      <Modal open={!!toRename} onClose={() => setToRename(null)} title="Rename dataset"
        footer={<><Button onClick={() => setToRename(null)}>Cancel</Button><Button variant="primary" onClick={rename} loading={busy} disabled={!toRename?.name?.trim()}>Save</Button></>}>
        <Field label="Name" htmlFor="ds-name">
          <input id="ds-name" className="input" value={toRename?.name || ''} maxLength={120}
            onChange={(e) => setToRename((r) => ({ ...r, name: e.target.value }))} onKeyDown={(e) => e.key === 'Enter' && toRename?.name?.trim() && rename()} />
        </Field>
      </Modal>
    </div>
  );
}
