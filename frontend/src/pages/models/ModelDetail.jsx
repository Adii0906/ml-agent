import { ArrowLeft, Copy, Download, FileJson, Pencil, Play, Rocket, ShieldCheck, Trash2 } from 'lucide-react';
import { useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { Badge, Button, Callout, ConfirmDialog, EmptyState, ErrorState, Field, Modal, PageHeader, Panel, Prov, SkeletonBlock } from '../../components/ui';
import { api, download } from '../../lib/api';
import { CLASSIFICATION_KEYS, fmtBytes, fmtDate, fmtNum, fmtPct, METRICS, REGRESSION_KEYS } from '../../lib/format';
import { useAsync } from '../../lib/hooks';
import { useToast } from '../../lib/toast';
import { downloadModel, ModelStatus } from './SavedModels';

function TestPrediction({ model }) {
  const cols = model.manifest.input_schema.columns;
  const [values, setValues] = useState(() => Object.fromEntries(cols.map((c) => [c.name, ''])));
  const [out, setOut] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const run = async () => {
    setBusy(true); setError(null);
    try {
      const row = Object.fromEntries(cols.map((c) => [c.name, values[c.name] === '' ? null : values[c.name]]));
      setOut((await api.modelPredict(model.id, [row])).predictions[0]);
    } catch (e) { setError(e); } finally { setBusy(false); }
  };
  return (
    <Panel title="Test prediction" icon={Play} actions={<Button size="sm" variant="primary" icon={Play} loading={busy} onClick={run} disabled={model.status !== 'ready'}>Predict</Button>}
      footer="Runs the stored artifact (hash-verified) through the same code path a future prediction API would use. Empty fields are treated as missing and imputed.">
      <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 10 }}>
        {cols.slice(0, 40).map((c) => (
          <Field key={c.name} label={<span className="truncate" title={`${c.name} · ${c.role}${c.dtype ? ` · ${c.dtype}` : ''}`}>{c.name}</span>} htmlFor={`tp-${c.name}`}>
            <input id={`tp-${c.name}`} className="input num" value={values[c.name]} placeholder={c.role === 'numeric' ? 'number' : 'text'}
              onChange={(e) => setValues((v) => ({ ...v, [c.name]: e.target.value }))} />
          </Field>
        ))}
      </div>
      {error && <div style={{ marginTop: 10 }}><ErrorState error={error} title="Prediction failed" /></div>}
      {out && (
        <div className="callout" style={{ marginTop: 12 }}>
          <div className="grow">
            <div>Prediction: <b className="num" style={{ fontSize: 15 }}>{typeof out.prediction === 'number' ? fmtNum(out.prediction, 4) : out.prediction}</b></div>
            {out.probabilities && <div className="small muted">{Object.entries(out.probabilities).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k}: ${fmtPct(v)}`).join(' · ')}</div>}
          </div>
        </div>
      )}
    </Panel>
  );
}

export default function ModelDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const toast = useToast();
  const m = useAsync(() => api.model(id), [id]);
  const [edit, setEdit] = useState(null);
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const [verify, setVerify] = useState(null);

  if (m.error) {
    return <div className="page">{m.error.status === 404
      ? <EmptyState title="Saved model not found" action={<Link className="btn" to="/models">All saved models</Link>}>It may have been deleted, or it belongs to another owner.</EmptyState>
      : <ErrorState error={m.error} onRetry={m.reload} />}</div>;
  }
  const model = m.data;
  if (!model) return <div className="page"><SkeletonBlock rows={8} /></div>;
  const man = model.manifest;
  const keys = model.task === 'classification' ? CLASSIFICATION_KEYS : REGRESSION_KEYS;
  const met = man.metrics;

  const saveEdit = async () => {
    setBusy(true);
    try {
      await api.updateModel(id, { name: edit.name.trim(), description: edit.description.trim() || null });
      setEdit(null);
      await m.reload();
      toast.success('Model details updated');
    } catch (e) { toast.error('Update failed', e.message); } finally { setBusy(false); }
  };
  const remove = async () => {
    setBusy(true);
    try {
      await api.deleteModel(id);
      toast.success('Saved model deleted', `${model.name} v${model.version}`);
      navigate('/models');
    } catch (e) { toast.error('Model not deleted', e.message); setBusy(false); }
  };
  const runVerify = async () => {
    setVerify({ loading: true });
    try { setVerify({ data: await api.verifyModel(id) }); } catch (e) { setVerify({ error: e.message }); }
  };
  const snippet = `from mlagent_runtime import load_artifact, predict   # backend/mlagent_runtime in the ML-Agent repo
# pip install "scikit-learn==${man.environment.scikit_learn}" "pandas==${man.environment.pandas}" "numpy==${man.environment.numpy}"
bundle = load_artifact("${model.lineage}-v${model.version}.joblib",
                       expected_sha256="${model.artifacts.joblib.sha256}")
predict(bundle, [{${man.input_schema.columns.slice(0, 3).map((c) => `"${c.name}": ...`).join(', ')}${man.input_schema.columns.length > 3 ? ', ...' : ''}}])`;

  return (
    <div className="page">
      <Link className="btn btn-ghost no-print" to="/models" style={{ marginBottom: 8, marginLeft: -8 }}><ArrowLeft size={14} />Saved models</Link>
      <PageHeader
        eyebrow={<span className="row" style={{ gap: 8 }}>Saved model <ModelStatus status={model.status} /></span>}
        title={<span className="row" style={{ gap: 8 }}>{model.name} <span className="mono subtle" style={{ fontSize: 15 }}>v{model.version}</span>
          <Button variant="ghost" size="sm" iconOnly icon={Pencil} aria-label="Edit name and description" onClick={() => setEdit({ name: model.name, description: model.description || '' })} /></span>}
        lede={<>{model.description || <span className="subtle">No description.</span>}<br />
          <span className="mono xs">{model.id}</span> · {model.algorithm_name} · {model.task} · target <b>{model.target}</b> · saved {fmtDate(model.created_at)}</>}
        actions={<>
          <Button icon={Download} disabled={model.status !== 'ready'} onClick={() => downloadModel(model, 'joblib', toast)}>.joblib</Button>
          <Button icon={Download} disabled={model.status !== 'ready'} onClick={() => downloadModel(model, 'pkl', toast)}>.pkl</Button>
          <Button icon={FileJson} onClick={() => download(`/models/${id}/manifest`, undefined, 'manifest.json').catch((e) => toast.error('Download failed', e.message))}>Manifest</Button>
          <Button variant="ghost" iconOnly icon={Trash2} aria-label="Delete saved model" title="Delete saved model" onClick={() => setConfirm(true)} />
        </>}
      />
      {model.status !== 'ready' && <div style={{ marginBottom: 16 }}><Callout tone="bad" title="Artifact problem.">One of the stored files is missing or no longer matches its recorded size. Downloads and predictions are blocked until this is resolved; delete and re-save from the experiment if it still exists.</Callout></div>}

      <div className="grid grid-main">
        <div className="stack-lg">
          <Panel title="Held-out evaluation" actions={<Prov kind="test" />} flush
            footer={`Recorded when the source experiment ran: ${man.dataset.n_test} test rows, seed ${man.training.random_state}.`}>
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>Metric</th><th className="r">Test</th><th className="r">CV (train split)</th><th className="r">Baseline</th></tr></thead>
                <tbody>
                  {keys.map((k) => (
                    <tr key={k}><td>{METRICS[k].label}</td><td className="r num" style={{ fontWeight: 600 }}>{fmtNum(met.test[k], 4)}</td>
                      <td className="r num">{met.cv?.mean?.[k] !== undefined ? `${fmtNum(met.cv.mean[k], 4)} ± ${fmtNum(met.cv.std[k], 3)}` : '—'}</td>
                      <td className="r num subtle">{fmtNum(met.baseline[k], 4)}</td></tr>
                  ))}
                  {met.roc_auc !== null && met.roc_auc !== undefined && <tr><td>ROC-AUC</td><td className="r num" style={{ fontWeight: 600 }}>{fmtNum(met.roc_auc, 4)}</td><td /><td className="r num subtle">0.5</td></tr>}
                </tbody>
              </table>
            </div>
          </Panel>

          <Panel title="Input schema" flush footer={`Missing values are allowed (imputed by the pipeline). Unseen categories are ignored by the encoder. Extra columns are ignored; column order does not matter.`}>
            <div className="table-wrap" style={{ maxHeight: 360 }}>
              <table className="table">
                <thead><tr><th>#</th><th>Column</th><th>Role</th><th>Training dtype</th></tr></thead>
                <tbody>
                  {man.input_schema.columns.map((c) => (
                    <tr key={c.name}><td className="num subtle">{man.input_schema.column_order.indexOf(c.name) + 1}</td><td className="mono small">{c.name}</td>
                      <td><Badge tone={c.role === 'numeric' ? 'info' : undefined}>{c.role}</Badge></td><td className="mono small subtle">{c.dtype || '—'}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>

          <div className="grid grid-2">
            <Panel title="Output">
              <dl className="kv">
                <dt>Type</dt><dd>{man.output_schema.type === 'class_label' ? 'Class label' : 'Number'}</dd>
                {man.output_schema.classes && <><dt>Classes</dt><dd className="small">{man.output_schema.classes.join(', ')}</dd></>}
                <dt>Probabilities</dt><dd>{man.output_schema.probabilities ? 'when the estimator supports them' : 'no'}</dd>
              </dl>
            </Panel>
            <Panel title="Hyperparameters">
              {Object.keys(man.algorithm.hyperparameters).length ? (
                <dl className="kv">{Object.entries(man.algorithm.hyperparameters).map(([k, v]) => <div key={k} style={{ display: 'contents' }}><dt className="mono small">{k}</dt><dd className="num">{v === null ? 'None' : String(v)}</dd></div>)}</dl>
              ) : <span className="small subtle">None.</span>}
            </Panel>
          </div>

          <Panel title="Preprocessing pipeline (fitted, stored in the artifact)">
            <ol style={{ margin: 0, paddingLeft: 18 }} className="stack-sm small">
              {man.pipeline_stages.map((s) => <li key={s.stage}><b>{s.stage}</b> <span className="subtle">· fitted on {s.fitted_on}</span><div className="muted">{s.detail}</div></li>)}
            </ol>
          </Panel>

          <TestPrediction model={model} />
        </div>

        <div className="stack-lg">
          <Panel title="Deployment" icon={Rocket} actions={<Badge>Coming soon</Badge>}>
            <p className="small muted">Deploying saved models as hosted endpoints is not available yet. This model is already stored in a deployment-ready form:</p>
            <ul className="small" style={{ margin: '8px 0 0', paddingLeft: 16 }}>
              <li>Stable id <span className="mono xs">{model.id}</span> and version {model.version}</li>
              <li>Complete fitted pipeline with a typed input schema</li>
              <li>SHA-256 verified artifacts + separate manifest</li>
              <li>Pinned library versions for a serving image</li>
            </ul>
          </Panel>

          <Panel title="Provenance">
            <dl className="kv">
              <dt>Source experiment</dt><dd className="small">{model.source_experiment_exists ? <Link to={`/experiments/${model.experiment_id}`}>{man.source.experiment_name}</Link> : <span className="subtle">{man.source.experiment_name} (deleted)</span>}</dd>
              <dt>Trained</dt><dd className="small">{fmtDate(model.trained_at)}</dd>
              <dt>Dataset</dt><dd className="small">{man.dataset.name} · {man.dataset.rows_used} rows used</dd>
              <dt>Dataset SHA-256</dt><dd className="mono xs">{man.dataset.sha256}</dd>
              <dt>Split</dt><dd className="small">{Math.round(man.training.test_size * 100)}% test · seed {man.training.random_state} · {man.training.stratified ? 'stratified' : 'not stratified'} · CV {man.training.cv_folds || 'off'}</dd>
              <dt>Owner</dt><dd className="small">{model.owner}</dd>
            </dl>
          </Panel>

          <Panel title="Artifacts" icon={ShieldCheck} actions={<Button size="sm" loading={verify?.loading} onClick={runVerify}>Verify</Button>}>
            <dl className="kv">
              {Object.entries(model.artifacts).map(([fmt, a]) => (
                <div key={fmt} style={{ display: 'contents' }}><dt className="mono small">.{fmt}</dt><dd><span className="small">{fmtBytes(a.bytes)}</span><div className="mono xs subtle">{a.sha256}</div></dd></div>
              ))}
            </dl>
            {verify?.data && <div style={{ marginTop: 10 }}>{verify.data.valid ? <Callout tone="good">Both artifacts match their recorded SHA-256.</Callout>
              : <Callout tone="bad">{Object.entries(verify.data.artifacts).map(([k, v]) => `${k}: ${v}`).join(' · ')}</Callout>}</div>}
            {verify?.error && <Callout tone="bad">{verify.error}</Callout>}
          </Panel>

          <Panel title="Environment">
            <dl className="kv">
              <dt>scikit-learn</dt><dd className="num">{man.environment.scikit_learn}</dd>
              <dt>pandas · numpy</dt><dd className="num">{man.environment.pandas} · {man.environment.numpy}</dd>
              <dt>Python</dt><dd className="num">{man.environment.python}</dd>
              <dt>Estimator</dt><dd className="mono small">{man.algorithm.estimator_class}</dd>
            </dl>
            <div className="row-between" style={{ marginTop: 12 }}><div className="eyebrow">Load in Python</div>
              <Button size="sm" variant="ghost" icon={Copy} onClick={() => navigator.clipboard?.writeText(snippet).then(() => toast.success('Copied'), () => toast.error('Copy failed'))}>Copy</Button></div>
            <pre className="log" style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', marginTop: 6 }}>{snippet}</pre>
            <p className="xs subtle">Only load artifacts you trust: joblib/pickle files can run code when loaded. Always pass the SHA-256 from the manifest.</p>
          </Panel>

          {model.versions.length > 1 && (
            <Panel title="Versions" flush>
              <div className="list">
                {model.versions.slice().reverse().map((v) => (
                  <Link key={v.id} to={`/models/${v.id}`} className="list-item" style={{ fontWeight: v.id === id ? 600 : 400 }}>
                    <span className="mono small">v{v.version}</span><span className="grow small subtle">{fmtDate(v.created_at)}</span>{v.id === id && <Badge>current</Badge>}
                  </Link>
                ))}
              </div>
            </Panel>
          )}
        </div>
      </div>

      <ConfirmDialog open={confirm} onClose={() => !busy && setConfirm(false)} onConfirm={remove} loading={busy} title="Delete saved model?" confirmLabel="Delete model">
        <strong style={{ color: 'var(--text)' }}>{model.name} v{model.version}</strong> and its .joblib, .pkl and manifest files will be permanently removed. Other versions and the source experiment are not affected.
      </ConfirmDialog>
      <Modal open={!!edit} onClose={() => setEdit(null)} title="Edit saved model"
        footer={<><Button onClick={() => setEdit(null)}>Cancel</Button><Button variant="primary" loading={busy} disabled={!edit?.name?.trim()} onClick={saveEdit}>Save</Button></>}>
        <div className="stack">
          <Field label="Name" htmlFor="md-name" hint="Renaming does not change the version lineage."><input id="md-name" className="input" maxLength={120} value={edit?.name || ''} onChange={(e) => setEdit((x) => ({ ...x, name: e.target.value }))} /></Field>
          <Field label="Description" htmlFor="md-desc"><textarea id="md-desc" className="textarea" rows={3} maxLength={2000} value={edit?.description || ''} onChange={(e) => setEdit((x) => ({ ...x, description: e.target.value }))} /></Field>
        </div>
      </Modal>
    </div>
  );
}
