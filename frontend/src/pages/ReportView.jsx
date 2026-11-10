import { ArrowLeft, Download, Printer, Sparkles } from 'lucide-react';
import { useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { BarList, ConfusionMatrix } from '../components/charts';
import { Button, Callout, EmptyState, ErrorState, Prov, SkeletonBlock } from '../components/ui';
import { api, fileUrl } from '../lib/api';
import { CLASSIFICATION_KEYS, fmtDate, fmtDuration, fmtInt, fmtNum, METRICS, REGRESSION_KEYS } from '../lib/format';
import { useAsync } from '../lib/hooks';
import { useToast } from '../lib/toast';
import { useWorkspace } from '../lib/workspace';

export default function ReportView() {
  const { id } = useParams();
  const toast = useToast();
  const { system } = useWorkspace();
  const rep = useAsync(() => api.report(id).then((d) => d.report), [id]);
  const exp = useAsync(() => api.experiment(id).then((d) => d.experiment), [id]);
  const [aiBusy, setAiBusy] = useState(false);
  const r = rep.data;
  const ai = exp.data?.results?.ai_narrative;

  const generate = async () => {
    setAiBusy(true);
    try { await api.narrative(id); exp.reload(); toast.success('AI narrative added', 'It is labelled as AI-generated in the report and exports.'); } catch (e) { toast.error('AI narrative failed', e.message); } finally { setAiBusy(false); }
  };

  if (rep.error) {
    return <div className="page">{rep.error.status === 409 || rep.error.status === 404
      ? <EmptyState title="No report for this experiment" action={<Link className="btn" to={`/experiments/${id}`}>Open experiment</Link>}>{rep.error.message}</EmptyState>
      : <ErrorState error={rep.error} onRetry={rep.reload} />}</div>;
  }
  if (!r) return <div className="page"><SkeletonBlock rows={10} /></div>;
  const keys = r.task === 'classification' ? CLASSIFICATION_KEYS : REGRESSION_KEYS;
  const m = r.metrics;
  const s = r.methodology.split;
  const rows = r.methodology.rows;
  const imp = r.importance?.permutation;

  return (
    <div className="page">
      <div className="row-between wrap no-print" style={{ marginBottom: 16, maxWidth: 920, marginInline: 'auto' }}>
        <Link className="btn btn-ghost" to={`/experiments/${id}`}><ArrowLeft size={14} />Experiment</Link>
        <div className="row wrap">
          {system?.llm?.available && !ai && <Button icon={Sparkles} loading={aiBusy} onClick={generate}>Add AI narrative</Button>}
          <a className="btn" href={fileUrl(`/experiments/${id}/export-report`, { format: 'md' })}><Download size={14} />Markdown</a>
          <a className="btn" href={fileUrl(`/experiments/${id}/export-report`, { format: 'txt' })}><Download size={14} />Text</a>
          <Button variant="primary" icon={Printer} onClick={() => window.print()}>Print / PDF</Button>
        </div>
      </div>
      <article className="doc">
        <div className="eyebrow">ML-Agent experiment report</div>
        <h1 style={{ marginTop: 6 }}>{r.experiment.name}</h1>
        <p className="small subtle" style={{ marginTop: 6 }}>Experiment <span className="mono">{r.experiment.id}</span> · completed {fmtDate(r.experiment.completed_at)} · generated {fmtDate(r.generated_at)}</p>

        <div className="grid grid-4" style={{ marginTop: 20, gap: 0, border: '1px solid var(--border)', borderRadius: 'var(--radius)' }}>
          {[['Task', r.task], ['Target', r.target], ['Algorithm', r.algorithm.name], [METRICS[r.primary_metric].label + ' (test)', fmtNum(m.test[r.primary_metric], 4)]].map(([k, v], i) => (
            <div key={k} style={{ padding: '10px 14px', borderLeft: i ? '1px solid var(--border)' : 0 }}><div className="xs subtle">{k}</div><div style={{ fontWeight: 600 }} className="truncate">{v}</div></div>
          ))}
        </div>

        <h2>Dataset</h2>
        <p>{r.dataset.name}{r.dataset.filename ? ` (${r.dataset.filename})` : ''} · {fmtInt(r.dataset.rows)} rows{r.dataset.columns ? ` × ${r.dataset.columns} columns` : ''} · SHA-256 <span className="mono xs">{r.dataset.sha256}</span>{!r.dataset.still_available && ' · the dataset has since been deleted'}</p>
        {r.data_quality.length > 0 && (
          <>
            <h3 style={{ marginTop: 12 }}>Data quality at training time</h3>
            <ul>{r.data_quality.map((q) => <li key={q.id}><b>{q.severity}</b> — {q.title}</li>)}</ul>
          </>
        )}

        <h2>Evaluation methodology</h2>
        <ul>
          <li>Hold-out test split: {Math.round(s.test_size * 100)}%, seed {s.random_state}, {s.stratified ? 'stratified by class' : 'not stratified'} (split id <span className="mono xs">{s.split_id}</span>).</li>
          <li>Rows: {fmtInt(rows.n_raw)} raw → {fmtInt(rows.n_used)} used ({rows.dropped_missing_target} without target, {rows.dropped_duplicates} duplicates, {rows.dropped_rare_classes} rare-class rows removed). Train {fmtInt(rows.n_train)} / test {fmtInt(rows.n_test)}.</li>
          <li>Cross-validation: {r.methodology.cv || 'disabled'}.</li>
          <li>Baseline: {r.methodology.baseline}.</li>
          <li>All preprocessing was fitted on the training split only, inside a scikit-learn Pipeline.</li>
        </ul>

        <h2>Results <Prov kind="test" /></h2>
        <div className="table-wrap">
          <table className="table">
            <thead><tr><th>Metric</th><th className="r">Test</th><th className="r">CV (train split)</th><th className="r">Train</th><th className="r">Baseline</th></tr></thead>
            <tbody>
              {keys.map((k) => (
                <tr key={k}><td>{METRICS[k].label}</td><td className="r num" style={{ fontWeight: 600 }}>{fmtNum(m.test[k], 4)}</td>
                  <td className="r num">{m.cv?.mean?.[k] !== undefined ? `${fmtNum(m.cv.mean[k], 4)} ± ${fmtNum(m.cv.std[k], 3)}` : '—'}</td>
                  <td className="r num">{fmtNum(m.train[k], 4)}</td><td className="r num subtle">{fmtNum(m.baseline[k], 4)}</td></tr>
              ))}
              {r.classification?.roc?.auc !== null && r.classification?.roc?.auc !== undefined && <tr><td>ROC-AUC</td><td className="r num" style={{ fontWeight: 600 }}>{fmtNum(r.classification.roc.auc, 4)}</td><td /><td /><td className="r num subtle">0.5</td></tr>}
            </tbody>
          </table>
        </div>

        {r.classification && (
          <div className="grid grid-2" style={{ marginTop: 16 }}>
            <div><h3 style={{ marginBottom: 8 }}>Confusion matrix (test)</h3><ConfusionMatrix labels={r.classification.labels} matrix={r.classification.confusion_matrix} /></div>
            <div>
              <h3 style={{ marginBottom: 8 }}>Per-class recall (test)</h3>
              <BarList items={r.classification.per_class.map((p) => ({ label: p.label, value: p.recall, sub: `n=${p.support}` }))} max={1} format={(v) => v.toFixed(3)} />
            </div>
          </div>
        )}
        {r.regression && (
          <p style={{ marginTop: 10 }}>Residuals on the test set: mean {fmtNum(r.regression.residual_mean, 4)}, standard deviation {fmtNum(r.regression.residual_std, 4)}.</p>
        )}

        {imp && (
          <>
            <h2>Feature importance</h2>
            <p className="small">{imp.method}</p>
            <div style={{ marginTop: 10 }}><BarList items={imp.features.slice(0, 10).map((f, i) => ({ label: f, value: imp.values[i] }))} diverging format={(v) => fmtNum(v, 4)} /></div>
          </>
        )}

        <h2>Preprocessing & configuration</h2>
        <ol>{r.pipeline_stages.map((st) => <li key={st.stage}><b>{st.stage}</b> ({st.fitted_on}) — {st.detail}</li>)}</ol>
        <p style={{ marginTop: 8 }}><b>Hyperparameters:</b> <span className="mono small">{Object.entries(r.algorithm.hyperparameters).map(([k, v]) => `${k}=${v}`).join(', ') || 'none'}</span></p>
        {r.model_info && <p><b>Model:</b> {r.model_info.estimator}{r.model_info.encoded_features ? ` · ${r.model_info.encoded_features} encoded features` : ''} · training time {fmtDuration(r.experiment.training_time)}</p>}

        <h2>Observations <Prov kind="rule" /></h2>
        <ul>{r.observations.map((o) => <li key={o.text}>{o.text}</li>)}</ul>

        {ai && (
          <>
            <h2>AI-generated narrative <Prov kind="ai" /></h2>
            <Callout tone="ai">Written by {ai.provider} ({ai.model}) from the numbers in this report. It may contain errors and is not a verified calculation.</Callout>
            <p style={{ marginTop: 10 }}>{ai.content.summary}</p>
            {ai.content.observations?.length > 0 && <ul>{ai.content.observations.map((o) => <li key={o}>{o}</li>)}</ul>}
            {ai.content.next_steps?.length > 0 && <><h3 style={{ marginTop: 10 }}>Suggested next steps (AI)</h3><ul>{ai.content.next_steps.map((o) => <li key={o}>{o}</li>)}</ul></>}
          </>
        )}

        <h2>Known limitations</h2>
        <ul>{r.limitations.map((l) => <li key={l}>{l}</li>)}</ul>

        <h2>Reproducibility</h2>
        <ul>
          <li>Dataset SHA-256: <span className="mono xs">{r.reproducibility.dataset_sha256}</span></li>
          <li>Random seed {r.reproducibility.random_state}; split id <span className="mono xs">{s.split_id}</span></li>
          <li>Python {r.reproducibility.python}, scikit-learn {r.reproducibility.scikit_learn}, pandas {r.reproducibility.pandas}, numpy {r.reproducibility.numpy}</li>
        </ul>
        <p className="xs subtle" style={{ marginTop: 24, paddingTop: 12, borderTop: '1px solid var(--border)' }}>{r.disclaimer}</p>
      </article>
    </div>
  );
}
