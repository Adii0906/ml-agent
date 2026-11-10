import { Activity, AlertTriangle, BarChart3, Bug, Gauge, Play, Settings2, Sigma } from 'lucide-react';
import { useMemo, useState } from 'react';
import { BarList, ConfusionMatrix, Histogram, LearningCurveChart, MiniBars, RocChart, ScatterPlot } from '../../components/charts';
import { Button, Callout, EmptyState, ErrorState, Field, Panel, Prov, Segmented, Tabs } from '../../components/ui';
import { api } from '../../lib/api';
import { CLASSIFICATION_KEYS, fmtDuration, fmtInt, fmtNum, fmtPct, METRICS, REGRESSION_KEYS } from '../../lib/format';

function MetricGrid({ res }) {
  const isClf = res.task === 'classification';
  const keys = isClf ? CLASSIFICATION_KEYS : REGRESSION_KEYS;
  const m = res.metrics;
  const extra = isClf && res.classification.roc.auc !== null ? [['roc_auc', res.classification.roc.auc]] : [];
  return (
    <div className="metric-grid">
      {[...keys.map((k) => [k, m.test[k]]), ...extra].map(([k, v]) => {
        const def = METRICS[k];
        const base = k === 'roc_auc' ? 0.5 : m.baseline[k];
        const delta = v !== null && base !== null && base !== undefined ? v - base : null;
        const better = delta !== null && (def.higher ? delta > 0 : delta < 0);
        return (
          <div key={k} className="metric-cell" title={def.help}>
            <div className="k">{def.label}</div>
            <div className="v">{k === 'mape' && v !== null ? fmtPct(v) : fmtNum(v, 4)}</div>
            <div className="d">
              {delta !== null ? <span className={better ? 'delta-good' : 'delta-bad'}>{delta > 0 ? '+' : ''}{fmtNum(delta, 3)}</span> : null}
              {delta !== null ? ` vs ${k === 'roc_auc' ? 'chance' : 'baseline'}` : k === 'mape' ? 'undefined (zeros in target)' : ''}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function SplitComparison({ res }) {
  const isClf = res.task === 'classification';
  const keys = isClf ? ['accuracy', 'balanced_accuracy', 'f1_macro'] : ['r2', 'rmse', 'mae'];
  const cv = res.metrics.cv;
  return (
    <div className="table-wrap">
      <table className="table">
        <thead><tr><th>Metric</th><th className="r"><Prov kind="test" /></th><th className="r"><Prov kind="cv" /></th><th className="r"><Prov kind="train" /></th><th className="r">Baseline</th></tr></thead>
        <tbody>
          {keys.map((k) => (
            <tr key={k}>
              <td title={METRICS[k].help}>{METRICS[k].label}</td>
              <td className="r num" style={{ fontWeight: 600 }}>{fmtNum(res.metrics.test[k], 4)}</td>
              <td className="r num">{cv?.mean?.[k] !== undefined ? <>{fmtNum(cv.mean[k], 4)} <span className="subtle">± {fmtNum(cv.std[k], 3)}</span></> : '—'}</td>
              <td className="r num">{fmtNum(res.metrics.train[k], 4)}</td>
              <td className="r num subtle">{fmtNum(res.metrics.baseline[k], 4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Observations({ report }) {
  if (!report) return null;
  return (
    <div className="stack-sm">
      {report.observations.map((o) => (
        <div key={o.text} className="row small" style={{ alignItems: 'flex-start', gap: 8 }}>
          {o.kind === 'warning' ? <AlertTriangle size={14} color="var(--warn)" style={{ marginTop: 2, flexShrink: 0 }} /> : <span className="dot" style={{ marginTop: 7, color: 'var(--text-3)' }} />}
          <span>{o.text}</span>
        </div>
      ))}
    </div>
  );
}

function ClassificationView({ res }) {
  const [norm, setNorm] = useState(false);
  const c = res.classification;
  return (
    <div className="stack-lg">
      <div className="grid grid-2">
        <Panel title="Confusion matrix" actions={<><Prov kind="test" /><Segmented ariaLabel="Normalisation" value={norm ? 'row' : 'count'} onChange={(v) => setNorm(v === 'row')}
          options={[{ value: 'count', label: 'Counts' }, { value: 'row', label: 'Row %' }]} /></>}>
          <ConfusionMatrix labels={c.labels} matrix={c.confusion_matrix} normalize={norm} />
        </Panel>
        <Panel title="ROC curve" actions={c.roc.auc !== null ? <span className="num small">AUC {c.roc.auc.toFixed(4)}</span> : null}>
          {c.roc.curves.length ? (
            <>
              <RocChart curves={c.roc.curves} />
              <div className="xs subtle">Scores: {c.roc.score_type}{c.roc.average ? ` · ${c.roc.average}` : ''}. Diagonal = random ranking.</div>
            </>
          ) : <EmptyState title="ROC not available">{c.roc.note}</EmptyState>}
        </Panel>
      </div>
      <Panel title="Per-class metrics" flush actions={<Prov kind="test" />}>
        <div className="table-wrap">
          <table className="table">
            <thead><tr><th>Class</th><th className="r">Precision</th><th className="r">Recall</th><th className="r">F1</th><th className="r">Support (test)</th><th className="r desktop-only">Train rows</th></tr></thead>
            <tbody>
              {c.per_class.map((p) => (
                <tr key={p.label}>
                  <td style={{ fontWeight: 550 }}>{p.label}</td>
                  <td className="r num">{fmtNum(p.precision, 3)}</td>
                  <td className="r num" style={{ color: p.recall < 0.5 ? 'var(--bad)' : undefined }}>{fmtNum(p.recall, 3)}</td>
                  <td className="r num">{fmtNum(p.f1, 3)}</td>
                  <td className="r num">{fmtInt(p.support)}</td>
                  <td className="r num subtle desktop-only">{fmtInt(c.train_class_counts[p.label] || 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}

function RegressionView({ res }) {
  const r = res.regression;
  const pts = r.points.map((p) => ({ ...p, x: p.actual, y: p.predicted }));
  const resid = r.points.map((p) => ({ ...p, x: p.predicted, y: p.residual }));
  return (
    <div className="stack-lg">
      <div className="grid grid-2">
        <Panel title="Actual vs predicted" actions={<Prov kind="test" />} footer={r.points_shown < r.points_total ? `Evenly sampled ${r.points_shown} of ${r.points_total} test rows.` : `${r.points_total} test rows. Points on the diagonal are perfect predictions.`}>
          <ScatterPlot points={pts} xLabel="Actual" yLabel="Predicted" diagonal height={300}
            tooltipRows={(p) => [['Actual', fmtNum(p.actual)], ['Predicted', fmtNum(p.predicted)], ['Residual', fmtNum(p.residual)]]} />
        </Panel>
        <Panel title="Residuals vs predicted" actions={<Prov kind="test" />} footer="Look for funnels (heteroscedasticity) or curves (missed non-linearity) around the zero line.">
          <ScatterPlot points={resid} xLabel="Predicted" yLabel="Residual (actual − predicted)" zeroLine height={300}
            tooltipRows={(p) => [['Predicted', fmtNum(p.predicted)], ['Residual', fmtNum(p.residual)]]} />
        </Panel>
      </div>
      <Panel title="Residual distribution" actions={<span className="small num subtle">mean {fmtNum(r.residual_mean, 3)} · sd {fmtNum(r.residual_std, 3)}</span>}>
        <Histogram edges={r.residual_hist.edges} counts={r.residual_hist.counts} height={180} xLabel="residual" />
      </Panel>
    </div>
  );
}

function ErrorsView({ res }) {
  const e = res.errors;
  if (!e?.rows?.length) return <Panel><EmptyState title={e?.kind === 'misclassified' ? 'No misclassified test rows' : 'No errors to show'}>Every held-out row was predicted correctly. Check the test-set size before trusting this.</EmptyState></Panel>;
  const isClf = e.kind === 'misclassified';
  return (
    <Panel flush title={isClf ? `Misclassified test rows (${fmtInt(e.total)})` : 'Largest test errors'} actions={<Prov kind="test" />}
      footer={isClf ? `Showing ${e.rows.length} of ${e.total}. Row numbers refer to the original dataset.` : `Top ${e.rows.length} rows by absolute residual.`}>
      <div className="table-wrap" style={{ maxHeight: 460 }}>
        <table className="table">
          <thead><tr><th>Row</th><th>Actual</th><th>Predicted</th><th className="r">{isClf ? 'Confidence' : 'Residual'}</th>{e.columns.map((c) => <th key={c} className="desktop-only">{c}</th>)}</tr></thead>
          <tbody>
            {e.rows.map((r) => (
              <tr key={r.row}>
                <td className="num subtle">{r.row}</td>
                <td>{isClf ? r.actual : <span className="num">{fmtNum(r.actual)}</span>}</td>
                <td style={{ color: 'var(--bad)' }}>{isClf ? r.predicted : <span className="num">{fmtNum(r.predicted)}</span>}</td>
                <td className="r num">{isClf ? (r.confidence !== null ? fmtPct(r.confidence) : '—') : fmtNum(r.residual)}</td>
                {e.columns.map((c) => <td key={c} className="num small desktop-only">{r.features[c] === null ? <span className="subtle">null</span> : typeof r.features[c] === 'number' ? fmtNum(r.features[c]) : String(r.features[c])}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function FeaturesView({ res }) {
  const { permutation, native } = res.importance || {};
  if (!permutation && !native) return <Panel><EmptyState title="No feature importance">Permutation importance was disabled and this model has no native importances.</EmptyState></Panel>;
  return (
    <div className="grid grid-2">
      {permutation && (
        <Panel title="Permutation importance" actions={<Prov kind="test" />} footer={permutation.method + '. Bars show the mean drop; the thin line is ±1 sd across repeats. Negative values mean shuffling the feature did not hurt.'}>
          <BarList items={permutation.features.slice(0, 20).map((f, i) => ({ label: f, value: permutation.values[i], err: permutation.std[i] }))} diverging format={(v) => fmtNum(v, 4)} />
        </Panel>
      )}
      {native && (
        <Panel title="Model-native importance" actions={<Prov kind="calc" label="Model" />} footer={`${native.method}. Computed on encoded features (after one-hot), so one source column may appear several times.`}>
          <BarList items={native.features.slice(0, 20).map((f, i) => ({ label: f.replace(/^(num|cat)__/, ''), value: native.values[i] }))} format={(v) => fmtNum(v, 4)} />
        </Panel>
      )}
    </div>
  );
}

function ValidationView({ res }) {
  const cv = res.metrics.cv;
  const key = res.task === 'classification' ? 'accuracy' : 'r2';
  return (
    <div className="stack-lg">
      <div className="grid grid-2">
        <Panel title="Cross-validation folds" actions={<Prov kind="cv" />} footer={cv ? cv.scheme : 'Cross-validation was disabled for this run.'}>
          {cv ? (
            <>
              <MiniBars data={cv.folds.map((f) => ({ fold: `F${f.fold}`, [key]: f[key] }))} dataKey={key} labelKey="fold" refLine={res.metrics.test[key]} />
              <div className="small muted" style={{ marginTop: 6 }}>Mean {METRICS[key].label} {fmtNum(cv.mean[key], 4)} ± {fmtNum(cv.std[key], 4)}. The horizontal line is the held-out test score.</div>
            </>
          ) : <EmptyState title="No cross-validation" />}
        </Panel>
        <Panel title="Learning curve" actions={<Prov kind="cv" />}>
          {res.learning_curve ? <LearningCurveChart lc={res.learning_curve} /> : <EmptyState title="Not computed">Enable “Learning curve” in Model Studio to see how performance scales with training data.</EmptyState>}
        </Panel>
      </div>
    </div>
  );
}

function PredictPlayground({ experiment }) {
  const res = experiment.results;
  const cols = [...res.data.features.numeric, ...res.data.features.categorical];
  const sample = res.errors?.rows?.[0]?.features || {};
  const [values, setValues] = useState(() => Object.fromEntries(cols.map((c) => [c, sample[c] ?? ''])));
  const [out, setOut] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const run = async () => {
    setBusy(true); setError(null);
    try {
      const row = Object.fromEntries(cols.map((c) => [c, values[c] === '' ? null : values[c]]));
      setOut((await api.predict(experiment.id, [row])).predictions[0]);
    } catch (e) { setError(e); } finally { setBusy(false); }
  };
  return (
    <Panel title="Try a prediction" icon={Play} actions={<Button size="sm" variant="primary" icon={Play} onClick={run} loading={busy}>Predict</Button>}
      footer="Runs the saved pipeline (preprocessing + model). Empty fields are treated as missing and imputed.">
      <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 10 }}>
        {cols.slice(0, 30).map((c) => (
          <Field key={c} label={<span className="truncate" title={c}>{c}</span>} htmlFor={`pp-${c}`}>
            <input id={`pp-${c}`} className="input num" value={values[c] ?? ''} onChange={(e) => setValues((v) => ({ ...v, [c]: e.target.value }))} />
          </Field>
        ))}
      </div>
      {error && <div style={{ marginTop: 10 }}><ErrorState error={error} /></div>}
      {out && (
        <div className="callout" style={{ marginTop: 12 }}>
          <Sigma size={15} />
          <div className="grow">
            <div>Prediction: <b className="num" style={{ fontSize: 15 }}>{typeof out.prediction === 'number' ? fmtNum(out.prediction, 4) : out.prediction}</b></div>
            {out.probabilities && <div className="small muted">{Object.entries(out.probabilities).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k}: ${fmtPct(v)}`).join(' · ')}</div>}
          </div>
        </div>
      )}
    </Panel>
  );
}

function ConfigView({ experiment }) {
  const res = experiment.results;
  const r = res.reproducibility;
  return (
    <div className="stack-lg">
      <Panel title="Preprocessing pipeline (as executed)">
        <ol style={{ margin: 0, paddingLeft: 18 }} className="stack-sm small">
          {res.pipeline_stages.map((s) => <li key={s.stage}><b>{s.stage}</b> <span className="subtle">· fitted on {s.fitted_on}</span><div className="muted">{s.detail}</div></li>)}
        </ol>
      </Panel>
      <div className="grid grid-3">
        <Panel title="Hyperparameters">
          {Object.keys(res.algorithm.hyperparameters).length ? (
            <dl className="kv">{Object.entries(res.algorithm.hyperparameters).map(([k, v]) => <div key={k} style={{ display: 'contents' }}><dt className="mono small">{k}</dt><dd className="num">{v === null ? 'None' : String(v)}</dd></div>)}</dl>
          ) : <span className="small subtle">No tunable hyperparameters.</span>}
        </Panel>
        <Panel title="Data & split">
          <dl className="kv">
            <dt>Rows used</dt><dd className="num">{fmtInt(res.data.n_used)} of {fmtInt(res.data.n_raw)}</dd>
            <dt>Dropped</dt><dd className="small">{res.data.dropped_missing_target} no target · {res.data.dropped_duplicates} dup · {res.data.dropped_rare_classes} rare</dd>
            <dt>Train / test</dt><dd className="num">{fmtInt(res.data.n_train)} / {fmtInt(res.data.n_test)}</dd>
            <dt>Stratified</dt><dd>{res.split.stratified ? 'yes' : 'no'}</dd>
            <dt>Split id</dt><dd className="mono small">{res.split.split_id}</dd>
            <dt>Encoded features</dt><dd className="num">{fmtInt(res.data.encoded_features)}</dd>
            {res.data.features.excluded.length > 0 && <><dt>Excluded</dt><dd className="small">{res.data.features.excluded.join(', ')}</dd></>}
          </dl>
        </Panel>
        <Panel title="Model & timing">
          <dl className="kv">
            {Object.entries(res.model_info).filter(([, v]) => v !== null).map(([k, v]) => <div key={k} style={{ display: 'contents' }}><dt className="small">{k.replace(/_/g, ' ')}</dt><dd className="num">{typeof v === 'number' ? fmtNum(v, 2) : v}</dd></div>)}
            {['cv', 'fit', 'predict_test', 'total'].map((k) => res.timing[k] !== undefined && <div key={k} style={{ display: 'contents' }}><dt className="small">time · {k.replace('_', ' ')}</dt><dd className="num">{fmtDuration(res.timing[k])}</dd></div>)}
          </dl>
        </Panel>
      </div>
      <Panel title="Reproducibility" actions={<Prov kind="calc" />}>
        <dl className="kv">
          <dt>Dataset SHA-256</dt><dd className="mono small">{r.dataset_sha256}</dd>
          <dt>Random seed</dt><dd className="num">{r.random_state}</dd>
          <dt>Environment</dt><dd className="small">Python {r.python} · scikit-learn {r.scikit_learn} · pandas {r.pandas} · numpy {r.numpy}</dd>
        </dl>
      </Panel>
      <PredictPlayground experiment={experiment} />
    </div>
  );
}

export default function Evaluation({ experiment, report }) {
  const res = experiment.results;
  const [tab, setTab] = useState('summary');
  const isClf = res.task === 'classification';
  const tabs = useMemo(() => [
    { value: 'summary', label: 'Summary', icon: Gauge },
    { value: 'detail', label: isClf ? 'Classes & ROC' : 'Residuals', icon: BarChart3 },
    { value: 'errors', label: 'Errors', icon: Bug, count: res.errors?.kind === 'misclassified' ? res.errors.total : null },
    { value: 'features', label: 'Features', icon: Sigma },
    { value: 'validation', label: 'Validation', icon: Activity },
    { value: 'config', label: 'Configuration', icon: Settings2 },
  ], [isClf, res.errors]);
  return (
    <div>
      <Tabs tabs={tabs} value={tab} onChange={setTab} id="eval" />
      {tab === 'summary' && (
        <div className="stack-lg">
          <div className="row wrap" style={{ gap: 8 }}>
            <Prov kind="test" />
            <span className="small muted">All headline metrics are measured on <b>{fmtInt(res.data.n_test)}</b> held-out test rows ({Math.round(res.split.test_size * 100)}%), never used for fitting. Baseline: {res.metrics.baseline.strategy.toLowerCase()}.</span>
          </div>
          <MetricGrid res={res} />
          <div className="grid grid-2">
            <Panel title="Test vs cross-validation vs train" flush footer="A large train–test gap signals overfitting. CV variance shows how much the score depends on which rows are held out.">
              <SplitComparison res={res} />
            </Panel>
            <Panel title="Observations" actions={<Prov kind="rule" />} footer="Generated by fixed rules from the numbers on this page — not by an LLM.">
              {report ? <Observations report={report} /> : <span className="small subtle">Loading…</span>}
            </Panel>
          </div>
          {res.warnings?.length > 0 && <Callout tone="warn" title="Run warnings:">{res.warnings.join(' · ')}</Callout>}
        </div>
      )}
      {tab === 'detail' && (isClf ? <ClassificationView res={res} /> : <RegressionView res={res} />)}
      {tab === 'errors' && <ErrorsView res={res} />}
      {tab === 'features' && <FeaturesView res={res} />}
      {tab === 'validation' && <ValidationView res={res} />}
      {tab === 'config' && <ConfigView experiment={experiment} />}
    </div>
  );
}
