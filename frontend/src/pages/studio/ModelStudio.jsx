import { AlertTriangle, Database, FlaskConical, Play } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import Dropzone from '../../components/Dropzone';
import { BarList } from '../../components/charts';
import { Badge, Button, Callout, Disclaimer, EmptyState, ErrorState, Field, PageHeader, Prov, Segmented, SeverityIcon, SkeletonBlock } from '../../components/ui';
import { api } from '../../lib/api';
import { fmtInt, fmtPct, titleCase } from '../../lib/format';
import { useAsync } from '../../lib/hooks';
import { useToast } from '../../lib/toast';
import { useWorkspace } from '../../lib/workspace';
import { SampleDatasets } from '../Datasets';
import {
  AiExplanation, AlgorithmCard, coerceParams, EXCLUDED_KINDS, FeaturePicker, ParamField, paramsValid, PipelineDiagram,
  PreprocessingForm, ReviewRow, Section, ValidationForm,
} from './sections';

const STEPS = [
  ['s-dataset', 'Dataset'], ['s-target', 'Target & problem'], ['s-quality', 'Data quality'], ['s-prep', 'Preprocessing'],
  ['s-algos', 'Algorithms'], ['s-params', 'Hyperparameters'], ['s-validation', 'Validation'], ['s-review', 'Review & train'],
];

const DEFAULT_PREP = {
  numeric_imputation: 'median', categorical_imputation: 'constant', encoding: 'onehot', scaling: 'auto', outliers: 'none',
  feature_selection: 'none', k_features: 10, drop_duplicates: true, drop_rare_classes: false,
};

export default function ModelStudio() {
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const toast = useToast();
  const { datasets, system, refreshExperiments, setActiveDatasetId, activeDatasetId } = useWorkspace();
  const datasetId = params.get('dataset') || '';
  const target = params.get('target') || '';
  const [taskOverride, setTaskOverride] = useState(null);
  const [excluded, setExcluded] = useState(null);
  const [prep, setPrep] = useState(DEFAULT_PREP);
  const [selected, setSelected] = useState([]);
  const [hyper, setHyper] = useState({});
  const [split, setSplit] = useState({ test_size: 0.2, random_state: 42, stratify: true, cv_folds: 5 });
  const [options, setOptions] = useState({ permutation_importance: true, learning_curve: false });
  const [name, setName] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);
  const [aiState, setAiState] = useState(null);
  const [activeStep, setActiveStep] = useState(STEPS[0][0]);
  const submitLock = useRef(false);

  // default to the active dataset
  useEffect(() => {
    if (!datasetId && activeDatasetId && datasets?.some((d) => d.id === activeDatasetId)) {
      setParams((p) => { const n = new URLSearchParams(p); n.set('dataset', activeDatasetId); return n; }, { replace: true });
    }
  }, [datasetId, activeDatasetId, datasets, setParams]);

  const setQuery = (k, v) => setParams((p) => { const n = new URLSearchParams(p); if (v) n.set(k, v); else n.delete(k); if (k === 'dataset') n.delete('target'); return n; }, { replace: true });

  const profile = useAsync(() => api.profile(datasetId), [datasetId], { enabled: !!datasetId });
  const algorithms = useAsync(() => api.algorithms(), []);
  const analysis = useAsync(() => api.target(datasetId, target, taskOverride || undefined), [datasetId, target, taskOverride], { enabled: !!(datasetId && target) });
  const algById = useMemo(() => Object.fromEntries((algorithms.data?.algorithms || []).map((a) => [a.id, a])), [algorithms.data]);

  const columns = useMemo(() => profile.data?.overview?.column_info || [], [profile.data]);
  const dataset = datasets?.find((d) => d.id === datasetId);
  const a = analysis.data?.analysis;
  const rec = analysis.data?.recommendation;
  const task = taskOverride || a?.problem_type;

  // reset dependent state when inputs change
  useEffect(() => { setTaskOverride(null); setSelected([]); setAiState(null); }, [datasetId, target]);
  useEffect(() => { setSelected([]); setAiState(null); }, [taskOverride]);
  useEffect(() => {
    if (!columns.length) return;
    setExcluded(columns.filter((c) => c.name !== target && EXCLUDED_KINDS.has(c.kind)).map((c) => c.name));
  }, [columns, target]);
  useEffect(() => { if (datasetId) setActiveDatasetId(datasetId); }, [datasetId, setActiveDatasetId]);
  // preselect the rule-based suggestions (user can change)
  useEffect(() => { if (rec?.suggested?.length && selected.length === 0) setSelected(rec.suggested.slice(0, 2)); }, [rec]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    setHyper((h) => {
      const next = { ...h };
      selected.forEach((id) => { if (!next[id] && algById[id]) next[id] = Object.fromEntries(algById[id].params.map((p) => [p.name, p.default ?? ''])); });
      return next;
    });
  }, [selected, algById]);

  // scroll-spy for the step rail
  useEffect(() => {
    const root = document.getElementById('main');
    if (!root) return undefined;
    const obs = new IntersectionObserver((entries) => {
      const visible = entries.filter((e) => e.isIntersecting).sort((x, y) => x.boundingClientRect.top - y.boundingClientRect.top)[0];
      if (visible) setActiveStep(visible.target.id);
    }, { root, rootMargin: '-10% 0px -70% 0px' });
    STEPS.forEach(([id]) => { const el = document.getElementById(id); if (el) obs.observe(el); });
    return () => obs.disconnect();
  }, [datasetId, target, task]);

  const features = columns.filter((c) => c.name !== target && !(excluded || []).includes(c.name));
  const nNumeric = features.filter((c) => c.kind === 'numeric' || c.kind === 'boolean').length;
  const nCategorical = features.length - nNumeric;
  const anyScaleSensitive = selected.some((id) => algById[id]?.scale_sensitive);
  const quality = (profile.data?.quality || []).filter((q) => q.severity !== 'info' && !(q.columns || []).every((c) => (excluded || []).includes(c)));
  const targetFindings = a?.findings || [];
  const critical = targetFindings.some((f) => f.severity === 'critical');
  const invalidParams = selected.filter((id) => !paramsValid(algById[id], hyper[id]));

  const done = {
    's-dataset': !!dataset, 's-target': !!(target && task), 's-quality': !!a, 's-prep': features.length > 0,
    's-algos': selected.length > 0, 's-params': selected.length > 0 && invalidParams.length === 0, 's-validation': true, 's-review': false,
  };
  const blockers = [
    !dataset && 'Choose a dataset',
    dataset && !target && 'Choose a target column',
    target && a && !a.problem_type && !taskOverride && 'The target cannot be learned',
    critical && 'Resolve critical target issues',
    !features.length && 'Include at least one feature',
    !selected.length && 'Select at least one algorithm',
    invalidParams.length > 0 && `Fix hyperparameters for ${invalidParams.map((id) => algById[id]?.name).join(', ')}`,
  ].filter(Boolean);

  const submit = async () => {
    if (submitLock.current || blockers.length) return;
    submitLock.current = true;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const body = {
        dataset_id: datasetId, target, task_type: task, name: name.trim() || undefined,
        algorithms: selected.map((id) => ({ id, hyperparameters: coerceParams(algById[id], hyper[id]) })),
        preprocessing: { ...prep, excluded_columns: excluded },
        split: { ...split, stratify: task === 'classification' ? split.stratify : false },
        options,
      };
      const res = await api.createExperiments(body);
      await refreshExperiments();
      toast.success(res.experiments.length > 1 ? `${res.experiments.length} experiments started` : 'Training started', 'Follow progress live on the experiment page.');
      if (res.experiments.length === 1) navigate(`/experiments/${res.experiments[0].id}`);
      else navigate(`/compare?ids=${res.experiments.map((e) => e.id).join(',')}`);
    } catch (e) {
      setSubmitError(e);
      if (e.status === 409 && e.body?.experiment_id) toast.info('Already running', 'An identical experiment is in progress.');
    } finally {
      setSubmitting(false);
      submitLock.current = false;
    }
  };

  const explain = async () => {
    setAiState({ loading: true });
    try {
      const data = await api.explainRecommendations({ dataset_id: datasetId, target, task_type: task });
      setAiState({ data });
    } catch (e) { setAiState({ error: e.message }); }
  };

  if (datasets && datasets.length === 0) {
    return (
      <div className="page page-narrow">
        <PageHeader eyebrow="Experimentation" title="Model Studio" lede="Design a reproducible experiment: dataset, target, preprocessing, algorithms and validation." />
        <div className="grid grid-2">
          <div className="panel" style={{ padding: 18 }}><h2 style={{ marginBottom: 10 }}>Upload a dataset to begin</h2><Dropzone onUploaded={(m) => setQuery('dataset', m.id)} /></div>
          <div className="panel" style={{ padding: 18 }}><h2 style={{ marginBottom: 10 }}>…or start with a sample</h2><SampleDatasets compact /></div>
        </div>
      </div>
    );
  }

  const groups = rec ? Object.entries(rec.algorithms.reduce((acc, r) => { (acc[r.family] = acc[r.family] || []).push(r); return acc; }, {})) : [];

  return (
    <div className="page">
      <PageHeader eyebrow="Experimentation" title="Model Studio"
        lede="Configure an experiment step by step. Everything that learns from data is fitted on the training split only."
        actions={<Link className="btn" to="/experiments">Experiments</Link>} />
      <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 1fr)', gap: 16 }}>
        <div className="grid grid-main">
          <div className="stack-lg">
            {/* 01 dataset */}
            <Section id="s-dataset" num={1} title="Dataset" subtitle="The file version is pinned by its SHA-256 hash in the experiment record.">
              {datasets === null ? <SkeletonBlock /> : (
                <div className="row wrap" style={{ gap: 12, alignItems: 'flex-end' }}>
                  <div style={{ width: 'min(360px, 100%)' }}>
                    <Field label="Dataset" htmlFor="st-ds">
                      <select id="st-ds" className="select" value={datasetId} onChange={(e) => setQuery('dataset', e.target.value)}>
                        <option value="">Choose a dataset…</option>
                        {datasets.map((d) => <option key={d.id} value={d.id}>{d.name} — {fmtInt(d.n_rows)} × {d.n_cols}</option>)}
                      </select>
                    </Field>
                  </div>
                  {dataset && <Link className="btn btn-ghost" to={`/datasets/${dataset.id}`}><Database size={14} />Explore</Link>}
                </div>
              )}
              {profile.error && <div style={{ marginTop: 10 }}><ErrorState error={profile.error} /></div>}
            </Section>

            {/* 02 target */}
            <Section id="s-target" num={2} title="Target & problem type" subtitle="The problem type is detected by fixed rules from the target's values; you can override it.">
              {!datasetId ? <p className="small subtle">Choose a dataset first.</p> : !profile.data ? <SkeletonBlock /> : (
                <div className="stack">
                  <div className="row wrap" style={{ gap: 16, alignItems: 'flex-end' }}>
                    <div style={{ width: 'min(320px, 100%)' }}>
                      <Field label="Target column" htmlFor="st-t">
                        <select id="st-t" className="select" value={target} onChange={(e) => setQuery('target', e.target.value)}>
                          <option value="">Choose the column to predict…</option>
                          {columns.filter((c) => c.kind !== 'empty' && c.kind !== 'constant').map((c) => <option key={c.name} value={c.name}>{c.name} ({c.kind})</option>)}
                        </select>
                      </Field>
                    </div>
                    {a && (
                      <Field label="Problem type">
                        <Segmented ariaLabel="Problem type" value={task || ''} onChange={(v) => setTaskOverride(v === a.problem_type ? null : v)}
                          options={[{ value: 'classification', label: 'Classification' }, { value: 'regression', label: 'Regression' }]} />
                      </Field>
                    )}
                  </div>
                  {analysis.loading && <SkeletonBlock rows={2} />}
                  {analysis.error && <ErrorState error={analysis.error} />}
                  {a && (
                    <div className="row wrap" style={{ gap: 8 }}>
                      <Prov kind="rule" label="Detected" />
                      <span className="small">{a.problem_type ? <>Detected <b>{a.problem_type}</b>: {a.reason}</> : a.reason}</span>
                      {taskOverride && <Badge tone="warn">Overridden by you</Badge>}
                    </div>
                  )}
                  {a?.class_distribution && task === 'classification' && (
                    <div style={{ maxWidth: 520 }}>
                      <BarList items={a.class_distribution.slice(0, 8).map((c) => ({ label: c.label, value: c.count, sub: `· ${fmtPct(c.count / (a.n_rows - a.n_missing))}` }))} format={fmtInt} />
                    </div>
                  )}
                </div>
              )}
            </Section>

            {/* 03 quality */}
            <Section id="s-quality" num={3} title="Data quality review" subtitle="Issues that can mislead training or evaluation. Fix them in the Data Explorer, which creates a new version."
              aside={dataset && <Link className="btn btn-sm" to={`/datasets/${datasetId}?tab=quality`}>Open quality report</Link>}>
              {!a ? <p className="small subtle">Choose a target to run the checks.</p> : (targetFindings.length + quality.length) === 0 ? (
                <Callout tone="good">No warnings for this dataset and target.</Callout>
              ) : (
                <div className="stack-sm">
                  {[...targetFindings, ...quality].slice(0, 8).map((f) => (
                    <div key={f.id} className="row small" style={{ alignItems: 'flex-start', gap: 8 }}>
                      <SeverityIcon severity={f.severity} size={14} />
                      <span><b>{f.title}</b> <span className="muted">— {f.detail}</span></span>
                    </div>
                  ))}
                </div>
              )}
            </Section>

            {/* 04 preprocessing */}
            <Section id="s-prep" num={4} title="Preprocessing" subtitle="What happens to your data, in order.">
              {!profile.data || !target ? <p className="small subtle">Choose a dataset and target first.</p> : (
                <div className="stack-lg">
                  <FeaturePicker columns={columns} target={target} excluded={excluded || []} onChange={setExcluded} />
                  <PreprocessingForm value={prep} onChange={setPrep} anyScaleSensitive={anyScaleSensitive} />
                  <PipelineDiagram prep={prep} nNumeric={nNumeric} nCategorical={nCategorical} anyScaleSensitive={anyScaleSensitive}
                    algorithms={selected.map((id) => algById[id]?.name).filter(Boolean)} />
                </div>
              )}
            </Section>

            {/* 05 algorithms */}
            <Section id="s-algos" num={5} title="Algorithms" subtitle="Only models that ML-Agent actually implements are listed. Select several to compare them on the identical split."
              aside={rec && <Prov kind="rule" label="Rule-ranked" />}>
              {!rec ? <p className="small subtle">{target ? 'Analysing…' : 'Choose a target first.'}</p> : (
                <div className="stack-lg">
                  {groups.map(([family, list]) => (
                    <div key={family}>
                      <div className="eyebrow" style={{ marginBottom: 8 }}>{family}</div>
                      <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: 10 }}>
                        {list.map((r) => (
                          <AlgorithmCard key={r.id} rec={r} alg={algById[r.id]} selected={selected.includes(r.id)}
                            onToggle={() => setSelected((s) => (s.includes(r.id) ? s.filter((x) => x !== r.id) : s.length >= 12 ? s : [...s, r.id]))} />
                        ))}
                      </div>
                    </div>
                  ))}
                  <p className="xs subtle">{rec.method}</p>
                  <AiExplanation state={aiState} onRequest={explain} available={!!system?.llm?.available} />
                </div>
              )}
            </Section>

            {/* 06 hyperparameters */}
            <Section id="s-params" num={6} title="Hyperparameters" subtitle="Defaults are sensible starting points. Values are validated before training starts.">
              {!selected.length ? <p className="small subtle">Select at least one algorithm.</p> : (
                <div className="stack-lg">
                  {selected.map((id) => {
                    const alg = algById[id];
                    if (!alg) return null;
                    return (
                      <div key={id} className="panel inset" style={{ padding: 14 }}>
                        <div className="row-between" style={{ marginBottom: alg.params.length ? 10 : 0 }}>
                          <b>{alg.name}</b>
                          {alg.params.length > 0 && <Button size="sm" variant="ghost" onClick={() => setHyper((h) => ({ ...h, [id]: Object.fromEntries(alg.params.map((p) => [p.name, p.default ?? ''])) }))}>Reset</Button>}
                        </div>
                        {alg.params.length === 0 ? <span className="small subtle"> — no tunable hyperparameters.</span> : (
                          <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 12 }}>
                            {alg.params.map((p) => (
                              <ParamField key={p.name} p={p} value={hyper[id]?.[p.name]} onChange={(v) => setHyper((h) => ({ ...h, [id]: { ...h[id], [p.name]: v } }))} />
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </Section>

            {/* 07 validation */}
            <Section id="s-validation" num={7} title="Validation strategy" subtitle="One stratified hold-out test set, plus optional cross-validation on the training rows.">
              <ValidationForm split={split} onSplit={setSplit} options={options} onOptions={setOptions} task={task} />
            </Section>

            {/* 08 review */}
            <Section id="s-review" num={8} title="Review & train">
              <div className="stack">
                <div>
                  <ReviewRow label="Dataset" ok={!!dataset}>{dataset ? `${dataset.name} · ${fmtInt(dataset.n_rows)} rows` : 'not selected'}</ReviewRow>
                  <ReviewRow label="Target" ok={!!(target && task)}>{target ? `${target} · ${task || 'unknown'}` : 'not selected'}</ReviewRow>
                  <ReviewRow label="Features" ok={features.length > 0}>{features.length} used ({nNumeric} numeric, {nCategorical} categorical){excluded?.length ? ` · ${excluded.length} excluded` : ''}</ReviewRow>
                  <ReviewRow label="Algorithms" ok={selected.length > 0 && !invalidParams.length}>{selected.map((id) => algById[id]?.name).join(', ') || 'none'}</ReviewRow>
                  <ReviewRow label="Evaluation">{Math.round(split.test_size * 100)}% hold-out{task === 'classification' && split.stratify ? ' (stratified)' : ''} · seed {split.random_state} · {split.cv_folds ? `${split.cv_folds}-fold CV` : 'no CV'}</ReviewRow>
                </div>
                <div className="row wrap" style={{ gap: 10, alignItems: 'flex-end' }}>
                  <div style={{ width: 'min(360px, 100%)' }}>
                    <Field label="Experiment name (optional)" htmlFor="st-name">
                      <input id="st-name" className="input" maxLength={120} value={name} placeholder={dataset && target ? `${dataset.name} · ${target}` : 'Untitled experiment'} onChange={(e) => setName(e.target.value)} />
                    </Field>
                  </div>
                  <Button variant="primary" size="lg" icon={selected.length > 1 ? FlaskConical : Play} onClick={submit} loading={submitting} disabled={blockers.length > 0}>
                    {selected.length > 1 ? `Train ${selected.length} models` : 'Train model'}
                  </Button>
                </div>
                {blockers.length > 0 && <div className="small" style={{ color: 'var(--warn)' }}><AlertTriangle size={13} style={{ verticalAlign: -2 }} /> {blockers.join(' · ')}</div>}
                {submitError && <ErrorState error={submitError} title={submitError.status === 409 ? 'Duplicate experiment' : 'Could not start training'} />}
                <Disclaimer />
              </div>
            </Section>
          </div>

          {/* step rail */}
          <aside className="desktop-only" style={{ position: 'sticky', top: 0, alignSelf: 'start' }}>
            <div className="panel" style={{ padding: 10 }}>
              <div className="eyebrow" style={{ padding: '4px 10px 8px' }}>Workflow</div>
              <nav className="steps" aria-label="Model Studio steps">
                {STEPS.map(([id, label], i) => (
                  <button key={id} className={`step ${activeStep === id ? 'active' : ''} ${done[id] ? 'done' : ''}`}
                    onClick={() => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })}>
                    <span className="step-num">{done[id] ? '✓' : i + 1}</span>
                    <span>{label}</span>
                  </button>
                ))}
              </nav>
              <hr className="sep" style={{ margin: '10px 0' }} />
              <div className="stack-sm" style={{ padding: '0 10px 6px' }}>
                <div className="small muted">{selected.length ? `${selected.length} model${selected.length > 1 ? 's' : ''} selected` : 'No models selected'}</div>
                <Button variant="primary" icon={Play} onClick={submit} loading={submitting} disabled={blockers.length > 0}>Train</Button>
                {task && <div className="xs subtle">{titleCase(task)} · {target}</div>}
              </div>
            </div>
          </aside>
        </div>
      </div>
      {datasetId && profile.data === undefined && !profile.loading && !profile.error && <EmptyState title="Loading dataset…" />}
    </div>
  );
}
