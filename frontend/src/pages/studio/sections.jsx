import { AnimatePresence, motion } from 'framer-motion';
import { AlertTriangle, ArrowRight, Check, ChevronDown, Sparkles } from 'lucide-react';
import { useState } from 'react';
import { Badge, Button, Callout, Field, Prov, Segmented, Switch, TypeChip } from '../../components/ui';
import { fmtNum } from '../../lib/format';

export const EXCLUDED_KINDS = new Set(['identifier', 'text', 'datetime', 'constant', 'empty']);

export function Section({ id, num, title, subtitle, children, aside }) {
  return (
    <section id={id} className="section panel" style={{ padding: 18 }} aria-labelledby={`${id}-title`}>
      <div className="row-between" style={{ marginBottom: 12, alignItems: 'flex-start' }}>
        <div>
          <div className="section-head" style={{ marginBottom: 2 }}>
            <span className="section-num">{String(num).padStart(2, '0')}</span>
            <h2 id={`${id}-title`}>{title}</h2>
          </div>
          {subtitle && <p className="small muted">{subtitle}</p>}
        </div>
        {aside}
      </div>
      {children}
    </section>
  );
}

// ---------------------------------------------------------------- features
export function FeaturePicker({ columns, target, excluded, onChange }) {
  const [open, setOpen] = useState(false);
  const usable = columns.filter((c) => c.name !== target);
  const included = usable.filter((c) => !excluded.includes(c.name));
  const toggle = (name) => onChange(excluded.includes(name) ? excluded.filter((x) => x !== name) : [...excluded, name]);
  return (
    <div className="panel inset" style={{ padding: 0 }}>
      <button className="list-item" style={{ borderTop: 0 }} onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="grow"><b>{included.length}</b> of {usable.length} columns used as features
          {excluded.length > 0 && <span className="subtle"> · excluded: {excluded.slice(0, 4).join(', ')}{excluded.length > 4 ? '…' : ''}</span>}</span>
        <ChevronDown size={15} style={{ transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 160ms' }} />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} style={{ overflow: 'hidden' }}>
            <div style={{ padding: '4px 16px 14px', display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '6px 16px' }}>
              {usable.map((c) => {
                const unsupported = c.kind === 'text' || c.kind === 'datetime' || c.kind === 'empty' || c.kind === 'constant';
                return (
                  <label key={c.name} className="check small" title={unsupported ? `${c.kind} columns are not supported by the tabular pipeline` : c.kind_reason}
                    style={{ opacity: unsupported ? 0.55 : 1, cursor: unsupported ? 'not-allowed' : 'pointer' }}>
                    <input type="checkbox" checked={!excluded.includes(c.name)} disabled={unsupported} onChange={() => toggle(c.name)} />
                    <TypeChip kind={c.kind} /><span className="truncate">{c.name}</span>
                    {c.kind === 'identifier' && <AlertTriangle size={12} color="var(--warn)" />}
                  </label>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ---------------------------------------------------------------- preprocessing
export function PreprocessingForm({ value, onChange, anyScaleSensitive }) {
  const set = (k, v) => onChange({ ...value, [k]: v });
  return (
    <div className="stack">
      <div className="grid grid-3" style={{ gap: 14 }}>
        <Field label="Numeric imputation" hint="Statistic learned from training rows" htmlFor="p-ni">
          <select id="p-ni" className="select" value={value.numeric_imputation} onChange={(e) => set('numeric_imputation', e.target.value)}>
            <option value="median">Median</option><option value="mean">Mean</option><option value="most_frequent">Most frequent</option><option value="zero">Constant 0</option>
          </select>
        </Field>
        <Field label="Categorical imputation" hint="Missing → own category, or the mode" htmlFor="p-ci">
          <select id="p-ci" className="select" value={value.categorical_imputation} onChange={(e) => set('categorical_imputation', e.target.value)}>
            <option value="constant">“Missing” category</option><option value="most_frequent">Most frequent</option>
          </select>
        </Field>
        <Field label="Categorical encoding" hint="One-hot caps at 50 levels per column" htmlFor="p-enc">
          <select id="p-enc" className="select" value={value.encoding} onChange={(e) => set('encoding', e.target.value)}>
            <option value="onehot">One-hot</option><option value="ordinal">Ordinal codes</option>
          </select>
        </Field>
        <Field label="Numeric scaling" hint={value.scaling === 'auto' ? (anyScaleSensitive ? 'Auto: standardise for scale-sensitive models only' : 'Auto: tree models need no scaling') : 'Applied to every selected model'} htmlFor="p-sc">
          <select id="p-sc" className="select" value={value.scaling} onChange={(e) => set('scaling', e.target.value)}>
            <option value="auto">Auto (per model)</option><option value="standard">Standardise (z-score)</option><option value="robust">Robust (median/IQR)</option><option value="minmax">Min–max [0, 1]</option><option value="none">None</option>
          </select>
        </Field>
        <Field label="Outlier handling" hint="Clip to 1.5×IQR fences from training rows" htmlFor="p-out">
          <select id="p-out" className="select" value={value.outliers} onChange={(e) => set('outliers', e.target.value)}>
            <option value="none">Keep as is</option><option value="clip_iqr">Clip (IQR)</option>
          </select>
        </Field>
        <Field label="Feature selection" hint="Scores computed on training rows only" htmlFor="p-fs">
          <div className="row" style={{ gap: 6 }}>
            <select id="p-fs" className="select" value={value.feature_selection} onChange={(e) => set('feature_selection', e.target.value)}>
              <option value="none">None</option><option value="variance">Drop zero-variance</option><option value="kbest">Top-k (F-test)</option>
            </select>
            {value.feature_selection === 'kbest' && (
              <input className="input num" type="number" min={1} style={{ width: 72 }} value={value.k_features} aria-label="k"
                onChange={(e) => set('k_features', Math.max(1, Number(e.target.value) || 1))} />
            )}
          </div>
        </Field>
      </div>
      <div className="row wrap" style={{ gap: 20 }}>
        <Switch checked={value.drop_duplicates} onChange={(v) => set('drop_duplicates', v)} label="Drop duplicate rows before splitting" />
        <Switch checked={value.drop_rare_classes} onChange={(v) => set('drop_rare_classes', v)} label="Drop classes with a single example" />
      </div>
    </div>
  );
}

export function PipelineDiagram({ prep, nNumeric, nCategorical, anyScaleSensitive, algorithms }) {
  const scaling = prep.scaling === 'auto' ? (anyScaleSensitive ? 'auto (z-score where needed)' : 'auto (none for trees)') : prep.scaling;
  const nodes = [
    { label: 'Rows', fit: false, lines: ['drop missing target', prep.drop_duplicates ? 'drop duplicates' : 'keep duplicates', prep.drop_rare_classes ? 'drop singleton classes' : null] },
    { label: 'Split', fit: false, lines: ['hold-out test set', 'test rows never used to fit'] },
    nNumeric ? { label: `Numeric · ${nNumeric}`, fit: true, lines: [`impute ${prep.numeric_imputation}`, prep.outliers === 'clip_iqr' ? 'clip IQR' : null, `scale: ${scaling}`] } : null,
    nCategorical ? { label: `Categorical · ${nCategorical}`, fit: true, lines: [`impute ${prep.categorical_imputation === 'constant' ? '“missing”' : 'mode'}`, prep.encoding === 'onehot' ? 'one-hot (≤50)' : 'ordinal'] } : null,
    prep.feature_selection !== 'none' ? { label: 'Select', fit: true, lines: [prep.feature_selection === 'kbest' ? `top ${prep.k_features} by F-test` : 'drop zero-variance'] } : null,
    { label: 'Model', fit: true, lines: algorithms.length ? [algorithms.length === 1 ? algorithms[0] : `${algorithms.length} models, same split`] : ['choose below'] },
  ].filter(Boolean);
  return (
    <div>
      <div className="pipeline-flow" role="list" aria-label="Preprocessing pipeline">
        {nodes.map((n, i) => (
          <div key={n.label} style={{ display: 'contents' }}>
            {i > 0 && <div className="pipe-arrow" aria-hidden="true"><ArrowRight size={14} /></div>}
            <motion.div role="listitem" className={`pipe-node ${n.fit ? 'fit-train' : ''}`} layout initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.03 }}>
              <span className="eyebrow" style={{ color: n.fit ? 'var(--accent-text)' : undefined }}>{n.fit ? 'fit on train' : 'deterministic'}</span>
              <b style={{ fontSize: 12.5 }}>{n.label}</b>
              {n.lines.filter(Boolean).map((l) => <span key={l} className="subtle">{l}</span>)}
            </motion.div>
          </div>
        ))}
      </div>
      <p className="xs subtle" style={{ marginTop: 8 }}>Blue-edged steps learn parameters. They run inside one scikit-learn Pipeline, fitted on the training split only, then applied unchanged to validation and test rows.</p>
    </div>
  );
}

// ---------------------------------------------------------------- algorithms
export function AlgorithmCard({ rec, alg, selected, onToggle }) {
  const [more, setMore] = useState(false);
  const disabled = !rec.compatible;
  return (
    <div className={`alg-card ${selected ? 'selected' : ''} ${disabled ? 'disabled' : ''}`} role="checkbox" aria-checked={selected} aria-disabled={disabled} tabIndex={0}
      onClick={() => !disabled && onToggle()} onKeyDown={(e) => { if ((e.key === ' ' || e.key === 'Enter') && !disabled) { e.preventDefault(); onToggle(); } }}>
      <div className="row-between" style={{ alignItems: 'flex-start' }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontWeight: 600 }} className="truncate">{rec.name}</div>
          <div className="xs subtle">{rec.family} · cost {rec.cost}</div>
        </div>
        <span style={{ width: 18, height: 18, borderRadius: 5, border: '1px solid var(--border-strong)', display: 'grid', placeItems: 'center', background: selected ? 'var(--accent)' : 'var(--surface)', borderColor: selected ? 'var(--accent)' : undefined, flexShrink: 0 }}>
          {selected && <Check size={12} color="#fff" strokeWidth={3} />}
        </span>
      </div>
      {rec.compatible && (
        <div className="row" style={{ gap: 8 }} title="Heuristic suitability score from dataset properties — not an accuracy prediction">
          <div className="score-bar grow"><div style={{ width: `${rec.score}%` }} /></div>
          <span className="xs num subtle">{rec.score}</span>
        </div>
      )}
      <div className="small muted">{alg?.summary}</div>
      {rec.reasons[0] && <div className="xs" style={{ color: 'var(--text-2)' }}>✓ {rec.reasons[0]}</div>}
      {rec.cautions.map((c) => <div key={c} className="xs" style={{ color: 'var(--warn)' }}>! {c}</div>)}
      {alg && (
        <button className="btn btn-ghost btn-sm" style={{ alignSelf: 'flex-start', marginLeft: -8 }} onClick={(e) => { e.stopPropagation(); setMore(!more); }} aria-expanded={more}>
          {more ? 'Less' : 'Strengths & limits'}
        </button>
      )}
      {more && alg && (
        <div className="grid grid-2 xs" style={{ gap: 8 }} onClick={(e) => e.stopPropagation()}>
          <div><div className="eyebrow">Strengths</div>{alg.strengths.map((s) => <div key={s}>+ {s}</div>)}</div>
          <div><div className="eyebrow">Limitations</div>{alg.limitations.map((s) => <div key={s}>− {s}</div>)}</div>
          {rec.reasons.slice(1).length > 0 && <div className="span-2"><div className="eyebrow">Why it may suit this data</div>{rec.reasons.slice(1).map((s) => <div key={s}>· {s}</div>)}</div>}
        </div>
      )}
    </div>
  );
}

export function AiExplanation({ state, onRequest, available }) {
  if (!available) {
    return <Callout tone="info">AI explanations are off (no LLM configured). Recommendations above are rule-based and fully functional without one.</Callout>;
  }
  if (!state) return <Button size="sm" icon={Sparkles} onClick={onRequest}>Explain these recommendations with AI</Button>;
  if (state.loading) return <Button size="sm" loading>Asking the model…</Button>;
  if (state.error) return <Callout tone="bad" title="AI explanation failed.">{state.error}</Callout>;
  const c = state.data.content;
  return (
    <div className="ai-block stack-sm">
      <div className="row" style={{ gap: 8 }}><Prov kind="ai" /><span className="xs subtle">{state.data.provider} · {state.data.model} · explains the rule-based ranking; does not change it</span></div>
      <p>{c.overview}</p>
      {c.per_algorithm?.map((p) => <div key={p.name} className="small"><b>{p.name}:</b> {p.note}</div>)}
      {c.watch_out?.length > 0 && <div className="small muted">Watch out: {c.watch_out.join(' · ')}</div>}
    </div>
  );
}

// ---------------------------------------------------------------- hyperparameters
export function ParamField({ p, value, onChange }) {
  const id = `hp-${p.name}`;
  if (p.type === 'choice') {
    return (
      <Field label={p.label} hint={p.help} htmlFor={id}>
        <select id={id} className="select" value={value} onChange={(e) => onChange(e.target.value)}>{p.choices.map((c) => <option key={c}>{c}</option>)}</select>
      </Field>
    );
  }
  if (p.type === 'bool') return <Switch checked={!!value} onChange={onChange} label={p.label} />;
  const invalid = value !== '' && value !== null && value !== undefined && (Number.isNaN(Number(value)) || (p.min !== undefined && Number(value) < p.min) || (p.max !== undefined && Number(value) > p.max) || ((p.type === 'int' || p.type === 'int_or_none') && !Number.isInteger(Number(value))));
  return (
    <Field label={p.label} htmlFor={id} error={invalid ? `Must be ${p.type.startsWith('int') ? 'a whole number' : 'a number'} between ${fmtNum(p.min)} and ${fmtNum(p.max)}` : null}
      hint={`${p.help}${p.type === 'int_or_none' ? ' Leave empty for none.' : ''}`}>
      <input id={id} className={`input num ${invalid ? 'invalid' : ''}`} inputMode="decimal" value={value ?? ''} placeholder={p.type === 'int_or_none' ? 'none' : String(p.default)}
        onChange={(e) => onChange(e.target.value)} />
    </Field>
  );
}

export function paramsValid(alg, values) {
  return (alg?.params || []).every((p) => {
    const v = values?.[p.name];
    if (p.type === 'choice' || p.type === 'bool') return true;
    if (v === '' || v === null || v === undefined) return p.type === 'int_or_none';
    const n = Number(v);
    if (Number.isNaN(n)) return false;
    if (p.type.startsWith('int') && !Number.isInteger(n)) return false;
    return (p.min === undefined || n >= p.min) && (p.max === undefined || n <= p.max);
  });
}

export function coerceParams(alg, values) {
  const out = {};
  (alg?.params || []).forEach((p) => {
    const v = values?.[p.name];
    if (p.type === 'choice' || p.type === 'bool') out[p.name] = v;
    else if (v === '' || v === null || v === undefined) out[p.name] = null;
    else out[p.name] = Number(v);
  });
  return out;
}

export function ValidationForm({ split, onSplit, options, onOptions, task }) {
  return (
    <div className="stack">
      <div className="grid grid-4" style={{ gap: 14 }}>
        <Field label="Test split" hint={`${Math.round(split.test_size * 100)}% held out for final evaluation`} htmlFor="v-ts">
          <input id="v-ts" type="range" min={0.1} max={0.4} step={0.05} value={split.test_size} onChange={(e) => onSplit({ ...split, test_size: Number(e.target.value) })} style={{ accentColor: 'var(--accent)' }} />
        </Field>
        <Field label="Cross-validation" hint="On the training split only" htmlFor="v-cv">
          <select id="v-cv" className="select" value={split.cv_folds} onChange={(e) => onSplit({ ...split, cv_folds: Number(e.target.value) })}>
            <option value={0}>Disabled</option>{[3, 5, 10].map((k) => <option key={k} value={k}>{k}-fold{task === 'classification' ? ' (stratified)' : ''}</option>)}
          </select>
        </Field>
        <Field label="Random seed" hint="Same seed + data = same split" htmlFor="v-seed">
          <input id="v-seed" className="input num" type="number" min={0} value={split.random_state} onChange={(e) => onSplit({ ...split, random_state: Math.max(0, Math.floor(Number(e.target.value) || 0)) })} />
        </Field>
        <Field label="Stratify by class" hint={task === 'classification' ? 'Keeps class ratios in both splits' : 'Classification only'}>
          <Segmented ariaLabel="Stratify" value={split.stratify && task === 'classification' ? 'on' : 'off'} onChange={(v) => onSplit({ ...split, stratify: v === 'on' })}
            options={[{ value: 'on', label: 'On' }, { value: 'off', label: 'Off' }]} />
        </Field>
      </div>
      <div className="row wrap" style={{ gap: 20 }}>
        <Switch checked={options.permutation_importance} onChange={(v) => onOptions({ ...options, permutation_importance: v })} label="Permutation importance on test set" />
        <Switch checked={options.learning_curve} onChange={(v) => onOptions({ ...options, learning_curve: v })} label="Learning curve (≈5× CV cost)" />
      </div>
      {split.test_size < 0.15 && <Callout tone="warn">Small test sets give noisy estimates. Keep cross-validation on to see the variance.</Callout>}
    </div>
  );
}

export function ReviewRow({ label, children, ok = true }) {
  return (
    <div className="row" style={{ gap: 10, alignItems: 'flex-start', padding: '6px 0', borderTop: '1px solid var(--hairline)' }}>
      <span style={{ color: ok ? 'var(--good)' : 'var(--warn)', marginTop: 2 }}>{ok ? <Check size={14} /> : <AlertTriangle size={14} />}</span>
      <span className="small" style={{ width: 120, flexShrink: 0, color: 'var(--text-2)' }}>{label}</span>
      <span className="small grow" style={{ minWidth: 0 }}>{children}</span>
    </div>
  );
}

export { Badge };
