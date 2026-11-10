import { AlertTriangle, CheckCircle2, GitCompare, Plus, X } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { BarList } from '../components/charts';
import { Button, Callout, EmptyState, ErrorState, Field, PageHeader, Panel, Prov, SkeletonBlock, StatusBadge } from '../components/ui';
import { api } from '../lib/api';
import { CLASSIFICATION_KEYS, fmtDuration, fmtNum, METRICS, REGRESSION_KEYS, timeAgo } from '../lib/format';
import { isActiveStatus, useWorkspace } from '../lib/workspace';

function useExperiments(ids, activeTick) {
  const [state, setState] = useState({ list: null, error: null });
  useEffect(() => {
    let alive = true;
    if (!ids.length) { setState({ list: [], error: null }); return undefined; }
    Promise.all(ids.map((id) => api.experiment(id).then((d) => d.experiment).catch(() => null)))
      .then((list) => alive && setState({ list: list.filter(Boolean), error: null }))
      .catch((error) => alive && setState({ list: null, error }));
    return () => { alive = false; };
  }, [ids.join(','), activeTick]); // eslint-disable-line react-hooks/exhaustive-deps
  return state;
}

function tradeoffs(list, metric) {
  const def = METRICS[metric];
  const val = (e) => e.results.metrics.test[metric];
  const ranked = [...list].filter((e) => val(e) !== null).sort((a, b) => (def.higher ? val(b) - val(a) : val(a) - val(b)));
  if (ranked.length < 2) return [];
  const best = ranked[0];
  const out = [`Highest held-out ${def.label}: ${best.algorithm} (${fmtNum(val(best), 4)}).`];
  const cvKey = best.task_type === 'classification' ? 'accuracy' : 'r2';
  const bestStd = best.results.metrics.cv?.std?.[cvKey];
  if (bestStd !== undefined && (metric === cvKey)) {
    const close = ranked.slice(1).filter((e) => Math.abs(val(e) - val(best)) <= bestStd);
    if (close.length) out.push(`${close.map((e) => e.algorithm).join(', ')} ${close.length > 1 ? 'are' : 'is'} within one CV standard deviation (±${fmtNum(bestStd, 3)}) — the difference may not be meaningful.`);
  }
  const fastest = [...list].sort((a, b) => a.results.timing.fit - b.results.timing.fit)[0];
  if (fastest.id !== best.id) {
    const ratio = best.results.timing.fit / Math.max(fastest.results.timing.fit, 1e-3);
    if (ratio > 2) out.push(`${best.algorithm} took ${ratio.toFixed(0)}× longer to fit than ${fastest.algorithm} (${fmtDuration(fastest.results.timing.fit)}), which scores ${fmtNum(val(fastest), 4)}.`);
  }
  const gapKey = cvKey;
  const gaps = list.map((e) => ({ e, gap: e.results.metrics.train[gapKey] - e.results.metrics.test[gapKey] })).filter((g) => g.gap > 0.1);
  if (gaps.length) out.push(`Large train–test gap (likely overfitting): ${gaps.map((g) => `${g.e.algorithm} (${fmtNum(g.gap, 3)})`).join(', ')}.`);
  return out;
}

export default function Compare() {
  const [params, setParams] = useSearchParams();
  const ids = useMemo(() => (params.get('ids') || '').split(',').filter(Boolean), [params]);
  const { experiments } = useWorkspace();
  const activeTick = (experiments || []).filter((e) => ids.includes(e.id)).map((e) => e.status).join(',');
  const { list, error } = useExperiments(ids, activeTick);
  const completed = (list || []).filter((e) => e.status === 'completed' && e.results?.metrics);
  const pending = (list || []).filter((e) => isActiveStatus(e.status));
  const failed = (list || []).filter((e) => ['failed', 'cancelled'].includes(e.status));
  const task = completed[0]?.task_type;
  const keys = task === 'regression' ? REGRESSION_KEYS : CLASSIFICATION_KEYS;
  const [metric, setMetric] = useState(null);
  const m = metric && keys.includes(metric) ? metric : keys[0];
  const setIds = (next) => setParams({ ids: next.join(',') }, { replace: true });

  const splitIds = new Set(completed.map((e) => e.results.split.split_id));
  const targets = new Set(completed.map((e) => e.results.target));
  const tasks = new Set(completed.map((e) => e.task_type));
  const candidates = (experiments || []).filter((e) => e.status === 'completed' && !ids.includes(e.id) && (!task || e.task_type === task));

  const sorted = [...completed].sort((a, b) => {
    const va = a.results.metrics.test[m]; const vb = b.results.metrics.test[m];
    return METRICS[m].higher ? (vb ?? -Infinity) - (va ?? -Infinity) : (va ?? Infinity) - (vb ?? Infinity);
  });
  const bestFor = (k) => {
    const vals = completed.map((e) => e.results.metrics.test[k]).filter((v) => v !== null && v !== undefined);
    if (!vals.length) return null;
    return METRICS[k].higher ? Math.max(...vals) : Math.min(...vals);
  };

  return (
    <div className="page">
      <PageHeader eyebrow="Experimentation" title="Model comparison"
        lede="Compare completed experiments under identical evaluation conditions. Differences smaller than the cross-validation spread are flagged rather than declared wins." />
      {error && <ErrorState error={error} />}
      {ids.length === 0 ? (
        <Panel>
          <EmptyState icon={GitCompare} title="Pick experiments to compare" action={<Link className="btn btn-primary" to="/experiments">Choose from history</Link>}>
            Train several algorithms at once in Model Studio — they share one split automatically — or select completed runs in Experiments.
          </EmptyState>
        </Panel>
      ) : list === null ? <SkeletonBlock rows={8} /> : (
        <div className="stack-lg">
          {pending.length > 0 && <Callout tone="info" title={`${pending.length} still training:`}>{pending.map((e) => e.algorithm).join(', ')}. This view updates when they finish.</Callout>}
          {failed.length > 0 && <Callout tone="warn" title="Not comparable:">{failed.map((e) => <span key={e.id}><Link to={`/experiments/${e.id}`}>{e.algorithm}</Link> ({e.status}) </span>)}</Callout>}
          {completed.length > 0 && (tasks.size > 1 ? (
            <Callout tone="bad" title="Mixed task types.">Classification and regression metrics cannot be compared. Remove experiments of one type.</Callout>
          ) : splitIds.size === 1 && targets.size === 1 ? (
            <div className="callout"><CheckCircle2 size={15} color="var(--good)" /><div><strong>Identical evaluation conditions.</strong> All {completed.length} models were scored on the same {completed[0].results.data.n_test} test rows (split <span className="mono xs">{[...splitIds][0]}</span>).</div></div>
          ) : (
            <Callout tone="warn" icon={AlertTriangle} title="Different evaluation conditions.">
              These experiments used {targets.size > 1 ? 'different targets' : 'different test splits'} (different data, seed, test size or row filtering). Their scores are not directly comparable.
            </Callout>
          ))}

          {completed.length > 0 && tasks.size === 1 && (
            <div className="grid grid-main">
              <Panel title="Ranking" actions={<>
                <Prov kind="test" />
                <select className="select" style={{ width: 180, height: 28 }} value={m} onChange={(e) => setMetric(e.target.value)} aria-label="Metric">
                  {keys.map((k) => <option key={k} value={k}>{METRICS[k].label}</option>)}
                </select>
              </>} footer={`${METRICS[m].help} ${METRICS[m].higher ? 'Higher' : 'Lower'} is better.`}>
                <BarList items={sorted.map((e) => ({ label: e.algorithm, value: e.results.metrics.test[m] ?? 0, title: e.name }))} format={(v) => fmtNum(v, 4)} />
              </Panel>
              <Panel title="Trade-offs" actions={<Prov kind="rule" />}>
                <ul className="stack-sm small" style={{ margin: 0, paddingLeft: 16 }}>
                  {tradeoffs(completed, m).map((t) => <li key={t}>{t}</li>)}
                  {completed.length < 2 && <li className="subtle">Add another experiment to compare.</li>}
                </ul>
              </Panel>
            </div>
          )}

          {completed.length > 0 && (
            <Panel title="Side by side" flush actions={candidates.length > 0 && (
              <select className="select" style={{ width: 220, height: 28 }} value="" onChange={(e) => e.target.value && setIds([...ids, e.target.value])} aria-label="Add experiment">
                <option value="">+ Add experiment…</option>{candidates.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
              </select>
            )}>
              <div className="table-wrap" style={{ marginTop: 12 }}>
                <table className="table">
                  <thead>
                    <tr><th style={{ minWidth: 170 }}>Property</th>{completed.map((e) => (
                      <th key={e.id} style={{ minWidth: 170 }}>
                        <div className="row-between"><Link to={`/experiments/${e.id}`} className="truncate" style={{ color: 'var(--text)' }}>{e.algorithm}</Link>
                          <button className="btn btn-ghost btn-sm btn-icon" aria-label={`Remove ${e.algorithm}`} onClick={() => setIds(ids.filter((x) => x !== e.id))}><X size={12} /></button></div>
                      </th>))}</tr>
                  </thead>
                  <tbody>
                    <tr><td colSpan={completed.length + 1} className="eyebrow" style={{ background: 'var(--surface-2)' }}>Test metrics</td></tr>
                    {keys.map((k) => {
                      const best = bestFor(k);
                      return (
                        <tr key={k}><td title={METRICS[k].help}>{METRICS[k].label}</td>{completed.map((e) => {
                          const v = e.results.metrics.test[k];
                          return <td key={e.id} className="num" style={{ fontWeight: v === best && completed.length > 1 ? 700 : 400 }}>{fmtNum(v, 4)}{v === best && completed.length > 1 ? ' ◆' : ''}</td>;
                        })}</tr>
                      );
                    })}
                    {task === 'classification' && <tr><td>ROC-AUC</td>{completed.map((e) => <td key={e.id} className="num">{fmtNum(e.results.classification.roc.auc, 4)}</td>)}</tr>}
                    <tr><td colSpan={completed.length + 1} className="eyebrow" style={{ background: 'var(--surface-2)' }}>Generalisation</td></tr>
                    <tr><td>CV mean ± sd</td>{completed.map((e) => { const cv = e.results.metrics.cv; const k = task === 'classification' ? 'accuracy' : 'r2'; return <td key={e.id} className="num">{cv ? `${fmtNum(cv.mean[k], 3)} ± ${fmtNum(cv.std[k], 3)}` : '—'}</td>; })}</tr>
                    <tr><td>Train − test gap</td>{completed.map((e) => { const k = task === 'classification' ? 'accuracy' : 'r2'; const g = e.results.metrics.train[k] - e.results.metrics.test[k]; return <td key={e.id} className="num" style={{ color: g > 0.1 ? 'var(--warn)' : undefined }}>{fmtNum(g, 3)}</td>; })}</tr>
                    <tr><td colSpan={completed.length + 1} className="eyebrow" style={{ background: 'var(--surface-2)' }}>Configuration</td></tr>
                    <tr><td>Dataset</td>{completed.map((e) => <td key={e.id} className="small">{e.results.data.dataset_name}</td>)}</tr>
                    <tr><td>Target</td>{completed.map((e) => <td key={e.id} className="small">{e.results.target}</td>)}</tr>
                    <tr><td>Split</td>{completed.map((e) => <td key={e.id} className="small num">{Math.round(e.results.split.test_size * 100)}% · seed {e.results.split.random_state} · <span className="mono xs">{e.results.split.split_id.slice(0, 8)}</span></td>)}</tr>
                    <tr><td>Rows train / test</td>{completed.map((e) => <td key={e.id} className="num small">{e.results.data.n_train} / {e.results.data.n_test}</td>)}</tr>
                    <tr><td>Scaling · encoding</td>{completed.map((e) => <td key={e.id} className="small">{e.results.preprocessing.scaling_resolved} · {e.results.preprocessing.encoding}</td>)}</tr>
                    <tr><td>Imputation</td>{completed.map((e) => <td key={e.id} className="small">{e.results.preprocessing.numeric_imputation} / {e.results.preprocessing.categorical_imputation}</td>)}</tr>
                    <tr><td>Feature selection</td>{completed.map((e) => <td key={e.id} className="small">{e.results.preprocessing.feature_selection}</td>)}</tr>
                    <tr><td>Hyperparameters</td>{completed.map((e) => <td key={e.id} className="mono xs" style={{ whiteSpace: 'normal' }}>{Object.entries(e.results.algorithm.hyperparameters).map(([k, v]) => `${k}=${v}`).join(' ') || '—'}</td>)}</tr>
                    <tr><td>Fit time</td>{completed.map((e) => <td key={e.id} className="num small">{fmtDuration(e.results.timing.fit)}</td>)}</tr>
                    <tr><td>Created</td>{completed.map((e) => <td key={e.id} className="small subtle">{timeAgo(e.created_at)}</td>)}</tr>
                    <tr><td>Status</td>{completed.map((e) => <td key={e.id}><StatusBadge status={e.status} /></td>)}</tr>
                  </tbody>
                </table>
              </div>
            </Panel>
          )}
          {completed.length === 0 && pending.length === 0 && <EmptyState title="Nothing comparable yet">None of the selected experiments has completed.</EmptyState>}
          {candidates.length > 0 && completed.length === 0 && (
            <Field label="Add a completed experiment">
              <select className="select" style={{ width: 280 }} value="" onChange={(e) => e.target.value && setIds([...ids, e.target.value])}>
                <option value="">Choose…</option>{candidates.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
              </select>
            </Field>
          )}
          <div><Button variant="ghost" size="sm" icon={Plus} onClick={() => setIds([])}>Start a new comparison</Button></div>
        </div>
      )}
    </div>
  );
}
