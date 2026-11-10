import { Crosshair, FlaskConical } from 'lucide-react';
import { useState } from 'react';
import { Link } from 'react-router-dom';
import { BarList, Histogram } from '../../components/charts';
import { Badge, EmptyState, ErrorState, Field, Panel, Prov, SeverityIcon, SkeletonBlock } from '../../components/ui';
import { api } from '../../lib/api';
import { fmtInt, fmtNum, fmtPct, titleCase } from '../../lib/format';
import { useAsync } from '../../lib/hooks';

export default function TargetPanel({ datasetId, columns }) {
  const [target, setTarget] = useState('');
  const res = useAsync(() => api.target(datasetId, target), [datasetId, target], { enabled: !!target });
  const a = res.data?.analysis;
  const rec = res.data?.recommendation;
  return (
    <div className="stack-lg">
      <Panel title="Target analysis" icon={Crosshair}>
        <div className="row wrap" style={{ gap: 16, alignItems: 'flex-end' }}>
          <div style={{ width: 280 }}>
            <Field label="Column to predict" htmlFor="tgt">
              <select id="tgt" className="select" value={target} onChange={(e) => setTarget(e.target.value)}>
                <option value="">Choose a column…</option>
                {(columns || []).map((c) => <option key={c.name} value={c.name}>{c.name} ({c.kind})</option>)}
              </select>
            </Field>
          </div>
          {target && a?.problem_type && (
            <Link className="btn btn-primary" to={`/studio?dataset=${datasetId}&target=${encodeURIComponent(target)}`}><FlaskConical size={15} />Train with this target</Link>
          )}
        </div>
        <p className="small subtle" style={{ marginTop: 10 }}>Inspect the label distribution, class balance and potential leakage before training. Nothing here is decided by an LLM.</p>
      </Panel>
      {!target ? null : res.error ? <ErrorState error={res.error} /> : !a ? <Panel><SkeletonBlock rows={5} /></Panel> : (
        <div className="grid grid-main">
          <div className="stack-lg">
            <Panel title="Problem type" actions={<Prov kind="rule" />}>
              <div className="row" style={{ gap: 10, marginBottom: 8 }}>
                <Badge tone={a.problem_type ? 'info' : 'bad'}>{a.problem_type ? titleCase(a.problem_type) : 'Not learnable'}</Badge>
                {a.ambiguous && <Badge tone="warn">Ambiguous — confirm in Model Studio</Badge>}
              </div>
              <p className="muted">{a.reason}</p>
              <dl className="kv" style={{ marginTop: 12 }}>
                <dt>Distinct values</dt><dd className="num">{fmtInt(a.n_unique)}</dd>
                <dt>Missing target</dt><dd className="num">{fmtInt(a.n_missing)} {a.n_missing ? <span className="subtle">(dropped before training)</span> : null}</dd>
                {a.imbalance_ratio && <><dt>Imbalance ratio</dt><dd className="num">{fmtNum(a.imbalance_ratio, 2)} : 1</dd></>}
              </dl>
            </Panel>
            {a.class_distribution && (
              <Panel title="Class distribution" actions={<Prov kind="calc" />}>
                <BarList items={a.class_distribution.map((c) => ({ label: c.label, value: c.count, sub: `· ${fmtPct(c.count / (a.n_rows - a.n_missing))}` }))} format={fmtInt} />
              </Panel>
            )}
            {a.numeric && !a.class_distribution && (
              <Panel title="Target distribution" actions={<Prov kind="calc" />}>
                <Histogram edges={a.numeric.histogram.edges} counts={a.numeric.histogram.counts} xLabel={target} height={200} />
              </Panel>
            )}
          </div>
          <div className="stack-lg">
            <Panel title="Target checks" flush actions={<Prov kind="rule" />}>
              {a.findings.length === 0 ? <div className="panel-body small muted">No target-specific issues detected.</div> : a.findings.map((f) => (
                <div key={f.id} className="finding" style={{ gridTemplateColumns: '18px minmax(0, 1fr)' }}>
                  <SeverityIcon severity={f.severity} />
                  <div><div className="ttl">{f.title}</div><div className="det">{f.detail}</div></div>
                </div>
              ))}
            </Panel>
            {rec && (
              <Panel title="Suggested starting points" actions={<Prov kind="rule" />}>
                <ol style={{ margin: 0, paddingLeft: 18 }} className="stack-sm small">
                  {rec.suggested.map((id) => {
                    const r = rec.algorithms.find((x) => x.id === id);
                    return <li key={id}><b>{r.name}</b> <span className="subtle">— {r.reasons[0]}</span></li>;
                  })}
                </ol>
                <p className="xs subtle" style={{ marginTop: 10 }}>{rec.method}</p>
              </Panel>
            )}
            {!a.problem_type && <EmptyState title="Choose another target">This column cannot be used as a supervised learning target.</EmptyState>}
          </div>
        </div>
      )}
    </div>
  );
}
