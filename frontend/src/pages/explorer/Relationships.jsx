import { useEffect, useMemo, useState } from 'react';
import { CorrelationHeatmap, MissingMap, ScatterPlot } from '../../components/charts';
import { EmptyState, ErrorState, Field, Panel, Prov, SkeletonBlock } from '../../components/ui';
import { api } from '../../lib/api';
import { fmtInt } from '../../lib/format';
import { useAsync } from '../../lib/hooks';

export default function Relationships({ datasetId, overview }) {
  const corr = useAsync(() => api.correlations(datasetId), [datasetId]);
  const missing = useAsync(() => api.missing(datasetId), [datasetId]);
  const numeric = useMemo(() => overview?.numeric_columns || [], [overview]);
  const categorical = (overview?.column_info || []).filter((c) => c.kind === 'categorical' || c.kind === 'boolean').map((c) => c.name);
  const [x, setX] = useState('');
  const [y, setY] = useState('');
  const [color, setColor] = useState('');
  useEffect(() => {
    if (numeric.length >= 2 && !x) { setX(numeric[0]); setY(numeric[1]); }
  }, [numeric, x]);
  const scatter = useAsync(() => api.scatter(datasetId, { x, y, color }), [datasetId, x, y, color], { enabled: !!(x && y) });

  return (
    <div className="stack-lg">
      <div className="grid grid-2">
        <Panel title="Correlation matrix" actions={<><Prov kind="calc" /><span className="small subtle">Pearson · numeric columns</span></>}
          footer={corr.data?.sampled ? 'Computed on a 100,000-row random sample.' : corr.data?.truncated ? 'Showing the 40 highest-variance numeric columns.' : 'Correlation measures linear association only; it does not imply causation.'}>
          {corr.error ? <ErrorState error={corr.error} /> : !corr.data ? <SkeletonBlock rows={6} /> : corr.data.columns.length < 2 ? (
            <EmptyState title="Not enough numeric columns">At least two numeric columns are needed.</EmptyState>
          ) : (
            <CorrelationHeatmap columns={corr.data.columns} matrix={corr.data.matrix} onSelect={(a, b) => { setX(a); setY(b); }} />
          )}
        </Panel>
        <Panel title="Scatter explorer" actions={scatter.data?.pearson_r !== undefined && scatter.data?.pearson_r !== null ? <span className="small num">r = {scatter.data.pearson_r.toFixed(3)}</span> : null}>
          {numeric.length < 2 ? <EmptyState title="Needs two numeric columns" /> : (
            <div className="stack">
              <div className="grid grid-3" style={{ gap: 10 }}>
                <Field label="X axis" htmlFor="sx"><select id="sx" className="select" value={x} onChange={(e) => setX(e.target.value)}>{numeric.map((c) => <option key={c}>{c}</option>)}</select></Field>
                <Field label="Y axis" htmlFor="sy"><select id="sy" className="select" value={y} onChange={(e) => setY(e.target.value)}>{numeric.map((c) => <option key={c}>{c}</option>)}</select></Field>
                <Field label="Colour by" htmlFor="sc"><select id="sc" className="select" value={color} onChange={(e) => setColor(e.target.value)}>
                  <option value="">None</option>{categorical.map((c) => <option key={c}>{c}</option>)}</select></Field>
              </div>
              {scatter.error ? <ErrorState error={scatter.error} /> : !scatter.data ? <SkeletonBlock rows={6} /> : (
                <>
                  <ScatterPlot points={scatter.data.points} xLabel={x} yLabel={y} groups={scatter.data.groups} height={300} />
                  <div className="small subtle">
                    {scatter.data.shown < scatter.data.total ? `Random sample of ${fmtInt(scatter.data.shown)} of ${fmtInt(scatter.data.total)} rows` : `${fmtInt(scatter.data.total)} rows`}
                    {color ? ' · top 3 categories coloured, the rest grouped as Other' : ''}
                  </div>
                </>
              )}
            </div>
          )}
        </Panel>
      </div>
      {corr.data?.high_pairs?.length > 0 && (
        <Panel title="Highly correlated pairs" actions={<span className="small subtle">|r| ≥ 0.95</span>} flush>
          <div className="list">
            {corr.data.high_pairs.map((p) => (
              <button key={`${p.a}-${p.b}`} className="list-item" onClick={() => { setX(p.a); setY(p.b); }}>
                <span className="grow small"><b>{p.a}</b> ↔ <b>{p.b}</b></span>
                <span className="num small">r = {p.r.toFixed(3)}</span>
              </button>
            ))}
          </div>
        </Panel>
      )}
      <Panel title="Missing-value pattern" actions={<Prov kind="calc" />}>
        {missing.error ? <ErrorState error={missing.error} /> : !missing.data ? <SkeletonBlock /> : missing.data.columns.length === 0 ? (
          <div className="small muted">No missing values in this dataset.</div>
        ) : (
          <div className="grid grid-main">
            <div>
              <MissingMap columns={missing.data.columns} grid={missing.data.grid} />
              <div className="small subtle" style={{ marginTop: 8 }}>Each row of the map is one dataset row (evenly sampled); red cells are missing. Aligned red bands across columns suggest values missing together.</div>
            </div>
            <div>
              <div className="eyebrow" style={{ marginBottom: 6 }}>Rows by number of missing fields</div>
              <table className="table">
                <tbody>
                  {missing.data.rows_by_missing_count.map((r) => (
                    <tr key={r.missing}><td>{r.missing === 0 ? 'Complete rows' : `${r.missing} missing`}</td><td className="r num">{fmtInt(r.rows)}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </Panel>
    </div>
  );
}
