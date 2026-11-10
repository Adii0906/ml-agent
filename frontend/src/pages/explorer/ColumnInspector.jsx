import { Search } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { BarList, BoxPlot, Histogram } from '../../components/charts';
import { EmptyState, ErrorState, Panel, Prov, SkeletonBlock, TypeChip } from '../../components/ui';
import { api } from '../../lib/api';
import { fmtInt, fmtNum, fmtPct } from '../../lib/format';
import { useAsync } from '../../lib/hooks';

function NumericStats({ s }) {
  const rows = [
    ['Mean', s.mean], ['Std. deviation', s.std], ['Minimum', s.min], ['25th pct (Q1)', s.q1], ['Median', s.median],
    ['75th pct (Q3)', s.q3], ['Maximum', s.max], ['Skewness', s.skew], ['Zeros', s.zeros],
  ];
  return (
    <dl className="kv">
      {rows.map(([k, v]) => <div key={k} style={{ display: 'contents' }}><dt>{k}</dt><dd className="num">{fmtNum(v, 4)}</dd></div>)}
      <dt>IQR outliers</dt><dd className="num">{fmtInt(s.outliers)} <span className="subtle">({fmtPct(s.outlier_pct)})</span></dd>
    </dl>
  );
}

export default function ColumnInspector({ datasetId, columns, selected, onSelect }) {
  const [filter, setFilter] = useState('');
  const list = useMemo(() => (columns || []).filter((c) => c.name.toLowerCase().includes(filter.toLowerCase())), [columns, filter]);
  const current = selected || columns?.[0]?.name;
  const detail = useAsync(() => (current ? api.column(datasetId, current) : null), [datasetId, current], { enabled: !!current });

  useEffect(() => {
    if (!selected && columns?.length) onSelect(columns[0].name);
  }, [selected, columns, onSelect]);

  const d = detail.data;
  return (
    <div className="grid grid-side">
      <Panel flush title="Columns" className="desktop-only" style={{ alignSelf: 'start' }}>
        <div style={{ padding: '10px 12px 6px' }}>
          <div className="input-icon"><Search size={14} /><input className="input" value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="Filter columns" aria-label="Filter columns" /></div>
        </div>
        <div className="list" style={{ maxHeight: '62vh', overflow: 'auto' }} role="listbox" aria-label="Columns">
          {list.map((c) => (
            <button key={c.name} className="list-item" role="option" aria-selected={c.name === current} onClick={() => onSelect(c.name)}
              style={{ padding: '7px 12px', background: c.name === current ? 'var(--surface-2)' : undefined, fontWeight: c.name === current ? 600 : 400 }}>
              <TypeChip kind={c.kind} />
              <span className="truncate grow small">{c.name}</span>
              {c.missing > 0 && <span className="xs" style={{ color: 'var(--warn)' }} title={`${c.missing} missing`}>{fmtPct(c.missing_pct, 0)}</span>}
            </button>
          ))}
        </div>
      </Panel>
      <div className="stack-lg">
        <div className="mobile-only" style={{ width: '100%' }}>
          <select className="select" value={current || ''} onChange={(e) => onSelect(e.target.value)} aria-label="Column">
            {(columns || []).map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
          </select>
        </div>
        {detail.error && <ErrorState error={detail.error} onRetry={detail.reload} />}
        {!d ? <Panel><SkeletonBlock rows={6} /></Panel> : (
          <>
            <Panel title={<span className="row" style={{ gap: 8 }}><TypeChip kind={d.kind} />{d.name}</span>} actions={<Prov kind="calc" />}>
              <div className="row wrap small muted" style={{ gap: 16 }}>
                <span>dtype <span className="mono">{d.dtype}</span></span>
                <span>{fmtInt(d.unique)} unique</span>
                <span style={{ color: d.missing ? 'var(--warn)' : undefined }}>{fmtInt(d.missing)} missing ({fmtPct(d.missing_pct)})</span>
                <span className="subtle">{d.kind_reason}</span>
              </div>
            </Panel>
            {d.numeric && d.numeric.count > 0 && (
              <div className="grid grid-main">
                <Panel title="Distribution" actions={<span className="small subtle">{d.numeric.histogram.counts.length} bins · {fmtInt(d.numeric.count)} values</span>}>
                  <Histogram edges={d.numeric.histogram.edges} counts={d.numeric.histogram.counts} xLabel={d.name} height={220} />
                  <div style={{ marginTop: 14 }}>
                    <div className="eyebrow" style={{ marginBottom: 4 }}>Box plot · whiskers at 1.5×IQR</div>
                    <BoxPlot box={d.numeric.box} min={d.numeric.min} max={d.numeric.max} />
                  </div>
                </Panel>
                <Panel title="Summary statistics"><NumericStats s={d.numeric} /></Panel>
              </div>
            )}
            {d.categorical && (
              <Panel title={d.numeric ? 'Value frequencies' : 'Category frequencies'}
                actions={<span className="small subtle">{fmtInt(d.categorical.unique)} distinct · cardinality {fmtPct(d.categorical.cardinality_ratio)}</span>}>
                {d.categorical.top_values.length ? (
                  <BarList items={d.categorical.top_values.map((t) => ({ label: t.value, value: t.count, sub: `· ${fmtPct(t.count / d.categorical.count)}` }))} format={fmtInt} />
                ) : <EmptyState title="No values" />}
                {d.categorical.other_count > 0 && <div className="small subtle" style={{ marginTop: 8 }}>+ {fmtInt(d.categorical.other_count)} rows in {fmtInt(d.categorical.unique - d.categorical.top_values.length)} other values</div>}
              </Panel>
            )}
          </>
        )}
      </div>
    </div>
  );
}
