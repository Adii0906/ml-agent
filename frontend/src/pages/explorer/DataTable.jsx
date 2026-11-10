import { ArrowDown, ArrowUp, ChevronLeft, ChevronRight, Search } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Button, ErrorState, Panel, Skeleton, TypeChip } from '../../components/ui';
import { api } from '../../lib/api';
import { fmtInt, fmtNum } from '../../lib/format';
import { useDebounced } from '../../lib/hooks';

const PAGE_SIZES = [50, 100, 250];

export default function DataTable({ datasetId, columns }) {
  const [offset, setOffset] = useState(0);
  const [limit, setLimit] = useState(50);
  const [sort, setSort] = useState(null);
  const [order, setOrder] = useState('asc');
  const [search, setSearch] = useState('');
  const [searchCol, setSearchCol] = useState('');
  const [missingOnly, setMissingOnly] = useState(false);
  const [page, setPage] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const q = useDebounced(search, 300);

  const kinds = useMemo(() => Object.fromEntries((columns || []).map((c) => [c.name, c.kind])), [columns]);

  useEffect(() => { setOffset(0); }, [q, searchCol, missingOnly, sort, order, limit]);

  useEffect(() => {
    const ctrl = new AbortController();
    setLoading(true);
    api.rows(datasetId, { offset, limit, sort, order, search: q, column: searchCol, missing_only: missingOnly || undefined }, ctrl.signal)
      .then((d) => { setPage(d); setError(null); })
      .catch((e) => { if (e.name !== 'AbortError') setError(e); })
      .finally(() => setLoading(false));
    return () => ctrl.abort();
  }, [datasetId, offset, limit, sort, order, q, searchCol, missingOnly]);

  const toggleSort = (c) => {
    if (sort !== c) { setSort(c); setOrder('asc'); } else if (order === 'asc') setOrder('desc'); else { setSort(null); setOrder('asc'); }
  };
  const total = page?.filtered ?? 0;
  const from = total ? offset + 1 : 0;
  const to = Math.min(offset + limit, total);

  return (
    <Panel flush title="Rows" actions={
      <div className="row wrap" style={{ gap: 8 }}>
        <div className="input-icon" style={{ width: 220 }}>
          <Search size={14} />
          <input className="input" placeholder="Search values…" value={search} onChange={(e) => setSearch(e.target.value)} aria-label="Search rows" />
        </div>
        <select className="select" style={{ width: 160 }} value={searchCol} onChange={(e) => setSearchCol(e.target.value)} aria-label="Search in column">
          <option value="">All columns</option>
          {(page?.columns || []).map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        <label className="check small"><input type="checkbox" checked={missingOnly} onChange={(e) => setMissingOnly(e.target.checked)} />Rows with missing values</label>
      </div>
    }>
      {error && <div style={{ padding: 16 }}><ErrorState error={error} /></div>}
      <div className="grid-wrap" style={{ marginTop: 12, opacity: loading && page ? 0.6 : 1, transition: 'opacity 120ms' }} aria-busy={loading}>
        {!page ? <div style={{ padding: 16 }} className="stack-sm">{Array.from({ length: 10 }, (_, i) => <Skeleton key={i} h={22} />)}</div> : (
          <table className="datagrid" aria-rowcount={total}>
            <thead>
              <tr>
                <th className="idx" scope="col" title="Original row index">#</th>
                {page.columns.map((c) => (
                  <th key={c} scope="col" onClick={() => toggleSort(c)} aria-sort={sort === c ? (order === 'asc' ? 'ascending' : 'descending') : 'none'}
                    title={`${c} · ${page.dtypes[c]} — click to sort`} tabIndex={0} onKeyDown={(e) => e.key === 'Enter' && toggleSort(c)}>
                    <span className="th-inner">
                      {kinds[c] && <TypeChip kind={kinds[c]} />}
                      <span className="truncate">{c}</span>
                      {sort === c && (order === 'asc' ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {page.rows.map((r, i) => (
                <tr key={page.index[i]}>
                  <td className="idx">{page.index[i]}</td>
                  {r.map((v, j) => {
                    const isNum = typeof v === 'number';
                    if (v === null) return <td key={j} className="is-missing" title="Missing value">null</td>;
                    return <td key={j} className={isNum ? 'is-num' : ''} title={String(v)}>{isNum ? fmtNum(v, 4) : String(v)}</td>;
                  })}
                </tr>
              ))}
              {page.rows.length === 0 && <tr><td colSpan={page.columns.length + 1} style={{ textAlign: 'center', padding: 24, fontFamily: 'var(--font-sans)' }} className="muted">No rows match.</td></tr>}
            </tbody>
          </table>
        )}
      </div>
      <div className="row-between wrap" style={{ padding: '10px 16px', borderTop: '1px solid var(--border)' }}>
        <span className="small muted num">
          {fmtInt(from)}–{fmtInt(to)} of {fmtInt(total)}{page && page.filtered !== page.total ? ` (filtered from ${fmtInt(page.total)})` : ''} rows
        </span>
        <div className="row" style={{ gap: 6 }}>
          <select className="select" style={{ width: 92, height: 28 }} value={limit} onChange={(e) => setLimit(Number(e.target.value))} aria-label="Rows per page">
            {PAGE_SIZES.map((s) => <option key={s} value={s}>{s} / page</option>)}
          </select>
          <Button size="sm" iconOnly icon={ChevronLeft} aria-label="Previous page" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))} />
          <Button size="sm" iconOnly icon={ChevronRight} aria-label="Next page" disabled={to >= total} onClick={() => setOffset(offset + limit)} />
        </div>
      </div>
    </Panel>
  );
}
