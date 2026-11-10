import { AnimatePresence, motion } from 'framer-motion';
import { AlertTriangle, BarChart3, Columns, Crosshair, FlaskConical, GitBranch, Grid3x3, LayoutGrid, ShieldAlert, Table2 } from 'lucide-react';
import { useEffect, useMemo } from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { Disclaimer, EmptyState, ErrorState, PageHeader, Panel, Prov, SkeletonBlock, Stat, Tabs, TypeChip } from '../../components/ui';
import { api } from '../../lib/api';
import { fmtBytes, fmtInt, fmtPct, timeAgo } from '../../lib/format';
import { useAsync } from '../../lib/hooks';
import { useWorkspace } from '../../lib/workspace';
import ColumnInspector from './ColumnInspector';
import DataTable from './DataTable';
import QualityPanel from './QualityPanel';
import Relationships from './Relationships';
import TargetPanel from './TargetPanel';

function ColumnInventory({ columns, rows, onOpen }) {
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr><th>Column</th><th>Type</th><th className="desktop-only">dtype</th><th style={{ minWidth: 150 }}>Missing</th><th className="r">Unique</th><th className="desktop-only">Sample values</th></tr>
        </thead>
        <tbody>
          {columns.map((c) => (
            <tr key={c.name} className="clickable" onClick={() => onOpen(c.name)} tabIndex={0} onKeyDown={(e) => e.key === 'Enter' && onOpen(c.name)}>
              <td style={{ fontWeight: 550, maxWidth: 220 }} className="truncate" title={c.name}>{c.name}</td>
              <td><TypeChip kind={c.kind} title={c.kind_reason} /></td>
              <td className="mono small subtle desktop-only">{c.dtype}</td>
              <td>
                <div className="row" style={{ gap: 8 }}>
                  <div className="progress" style={{ width: 70, height: 5 }}><div style={{ width: `${c.missing_pct * 100}%`, background: c.missing_pct > 0.5 ? 'var(--bad)' : c.missing_pct > 0 ? 'var(--status-warning)' : 'transparent' }} /></div>
                  <span className="num small" style={{ color: c.missing ? 'var(--text)' : 'var(--text-3)' }}>{c.missing ? `${fmtInt(c.missing)} · ${fmtPct(c.missing_pct)}` : '0'}</span>
                </div>
              </td>
              <td className="r num">{fmtInt(c.unique)}{rows ? <span className="subtle xs"> {c.unique === rows ? '(all)' : ''}</span> : null}</td>
              <td className="mono small subtle truncate desktop-only" style={{ maxWidth: 260 }}>{c.sample.map((v) => (v === null ? '∅' : String(v))).join(', ')}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function DataExplorer() {
  const { id } = useParams();
  const [params, setParams] = useSearchParams();
  const tab = params.get('tab') || 'overview';
  const column = params.get('column');
  const { setActiveDatasetId, experiments } = useWorkspace();
  const meta = useAsync(() => api.dataset(id), [id]);
  const profile = useAsync(() => api.profile(id), [id]);

  useEffect(() => { setActiveDatasetId(id); }, [id, setActiveDatasetId]);

  const setTab = (t, extra = {}) => {
    const next = new URLSearchParams(params);
    next.set('tab', t);
    Object.entries(extra).forEach(([k, v]) => (v ? next.set(k, v) : next.delete(k)));
    setParams(next, { replace: false });
  };
  const openColumn = (name) => setTab('columns', { column: name });

  const ov = profile.data?.overview;
  const quality = profile.data?.quality || [];
  const issues = quality.filter((q) => q.severity !== 'info').length;
  const dsExperiments = useMemo(() => (experiments || []).filter((e) => e.dataset_id === id), [experiments, id]);

  if (meta.error?.status === 404) {
    return <div className="page"><EmptyState icon={Table2} title="Dataset not found" action={<Link className="btn" to="/datasets">All datasets</Link>}>It may have been deleted.</EmptyState></div>;
  }

  const tabs = [
    { value: 'overview', label: 'Overview', icon: LayoutGrid },
    { value: 'table', label: 'Table', icon: Table2 },
    { value: 'columns', label: 'Columns', icon: Columns },
    { value: 'relationships', label: 'Relationships', icon: Grid3x3 },
    { value: 'quality', label: 'Quality', icon: ShieldAlert, count: profile.data ? issues : null },
    { value: 'target', label: 'Target', icon: Crosshair },
  ];

  return (
    <div className="page">
      <PageHeader
        eyebrow={<span className="row" style={{ gap: 8 }}>Data Explorer {meta.data?.parent_id && <span className="row" style={{ gap: 4, textTransform: 'none', letterSpacing: 0 }}><GitBranch size={11} /> derived from <Link to={`/datasets/${meta.data.parent_id}`}>{meta.data.parent_name || 'parent'}</Link></span>}</span>}
        title={meta.data?.name || '…'}
        lede={meta.data ? <>{meta.data.filename} · uploaded {timeAgo(meta.data.created_at)} · <span className="mono xs" title={meta.data.sha256}>sha256 {meta.data.sha256?.slice(0, 12)}</span>{meta.data.note ? ` · ${meta.data.note}` : ''}</> : null}
        actions={<>
          {dsExperiments.length > 0 && <Link className="btn" to={`/experiments?dataset=${id}`}>{dsExperiments.length} experiment{dsExperiments.length > 1 ? 's' : ''}</Link>}
          <Link className="btn btn-primary" to={`/studio?dataset=${id}`}><FlaskConical size={15} />Train a model</Link>
        </>}
      />

      {profile.error && <ErrorState error={profile.error} onRetry={profile.reload} title="Could not profile this dataset" />}

      <div className="stat-strip" style={{ marginBottom: 20 }}>
        {ov ? <>
          <Stat label="Rows" value={fmtInt(ov.rows)} prov="calc" />
          <Stat label="Columns" value={ov.columns} sub={`${ov.numeric_columns.length} numeric · ${ov.categorical_columns.length} categorical`} />
          <Stat label="Missing cells" value={fmtPct(ov.missing_pct)} sub={`${fmtInt(ov.missing_cells)} cells · ${fmtInt(ov.rows_with_missing)} rows`} />
          <Stat label="Duplicate rows" value={fmtInt(ov.duplicate_rows)} sub={ov.rows ? fmtPct(ov.duplicate_rows / ov.rows) : null} />
          <Stat label="Memory" value={fmtBytes(ov.memory_bytes)} sub="in-memory (pandas)" />
          <Stat label="Quality findings" value={<span style={{ color: issues ? 'var(--warn)' : 'var(--good)' }}>{issues ? <><AlertTriangle size={15} style={{ verticalAlign: -2 }} /> {issues}</> : 'Clean'}</span>} sub={`${quality.length - issues} informational`} />
        </> : Array.from({ length: 6 }, (_, i) => <div key={i} className="stat"><SkeletonBlock rows={2} /></div>)}
      </div>

      <Tabs tabs={tabs} value={tab} onChange={(t) => setTab(t)} id="explorer" />

      <AnimatePresence mode="wait" initial={false}>
        <motion.div key={tab} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.16 }}>
          {tab === 'overview' && (
            <div className="grid grid-main-wide">
              <Panel title="Columns" icon={Columns} flush actions={<Prov kind="calc" />}
                footer="Types are inferred from values. Identifiers, free text, dates and constant columns are excluded from training by default.">
                {ov ? <ColumnInventory columns={ov.column_info} rows={ov.rows} onOpen={openColumn} /> : <div style={{ padding: 16 }}><SkeletonBlock rows={6} /></div>}
              </Panel>
              <div className="stack-lg">
                <Panel title="Top findings" icon={ShieldAlert} actions={<button className="btn btn-ghost btn-sm" onClick={() => setTab('quality')}>All findings</button>}>
                  {!profile.data ? <SkeletonBlock /> : quality.length === 0 ? <div className="small muted">No data-quality issues detected.</div> : (
                    <div className="stack-sm">
                      {quality.slice(0, 5).map((q) => (
                        <div key={q.id} className="row small" style={{ alignItems: 'flex-start', gap: 8 }}>
                          <span className={`sev-${q.severity}`} style={{ marginTop: 3 }}><span className="dot" /></span>
                          <span>{q.title}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </Panel>
                <Panel title="Type mix" icon={BarChart3}>
                  {ov ? (
                    <div className="stack-sm">
                      {Object.entries(ov.kind_counts).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
                        <div key={k} className="row-between small"><span className="row" style={{ gap: 8 }}><TypeChip kind={k} />{k}</span><span className="num">{v}</span></div>
                      ))}
                    </div>
                  ) : <SkeletonBlock />}
                </Panel>
                <Disclaimer />
              </div>
            </div>
          )}
          {tab === 'table' && <DataTable datasetId={id} columns={ov?.column_info} />}
          {tab === 'columns' && <ColumnInspector datasetId={id} columns={ov?.column_info} selected={column} onSelect={openColumn} />}
          {tab === 'relationships' && <Relationships datasetId={id} overview={ov} />}
          {tab === 'quality' && <QualityPanel datasetId={id} findings={quality} loading={!profile.data} />}
          {tab === 'target' && <TargetPanel datasetId={id} columns={ov?.column_info} />}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
