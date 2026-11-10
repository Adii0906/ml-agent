import { motion } from 'framer-motion';
import { ArrowRight, BookOpen, Database, FileText, FlaskConical, Layers, Sparkles, Table2, Trophy, Upload } from 'lucide-react';
import { useMemo } from 'react';
import { Link } from 'react-router-dom';
import { MiniBars } from '../components/charts';
import Dropzone from '../components/Dropzone';
import { Logo } from '../components/shell/Shell';
import { Badge, EmptyState, ErrorState, Panel, Prov, SkeletonBlock, Stat, StatusBadge } from '../components/ui';
import { fmtInt, fmtNum, METRICS, primaryValue, timeAgo } from '../lib/format';
import { isActiveStatus, useWorkspace } from '../lib/workspace';
import { SampleDatasets } from './Datasets';

function Onboarding({ system }) {
  const steps = [
    { icon: Database, title: 'Load data', text: 'Upload a CSV or Parquet file, or start from a sample dataset.' },
    { icon: Table2, title: 'Explore', text: 'Profile every column, check quality findings, inspect the target.' },
    { icon: FlaskConical, title: 'Experiment', text: 'Pick a target, configure a leak-free pipeline, train several models on one split.' },
    { icon: FileText, title: 'Evaluate & report', text: 'Compare held-out metrics, inspect errors and export reproducible reports.' },
  ];
  return (
    <div className="page page-narrow">
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }} className="stack-lg">
        <div className="row" style={{ gap: 14 }}>
          <Logo size={40} />
          <div>
            <h1 style={{ fontSize: 24 }}>Welcome to ML-Agent</h1>
            <p className="muted">A workspace for exploring tabular data and running honest, reproducible model experiments.</p>
          </div>
        </div>
        <div className="grid grid-4">
          {steps.map((s, i) => (
            <motion.div key={s.title} className="panel" style={{ padding: 14 }} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.08 + i * 0.05 }}>
              <div className="row" style={{ gap: 8, marginBottom: 6 }}><span className="section-num">{String(i + 1).padStart(2, '0')}</span><s.icon size={15} color="var(--text-2)" /></div>
              <div style={{ fontWeight: 600 }}>{s.title}</div>
              <div className="small muted">{s.text}</div>
            </motion.div>
          ))}
        </div>
        <div className="grid grid-2">
          <Panel title="Upload your first dataset" icon={Upload}><Dropzone /></Panel>
          <Panel title="Or try a sample" icon={Sparkles}><SampleDatasets compact /></Panel>
        </div>
        <div className="callout">
          <Sparkles size={15} color="var(--ai)" />
          <div className="grow small">
            {system?.llm?.available
              ? <>AI explanations are available via <b>{system.llm.active_provider}</b>. They are always labelled and never replace calculated results.</>
              : <>AI explanations are <b>off</b> — no LLM is configured. Data exploration, training, evaluation and reports work fully without one. <Link to="/settings">Configure in Settings</Link>.</>}
          </div>
        </div>
      </motion.div>
    </div>
  );
}

export default function Overview() {
  const { datasets, experiments, error, system } = useWorkspace();
  const exps = useMemo(() => experiments || [], [experiments]);
  const completed = exps.filter((e) => e.status === 'completed');
  const active = exps.filter((e) => isActiveStatus(e.status));

  const bestByProblem = useMemo(() => {
    const groups = {};
    completed.forEach((e) => {
      const key = `${e.dataset_id}|${e.target}|${e.task_type}`;
      (groups[key] = groups[key] || []).push(e);
    });
    return Object.values(groups).map((list) => {
      const pk = list[0].task_type === 'classification' ? 'accuracy' : 'r2';
      const best = [...list].sort((a, b) => (primaryValue(b).value ?? -Infinity) - (primaryValue(a).value ?? -Infinity))[0];
      return { best, count: list.length, pk, splits: new Set(list.map((e) => e.metrics?.split_id)).size };
    }).sort((a, b) => new Date(b.best.created_at) - new Date(a.best.created_at)).slice(0, 6);
  }, [completed]);

  const activity = useMemo(() => {
    const days = [];
    for (let i = 13; i >= 0; i -= 1) {
      const d = new Date();
      d.setHours(0, 0, 0, 0);
      d.setDate(d.getDate() - i);
      days.push({ key: d.toDateString(), label: d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }), runs: 0 });
    }
    const idx = Object.fromEntries(days.map((d, i) => [d.key, i]));
    exps.forEach((e) => { const k = new Date(e.created_at).toDateString(); if (k in idx) days[idx[k]].runs += 1; });
    return days;
  }, [exps]);

  if (datasets === null || experiments === null) {
    return <div className="page">{error ? <ErrorState error={error} title="Cannot reach the ML-Agent server" /> : <SkeletonBlock rows={6} />}</div>;
  }
  if (datasets.length === 0 && exps.length === 0) return <Onboarding system={system} />;
  const dsName = (id) => datasets.find((d) => d.id === id)?.name || 'deleted dataset';

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <div className="eyebrow" style={{ marginBottom: 6 }}>Home</div>
          <h1>Overview</h1>
          <p className="lede">Pick up where you left off.</p>
        </div>
        <div className="row wrap">
          <Link className="btn" to="/datasets?upload=1"><Upload size={14} />Upload data</Link>
          <Link className="btn" to="/research"><BookOpen size={14} />Papers</Link>
          <Link className="btn btn-primary" to="/studio"><FlaskConical size={15} />New experiment</Link>
        </div>
      </div>

      <div className="stat-strip" style={{ marginBottom: 20 }}>
        <Stat label="Datasets" value={datasets.length} sub={`${fmtInt(datasets.reduce((a, d) => a + (d.n_rows || 0), 0))} rows total`} />
        <Stat label="Experiments" value={exps.length} sub={`${completed.length} completed`} />
        <Stat label="Running now" value={active.length} sub={active.length ? active.map((e) => e.algorithm).slice(0, 2).join(', ') : 'idle'} />
        <Stat label="Failed / cancelled" value={exps.filter((e) => ['failed', 'cancelled'].includes(e.status)).length} sub="kept in history" />
      </div>

      <div className="grid grid-main">
        <div className="stack-lg">
          {active.length > 0 && (
            <Panel title="Training now" icon={Layers} flush>
              <div className="list">
                {active.map((e) => (
                  <Link key={e.id} to={`/experiments/${e.id}`} className="list-item">
                    <StatusBadge status={e.status} />
                    <span className="grow truncate"><b>{e.algorithm}</b> <span className="subtle">· {e.name}</span></span>
                    <span className="small" style={{ color: 'var(--accent-text)' }}>Watch <ArrowRight size={12} style={{ verticalAlign: -1 }} /></span>
                  </Link>
                ))}
              </div>
            </Panel>
          )}
          <Panel title="Recent experiments" icon={Layers} flush actions={<Link className="btn btn-ghost btn-sm" to="/experiments">All</Link>}>
            {exps.length === 0 ? (
              <EmptyState icon={FlaskConical} title="No experiments yet" action={<Link className="btn btn-primary" to="/studio">Open Model Studio</Link>}>Your datasets are ready. Train a first model to see results here.</EmptyState>
            ) : (
              <div className="table-wrap">
                <table className="table">
                  <thead><tr><th>Experiment</th><th>Status</th><th className="r">Test score <Prov kind="test" /></th><th className="desktop-only">When</th></tr></thead>
                  <tbody>
                    {exps.slice(0, 7).map((e) => {
                      const pv = primaryValue(e);
                      return (
                        <tr key={e.id}>
                          <td style={{ maxWidth: 320 }}><Link to={`/experiments/${e.id}`} className="truncate" style={{ color: 'var(--text)', fontWeight: 550, display: 'block' }}>{e.name}</Link><span className="xs subtle">{e.algorithm} · {e.target}</span></td>
                          <td><StatusBadge status={e.status} /></td>
                          <td className="r num">{pv.value !== null && pv.value !== undefined ? <>{fmtNum(pv.value, 4)} <span className="xs subtle">{METRICS[pv.key].label}</span></> : <span className="subtle">—</span>}</td>
                          <td className="small subtle nowrap desktop-only">{timeAgo(e.created_at)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </Panel>
          {bestByProblem.length > 0 && (
            <Panel title="Best model per problem" icon={Trophy} flush actions={<Prov kind="test" />}
              footer="Best held-out score for each dataset + target. If runs used different splits, scores are not strictly comparable (flagged).">
              <div className="list">
                {bestByProblem.map(({ best, count, pk, splits }) => (
                  <Link key={best.id} to={`/experiments/${best.id}`} className="list-item">
                    <span className="list-icon"><Trophy size={14} /></span>
                    <span className="grow" style={{ minWidth: 0 }}>
                      <span className="truncate" style={{ display: 'block', fontWeight: 550 }}>{dsName(best.dataset_id)} → {best.target}</span>
                      <span className="xs subtle">{best.algorithm} · best of {count} run{count > 1 ? 's' : ''}{splits > 1 ? ' · mixed splits' : ''}</span>
                    </span>
                    <span className="num" style={{ fontWeight: 600 }}>{fmtNum(primaryValue(best).value, 4)}</span>
                    <span className="xs subtle" style={{ width: 64 }}>{METRICS[pk].label}</span>
                  </Link>
                ))}
              </div>
            </Panel>
          )}
        </div>
        <div className="stack-lg">
          <Panel title="Training activity" footer="Experiments started per day, last 14 days.">
            {exps.length ? <MiniBars data={activity} dataKey="runs" labelKey="label" height={130} format={(v) => `${v} run${v === 1 ? '' : 's'}`} /> : <span className="small subtle">No runs yet.</span>}
          </Panel>
          <Panel title="Recent datasets" icon={Database} flush actions={<Link className="btn btn-ghost btn-sm" to="/datasets">All</Link>}>
            <div className="list">
              {datasets.slice(0, 5).map((d) => (
                <Link key={d.id} to={`/datasets/${d.id}`} className="list-item">
                  <span className="list-icon"><Table2 size={14} /></span>
                  <span className="grow" style={{ minWidth: 0 }}><span className="truncate" style={{ display: 'block', fontWeight: 550 }}>{d.name}</span><span className="xs subtle num">{fmtInt(d.n_rows)} × {d.n_cols} · {timeAgo(d.last_opened_at || d.created_at)}</span></span>
                  {d.experiment_count > 0 && <Badge>{d.experiment_count} runs</Badge>}
                </Link>
              ))}
              {datasets.length === 0 && <div className="panel-body small subtle">No datasets.</div>}
            </div>
          </Panel>
          <Panel title="Recent reports" icon={FileText} flush actions={<Link className="btn btn-ghost btn-sm" to="/reports">All</Link>}>
            <div className="list">
              {completed.slice(0, 4).map((e) => (
                <Link key={e.id} to={`/reports/${e.id}`} className="list-item">
                  <span className="list-icon"><FileText size={14} /></span>
                  <span className="grow truncate small">{e.name}</span>
                  <span className="xs subtle nowrap">{timeAgo(e.completed_at || e.created_at)}</span>
                </Link>
              ))}
              {completed.length === 0 && <div className="panel-body small subtle">Reports appear once an experiment completes.</div>}
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
