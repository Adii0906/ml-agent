import { FileText } from 'lucide-react';
import { Link } from 'react-router-dom';
import { EmptyState, PageHeader, Panel, SkeletonBlock } from '../components/ui';
import { fmtNum, METRICS, primaryValue, timeAgo } from '../lib/format';
import { useWorkspace } from '../lib/workspace';

export default function Reports() {
  const { experiments, datasets } = useWorkspace();
  const completed = (experiments || []).filter((e) => e.status === 'completed');
  return (
    <div className="page page-narrow">
      <PageHeader eyebrow="Intelligence" title="Insights & reports"
        lede="A shareable report exists for every completed experiment. Reports contain only numbers recorded by that experiment, plus rule-based observations. AI narrative is optional and labelled." />
      <Panel flush>
        {experiments === null ? <div style={{ padding: 16 }}><SkeletonBlock /></div> : completed.length === 0 ? (
          <EmptyState icon={FileText} title="No reports yet" action={<Link className="btn btn-primary" to="/studio">Train a model</Link>}>Reports are generated from completed experiments, so they can always be traced back to a real run.</EmptyState>
        ) : (
          <div className="list">
            {completed.map((e) => {
              const pv = primaryValue(e);
              return (
                <Link key={e.id} to={`/reports/${e.id}`} className="list-item">
                  <span className="list-icon"><FileText size={14} /></span>
                  <span className="grow" style={{ minWidth: 0 }}>
                    <span className="truncate" style={{ display: 'block', fontWeight: 550 }}>{e.name}</span>
                    <span className="xs subtle">{e.algorithm} · {datasets?.find((d) => d.id === e.dataset_id)?.name || 'deleted dataset'} → {e.target}</span>
                  </span>
                  <span className="num small desktop-only">{pv.key ? `${METRICS[pv.key].label} ${fmtNum(pv.value, 4)}` : ''}</span>
                  <span className="xs subtle nowrap" style={{ width: 80, textAlign: 'right' }}>{timeAgo(e.completed_at || e.created_at)}</span>
                </Link>
              );
            })}
          </div>
        )}
      </Panel>
    </div>
  );
}
