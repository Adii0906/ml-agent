import { Database } from 'lucide-react';
import { Link, Navigate } from 'react-router-dom';
import { EmptyState, SkeletonBlock } from '../../components/ui';
import { useWorkspace } from '../../lib/workspace';

/** "Data Explorer" opens the active dataset, else the most recent one, else onboarding. */
export default function ExplorerRedirect() {
  const { datasets, activeDatasetId } = useWorkspace();
  if (datasets === null) return <div className="page"><SkeletonBlock /></div>;
  const target = datasets.find((d) => d.id === activeDatasetId) || datasets[0];
  if (target) return <Navigate to={`/datasets/${target.id}`} replace />;
  return (
    <div className="page">
      <EmptyState icon={Database} title="No dataset to explore yet"
        action={<Link className="btn btn-primary" to="/datasets?upload=1">Upload a dataset</Link>}>
        Upload a CSV or load a sample dataset. The explorer profiles every column, flags data-quality issues and plots distributions.
      </EmptyState>
    </div>
  );
}
