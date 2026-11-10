import { Compass } from 'lucide-react';
import { Link } from 'react-router-dom';
import { EmptyState } from '../components/ui';

export default function NotFound() {
  return (
    <div className="page">
      <EmptyState icon={Compass} title="Page not found" action={<Link className="btn" to="/">Back to overview</Link>}>
        The page you are looking for does not exist or was moved.
      </EmptyState>
    </div>
  );
}
