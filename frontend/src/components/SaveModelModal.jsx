import { Save } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../lib/api';
import { useToast } from '../lib/toast';
import { Button, ErrorState, Field, Modal } from './ui';

/** Save a completed experiment's fitted pipeline as a named, versioned model. */
export default function SaveModelModal({ open, onClose, experiment, onSaved }) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const toast = useToast();

  useEffect(() => {
    if (open) {
      setName(`${experiment.algorithm} · ${experiment.target || experiment.config?.target || 'model'}`.slice(0, 120));
      setDescription('');
      setError(null);
    }
  }, [open, experiment]);

  const trimmed = name.trim();
  const save = async () => {
    if (!trimmed || busy) return;
    setBusy(true);
    setError(null);
    try {
      const m = await api.saveModel(experiment.id, { name: trimmed, description: description.trim() || null });
      toast.success(`Saved ${m.name} v${m.version}`, 'The fitted pipeline and its metadata are stored in Saved Models.');
      onSaved?.(m);
      onClose();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal open={open} onClose={onClose} title="Save model" width={540}
      footer={<><Button onClick={onClose}>Cancel</Button><Button variant="primary" icon={Save} loading={busy} disabled={!trimmed} onClick={save}>Save model</Button></>}>
      <div className="stack">
        <p className="small">Stores a copy of the complete fitted pipeline (preprocessing + {experiment.algorithm}) with its input schema, metrics and reproducibility metadata. It stays available even if this experiment is deleted.</p>
        <Field label="Name" htmlFor="sm-name" hint="Saving again under the same name creates a new version." error={!trimmed ? 'A name is required.' : null}>
          <input id="sm-name" className="input" maxLength={120} value={name} onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && save()} autoComplete="off" />
        </Field>
        <Field label="Description (optional)" htmlFor="sm-desc" aside={<span className="xs subtle">{description.length}/2000</span>}>
          <textarea id="sm-desc" className="textarea" rows={3} maxLength={2000} value={description} onChange={(e) => setDescription(e.target.value)}
            placeholder="What is this model for? Any caveats?" />
        </Field>
        {error && <ErrorState error={error} title="The model was not saved" />}
        <p className="xs subtle">Manage saved models in <Link to="/models" onClick={onClose}>Saved Models</Link>.</p>
      </div>
    </Modal>
  );
}
