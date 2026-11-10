import { AnimatePresence, motion } from 'framer-motion';
import { CheckCircle2, FileSpreadsheet, UploadCloud } from 'lucide-react';
import { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../lib/api';
import { fmtBytes } from '../lib/format';
import { useToast } from '../lib/toast';
import { useWorkspace } from '../lib/workspace';

const ACCEPT = ['.csv', '.tsv', '.parquet'];

export default function Dropzone({ autoFocus, compact, onUploaded }) {
  const inputRef = useRef(null);
  const [over, setOver] = useState(false);
  const [state, setState] = useState(null); // {file, progress, done, error}
  const toast = useToast();
  const navigate = useNavigate();
  const { system, refreshDatasets, setActiveDatasetId } = useWorkspace();
  const maxMb = system?.max_upload_mb ?? 200;

  const start = async (file) => {
    if (!file) return;
    const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
    if (!ACCEPT.includes(ext)) {
      setState({ file, error: `"${ext}" files are not supported. Upload CSV, TSV or Parquet.` });
      return;
    }
    if (file.size > maxMb * 1024 * 1024) {
      setState({ file, error: `${fmtBytes(file.size)} exceeds the ${maxMb} MB limit.` });
      return;
    }
    if (file.size === 0) {
      setState({ file, error: 'The file is empty.' });
      return;
    }
    setState({ file, progress: 0 });
    try {
      const meta = await api.uploadDataset(file, (p) => setState((s) => ({ ...s, progress: p })));
      setState({ file, progress: 1, done: true });
      await refreshDatasets();
      setActiveDatasetId(meta.id);
      toast.success('Dataset uploaded', `${meta.name} · ${meta.n_rows.toLocaleString()} rows × ${meta.n_cols} columns`);
      if (onUploaded) onUploaded(meta);
      else setTimeout(() => navigate(`/datasets/${meta.id}`), 450);
    } catch (e) {
      setState({ file, error: e.message });
    }
  };

  const busy = state && !state.error && !state.done;
  return (
    <div>
      <div
        className={`dropzone ${over ? 'over' : ''}`}
        role="button" tabIndex={0} aria-label="Upload a dataset: drop a file or press Enter to browse"
        style={compact ? { padding: '18px 16px' } : undefined}
        ref={(el) => { if (autoFocus && el) el.focus(); }}
        onClick={() => !busy && inputRef.current?.click()}
        onKeyDown={(e) => { if ((e.key === 'Enter' || e.key === ' ') && !busy) { e.preventDefault(); inputRef.current?.click(); } }}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); if (!busy) start(e.dataTransfer.files?.[0]); }}
      >
        <input ref={inputRef} type="file" accept={ACCEPT.join(',')} hidden onChange={(e) => { start(e.target.files?.[0]); e.target.value = ''; }} />
        <AnimatePresence mode="wait" initial={false}>
          {!state || state.error ? (
            <motion.div key="idle" className="stack-sm" style={{ alignItems: 'center' }} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <motion.div animate={over ? { y: -3, scale: 1.05 } : { y: 0, scale: 1 }} className="empty-icon"><UploadCloud size={18} /></motion.div>
              <div style={{ fontWeight: 600 }}>{over ? 'Drop to upload' : 'Drop a dataset here, or browse'}</div>
              <div className="small subtle">CSV, TSV or Parquet · up to {maxMb} MB · stored on your ML-Agent server</div>
            </motion.div>
          ) : (
            <motion.div key="busy" className="stack-sm" style={{ alignItems: 'center', width: 'min(360px, 100%)' }} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
              <div className="row" style={{ gap: 8 }}>
                {state.done ? <CheckCircle2 size={16} color="var(--good)" /> : <FileSpreadsheet size={16} />}
                <span className="truncate" style={{ fontWeight: 600 }}>{state.file.name}</span>
                <span className="small subtle nowrap">{fmtBytes(state.file.size)}</span>
              </div>
              <div className="progress" role="progressbar" aria-valuenow={Math.round((state.progress || 0) * 100)} aria-valuemin={0} aria-valuemax={100}>
                <div style={{ width: `${Math.round((state.progress || 0) * 100)}%`, background: state.done ? 'var(--good)' : undefined }} />
              </div>
              <div className="small subtle">{state.done ? 'Profiled and ready' : state.progress >= 1 ? 'Validating and profiling…' : `Uploading ${Math.round((state.progress || 0) * 100)}%`}</div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      {state?.error && <div className="field-error" role="alert" style={{ marginTop: 8 }}>{state.file?.name}: {state.error}</div>}
    </div>
  );
}
