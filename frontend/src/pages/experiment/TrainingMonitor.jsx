import { motion } from 'framer-motion';
import { Check, Loader2, Square, Wifi, WifiOff, X } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { Badge, Button, Panel, StatusBadge } from '../../components/ui';
import { api, wsUrl } from '../../lib/api';
import { fmtDuration } from '../../lib/format';
import { useToast } from '../../lib/toast';

const STAGES = [
  ['prepare', 'Validate & filter rows'],
  ['split', 'Train / test split'],
  ['cv', 'Cross-validation (training split)'],
  ['fit', 'Fit final pipeline'],
  ['evaluate', 'Evaluate on held-out test set'],
  ['explain', 'Feature importance & diagnostics'],
  ['persist', 'Save model artifact'],
];

/** Live training view. Uses the experiment WebSocket (which replays history on connect),
 *  reconnects with backoff, and falls back to polling if sockets are unavailable.
 *  Progress is shown per stage / CV fold — never as an invented percentage. */
export default function TrainingMonitor({ experiment, onFinished }) {
  const [events, setEvents] = useState(() => (experiment.logs || []).filter((e) => e.seq));
  const [conn, setConn] = useState('connecting');
  const [cancelling, setCancelling] = useState(false);
  const [now, setNow] = useState(Date.now());
  const toast = useToast();
  const finished = useRef(false);
  const id = experiment.id;

  useEffect(() => {
    let ws;
    let retry = 0;
    let closed = false;
    let pollTimer;
    const seen = new Set();
    const handle = (evt) => {
      if (evt.type === 'heartbeat') return;
      if (evt.seq) {
        if (seen.has(evt.seq)) return;
        seen.add(evt.seq);
        setEvents((list) => [...list.filter((e) => e.seq !== evt.seq), evt].sort((x, y) => x.seq - y.seq));
      }
      if (evt.terminal && !finished.current) {
        finished.current = true;
        onFinished(evt);
      }
    };
    const poll = async () => {
      try {
        const { experiment: exp } = await api.experiment(id);
        (exp.logs || []).forEach(handle);
        if (['completed', 'failed', 'cancelled'].includes(exp.status) && !finished.current) {
          finished.current = true;
          onFinished({ status: exp.status, error: exp.error });
          return;
        }
      } catch { /* keep polling */ }
      if (!closed) pollTimer = setTimeout(poll, 2500);
    };
    const connect = () => {
      if (closed) return;
      setConn(retry ? 'reconnecting' : 'connecting');
      try { ws = new WebSocket(wsUrl(`/ws/experiments/${id}`)); } catch { setConn('polling'); poll(); return; }
      ws.onopen = () => { retry = 0; setConn('live'); };
      ws.onmessage = (m) => { try { handle(JSON.parse(m.data)); } catch { /* ignore malformed */ } };
      ws.onclose = () => {
        if (closed || finished.current) return;
        retry += 1;
        if (retry > 4) { setConn('polling'); poll(); return; }
        setTimeout(connect, Math.min(800 * 2 ** retry, 8000));
      };
      ws.onerror = () => ws.close();
    };
    connect();
    return () => { closed = true; clearTimeout(pollTimer); ws?.close(); };
  }, [id, onFinished]);

  useEffect(() => { const t = setInterval(() => setNow(Date.now()), 500); return () => clearInterval(t); }, []);

  const progress = events.filter((e) => e.type === 'progress');
  const last = progress[progress.length - 1];
  const currentIdx = last?.stage_index ?? -1;
  const status = [...events].reverse().find((e) => e.type === 'status')?.status || experiment.status;
  const started = experiment.started_at || events.find((e) => e.status === 'running')?.time;
  const elapsed = started ? (now - new Date(started).getTime()) / 1000 : null;
  const foldEvt = [...progress].reverse().find((e) => e.stage === 'cv' && e.progress);
  const stageMsg = useMemo(() => {
    const m = {};
    progress.forEach((e) => { m[e.stage] = e.message; });
    return m;
  }, [progress]);

  const cancel = async () => {
    setCancelling(true);
    try { await api.cancelExperiment(id); toast.info('Cancellation requested', 'Training stops after the current step.'); } catch (e) { toast.error('Could not cancel', e.message); setCancelling(false); }
  };

  const cfg = experiment.config || {};
  return (
    <div className="grid grid-main">
      <Panel title={<span className="row" style={{ gap: 10 }}>Training <StatusBadge status={status} /></span>}
        actions={<>
          <Badge tone={conn === 'live' ? 'good' : conn === 'polling' ? 'warn' : undefined} title={conn === 'polling' ? 'WebSocket unavailable — polling every 2.5 s' : 'Live event stream'}>
            {conn === 'live' ? <Wifi size={11} /> : <WifiOff size={11} />}{conn}
          </Badge>
          <Button size="sm" variant="danger" icon={Square} onClick={cancel} loading={cancelling} disabled={!['queued', 'running'].includes(status)}>Cancel</Button>
        </>}>
        <div className="row wrap" style={{ gap: 24, marginBottom: 14 }}>
          <div><div className="eyebrow">Elapsed</div><div className="num" style={{ fontSize: 20, fontWeight: 600 }}>{elapsed !== null ? fmtDuration(elapsed) : status === 'queued' ? 'waiting' : '—'}</div></div>
          <div><div className="eyebrow">Stage</div><div style={{ fontSize: 15, fontWeight: 600 }}>{currentIdx >= 0 ? `${currentIdx + 1} of ${STAGES.length}` : status === 'queued' ? 'Queued for a worker' : 'Starting'}</div></div>
          {foldEvt && foldEvt.progress.current < foldEvt.progress.total + 1 && currentIdx === 2 && (
            <div><div className="eyebrow">CV fold</div><div className="num" style={{ fontSize: 20, fontWeight: 600 }}>{foldEvt.progress.current}/{foldEvt.progress.total}</div></div>
          )}
        </div>
        <div className="stage-list" role="list" aria-label="Training stages">
          {STAGES.map(([key, label], i) => {
            const state = status === 'failed' && i === currentIdx ? 'failed' : i < currentIdx || (i === currentIdx && status === 'completed') ? 'done' : i === currentIdx ? 'active' : 'pending';
            return (
              <motion.div key={key} role="listitem" className={`stage-row ${state}`} layout>
                <span className="stage-icon">
                  {state === 'done' ? <Check size={11} strokeWidth={3} /> : state === 'active' ? <Loader2 size={11} className="spin" /> : state === 'failed' ? <X size={11} /> : <span style={{ fontSize: 9 }}>{i + 1}</span>}
                </span>
                <div style={{ minWidth: 0 }}>
                  <div style={{ fontWeight: state === 'active' ? 600 : 500 }}>{label}</div>
                  {stageMsg[key] && state !== 'pending' && <div className="xs subtle truncate">{stageMsg[key]}</div>}
                  {key === 'cv' && foldEvt && state === 'active' && (
                    <div className="row" style={{ gap: 3, marginTop: 4 }} aria-label={`Fold ${foldEvt.progress.current} of ${foldEvt.progress.total}`}>
                      {Array.from({ length: foldEvt.progress.total }, (_, k) => (
                        <span key={k} style={{ width: 18, height: 4, borderRadius: 2, background: k < foldEvt.progress.current ? 'var(--accent)' : 'var(--surface-3)' }} />
                      ))}
                    </div>
                  )}
                </div>
                <span />
              </motion.div>
            );
          })}
        </div>
        <p className="xs subtle" style={{ marginTop: 10 }}>Progress is reported per stage and per CV fold as it actually happens. Fitting a single model has no reliable percentage, so none is shown.</p>
        <div className="eyebrow" style={{ margin: '16px 0 6px' }}>Event log</div>
        <div className="log" role="log" aria-live="polite">
          {events.length === 0 && <div className="subtle">Waiting for events…</div>}
          {events.map((e) => (
            <div key={e.seq} className={`log-line ${e.level || ''} ${e.error ? 'error' : ''}`}>
              <span className="t">{e.time ? new Date(e.time).toLocaleTimeString([], { hour12: false }) : ''}</span>
              <span className="s">{e.stage || e.status}</span>
              <span className="m">{e.message}</span>
            </div>
          ))}
        </div>
      </Panel>
      <div className="stack-lg">
        <Panel title="Configuration">
          <dl className="kv">
            <dt>Algorithm</dt><dd>{experiment.algorithm}</dd>
            <dt>Dataset</dt><dd className="truncate">{cfg.dataset_name || '—'}</dd>
            <dt>Target</dt><dd>{cfg.target} · {cfg.task_type}</dd>
            <dt>Test split</dt><dd className="num">{cfg.split ? `${Math.round(cfg.split.test_size * 100)}% · seed ${cfg.split.random_state}` : '—'}</dd>
            <dt>CV</dt><dd>{cfg.split?.cv_folds ? `${cfg.split.cv_folds}-fold` : 'off'}</dd>
            <dt>Features</dt><dd>{cfg.features ? `${cfg.features.numeric.length} num · ${cfg.features.categorical.length} cat` : '—'}</dd>
          </dl>
        </Panel>
      </div>
    </div>
  );
}
