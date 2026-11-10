import { AnimatePresence, motion } from 'framer-motion';
import { AlertCircle, CheckCircle2, Info, X } from 'lucide-react';
import { createContext, useCallback, useContext, useMemo, useRef, useState } from 'react';

const ToastCtx = createContext(null);

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);
  const nextId = useRef(1);
  const dismiss = useCallback((id) => setToasts((t) => t.filter((x) => x.id !== id)), []);
  const push = useCallback((toast) => {
    const id = nextId.current++;
    setToasts((t) => [...t.slice(-3), { id, kind: 'info', ...toast }]);
    setTimeout(() => dismiss(id), toast.duration ?? (toast.kind === 'error' ? 7000 : 4000));
    return id;
  }, [dismiss]);
  const api = useMemo(() => ({
    push,
    success: (title, message) => push({ kind: 'success', title, message }),
    error: (title, message) => push({ kind: 'error', title, message }),
    info: (title, message) => push({ kind: 'info', title, message }),
  }), [push]);
  const icon = { success: <CheckCircle2 size={16} color="var(--good)" />, error: <AlertCircle size={16} color="var(--bad)" />, info: <Info size={16} color="var(--info)" /> };
  return (
    <ToastCtx.Provider value={api}>
      {children}
      <div className="toasts" role="status" aria-live="polite">
        <AnimatePresence initial={false}>
          {toasts.map((t) => (
            <motion.div key={t.id} className="toast" layout initial={{ opacity: 0, y: 12, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, x: 24 }} transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}>
              <span style={{ marginTop: 1 }}>{icon[t.kind]}</span>
              <div className="grow">
                <div className="toast-title">{t.title}</div>
                {t.message && <div className="muted">{t.message}</div>}
              </div>
              <button className="btn btn-ghost btn-sm btn-icon" onClick={() => dismiss(t.id)} aria-label="Dismiss notification"><X size={14} /></button>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </ToastCtx.Provider>
  );
}

export const useToast = () => useContext(ToastCtx);
