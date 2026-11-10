export const fmtInt = (n) => (n === null || n === undefined ? '—' : Number(n).toLocaleString('en-US'));

export function fmtNum(v, digits = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  if (typeof v !== 'number') return String(v);
  if (Number.isInteger(v) && Math.abs(v) < 1e9) return v.toLocaleString('en-US');
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(2);
  return v.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: a < 10 ? Math.min(digits, 3) : 0 });
}

export const fmtPct = (v, digits = 1) => (v === null || v === undefined ? '—' : `${(v * 100).toFixed(digits)}%`);

export function fmtBytes(b) {
  if (b === null || b === undefined) return '—';
  const u = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  let n = b;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i += 1; }
  return `${n.toFixed(n < 10 && i ? 1 : 0)} ${u[i]}`;
}

export function fmtDuration(s) {
  if (s === null || s === undefined) return '—';
  if (s < 1) return `${Math.round(s * 1000)} ms`;
  if (s < 60) return `${s.toFixed(s < 10 ? 2 : 1)} s`;
  const m = Math.floor(s / 60);
  return `${m}m ${Math.round(s % 60)}s`;
}

export function timeAgo(iso) {
  if (!iso) return '—';
  const t = new Date(iso).getTime();
  const d = (Date.now() - t) / 1000;
  if (d < 45) return 'just now';
  if (d < 3600) return `${Math.round(d / 60)}m ago`;
  if (d < 86400) return `${Math.round(d / 3600)}h ago`;
  if (d < 86400 * 7) return `${Math.round(d / 86400)}d ago`;
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

export const fmtDate = (iso) => (iso ? new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) : '—');

/** Metric definitions: label, direction (higher is better?), plain-language meaning. */
export const METRICS = {
  accuracy: { label: 'Accuracy', higher: true, help: 'Share of test rows predicted correctly. Misleading when classes are imbalanced.' },
  balanced_accuracy: { label: 'Balanced accuracy', higher: true, help: 'Average recall across classes — every class counts equally regardless of size.' },
  f1_macro: { label: 'F1 (macro)', higher: true, help: 'Harmonic mean of precision and recall, averaged equally over classes.' },
  f1_weighted: { label: 'F1 (weighted)', higher: true, help: 'F1 averaged over classes, weighted by class frequency.' },
  precision_macro: { label: 'Precision (macro)', higher: true, help: 'Of the rows predicted as a class, the share that truly belong to it (class average).' },
  recall_macro: { label: 'Recall (macro)', higher: true, help: 'Of the rows that truly belong to a class, the share the model found (class average).' },
  precision_weighted: { label: 'Precision (weighted)', higher: true, help: 'Precision weighted by class frequency.' },
  recall_weighted: { label: 'Recall (weighted)', higher: true, help: 'Recall weighted by class frequency.' },
  roc_auc: { label: 'ROC-AUC', higher: true, help: 'Probability that a random positive is ranked above a random negative. 0.5 = chance.' },
  r2: { label: 'R²', higher: true, help: 'Share of target variance explained. 0 = predicting the mean; negative = worse than the mean.' },
  rmse: { label: 'RMSE', higher: false, help: 'Root mean squared error, in target units. Penalises large errors.' },
  mae: { label: 'MAE', higher: false, help: 'Mean absolute error, in target units.' },
  mse: { label: 'MSE', higher: false, help: 'Mean squared error (target units squared).' },
  median_ae: { label: 'Median AE', higher: false, help: 'Median absolute error — robust to outliers.' },
  max_error: { label: 'Max error', higher: false, help: 'Largest absolute error on a single test row.' },
  mape: { label: 'MAPE', higher: false, help: 'Mean absolute percentage error. Undefined when the target contains zeros.' },
};

export const CLASSIFICATION_KEYS = ['accuracy', 'balanced_accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'f1_weighted'];
export const REGRESSION_KEYS = ['r2', 'rmse', 'mae', 'median_ae', 'max_error', 'mape'];

export function primaryValue(exp) {
  const m = exp?.metrics || {};
  if (m.task_type === 'classification') return { key: 'accuracy', value: m.accuracy };
  if (m.task_type === 'regression') return { key: 'r2', value: m.r2_score };
  return { key: null, value: null };
}

export const titleCase = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
