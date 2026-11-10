import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';

const ThemeCtx = createContext(null);
const KEY = 'mlagent.theme';

function readStored() {
  try { return localStorage.getItem(KEY) || 'system'; } catch { return 'system'; }
}

function systemDark() {
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches;
}

export function ThemeProvider({ children }) {
  const [preference, setPreference] = useState(readStored);
  const [systemIsDark, setSystemIsDark] = useState(systemDark);

  useEffect(() => {
    const mq = window.matchMedia?.('(prefers-color-scheme: dark)');
    if (!mq) return undefined;
    const onChange = (e) => setSystemIsDark(e.matches);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);

  const resolved = preference === 'system' ? (systemIsDark ? 'dark' : 'light') : preference;

  useEffect(() => {
    const root = document.documentElement;
    if (preference === 'system') root.removeAttribute('data-theme');
    else root.setAttribute('data-theme', preference);
    try { localStorage.setItem(KEY, preference); } catch { /* storage unavailable */ }
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', resolved === 'dark' ? '#0f0f0e' : '#f5f4f0');
  }, [preference, resolved]);

  const value = useMemo(() => ({ preference, setPreference, resolved }), [preference, resolved]);
  return <ThemeCtx.Provider value={value}>{children}</ThemeCtx.Provider>;
}

export const useTheme = () => useContext(ThemeCtx);

const CHART_VARS = ['series-1', 'series-2', 'series-3', 'series-4', 'series-5', 'series-6', 'series-7', 'series-8',
  'series-muted', 'chart-grid', 'chart-axis', 'chart-label', 'text', 'text-2', 'text-3', 'surface', 'surface-2',
  'border', 'accent', 'good', 'bad', 'warn', 'status-good', 'status-critical', 'div-neg', 'div-mid', 'div-pos',
  'seq-0', 'seq-1', 'seq-2', 'seq-3', 'seq-4', 'seq-5', 'seq-6', 'seq-7'];

/** Resolved chart colours for the active theme (SVG attributes need concrete values). */
export function useChartColors() {
  const { resolved } = useTheme();
  const read = useCallback(() => {
    const cs = getComputedStyle(document.documentElement);
    const out = {};
    CHART_VARS.forEach((v) => { out[v.replace(/-(\w)/g, (_, c) => c.toUpperCase())] = cs.getPropertyValue(`--${v}`).trim(); });
    out.series = [1, 2, 3, 4, 5, 6, 7, 8].map((i) => out[`series${i}`]);
    out.seq = [0, 1, 2, 3, 4, 5, 6, 7].map((i) => out[`seq${i}`]);
    return out;
  }, []);
  const [colors, setColors] = useState(read);
  useEffect(() => { setColors(read()); }, [resolved, read]);
  return colors;
}
