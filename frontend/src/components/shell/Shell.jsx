import { AnimatePresence, motion } from 'framer-motion';
import {
  BookOpen, Command, Database, FileText, FlaskConical, GitCompare, Layers, LayoutDashboard, Menu, Monitor, Moon,
  PanelLeftClose, PanelLeftOpen, Search, Settings, Sun, Table2,
} from 'lucide-react';
import { Suspense, useEffect, useMemo, useRef, useState } from 'react';
import { Link, NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom';
import { useLocalStorage } from '../../lib/hooks';
import { useTheme } from '../../lib/theme';
import { useWorkspace } from '../../lib/workspace';
import { Segmented, SkeletonBlock } from '../ui';

export function Logo({ size = 22 }) {
  // A tiny decision boundary: two point clouds split by a line.
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <rect x="0.5" y="0.5" width="23" height="23" rx="6" fill="var(--text)" />
      <path d="M5 18.5 L19 5.5" stroke="var(--bg)" strokeWidth="1.6" strokeLinecap="round" />
      <circle cx="7" cy="8" r="1.7" fill="var(--series-1)" />
      <circle cx="10.5" cy="6" r="1.7" fill="var(--series-1)" />
      <circle cx="6.5" cy="12" r="1.7" fill="var(--series-1)" />
      <circle cx="17" cy="16" r="1.7" fill="var(--series-2)" />
      <circle cx="13.5" cy="18" r="1.7" fill="var(--series-2)" />
      <circle cx="18" cy="11.5" r="1.7" fill="var(--series-2)" />
    </svg>
  );
}

const NAV = [
  { group: 'Home', items: [{ to: '/', label: 'Overview', icon: LayoutDashboard, end: true }] },
  { group: 'Data', items: [
    { to: '/datasets', label: 'Datasets', icon: Database, count: 'datasets', end: true },
    { to: '/explorer', label: 'Data Explorer', icon: Table2, match: /^\/(explorer|datasets\/[^/]+)/ },
  ] },
  { group: 'Experimentation', items: [
    { to: '/studio', label: 'Model Studio', icon: FlaskConical },
    { to: '/experiments', label: 'Experiments', icon: Layers, count: 'experiments' },
    { to: '/compare', label: 'Model Comparison', icon: GitCompare },
  ] },
  { group: 'Intelligence', items: [
    { to: '/research', label: 'Research Papers', icon: BookOpen },
    { to: '/reports', label: 'Insights & Reports', icon: FileText },
  ] },
];

function Sidebar({ collapsed, onToggle, mobileOpen, onNavigate }) {
  const { datasets, experiments, activeCount } = useWorkspace();
  const location = useLocation();
  const counts = { datasets: datasets?.length, experiments: experiments?.length };
  return (
    <aside className={`sidebar ${mobileOpen ? 'open' : ''}`} aria-label="Primary">
      <div className="sidebar-brand">
        <Link to="/" className="row" style={{ color: 'var(--text)', textDecoration: 'none', gap: 10 }} onClick={onNavigate}>
          <Logo />
          <span className="brand-name">ML-Agent <span>workspace</span></span>
        </Link>
      </div>
      <nav className="sidebar-nav">
        {NAV.map((g) => (
          <div className="nav-group" key={g.group}>
            <div className="nav-group-label eyebrow">{g.group}</div>
            {g.items.map((item) => {
              const forced = item.match?.test(location.pathname) && location.pathname !== '/datasets';
              return (
                <NavLink key={item.to} to={item.to} end={item.end} onClick={onNavigate} title={collapsed ? item.label : undefined}
                  className={({ isActive }) => `nav-item ${isActive || forced ? 'active' : ''}`}>
                  {({ isActive }) => (
                    <>
                      {(isActive || forced) && <motion.span layoutId="nav-indicator" className="nav-indicator" transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }} />}
                      <item.icon size={16} strokeWidth={1.8} />
                      <span className="nav-label">{item.label}</span>
                      {item.to === '/experiments' && activeCount > 0 ? (
                        <span className="nav-count" style={{ color: 'var(--accent-text)' }} title={`${activeCount} running`}>● {activeCount}</span>
                      ) : item.count && counts[item.count] ? <span className="nav-count">{counts[item.count]}</span> : null}
                    </>
                  )}
                </NavLink>
              );
            })}
          </div>
        ))}
      </nav>
      <div className="sidebar-foot">
        <NavLink to="/settings" onClick={onNavigate} className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`} title={collapsed ? 'Settings' : undefined}>
          <Settings size={16} strokeWidth={1.8} /><span className="nav-label">Settings</span>
        </NavLink>
        <button className="nav-item desktop-only" style={{ border: 0, background: 'none', cursor: 'pointer', width: '100%' }} onClick={onToggle}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'} title={collapsed ? 'Expand sidebar' : undefined}>
          {collapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
          <span className="nav-label">Collapse</span>
        </button>
      </div>
    </aside>
  );
}

function useCrumbs() {
  const { pathname } = useLocation();
  const { datasets, experiments } = useWorkspace();
  return useMemo(() => {
    const parts = pathname.split('/').filter(Boolean);
    const out = [{ label: 'ML-Agent', to: '/' }];
    const names = { datasets: 'Datasets', explorer: 'Data Explorer', studio: 'Model Studio', experiments: 'Experiments', compare: 'Model Comparison', research: 'Research Papers', reports: 'Insights & Reports', settings: 'Settings' };
    if (!parts.length) return [...out, { label: 'Overview' }];
    out.push({ label: names[parts[0]] || parts[0], to: `/${parts[0]}` });
    if (parts[1]) {
      let label = parts[1];
      if (parts[0] === 'datasets') label = datasets?.find((d) => d.id === parts[1])?.name || 'Dataset';
      if (parts[0] === 'experiments' || parts[0] === 'reports') label = experiments?.find((e) => e.id === parts[1])?.name || 'Experiment';
      out.push({ label });
    }
    return out;
  }, [pathname, datasets, experiments]);
}

function ThemeSwitch() {
  const { preference, setPreference } = useTheme();
  return (
    <Segmented ariaLabel="Colour theme" value={preference} onChange={setPreference} options={[
      { value: 'light', icon: Sun, title: 'Light theme' },
      { value: 'system', icon: Monitor, title: 'Match system' },
      { value: 'dark', icon: Moon, title: 'Dark theme' },
    ]} />
  );
}

function Header({ onMenu, onCommand }) {
  const crumbs = useCrumbs();
  const { activeDataset } = useWorkspace();
  return (
    <header className="header">
      <button className="btn btn-ghost btn-icon mobile-only" onClick={onMenu} aria-label="Open navigation"><Menu size={18} /></button>
      <nav className="crumbs" aria-label="Breadcrumb">
        {crumbs.map((c, i) => (
          <span key={i} className="row" style={{ gap: 6, minWidth: 0 }}>
            {i > 0 && <span aria-hidden="true">/</span>}
            {c.to && i < crumbs.length - 1 ? <Link className="crumb" to={c.to}>{c.label}</Link>
              : <span className={`crumb ${i === crumbs.length - 1 ? 'current' : ''}`} aria-current={i === crumbs.length - 1 ? 'page' : undefined}>{c.label}</span>}
          </span>
        ))}
      </nav>
      {activeDataset && (
        <Link to={`/datasets/${activeDataset.id}`} className="context-chip desktop-only" title="Active dataset">
          <Database size={12} />
          <span className="truncate">{activeDataset.name}</span>
          <span className="num subtle">{activeDataset.n_rows?.toLocaleString()}×{activeDataset.n_cols}</span>
        </Link>
      )}
      <button className="btn btn-sm desktop-only" onClick={onCommand} aria-label="Open command palette" style={{ color: 'var(--text-3)', gap: 8 }}>
        <Search size={13} /> Jump to… <kbd><Command size={10} />K</kbd>
      </button>
      <ThemeSwitch />
    </header>
  );
}

function CommandPalette({ open, onClose }) {
  const navigate = useNavigate();
  const { datasets, experiments } = useWorkspace();
  const [q, setQ] = useState('');
  const [sel, setSel] = useState(0);
  const inputRef = useRef(null);
  useEffect(() => { if (open) { setQ(''); setSel(0); setTimeout(() => inputRef.current?.focus(), 20); } }, [open]);
  const items = useMemo(() => {
    const pages = NAV.flatMap((g) => g.items).concat([{ to: '/settings', label: 'Settings', icon: Settings }])
      .map((i) => ({ group: 'Go to', label: i.label, icon: i.icon, to: i.to }));
    const ds = (datasets || []).slice(0, 30).map((d) => ({ group: 'Datasets', label: d.name, icon: Database, to: `/datasets/${d.id}`, meta: `${d.n_rows} rows` }));
    const ex = (experiments || []).slice(0, 30).map((e) => ({ group: 'Experiments', label: e.name, icon: Layers, to: `/experiments/${e.id}`, meta: e.status }));
    const all = [...pages, { group: 'Actions', label: 'Upload a dataset', icon: Database, to: '/datasets?upload=1' }, { group: 'Actions', label: 'New experiment', icon: FlaskConical, to: '/studio' }, ...ds, ...ex];
    const needle = q.trim().toLowerCase();
    return needle ? all.filter((i) => i.label.toLowerCase().includes(needle)) : all;
  }, [q, datasets, experiments]);
  const go = (item) => { onClose(); navigate(item.to); };
  const onKey = (e) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setSel((s) => Math.min(s + 1, items.length - 1)); }
    if (e.key === 'ArrowUp') { e.preventDefault(); setSel((s) => Math.max(s - 1, 0)); }
    if (e.key === 'Enter' && items[sel]) go(items[sel]);
    if (e.key === 'Escape') onClose();
  };
  let lastGroup = null;
  return (
    <AnimatePresence>
      {open && (
        <motion.div className="cmdk" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
          <motion.div className="cmdk-panel" role="dialog" aria-modal="true" aria-label="Command palette"
            initial={{ y: -8, scale: 0.98 }} animate={{ y: 0, scale: 1 }} exit={{ y: -8, scale: 0.98 }} transition={{ duration: 0.16 }}>
            <input ref={inputRef} className="cmdk-input" placeholder="Search pages, datasets, experiments…" value={q}
              onChange={(e) => { setQ(e.target.value); setSel(0); }} onKeyDown={onKey} aria-label="Search" role="combobox" aria-expanded="true" aria-controls="cmdk-list" />
            <div className="cmdk-list" id="cmdk-list" role="listbox">
              {items.length === 0 && <div className="empty small">No matches</div>}
              {items.map((item, i) => {
                const header = item.group !== lastGroup ? <div className="cmdk-group eyebrow" key={`g-${item.group}`}>{item.group}</div> : null;
                lastGroup = item.group;
                return (
                  <div key={`${item.group}-${item.to}-${i}`}>
                    {header}
                    <button className="cmdk-item" role="option" aria-selected={i === sel} onMouseEnter={() => setSel(i)} onClick={() => go(item)}>
                      <item.icon size={15} /><span className="truncate grow">{item.label}</span>
                      {item.meta && <span className="subtle small">{item.meta}</span>}
                    </button>
                  </div>
                );
              })}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export function AppShell() {
  const [collapsed, setCollapsed] = useLocalStorage('mlagent.sidebarCollapsed', false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [cmdOpen, setCmdOpen] = useState(false);
  const location = useLocation();
  useEffect(() => { setMobileOpen(false); }, [location.pathname]);
  useEffect(() => {
    const onKey = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); setCmdOpen((o) => !o); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  const section = location.pathname.split('/').slice(0, 3).join('/');
  return (
    <div className={`shell ${collapsed ? 'collapsed' : ''}`}>
      <a href="#main" className="sr-only">Skip to content</a>
      <Sidebar collapsed={collapsed} onToggle={() => setCollapsed(!collapsed)} mobileOpen={mobileOpen} onNavigate={() => setMobileOpen(false)} />
      {mobileOpen && <div className="sidebar-scrim" onClick={() => setMobileOpen(false)} aria-hidden="true" />}
      <div className="main-col">
        <Header onMenu={() => setMobileOpen(true)} onCommand={() => setCmdOpen(true)} />
        <main className="content" id="main">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div key={section} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              transition={{ duration: 0.18, ease: [0.22, 1, 0.36, 1] }}>
              <Suspense fallback={<div className="page"><div style={{ maxWidth: 520 }}><SkeletonBlock rows={5} /></div></div>}>
                <Outlet />
              </Suspense>
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
      <CommandPalette open={cmdOpen} onClose={() => setCmdOpen(false)} />
    </div>
  );
}
