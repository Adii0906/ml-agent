import { motion } from 'framer-motion';
import { useMemo, useState } from 'react';
import {
  Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart,
  Tooltip, XAxis, YAxis, ZAxis,
} from 'recharts';
import { fmtNum } from '../../lib/format';
import { useChartColors } from '../../lib/theme';

// ------------------------------------------------------------ colour helpers
function hexToRgb(hex) {
  const h = hex.replace('#', '');
  const n = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function mix(a, b, t) {
  if (!a?.startsWith('#') || !b?.startsWith('#')) return b || a;
  const [r1, g1, b1] = hexToRgb(a);
  const [r2, g2, b2] = hexToRgb(b);
  const c = (x, y) => Math.round(x + (y - x) * t);
  return `rgb(${c(r1, r2)}, ${c(g1, g2)}, ${c(b1, b2)})`;
}
function rampColor(stops, t) {
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.min(Math.floor(x), stops.length - 2);
  return mix(stops[i], stops[i + 1], x - i);
}
function fmtCorr(r) {
  if (Math.abs(r) >= 0.995) return r > 0 ? '1' : '−1';
  if (Math.abs(r) < 0.005) return '0';
  return r.toFixed(2).replace(/^(-?)0\./, '$1.').replace('-', '−');
}
function luminance(color) {
  const m = color.match(/\d+/g);
  if (!m) return 1;
  const [r, g, b] = m.map(Number);
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
}

const axisProps = (c) => ({
  stroke: c.chartAxis, tick: { fill: c.chartLabel, fontSize: 11, fontFamily: 'var(--font-mono)' }, tickLine: false,
});

export function ChartTooltip({ active, payload, title, rows }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="chart-tooltip">
      {title && <div className="tt-title">{typeof title === 'function' ? title(p) : title}</div>}
      {(rows ? rows(p) : payload.map((x) => [x.name, fmtNum(x.value)])).map(([k, v]) => (
        <div className="tt-row" key={k}><span>{k}</span><b>{v}</b></div>
      ))}
    </div>
  );
}

// ------------------------------------------------------------ histogram
export function Histogram({ edges, counts, height = 180, xLabel }) {
  const c = useChartColors();
  const data = useMemo(() => counts.map((n, i) => ({ lo: edges[i], hi: edges[i + 1], n, mid: (edges[i] + edges[i + 1]) / 2 })), [edges, counts]);
  if (!counts?.length) return null;
  const total = counts.reduce((a, b) => a + b, 0);
  return (
    <div className="chart" style={{ height }} role="img" aria-label={`Histogram${xLabel ? ` of ${xLabel}` : ''}`}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 6, right: 6, bottom: 0, left: -12 }} barCategoryGap={1}>
          <CartesianGrid vertical={false} stroke={c.chartGrid} />
          <XAxis dataKey="mid" {...axisProps(c)} tickFormatter={(v) => fmtNum(v, 2)} minTickGap={24} />
          <YAxis {...axisProps(c)} allowDecimals={false} width={44} />
          <Tooltip cursor={{ fill: c.surface2 }} content={<ChartTooltip title={(p) => `${fmtNum(p.lo)} – ${fmtNum(p.hi)}`}
            rows={(p) => [['Rows', p.n.toLocaleString()], ['Share', `${((p.n / total) * 100).toFixed(1)}%`]]} />} />
          <Bar dataKey="n" fill={c.series1} radius={[3, 3, 0, 0]} isAnimationActive animationDuration={450} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ------------------------------------------------------------ box plot
export function BoxPlot({ box, min, max, height = 64 }) {
  const c = useChartColors();
  const [hover, setHover] = useState(null);
  if (!box) return null;
  const lo = Math.min(min, box.whisker_low);
  const hi = Math.max(max, box.whisker_high);
  const span = hi - lo || 1;
  const x = (v) => 8 + ((v - lo) / span) * 84;
  const mid = 28;
  const stats = [['Lower whisker', box.whisker_low], ['Q1', box.q1], ['Median', box.median], ['Q3', box.q3], ['Upper whisker', box.whisker_high]];
  return (
    <div style={{ position: 'relative' }}>
      <svg viewBox="0 0 100 56" preserveAspectRatio="none" width="100%" height={height} role="img"
        aria-label={`Box plot: median ${fmtNum(box.median)}, IQR ${fmtNum(box.q1)} to ${fmtNum(box.q3)}`}
        onMouseLeave={() => setHover(null)}>
        <line x1={x(box.whisker_low)} x2={x(box.whisker_high)} y1={mid} y2={mid} stroke={c.chartAxis} strokeWidth="0.6" vectorEffect="non-scaling-stroke" />
        {[box.whisker_low, box.whisker_high].map((v, i) => <line key={i} x1={x(v)} x2={x(v)} y1={mid - 6} y2={mid + 6} stroke={c.text3} strokeWidth="1.5" vectorEffect="non-scaling-stroke" />)}
        <rect x={x(box.q1)} y={mid - 11} width={Math.max(0.5, x(box.q3) - x(box.q1))} height={22} rx="1.2" fill={mix(c.surface, c.series1, 0.22)} stroke={c.series1} strokeWidth="1.5" vectorEffect="non-scaling-stroke"
          onMouseEnter={() => setHover(stats)} />
        <line x1={x(box.median)} x2={x(box.median)} y1={mid - 11} y2={mid + 11} stroke={c.text} strokeWidth="2" vectorEffect="non-scaling-stroke" />
        {(box.outlier_sample || []).map((v, i) => <circle key={i} cx={x(v)} cy={mid + ((i % 5) - 2) * 2.2} r="0.9" fill={c.series2} opacity="0.8" />)}
      </svg>
      <div className="row-between xs subtle num" style={{ marginTop: -4 }}><span>{fmtNum(lo)}</span><span>{fmtNum(hi)}</span></div>
      {hover && (
        <div className="chart-tooltip" style={{ position: 'absolute', top: -8, right: 0, transform: 'translateY(-100%)', zIndex: 2 }}>
          {hover.map(([k, v]) => <div className="tt-row" key={k}><span>{k}</span><b>{fmtNum(v)}</b></div>)}
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------ bar list
export function BarList({ items, format = (v) => fmtNum(v), max, color, diverging = false, onSelect, selected }) {
  const c = useChartColors();
  const m = max ?? Math.max(...items.map((i) => Math.abs(i.value)), 1e-12);
  return (
    <div className="bar-list" role="list">
      {items.map((it, idx) => (
        <div key={it.label} className="bar-row" role="listitem" title={it.title || `${it.label}: ${format(it.value)}`}
          style={{ cursor: onSelect ? 'pointer' : undefined, opacity: selected && selected !== it.label ? 0.55 : 1 }}
          onClick={onSelect ? () => onSelect(it.label) : undefined}>
          <span className="label">{it.label}</span>
          <div className="bar-track">
            <motion.div className="bar-fill" initial={{ width: 0 }} animate={{ width: `${(Math.abs(it.value) / m) * 100}%` }}
              transition={{ duration: 0.5, delay: Math.min(idx * 0.02, 0.3), ease: [0.22, 1, 0.36, 1] }}
              style={{ background: it.color || (diverging && it.value < 0 ? c.series8 : color || c.series1) }} />
            {it.err ? <span style={{ position: 'absolute', top: 3, height: 2, background: c.text3, left: `${(Math.max(0, it.value - it.err) / m) * 100}%`, width: `${((2 * it.err) / m) * 100}%` }} /> : null}
          </div>
          <span className="value">{format(it.value)}{it.sub && <span className="subtle"> {it.sub}</span>}</span>
        </div>
      ))}
    </div>
  );
}

// ------------------------------------------------------------ correlation heatmap
export function CorrelationHeatmap({ columns, matrix, onSelect }) {
  const c = useChartColors();
  const [hover, setHover] = useState(null);
  const n = columns.length;
  const cell = (r) => {
    if (r === null || r === undefined) return c.surface2;
    return r < 0 ? mix(c.divMid, c.divNeg, Math.abs(r)) : mix(c.divMid, c.divPos, r);
  };
  const showText = n <= 12;
  return (
    <div>
      <div className="heatmap" style={{ gridTemplateColumns: `minmax(70px, 130px) repeat(${n}, minmax(14px, ${n <= 6 ? 64 : 48}px))`, justifyContent: 'start' }} role="table" aria-label="Correlation matrix">
        {matrix.map((row, i) => (
          <div key={columns[i]} style={{ display: 'contents' }} role="row">
            <div className="heat-label" role="rowheader" title={columns[i]}>{columns[i]}</div>
            {row.map((r, j) => {
              const bg = cell(r);
              return (
                <div key={j} role="cell" className="heat-cell" style={{ background: bg, color: luminance(bg) < 0.55 ? '#fff' : '#111', cursor: onSelect && i !== j ? 'pointer' : 'default' }}
                  onMouseEnter={() => setHover({ a: columns[i], b: columns[j], r })} onMouseLeave={() => setHover(null)}
                  onClick={() => onSelect && i !== j && onSelect(columns[i], columns[j])}
                  aria-label={`${columns[i]} vs ${columns[j]}: ${r === null ? 'n/a' : r.toFixed(2)}`}>
                  {showText && r !== null ? fmtCorr(r) : ''}
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <div className="row-between" style={{ marginTop: 10, gap: 16, flexWrap: 'wrap' }}>
        <div className="row xs subtle" style={{ gap: 6 }}>
          <span>−1</span>
          <div style={{ width: 140, height: 8, borderRadius: 4, background: `linear-gradient(90deg, ${c.divNeg}, ${c.divMid}, ${c.divPos})` }} />
          <span>+1</span>
          <span style={{ marginLeft: 6 }}>Pearson r</span>
        </div>
        <div className="small num" style={{ minHeight: 18 }} aria-live="polite">
          {hover ? <>{hover.a} × {hover.b}: <b>{hover.r === null ? 'n/a' : hover.r.toFixed(3)}</b></> : <span className="subtle">Hover a cell{onSelect ? ' · click to plot the pair' : ''}</span>}
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------ confusion matrix
export function ConfusionMatrix({ labels, matrix, normalize = false }) {
  const c = useChartColors();
  const [hover, setHover] = useState(null);
  const rowSums = matrix.map((r) => r.reduce((a, b) => a + b, 0));
  const maxVal = normalize ? 1 : Math.max(...matrix.flat(), 1);
  const n = labels.length;
  const truncate = (s) => (s.length > 14 ? `${s.slice(0, 13)}…` : s);
  return (
    <div>
      <div className="row xs subtle" style={{ justifyContent: 'center', marginBottom: 4 }}>Predicted →</div>
      <div className="cm-grid" style={{ gridTemplateColumns: `minmax(60px, 110px) repeat(${n}, minmax(34px, 1fr))` }} role="table" aria-label="Confusion matrix">
        <div />
        {labels.map((l) => <div key={l} className="heat-label xs" style={{ justifyContent: 'center' }} title={l}>{truncate(l)}</div>)}
        {matrix.map((row, i) => (
          <div key={labels[i]} style={{ display: 'contents' }} role="row">
            <div className="heat-label xs" title={labels[i]} role="rowheader">{truncate(labels[i])}</div>
            {row.map((v, j) => {
              const share = rowSums[i] ? v / rowSums[i] : 0;
              const t = normalize ? share : v / maxVal;
              const bg = v === 0 ? c.surface2 : rampColor(c.seq.slice(1), t);
              const fg = luminance(bg) < 0.55 ? '#fff' : '#111';
              return (
                <div key={j} role="cell" className="cm-cell" style={{ background: bg, color: v === 0 ? c.text3 : fg, boxShadow: i === j ? `inset 0 0 0 1.5px ${c.text}` : undefined }}
                  onMouseEnter={() => setHover({ a: labels[i], p: labels[j], v, share })} onMouseLeave={() => setHover(null)}
                  aria-label={`Actual ${labels[i]}, predicted ${labels[j]}: ${v}`}>
                  {normalize ? `${Math.round(share * 100)}%` : v}
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <div className="small num" style={{ marginTop: 8, minHeight: 18 }} aria-live="polite">
        {hover ? <>Actual <b>{hover.a}</b> → predicted <b>{hover.p}</b>: {hover.v} rows ({(hover.share * 100).toFixed(1)}% of actual {hover.a})</>
          : <span className="subtle">Rows = actual class · diagonal (outlined) = correct</span>}
      </div>
    </div>
  );
}

// ------------------------------------------------------------ scatter
export function ScatterPlot({ points, xLabel, yLabel, groups, height = 320, diagonal = false, zeroLine = false, xKey = 'x', yKey = 'y', tooltipRows }) {
  const c = useChartColors();
  const series = useMemo(() => {
    if (!groups) return [{ name: 'all', data: points, color: c.series1 }];
    return groups.map((g, i) => ({ name: g, data: points.filter((p) => p.g === g), color: g === 'Other' ? c.seriesMuted : c.series[i] }));
  }, [points, groups, c]);
  const domain = useMemo(() => {
    if (!diagonal || !points.length) return null;
    const vals = points.flatMap((p) => [p[xKey], p[yKey]]);
    return [Math.min(...vals), Math.max(...vals)];
  }, [points, diagonal, xKey, yKey]);
  return (
    <div>
      <div className="xs subtle" style={{ marginBottom: 2 }}>↑ {yLabel}</div>
      <div className="chart" style={{ height }} role="img" aria-label={`Scatter plot of ${yLabel} against ${xLabel}`}>
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 6, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid stroke={c.chartGrid} />
            <XAxis type="number" dataKey={xKey} name={xLabel} {...axisProps(c)} domain={domain || ['auto', 'auto']} tickFormatter={(v) => fmtNum(v, 2)} />
            <YAxis type="number" dataKey={yKey} name={yLabel} {...axisProps(c)} domain={domain || ['auto', 'auto']} tickFormatter={(v) => fmtNum(v, 2)} width={56} />
            <ZAxis range={[22, 22]} />
            {diagonal && domain && <ReferenceLine segment={[{ x: domain[0], y: domain[0] }, { x: domain[1], y: domain[1] }]} stroke={c.text3} strokeDasharray="0" ifOverflow="extendDomain" />}
            {zeroLine && <ReferenceLine y={0} stroke={c.text3} />}
            <Tooltip cursor={{ stroke: c.chartAxis }} content={<ChartTooltip title={(p) => p.g}
              rows={tooltipRows || ((p) => [[xLabel, fmtNum(p[xKey])], [yLabel, fmtNum(p[yKey])]])} />} />
            {series.map((s) => (
              <Scatter key={s.name} name={s.name} data={s.data} fill={s.color} fillOpacity={0.7} stroke={c.surface} strokeWidth={0.5} isAnimationActive={s.data.length < 800} animationDuration={400} />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <div className="xs subtle" style={{ textAlign: 'right' }}>{xLabel} →</div>
      {groups && (
        <div className="chart-legend" style={{ marginTop: 6 }}>
          {series.map((s) => <span key={s.name} className="row" style={{ gap: 6 }}><span className="legend-swatch" style={{ background: s.color }} />{s.name}</span>)}
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------ line charts
export function RocChart({ curves, height = 280 }) {
  const c = useChartColors();
  return (
    <div>
      <div className="xs subtle" style={{ marginBottom: 2 }}>↑ True positive rate</div>
      <div className="chart" style={{ height }} role="img" aria-label="ROC curve">
        <ResponsiveContainer>
          <LineChart margin={{ top: 6, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid stroke={c.chartGrid} />
            <XAxis type="number" dataKey="fpr" domain={[0, 1]} {...axisProps(c)} tickFormatter={(v) => v.toFixed(1)} />
            <YAxis type="number" dataKey="tpr" domain={[0, 1]} {...axisProps(c)} tickFormatter={(v) => v.toFixed(1)} width={40} />
            <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={c.text3} />
            <Tooltip content={<ChartTooltip rows={(p) => [['FPR', p.fpr.toFixed(3)], ['TPR', p.tpr.toFixed(3)]]} />} />
            {curves.map((cv, i) => (
              <Line key={cv.label} data={cv.points} dataKey="tpr" name={cv.label} type="linear" dot={false} stroke={c.series[i % 8]} strokeWidth={2} isAnimationActive animationDuration={500} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="xs subtle" style={{ textAlign: 'right' }}>False positive rate →</div>
      {curves.length > 1 && (
        <div className="chart-legend" style={{ marginTop: 6 }}>
          {curves.map((cv, i) => <span key={cv.label} className="row" style={{ gap: 6 }}><span className="legend-swatch" style={{ background: c.series[i % 8] }} />{cv.label} vs rest</span>)}
        </div>
      )}
    </div>
  );
}

export function LearningCurveChart({ lc, height = 260 }) {
  const c = useChartColors();
  const data = lc.train_sizes.map((n, i) => ({ n, train: lc.train_mean[i], val: lc.val_mean[i], trainStd: lc.train_std[i], valStd: lc.val_std[i] }));
  return (
    <div>
      <div className="chart" style={{ height }} role="img" aria-label="Learning curve">
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 6, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid stroke={c.chartGrid} />
            <XAxis dataKey="n" type="number" domain={['dataMin', 'dataMax']} {...axisProps(c)} />
            <YAxis {...axisProps(c)} domain={['auto', 'auto']} tickFormatter={(v) => v.toFixed(2)} width={44} />
            <Tooltip content={<ChartTooltip title={(p) => `${p.n} training rows`} rows={(p) => [['Train', `${p.train.toFixed(3)} ± ${p.trainStd.toFixed(3)}`], ['Validation', `${p.val.toFixed(3)} ± ${p.valStd.toFixed(3)}`]]} />} />
            <Line dataKey="train" name="Train" stroke={c.series2} strokeWidth={2} dot={{ r: 3 }} />
            <Line dataKey="val" name="Validation" stroke={c.series1} strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="xs subtle" style={{ textAlign: 'right' }}>Training rows →</div>
      <div className="chart-legend" style={{ marginTop: 6 }}>
        <span className="row" style={{ gap: 6 }}><span className="legend-swatch" style={{ background: c.series2 }} />Train ({lc.metric})</span>
        <span className="row" style={{ gap: 6 }}><span className="legend-swatch" style={{ background: c.series1 }} />Validation ({lc.metric})</span>
      </div>
    </div>
  );
}

// ------------------------------------------------------------ simple bars (CV folds, residual hist)
export function MiniBars({ data, dataKey, labelKey, height = 140, format = (v) => fmtNum(v), refLine }) {
  const c = useChartColors();
  return (
    <div className="chart" style={{ height }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 6, right: 6, bottom: 0, left: -12 }}>
          <CartesianGrid vertical={false} stroke={c.chartGrid} />
          <XAxis dataKey={labelKey} {...axisProps(c)} />
          <YAxis {...axisProps(c)} width={44} tickFormatter={(v) => fmtNum(v, 2)} domain={['auto', 'auto']} />
          {refLine !== undefined && <ReferenceLine y={refLine} stroke={c.text3} />}
          <Tooltip cursor={{ fill: c.surface2 }} content={<ChartTooltip title={(p) => p[labelKey]} rows={(p) => [[dataKey, format(p[dataKey])]]} />} />
          <Bar dataKey={dataKey} fill={c.series1} radius={[3, 3, 0, 0]} maxBarSize={36}>
            {data.map((d, i) => <Cell key={i} fill={c.series1} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ------------------------------------------------------------ missing map
export function MissingMap({ columns, grid }) {
  if (!columns.length) return null;
  return (
    <div>
      <div className="miss-map" style={{ gridTemplateColumns: `repeat(${columns.length}, minmax(6px, 1fr))` }} role="img"
        aria-label={`Missing value map for ${columns.length} columns`}>
        {grid.map((row, i) => row.map((v, j) => <div key={`${i}-${j}`} className={`miss-cell ${v ? 'on' : ''}`} />))}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${columns.length}, minmax(6px, 1fr))`, gap: 1, marginTop: 6 }}>
        {columns.map((col) => (
          <div key={col} className="xs subtle" title={col}
            style={{ writingMode: columns.length > 8 ? 'vertical-rl' : undefined, transform: columns.length > 8 ? 'rotate(180deg)' : undefined, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxHeight: 90, textAlign: columns.length > 8 ? 'left' : 'center' }}>
            {col}
          </div>
        ))}
      </div>
    </div>
  );
}
