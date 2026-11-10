import { motion } from 'framer-motion';
import { BookOpen, ExternalLink, FileDown, History, Search, Sparkles } from 'lucide-react';
import { useState } from 'react';
import { Badge, Button, Callout, Disclaimer, EmptyState, ErrorState, PageHeader, Panel, Prov, Segmented, SkeletonBlock, Switch } from '../components/ui';
import { api } from '../lib/api';
import { useLocalStorage } from '../lib/hooks';
import { useWorkspace } from '../lib/workspace';

function Paper({ p, note, context, aiAvailable, index }) {
  const [open, setOpen] = useState(false);
  const [summary, setSummary] = useState(null);
  const summarize = async () => {
    setSummary({ loading: true });
    try { setSummary({ data: await api.summarizePaper({ title: p.title, abstract: p.abstract, context: context || undefined }) }); } catch (e) { setSummary({ error: e.message }); }
  };
  const authors = p.authors.length > 6 ? `${p.authors.slice(0, 6).join(', ')} +${p.authors.length - 6}` : p.authors.join(', ');
  const s = summary?.data?.content;
  return (
    <motion.article className="paper" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: Math.min(index * 0.03, 0.3) }}>
      <div className="row wrap" style={{ gap: 6 }}>
        <Badge className="mono" title="arXiv identifier">arXiv:{p.arxiv_id}</Badge>
        {p.primary_category && <Badge tone="info">{p.primary_category}</Badge>}
        <span className="small subtle">{p.published ? new Date(p.published).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' }) : ''}
          {p.updated && p.updated.slice(0, 10) !== p.published?.slice(0, 10) ? ` · updated ${new Date(p.updated).toLocaleDateString()}` : ''}</span>
      </div>
      <a className="paper-title" href={p.abs_url} target="_blank" rel="noopener noreferrer">{p.title}</a>
      <div className="small muted">{authors}</div>
      {note && <div className="ai-block small"><Prov kind="ai" /> <span style={{ marginLeft: 6 }}>{note}</span></div>}
      <div>
        <div className="eyebrow" style={{ marginBottom: 4 }}>Original abstract · arXiv</div>
        <p className="paper-abstract" style={open ? undefined : { display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>{p.abstract}</p>
        <button className="btn btn-ghost btn-sm" style={{ marginLeft: -8 }} onClick={() => setOpen(!open)} aria-expanded={open}>{open ? 'Show less' : 'Read full abstract'}</button>
      </div>
      {summary?.loading && <SkeletonBlock rows={3} />}
      {summary?.error && <Callout tone="bad" title="Summary failed.">{summary.error}</Callout>}
      {s && (
        <div className="ai-block stack-sm">
          <div className="row" style={{ gap: 8 }}><Prov kind="ai" /><span className="xs subtle">{summary.data.provider} · {summary.data.model} · from the abstract only, not the full paper</span></div>
          <div><b>TL;DR:</b> {s.tldr}</div>
          {s.key_points?.length > 0 && <ul style={{ margin: 0, paddingLeft: 16 }}>{s.key_points.map((k) => <li key={k}>{k}</li>)}</ul>}
          {s.practical_takeaways?.length > 0 && <div className="small"><b>Possible takeaways:</b> {s.practical_takeaways.join(' · ')}</div>}
          {s.limitations_unknown && <div className="small muted">Unknown from the abstract: {s.limitations_unknown}</div>}
        </div>
      )}
      <div className="row wrap" style={{ gap: 6 }}>
        <a className="btn btn-sm" href={p.abs_url} target="_blank" rel="noopener noreferrer"><ExternalLink size={13} />arXiv page</a>
        {p.pdf_url && <a className="btn btn-sm" href={p.pdf_url} target="_blank" rel="noopener noreferrer"><FileDown size={13} />PDF</a>}
        {p.doi && <a className="btn btn-sm btn-ghost" href={`https://doi.org/${p.doi}`} target="_blank" rel="noopener noreferrer">DOI</a>}
        {aiAvailable && !s && !summary?.loading && <Button size="sm" variant="ghost" icon={Sparkles} onClick={summarize}>Summarise with AI</Button>}
      </div>
    </motion.article>
  );
}

export default function Research() {
  const { system } = useWorkspace();
  const aiAvailable = !!system?.llm?.available;
  const [q, setQ] = useState('');
  const [sort, setSort] = useState('relevance');
  const [explain, setExplain] = useState(false);
  const [context, setContext] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useLocalStorage('mlagent.paperSearches', []);

  const search = async (query = q) => {
    if (query.trim().length < 2) return;
    setQ(query);
    setLoading(true); setError(null);
    try {
      const r = await api.searchPapers({ q: query.trim(), sort, limit: 12, explain: explain && aiAvailable ? 'true' : undefined });
      setResult(r);
      setHistory((h) => [query.trim(), ...h.filter((x) => x !== query.trim())].slice(0, 8));
    } catch (e) { setError(e); } finally { setLoading(false); }
  };
  const notes = Object.fromEntries((result?.ai?.content?.notes || []).map((n) => [n.i, n.relevance]));

  return (
    <div className="page">
      <PageHeader eyebrow="Intelligence" title="Research papers" lede="Search arXiv for methods relevant to your problem. Paper metadata and abstracts are shown verbatim from arXiv; AI notes are optional and labelled." />
      <div className="grid grid-main">
        <div className="stack-lg">
          <Panel>
            <form className="stack" onSubmit={(e) => { e.preventDefault(); search(); }}>
              <div className="row" style={{ gap: 8 }}>
                <div className="input-icon grow"><Search size={14} />
                  <input className="input" style={{ height: 38 }} value={q} onChange={(e) => setQ(e.target.value)} placeholder="e.g. gradient boosting imbalanced tabular data" aria-label="Search arXiv" /></div>
                <Button type="submit" variant="primary" size="lg" loading={loading} disabled={q.trim().length < 2}>Search</Button>
              </div>
              <div className="row wrap" style={{ gap: 16 }}>
                <Segmented ariaLabel="Sort" value={sort} onChange={setSort} options={[{ value: 'relevance', label: 'Relevance' }, { value: 'date', label: 'Newest' }]} />
                {aiAvailable ? <Switch checked={explain} onChange={setExplain} label="Add AI relevance notes" /> : <span className="small subtle">AI notes unavailable (no LLM configured)</span>}
              </div>
              {aiAvailable && (
                <input className="input" value={context} onChange={(e) => setContext(e.target.value)} placeholder="Optional context for AI summaries, e.g. “churn prediction, 20k rows, 8% positive class”" aria-label="Context for AI summaries" />
              )}
            </form>
          </Panel>
          {error && <ErrorState error={error} title="Search failed" onRetry={() => search()} />}
          {result?.ai_error && <Callout tone="warn" title="AI notes unavailable:">{result.ai_error} Results below are unaffected.</Callout>}
          {loading && !result ? <Panel><SkeletonBlock rows={8} /></Panel> : result ? (
            <Panel flush title={`${result.papers.length} papers`} actions={<span className="small subtle">Source: arXiv · {result.query}</span>}>
              {result.papers.length === 0 ? <EmptyState icon={BookOpen} title="No papers found">Try broader terms.</EmptyState> :
                result.papers.map((p, i) => <Paper key={p.arxiv_id} p={p} note={notes[i]} context={context} aiAvailable={aiAvailable} index={i} />)}
            </Panel>
          ) : (
            <Panel><EmptyState icon={BookOpen} title="Search the literature">Results come straight from the arXiv API. Ranking is arXiv's own relevance or date order — the AI never re-orders results.</EmptyState></Panel>
          )}
        </div>
        <div className="stack-lg">
          <Panel title="Recent searches" icon={History}>
            {history.length === 0 ? <span className="small subtle">None yet.</span> : (
              <div className="stack-sm">
                {history.map((h) => <button key={h} className="btn btn-ghost btn-sm" style={{ justifyContent: 'flex-start' }} onClick={() => search(h)}>{h}</button>)}
              </div>
            )}
          </Panel>
          <Panel title="How to read this page">
            <div className="stack-sm small muted">
              <div><Prov kind="calc" label="arXiv" /> Titles, authors, dates and abstracts are the original source text.</div>
              <div><Prov kind="ai" /> Relevance notes and summaries are generated from the abstract only. They may be wrong and are not verified scientific claims.</div>
            </div>
          </Panel>
          <Disclaimer />
        </div>
      </div>
    </div>
  );
}
