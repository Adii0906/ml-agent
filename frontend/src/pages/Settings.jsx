import { Cpu, KeyRound, Monitor, Moon, Palette, Plug, Server, Sun } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Badge, Button, Callout, ErrorState, Field, PageHeader, Panel, Segmented, SkeletonBlock } from '../components/ui';
import { api, getAccessToken, setAccessToken } from '../lib/api';
import { useAsync } from '../lib/hooks';
import { useTheme } from '../lib/theme';
import { useToast } from '../lib/toast';
import { useWorkspace } from '../lib/workspace';

export default function Settings() {
  const { preference, setPreference } = useTheme();
  const { system, refreshSystem } = useWorkspace();
  const settings = useAsync(() => api.settings(), []);
  const [form, setForm] = useState(null);
  const [saving, setSaving] = useState(false);
  const [probe, setProbe] = useState(null);
  const [token, setToken] = useState(getAccessToken);
  const [whoami, setWhoami] = useState(null);
  const saveToken = async () => {
    setAccessToken(token.trim());
    setWhoami({ loading: true });
    try { setWhoami({ data: await api.whoami() }); } catch (e) { setWhoami({ error: e.message }); }
  };
  const toast = useToast();
  useEffect(() => { if (settings.data && !form) setForm(settings.data.llm); }, [settings.data, form]);

  const save = async () => {
    setSaving(true);
    try {
      const res = await api.saveSettings(form);
      settings.setData(res);
      setForm(res.llm);
      refreshSystem();
      toast.success('Settings saved', res.status.available ? `AI provider: ${res.status.active_provider}` : 'AI features are disabled.');
    } catch (e) { toast.error('Could not save', e.message); } finally { setSaving(false); }
  };
  const test = async () => {
    setProbe({ loading: true });
    try { setProbe({ data: await api.testLlm() }); } catch (e) { setProbe({ error: e.message }); }
  };
  const status = settings.data?.status;
  const dirty = form && settings.data && JSON.stringify(form) !== JSON.stringify(settings.data.llm);

  return (
    <div className="page page-narrow">
      <PageHeader eyebrow="System" title="Settings" lede="Appearance, AI provider and server information. Model training never depends on an AI provider." />
      <div className="stack-lg">
        <Panel title="Appearance" icon={Palette}>
          <Field label="Theme" hint="Saved in this browser. “System” follows your operating system.">
            <Segmented ariaLabel="Theme" value={preference} onChange={setPreference} options={[
              { value: 'light', label: 'Light', icon: Sun }, { value: 'dark', label: 'Dark', icon: Moon }, { value: 'system', label: 'System', icon: Monitor }]} />
          </Field>
        </Panel>

        <Panel title="Access" icon={KeyRound} actions={system?.auth && <Badge tone={system.auth.mode === 'tokens' ? 'info' : undefined}>{system.auth.mode === 'tokens' ? 'Tokens required' : 'Local mode'}</Badge>}
          footer="The token is kept only in this browser and sent as an Authorization header to protected endpoints (saved models, downloads, deletions).">
          <div className="stack">
            <p className="small muted">{system?.auth?.detail || 'Checking server…'}</p>
            <div className="row wrap" style={{ gap: 8, alignItems: 'flex-end' }}>
              <div style={{ width: 'min(360px, 100%)' }}>
                <Field label="Access token" htmlFor="tok"><input id="tok" className="input mono" type="password" autoComplete="off" value={token} onChange={(e) => setToken(e.target.value)} placeholder={system?.auth?.mode === 'tokens' ? 'Paste your token' : 'Not required in local mode'} /></Field>
              </div>
              <Button onClick={saveToken} loading={whoami?.loading}>Save & check</Button>
              {token && <Button variant="ghost" onClick={() => { setToken(''); setAccessToken(''); setWhoami(null); }}>Clear</Button>}
            </div>
            {whoami?.data && <Callout tone="good">Signed in as <b>{whoami.data.owner}</b> ({whoami.data.mode} mode).</Callout>}
            {whoami?.error && <Callout tone="bad">{whoami.error}</Callout>}
          </div>
        </Panel>

        <Panel title="AI provider" icon={Plug} actions={status && <Badge tone={status.available ? 'good' : undefined}>{status.available ? `Active: ${status.active_provider}` : 'Disabled'}</Badge>}
          footer="API keys are read from the server's environment (.env) and are never sent to or stored by the browser.">
          {settings.error ? <ErrorState error={settings.error} onRetry={settings.reload} /> : !form ? <SkeletonBlock /> : (
            <div className="stack">
              <Field label="Provider">
                <Segmented ariaLabel="Provider" value={form.llm_provider} onChange={(v) => setForm({ ...form, llm_provider: v })} options={[
                  { value: 'auto', label: 'Auto' }, { value: 'groq', label: 'Groq' }, { value: 'ollama', label: 'Ollama (local)' }, { value: 'none', label: 'Off' }]} />
              </Field>
              <div className="grid grid-2" style={{ gap: 14 }}>
                <Field label="Groq model" htmlFor="gm" hint={status?.groq_key_present ? 'GROQ_API_KEY detected on the server' : 'GROQ_API_KEY is not set on the server'}>
                  <input id="gm" className="input mono" value={form.groq_model} onChange={(e) => setForm({ ...form, groq_model: e.target.value })} />
                </Field>
                <div />
                <Field label="Ollama URL" htmlFor="ou"><input id="ou" className="input mono" value={form.ollama_url} onChange={(e) => setForm({ ...form, ollama_url: e.target.value })} /></Field>
                <Field label="Ollama model" htmlFor="om"><input id="om" className="input mono" value={form.ollama_model} onChange={(e) => setForm({ ...form, ollama_model: e.target.value })} /></Field>
              </div>
              <div className="row wrap">
                <Button variant="primary" onClick={save} loading={saving} disabled={!dirty}>Save</Button>
                <Button onClick={test} loading={probe?.loading} disabled={dirty}>Test connection</Button>
                {dirty && <span className="small subtle">Save before testing.</span>}
              </div>
              {probe?.data && (probe.data.reachable
                ? <Callout tone="good">Provider responded ({probe.data.active_provider}).</Callout>
                : <Callout tone="warn">{probe.data.reason || 'Provider did not respond.'}</Callout>)}
              {probe?.error && <Callout tone="bad">{probe.error}</Callout>}
              <p className="small muted">“Auto” uses Groq when a key is present and otherwise turns AI features off. AI is only used to explain recommendations, summarise papers and draft optional report narratives — always labelled.</p>
            </div>
          )}
        </Panel>

        <Panel title="Server" icon={Server}>
          {!system ? <SkeletonBlock /> : (
            <dl className="kv">
              <dt>API version</dt><dd className="num">{system.version}</dd>
              <dt>Python</dt><dd className="num">{system.versions.python}</dd>
              <dt>scikit-learn</dt><dd className="num">{system.versions.scikit_learn}</dd>
              <dt>pandas · numpy</dt><dd className="num">{system.versions.pandas} · {system.versions.numpy}</dd>
              <dt>XGBoost</dt><dd>{system.xgboost_available ? 'installed' : 'not installed (XGBoost models hidden)'}</dd>
              <dt>Upload limit</dt><dd className="num">{system.max_upload_mb} MB</dd>
              <dt>Concurrent training jobs</dt><dd className="num">{system.max_concurrent_jobs}</dd>
            </dl>
          )}
        </Panel>
        <Panel title="Keyboard" icon={Cpu}>
          <dl className="kv"><dt><kbd>⌘</kbd> / <kbd>Ctrl</kbd> + <kbd>K</kbd></dt><dd>Jump to any page, dataset or experiment</dd><dt><kbd>Esc</kbd></dt><dd>Close dialogs</dd></dl>
        </Panel>
      </div>
    </div>
  );
}
