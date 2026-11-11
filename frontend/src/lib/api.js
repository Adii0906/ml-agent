// Thin fetch wrapper. All calls go through /api so the production build can serve
// the SPA from "/" without clashing with API routes.
const BASE = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '') + '/api';

const TOKEN_KEY = 'mlagent.accessToken';

/** Access token for servers started with MLAGENT_ACCESS_TOKENS (kept only in this browser). */
export function getAccessToken() {
  try { return localStorage.getItem(TOKEN_KEY) || ''; } catch { return ''; }
}
export function setAccessToken(token) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch { /* storage unavailable */ }
}
function authHeaders() {
  const t = getAccessToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

export class ApiError extends Error {
  constructor(message, status, body) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

async function request(path, { method = 'GET', body, query, signal, raw = false } = {}) {
  const url = new URL(BASE + path, window.location.origin);
  if (query) {
    Object.entries(query).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, v);
    });
  }
  const init = { method, signal, headers: { ...authHeaders() } };
  if (body instanceof FormData) init.body = body;
  else if (body !== undefined) {
    init.body = JSON.stringify(body);
    init.headers['Content-Type'] = 'application/json';
  }
  let res;
  try {
    res = await fetch(url, init);
  } catch (err) {
    if (err.name === 'AbortError') throw err;
    throw new ApiError('Cannot reach the ML-Agent server. Is the backend running on port 8000?', 0);
  }
  if (res.status === 401) {
    throw new ApiError('This server requires an access token. Add yours in Settings → Access.', 401);
  }
  if (raw) {
    if (!res.ok) {
      let detail = `Request failed (${res.status})`;
      try { detail = (await res.json()).detail || detail; } catch { /* not json */ }
      throw new ApiError(detail, res.status);
    }
    return res;
  }
  let data = null;
  const text = await res.text();
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!res.ok) {
    let msg = data?.detail ?? data?.message ?? `Request failed (${res.status})`;
    if (Array.isArray(msg)) msg = msg.map((d) => `${(d.loc || []).slice(1).join('.')}: ${d.msg}`).join('; ');
    throw new ApiError(String(msg), res.status, data);
  }
  return data;
}

/** Upload with progress (fetch has no upload progress events). */
function upload(path, file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', BASE + path);
    Object.entries(authHeaders()).forEach(([k, v]) => xhr.setRequestHeader(k, v));
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
    xhr.onload = () => {
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch { /* not json */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else reject(new ApiError(data?.detail || `Upload failed (${xhr.status})`, xhr.status, data));
    };
    xhr.onerror = () => reject(new ApiError('Upload failed: the server could not be reached.', 0));
    const fd = new FormData();
    fd.append('file', file);
    xhr.send(fd);
  });
}

export function wsUrl(path) {
  const origin = import.meta.env.VITE_API_URL || window.location.origin;
  return origin.replace(/^http/, 'ws').replace(/\/$/, '') + '/api' + path;
}

/** Authenticated file download: fetch → blob → save. Reports failures instead of
 *  silently navigating to an error page. Returns {filename, bytes, sha256Header}. */
export async function download(path, query, fallbackName = 'download') {
  const res = await request(path, { query, raw: true });
  const cd = res.headers.get('content-disposition') || '';
  const match = cd.match(/filename\*?=(?:UTF-8'')?"?([^";]+)"?/i);
  const filename = match ? decodeURIComponent(match[1]) : fallbackName;
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
  return { filename, bytes: blob.size, sha256: res.headers.get('x-content-sha256') };
}

export const fileUrl = (path, query) => {
  const url = new URL(BASE + path, window.location.origin);
  Object.entries(query || {}).forEach(([k, v]) => url.searchParams.set(k, v));
  return url.toString();
};

export const api = {
  system: () => request('/system'),
  settings: () => request('/settings'),
  saveSettings: (values) => request('/settings', { method: 'PUT', body: values }),
  testLlm: () => request('/settings/test-llm', { method: 'POST' }),

  datasets: () => request('/datasets'),
  dataset: (id) => request(`/datasets/${id}`),
  uploadDataset: (file, onProgress) => upload('/datasets', file, onProgress),
  renameDataset: (id, name) => request(`/datasets/${id}`, { method: 'PATCH', body: { name } }),
  deleteDataset: (id) => request(`/datasets/${id}`, { method: 'DELETE' }),
  profile: (id) => request(`/datasets/${id}/profile`),
  rows: (id, query, signal) => request(`/datasets/${id}/rows`, { query, signal }),
  column: (id, col) => request(`/datasets/${id}/columns/${encodeURIComponent(col)}`),
  correlations: (id) => request(`/datasets/${id}/correlations`),
  missing: (id) => request(`/datasets/${id}/missing`),
  scatter: (id, query) => request(`/datasets/${id}/scatter`, { query }),
  quality: (id, target) => request(`/datasets/${id}/quality`, { query: { target } }),
  target: (id, target, task_type) => request(`/datasets/${id}/target`, { query: { target, task_type } }),
  clean: (id, actions) => request(`/datasets/${id}/clean`, { method: 'POST', body: { actions } }),
  samples: () => request('/samples'),
  importSample: (name) => request(`/samples/${name}/import`, { method: 'POST' }),

  algorithms: (task) => request('/algorithms', { query: { task } }),
  explainRecommendations: (body) => request('/recommendations/explain', { method: 'POST', body }),

  experiments: (query) => request('/experiments', { query }),
  experiment: (id) => request(`/experiments/${id}`),
  createExperiments: (body) => request('/experiments', { method: 'POST', body }),
  renameExperiment: (id, name) => request(`/experiments/${id}`, { method: 'PATCH', body: { name } }),
  cancelExperiment: (id) => request(`/experiments/${id}/cancel`, { method: 'POST' }),
  deleteExperiment: (id) => request(`/experiments/${id}`, { method: 'DELETE' }),
  report: (id) => request(`/experiments/${id}/report`),
  narrative: (id) => request(`/experiments/${id}/report/narrative`, { method: 'POST' }),
  predict: (id, rows) => request(`/experiments/${id}/predict`, { method: 'POST', body: { rows } }),

  whoami: () => request('/auth/whoami'),
  saveModel: (expId, body) => request(`/experiments/${expId}/save-model`, { method: 'POST', body }),
  experimentModels: (expId) => request(`/experiments/${expId}/saved-models`),
  models: () => request('/models'),
  model: (id) => request(`/models/${id}`),
  updateModel: (id, body) => request(`/models/${id}`, { method: 'PATCH', body }),
  deleteModel: (id) => request(`/models/${id}`, { method: 'DELETE' }),
  verifyModel: (id) => request(`/models/${id}/verify`),
  modelPredict: (id, rows) => request(`/models/${id}/predict`, { method: 'POST', body: { rows } }),

  searchPapers: (query) => request('/papers/search', { query }),
  summarizePaper: (body) => request('/papers/summarize', { method: 'POST', body }),
};
