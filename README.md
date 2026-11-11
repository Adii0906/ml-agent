# ML-Agent

A workspace for exploring tabular datasets, running leak-free model experiments, comparing them under identical evaluation conditions, and producing reproducible reports.

ML-Agent is built so that every number you see can be traced to a calculation on your data. Language models (Groq or a local Ollama) are optional and are only used to *explain* results. They never pick your target, decide which algorithms are compatible, or produce metrics.

```
Upload ─▶ Explore & clean ─▶ Choose target ─▶ Configure pipeline ─▶ Train (one or many models)
                                                                         │
             Reports ◀─ Compare on the same split ◀─ Evaluate on held-out data
```

---

## Workspaces

| Area | What it does |
|---|---|
| **Overview** | Resume work: running jobs, recent experiments, best held-out score per dataset/target, 14-day activity, recent datasets and reports. New users get an onboarding flow with sample datasets. |
| **Datasets** | Drag-and-drop upload (CSV, TSV, Parquet) with progress and validation, library with rename/delete/download, sample datasets. Every file is pinned by SHA-256. |
| **Data Explorer** | Overview (rows, columns, missing cells, duplicates, memory, inferred types). Spreadsheet-style table (server-side sort, search and pagination, missing-value highlighting). Per-column statistics, histograms, box plots and frequencies. Correlation heatmap, scatter explorer, missing-value pattern. Data-quality findings with severity. Target analysis (problem type, class balance, leakage suspects). |
| **Model Studio** | Step-by-step experiment design: dataset → target → problem type → quality review → preprocessing (with a live pipeline diagram) → algorithms (rule-ranked, with strengths and limits) → hyperparameters (validated) → validation strategy → review and train. Several algorithms can be trained in one batch on the same split. |
| **Experiments** | Full history, including failed and cancelled runs, with filtering, selection for comparison, and deletion (trash icon, confirmation dialog, removes the record and working artifact; saved models are kept). Each experiment page shows a **live training monitor** (real stages and CV folds over WebSocket, cancellation) followed by **evaluation**: test/CV/train/baseline metrics, confusion matrix, per-class metrics, ROC, residual plots, error inspection, permutation importance, CV folds, learning curve, the configuration as executed, and a prediction playground. |
| **Saved Models** | Save a completed experiment's fitted pipeline under a name and optional description. Each save gets a stable id and a version number per name. Browse the list, view metrics and input schema, run a test prediction, verify artifact integrity, and download as `.joblib` or `.pkl`, plus a `manifest.json` of metadata. Deployment is shown only as *Coming soon*. |
| **Model Comparison** | Checks that runs share the same split and target, ranks by any metric, and states trade-offs (differences within CV noise, training cost, overfitting) rather than naming a single winner. |
| **Research Papers** | arXiv search showing original titles, authors, dates, abstracts and links. Optional AI relevance notes and summaries are clearly labelled. |
| **Insights & Reports** | A printable report for every completed experiment, built from recorded results with rule-based observations and limitations. Export as Markdown, text, or print to PDF. An optional AI narrative can be added and is clearly labelled. |
| **Settings** | Light, dark or system theme (persisted, no flash on load), AI provider configuration with a connection test, and server information. |

Each number in the UI carries a provenance tag: `CALCULATED`, `TEST` (held-out), `CV` (training split), `TRAIN`, `RULE` or `AI`.

---

## ML methodology (enforced in code)

- **No preprocessing leakage.** Imputation, scaling, IQR clipping, one-hot categories and feature selection all live inside one scikit-learn `Pipeline`/`ColumnTransformer` that is fitted on the training split only. Tests verify this by checking the fitted statistics against the training rows.
- **Duplicates removed before splitting** (on by default). Rows are compared on the columns the model actually uses, so rows that differ only in an excluded ID column still count as duplicates and cannot sit in both train and test.
- **Stratified hold-out test set**, reproducible from a seed. A `split_id` fingerprints the exact test rows, so the comparison view can confirm that models were scored on identical data.
- **Cross-validation runs on the training split only.** The test split is used exactly once, by the final model.
- **Train, CV, test and baseline metrics are reported separately**, and every result is compared with a dummy baseline (majority class, or the mean for regression).
- **Columns excluded from training by default:** identifier-like, constant, empty, free-text and date columns. You can override this.
- **Strict validation.** The engine rejects:
  - unknown algorithms (no silent fallback to Random Forest);
  - a task that doesn't match the algorithm;
  - invalid hyperparameters or split settings;
  - continuous targets used for classification;
  - text targets used for regression;
  - single-class targets;
  - singleton classes (unless you opt in to dropping them);
  - datasets with fewer than 10 usable rows;
  - impossible stratified splits.
- **Failed and cancelled runs are kept**, with their error and event log.
- **Reproducibility** is recorded per experiment: dataset SHA-256, seed, split id, and Python, scikit-learn, pandas and numpy versions.

### Supported algorithms

| Classification | Regression |
|---|---|
| Logistic Regression | Linear Regression |
| Decision Tree | Ridge, Lasso |
| Random Forest | Decision Tree Regressor |
| Gradient Boosting | Random Forest Regressor |
| Histogram Gradient Boosting | Gradient Boosting Regressor |
| Support Vector Machine (SVC) | Histogram Gradient Boosting Regressor |
| K-Nearest Neighbors | Support Vector Regression |
| Gaussian Naive Bayes | K-Nearest Neighbors Regressor |
| XGBoost (if installed) | XGBoost Regressor (if installed) |

Every algorithm listed is trained and evaluated in the automated tests, for both binary and multiclass classification.

---

## Saved models

Saving copies the experiment's fitted pipeline into a separate, versioned model package:

```
<MLAGENT_DATA_DIR>/saved_models/<model_id>/
  model.joblib    fitted Pipeline (preprocessing + estimator), label classes, embedded manifest
  model.pkl       the same object, pickle protocol 5
  manifest.json   metadata, also stored in SQLite (table saved_models)
```

The manifest records:
- algorithm, estimator class and hyperparameters;
- task and target;
- an input schema (feature names, roles, training dtypes and column order) and an output schema;
- the preprocessing configuration and pipeline stages;
- metrics (test, CV, train, baseline);
- dataset id and SHA-256;
- split, random seed and CV settings;
- the source experiment;
- Python, scikit-learn, pandas, numpy and joblib versions;
- each artifact's SHA-256 and size.

**Loading outside ML-Agent.** An artifact needs only scikit-learn, pandas, numpy and the small `backend/mlagent_runtime` module, which holds the two custom pipeline components and the load and predict helpers:

```python
from mlagent_runtime import load_artifact, predict
bundle = load_artifact("model.joblib", expected_sha256="<sha256 from the manifest>")
predict(bundle, [{"age": 41, "plan": "pro"}])
```

**Designed for deployment later.** Deployment is not implemented yet. The storage is built so it won't need replacing when it is:
- **Stable ids:** `mdl_…` ids are never reused, and each (owner, name) pair has its own version history.
- **Separate storage:** artifacts live on disk; metadata lives in SQLite and `manifest.json`.
- **Prediction already exists:** `POST /api/models/{id}/predict` loads the hash-verified artifact through the same `mlagent_runtime` code a serving container would use.

**Integrity and safety:**
- Every artifact is SHA-256-verified before it's downloaded or deserialised.
- A missing or tampered artifact is refused (HTTP 410 or 409).
- Only artifacts this server wrote are ever loaded. There is deliberately no endpoint for uploading pickles.
- Saves and deletes are atomic:
  - files are written to a temporary directory and moved into place;
  - on delete, files are moved aside and restored if the database step fails;
  - leftovers are swept on startup.

**Deleting experiments.** Deleting an experiment removes its record and working artifact. Saved models are independent copies and are kept; they show their source as *(deleted)*. If an experiment is deleted while it's training, the run is cancelled, and any artifact it finishes writing afterwards is discarded.

**Access control.** ML-Agent has no user accounts. For shared deployments, set `MLAGENT_ACCESS_TOKENS="alice:<token>,bob:<token>"`; tokens must be at least 16 characters. Every saved-model endpoint, plus experiment and dataset deletion and the experiment model export, then requires `Authorization: Bearer <token>`. Saved models belong to the token's owner, and other owners get 404. Users enter their token under **Settings → Access**; it's kept in the browser and sent as a header, never in URLs. Without tokens the server runs in single-user local mode, and `/api/system` reports which mode is active.

---

## Run it

### Local development

Requirements: Python 3.10+ and Node 18+.

```bash
./setup.sh
```

Then, in two terminals:

```bash
cd backend && ../.venv/bin/python main.py
```

```bash
cd frontend && npm run dev
```

Open <http://localhost:5173>. The Vite dev server proxies `/api` (HTTP and WebSocket) to the backend on port 8000. Load `iris.csv` from **Datasets → Sample datasets** to try the full workflow.

### Docker

```bash
docker compose up --build
```

The UI is served at <http://localhost:8080> (nginx proxies `/api` to the backend). Data is kept in the `mlagent-data` volume.

For a single container (for example Hugging Face Spaces), use the root `Dockerfile`. FastAPI serves the built frontend and the API on port 7860.

### Configuration

Copy `.env.example` to `.env`. Every setting is optional:

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY`, `GROQ_MODEL` | Enable AI explanations through Groq. |
| `OLLAMA_URL`, `OLLAMA_MODEL`, `USE_OLLAMA` | Use a local Ollama model instead (also selectable in Settings). |
| `MLAGENT_DATA_DIR` | Where datasets, models and the SQLite database are stored (default `backend/var`). |
| `MLAGENT_MAX_UPLOAD_MB`, `MLAGENT_MAX_JOBS` | Upload limit and number of concurrent training jobs. |
| `MLAGENT_MAX_MODEL_MB` | Largest model artifact that can be saved (default 1024). |
| `MLAGENT_ACCESS_TOKENS` | `owner:token` pairs that turn on access control (see *Saved models*). |
| `MLAGENT_CORS_ORIGINS`, `VITE_API_URL` | For deployments where the UI and API are on different origins. |

Without an LLM, every core workflow still works: upload, explore, clean, train, evaluate, compare and report.

---

## Architecture

```
frontend/  React 18 + Vite · react-router · framer-motion · recharts · lucide
  src/styles/tokens.css   design tokens (light + dark, validated chart palette)
  src/lib/                api client, theme, toasts, workspace context, formatting
  src/components/         shell (sidebar, header, ⌘K palette), UI kit, charts, dropzone
  src/pages/              overview, datasets, explorer/*, studio/*, experiment/*, compare, research, reports, settings

backend/   FastAPI · scikit-learn · pandas · SQLite
  main.py                 routes (served at / and /api), WebSocket stream, SPA serving
  core/datasets.py        storage under generated ids, parsing, type inference, consent-based cleaning
  core/profiling.py       statistics, distributions, correlations, quality findings, target analysis
  core/algorithms.py      registry of implemented models + hyperparameter schemas
  core/recommend.py       deterministic suitability rules
  core/pipeline.py        leak-free preprocessing pipeline builder
  core/training.py        validation, split, CV, fit, evaluation, importance, reproducibility
  core/jobs.py            thread-pool job manager, replayable events, cancellation, dedupe
  core/reports.py         rule-based observations, limitations, Markdown/text export
  core/model_registry.py  saved models: versioning, integrity, atomic save/delete
  core/auth.py            optional owner tokens
  mlagent_runtime/        standalone loader/predictor + custom transformers used by artifacts
  core/llm.py             optional Groq/Ollama layer (explanations only, always labelled)
  core/papers.py          arXiv search
  tests/                  pytest suite
```

Training runs in a worker thread pool, so the API stays responsive. Each job keeps an ordered event log; WebSocket clients get a replay on connect and then live events, so a page opened late or reconnected never misses the outcome. Identical in-flight submissions are rejected with HTTP 409. Experiments left running by a server restart are marked as interrupted.

### API

The new endpoints live under `/api` (and remain available at `/`):

```
GET  /api/system                         versions, LLM status, limits
POST /api/datasets                       upload (multipart)
GET  /api/datasets[/{id}]                list / metadata
GET  /api/datasets/{id}/profile|rows|columns/{c}|correlations|missing|scatter|quality|target
POST /api/datasets/{id}/clean            apply reviewed actions → new dataset version
GET  /api/algorithms                     registry with hyperparameter schemas
POST /api/experiments                    train one or more algorithms on one split
GET  /api/experiments[/{id}]             history / full results
POST /api/experiments/{id}/cancel|predict
GET  /api/experiments/{id}/report|export-report?format=md|txt|export-model
DELETE /api/experiments/{id}             record + working artifact (saved models kept)
POST /api/experiments/{id}/save-model    {name, description} → versioned saved model
GET  /api/models[/{id}]                  saved models (owner-scoped), detail incl. manifest + versions
PATCH|DELETE /api/models/{id}            rename / describe, delete record + files
GET  /api/models/{id}/download?format=joblib|pkl   hash-verified artifact
GET  /api/models/{id}/manifest|verify    metadata JSON / full integrity check
POST /api/models/{id}/predict            predictions from the stored artifact
WS   /api/ws/experiments/{id}            replayable training events
GET  /api/papers/search                  arXiv (+ optional AI relevance notes)
```

The original endpoints still work: `/upload-dataset`, `/get-dataset-preview`, `/analyze-dataset`, `/recommend-algorithm`, `/train-model`, `/experiments…`, `/search-papers` and `/ws/train/{id}`. They now validate their input:
- file paths are restricted to managed upload directories;
- algorithm names must match a registered model;
- `recommend-algorithm` reports `confidence: null` instead of an invented number.

---

## Testing

```bash
.venv/bin/python -m pytest backend/tests -q        # 131 tests
cd frontend && npm run lint && npm run build
```

The backend suite covers:
- **Every algorithm:** training on synthetic data with known structure (binary and multiclass, plus regression), beating the baseline, prediction shapes and label space, reproducibility, identical splits across algorithms, hyperparameters actually being applied, and rejection of invalid input.
- **Leakage:** imputer and scaler statistics equal the training-split statistics; feature selection on pure noise stays near chance; IQR fences come from training rows; duplicates hidden behind an ID never cross the split; the target and IDs are never used as features; categories unseen in training don't crash prediction; CV folds use training rows only.
- **Edge cases:**
  - empty, header-only, unsupported, semicolon-delimited and Latin-1 files;
  - single-column data;
  - missing target or feature values; constant and empty columns;
  - single-class targets, continuous targets used for classification, text targets used for regression;
  - tiny datasets and singleton classes;
  - imbalanced data, impossible stratification, high cardinality;
  - boolean and mixed-type columns;
  - invalid split or preprocessing options;
  - cancellation.
- **Metrics:** checked against hand-computed values. Train and test scores are reported separately, and the baseline equals the majority share.
- **Saved models:**
  - the full save → restart → list → download (`.joblib` and `.pkl`) → load-and-predict cycle, run in an isolated interpreter that can import only `mlagent_runtime`, with predictions identical to the server's;
  - manifest contents and versioning;
  - deleting an experiment keeps its saved models; deleting a saved model removes its record and files;
  - tampered and missing artifacts are refused, including a tampered experiment artifact at save time;
  - simulated disk and database failures leave nothing behind or are rolled back;
  - deleting an experiment mid-training leaves no orphaned artifact;
  - path-traversal ids are rejected, and pickle uploads are refused;
  - owner isolation and 401/404 behaviour, and weak tokens are rejected at startup.
- **API and WebSocket:** upload validation and path traversal, the size limit, pagination, sorting and search, cleaning versions, the full experiment lifecycle and exports, failed-run visibility, request validation, duplicate rejection and cancellation, WebSocket replay for late subscribers, persistence across restarts and recovery of interrupted runs, settings validation, LLM-unavailable behaviour, and the legacy endpoints.

---

## Known limitations

- Supports tabular data only (CSV, TSV, Parquet). Date and free-text columns are excluded rather than featurised, and there is no time-series or image support.
- There is one hold-out split per experiment and no automated hyperparameter search. Nested CV is not implemented.
- Cancellation is cooperative: it takes effect between pipeline stages and CV folds, not in the middle of a single `fit()`.
- Training progress is reported per stage and per fold. A single model fit has no percentage, so none is shown.
- SVR does not rescale the target, so its error metrics are sensitive to the target's scale. This is noted in its reports.
- The job queue lives in process memory. Restarting the server interrupts running jobs, which are then marked as failed.
- There are no user accounts. Access control is an optional set of owner tokens configured on the server. Only **saved models** are owner-scoped; datasets and experiments are shared by everyone who can reach the server, and only their deletion and model export require a token.
- Saved models can't be uploaded or imported: loading a third-party pickle would let it run code. Deployment and hosted endpoints are not implemented yet.
- Downloaded models need the same scikit-learn version recorded in the manifest, plus the `mlagent_runtime` module. Pickles are not portable across scikit-learn versions.
- Browser downloads are buffered in memory before saving. For very large artifacts, use the API directly (for example `curl -H "Authorization: Bearer …" -OJ …/download`).
- Infinite values are treated as missing and imputed. Numbers stored with currency or percent symbols (`$1,234`, `45%`) are still treated as categories.
- Training time grows quickly for kernel and distance models, and with permutation importance on wide data. In testing, 200,000 rows took about 11 minutes across 7 algorithms, and 400 columns about 5 minutes.
- arXiv search needs outbound network access to `export.arxiv.org`.

AI-generated recommendations and explanations may contain errors. Verify your dataset, preprocessing decisions and evaluation methodology independently. Model performance depends on data quality, validation strategy and real-world conditions.
