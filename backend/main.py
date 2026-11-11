"""ML-Agent API server.

All routes are served at both ``/`` (backwards compatible with the original
API) and ``/api`` (used by the new frontend, which lets the production build
serve the single-page app from ``/`` without path clashes).
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import joblib
import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from core import algorithms as algs
from core import papers as paper_search
from core import profiling, recommend, reports
from core.auth import Authenticator, Principal
from core.config import BACKEND_DIR, ROOT_DIR, Settings
from core.datasets import DatasetError, DatasetRegistry
from core.jobs import ACTIVE, TERMINAL, DuplicateJob, JobManager, make_work
from core.llm import DISCLAIMER, LLM, LLMUnavailable
from core.model_registry import FORMATS, ModelRegistry, RegistryError
from core.store import Store
from core.training import TrainingError, fingerprint, validate_config
from core.util import clean_json, safe_filename, utcnow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ml-agent")

SAMPLES_DIR = ROOT_DIR / "data"


# ------------------------------------------------------------------ schemas
class AlgorithmSpec(BaseModel):
    id: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


class ExperimentCreate(BaseModel):
    dataset_id: str
    target: str
    name: Optional[str] = None
    task_type: Literal["auto", "classification", "regression"] = "auto"
    algorithms: List[AlgorithmSpec] = Field(min_length=1, max_length=12)
    preprocessing: Dict[str, Any] = Field(default_factory=dict)
    split: Dict[str, Any] = Field(default_factory=dict)
    options: Dict[str, Any] = Field(default_factory=dict)


class CleanRequest(BaseModel):
    actions: List[Dict[str, Any]] = Field(min_length=1)


class RenameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class SaveModelRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: Optional[str] = Field(default=None, max_length=2000)


class ModelUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    description: Optional[str] = Field(default=None, max_length=2000)


class PredictRequest(BaseModel):
    rows: List[Dict[str, Any]] = Field(min_length=1, max_length=1000)


class SummarizeRequest(BaseModel):
    title: str
    abstract: str
    context: Optional[str] = None


class RecommendExplainRequest(BaseModel):
    dataset_id: str
    target: str
    task_type: Optional[Literal["classification", "regression"]] = None


# Legacy request models (original API)
class DatasetRequest(BaseModel):
    file_path: str
    dataset_type: str = "tabular"
    task_type: Optional[str] = None
    target_column: Optional[str] = None


class TrainingRequest(BaseModel):
    experiment_name: str
    algorithm: str
    dataset_path: str
    config: Dict[str, Any] = Field(default_factory=dict)
    hyperparameters: Optional[Dict[str, Any]] = None
    validation_split: float = 0.2
    random_state: int = 42


# ------------------------------------------------------------------ app
def create_app(settings: Optional[Settings] = None, static_dir: Optional[Path] = None) -> FastAPI:
    settings = settings or Settings()
    store = Store(settings.db_path)
    registry = DatasetRegistry(settings, store)
    manager = JobManager(store, settings.models_dir, settings.max_concurrent_jobs)
    llm = LLM(store)
    models = ModelRegistry(settings, store)
    auth = Authenticator(settings.access_tokens)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager.bind_loop(asyncio.get_running_loop())
        n = manager.recover_interrupted()
        swept = models.sweep()
        if swept:
            log.info("Removed %d leftover temporary/trash artifacts", swept)
        if n:
            log.warning("Marked %d interrupted experiments as failed", n)
        yield
        manager.shutdown()

    app = FastAPI(title="ML-Agent", version="2.0.0", lifespan=lifespan)
    app.state.settings, app.state.store, app.state.registry = settings, store, registry
    app.state.manager, app.state.llm, app.state.models = manager, llm, models

    def principal(request: Request) -> Principal:
        """Resolve the caller. Protected endpoints depend on this; it raises 401 when tokens
        are configured and the request does not carry a valid one."""
        who = auth.resolve(request.headers.get("authorization"))
        if who is None:
            raise HTTPException(401, "Authentication required: send 'Authorization: Bearer <access token>'.",
                                headers={"WWW-Authenticate": "Bearer"})
        return who

    origins = [o.strip() for o in os.getenv("MLAGENT_CORS_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=origins != ["*"],
                       allow_methods=["*"], allow_headers=["*"])

    api = APIRouter()

    # -------------------------------------------------------------- helpers
    def dataset_or_404(dataset_id: str) -> Dict[str, Any]:
        meta = store.get_dataset(dataset_id)
        if not meta:
            raise HTTPException(404, "Dataset not found.")
        return meta

    async def load_df(dataset_id: str) -> pd.DataFrame:
        dataset_or_404(dataset_id)
        try:
            return await run_in_threadpool(registry.load, dataset_id)
        except DatasetError as exc:
            raise HTTPException(422, str(exc)) from None

    def experiment_or_404(exp_id: str) -> Dict[str, Any]:
        exp = store.get_experiment(exp_id)
        if not exp:
            raise HTTPException(404, "Experiment not found.")
        return exp

    def resolve_legacy_path(file_path: str) -> Path:
        """Old endpoints accepted arbitrary server paths. Only allow files inside
        the managed dataset directory or the legacy uploads directory."""
        p = Path(file_path).resolve()
        allowed = [settings.datasets_dir.resolve(), settings.legacy_upload_dir.resolve(), SAMPLES_DIR.resolve()]
        if not any(p.is_relative_to(a) for a in allowed):
            raise HTTPException(403, "Only uploaded datasets can be accessed.")
        if not p.is_file():
            raise HTTPException(404, "File not found.")
        return p

    def legacy_dataset_id(file_path: str) -> str:
        p = resolve_legacy_path(file_path)
        meta = store.find_dataset_by_path(str(p))
        if meta:
            return meta["id"]
        try:
            return registry.register_existing_file(p)["id"]
        except DatasetError as exc:
            raise HTTPException(422, str(exc)) from None

    # -------------------------------------------------------------- system
    @api.get("/health")
    async def health():
        return {"status": "healthy", "timestamp": utcnow(), "version": app.version,
                "active_jobs": manager.active_count()}

    @api.get("/system")
    async def system():
        import platform

        import numpy
        import sklearn
        return {
            "version": app.version,
            "llm": llm.status(),
            "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                         "pandas": pd.__version__, "numpy": numpy.__version__},
            "xgboost_available": algs.XGBOOST_AVAILABLE,
            "max_upload_mb": settings.max_upload_mb,
            "max_concurrent_jobs": settings.max_concurrent_jobs,
            "disclaimer": DISCLAIMER,
            "auth": {"mode": "tokens" if auth.enabled else "local",
                     "detail": "Access tokens required for model, download and delete endpoints." if auth.enabled
                     else "Single-user local mode: no access tokens configured (MLAGENT_ACCESS_TOKENS)."},
            "max_model_mb": settings.max_model_mb,
        }

    @api.get("/settings")
    async def get_settings():
        return {"llm": llm.config(), "status": llm.status()}

    @api.put("/settings")
    async def put_settings(values: Dict[str, Any]):
        try:
            cfg = llm.update_config(values)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return {"llm": cfg, "status": llm.status()}

    @api.post("/settings/test-llm")
    async def test_llm():
        return await run_in_threadpool(llm.status, True)

    # -------------------------------------------------------------- datasets
    @api.get("/datasets")
    async def list_datasets():
        datasets = registry.list()
        counts: Dict[str, int] = {}
        for e in store.list_experiments():
            if e.get("dataset_id"):
                counts[e["dataset_id"]] = counts.get(e["dataset_id"], 0) + 1
        for d in datasets:
            d["experiment_count"] = counts.get(d["id"], 0)
            d.pop("path", None)
        return {"datasets": datasets}

    @api.post("/datasets")
    async def upload_dataset(file: UploadFile = File(...)):
        try:
            meta = await run_in_threadpool(registry.register_upload, file.filename or "", file.file)
        except DatasetError as exc:
            raise HTTPException(422, str(exc)) from None
        meta = dict(meta)
        meta.pop("path", None)
        return meta

    @api.get("/datasets/{dataset_id}")
    async def get_dataset(dataset_id: str):
        meta = dict(dataset_or_404(dataset_id))
        store.touch_dataset(dataset_id)
        meta.pop("path", None)
        if meta.get("parent_id"):
            parent = store.get_dataset(meta["parent_id"])
            meta["parent_name"] = parent["name"] if parent else None
        return meta

    @api.patch("/datasets/{dataset_id}")
    async def rename_dataset(dataset_id: str, body: RenameRequest):
        dataset_or_404(dataset_id)
        store.rename_dataset(dataset_id, body.name.strip())
        return await get_dataset(dataset_id)

    @api.delete("/datasets/{dataset_id}")
    async def delete_dataset(dataset_id: str, who: Principal = Depends(principal)):
        dataset_or_404(dataset_id)
        running = [e for e in store.list_experiments() if e.get("dataset_id") == dataset_id and e["status"] in ACTIVE]
        if running:
            raise HTTPException(409, "This dataset has experiments in progress.")
        registry.delete(dataset_id)
        return {"status": "deleted"}

    @api.get("/datasets/{dataset_id}/profile")
    async def dataset_profile(dataset_id: str):
        df = await load_df(dataset_id)
        return await run_in_threadpool(profiling.full_profile, df)

    @api.get("/datasets/{dataset_id}/rows")
    async def dataset_rows(dataset_id: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=500),
                           sort: Optional[str] = None, order: Literal["asc", "desc"] = "asc",
                           search: Optional[str] = None, column: Optional[str] = None,
                           missing_only: bool = False):
        df = await load_df(dataset_id)

        def work():
            view = df
            if search:
                cols = [column] if column and column in df.columns else list(df.columns)
                needle = search.lower()
                mask = pd.Series(False, index=df.index)
                for c in cols:
                    mask |= df[c].astype(str).str.lower().str.contains(needle, regex=False, na=False)
                view = view[mask]
            if missing_only:
                view = view[view.isna().any(axis=1)]
            if sort:
                if sort not in df.columns:
                    raise HTTPException(400, f"Unknown column '{sort}'.")
                view = view.sort_values(sort, ascending=order == "asc", na_position="last", kind="mergesort")
            page = view.iloc[offset: offset + limit]
            return {
                "columns": list(df.columns),
                "dtypes": {c: str(df[c].dtype) for c in df.columns},
                "total": int(len(df)), "filtered": int(len(view)),
                "offset": offset, "limit": limit,
                "index": [int(i) for i in page.index],
                "rows": clean_json(page.to_numpy(dtype=object).tolist()),
            }
        return await run_in_threadpool(work)

    @api.get("/datasets/{dataset_id}/columns/{column}")
    async def column_detail(dataset_id: str, column: str):
        df = await load_df(dataset_id)
        try:
            return await run_in_threadpool(profiling.column_detail, df, column)
        except DatasetError as exc:
            raise HTTPException(404, str(exc)) from None

    @api.get("/datasets/{dataset_id}/correlations")
    async def dataset_correlations(dataset_id: str):
        df = await load_df(dataset_id)
        return await run_in_threadpool(profiling.correlations, df)

    @api.get("/datasets/{dataset_id}/missing")
    async def dataset_missing(dataset_id: str):
        df = await load_df(dataset_id)
        return await run_in_threadpool(profiling.missing_pattern, df)

    @api.get("/datasets/{dataset_id}/scatter")
    async def dataset_scatter(dataset_id: str, x: str, y: str, color: Optional[str] = None):
        df = await load_df(dataset_id)
        try:
            return await run_in_threadpool(profiling.scatter, df, x, y, color)
        except DatasetError as exc:
            raise HTTPException(404, str(exc)) from None

    @api.get("/datasets/{dataset_id}/quality")
    async def dataset_quality(dataset_id: str, target: Optional[str] = None):
        df = await load_df(dataset_id)
        if target and target not in df.columns:
            raise HTTPException(400, f"Unknown target column '{target}'.")
        return {"findings": await run_in_threadpool(profiling.quality_findings, df, target)}

    @api.get("/datasets/{dataset_id}/target")
    async def dataset_target(dataset_id: str, target: str, task_type: Optional[str] = None):
        df = await load_df(dataset_id)
        try:
            analysis = await run_in_threadpool(profiling.target_analysis, df, target)
        except DatasetError as exc:
            raise HTTPException(400, str(exc)) from None
        task = task_type if task_type in ("classification", "regression") else analysis["problem_type"]
        rec = await run_in_threadpool(recommend.recommend, df, target, task) if task else None
        return {"analysis": analysis, "recommendation": rec}

    @api.post("/datasets/{dataset_id}/clean")
    async def dataset_clean(dataset_id: str, body: CleanRequest):
        await load_df(dataset_id)
        try:
            meta = await run_in_threadpool(registry.apply_actions, dataset_id, body.actions)
        except DatasetError as exc:
            raise HTTPException(400, str(exc)) from None
        meta = dict(meta)
        meta.pop("path", None)
        return meta

    @api.get("/datasets/{dataset_id}/download")
    async def dataset_download(dataset_id: str):
        meta = dataset_or_404(dataset_id)
        return FileResponse(meta["path"], filename=meta["filename"] or f"{dataset_id}.csv")

    @api.get("/samples")
    async def list_samples():
        out = []
        for p in sorted(SAMPLES_DIR.glob("*.csv")):
            out.append({"name": p.stem, "filename": p.name, "size_bytes": p.stat().st_size})
        return {"samples": out}

    @api.post("/samples/{name}/import")
    async def import_sample(name: str):
        path = SAMPLES_DIR / f"{safe_filename(name)}.csv"
        if not path.is_file():
            raise HTTPException(404, "Sample not found.")
        meta = dict(await run_in_threadpool(registry.register_existing_file, path, path.name))
        meta.pop("path", None)
        return meta

    # -------------------------------------------------------------- algorithms
    @api.get("/algorithms")
    async def list_algorithms(task: Optional[str] = None):
        return {"algorithms": [a.to_dict() for a in algs.for_task(task)]}

    @api.post("/recommendations/explain")
    async def explain_recommendations(body: RecommendExplainRequest):
        df = await load_df(body.dataset_id)
        if body.target not in df.columns:
            raise HTTPException(400, "Unknown target column.")
        rec = await run_in_threadpool(recommend.recommend, df, body.target, body.task_type)
        try:
            return await run_in_threadpool(llm.explain_recommendations, rec, body.target)
        except LLMUnavailable as exc:
            raise HTTPException(503, str(exc)) from None
        except Exception as exc:
            raise HTTPException(502, f"AI provider error: {exc}") from None

    # -------------------------------------------------------------- experiments
    def start_experiments(body: ExperimentCreate, df: pd.DataFrame) -> Dict[str, Any]:
        meta = dataset_or_404(body.dataset_id)
        resolved_all = []
        for spec in body.algorithms:
            cfg = {"target": body.target, "task_type": body.task_type, "algorithm_id": spec.id,
                   "hyperparameters": spec.hyperparameters, "preprocessing": body.preprocessing,
                   "split": body.split, "options": body.options}
            try:
                resolved = validate_config(cfg, df)
            except TrainingError as exc:
                raise HTTPException(400, str(exc)) from None
            resolved_all.append(resolved)
        if len({fingerprint(body.dataset_id, r) for r in resolved_all}) != len(resolved_all):
            raise HTTPException(400, "The same algorithm configuration is listed twice.")
        group_id = uuid.uuid4().hex[:10] if len(resolved_all) > 1 else None
        created = []
        for resolved in resolved_all:
            fp = fingerprint(body.dataset_id, resolved)
            alg = algs.get(resolved["algorithm_id"])
            exp_id = str(uuid.uuid4())
            name = (body.name or f"{meta['name']} · {resolved['target']}").strip()
            if len(resolved_all) > 1:
                name = f"{name} · {alg.name}"
            exp = {"id": exp_id, "name": name[:160], "algorithm": alg.name, "algorithm_id": alg.id,
                   "dataset_id": body.dataset_id, "dataset_path": meta["path"],
                   "config": {**resolved, "dataset_id": body.dataset_id, "dataset_name": meta["name"]},
                   "status": "queued", "metrics": {}, "logs": [], "results": {}, "training_time": 0,
                   "created_at": utcnow(), "task_type": resolved["task_type"], "target": resolved["target"],
                   "group_id": group_id, "fingerprint": fp}
            try:
                # reserve the fingerprint before persisting, so duplicates are rejected atomically
                work = make_work(lambda: registry.load(body.dataset_id), resolved, meta, manager)

                def with_quality(job, _work=work, _target=resolved["target"]):
                    out = _work(job)
                    out["results"]["data_quality"] = profiling.quality_findings(registry.load(body.dataset_id), _target)
                    return out
                store.save_experiment(exp)
                manager.submit(exp_id, fp, with_quality)
            except DuplicateJob as dup:
                store.delete_experiment(exp_id)
                for c in created:  # roll back the batch
                    manager.cancel(c["id"])
                raise HTTPException(409, {"message": "An identical experiment is already running.",
                                          "experiment_id": dup.existing_id}) from None
            created.append({"id": exp_id, "algorithm_id": alg.id, "name": exp["name"]})
        return {"group_id": group_id, "experiments": created}

    @api.post("/experiments")
    async def create_experiments(body: ExperimentCreate):
        df = await load_df(body.dataset_id)
        return await run_in_threadpool(start_experiments, body, df)

    @api.get("/experiments")
    async def list_experiments(dataset_id: Optional[str] = None, status: Optional[str] = None):
        exps = store.list_experiments()
        if dataset_id:
            exps = [e for e in exps if e.get("dataset_id") == dataset_id]
        if status:
            exps = [e for e in exps if e["status"] == status]
        return {"status": "success", "experiments": exps}

    @api.get("/experiments/{exp_id}")
    async def get_experiment(exp_id: str):
        exp = experiment_or_404(exp_id)
        job = manager.jobs.get(exp_id)
        if job and job.status in ACTIVE:
            exp["logs"] = list(job.events)
        return {"status": "success", "experiment": exp}

    @api.patch("/experiments/{exp_id}")
    async def rename_experiment(exp_id: str, body: RenameRequest):
        experiment_or_404(exp_id)
        store.update_experiment(exp_id, name=body.name.strip())
        return {"status": "success"}

    @api.post("/experiments/{exp_id}/cancel")
    async def cancel_experiment(exp_id: str):
        exp = experiment_or_404(exp_id)
        if exp["status"] in TERMINAL:
            raise HTTPException(409, f"Experiment already {exp['status']}.")
        if not manager.cancel(exp_id):
            raise HTTPException(409, "Experiment is not running in this server process.")
        return {"status": "cancelling"}

    @api.get("/auth/whoami")
    async def whoami(who: Principal = Depends(principal)):
        return {"owner": who.owner, "authenticated": who.authenticated, "mode": "tokens" if auth.enabled else "local"}

    @api.delete("/experiments/{exp_id}")
    async def delete_experiment(exp_id: str, who: Principal = Depends(principal)):
        """Delete the experiment record and its working model file.

        Saved models are independent copies (see core/model_registry.py) and are kept.
        The artifact is moved aside before the record is deleted so a database failure can
        be rolled back; a job still running for this experiment is cancelled, and the job
        manager removes any artifact it finishes writing afterwards."""
        exp = experiment_or_404(exp_id)
        if exp["status"] in ACTIVE:
            manager.cancel(exp_id)
        artifact = settings.models_dir / f"{exp_id}.joblib"
        trash = settings.models_dir / f".trash-{exp_id}"
        moved = False
        if artifact.exists():
            try:
                os.replace(artifact, trash)
                moved = True
            except OSError as exc:
                raise HTTPException(500, f"Could not remove the experiment's model file: {exc}") from None
        try:
            if not store.delete_experiment(exp_id):
                raise RuntimeError("record not found during delete")
        except Exception as exc:
            if moved:
                os.replace(trash, artifact)
            raise HTTPException(500, f"Could not delete the experiment record: {exc}") from None
        if moved:
            trash.unlink(missing_ok=True)
        kept = store.models_for_experiment(exp_id)
        return {"status": "success", "message": "Experiment deleted", "deleted_id": exp_id,
                "saved_models_retained": len(kept)}

    def completed(exp_id: str) -> Dict[str, Any]:
        exp = experiment_or_404(exp_id)
        if exp["status"] != "completed" or not exp.get("results"):
            raise HTTPException(409, f"Experiment is {exp['status']}; this requires a completed experiment.")
        return exp

    def build_report(exp_id: str) -> Dict[str, Any]:
        exp = completed(exp_id)
        ds = store.get_dataset(exp.get("dataset_id") or "")
        quality = exp["results"].get("data_quality")
        return reports.build(exp, ds, quality)

    @api.get("/experiments/{exp_id}/report")
    async def get_report(exp_id: str):
        return {"status": "success", "report": build_report(exp_id)}

    @api.post("/experiments/{exp_id}/report/narrative")
    async def report_narrative(exp_id: str):
        rep = build_report(exp_id)
        facts = {k: rep[k] for k in ("task", "target", "algorithm", "primary_metric", "observations", "limitations")}
        facts["metrics"] = {k: rep["metrics"][k] for k in ("test", "train", "baseline")}
        facts["cv"] = (rep["metrics"].get("cv") or {}).get("mean")
        facts["rows"] = rep["methodology"]["rows"]
        try:
            ai = await run_in_threadpool(llm.report_narrative, facts)
        except LLMUnavailable as exc:
            raise HTTPException(503, str(exc)) from None
        except Exception as exc:
            raise HTTPException(502, f"AI provider error: {exc}") from None
        exp = store.get_experiment(exp_id)
        results = exp["results"]
        results["ai_narrative"] = {**ai, "generated_at": utcnow()}
        store.update_experiment(exp_id, results=results)
        return results["ai_narrative"]

    @api.get("/experiments/{exp_id}/export-report")
    async def export_report(exp_id: str, format: Literal["txt", "md"] = "txt"):
        rep = build_report(exp_id)
        exp = store.get_experiment(exp_id)
        ai = exp["results"].get("ai_narrative")
        body = reports.to_markdown(rep, ai) if format == "md" else reports.to_text(rep, ai)
        fname = f"ML-Agent_Report_{safe_filename(exp['name'], 'experiment')}.{format}"
        return PlainTextResponse(body, media_type="text/markdown" if format == "md" else "text/plain",
                                 headers={"Content-Disposition": f'attachment; filename="{fname}"'})

    @api.get("/experiments/{exp_id}/export-model")
    async def export_model(exp_id: str, who: Principal = Depends(principal)):
        exp = completed(exp_id)
        path = settings.models_dir / f"{exp_id}.joblib"
        if not path.exists():
            raise HTTPException(404, "Model file not found.")
        return FileResponse(path, media_type="application/octet-stream",
                            filename=f"ML-Agent_{safe_filename(exp['name'], 'model')}.joblib")

    @api.post("/experiments/{exp_id}/predict")
    async def predict(exp_id: str, body: PredictRequest):
        completed(exp_id)
        path = settings.models_dir / f"{exp_id}.joblib"
        if not path.exists():
            raise HTTPException(404, "Model file not found.")

        def work():
            bundle = joblib.load(path)
            cols = bundle["feature_columns"]
            frame = pd.DataFrame(body.rows)
            missing = [c for c in cols if c not in frame.columns]
            if missing:
                raise HTTPException(400, f"Rows are missing feature columns: {', '.join(missing)}")
            frame = frame[cols]
            for c in bundle["numeric"]:
                frame[c] = pd.to_numeric(frame[c], errors="coerce")
            pipe = bundle["pipeline"]
            pred = pipe.predict(frame)
            out = {"predictions": [], "task": bundle["task"]}
            proba = None
            if bundle["task"] == "classification" and hasattr(pipe, "predict_proba"):
                try:
                    proba = pipe.predict_proba(frame)
                except AttributeError:
                    proba = None
            for i, p in enumerate(pred):
                item: Dict[str, Any] = {"prediction": bundle["classes"][int(p)] if bundle["classes"] else float(p)}
                if proba is not None:
                    item["probabilities"] = {bundle["classes"][int(c)]: float(v) for c, v in zip(pipe.classes_, proba[i])}
                out["predictions"].append(item)
            return clean_json(out)
        return await run_in_threadpool(work)

    # -------------------------------------------------------------- saved models
    def registry_error(exc: RegistryError) -> HTTPException:
        return HTTPException(exc.status, str(exc))

    def owned_model(model_id: str, who: Principal) -> Dict[str, Any]:
        # Models of other owners are reported as not found, so their existence is not leaked.
        m = models.get_owned(model_id, who.owner)
        if not m:
            raise HTTPException(404, "Saved model not found.")
        return m

    @api.post("/experiments/{exp_id}/save-model", status_code=201)
    async def save_model(exp_id: str, body: SaveModelRequest, who: Principal = Depends(principal)):
        exp = experiment_or_404(exp_id)
        try:
            saved = await run_in_threadpool(models.save_from_experiment, exp, who.owner, body.name, body.description)
        except RegistryError as exc:
            raise registry_error(exc) from None
        return models.public(saved)

    @api.get("/experiments/{exp_id}/saved-models")
    async def experiment_saved_models(exp_id: str, who: Principal = Depends(principal)):
        return {"models": [models.public(m) for m in store.models_for_experiment(exp_id) if m["owner"] == who.owner]}

    @api.get("/models")
    async def list_models(lineage: Optional[str] = None, who: Principal = Depends(principal)):
        items = [m for m in store.list_saved_models(owner=who.owner) if not lineage or m["lineage"] == lineage]
        return {"models": [models.public(m) for m in items], "owner": who.owner}

    @api.get("/models/{model_id}")
    async def get_model(model_id: str, who: Principal = Depends(principal)):
        m = owned_model(model_id, who)
        versions = [{"id": v["id"], "version": v["version"], "created_at": v["created_at"]}
                    for v in store.list_saved_models(owner=who.owner) if v["lineage"] == m["lineage"]]
        return {**models.public(m, include_manifest=True), "versions": sorted(versions, key=lambda v: v["version"])}

    @api.patch("/models/{model_id}")
    async def update_model(model_id: str, body: ModelUpdate, who: Principal = Depends(principal)):
        owned_model(model_id, who)
        fields = {k: (v.strip() if isinstance(v, str) else v) for k, v in body.model_dump(exclude_unset=True).items()}
        if "name" in fields and not fields["name"]:
            raise HTTPException(400, "A model name is required.")
        if fields:
            store.update_saved_model(model_id, **fields)
        return models.public(store.get_saved_model(model_id))

    @api.delete("/models/{model_id}")
    async def delete_model(model_id: str, who: Principal = Depends(principal)):
        m = owned_model(model_id, who)
        try:
            await run_in_threadpool(models.delete, m)
        except RegistryError as exc:
            raise registry_error(exc) from None
        return {"status": "success", "deleted_id": model_id}

    @api.get("/models/{model_id}/verify")
    async def verify_model(model_id: str, who: Principal = Depends(principal)):
        m = owned_model(model_id, who)
        out = {}
        for fmt in FORMATS:
            try:
                await run_in_threadpool(models.verify, m, fmt)
                out[fmt] = "ok"
            except RegistryError as exc:
                out[fmt] = str(exc)
        return {"id": model_id, "artifacts": out, "valid": all(v == "ok" for v in out.values())}

    @api.get("/models/{model_id}/download")
    async def download_model(model_id: str, format: Literal["joblib", "pkl"] = "joblib", who: Principal = Depends(principal)):
        m = owned_model(model_id, who)
        try:
            path = await run_in_threadpool(models.verify, m, format)
        except RegistryError as exc:
            raise registry_error(exc) from None
        fname = f"{safe_filename(m['lineage'], 'model')}-v{m['version']}.{format}"
        return FileResponse(path, media_type="application/octet-stream", filename=fname,
                            headers={"X-Content-SHA256": m["artifacts"][format]["sha256"], "Cache-Control": "no-store"})

    @api.get("/models/{model_id}/manifest")
    async def model_manifest(model_id: str, who: Principal = Depends(principal)):
        m = owned_model(model_id, who)
        fname = f"{safe_filename(m['lineage'], 'model')}-v{m['version']}.manifest.json"
        return JSONResponse(clean_json(m["manifest"]), headers={"Content-Disposition": f'attachment; filename="{fname}"'})

    @api.post("/models/{model_id}/predict")
    async def model_predict(model_id: str, body: PredictRequest, who: Principal = Depends(principal)):
        m = owned_model(model_id, who)
        try:
            preds = await run_in_threadpool(models.predict, m, body.rows)
        except RegistryError as exc:
            raise registry_error(exc) from None
        except KeyError as exc:
            raise HTTPException(400, str(exc).strip("'\"")) from None
        return {"model_id": model_id, "version": m["version"], "task": m["task"], "predictions": clean_json(preds)}

    # -------------------------------------------------------------- research
    @api.get("/papers/search")
    async def papers_search(q: str, limit: int = Query(10, ge=1, le=30), sort: Literal["relevance", "date"] = "relevance",
                            explain: bool = False):
        try:
            found = await run_in_threadpool(paper_search.search, q, limit, sort)
        except paper_search.PaperSearchError as exc:
            raise HTTPException(502, str(exc)) from None
        out: Dict[str, Any] = {"query": q, "papers": found, "source": "arXiv", "ai": None}
        if explain and found:
            try:
                out["ai"] = await run_in_threadpool(llm.paper_relevance, q, found)
            except LLMUnavailable as exc:
                out["ai_error"] = str(exc)
            except Exception as exc:
                out["ai_error"] = f"AI provider error: {exc}"
        return out

    @api.post("/papers/summarize")
    async def papers_summarize(body: SummarizeRequest):
        try:
            return await run_in_threadpool(llm.summarize_paper, body.model_dump(), body.context)
        except LLMUnavailable as exc:
            raise HTTPException(503, str(exc)) from None
        except Exception as exc:
            raise HTTPException(502, f"AI provider error: {exc}") from None

    # -------------------------------------------------------------- legacy API
    @api.post("/upload-dataset")
    async def legacy_upload(file: UploadFile = File(...)):
        meta = await upload_dataset(file)
        full = store.get_dataset(meta["id"])
        return {"status": "success", "file_path": full["path"], "filename": meta["filename"], "dataset_id": meta["id"]}

    @api.post("/get-dataset-preview")
    async def legacy_preview(file_path: str, rows: int = Query(20, ge=1, le=500)):
        ds_id = legacy_dataset_id(file_path)
        page = await dataset_rows(ds_id, offset=0, limit=rows)
        return {"status": "success", "columns": page["columns"], "dtypes": page["dtypes"],
                "data": [dict(zip(page["columns"], r)) for r in page["rows"]], "total_rows": page["total"]}

    @api.post("/analyze-dataset")
    async def legacy_analyze(request: DatasetRequest):
        if request.dataset_type != "tabular":
            raise HTTPException(400, "Only tabular datasets are supported.")
        ds_id = legacy_dataset_id(request.file_path)
        df = await load_df(ds_id)
        ov = await run_in_threadpool(profiling.overview, df)
        target = request.target_column if request.target_column in df.columns else None
        guess = target or df.columns[-1]
        det = profiling.detect_problem_type(df[guess])
        num = ov["numeric_columns"]
        analysis = {
            "rows": ov["rows"], "columns": ov["columns"], "column_names": list(df.columns),
            "data_types": {c: str(df[c].dtype) for c in df.columns},
            "missing_values": {c["name"]: c["missing"] for c in ov["column_info"]},
            "duplicate_rows": ov["duplicate_rows"], "numeric_features": num,
            "categorical_features": ov["categorical_columns"],
            "numeric_stats": clean_json(df[num].describe().to_dict()) if num else {},
            "target_column": guess, "task_type": det["problem_type"],
            "explanation": ("Target provided by the user. " if target else
                            "No target supplied; showing the last column — confirm it in Model Studio. ") + det["reason"],
            "dataset_id": ds_id,
        }
        return {"status": "success", "analysis": analysis}

    @api.post("/recommend-algorithm")
    async def legacy_recommend(request: DatasetRequest):
        ds_id = legacy_dataset_id(request.file_path)
        df = await load_df(ds_id)
        target = request.target_column if request.target_column in df.columns else df.columns[-1]
        task = request.task_type if request.task_type in ("classification", "regression") else None
        rec = await run_in_threadpool(recommend.recommend, df, target, task)
        top = next((a for a in rec["algorithms"] if a["compatible"]), None)
        if not top:
            raise HTTPException(422, "No compatible algorithm for this target.")
        alg = algs.get(top["id"])
        return {"status": "success", "recommendation": {
            "algorithm": alg.name, "algorithm_id": alg.id, "confidence": None,
            "reasoning": " ".join(top["reasons"]), "hyperparameters": alg.validate_params({}),
            "expected_performance": "Not estimated. Train the model to measure performance.",
            "pros": alg.strengths, "cons": alg.limitations + top["cautions"],
            "method": rec["method"]}}

    @api.post("/train-model")
    async def legacy_train(request: TrainingRequest):
        ds_id = legacy_dataset_id(request.dataset_path)
        df = await load_df(ds_id)
        target = request.config.get("target_column")
        if not target or target not in df.columns:
            raise HTTPException(400, f"config.target_column '{target}' is not a column of the dataset.")
        task = request.config.get("task_type") or "auto"
        if task == "auto":
            task = profiling.detect_problem_type(df[target])["problem_type"]
        try:
            alg = algs.resolve_name(request.algorithm, task)
        except algs.AlgorithmError as exc:
            raise HTTPException(400, str(exc)) from None
        hyper = {k: v for k, v in (request.hyperparameters or {}).items() if k in {p.name for p in alg.params}}
        body = ExperimentCreate(dataset_id=ds_id, target=target, name=request.experiment_name, task_type=task,
                                algorithms=[AlgorithmSpec(id=alg.id, hyperparameters=hyper)],
                                split={"test_size": min(max(request.validation_split, 0.05), 0.5),
                                       "random_state": request.random_state})
        out = await run_in_threadpool(start_experiments, body, df)
        return {"status": "success", "experiment_id": out["experiments"][0]["id"], "message": "Training started"}

    @api.post("/search-papers")
    async def legacy_search(query: str, limit: int = 5):
        try:
            found = await run_in_threadpool(paper_search.search, query, limit, "relevance")
        except paper_search.PaperSearchError as exc:
            raise HTTPException(502, str(exc)) from None
        papers = [{**p, "authors": ", ".join(p["authors"][:3]), "summary": p["abstract"]} for p in found]
        return {"status": "success", "papers": papers, "count": len(papers)}

    app.include_router(api)
    app.include_router(api, prefix="/api")

    # -------------------------------------------------------------- websocket
    async def stream(websocket: WebSocket, exp_id: str):
        await websocket.accept()
        exp = store.get_experiment(exp_id)
        if not exp:
            await websocket.send_json({"type": "error", "message": "Experiment not found."})
            await websocket.close(code=4404)
            return
        queue, history = manager.subscribe(exp_id)
        try:
            if queue is None:
                # Not running in this process: replay the persisted log and final state.
                for evt in exp.get("logs") or []:
                    await websocket.send_json(evt)
                await websocket.send_json({"type": "status", "status": exp["status"], "terminal": exp["status"] in TERMINAL,
                                           "message": exp.get("error") or exp["status"], "error": exp.get("error"),
                                           "summary": exp.get("metrics"), "experiment_id": exp_id})
                return
            last_seq = 0
            for evt in history:
                await websocket.send_json(evt)
                last_seq = evt["seq"]
                if evt.get("terminal"):
                    return
            while True:
                try:
                    evt = await asyncio.wait_for(queue.get(), timeout=25)
                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "heartbeat"})
                    continue
                if evt["seq"] <= last_seq:
                    continue
                await websocket.send_json(evt)
                last_seq = evt["seq"]
                if evt.get("terminal"):
                    return
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            if queue is not None:
                manager.unsubscribe(exp_id, queue)
            try:
                await websocket.close()
            except RuntimeError:
                pass

    for prefix in ("", "/api"):
        app.add_api_websocket_route(f"{prefix}/ws/experiments/{{exp_id}}", stream)
        app.add_api_websocket_route(f"{prefix}/ws/train/{{exp_id}}", stream)

    # -------------------------------------------------------------- SPA
    static_dir = static_dir or (BACKEND_DIR / "static")
    if static_dir.is_dir() and (static_dir / "index.html").exists():
        index = static_dir / "index.html"
        if (static_dir / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

        @app.middleware("http")
        async def spa_fallback(request: Request, call_next):
            path = request.url.path
            if (request.method == "GET" and "text/html" in request.headers.get("accept", "")
                    and not path.startswith(("/api", "/ws", "/docs", "/openapi", "/redoc", "/assets"))):
                candidate = (static_dir / path.lstrip("/")).resolve()
                if path != "/" and candidate.is_file() and candidate.is_relative_to(static_dir.resolve()):
                    return FileResponse(candidate)
                return FileResponse(index)
            return await call_next(request)

        @app.get("/{asset_path:path}", include_in_schema=False)
        async def static_files(asset_path: str):
            candidate = (static_dir / asset_path).resolve()
            if asset_path and candidate.is_file() and candidate.is_relative_to(static_dir.resolve()):
                return FileResponse(candidate)
            raise HTTPException(404, "Not found")

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        detail = exc.detail
        payload = detail if isinstance(detail, dict) else {"message": str(detail)}
        payload.setdefault("message", "Request failed")
        return JSONResponse({"detail": payload["message"], **payload}, status_code=exc.status_code,
                            headers=getattr(exc, "headers", None))

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=port)
