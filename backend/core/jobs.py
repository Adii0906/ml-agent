"""Training job manager.

* Training runs in a bounded thread pool so the event loop (and the API) stays
  responsive while scikit-learn fits.
* Each job keeps an ordered event log with sequence numbers. WebSocket clients
  receive a replay of past events on connect, then live events — a client that
  connects late (or reconnects) never misses the outcome.
* Identical in-flight submissions are rejected (fingerprint check).
* Cancellation is cooperative: it takes effect between stages / CV folds.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import joblib

from .store import Store
from .training import Cancelled, TrainingError, run_training
from .util import clean_json, utcnow

log = logging.getLogger(__name__)

TERMINAL = {"completed", "failed", "cancelled"}
ACTIVE = {"queued", "running"}


@dataclass
class Job:
    exp_id: str
    fingerprint: str
    status: str = "queued"
    events: List[Dict[str, Any]] = field(default_factory=list)
    cancel: threading.Event = field(default_factory=threading.Event)
    subscribers: Set[asyncio.Queue] = field(default_factory=set)
    started: Optional[float] = None


class DuplicateJob(Exception):
    def __init__(self, existing_id: str):
        super().__init__(existing_id)
        self.existing_id = existing_id


class JobManager:
    def __init__(self, store: Store, models_dir: Path, max_workers: int = 2):
        self.store = store
        self.models_dir = models_dir
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="train")
        self.jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def recover_interrupted(self) -> int:
        """Experiments left queued/running by a previous process can never finish."""
        stale = self.store.experiments_with_status(ACTIVE | {"initializing", "training"})
        for exp in stale:
            self.store.update_experiment(exp["id"], status="failed", completed_at=utcnow(),
                                         error="Interrupted: the server stopped before training finished.")
        return len(stale)

    # ---------------------------------------------------------------- submit
    def submit(self, exp_id: str, fingerprint: str, work: Callable[[], Dict[str, Any]]) -> None:
        with self._lock:
            for job in self.jobs.values():
                if job.fingerprint == fingerprint and job.status in ACTIVE:
                    raise DuplicateJob(job.exp_id)
            job = Job(exp_id=exp_id, fingerprint=fingerprint)
            self.jobs[exp_id] = job
        self._publish(job, {"type": "status", "status": "queued", "message": "Queued"})
        self.executor.submit(self._run, job, work)

    def _run(self, job: Job, work: Callable[[], Dict[str, Any]]) -> None:
        with self._lock:
            if job.status in TERMINAL:  # cancelled while queued
                return
            job.status = "running"
        job.started = time.time()
        self.store.update_experiment(job.exp_id, status="running", started_at=utcnow())
        self._publish(job, {"type": "status", "status": "running", "message": "Training started"})
        try:
            out = work(job)
            model_path = self.models_dir / f"{job.exp_id}.joblib"
            self.emit(job, stage="persist", stage_index=6, stage_count=7, message="Serialising fitted pipeline")
            joblib.dump(out["bundle"], model_path)
            elapsed = round(time.time() - job.started, 3)
            self.store.update_experiment(job.exp_id, results=out["results"], metrics=out["summary"],
                                         training_time=elapsed)
            self._finish(job, "completed", summary=out["summary"])
        except Cancelled:
            self._finish(job, "cancelled", error="Cancelled by user.")
        except TrainingError as exc:
            self._finish(job, "failed", error=str(exc))
        except Exception as exc:  # unexpected: keep the trace in logs, a clean message for users
            log.error("Training %s crashed: %s\n%s", job.exp_id, exc, traceback.format_exc())
            self._finish(job, "failed", error=f"{type(exc).__name__}: {exc}")

    def _finish(self, job: Job, status: str, error: Optional[str] = None, summary=None) -> None:
        job.status = status
        elapsed = round(time.time() - job.started, 3) if job.started else 0
        evt = {"type": "status", "status": status, "message": error or "Training completed", "terminal": True}
        if summary is not None:
            evt["summary"] = summary
        if error:
            evt["error"] = error
        # Persist first (so a client reacting to the terminal event reads final state), with the
        # terminal event included in the stored log under the sequence number it will be published with.
        stored_evt = clean_json({**evt, "seq": len(job.events) + 1, "time": utcnow(), "experiment_id": job.exp_id})
        fields: Dict[str, Any] = {"status": status, "completed_at": utcnow(), "error": error,
                                  "logs": job.events + [stored_evt]}
        if status != "completed":
            fields["training_time"] = elapsed
        self.store.update_experiment(job.exp_id, **fields)
        self._publish(job, evt)

    # ---------------------------------------------------------------- events
    def emit(self, job: Job, **event: Any) -> None:
        self._publish(job, {"type": "progress", **event})

    def _publish(self, job: Job, event: Dict[str, Any]) -> None:
        event = clean_json({**event, "seq": len(job.events) + 1, "time": utcnow(), "experiment_id": job.exp_id})
        with self._lock:
            job.events.append(event)
            subscribers = list(job.subscribers)
        if self.loop is None:
            return
        for q in subscribers:
            try:
                self.loop.call_soon_threadsafe(q.put_nowait, event)
            except RuntimeError:  # loop closed during shutdown
                pass

    def subscribe(self, exp_id: str) -> tuple[Optional[asyncio.Queue], List[Dict[str, Any]]]:
        """Register a queue and return (queue, history). Registration and history
        snapshot happen under one lock so no event is lost or duplicated."""
        with self._lock:
            job = self.jobs.get(exp_id)
            if job is None:
                return None, []
            q: asyncio.Queue = asyncio.Queue()
            job.subscribers.add(q)
            return q, list(job.events)

    def unsubscribe(self, exp_id: str, q: asyncio.Queue) -> None:
        with self._lock:
            job = self.jobs.get(exp_id)
            if job:
                job.subscribers.discard(q)

    def cancel(self, exp_id: str) -> bool:
        job = self.jobs.get(exp_id)
        if not job or job.status in TERMINAL:
            return False
        job.cancel.set()
        with self._lock:
            was_queued = job.status == "queued"
            if was_queued:
                job.status = "cancelled"
        if was_queued:
            self._finish(job, "cancelled", error="Cancelled before training started.")
        else:
            self._publish(job, {"type": "status", "status": job.status,
                                "message": "Cancellation requested — stops after the current step"})
        return True

    def active_count(self) -> int:
        return sum(1 for j in self.jobs.values() if j.status in ACTIVE)

    def shutdown(self) -> None:
        for job in self.jobs.values():
            job.cancel.set()
        self.executor.shutdown(wait=False, cancel_futures=True)


def make_work(df_loader: Callable[[], Any], resolved: Dict[str, Any], dataset_meta: Dict[str, Any], manager: JobManager):
    def work(job: Job) -> Dict[str, Any]:
        df = df_loader()
        return run_training(df, resolved, dataset_meta,
                            emit=lambda **e: manager.emit(job, **e),
                            is_cancelled=job.cancel.is_set)
    return work
