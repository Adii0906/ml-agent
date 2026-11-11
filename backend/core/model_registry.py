"""Saved-model registry: versioned, owner-scoped, integrity-checked model packages.

Layout (one directory per saved model, named by its stable id — never exposed to clients):

    <data>/saved_models/<model_id>/
        model.joblib     fitted Pipeline (preprocessing + estimator) + label classes + manifest
        model.pkl        the same object, pickle protocol 5
        manifest.json    metadata (also stored in the database)

Design notes for future deployment:
* ``model_id`` is stable and never reused; (owner, lineage, version) tracks versions of a name.
* Artifacts are *copies* made from the experiment's working artifact, so deleting the
  experiment never affects a saved model.
* Every artifact has a recorded SHA-256 that is verified before it is served or
  deserialised. Only artifacts written by this server are ever loaded — there is no
  upload path for pickles.
* ``mlagent_runtime`` is the only non-library code an artifact needs, so a future
  serving container can load it without the rest of the backend.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import re
import shutil
import sqlite3
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import sklearn

from mlagent_runtime import ARTIFACT_FORMAT, file_sha256, load_artifact
from mlagent_runtime import predict as runtime_predict

from .config import Settings
from .store import Store
from .util import clean_json, utcnow

log = logging.getLogger(__name__)

MODEL_ID_RE = re.compile(r"^mdl_[0-9a-f]{16}$")
FORMATS = ("joblib", "pkl")
API_VERSION = "2.1.0"


class RegistryError(Exception):
    status = 400


class NotSavable(RegistryError):
    status = 409


class IntegrityError(RegistryError):
    status = 409


class ArtifactMissing(RegistryError):
    status = 410


class StorageError(RegistryError):
    status = 500


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s[:80] or "model"


class ModelRegistry:
    def __init__(self, settings: Settings, store: Store, cache_size: int = 4):
        self.settings = settings
        self.store = store
        self.root = settings.saved_models_dir
        self._cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- housekeeping
    def sweep(self) -> int:
        """Remove leftovers from interrupted saves/deletes (temp + trash dirs, partial files)."""
        removed = 0
        for p in list(self.root.glob(".tmp-*")) + list(self.root.glob(".trash-*")):
            shutil.rmtree(p, ignore_errors=True)
            removed += 1
        for p in list(self.settings.models_dir.glob(".trash-*")) + list(self.settings.models_dir.glob("*.partial")):
            p.unlink(missing_ok=True)
            removed += 1
        return removed

    def _dir(self, model_id: str) -> Path:
        if not MODEL_ID_RE.match(model_id or ""):
            raise RegistryError("Invalid model id.")
        return self.root / model_id

    def artifact_path(self, model: Dict[str, Any], fmt: str) -> Path:
        if fmt not in FORMATS:
            raise RegistryError(f"Format must be one of: {', '.join(FORMATS)}.")
        return self._dir(model["id"]) / f"model.{fmt}"

    # ---------------------------------------------------------------- status
    def status(self, model: Dict[str, Any]) -> str:
        """Cheap check used for listings: 'ready' | 'missing' | 'corrupted' (size mismatch)."""
        for fmt in FORMATS:
            info = (model.get("artifacts") or {}).get(fmt)
            p = self.artifact_path(model, fmt)
            if not info or not p.is_file():
                return "missing"
            if p.stat().st_size != info["bytes"]:
                return "corrupted"
        return "ready"

    def verify(self, model: Dict[str, Any], fmt: str) -> Path:
        """Full SHA-256 verification of one artifact. Returns the path when valid."""
        p = self.artifact_path(model, fmt)
        info = (model.get("artifacts") or {}).get(fmt)
        if not info or not p.is_file():
            raise ArtifactMissing("The model artifact is missing from storage.")
        if file_sha256(p) != info["sha256"]:
            raise IntegrityError("Integrity check failed: the stored artifact does not match its recorded SHA-256.")
        return p

    def public(self, model: Dict[str, Any], include_manifest: bool = False) -> Dict[str, Any]:
        out = {k: v for k, v in model.items() if k not in ("manifest",)}
        out["artifacts"] = {fmt: {"sha256": a["sha256"], "bytes": a["bytes"]} for fmt, a in (model.get("artifacts") or {}).items()}
        out["status"] = self.status(model)
        out["source_experiment_exists"] = bool(model.get("experiment_id")) and self.store.get_experiment(model["experiment_id"]) is not None
        if include_manifest:
            out["manifest"] = model.get("manifest")
        return out

    # ---------------------------------------------------------------- save
    def save_from_experiment(self, exp: Dict[str, Any], owner: str, name: str, description: Optional[str]) -> Dict[str, Any]:
        name = (name or "").strip()
        description = (description or "").strip() or None
        if not name:
            raise RegistryError("A model name is required.")
        if len(name) > 120:
            raise RegistryError("Model names are limited to 120 characters.")
        if description and len(description) > 2000:
            raise RegistryError("Descriptions are limited to 2000 characters.")
        if exp.get("status") != "completed" or not exp.get("results"):
            raise NotSavable(f"Only completed experiments can be saved (this one is {exp.get('status')}).")
        res = exp["results"]
        src = self.settings.models_dir / f"{exp['id']}.joblib"
        if not src.is_file():
            raise NotSavable("The experiment's trained model file is missing; re-run the experiment to save it.")
        limit = self.settings.max_model_mb * 1024 * 1024
        if src.stat().st_size > limit:
            raise NotSavable(f"The model artifact exceeds the {self.settings.max_model_mb} MB limit (MLAGENT_MAX_MODEL_MB).")
        recorded = (res.get("artifact") or {}).get("sha256")
        if recorded and file_sha256(src) != recorded:
            raise IntegrityError("The experiment's model file failed its integrity check and will not be loaded.")
        bundle = joblib.load(src)  # server-written artifact (hash-verified when a hash was recorded)

        model_id = f"mdl_{uuid.uuid4().hex[:16]}"
        lineage = slugify(name)
        created = utcnow()
        tmp = self.root / f".tmp-{model_id}"
        final = self._dir(model_id)
        for attempt in range(3):
            version = self.store.next_model_version(owner, lineage)
            manifest = self._manifest(exp, bundle, model_id, owner, name, description, lineage, version, created)
            try:
                tmp.mkdir(parents=True)
                artifact = {"format": ARTIFACT_FORMAT, "pipeline": bundle["pipeline"], "classes": bundle["classes"],
                            "feature_columns": bundle["feature_columns"], "numeric": bundle["numeric"],
                            "categorical": bundle["categorical"], "target": bundle["target"], "task": bundle["task"],
                            "manifest": manifest}
                joblib.dump(artifact, tmp / "model.joblib", compress=3)
                with (tmp / "model.pkl").open("wb") as f:
                    pickle.dump(artifact, f, protocol=5)
                artifacts = {fmt: {"file": f"model.{fmt}", "sha256": file_sha256(tmp / f"model.{fmt}"),
                                   "bytes": (tmp / f"model.{fmt}").stat().st_size} for fmt in FORMATS}
                (tmp / "manifest.json").write_text(json.dumps(clean_json({**manifest, "artifacts": artifacts}), indent=2))
            except OSError as exc:
                shutil.rmtree(tmp, ignore_errors=True)
                raise StorageError(f"Could not write the model artifact: {exc}") from exc
            row = {
                "id": model_id, "owner": owner, "name": name, "description": description, "lineage": lineage,
                "version": version, "experiment_id": exp["id"], "algorithm_id": res["algorithm"]["id"],
                "algorithm_name": res["algorithm"]["name"], "task": res["task"], "target": res["target"],
                "dataset_id": res["data"].get("dataset_id"), "dataset_name": res["data"].get("dataset_name"),
                "dataset_sha256": res["reproducibility"].get("dataset_sha256"),
                "primary_metric": res["primary_metric"], "primary_value": res["metrics"]["test"].get(res["primary_metric"]),
                "metrics": manifest["metrics"], "manifest": {**manifest, "artifacts": artifacts}, "artifacts": artifacts,
                "trained_at": exp.get("completed_at"), "created_at": created,
            }
            try:
                saved = self.store.insert_saved_model(row)
            except sqlite3.IntegrityError:  # concurrent save took this version number: retry
                shutil.rmtree(tmp, ignore_errors=True)
                continue
            except sqlite3.Error as exc:
                shutil.rmtree(tmp, ignore_errors=True)
                raise StorageError(f"Could not record the saved model: {exc}") from exc
            try:
                os.replace(tmp, final)
            except OSError as exc:
                self.store.delete_saved_model(model_id)
                shutil.rmtree(tmp, ignore_errors=True)
                raise StorageError(f"Could not move the model artifact into place: {exc}") from exc
            return saved
        raise StorageError("Could not allocate a version number; try again.")

    def _manifest(self, exp, bundle, model_id, owner, name, description, lineage, version, created) -> Dict[str, Any]:
        res = exp["results"]
        dtypes = res["data"].get("feature_dtypes") or {}
        columns = [{"name": c, "role": "numeric", "dtype": dtypes.get(c)} for c in bundle["numeric"]] + \
                  [{"name": c, "role": "categorical", "dtype": dtypes.get(c)} for c in bundle["categorical"]]
        cv = res["metrics"].get("cv")
        rep = res["reproducibility"]
        return {
            "format": "mlagent.saved_model", "format_version": 1, "artifact_format": ARTIFACT_FORMAT,
            "model_id": model_id, "name": name, "description": description, "lineage": lineage, "version": version,
            "owner": owner, "created_at": created,
            "source": {"experiment_id": exp["id"], "experiment_name": exp["name"], "trained_at": exp.get("completed_at"),
                       "split_id": res["split"]["split_id"]},
            "task": res["task"], "target": res["target"],
            "algorithm": {**res["algorithm"], "estimator_class": res["model_info"].get("estimator")},
            "input_schema": {
                "columns": columns,
                "column_order": bundle["feature_columns"],
                "excluded_columns": res["data"]["features"]["excluded"],
                "missing_values": "allowed — imputed by the pipeline",
                "unknown_categories": "allowed — ignored by the encoder",
                "extra_columns": "ignored",
            },
            "output_schema": {"type": "class_label" if res["task"] == "classification" else "number",
                              "classes": bundle["classes"], "probabilities": res["task"] == "classification"},
            "dataset": {"id": res["data"].get("dataset_id"), "name": res["data"].get("dataset_name"),
                        "sha256": rep.get("dataset_sha256"), "rows_used": res["data"]["n_used"],
                        "n_train": res["data"]["n_train"], "n_test": res["data"]["n_test"]},
            "preprocessing": res["preprocessing"],
            "pipeline_stages": res["pipeline_stages"],
            "training": {"split": res["split"], "random_state": res["split"]["random_state"],
                         "test_size": res["split"]["test_size"], "stratified": res["split"]["stratified"],
                         "cv_folds": res["split"]["cv_folds"]},
            "metrics": {"primary_metric": res["primary_metric"], "test": res["metrics"]["test"],
                        "baseline": res["metrics"]["baseline"], "train": res["metrics"]["train"],
                        "cv": {"mean": cv["mean"], "std": cv["std"], "k": cv["k"]} if cv else None,
                        "roc_auc": (res.get("classification") or {}).get("roc", {}).get("auc")},
            "environment": {"python": rep["python"], "scikit_learn": rep["scikit_learn"], "pandas": rep["pandas"],
                            "numpy": rep["numpy"], "joblib": joblib.__version__, "saved_with_scikit_learn": sklearn.__version__,
                            "mlagent_api": API_VERSION},
            "requirements": {"packages": {"scikit-learn": rep["scikit_learn"], "pandas": rep["pandas"], "numpy": rep["numpy"]},
                             "runtime_module": "mlagent_runtime (backend/mlagent_runtime in the ML-Agent repository)"},
            "deployment": {"status": "not_deployed"},
        }

    # ---------------------------------------------------------------- access
    def get_owned(self, model_id: str, owner: str) -> Optional[Dict[str, Any]]:
        if not MODEL_ID_RE.match(model_id or ""):
            return None
        m = self.store.get_saved_model(model_id)
        return m if m and m["owner"] == owner else None

    def load(self, model: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if model["id"] in self._cache:
                self._cache.move_to_end(model["id"])
                return self._cache[model["id"]]
        path = self.verify(model, "joblib")
        bundle = load_artifact(path, expected_sha256=model["artifacts"]["joblib"]["sha256"])
        with self._lock:
            self._cache[model["id"]] = bundle
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return bundle

    def predict(self, model: Dict[str, Any], rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return runtime_predict(self.load(model), rows)

    # ---------------------------------------------------------------- delete
    def delete(self, model: Dict[str, Any]) -> None:
        """Remove files and record together: the directory is moved aside first so a failed
        database delete can be rolled back, and a failed file removal never leaves a record
        pointing at nothing."""
        d = self._dir(model["id"])
        trash = self.root / f".trash-{model['id']}-{uuid.uuid4().hex[:6]}"
        moved = False
        if d.exists():
            try:
                os.replace(d, trash)
                moved = True
            except OSError as exc:
                raise StorageError(f"Could not remove the model files: {exc}") from exc
        try:
            if not self.store.delete_saved_model(model["id"]):
                raise StorageError("The model record could not be deleted.")
        except (sqlite3.Error, StorageError) as exc:
            if moved:
                os.replace(trash, d)
            raise StorageError(f"Could not delete the model record: {exc}") from exc
        with self._lock:
            self._cache.pop(model["id"], None)
        if moved:
            shutil.rmtree(trash, ignore_errors=True)  # leftovers are swept at startup
