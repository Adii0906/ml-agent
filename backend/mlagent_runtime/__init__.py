"""mlagent_runtime — the only code needed to load and run an ML-Agent model artifact.

Saved ML-Agent models are scikit-learn Pipelines. Two of their preprocessing
components are custom and live here so that an artifact can be loaded outside
the ML-Agent server (a notebook, a batch job, a future deployment service):

    pip install "scikit-learn==<version in manifest>" pandas numpy
    # put this single file/package on PYTHONPATH, then:
    from mlagent_runtime import load_artifact, predict
    bundle = load_artifact("model.joblib", expected_sha256="<from manifest>")
    predict(bundle, [{"age": 41, "plan": "pro", ...}])

Dependencies: numpy, pandas, scikit-learn (+ joblib, which ships with scikit-learn).
Never load artifacts from untrusted sources: joblib/pickle files can execute code
when loaded. ``load_artifact`` verifies a SHA-256 before deserialising.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["IQRClipper", "categorical_to_str", "infinite_to_nan", "load_artifact", "predict", "file_sha256", "ARTIFACT_FORMAT"]

ARTIFACT_FORMAT = "mlagent.model/v1"


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip each numeric column to [Q1 - k·IQR, Q3 + k·IQR] using fences learned in ``fit``."""

    def __init__(self, k: float = 1.5):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.k * iqr
        self.upper_ = q3 + self.k * iqr
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)


def categorical_to_str(X):
    """Categorical values as strings so mixed int/str columns encode consistently."""
    if isinstance(X, pd.DataFrame):
        return X.apply(lambda s: s.astype(str).astype(object).where(s.notna(), np.nan))
    return X


def infinite_to_nan(X):
    """Treat +/-inf as missing so the imputer (fitted on training rows) fills it."""
    if isinstance(X, pd.DataFrame):
        return X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = np.asarray(X, dtype=float)
    return np.where(np.isfinite(X), X, np.nan)


def file_sha256(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_artifact(path, expected_sha256: Optional[str] = None) -> Dict[str, Any]:
    """Load a .joblib or .pkl ML-Agent artifact.

    If ``expected_sha256`` is given (it is recorded in the model manifest), the file is
    hashed first and refused on mismatch — before any deserialisation happens.
    """
    path = Path(path)
    if expected_sha256 is not None:
        actual = file_sha256(path)
        if actual != expected_sha256:
            raise ValueError(f"Artifact integrity check failed: sha256 {actual} != expected {expected_sha256}")
    if path.suffix == ".pkl":
        with path.open("rb") as f:
            bundle = pickle.load(f)  # noqa: S301 - hash verified above when provided
    else:
        import joblib
        bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle or "feature_columns" not in bundle:
        raise ValueError("Not an ML-Agent model artifact.")
    return bundle


def _frame(bundle: Dict[str, Any], rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    cols: List[str] = bundle["feature_columns"]
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f"Rows are missing feature columns: {', '.join(missing)}")
    frame = frame[cols].copy()
    for c in bundle.get("numeric", []):
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    return frame


def predict(bundle: Dict[str, Any], rows: Iterable[Dict[str, Any]], proba: bool = True) -> List[Dict[str, Any]]:
    """Run the full preprocessing + model pipeline on raw input rows (dicts of column → value)."""
    frame = _frame(bundle, rows)
    pipe = bundle["pipeline"]
    pred = pipe.predict(frame)
    classes = bundle.get("classes")
    probabilities = None
    if proba and classes and hasattr(pipe, "predict_proba"):
        try:
            probabilities = pipe.predict_proba(frame)
        except AttributeError:  # e.g. SVC without probability estimates
            probabilities = None
    out = []
    for i, p in enumerate(pred):
        item: Dict[str, Any] = {"prediction": classes[int(p)] if classes else float(p)}
        if probabilities is not None:
            item["probabilities"] = {classes[int(c)]: float(v) for c, v in zip(pipe.classes_, probabilities[i])}
        out.append(item)
    return out
