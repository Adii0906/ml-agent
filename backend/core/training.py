"""Experiment execution: validation, splitting, CV, fitting, evaluation.

Methodology (enforced, not configurable away):
* Rows without a target are dropped; exact duplicates are dropped *before* the
  split (by default) so identical rows cannot sit in both train and test.
* One held-out test split. All learned preprocessing lives in the Pipeline and
  is fitted on the training split only.
* Cross-validation runs on the training split only; the test split is touched
  exactly once, by the final fitted pipeline.
* Train, CV and test metrics are reported separately and labelled as such.
"""
from __future__ import annotations

import hashlib
import math
import platform
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, StratifiedKFold, learning_curve, train_test_split
from sklearn.preprocessing import LabelEncoder

from . import algorithms as algs
from .pipeline import (
    PipelineConfigError,
    build_pipeline,
    describe_pipeline,
    resolve_scaling,
    select_features,
    validate_preprocessing,
)
from .profiling import CLASSIFICATION_MAX_CLASSES, detect_problem_type
from .util import clean_json

MIN_ROWS = 10


class TrainingError(ValueError):
    """Invalid configuration or data that makes training impossible."""


class Cancelled(Exception):
    pass


STAGES = [
    ("prepare", "Validate & filter rows"),
    ("split", "Train / test split"),
    ("cv", "Cross-validation (training split)"),
    ("fit", "Fit final pipeline"),
    ("evaluate", "Evaluate on held-out test set"),
    ("explain", "Feature importance & diagnostics"),
    ("persist", "Save model artifact"),
]
STAGE_INDEX = {s: i for i, (s, _) in enumerate(STAGES)}

DEFAULT_SPLIT = {"test_size": 0.2, "random_state": 42, "stratify": True, "cv_folds": 5}
DEFAULT_OPTIONS = {"learning_curve": False, "permutation_importance": True}


# ------------------------------------------------------------ config
def validate_config(cfg: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """Validate and resolve a training request. Raises TrainingError with a
    user-facing message. Cheap enough to call synchronously at submit time."""
    target = cfg.get("target")
    if not target or target not in df.columns:
        raise TrainingError(f"Target column '{target}' does not exist in the dataset.")
    if df.shape[1] < 2:
        raise TrainingError("The dataset needs at least one feature column besides the target.")

    task = cfg.get("task_type") or "auto"
    if task == "auto":
        task = detect_problem_type(df[target])["problem_type"]
        if task is None:
            raise TrainingError(f"Target '{target}' has fewer than two distinct values.")
    if task not in ("classification", "regression"):
        raise TrainingError("task_type must be 'classification', 'regression' or 'auto'.")

    try:
        alg = algs.get(cfg.get("algorithm_id", ""))
    except algs.AlgorithmError as exc:
        raise TrainingError(str(exc)) from None
    if alg.task != task:
        raise TrainingError(f"{alg.name} is a {alg.task} algorithm but the task is {task}.")
    if not alg.available:
        raise TrainingError(f"{alg.name} is unavailable: {alg.unavailable_reason}")
    try:
        hyper = alg.validate_params(cfg.get("hyperparameters"))
        prep = validate_preprocessing(cfg.get("preprocessing"))
    except (algs.AlgorithmError, PipelineConfigError) as exc:
        raise TrainingError(str(exc)) from None

    split = dict(DEFAULT_SPLIT)
    unknown = set(cfg.get("split") or {}) - set(split)
    if unknown:
        raise TrainingError(f"Unknown split options: {', '.join(sorted(unknown))}.")
    split.update(cfg.get("split") or {})
    try:
        split["test_size"] = float(split["test_size"])
        split["random_state"] = int(split["random_state"])
        split["cv_folds"] = int(split["cv_folds"])
    except (TypeError, ValueError):
        raise TrainingError("test_size, random_state and cv_folds must be numbers.") from None
    split["stratify"] = bool(split["stratify"])
    if not 0.05 <= split["test_size"] <= 0.5:
        raise TrainingError("test_size must be between 0.05 and 0.5.")
    if split["cv_folds"] != 0 and not 2 <= split["cv_folds"] <= 10:
        raise TrainingError("cv_folds must be 0 (disabled) or between 2 and 10.")
    if not 0 <= split["random_state"] <= 2**31 - 1:
        raise TrainingError("random_state must be a non-negative 32-bit integer.")

    options = dict(DEFAULT_OPTIONS)
    options.update({k: bool(v) for k, v in (cfg.get("options") or {}).items() if k in DEFAULT_OPTIONS})

    try:
        features = select_features(df, target, prep["excluded_columns"])
    except PipelineConfigError as exc:
        raise TrainingError(str(exc)) from None
    if task == "regression":
        y = df[target].dropna()
        if not pd.api.types.is_numeric_dtype(y):
            coerced = pd.to_numeric(y, errors="coerce")
            bad = y[coerced.isna()]
            if len(bad):
                raise TrainingError(f"Regression needs a numeric target, but '{target}' contains values like {bad.iloc[0]!r}.")

    return {
        "target": target,
        "task_type": task,
        "algorithm_id": alg.id,
        "hyperparameters": hyper,
        "preprocessing": prep,
        "split": split,
        "options": options,
        "features": {k: features[k] for k in ("numeric", "categorical", "excluded", "skipped")},
    }


def fingerprint(dataset_id: str, resolved: Dict[str, Any]) -> str:
    import json
    payload = json.dumps({"d": dataset_id, **{k: resolved[k] for k in
                          ("target", "task_type", "algorithm_id", "hyperparameters", "preprocessing", "split", "options")}},
                         sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()


# ------------------------------------------------------------ metrics
def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def regression_metrics(y_true, y_pred) -> Dict[str, Optional[float]]:
    mse = mean_squared_error(y_true, y_pred)
    out = {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "r2": r2_score(y_true, y_pred) if len(y_true) > 1 else None,
        "median_ae": median_absolute_error(y_true, y_pred),
        "max_error": max_error(y_true, y_pred),
        "mape": None,
    }
    if np.all(np.asarray(y_true) != 0):
        out["mape"] = mean_absolute_percentage_error(y_true, y_pred)
    return out


def _scores(model, X):
    """Return (kind, scores) for ROC computations."""
    if hasattr(model, "predict_proba"):
        try:
            return "proba", model.predict_proba(X)
        except (AttributeError, NotImplementedError):
            pass
    if hasattr(model, "decision_function"):
        return "decision", model.decision_function(X)
    return None, None


def _downsample_curve(fpr, tpr, thr, max_points=200):
    if len(fpr) <= max_points:
        idx = np.arange(len(fpr))
    else:
        idx = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
    return [{"fpr": float(fpr[i]), "tpr": float(tpr[i])} for i in idx]


# ------------------------------------------------------------ runner
class Runner:
    def __init__(self, emit: Callable[..., None], is_cancelled: Callable[[], bool]):
        self._emit = emit
        self._is_cancelled = is_cancelled
        self.timing: Dict[str, float] = {}
        self.warnings: List[str] = []

    def stage(self, stage: str, message: str, **extra):
        self._emit(stage=stage, stage_index=STAGE_INDEX[stage], stage_count=len(STAGES), message=message, **extra)

    def warn(self, stage: str, message: str):
        self.warnings.append(message)
        self._emit(stage=stage, stage_index=STAGE_INDEX[stage], stage_count=len(STAGES), message=message, level="warning")

    def check(self):
        if self._is_cancelled():
            raise Cancelled()


def run_training(df: pd.DataFrame, resolved: Dict[str, Any], dataset_meta: Dict[str, Any],
                 emit: Callable[..., None], is_cancelled: Callable[[], bool] = lambda: False) -> Dict[str, Any]:
    """Execute an experiment. Returns {'results': ..., 'bundle': ..., 'summary': ...}."""
    r = Runner(emit, is_cancelled)
    t_total = time.perf_counter()
    target, task = resolved["target"], resolved["task_type"]
    prep, split_cfg, options = resolved["preprocessing"], resolved["split"], resolved["options"]
    alg = algs.get(resolved["algorithm_id"])
    rs = split_cfg["random_state"]

    # ---------------------------------------------------------- prepare
    t0 = time.perf_counter()
    r.stage("prepare", f"Validating {len(df):,} rows")
    data = df
    n_raw = len(data)
    missing_target = int(data[target].isna().sum())
    if missing_target:
        data = data[data[target].notna()]
        r.warn("prepare", f"Dropped {missing_target} rows with no target value.")
    features = select_features(data, target, prep["excluded_columns"])
    feature_cols = features["numeric"] + features["categorical"]
    n_dups = 0
    if prep["drop_duplicates"]:
        # Duplicates are judged on the columns the model sees (features + target), so rows that
        # differ only in an excluded ID column still count as duplicates.
        dup_mask = data.duplicated(subset=feature_cols + [target])
        n_dups = int(dup_mask.sum())
        if n_dups:
            data = data[~dup_mask]
            r.stage("prepare", f"Dropped {n_dups} duplicate rows (identical features and target) before splitting")
    if features["excluded"]:
        r.stage("prepare", f"Excluded columns: {', '.join(features['excluded'])}")

    y_raw = data[target]
    classes: Optional[List[str]] = None
    n_rare_dropped = 0
    if task == "classification":
        y_str = y_raw.astype(str)
        counts = y_str.value_counts()
        if counts.size < 2:
            raise TrainingError("The target has a single class after filtering; a classifier needs at least two.")
        if counts.size > CLASSIFICATION_MAX_CLASSES and counts.size > 0.5 * len(y_str):
            raise TrainingError(f"The target has {counts.size} distinct values — it looks continuous. Use regression.")
        rare = counts[counts < 2]
        if len(rare):
            if prep["drop_rare_classes"]:
                keep = ~y_str.isin(rare.index)
                n_rare_dropped = int((~keep).sum())
                data, y_str = data[keep], y_str[keep]
                r.warn("prepare", f"Dropped {n_rare_dropped} rows from {len(rare)} classes with a single example.")
                if y_str.nunique() < 2:
                    raise TrainingError("Fewer than two classes remain after dropping rare classes.")
            else:
                raise TrainingError(
                    f"{len(rare)} classes have only one example ({', '.join(map(str, rare.index[:5]))}). "
                    "They cannot appear in both training and test data. Enable 'Drop classes with a single example' "
                    "or remove them in the Data Explorer.")
        encoder = LabelEncoder().fit(y_str)
        classes = encoder.classes_.tolist()
        y = pd.Series(encoder.transform(y_str), index=data.index)
    else:
        y = pd.to_numeric(y_raw, errors="coerce").astype(float)
        if y.isna().any():
            raise TrainingError("The regression target contains non-numeric values.")
        if y.nunique() < 2:
            raise TrainingError("The regression target is constant.")
    X = data[feature_cols]
    n_used = len(X)
    if n_used < MIN_ROWS:
        raise TrainingError(f"Only {n_used} usable rows remain; at least {MIN_ROWS} are needed for a held-out evaluation.")
    r.timing["prepare"] = time.perf_counter() - t0
    r.stage("prepare", f"{n_used:,} rows · {len(feature_cols)} features · task: {task}")
    r.check()

    # ---------------------------------------------------------- split
    t0 = time.perf_counter()
    test_size = split_cfg["test_size"]
    n_test = math.ceil(test_size * n_used)
    n_train = n_used - n_test
    stratified = False
    if task == "classification":
        n_classes = len(classes)
        if split_cfg["stratify"]:
            if n_test < n_classes or n_train < n_classes:
                raise TrainingError(
                    f"A stratified split needs at least {n_classes} rows in both train and test "
                    f"(currently {n_train} / {n_test}). Increase the test size or use more data.")
            stratified = True
        else:
            r.warn("split", "Stratification disabled: some classes may be missing from the training or test split.")
    if n_test < 3 or n_train < 5:
        raise TrainingError(f"Split leaves {n_train} training and {n_test} test rows; use more data or a different test size.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=rs, shuffle=True, stratify=y if stratified else None)
    test_idx = sorted(int(i) for i in X_test.index)
    split_id = hashlib.sha1(
        f"{dataset_meta.get('sha256')}|{target}|{','.join(map(str, test_idx))}".encode()).hexdigest()[:16]
    r.timing["split"] = time.perf_counter() - t0
    r.stage("split", f"Train {len(X_train):,} rows · test {len(X_test):,} rows"
                     f"{' · stratified by class' if stratified else ''} · seed {rs}")
    r.check()

    estimator = alg.build(resolved["hyperparameters"], rs)
    pipeline = build_pipeline(estimator, features, prep, alg.scale_sensitive, task)

    # ---------------------------------------------------------- cross-validation
    t0 = time.perf_counter()
    cv_result = None
    k = split_cfg["cv_folds"]
    cv = None
    if k:
        if task == "classification":
            min_count = int(pd.Series(y_train).value_counts().min())
            if min_count < k:
                if min_count >= 2:
                    r.warn("cv", f"Reduced CV folds from {k} to {min_count}: the rarest class has {min_count} training rows.")
                    k = min_count
                else:
                    r.warn("cv", "Skipped cross-validation: a class has fewer than 2 training rows.")
                    k = 0
            if k:
                cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=rs)
        else:
            k = min(k, len(X_train))
            cv = KFold(n_splits=k, shuffle=True, random_state=rs)
    if cv is not None:
        folds = []
        for i, (tr, va) in enumerate(cv.split(X_train, y_train), start=1):
            r.check()
            model = clone(pipeline).fit(X_train.iloc[tr], y_train.iloc[tr])
            pred = model.predict(X_train.iloc[va])
            yv = y_train.iloc[va]
            if task == "classification":
                m = {"accuracy": accuracy_score(yv, pred), "balanced_accuracy": balanced_accuracy_score(yv, pred),
                     "f1_macro": f1_score(yv, pred, average="macro", zero_division=0)}
            else:
                mse = mean_squared_error(yv, pred)
                m = {"r2": r2_score(yv, pred), "rmse": math.sqrt(mse), "mae": mean_absolute_error(yv, pred)}
            folds.append({"fold": i, "n_train": len(tr), "n_val": len(va), **m})
            r.stage("cv", f"Fold {i}/{k} · " + " · ".join(f"{kk} {vv:.3f}" for kk, vv in m.items()),
                    progress={"current": i, "total": k})
        names = [n for n in folds[0] if n not in ("fold", "n_train", "n_val")]
        cv_result = {
            "folds": folds, "k": k,
            "mean": {n: float(np.mean([f[n] for f in folds])) for n in names},
            "std": {n: float(np.std([f[n] for f in folds], ddof=1)) if k > 1 else 0.0 for n in names},
            "scheme": f"{'Stratified ' if task == 'classification' else ''}{k}-fold on the training split (shuffled, seed {rs})",
        }
    else:
        r.stage("cv", "Cross-validation disabled")
    r.timing["cv"] = time.perf_counter() - t0
    r.check()

    # ---------------------------------------------------------- fit
    t0 = time.perf_counter()
    r.stage("fit", f"Fitting {alg.name} on {len(X_train):,} training rows")
    pipeline.fit(X_train, y_train)
    r.timing["fit"] = time.perf_counter() - t0
    r.stage("fit", f"Fitted in {r.timing['fit']:.2f}s")
    r.check()

    # ---------------------------------------------------------- evaluate
    t0 = time.perf_counter()
    r.stage("evaluate", f"Scoring {len(X_test):,} held-out rows")
    tp = time.perf_counter()
    y_pred_test = pipeline.predict(X_test)
    predict_seconds = time.perf_counter() - tp
    y_pred_train = pipeline.predict(X_train)
    detail: Dict[str, Any] = {}
    if task == "classification":
        test_m = classification_metrics(y_test, y_pred_test)
        train_m = classification_metrics(y_train, y_pred_train)
        dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
        baseline = classification_metrics(y_test, dummy.predict(X_test))
        baseline["strategy"] = "Always predict the most frequent training class"
        labels = list(range(len(classes)))
        cm = confusion_matrix(y_test, y_pred_test, labels=labels)
        p, rc, f, s = precision_recall_fscore_support(y_test, y_pred_test, labels=labels, zero_division=0)
        per_class = [{"label": classes[i], "precision": p[i], "recall": rc[i], "f1": f[i], "support": int(s[i])}
                     for i in labels]
        roc = {"auc": None, "note": None, "curves": []}
        kind, scores = _scores(pipeline, X_test)
        present = np.unique(y_test)
        model_classes = getattr(pipeline, "classes_", np.arange(len(classes)))
        if kind is None:
            roc["note"] = "The model exposes neither probabilities nor decision scores."
        elif len(present) < len(classes):
            roc["note"] = "Not computed: some classes are absent from the test split."
        elif len(model_classes) != len(classes):
            roc["note"] = "Not computed: some classes were absent from the training split."
        elif len(classes) == 2:
            s1 = scores[:, 1] if (kind == "proba" and scores.ndim == 2) else scores
            roc["auc"] = roc_auc_score(y_test, s1)
            fpr, tpr, thr = roc_curve(y_test, s1)
            roc["curves"] = [{"label": classes[1], "points": _downsample_curve(fpr, tpr, thr)}]
            roc["score_type"] = "probability" if kind == "proba" else "decision function"
        elif kind == "proba":
            roc["auc"] = roc_auc_score(y_test, scores, multi_class="ovr", average="macro", labels=labels)
            roc["average"] = "macro one-vs-rest"
            for i in labels[:8]:
                fpr, tpr, thr = roc_curve((np.asarray(y_test) == i).astype(int), scores[:, i])
                roc["curves"].append({"label": classes[i], "points": _downsample_curve(fpr, tpr, thr)})
            roc["score_type"] = "probability"
        else:
            roc["note"] = "Multiclass ROC-AUC needs class probabilities; this model only provides decision scores."
        detail["classification"] = {"labels": classes, "confusion_matrix": cm.tolist(), "per_class": per_class, "roc": roc,
                                    "test_class_counts": {classes[i]: int(c) for i, c in zip(*np.unique(y_test, return_counts=True))},
                                    "train_class_counts": {classes[i]: int(c) for i, c in zip(*np.unique(y_train, return_counts=True))}}
        # misclassified examples for inspection
        conf = None
        if kind == "proba":
            conf = scores.max(axis=1)
        wrong = np.flatnonzero(np.asarray(y_test) != y_pred_test)
        show_cols = feature_cols[:8]
        errors = []
        for j in wrong[:50]:
            row = X_test.iloc[j]
            errors.append({"row": int(X_test.index[j]), "actual": classes[int(y_test.iloc[j])],
                           "predicted": classes[int(y_pred_test[j])],
                           "confidence": float(conf[j]) if conf is not None else None,
                           "features": {c: row[c] for c in show_cols}})
        detail["errors"] = {"kind": "misclassified", "total": int(wrong.size), "rows": errors, "columns": show_cols}
    else:
        test_m = regression_metrics(y_test, y_pred_test)
        train_m = regression_metrics(y_train, y_pred_train)
        dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
        baseline = regression_metrics(y_test, dummy.predict(X_test))
        baseline["strategy"] = "Always predict the training-set mean"
        residuals = np.asarray(y_test) - y_pred_test
        n_pts = len(residuals)
        idx = np.arange(n_pts) if n_pts <= 600 else np.unique(np.linspace(0, n_pts - 1, 600).astype(int))
        counts, edges = np.histogram(residuals, bins=min(30, max(5, int(np.sqrt(n_pts)))))
        detail["regression"] = {
            "points": [{"actual": float(np.asarray(y_test)[i]), "predicted": float(y_pred_test[i]),
                        "residual": float(residuals[i])} for i in idx],
            "points_shown": int(len(idx)), "points_total": int(n_pts),
            "residual_hist": {"edges": edges.tolist(), "counts": counts.tolist()},
            "residual_mean": float(residuals.mean()), "residual_std": float(residuals.std(ddof=1)) if n_pts > 1 else 0.0,
        }
        order = np.argsort(-np.abs(residuals))[:30]
        show_cols = feature_cols[:8]
        detail["errors"] = {"kind": "largest_residuals", "total": int(n_pts), "columns": show_cols, "rows": [
            {"row": int(X_test.index[j]), "actual": float(np.asarray(y_test)[j]), "predicted": float(y_pred_test[j]),
             "residual": float(residuals[j]), "features": {c: X_test.iloc[j][c] for c in show_cols}} for j in order]}
    r.timing["evaluate"] = time.perf_counter() - t0
    r.stage("evaluate", "Test metrics computed")
    r.check()

    # ---------------------------------------------------------- explain
    t0 = time.perf_counter()
    importance: Dict[str, Any] = {}
    try:
        names_out = pipeline[:-1].get_feature_names_out()
        est = pipeline[-1]
        native = None
        if hasattr(est, "feature_importances_"):
            native = np.asarray(est.feature_importances_, dtype=float)
            nkind = "impurity-based importance (training split)"
        elif hasattr(est, "coef_"):
            coef = np.asarray(est.coef_, dtype=float)
            native = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
            nkind = "absolute coefficient" + (" (features standardised)" if resolve_scaling(prep, alg.scale_sensitive) == "standard" else " (features not standardised — magnitudes not comparable)")
        if native is not None and len(native) == len(names_out):
            order = np.argsort(-native)[:30]
            importance["native"] = {"method": nkind, "features": [str(names_out[i]) for i in order],
                                    "values": [float(native[i]) for i in order]}
        encoded_features = int(len(names_out))
    except Exception:  # pragma: no cover - defensive; some estimators lack names
        encoded_features = None
    if options["permutation_importance"]:
        n_eval = min(len(X_test), 2000)
        Xp = X_test if n_eval == len(X_test) else X_test.sample(n_eval, random_state=rs)
        yp = y_test.loc[Xp.index]
        repeats = 5 if len(feature_cols) * n_eval < 400_000 else 3
        r.stage("explain", f"Permutation importance · {len(feature_cols)} features × {repeats} repeats on test rows")
        scoring = "accuracy" if task == "classification" else "r2"
        pi = permutation_importance(pipeline, Xp, yp, n_repeats=repeats, random_state=rs, scoring=scoring, n_jobs=1)
        order = np.argsort(-pi.importances_mean)
        importance["permutation"] = {
            "method": f"Permutation importance on the held-out test set (drop in {scoring}, {repeats} repeats)",
            "features": [feature_cols[i] for i in order],
            "values": [float(pi.importances_mean[i]) for i in order],
            "std": [float(pi.importances_std[i]) for i in order],
        }
    r.check()
    lc = None
    if options["learning_curve"] and cv is not None:
        r.stage("explain", "Computing learning curve on the training split")
        scoring = "accuracy" if task == "classification" else "r2"
        sizes, tr_scores, va_scores = learning_curve(
            pipeline, X_train, y_train, cv=cv, scoring=scoring, train_sizes=np.linspace(0.2, 1.0, 5),
            shuffle=True, random_state=rs, n_jobs=1)
        lc = {"metric": scoring, "train_sizes": sizes.tolist(),
              "train_mean": tr_scores.mean(axis=1).tolist(), "train_std": tr_scores.std(axis=1).tolist(),
              "val_mean": va_scores.mean(axis=1).tolist(), "val_std": va_scores.std(axis=1).tolist()}
    elif options["learning_curve"]:
        r.warn("explain", "Learning curve skipped: it requires cross-validation to be enabled.")
    r.timing["explain"] = time.perf_counter() - t0

    est = pipeline[-1]
    model_info: Dict[str, Any] = {"estimator": type(est).__name__, "encoded_features": encoded_features}
    for attr in ("n_estimators", "n_iter_", "n_features_in_"):
        if hasattr(est, attr):
            val = getattr(est, attr)
            if val is None:
                continue
            model_info[attr.rstrip("_")] = int(np.max(val)) if np.ndim(val) else int(val)
    if hasattr(est, "get_depth"):
        model_info["tree_depth"] = int(est.get_depth())
        model_info["tree_leaves"] = int(est.get_n_leaves())
    if hasattr(est, "estimators_") and hasattr(est.estimators_, "__len__") and len(est.estimators_) and hasattr(est.estimators_[0], "get_depth"):
        depths = [e.get_depth() for e in est.estimators_]
        model_info["mean_tree_depth"] = float(np.mean(depths))
    if hasattr(est, "n_support_"):
        model_info["support_vectors"] = int(np.sum(est.n_support_))
    if hasattr(est, "coef_"):
        model_info["nonzero_coefficients"] = int(np.count_nonzero(est.coef_))

    total = time.perf_counter() - t_total
    primary = "accuracy" if task == "classification" else "r2"
    results = {
        "version": 2,
        "task": task,
        "target": target,
        "algorithm": {"id": alg.id, "name": alg.name, "family": alg.family, "hyperparameters": resolved["hyperparameters"]},
        "data": {
            "dataset_id": dataset_meta.get("id"), "dataset_name": dataset_meta.get("name"),
            "n_raw": n_raw, "dropped_missing_target": missing_target, "dropped_duplicates": n_dups,
            "dropped_rare_classes": n_rare_dropped, "n_used": n_used,
            "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "classes": classes,
            "features": {k: features[k] for k in ("numeric", "categorical", "excluded", "skipped")},
            "encoded_features": encoded_features,
        },
        "preprocessing": {**prep, "scaling_resolved": resolve_scaling(prep, alg.scale_sensitive)},
        "pipeline_stages": describe_pipeline(features, prep, alg.scale_sensitive),
        "split": {"strategy": "holdout", "test_size": test_size, "random_state": rs, "stratified": stratified,
                  "shuffle": True, "split_id": split_id, "cv_folds": k if cv is not None else 0},
        "primary_metric": primary,
        "metrics": {"test": test_m, "train": train_m, "baseline": baseline, "cv": cv_result},
        **detail,
        "importance": importance,
        "learning_curve": lc,
        "model_info": model_info,
        "timing": {**r.timing, "predict_test": predict_seconds, "total": total},
        "warnings": r.warnings,
        "reproducibility": {
            "dataset_sha256": dataset_meta.get("sha256"),
            "random_state": rs,
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    bundle = {"pipeline": pipeline, "classes": classes, "target": target, "task": task,
              "feature_columns": feature_cols, "numeric": features["numeric"], "categorical": features["categorical"],
              "algorithm_id": alg.id, "sklearn_version": sklearn.__version__}
    return {"results": clean_json(results), "bundle": bundle, "summary": legacy_metrics(results)}


def legacy_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Flat summary stored in the legacy `metrics` column (and used by list views)."""
    task = results["task"]
    test = results["metrics"]["test"]
    imp = results["importance"].get("permutation") or results["importance"].get("native") or {}
    out = {
        "task_type": task,
        "primary_metric": results["primary_metric"],
        "validation_strategy": f"Hold-out {results['split']['test_size']:.0%} test split"
                               + (f" + {results['split']['cv_folds']}-fold CV on training split" if results["split"]["cv_folds"] else ""),
        "n_samples": results["data"]["n_used"],
        "n_train": results["data"]["n_train"],
        "n_test": results["data"]["n_test"],
        "n_features": len(results["data"]["features"]["numeric"]) + len(results["data"]["features"]["categorical"]),
        "target_column": results["target"],
        "algorithm_used": results["algorithm"]["name"],
        "split_id": results["split"]["split_id"],
        "feature_importance": {"features": imp.get("features", [])[:15], "importances": imp.get("values", [])[:15]},
    }
    if task == "classification":
        out.update(accuracy=test["accuracy"], balanced_accuracy=test["balanced_accuracy"], f1_score=test["f1_weighted"],
                   f1_macro=test["f1_macro"], precision=test["precision_weighted"], recall=test["recall_weighted"],
                   roc_auc=results["classification"]["roc"]["auc"], train_accuracy=results["metrics"]["train"]["accuracy"])
    else:
        out.update(r2_score=test["r2"], rmse=test["rmse"], mse=test["mse"], mae=test["mae"],
                   train_r2=results["metrics"]["train"]["r2"])
    cv = results["metrics"]["cv"]
    if cv:
        key = "accuracy" if task == "classification" else "r2"
        out["cv_mean"], out["cv_std"] = cv["mean"][key], cv["std"][key]
    return clean_json(out)
