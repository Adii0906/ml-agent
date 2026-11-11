"""Leak-free preprocessing pipelines.

Every transformation that learns from data (imputation statistics, scaling
parameters, IQR fences, one-hot categories, feature-selection scores) lives
inside a scikit-learn ``Pipeline`` so it is fitted on the training split only
and merely *applied* to validation/test data.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler

from mlagent_runtime import IQRClipper as _RuntimeIQRClipper
from mlagent_runtime import categorical_to_str, infinite_to_nan

from .datasets import DEFAULT_EXCLUDED_KINDS, column_kinds


class PipelineConfigError(ValueError):
    pass


DEFAULT_PREPROCESSING: Dict[str, Any] = {
    "numeric_imputation": "median",     # mean | median | most_frequent | zero
    "categorical_imputation": "constant",  # most_frequent | constant
    "encoding": "onehot",               # onehot | ordinal
    "scaling": "auto",                  # auto | standard | minmax | robust | none
    "outliers": "none",                 # none | clip_iqr
    "feature_selection": "none",        # none | variance | kbest
    "k_features": 10,
    "drop_duplicates": True,
    "drop_rare_classes": False,
    "excluded_columns": None,           # None = use defaults from type inference
}

_CHOICES = {
    "numeric_imputation": {"mean", "median", "most_frequent", "zero"},
    "categorical_imputation": {"most_frequent", "constant"},
    "encoding": {"onehot", "ordinal"},
    "scaling": {"auto", "standard", "minmax", "robust", "none"},
    "outliers": {"none", "clip_iqr"},
    "feature_selection": {"none", "variance", "kbest"},
}


# The custom transformers live in the standalone ``mlagent_runtime`` module so that saved
# artifacts can be loaded without the ML-Agent server. These aliases keep artifacts that were
# pickled before the move (which reference ``core.pipeline``) loadable.
IQRClipper = _RuntimeIQRClipper
_to_str = categorical_to_str


def validate_preprocessing(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = dict(DEFAULT_PREPROCESSING)
    raw = dict(raw or {})
    unknown = set(raw) - set(cfg)
    if unknown:
        raise PipelineConfigError(f"Unknown preprocessing options: {', '.join(sorted(unknown))}.")
    cfg.update(raw)
    for key, allowed in _CHOICES.items():
        if cfg[key] not in allowed:
            raise PipelineConfigError(f"'{key}' must be one of {sorted(allowed)}.")
    try:
        cfg["k_features"] = int(cfg["k_features"])
    except (TypeError, ValueError):
        raise PipelineConfigError("'k_features' must be an integer.") from None
    if cfg["k_features"] < 1:
        raise PipelineConfigError("'k_features' must be at least 1.")
    for key in ("drop_duplicates", "drop_rare_classes"):
        cfg[key] = bool(cfg[key])
    if cfg["excluded_columns"] is not None and not isinstance(cfg["excluded_columns"], list):
        raise PipelineConfigError("'excluded_columns' must be a list of column names.")
    return cfg


def select_features(df: pd.DataFrame, target: str, excluded: Optional[List[str]]) -> Dict[str, Any]:
    """Decide which columns feed the model and how each is treated."""
    kinds = column_kinds(df)
    if excluded is None:
        excluded = [c for c, k in kinds.items() if c != target and k["kind"] in DEFAULT_EXCLUDED_KINDS]
    excluded = [c for c in excluded if c in df.columns and c != target]
    numeric, categorical, skipped = [], [], []
    for c in df.columns:
        if c == target or c in excluded:
            continue
        kind = kinds[c]["kind"]
        if kind in ("numeric", "boolean") and pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
        elif kind in ("empty", "constant"):
            skipped.append({"column": c, "reason": kinds[c]["reason"]})
        elif kind in ("text", "datetime"):
            skipped.append({"column": c, "reason": f"{kind} columns are not supported by the tabular pipeline."})
        else:
            # categorical, boolean-as-text, or identifiers the user chose to include
            categorical.append(c)
    if not numeric and not categorical:
        raise PipelineConfigError("No usable feature columns remain after exclusions.")
    return {"numeric": numeric, "categorical": categorical, "excluded": excluded, "skipped": skipped, "kinds": kinds}


def resolve_scaling(cfg: Dict[str, Any], scale_sensitive: bool) -> str:
    if cfg["scaling"] != "auto":
        return cfg["scaling"]
    return "standard" if scale_sensitive else "none"


def build_pipeline(estimator, features: Dict[str, Any], cfg: Dict[str, Any], scale_sensitive: bool, task: str) -> Pipeline:
    scaling = resolve_scaling(cfg, scale_sensitive)
    transformers = []
    if features["numeric"]:
        if cfg["numeric_imputation"] == "zero":
            imputer = SimpleImputer(strategy="constant", fill_value=0)
        else:
            imputer = SimpleImputer(strategy=cfg["numeric_imputation"])
        steps = [("finite", FunctionTransformer(infinite_to_nan, feature_names_out="one-to-one")), ("impute", imputer)]
        if cfg["outliers"] == "clip_iqr":
            steps.append(("clip", IQRClipper(1.5)))
        scaler = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}.get(scaling)
        if scaler is not None:
            steps.append(("scale", scaler))
        transformers.append(("num", Pipeline(steps), features["numeric"]))
    if features["categorical"]:
        if cfg["categorical_imputation"] == "constant":
            cat_imputer = SimpleImputer(strategy="constant", fill_value="__missing__")
        else:
            cat_imputer = SimpleImputer(strategy="most_frequent")
        if cfg["encoding"] == "onehot":
            encoder = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=None,
                                    max_categories=50, sparse_output=False)
        else:
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        cat_steps = [("to_str", FunctionTransformer(categorical_to_str, feature_names_out="one-to-one")),
                     ("impute", cat_imputer), ("encode", encoder)]
        if cfg["encoding"] == "ordinal" and scaling != "none":
            cat_steps.append(("scale", StandardScaler()))
        transformers.append(("cat", Pipeline(cat_steps), features["categorical"]))

    preprocess = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)
    steps = [("preprocess", preprocess)]
    if cfg["feature_selection"] == "variance":
        steps.append(("select", VarianceThreshold(0.0)))
    elif cfg["feature_selection"] == "kbest":
        score_fn = f_classif if task == "classification" else f_regression
        steps.append(("select", SelectKBest(score_fn, k=cfg["k_features"])))
    steps.append(("model", estimator))
    return Pipeline(steps)


def describe_pipeline(features: Dict[str, Any], cfg: Dict[str, Any], scale_sensitive: bool) -> List[Dict[str, Any]]:
    """Human-readable, ordered description of exactly what the pipeline does."""
    scaling = resolve_scaling(cfg, scale_sensitive)
    stages: List[Dict[str, Any]] = []
    stages.append({"stage": "Row filtering", "fitted_on": "full dataset (no statistics learned)",
                   "detail": ("Drop rows with a missing target" +
                              ("; drop rows with identical features and target" if cfg["drop_duplicates"] else "") +
                              ("; drop classes with fewer than 2 rows" if cfg["drop_rare_classes"] else ""))})
    stages.append({"stage": "Train/test split", "fitted_on": "—", "detail": "Held-out test set is never used for fitting."})
    if features["numeric"]:
        imp = "constant 0" if cfg["numeric_imputation"] == "zero" else cfg["numeric_imputation"]
        num_detail = f"{len(features['numeric'])} columns · ±inf → missing · impute {imp}"
        if cfg["outliers"] == "clip_iqr":
            num_detail += " · clip to 1.5×IQR fences"
        num_detail += f" · scaling: {scaling}"
        stages.append({"stage": "Numeric features", "fitted_on": "training split", "detail": num_detail,
                       "columns": features["numeric"]})
    if features["categorical"]:
        imp = "'__missing__' category" if cfg["categorical_imputation"] == "constant" else "most frequent"
        enc = "one-hot (≤50 levels, rare levels grouped, unseen → ignored)" if cfg["encoding"] == "onehot" else "ordinal codes (unseen → -1)"
        stages.append({"stage": "Categorical features", "fitted_on": "training split",
                       "detail": f"{len(features['categorical'])} columns · impute {imp} · {enc}",
                       "columns": features["categorical"]})
    if cfg["feature_selection"] == "variance":
        stages.append({"stage": "Feature selection", "fitted_on": "training split", "detail": "Remove zero-variance encoded features"})
    elif cfg["feature_selection"] == "kbest":
        stages.append({"stage": "Feature selection", "fitted_on": "training split",
                       "detail": f"Keep the {cfg['k_features']} best features by univariate F-test"})
    stages.append({"stage": "Model", "fitted_on": "training split", "detail": "Estimator fitted on the transformed training data"})
    return stages
