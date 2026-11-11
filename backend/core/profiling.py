"""Deterministic dataset profiling and data-quality inspection.

Everything here is calculated from the data; no LLM is involved.
Expensive analyses run on a fixed-seed sample when the dataset is large and
the response says so (``sampled``).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .datasets import (
    DEFAULT_EXCLUDED_KINDS,
    DatasetError,
    column_kinds,
    numeric_coercible_share,
)
from .util import clean_json

SAMPLE_ROWS = 100_000
MAX_CORR_COLUMNS = 40
HIGH_CORR = 0.95


def _sample(df: pd.DataFrame, n: int = SAMPLE_ROWS) -> tuple[pd.DataFrame, bool]:
    if len(df) > n:
        return df.sample(n, random_state=0), True
    return df, False


# ------------------------------------------------------------- overview
def overview(df: pd.DataFrame) -> Dict[str, Any]:
    kinds = column_kinds(df)
    n_rows, n_cols = df.shape
    missing = df.isna().sum()
    columns = []
    for c in df.columns:
        s = df[c]
        non_null = s.dropna()
        columns.append({
            "name": c,
            "dtype": str(s.dtype),
            "kind": kinds[c]["kind"],
            "kind_reason": kinds[c]["reason"],
            "missing": int(missing[c]),
            "missing_pct": float(missing[c] / n_rows) if n_rows else 0.0,
            "unique": int(non_null.nunique()),
            "sample": [clean_json(v) for v in non_null.head(3).tolist()],
        })
    kind_counts: Dict[str, int] = {}
    for k in kinds.values():
        kind_counts[k["kind"]] = kind_counts.get(k["kind"], 0) + 1
    total_cells = n_rows * n_cols
    return clean_json({
        "rows": n_rows,
        "columns": n_cols,
        "missing_cells": int(missing.sum()),
        "missing_pct": float(missing.sum() / total_cells) if total_cells else 0.0,
        "rows_with_missing": int(df.isna().any(axis=1).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
        "kind_counts": kind_counts,
        "numeric_columns": [c for c, k in kinds.items() if k["kind"] == "numeric"],
        "categorical_columns": [c for c, k in kinds.items() if k["kind"] in ("categorical", "boolean")],
        "column_info": columns,
    })


# ---------------------------------------------------------- column detail
def _histogram(values: np.ndarray, max_bins: int = 40) -> Dict[str, Any]:
    if values.size == 0:
        return {"edges": [], "counts": []}
    lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        return {"edges": [lo, hi], "counts": [int(values.size)]}
    edges = np.histogram_bin_edges(values, bins="auto")
    if len(edges) - 1 > max_bins:
        edges = np.linspace(lo, hi, max_bins + 1)
    counts, edges = np.histogram(values, bins=edges)
    return {"edges": edges.tolist(), "counts": counts.tolist()}


def _evenly(values: np.ndarray, k: int) -> list:
    if values.size <= k:
        return values.tolist()
    return values[np.linspace(0, values.size - 1, k).astype(int)].tolist()


def numeric_summary(s: pd.Series) -> Dict[str, Any]:
    v = pd.to_numeric(s, errors="coerce").dropna().astype(float).to_numpy()
    if v.size == 0:
        return {"count": 0}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_mask = (v < lo_fence) | (v > hi_fence)
    inside = v[~outlier_mask]
    return {
        "count": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "min": float(v.min()),
        "q1": float(q1),
        "median": float(med),
        "q3": float(q3),
        "max": float(v.max()),
        "skew": float(pd.Series(v).skew()) if v.size > 2 else 0.0,
        "zeros": int((v == 0).sum()),
        "outliers": int(outlier_mask.sum()),
        "outlier_pct": float(outlier_mask.mean()),
        "box": {
            "whisker_low": float(inside.min()) if inside.size else float(q1),
            "q1": float(q1), "median": float(med), "q3": float(q3),
            "whisker_high": float(inside.max()) if inside.size else float(q3),
            "outlier_sample": _evenly(np.sort(v[outlier_mask]), 60),
        },
        "histogram": _histogram(v),
    }


def categorical_summary(s: pd.Series, top: int = 20) -> Dict[str, Any]:
    non_null = s.dropna().astype(str)
    counts = non_null.value_counts()
    top_counts = counts.head(top)
    other = int(counts.iloc[top:].sum()) if len(counts) > top else 0
    return {
        "count": int(non_null.size),
        "unique": int(counts.size),
        "cardinality_ratio": float(counts.size / non_null.size) if non_null.size else 0.0,
        "mode": str(counts.index[0]) if counts.size else None,
        "mode_freq": int(counts.iloc[0]) if counts.size else 0,
        "top_values": [{"value": str(k), "count": int(v)} for k, v in top_counts.items()],
        "other_count": other,
    }


def column_detail(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    if column not in df.columns:
        raise DatasetError(f"Unknown column '{column}'.")
    s = df[column]
    kind = column_kinds(df[[column]]).get(column, {"kind": "categorical", "reason": ""})
    out: Dict[str, Any] = {
        "name": column, "dtype": str(s.dtype), "kind": kind["kind"], "kind_reason": kind["reason"],
        "missing": int(s.isna().sum()), "missing_pct": float(s.isna().mean()),
        "unique": int(s.nunique(dropna=True)),
    }
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        out["numeric"] = numeric_summary(s)
        if out["unique"] <= 30:
            out["categorical"] = categorical_summary(s, top=30)
    else:
        out["categorical"] = categorical_summary(s)
        if kind["kind"] == "categorical" and numeric_coercible_share(s) > 0.95:
            out["numeric"] = numeric_summary(s.astype(str).str.replace(",", "", regex=False))
    return clean_json(out)


# ------------------------------------------------------------ correlations
def correlations(df: pd.DataFrame) -> Dict[str, Any]:
    kinds = column_kinds(df)
    num_cols = [c for c, k in kinds.items() if k["kind"] == "numeric"]
    sampled_df, sampled = _sample(df)
    truncated = False
    if len(num_cols) > MAX_CORR_COLUMNS:
        variances = sampled_df[num_cols].var().sort_values(ascending=False)
        num_cols = variances.index[:MAX_CORR_COLUMNS].tolist()
        truncated = True
    if len(num_cols) < 2:
        return {"columns": num_cols, "matrix": [], "high_pairs": [], "sampled": sampled, "truncated": truncated}
    corr = sampled_df[num_cols].astype(float).corr(method="pearson", min_periods=3)
    pairs = []
    for i, a in enumerate(num_cols):
        for b in num_cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= HIGH_CORR:
                pairs.append({"a": a, "b": b, "r": float(r)})
    pairs.sort(key=lambda p: -abs(p["r"]))
    return clean_json({
        "method": "pearson",
        "columns": num_cols,
        "matrix": corr.to_numpy().tolist(),
        "high_pairs": pairs,
        "sampled": sampled,
        "truncated": truncated,
    })


# ---------------------------------------------------------- missing values
def missing_pattern(df: pd.DataFrame, max_rows: int = 120) -> Dict[str, Any]:
    miss = df.isna()
    cols = [c for c in df.columns if miss[c].any()]
    per_row = miss.sum(axis=1)
    dist = per_row.value_counts().sort_index()
    if cols:
        step = max(1, len(df) // max_rows)
        idx = np.arange(0, len(df), step)[:max_rows]
        grid = miss.iloc[idx][cols].to_numpy().astype(int).tolist()
        row_index = idx.tolist()
    else:
        grid, row_index = [], []
    return clean_json({
        "columns": cols,
        "per_column": {c: int(miss[c].sum()) for c in cols},
        "rows_by_missing_count": [{"missing": int(k), "rows": int(v)} for k, v in dist.items()],
        "grid": grid,
        "grid_row_index": row_index,
    })


# ---------------------------------------------------------- scatter sample
def scatter(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None, limit: int = 1500) -> Dict[str, Any]:
    for c in (x, y, color):
        if c and c not in df.columns:
            raise DatasetError(f"Unknown column '{c}'.")
    cols = [x, y] + ([color] if color else [])
    sub = df[cols].copy()
    sub[x] = pd.to_numeric(sub[x], errors="coerce")
    sub[y] = pd.to_numeric(sub[y], errors="coerce")
    sub = sub.dropna(subset=[x, y])
    total = len(sub)
    if total > limit:
        sub = sub.sample(limit, random_state=0)
    groups = None
    if color:
        top = df[color].astype(str).value_counts().head(3).index.tolist()  # 3 hues validate all-pairs
        sub[color] = sub[color].astype(str).where(sub[color].astype(str).isin(top), "Other")
        groups = top + (["Other"] if (sub[color] == "Other").any() else [])
    points = [
        {"x": r[0], "y": r[1], **({"g": r[2]} if color else {})}
        for r in sub[cols].itertuples(index=False, name=None)
    ]
    r = float(sub[x].corr(sub[y])) if len(sub) > 2 else None
    return clean_json({"x": x, "y": y, "color": color, "groups": groups, "points": points,
                       "total": total, "shown": len(points), "pearson_r": r})


# ----------------------------------------------------------- quality checks
_SEV_ORDER = {"critical": 0, "warning": 1, "info": 2}


def quality_findings(df: pd.DataFrame, target: Optional[str] = None) -> List[Dict[str, Any]]:
    n = len(df)
    findings: List[Dict[str, Any]] = []
    kinds = column_kinds(df)

    def add(fid, severity, title, detail, columns=None, action=None):
        findings.append({"id": fid, "severity": severity, "title": title, "detail": detail,
                         "columns": columns or [], "action": action})

    if n < 30:
        add("small_dataset", "warning", f"Only {n} rows",
            "Evaluation on very small datasets is noisy. Treat any test score as a rough estimate and prefer cross-validation.")

    dups = int(df.duplicated().sum())
    if dups:
        add("duplicates", "warning", f"{dups} duplicate rows ({dups / n:.1%})",
            "Identical rows can land in both the training and test split, which inflates test scores. "
            "Model Studio drops exact duplicates before splitting by default.",
            action={"type": "drop_duplicates", "params": {}, "label": "Drop duplicate rows"})
    else:
        id_cols = [c for c, k in kinds.items() if k["kind"] == "identifier" and c != target]
        rest = [c for c in df.columns if c not in id_cols]
        if id_cols and rest:
            near = int(df.duplicated(subset=rest).sum())
            if near:
                add("duplicates_ignoring_ids", "warning",
                    f"{near} rows duplicate other rows apart from {', '.join(id_cols)}",
                    "These rows are identical in every other column, so they can sit in both the training and test "
                    "split. Model Studio drops them before splitting by default (identifiers are excluded from training).",
                    id_cols, {"type": "drop_duplicates", "params": {"subset": rest},
                              "label": f"Drop duplicates (ignoring {', '.join(id_cols)})"})

    for c, k in kinds.items():
        if c == target:
            continue
        if k["kind"] == "empty":
            add(f"empty:{c}", "warning", f"'{c}' is entirely empty", "The column carries no information.",
                [c], {"type": "drop_columns", "params": {"columns": [c]}, "label": f"Drop '{c}'"})
        elif k["kind"] == "constant":
            add(f"constant:{c}", "warning", f"'{c}' is constant", "A single value cannot help a model and is excluded from training by default.",
                [c], {"type": "drop_columns", "params": {"columns": [c]}, "label": f"Drop '{c}'"})
        elif k["kind"] == "identifier":
            add(f"identifier:{c}", "warning", f"'{c}' looks like an identifier",
                f"{k['reason']} Identifiers rarely generalise and can leak ordering information "
                "(e.g. when rows are sorted by class). It is excluded from training by default.",
                [c], {"type": "drop_columns", "params": {"columns": [c]}, "label": f"Drop '{c}'"})
        elif k["kind"] == "text":
            add(f"text:{c}", "info", f"'{c}' is free text",
                "Free-text columns are not used by the tabular pipeline and are excluded by default.", [c])
        elif k["kind"] == "datetime":
            add(f"datetime:{c}", "info", f"'{c}' contains dates",
                "Date columns are excluded by default. Derive numeric features (year, month, elapsed days) if they matter.", [c])

    for c, k in kinds.items():
        if k["kind"] == "numeric" and pd.api.types.is_float_dtype(df[c]):
            n_inf = int(np.isinf(df[c].to_numpy(dtype=float)).sum())
            if n_inf:
                add(f"infinite:{c}", "warning", f"'{c}' contains {n_inf} infinite values",
                    "Infinite values are treated as missing and imputed during training. Check how they were produced "
                    "(e.g. division by zero).", [c])

    miss = df.isna().mean()
    for c, pct in miss.items():
        if c == target or pct == 0 or kinds[c]["kind"] == "empty":
            continue
        if pct >= 0.5:
            add(f"missing_high:{c}", "warning", f"'{c}' is {pct:.0%} missing",
                "Heavily imputed columns can mislead a model. Consider dropping it.",
                [c], {"type": "drop_columns", "params": {"columns": [c]}, "label": f"Drop '{c}'"})
        elif pct >= 0.05:
            add(f"missing:{c}", "info", f"'{c}' is {pct:.0%} missing",
                "Missing values are imputed inside the training pipeline (fitted on the training split only).", [c])

    for c, k in kinds.items():
        s = df[c]
        if k["kind"] == "categorical":
            share = numeric_coercible_share(s)
            if share > 0.95:
                add(f"numeric_text:{c}", "warning", f"'{c}' stores numbers as text",
                    f"{share:.0%} of values parse as numbers, so it is currently treated as categorical.",
                    [c], {"type": "cast_numeric", "params": {"columns": [c]}, "label": f"Convert '{c}' to numeric"})
                continue
            nun = s.nunique()
            if nun > 50 and nun > 0.3 * s.notna().sum():
                add(f"high_card:{c}", "warning", f"'{c}' has {nun} categories",
                    "High-cardinality categories expand into many one-hot columns; rare levels are grouped as 'infrequent' during training.", [c])
            types = s.dropna().map(type).value_counts()
            if len(types) > 1:
                add(f"mixed:{c}", "warning", f"'{c}' mixes value types",
                    f"Found {', '.join(t.__name__ for t in types.index)}. Check how the column was exported.", [c])

    sampled_df, _ = _sample(df)
    for c, k in kinds.items():
        if k["kind"] != "numeric" or c == target:
            continue
        v = sampled_df[c].dropna().astype(float)
        if len(v) < 8:
            continue
        q1, q3 = v.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        share = float(((v < q1 - 3 * iqr) | (v > q3 + 3 * iqr)).mean())
        if share >= 0.01:
            add(f"outliers:{c}", "info", f"'{c}' has extreme values ({share:.1%})",
                "Values beyond 3×IQR. Consider robust scaling or IQR clipping in Model Studio, and check whether they are data errors.", [c])

    corr = correlations(df)
    for p in corr["high_pairs"][:10]:
        if target in (p["a"], p["b"]):
            continue
        add(f"corr:{p['a']}:{p['b']}", "info", f"'{p['a']}' and '{p['b']}' are highly correlated (r = {p['r']:.2f})",
            "Redundant features can destabilise linear model coefficients and split importance between them.", [p["a"], p["b"]])

    if target:
        findings.extend(target_findings(df, target))

    findings.sort(key=lambda f: _SEV_ORDER[f["severity"]])
    return clean_json(findings)


# ------------------------------------------------------------ target analysis
CLASSIFICATION_MAX_CLASSES = 50


def detect_problem_type(y: pd.Series) -> Dict[str, Any]:
    """Deterministic classification/regression decision with its reasoning."""
    y = y.dropna()
    n_unique = int(y.nunique())
    if n_unique < 2:
        return {"problem_type": None, "ambiguous": False,
                "reason": "The target has fewer than two distinct values, so nothing can be learned."}
    if pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(y):
        if numeric_coercible_share(y) > 0.95 and n_unique > CLASSIFICATION_MAX_CLASSES:
            return {"problem_type": "regression", "ambiguous": True,
                    "reason": "Values are numbers stored as text with many distinct values; convert the column to numeric first."}
        return {"problem_type": "classification", "ambiguous": False,
                "reason": f"Non-numeric target with {n_unique} classes."}
    is_integral = bool(np.all(np.mod(y.astype(float), 1) == 0))
    if is_integral and n_unique <= 20 and n_unique <= max(2, 0.05 * len(y)):
        return {"problem_type": "classification", "ambiguous": n_unique > 2,
                "reason": f"Integer target with only {n_unique} distinct values; treated as class labels. "
                          "Switch to regression if the values are ordered quantities."}
    return {"problem_type": "regression", "ambiguous": is_integral and n_unique <= 50,
            "reason": f"Numeric target with {n_unique} distinct values."}


def target_findings(df: pd.DataFrame, target: str, problem_type: Optional[str] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    y = df[target]
    n_missing = int(y.isna().sum())
    if n_missing:
        out.append({"id": "target_missing", "severity": "warning",
                    "title": f"{n_missing} rows have no target value",
                    "detail": "These rows cannot be used for supervised training and are dropped before splitting.",
                    "columns": [target],
                    "action": {"type": "drop_missing_rows", "params": {"columns": [target]}, "label": "Drop rows without a target"}})
    ptype = problem_type or detect_problem_type(y)["problem_type"]
    yv = y.dropna()
    if ptype == "classification":
        counts = yv.astype(str).value_counts()
        if len(counts) < 2:
            out.append({"id": "target_single_class", "severity": "critical", "title": "Target has a single class",
                        "detail": "A classifier needs at least two classes.", "columns": [target], "action": None})
        else:
            ratio = counts.iloc[0] / counts.iloc[-1]
            if counts.iloc[-1] < 2:
                out.append({"id": "target_rare_class", "severity": "warning",
                            "title": f"{int((counts < 2).sum())} classes have a single example",
                            "detail": "Stratified splitting needs at least 2 examples per class; these classes cannot be evaluated reliably.",
                            "columns": [target], "action": None})
            if ratio >= 10:
                out.append({"id": "target_imbalance", "severity": "warning",
                            "title": f"Strong class imbalance ({ratio:.0f}:1)",
                            "detail": "Accuracy is misleading here. Compare balanced accuracy and macro F1, and consider class_weight='balanced'.",
                            "columns": [target], "action": None})
            elif ratio >= 3:
                out.append({"id": "target_imbalance", "severity": "info",
                            "title": f"Moderate class imbalance ({ratio:.1f}:1)",
                            "detail": "Check per-class recall, not just accuracy.", "columns": [target], "action": None})
            if len(counts) > CLASSIFICATION_MAX_CLASSES:
                out.append({"id": "target_many_classes", "severity": "warning",
                            "title": f"{len(counts)} classes",
                            "detail": "This many classes usually means the target is continuous or an identifier.",
                            "columns": [target], "action": None})
    elif ptype == "regression":
        yn = pd.to_numeric(yv, errors="coerce").dropna()
        if len(yn) > 3 and abs(float(yn.skew())) > 2:
            out.append({"id": "target_skew", "severity": "info", "title": f"Target is heavily skewed (skew {float(yn.skew()):.1f})",
                        "detail": "Errors on large values will dominate MSE/RMSE. A log transform of the target may help.",
                        "columns": [target], "action": None})
    out.extend(leakage_suspects(df, target, ptype))
    return out


def leakage_suspects(df: pd.DataFrame, target: str, ptype: Optional[str]) -> List[Dict[str, Any]]:
    """Flag features that almost perfectly determine the target (possible target leakage)."""
    out = []
    sub, _ = _sample(df.dropna(subset=[target]), 50_000)
    y = sub[target]
    kinds = column_kinds(sub)
    for c in sub.columns:
        if c == target or kinds[c]["kind"] in DEFAULT_EXCLUDED_KINDS:
            continue
        x = sub[c]
        suspicious, why = False, ""
        if ptype == "regression" and kinds[c]["kind"] == "numeric":
            yn = pd.to_numeric(y, errors="coerce")
            r = x.astype(float).corr(yn)
            if pd.notna(r) and abs(r) >= 0.98:
                suspicious, why = True, f"Pearson r with the target is {r:.3f}."
        elif ptype == "classification":
            nun = x.nunique()
            if 1 < nun < 0.5 * len(x):
                purity = sub.groupby(x.astype(str), observed=True)[target].nunique()
                if len(purity) and (purity == 1).all() and y.nunique() > 1 and nun >= y.nunique():
                    suspicious, why = True, "Every value of this feature maps to exactly one class."
        if suspicious:
            out.append({"id": f"leak:{c}", "severity": "warning", "title": f"Possible target leakage via '{c}'",
                        "detail": f"{why} Check that this information is available at prediction time; otherwise exclude it.",
                        "columns": [c], "action": {"type": "drop_columns", "params": {"columns": [c]}, "label": f"Drop '{c}'"}})
    return out


def target_analysis(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    if target not in df.columns:
        raise DatasetError(f"Unknown target column '{target}'.")
    y = df[target]
    detection = detect_problem_type(y)
    out: Dict[str, Any] = {
        "target": target,
        "n_rows": int(len(df)),
        "n_missing": int(y.isna().sum()),
        "n_unique": int(y.nunique()),
        **detection,
    }
    yv = y.dropna()
    if detection["problem_type"] == "classification" or y.nunique() <= 30:
        counts = yv.astype(str).value_counts()
        out["class_distribution"] = [{"label": str(k), "count": int(v)} for k, v in counts.head(50).items()]
        out["n_classes"] = int(counts.size)
        out["imbalance_ratio"] = float(counts.iloc[0] / counts.iloc[-1]) if counts.size > 1 else None
        out["min_class_count"] = int(counts.iloc[-1]) if counts.size else 0
    if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y):
        out["numeric"] = numeric_summary(y)
    out["findings"] = target_findings(df, target, detection["problem_type"])
    return clean_json(out)


def full_profile(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "overview": overview(df),
        "quality": quality_findings(df),
    }
