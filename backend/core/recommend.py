"""Deterministic algorithm compatibility and recommendation.

Scores are heuristic suitability ratings derived from dataset properties —
they are NOT predictions of model performance. The LLM may later *explain*
these results but never changes them.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from . import algorithms as algs
from .datasets import DEFAULT_EXCLUDED_KINDS, column_kinds
from .profiling import detect_problem_type


def dataset_facts(df: pd.DataFrame, target: str, task: str) -> Dict[str, Any]:
    kinds = column_kinds(df)
    features = [c for c in df.columns if c != target and kinds[c]["kind"] not in DEFAULT_EXCLUDED_KINDS]
    numeric = [c for c in features if kinds[c]["kind"] in ("numeric", "boolean")]
    categorical = [c for c in features if kinds[c]["kind"] == "categorical"]
    encoded_width = len(numeric) + sum(min(int(df[c].nunique()), 50) for c in categorical)
    y = df[target].dropna()
    facts: Dict[str, Any] = {
        "n_rows": int(y.size),
        "n_features": len(features),
        "n_numeric": len(numeric),
        "n_categorical": len(categorical),
        "encoded_width": encoded_width,
        "missing_share": float(df[features].isna().mean().mean()) if features else 0.0,
        "task": task,
    }
    if task == "classification":
        counts = y.astype(str).value_counts()
        facts["n_classes"] = int(counts.size)
        facts["imbalance_ratio"] = float(counts.iloc[0] / counts.iloc[-1]) if counts.size > 1 else None
    return facts


def recommend(df: pd.DataFrame, target: str, task: str | None = None) -> Dict[str, Any]:
    if task is None:
        task = detect_problem_type(df[target])["problem_type"]
    if task not in ("classification", "regression"):
        return {"task": task, "facts": {}, "algorithms": [], "note": "Target is not suitable for supervised learning."}
    f = dataset_facts(df, target, task)
    n, width = f["n_rows"], f["encoded_width"]
    imbalanced = (f.get("imbalance_ratio") or 1) >= 3
    results: List[Dict[str, Any]] = []

    for a in algs.ALGORITHMS:
        entry: Dict[str, Any] = {"id": a.id, "name": a.name, "family": a.family, "task": a.task}
        if a.task != task:
            continue
        if not a.available:
            entry.update(compatible=False, score=0, reasons=[], cautions=[a.unavailable_reason], cost="—")
            results.append(entry)
            continue
        score, reasons, cautions = 60, [], []
        fam = a.family
        cost = "low"

        if fam == "Linear":
            reasons.append("Fast, interpretable baseline — always worth training first for comparison.")
            score += 8
            if n < 200:
                score += 6
                reasons.append(f"With {n} rows, a low-variance model is less likely to overfit.")
            if width > 0 and n / max(width, 1) < 5 and a.id in ("ridge", "lasso", "logistic_regression"):
                score += 4
                reasons.append("Regularisation helps when features are numerous relative to rows.")
            if a.id == "linear_regression" and width > n / 3:
                score -= 10
                cautions.append("Unregularised OLS is unstable with this many features relative to rows; prefer Ridge.")
        if fam in ("Ensemble", "Tree"):
            reasons.append("Handles non-linear effects and mixed feature types without scaling.")
            score += 6
            if n >= 1000:
                score += 8
                reasons.append(f"{n:,} rows is enough for tree ensembles to learn interactions.")
            if n < 100 and fam == "Ensemble":
                score -= 6
                cautions.append(f"Only {n} rows: ensemble gains over simpler models are often within noise.")
            if fam == "Tree":
                score -= 4
                cautions.append("A single tree has high variance; compare with Random Forest.")
            if "hist_gradient" in a.id:
                cost = "low"
                if n >= 10000:
                    score += 10
                    reasons.append("Histogram boosting is designed for datasets of this size.")
                elif n < 500:
                    score -= 8
                    cautions.append("Designed for larger datasets; small data gains little.")
            elif fam == "Ensemble":
                cost = "medium" if n < 50000 else "high"
                if a.id.startswith("gradient_boosting") and n > 50000:
                    score -= 8
                    cautions.append("Classic gradient boosting is sequential and slow at this size; prefer Histogram Gradient Boosting.")
        if fam == "Kernel":
            cost = "low" if n < 5000 else "medium" if n < 20000 else "high"
            if n <= 5000:
                score += 4
                reasons.append("Kernel methods work well on small-to-medium datasets.")
            else:
                score -= 12 if n > 20000 else 6
                cautions.append(f"Training time grows roughly quadratically; {n:,} rows will be slow.")
        if fam == "Instance-based":
            cost = "low" if n < 20000 else "high"
            if width > 50:
                score -= 12
                cautions.append(f"Distances become less meaningful across ~{width} encoded features.")
            if n > 50000:
                score -= 8
                cautions.append("Prediction requires comparing against every training row.")
            if width <= 20 and n <= 20000:
                reasons.append("Low dimensionality suits distance-based methods.")
        if fam == "Probabilistic":
            reasons.append("Very fast; a useful sanity-check baseline.")
            if f["n_categorical"] > 0:
                score -= 6
                cautions.append("One-hot features violate the Gaussian assumption.")
            if n < 100:
                score += 4
        if a.scale_sensitive:
            reasons.append("Features will be standardised automatically (scale-sensitive model).")
        if task == "classification" and imbalanced:
            if any(p.name == "class_weight" for p in a.params):
                reasons.append(f"Supports class_weight='balanced' for the {f['imbalance_ratio']:.1f}:1 class imbalance.")
                score += 3
            else:
                cautions.append("No class weighting option; watch minority-class recall.")
        entry.update(
            compatible=True,
            score=int(max(0, min(100, score))),
            reasons=[r for r in reasons if r],
            cautions=cautions,
            cost=cost,
        )
        results.append(entry)

    results.sort(key=lambda r: (-r["compatible"], -r["score"], r["name"]))
    compatible = [r for r in results if r["compatible"]]
    baseline = next((r["id"] for r in compatible if r["family"] == "Linear"), None)
    return {
        "task": task,
        "facts": f,
        "algorithms": results,
        "suggested": [r["id"] for r in compatible[:3]],
        "baseline": baseline,
        "method": "Deterministic rules over dataset size, feature count/types, missingness and class balance. "
                  "Scores rank suitability, not expected accuracy — train to measure performance.",
    }
