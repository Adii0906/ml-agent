"""Experiment reports built only from stored experiment results.

Observations and limitations are produced by deterministic rules so every
statement can be traced to a number in the experiment record. An optional
AI narrative is attached separately and clearly labelled.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .llm import DISCLAIMER
from .util import utcnow

METRIC_LABELS = {
    "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "precision_macro": "Precision (macro)",
    "recall_macro": "Recall (macro)", "f1_macro": "F1 (macro)", "precision_weighted": "Precision (weighted)",
    "recall_weighted": "Recall (weighted)", "f1_weighted": "F1 (weighted)",
    "mae": "MAE", "mse": "MSE", "rmse": "RMSE", "r2": "R²", "median_ae": "Median AE", "max_error": "Max error", "mape": "MAPE",
}


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def observations(exp: Dict[str, Any]) -> List[Dict[str, str]]:
    res = exp["results"]
    task = res["task"]
    m = res["metrics"]
    test, train, base, cv = m["test"], m["train"], m["baseline"], m.get("cv")
    obs: List[Dict[str, str]] = []
    if task == "classification":
        gap = train["accuracy"] - test["accuracy"]
        obs.append({"kind": "baseline", "text":
                    f"Test accuracy {test['accuracy']:.3f} vs. {base['accuracy']:.3f} for always predicting the majority class."})
        if test["accuracy"] <= base["accuracy"] + 0.01:
            obs.append({"kind": "warning", "text": "The model is no better than the majority-class baseline."})
        if abs(test["accuracy"] - test["balanced_accuracy"]) > 0.05:
            obs.append({"kind": "warning", "text":
                        f"Balanced accuracy ({test['balanced_accuracy']:.3f}) differs from accuracy — performance is uneven across classes."})
        weakest = min(res["classification"]["per_class"], key=lambda c: c["recall"])
        obs.append({"kind": "info", "text": f"Lowest recall: class '{weakest['label']}' ({weakest['recall']:.3f}, support {weakest['support']})."})
        if gap > 0.1:
            obs.append({"kind": "warning", "text": f"Training accuracy exceeds test accuracy by {gap:.3f}: likely overfitting."})
        auc = res["classification"]["roc"].get("auc")
        if auc is not None:
            obs.append({"kind": "info", "text": f"ROC-AUC {auc:.3f} ({res['classification']['roc'].get('average', 'binary')})."})
    else:
        obs.append({"kind": "baseline", "text":
                    f"Test R² {_fmt(test['r2'], 3)}; predicting the training mean scores R² {_fmt(base['r2'], 3)} (RMSE {_fmt(base['rmse'], 3)})."})
        if test["r2"] is not None and test["r2"] < 0:
            obs.append({"kind": "warning", "text": "Negative R²: the model is worse than predicting the mean."})
        if train["r2"] is not None and test["r2"] is not None and train["r2"] - test["r2"] > 0.15:
            obs.append({"kind": "warning", "text": f"Training R² ({train['r2']:.3f}) far exceeds test R² ({test['r2']:.3f}): likely overfitting."})
        reg = res.get("regression") or {}
        if reg and reg.get("residual_std"):
            if abs(reg["residual_mean"]) > 0.25 * reg["residual_std"]:
                obs.append({"kind": "warning", "text": f"Residuals are biased (mean {reg['residual_mean']:.3f}); predictions are systematically off."})
    if cv:
        key = "accuracy" if task == "classification" else "r2"
        obs.append({"kind": "info", "text": f"Cross-validation on the training split: {key} {cv['mean'][key]:.3f} ± {cv['std'][key]:.3f} over {cv['k']} folds."})
        if cv["std"][key] > 0.05:
            obs.append({"kind": "warning", "text": "Fold-to-fold variation is high; the test score is an uncertain estimate."})
    if res["data"]["n_test"] < 50:
        obs.append({"kind": "warning", "text": f"The test set has only {res['data']['n_test']} rows; metrics have wide uncertainty."})
    imp = res["importance"].get("permutation")
    if imp and imp["features"]:
        obs.append({"kind": "info", "text": f"Most influential feature on held-out data: '{imp['features'][0]}' (permutation importance {imp['values'][0]:.3f})."})
    for w in res.get("warnings", []):
        obs.append({"kind": "warning", "text": w})
    return obs


def limitations(exp: Dict[str, Any]) -> List[str]:
    res = exp["results"]
    out = [
        "Metrics come from a single held-out split of this dataset; performance on new data drawn from a different distribution can be substantially worse.",
        "No hyperparameter search was performed beyond the values listed.",
        "Feature importance describes the model's behaviour on this data, not causal effects.",
    ]
    if res["data"]["features"]["excluded"]:
        out.append(f"Columns excluded from training: {', '.join(res['data']['features']['excluded'])}.")
    if res["algorithm"]["id"] == "svr":
        out.append("SVR is sensitive to the target's scale; the target was not rescaled.")
    if res["task"] == "classification" and res["classification"]["roc"].get("note"):
        out.append(f"ROC-AUC: {res['classification']['roc']['note']}")
    return out


def build(exp: Dict[str, Any], dataset: Optional[Dict[str, Any]], quality: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    if exp.get("status") != "completed" or not exp.get("results"):
        raise ValueError("Reports can only be generated for completed experiments.")
    res = exp["results"]
    obs = observations(exp)
    report = {
        "generated_at": utcnow(),
        "experiment": {"id": exp["id"], "name": exp["name"], "created_at": exp["created_at"],
                       "completed_at": exp.get("completed_at"), "training_time": exp.get("training_time")},
        "dataset": {
            "id": res["data"].get("dataset_id"), "name": (dataset or {}).get("name") or res["data"].get("dataset_name"),
            "filename": (dataset or {}).get("filename"), "rows": res["data"]["n_raw"], "columns": (dataset or {}).get("n_cols"),
            "sha256": res["reproducibility"]["dataset_sha256"], "still_available": dataset is not None,
        },
        "data_quality": [q for q in (quality or []) if q["severity"] in ("critical", "warning")][:12],
        "task": res["task"], "target": res["target"],
        "algorithm": res["algorithm"],
        "preprocessing": res["preprocessing"], "pipeline_stages": res["pipeline_stages"],
        "methodology": {
            "split": res["split"], "rows": {k: res["data"][k] for k in ("n_raw", "dropped_missing_target", "dropped_duplicates", "dropped_rare_classes", "n_used", "n_train", "n_test")},
            "cv": (res["metrics"]["cv"] or {}).get("scheme"),
            "baseline": res["metrics"]["baseline"].get("strategy"),
        },
        "metrics": res["metrics"],
        "primary_metric": res["primary_metric"],
        "classification": res.get("classification"),
        "regression": {k: v for k, v in (res.get("regression") or {}).items() if k != "points"} or None,
        "importance": res.get("importance"),
        "model_info": res.get("model_info"),
        "observations": obs,
        "limitations": limitations(exp),
        "reproducibility": res["reproducibility"],
        "disclaimer": DISCLAIMER,
        # legacy keys (old /report consumers)
        "summary": obs[0]["text"] if obs else "",
        "insights": [o["text"] for o in obs[1:6]],
    }
    return report


def to_markdown(report: Dict[str, Any], ai: Optional[Dict[str, Any]] = None) -> str:
    e, d, a = report["experiment"], report["dataset"], report["algorithm"]
    m = report["metrics"]
    lines = [f"# Experiment report — {e['name']}", "",
             f"Generated {report['generated_at']} by ML-Agent · experiment `{e['id']}`", "",
             "## Summary", "",
             f"- **Task:** {report['task']} · **Target:** `{report['target']}`",
             f"- **Algorithm:** {a['name']} (`{a['id']}`)",
             f"- **Dataset:** {d['name']} ({d['rows']} rows) · SHA-256 `{(d['sha256'] or '')[:16]}…`", ""]
    lines += ["## Evaluation methodology", ""]
    s = report["methodology"]["split"]
    rows = report["methodology"]["rows"]
    lines += [f"- Hold-out test split: {s['test_size']:.0%}, seed {s['random_state']}, stratified: {s['stratified']}",
              f"- Rows: {rows['n_raw']} raw → {rows['n_used']} used ({rows['dropped_missing_target']} missing target, "
              f"{rows['dropped_duplicates']} duplicates, {rows['dropped_rare_classes']} rare-class rows removed)",
              f"- Train {rows['n_train']} / test {rows['n_test']} rows",
              f"- Cross-validation: {report['methodology']['cv'] or 'disabled'}",
              f"- Baseline: {report['methodology']['baseline']}", ""]
    lines += ["## Metrics (calculated)", "", "| Metric | Test | Train | Baseline (test) |", "|---|---|---|---|"]
    for k, label in METRIC_LABELS.items():
        if k in m["test"]:
            lines.append(f"| {label} | {_fmt(m['test'][k])} | {_fmt(m['train'].get(k))} | {_fmt(m['baseline'].get(k))} |")
    if m.get("cv"):
        lines += ["", "Cross-validation (training split): " + ", ".join(
            f"{k} {v:.4f} ± {m['cv']['std'][k]:.4f}" for k, v in m["cv"]["mean"].items())]
    if report.get("classification"):
        c = report["classification"]
        lines += ["", "### Per-class metrics", "", "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
        for pc in c["per_class"]:
            lines.append(f"| {pc['label']} | {pc['precision']:.3f} | {pc['recall']:.3f} | {pc['f1']:.3f} | {pc['support']} |")
        lines += ["", "### Confusion matrix (rows = actual, columns = predicted)", "",
                  "| | " + " | ".join(c["labels"]) + " |", "|---" * (len(c["labels"]) + 1) + "|"]
        for lab, row in zip(c["labels"], c["confusion_matrix"]):
            lines.append(f"| **{lab}** | " + " | ".join(str(v) for v in row) + " |")
    imp = (report.get("importance") or {}).get("permutation")
    if imp:
        lines += ["", "## Feature importance", "", f"_{imp['method']}_", ""]
        lines += [f"{i + 1}. `{f}` — {v:.4f}" for i, (f, v) in enumerate(zip(imp["features"][:10], imp["values"][:10]))]
    lines += ["", "## Preprocessing pipeline", ""]
    lines += [f"{i + 1}. **{st['stage']}** ({st['fitted_on']}): {st['detail']}" for i, st in enumerate(report["pipeline_stages"])]
    hp_lines = [f"- `{k}` = `{v}`" for k, v in a["hyperparameters"].items()] or ["- (no tunable hyperparameters)"]
    lines += ["", "## Hyperparameters", ""] + hp_lines
    if report["data_quality"]:
        lines += ["", "## Data quality findings", ""] + [f"- **{q['severity']}** — {q['title']}" for q in report["data_quality"]]
    lines += ["", "## Observations (rule-based, from the numbers above)", ""] + [f"- {o['text']}" for o in report["observations"]]
    lines += ["", "## Known limitations", ""] + [f"- {x}" for x in report["limitations"]]
    r = report["reproducibility"]
    lines += ["", "## Reproducibility", "",
              f"- Dataset SHA-256: `{r['dataset_sha256']}`", f"- Random seed: {r['random_state']} · split id `{s['split_id']}`",
              f"- Python {r['python']}, scikit-learn {r['scikit_learn']}, pandas {r['pandas']}, numpy {r['numpy']}"]
    if ai:
        c = ai["content"]
        lines += ["", f"## AI-generated narrative ({ai['provider']} · {ai['model']})", "",
                  "> Generated by a language model from the facts above. It may contain errors and is not a verified calculation.", "",
                  c.get("summary", "")]
        if c.get("observations"):
            lines += [""] + [f"- {x}" for x in c["observations"]]
        if c.get("next_steps"):
            lines += ["", "**Suggested next steps (AI):**", ""] + [f"- {x}" for x in c["next_steps"]]
    lines += ["", "---", "", f"_{report['disclaimer']}_", ""]
    return "\n".join(lines)


def to_text(report: Dict[str, Any], ai: Optional[Dict[str, Any]] = None) -> str:
    """Plain-text export (keeps the original .txt download working)."""
    md = to_markdown(report, ai)
    out = []
    for line in md.splitlines():
        line = line.replace("**", "").replace("`", "")
        if line.startswith("# "):
            out += ["=" * 64, line[2:].upper(), "=" * 64]
        elif line.startswith("## "):
            out += ["", line[3:].upper(), "-" * 64]
        elif line.startswith("### "):
            out += ["", line[4:]]
        elif line.startswith("|---"):
            continue
        else:
            out.append(line)
    return "\n".join(out)
