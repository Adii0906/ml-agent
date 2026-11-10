"""Metric calculations and dataset profiling against hand-computed values."""
import numpy as np
import pandas as pd
import pytest

from core import profiling, recommend
from core.training import classification_metrics, regression_metrics, run_training, validate_config

from .conftest import make_classification, make_regression


def test_classification_metrics_match_hand_computation():
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    m = classification_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(4 / 6)
    # recall per class: 2/3, 1/2, 1/1 ; precision: 2/2, 1/2, 1/2
    assert m["recall_macro"] == pytest.approx((2 / 3 + 0.5 + 1) / 3)
    assert m["precision_macro"] == pytest.approx((1 + 0.5 + 0.5) / 3)
    assert m["balanced_accuracy"] == pytest.approx(m["recall_macro"])


def test_regression_metrics_match_hand_computation():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.5, 2.0, 2.0, 5.0])
    m = regression_metrics(y, p)
    assert m["mae"] == pytest.approx((0.5 + 0 + 1 + 1) / 4)
    assert m["mse"] == pytest.approx((0.25 + 0 + 1 + 1) / 4)
    assert m["rmse"] == pytest.approx(np.sqrt(2.25 / 4))
    assert m["r2"] == pytest.approx(1 - 2.25 / 5.0)
    assert m["max_error"] == 1.0
    assert m["mape"] is not None
    assert regression_metrics(np.array([0.0, 1.0]), np.array([0.1, 1.0]))["mape"] is None


def test_train_and_test_metrics_are_distinct_and_baseline_is_majority():
    df = make_classification(n=300, n_classes=2, seed=1)
    out = run_training(df, validate_config({"target": "label", "algorithm_id": "decision_tree_classifier",
                                            "hyperparameters": {"max_depth": None}}, df),
                       {"sha256": "x"}, emit=lambda **e: None)["results"]
    m = out["metrics"]
    assert m["train"]["accuracy"] == pytest.approx(1.0)  # unlimited tree memorises training data
    assert m["test"]["accuracy"] < 1.0                   # ...but the reported test score does not
    counts = out["classification"]["test_class_counts"]
    assert m["baseline"]["accuracy"] == pytest.approx(max(counts.values()) / sum(counts.values()))
    per_class_support = sum(c["support"] for c in out["classification"]["per_class"])
    assert per_class_support == out["data"]["n_test"]


def test_overview_counts():
    df = pd.DataFrame({"a": [1, 2, 2, None], "b": ["x", "y", "y", "z"], "c": [1.0, 1.0, 1.0, 1.0]})
    ov = profiling.overview(df)
    assert ov["rows"] == 4 and ov["columns"] == 3
    assert ov["missing_cells"] == 1
    assert ov["duplicate_rows"] == 1
    info = {c["name"]: c for c in ov["column_info"]}
    assert info["c"]["kind"] == "constant"
    assert info["a"]["missing"] == 1


def test_numeric_summary_and_histogram():
    s = pd.Series(list(range(1, 101)) + [1000])
    summ = profiling.numeric_summary(s)
    assert summ["median"] == 51
    assert summ["outliers"] == 1
    assert sum(summ["histogram"]["counts"]) == 101
    assert summ["box"]["whisker_high"] == 100


def test_categorical_summary():
    s = pd.Series(["a"] * 5 + ["b"] * 3 + ["c"] + [None])
    c = profiling.categorical_summary(s)
    assert c["unique"] == 3 and c["mode"] == "a" and c["mode_freq"] == 5 and c["count"] == 9


def test_correlations_flag_redundant_pairs():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    df = pd.DataFrame({"x": x, "x2": x * 2 + rng.normal(scale=0.01, size=200), "z": rng.normal(size=200)})
    corr = profiling.correlations(df)
    assert corr["high_pairs"][0]["a"] == "x" and corr["high_pairs"][0]["b"] == "x2"
    assert len(corr["matrix"]) == 3


def test_quality_findings():
    rng = np.random.default_rng(1)
    n = 100
    df = pd.DataFrame({"id": np.arange(n), "const": 1, "mostly_missing": np.where(rng.random(n) < 0.7, np.nan, 1.0),
                       "num_text": [str(v) for v in rng.integers(0, 100, n)], "x": rng.normal(size=n)})
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    ids = {f["id"].split(":")[0] for f in profiling.quality_findings(df)}
    assert {"duplicates", "constant", "missing_high", "numeric_text"} <= ids


def test_quality_finding_actions_are_applicable(settings):
    from core.datasets import DatasetRegistry
    from core.store import Store
    reg = DatasetRegistry(settings, Store(settings.db_path))
    df = pd.DataFrame({"a": [1, 1, 2, 3] * 5, "b": ["1", "1", "2", "x"] * 5, "y": [0, 0, 1, 1] * 5})
    meta = reg.register_dataframe(df, "t", None, "")
    new = reg.apply_actions(meta["id"], [{"type": "drop_duplicates"}, {"type": "cast_numeric", "params": {"columns": ["b"]}}])
    cleaned = reg.load(new["id"])
    assert len(cleaned) == 3 and pd.api.types.is_numeric_dtype(cleaned["b"])
    assert len(reg.load(meta["id"])) == 20  # original untouched
    assert new["parent_id"] == meta["id"]


@pytest.mark.parametrize("values,expected", [
    (["a", "b"] * 20, "classification"),
    ([0, 1] * 20, "classification"),
    (list(np.linspace(0, 1, 40)), "regression"),
    ([True, False] * 20, "classification"),
])
def test_problem_type_detection(values, expected):
    assert profiling.detect_problem_type(pd.Series(values))["problem_type"] == expected


def test_single_value_target_has_no_problem_type():
    assert profiling.detect_problem_type(pd.Series([1] * 10))["problem_type"] is None


def test_target_analysis_imbalance_and_leakage():
    rng = np.random.default_rng(2)
    n = 300
    y = np.where(rng.random(n) < 0.05, "fraud", "ok")
    df = pd.DataFrame({"amount": rng.normal(size=n), "status_code": np.where(y == "fraud", "F1", "OK1"), "y": y})
    t = profiling.target_analysis(df, "y")
    ids = {f["id"].split(":")[0] for f in t["findings"]}
    assert "target_imbalance" in ids
    assert "leak" in ids
    assert t["problem_type"] == "classification"


def test_regression_leakage_by_correlation():
    rng = np.random.default_rng(3)
    df = make_regression(seed=3)
    df["target_copy"] = df["target"] * 1.0001 + rng.normal(scale=1e-4, size=len(df))
    ids = [f["id"] for f in profiling.target_findings(df, "target", "regression")]
    assert "leak:target_copy" in ids


def test_recommendations_are_deterministic_and_task_aware():
    df = make_classification(n=300, seed=4)
    a = recommend.recommend(df, "label")
    b = recommend.recommend(df, "label")
    assert a == b
    assert a["task"] == "classification"
    assert all(r["task"] == "classification" for r in a["algorithms"])
    assert a["baseline"] == "logistic_regression"
    big = make_classification(n=30000, seed=4)
    rec = {r["id"]: r for r in recommend.recommend(big, "label")["algorithms"]}
    assert rec["svc"]["cautions"]  # quadratic cost warning
    assert rec["hist_gradient_boosting_classifier"]["score"] > rec["svc"]["score"]


def test_scatter_sampling():
    df = make_regression(n=5000)
    sc = profiling.scatter(df, "a", "target", color="kind", limit=500)
    assert sc["shown"] == 500 and sc["total"] == 5000
    assert set(sc["groups"]) <= {"x", "y", "z"}
