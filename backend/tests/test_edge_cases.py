"""Invalid and awkward datasets must fail clearly or be handled explicitly."""
import io

import numpy as np
import pandas as pd
import pytest

from core.datasets import DatasetError, read_table
from core.training import TrainingError, run_training, validate_config

from .conftest import make_classification

META = {"id": "t", "sha256": "s"}


def run(df, **cfg):
    return run_training(df, validate_config(cfg, df), META, emit=lambda **e: None)


def test_empty_and_header_only_files_rejected(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(DatasetError, match="empty"):
        read_table(empty)
    unsupported = tmp_path / "x.xlsx"
    unsupported.write_bytes(b"PK")
    with pytest.raises(DatasetError, match="Unsupported"):
        read_table(unsupported)


def test_semicolon_and_latin1_csv(tmp_path):
    p = tmp_path / "semi.csv"
    p.write_bytes("name;value\ncafé;1\nnaïve;2\n".encode("latin1"))
    df = read_table(p)
    assert list(df.columns) == ["name", "value"]
    assert df["name"].tolist() == ["café", "naïve"]


def test_single_column_dataset_rejected():
    df = pd.DataFrame({"y": np.arange(50)})
    with pytest.raises(TrainingError, match="at least one feature"):
        validate_config({"target": "y", "algorithm_id": "linear_regression"}, df)


def test_invalid_target_rejected():
    with pytest.raises(TrainingError, match="does not exist"):
        validate_config({"target": "nope", "algorithm_id": "logistic_regression"}, make_classification())


def test_missing_target_rows_dropped_and_reported():
    df = make_classification(n=200, seed=1)
    df.loc[:9, "label"] = np.nan
    out = run(df, target="label", algorithm_id="logistic_regression", options={"permutation_importance": False})
    d = out["results"]["data"]
    assert d["dropped_missing_target"] == 10
    assert d["n_used"] == 190 - d["dropped_duplicates"]
    assert any("no target value" in w for w in out["results"]["warnings"])


def test_missing_feature_values_are_imputed():
    df = make_classification(n=200, seed=2, missing=0.3)
    out = run(df, target="label", algorithm_id="svc", options={"permutation_importance": False})
    assert out["results"]["metrics"]["test"]["accuracy"] > 0.5


def test_constant_and_empty_columns_excluded():
    df = make_classification(n=200, seed=3)
    df["const"] = 7
    df["blank"] = np.nan
    out = run(df, target="label", algorithm_id="logistic_regression", options={"permutation_importance": False})
    feats = out["bundle"]["feature_columns"]
    assert "const" not in feats and "blank" not in feats


def test_single_class_target_fails():
    df = pd.DataFrame({"x": np.arange(30.0), "y": ["a"] * 30})
    with pytest.raises(TrainingError):
        run(df, target="y", algorithm_id="logistic_regression", task_type="classification")


def test_continuous_target_as_classification_fails():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=200), "y": rng.normal(size=200)})
    with pytest.raises(TrainingError, match="continuous"):
        run(df, target="y", algorithm_id="logistic_regression", task_type="classification")


def test_text_target_for_regression_fails():
    df = pd.DataFrame({"x": np.arange(40.0), "y": ["a", "b"] * 20})
    with pytest.raises(TrainingError, match="numeric target"):
        validate_config({"target": "y", "algorithm_id": "ridge", "task_type": "regression"}, df)


def test_too_few_rows_fails():
    df = pd.DataFrame({"x": np.arange(8.0), "y": np.arange(8.0) * 2})
    with pytest.raises(TrainingError, match="at least 10"):
        run(df, target="y", algorithm_id="linear_regression")


def test_rare_singleton_class_requires_explicit_choice():
    df = make_classification(n=120, n_classes=2, seed=4)
    df.loc[0, "label"] = "unicorn"
    with pytest.raises(TrainingError, match="only one example"):
        run(df, target="label", algorithm_id="logistic_regression")
    out = run(df, target="label", algorithm_id="logistic_regression", preprocessing={"drop_rare_classes": True},
              options={"permutation_importance": False})
    assert out["results"]["data"]["dropped_rare_classes"] == 1
    assert "unicorn" not in out["results"]["data"]["classes"]


def test_small_dataset_reduces_cv_folds_with_warning():
    rng = np.random.default_rng(5)
    labels = ["a"] * 40 + ["b"] * 12 + ["c"] * 4  # 'c' leaves ~3 training rows: 5 folds impossible
    df = pd.DataFrame({"x": rng.normal(size=56), "y": labels})
    df["x"] += df["y"].map({"a": 0, "b": 2, "c": 4})
    out = run(df, target="y", algorithm_id="decision_tree_classifier", split={"cv_folds": 5},
              options={"permutation_importance": False})
    cv = out["results"]["metrics"]["cv"]
    train_c = out["results"]["classification"]["train_class_counts"]["c"]
    assert cv["k"] == train_c < 5
    assert any("Reduced CV folds" in w for w in out["results"]["warnings"])


def test_imbalanced_stratified_split_keeps_minority_in_both_splits():
    df = make_classification(n=400, n_classes=2, seed=6, weights=[0.95, 0.05])
    out = run(df, target="label", algorithm_id="logistic_regression",
              hyperparameters={"class_weight": "balanced"}, options={"permutation_importance": False})
    c = out["results"]["classification"]
    assert out["results"]["split"]["stratified"]
    assert len(c["test_class_counts"]) == 2 and len(c["train_class_counts"]) == 2


def test_stratified_split_impossible_gives_clear_error():
    df = make_classification(n=30, n_classes=2, seed=7)
    df["label"] = [f"c{i % 10}" for i in range(30)]
    with pytest.raises(TrainingError, match="stratified split needs"):
        run(df, target="label", algorithm_id="logistic_regression", split={"test_size": 0.1})


def test_numeric_integer_target_detected_as_classification():
    df = make_classification(n=200, seed=8)
    df["label"] = df["label"].map({"class_0": 0, "class_1": 1})
    r = validate_config({"target": "label", "algorithm_id": "logistic_regression"}, df)
    assert r["task_type"] == "classification"


def test_high_cardinality_categorical_is_capped():
    rng = np.random.default_rng(9)
    n = 600
    df = pd.DataFrame({"x": rng.normal(size=n), "city": [f"city_{i}" for i in rng.integers(0, 300, n)]})
    df["y"] = df.x * 2 + rng.normal(size=n)
    out = run(df, target="y", algorithm_id="ridge", options={"permutation_importance": False})
    assert out["results"]["data"]["encoded_features"] <= 1 + 50


def test_boolean_and_mixed_type_columns():
    rng = np.random.default_rng(10)
    n = 200
    df = pd.DataFrame({"flag": rng.random(n) > 0.5, "mixed": [1 if i % 3 else "one" for i in range(n)],
                       "x": rng.normal(size=n)})
    df["y"] = np.where(df.flag, "yes", "no")
    out = run(df, target="y", algorithm_id="random_forest_classifier", options={"permutation_importance": False})
    assert out["results"]["metrics"]["test"]["accuracy"] > 0.9


def test_invalid_split_options_rejected():
    df = make_classification()
    for bad in ({"test_size": 0.9}, {"test_size": 0.0}, {"cv_folds": 1}, {"cv_folds": 50}, {"random_state": -1}, {"seed": 1}):
        with pytest.raises(TrainingError):
            validate_config({"target": "label", "algorithm_id": "logistic_regression", "split": bad}, df)


def test_invalid_preprocessing_rejected():
    df = make_classification()
    for bad in ({"scaling": "magic"}, {"encoding": "hash"}, {"k_features": 0}, {"nonsense": True}):
        with pytest.raises(TrainingError):
            validate_config({"target": "label", "algorithm_id": "logistic_regression", "preprocessing": bad}, df)


def test_all_features_excluded_fails():
    df = make_classification()
    with pytest.raises(TrainingError, match="No usable feature"):
        validate_config({"target": "label", "algorithm_id": "logistic_regression",
                         "preprocessing": {"excluded_columns": ["f1", "f2", "f3", "f4", "colour"]}}, df)


def test_cancellation_between_stages():
    from core.training import Cancelled
    df = make_classification(n=200, seed=11)
    resolved = validate_config({"target": "label", "algorithm_id": "logistic_regression"}, df)
    calls = {"n": 0}

    def cancelled():
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(Cancelled):
        run_training(df, resolved, META, emit=lambda **e: None, is_cancelled=cancelled)
