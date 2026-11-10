"""Preprocessing and feature selection must never learn from the test split."""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from core.training import run_training, validate_config

META = {"id": "t", "sha256": "s"}


def run(df, **cfg):
    return run_training(df, validate_config(cfg, df), META, emit=lambda **e: None)


def _reconstruct_split(df, target, resolved_split, stratify):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=resolved_split["test_size"], random_state=resolved_split["random_state"],
                            shuffle=True, stratify=y if stratify else None)


def test_scaler_and_imputer_fitted_on_training_split_only():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    df.loc[::7, "z"] = np.nan
    df["y"] = 2 * df.x + rng.normal(scale=0.1, size=n)
    out = run(df, target="y", algorithm_id="ridge", preprocessing={"numeric_imputation": "mean"},
              options={"permutation_importance": False})
    X_train, X_test, _, _ = _reconstruct_split(df, "y", out["results"]["split"], False)
    assert out["results"]["data"]["n_test"] == len(X_test)
    num = out["bundle"]["pipeline"].named_steps["preprocess"].named_transformers_["num"]
    imputer, scaler = num.named_steps["impute"], num.named_steps["scale"]
    assert imputer.statistics_ == pytest.approx(X_train[["x", "z"]].mean().to_numpy())
    assert not np.allclose(imputer.statistics_, df[["x", "z"]].mean().to_numpy())
    assert int(np.max(scaler.n_samples_seen_)) == len(X_train)


def test_feature_selection_does_not_leak_on_pure_noise():
    """Classic leakage trap: selecting features on all rows makes noise look predictive.
    Inside the pipeline, test accuracy on random labels must stay near chance."""
    rng = np.random.default_rng(1)
    n, p = 120, 400
    df = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"n{i}" for i in range(p)])
    df["y"] = rng.choice(["a", "b"], size=n)
    out = run(df, target="y", algorithm_id="logistic_regression",
              preprocessing={"feature_selection": "kbest", "k_features": 10},
              split={"cv_folds": 5}, options={"permutation_importance": False})
    m = out["results"]["metrics"]
    assert m["cv"]["mean"]["accuracy"] < 0.7
    assert m["test"]["accuracy"] < 0.75
    # The selector saw only training rows
    selector = out["bundle"]["pipeline"].named_steps["select"]
    assert selector.scores_.shape == (p,)


def test_iqr_clipping_fences_learned_from_training_split():
    rng = np.random.default_rng(2)
    n = 200
    df = pd.DataFrame({"x": rng.normal(size=n)})
    df["y"] = df.x * 3 + rng.normal(scale=0.2, size=n)
    out = run(df, target="y", algorithm_id="linear_regression", preprocessing={"outliers": "clip_iqr"},
              options={"permutation_importance": False})
    X_train, *_ = _reconstruct_split(df, "y", out["results"]["split"], False)
    clip = out["bundle"]["pipeline"].named_steps["preprocess"].named_transformers_["num"].named_steps["clip"]
    q1, q3 = np.percentile(X_train["x"], [25, 75])
    assert clip.lower_[0] == pytest.approx(q1 - 1.5 * (q3 - q1))


def test_duplicates_removed_before_split_so_none_cross_train_and_test():
    rng = np.random.default_rng(3)
    base = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})
    base["y"] = (base.a > 0).map({True: "pos", False: "neg"})
    df = pd.concat([base] * 4, ignore_index=True)  # every row appears 4 times
    df.insert(0, "row_id", range(len(df)))  # unique ID that would hide duplicates
    out = run(df, target="y", algorithm_id="knn_classifier", options={"permutation_importance": False})
    data = out["results"]["data"]
    assert "row_id" in data["features"]["excluded"]
    assert data["dropped_duplicates"] == 180
    assert data["n_used"] == 60


def test_duplicates_kept_when_disabled_and_reported():
    rng = np.random.default_rng(3)
    base = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})
    base["y"] = (base.a > 0).map({True: "pos", False: "neg"})
    df = pd.concat([base] * 2, ignore_index=True)
    out = run(df, target="y", algorithm_id="knn_classifier", preprocessing={"drop_duplicates": False},
              options={"permutation_importance": False})
    assert out["results"]["data"]["dropped_duplicates"] == 0
    assert out["results"]["data"]["n_used"] == 120


def test_target_never_used_as_feature_and_sorted_ids_excluded():
    rng = np.random.default_rng(4)
    n = 150
    y = np.repeat(["a", "b", "c"], n // 3)  # sorted by class, like iris
    df = pd.DataFrame({"Id": np.arange(1, n + 1), "x": rng.normal(size=n) + (y == "a"), "y": y})
    out = run(df, target="y", algorithm_id="random_forest_classifier", options={"permutation_importance": False})
    feats = out["bundle"]["feature_columns"]
    assert "y" not in feats and "Id" not in feats
    assert out["results"]["data"]["features"]["excluded"] == ["Id"]


def test_unseen_category_in_test_does_not_crash():
    rng = np.random.default_rng(5)
    n = 200
    df = pd.DataFrame({"x": rng.normal(size=n), "cat": rng.choice(["a", "b", "c"], n)})
    df.loc[n - 1, "cat"] = "only_once"
    df["y"] = (df.x > 0).map({True: "p", False: "q"})
    out = run(df, target="y", algorithm_id="logistic_regression", options={"permutation_importance": False})
    pipe = out["bundle"]["pipeline"]
    pred = pipe.predict(pd.DataFrame({"x": [0.1], "cat": ["never_seen"]}))
    assert pred.shape == (1,)


def test_cross_validation_uses_only_training_rows():
    df = pd.DataFrame({"x": np.arange(100.0), "y": np.arange(100.0) * 2})
    events = []
    resolved = validate_config({"target": "y", "algorithm_id": "linear_regression", "split": {"cv_folds": 4}}, df)
    out = run_training(df, resolved, META, emit=lambda **e: events.append(e))
    folds = out["results"]["metrics"]["cv"]["folds"]
    n_train = out["results"]["data"]["n_train"]
    assert all(f["n_train"] + f["n_val"] == n_train for f in folds)
    assert [e["progress"]["current"] for e in events if e.get("progress")] == [1, 2, 3, 4]
