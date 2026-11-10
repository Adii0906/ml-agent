"""Every advertised algorithm must actually train, predict and evaluate correctly."""
import numpy as np
import pandas as pd
import pytest

from core import algorithms as algs
from core.training import TrainingError, run_training, validate_config

from .conftest import make_classification, make_regression

META = {"id": "test", "sha256": "abc", "name": "synthetic"}
CLF = [a for a in algs.for_task("classification") if a.available]
REG = [a for a in algs.for_task("regression") if a.available]


def train(df, **cfg):
    resolved = validate_config(cfg, df)
    return run_training(df, resolved, META, emit=lambda **e: None)


@pytest.mark.parametrize("alg", CLF, ids=lambda a: a.id)
@pytest.mark.parametrize("n_classes", [2, 3])
def test_classifier_trains_and_beats_baseline(alg, n_classes):
    df = make_classification(n=300, n_classes=n_classes, seed=1)
    out = train(df, target="label", algorithm_id=alg.id, split={"cv_folds": 3}, options={"permutation_importance": False})
    res = out["results"]
    test = res["metrics"]["test"]
    assert res["algorithm"]["id"] == alg.id  # no silent substitution
    assert type(out["bundle"]["pipeline"][-1]).__name__ == res["model_info"]["estimator"]
    for k in ("accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro"):
        assert 0.0 <= test[k] <= 1.0
    assert test["accuracy"] > res["metrics"]["baseline"]["accuracy"] + 0.1, alg.id
    cm = np.array(res["classification"]["confusion_matrix"])
    assert cm.shape == (n_classes, n_classes)
    assert cm.sum() == res["data"]["n_test"]
    assert np.isclose(np.trace(cm) / cm.sum(), test["accuracy"])
    # prediction dimensions and label space
    pipe = out["bundle"]["pipeline"]
    X = df.drop(columns=["label"]).head(17)
    pred = pipe.predict(X)
    assert pred.shape == (17,)
    assert set(np.unique(pred)) <= set(range(n_classes))
    assert len(res["metrics"]["cv"]["folds"]) == 3


@pytest.mark.parametrize("alg", REG, ids=lambda a: a.id)
def test_regressor_trains_and_beats_baseline(alg):
    df = make_regression(n=300, seed=2)
    out = train(df, target="target", algorithm_id=alg.id, split={"cv_folds": 3}, options={"permutation_importance": False})
    res = out["results"]
    test = res["metrics"]["test"]
    assert res["algorithm"]["id"] == alg.id
    assert test["rmse"] == pytest.approx(np.sqrt(test["mse"]))
    assert test["r2"] > res["metrics"]["baseline"]["r2"], alg.id
    if alg.id not in ("svr",):  # SVR with an unscaled target is weak by design (documented limitation)
        assert test["r2"] > 0.5, (alg.id, test["r2"])
    pred = out["bundle"]["pipeline"].predict(df.drop(columns=["target"]).head(11))
    assert pred.shape == (11,)
    assert np.issubdtype(pred.dtype, np.floating)
    assert res["regression"]["points_total"] == res["data"]["n_test"]


@pytest.mark.parametrize("alg", CLF[:3] + REG[:3], ids=lambda a: a.id)
def test_training_is_reproducible(alg):
    df = make_classification(seed=3) if alg.task == "classification" else make_regression(seed=3)
    target = "label" if alg.task == "classification" else "target"
    a = train(df, target=target, algorithm_id=alg.id)["results"]
    b = train(df, target=target, algorithm_id=alg.id)["results"]
    assert a["metrics"] == b["metrics"]
    assert a["split"]["split_id"] == b["split"]["split_id"]
    c = train(df, target=target, algorithm_id=alg.id, split={"random_state": 7})["results"]
    assert c["split"]["split_id"] != a["split"]["split_id"]


def test_same_split_across_algorithms():
    df = make_classification(seed=4)
    ids = {train(df, target="label", algorithm_id=a)["results"]["split"]["split_id"]
           for a in ("logistic_regression", "random_forest_classifier", "knn_classifier")}
    assert len(ids) == 1


def test_hyperparameters_are_applied():
    df = make_classification(seed=5)
    out = train(df, target="label", algorithm_id="random_forest_classifier",
                hyperparameters={"n_estimators": 13, "max_depth": 2})
    model = out["bundle"]["pipeline"][-1]
    assert model.n_estimators == 13 and model.max_depth == 2
    assert out["results"]["algorithm"]["hyperparameters"]["n_estimators"] == 13


@pytest.mark.parametrize("bad", [{"n_estimators": 0}, {"n_estimators": 2.5}, {"max_depth": "deep"}, {"unknown": 1}])
def test_invalid_hyperparameters_rejected(bad):
    df = make_classification(seed=6)
    with pytest.raises(TrainingError):
        validate_config({"target": "label", "algorithm_id": "random_forest_classifier", "hyperparameters": bad}, df)


def test_choice_hyperparameter_validated():
    df = make_classification(seed=6)
    with pytest.raises(TrainingError):
        validate_config({"target": "label", "algorithm_id": "svc", "hyperparameters": {"kernel": "magic"}}, df)


def test_unknown_algorithm_never_falls_back():
    df = make_classification(seed=7)
    with pytest.raises(TrainingError, match="Unknown algorithm"):
        validate_config({"target": "label", "algorithm_id": "foo_boost_9000"}, df)
    with pytest.raises(algs.AlgorithmError):
        algs.resolve_name("Foo Boost 9000", "classification")


def test_task_mismatch_rejected():
    df = make_regression()
    with pytest.raises(TrainingError, match="classification algorithm"):
        validate_config({"target": "target", "algorithm_id": "gaussian_nb", "task_type": "regression"}, df)
    with pytest.raises(TrainingError, match="regression algorithm"):
        validate_config({"target": "label", "algorithm_id": "linear_regression"}, make_classification())


def test_legacy_names_resolve_by_task():
    assert algs.resolve_name("Random Forest", "classification").id == "random_forest_classifier"
    assert algs.resolve_name("Random Forest", "regression").id == "random_forest_regressor"
    assert algs.resolve_name("SVM", "regression").id == "svr"
    with pytest.raises(algs.AlgorithmError):
        algs.resolve_name("Naive Bayes", "regression")


def test_registry_metadata_complete():
    for a in algs.ALGORITHMS:
        assert a.summary and a.strengths and a.limitations
        defaults = a.validate_params({})
        assert set(defaults) == {p.name for p in a.params}


def test_roc_auc_binary_and_multiclass():
    binary = train(make_classification(n=300, n_classes=2, seed=8), target="label", algorithm_id="logistic_regression")["results"]
    assert 0.5 < binary["classification"]["roc"]["auc"] <= 1.0
    multi = train(make_classification(n=300, n_classes=3, seed=8), target="label", algorithm_id="logistic_regression")["results"]
    assert multi["classification"]["roc"]["average"] == "macro one-vs-rest"
    svc_multi = train(make_classification(n=300, n_classes=3, seed=8), target="label", algorithm_id="svc")["results"]
    assert svc_multi["classification"]["roc"]["auc"] is None
    assert "probabilities" in svc_multi["classification"]["roc"]["note"]
    svc_bin = train(make_classification(n=300, n_classes=2, seed=8), target="label", algorithm_id="svc")["results"]
    assert svc_bin["classification"]["roc"]["score_type"] == "decision function"
