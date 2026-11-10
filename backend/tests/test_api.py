"""HTTP + WebSocket behaviour, persistence and legacy compatibility."""
import io
import time

import numpy as np
import pandas as pd
import pytest

from .conftest import make_classification, make_regression, upload, wait


def test_health_and_system(client):
    assert client.get("/health").json()["status"] == "healthy"
    sysinfo = client.get("/api/system").json()
    assert sysinfo["llm"]["available"] is False  # no key in tests
    assert "scikit_learn" in sysinfo["versions"]


def test_upload_validation(client):
    r = client.post("/api/datasets", files={"file": ("x.exe", io.BytesIO(b"MZ"), "application/octet-stream")})
    assert r.status_code == 422
    r = client.post("/api/datasets", files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")})
    assert r.status_code == 422
    r = client.post("/api/datasets", files={"file": ("hdr.csv", io.BytesIO(b"a,b\n"), "text/csv")})
    assert r.status_code == 422 and "no data rows" in r.json()["detail"]
    # path traversal in filename cannot escape the managed directory
    r = client.post("/api/datasets", files={"file": ("../../evil.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")})
    assert r.status_code == 200
    assert ".." not in r.json()["filename"]
    assert "path" not in r.json()


def test_upload_size_limit(client):
    client.app_ref.state.settings.max_upload_mb = 0
    r = client.post("/api/datasets", files={"file": ("big.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")})
    assert r.status_code == 422 and "limit" in r.json()["detail"]


def test_dataset_browse_endpoints(client):
    df = make_classification(n=120, seed=1, missing=0.1)
    ds = upload(client, df)
    listed = client.get("/api/datasets").json()["datasets"]
    assert listed[0]["id"] == ds and listed[0]["n_rows"] == 120
    page = client.get(f"/api/datasets/{ds}/rows", params={"offset": 10, "limit": 5, "sort": "f1", "order": "desc"}).json()
    assert len(page["rows"]) == 5 and page["total"] == 120
    f1 = page["columns"].index("f1")
    vals = [r[f1] for r in page["rows"]]
    assert vals == sorted(vals, reverse=True)
    searched = client.get(f"/api/datasets/{ds}/rows", params={"search": "class_1", "column": "label"}).json()
    assert searched["filtered"] == int((df.label == "class_1").sum())
    missing = client.get(f"/api/datasets/{ds}/rows", params={"missing_only": True}).json()
    assert missing["filtered"] == int(df.isna().any(axis=1).sum())
    assert client.get(f"/api/datasets/{ds}/rows", params={"sort": "nope"}).status_code == 400
    prof = client.get(f"/api/datasets/{ds}/profile").json()
    assert prof["overview"]["rows"] == 120
    col = client.get(f"/api/datasets/{ds}/columns/f2").json()
    assert col["missing"] == int(df.f2.isna().sum())
    assert client.get(f"/api/datasets/{ds}/columns/zzz").status_code == 404
    assert client.get(f"/api/datasets/{ds}/correlations").json()["method"] == "pearson"
    assert "f2" in client.get(f"/api/datasets/{ds}/missing").json()["columns"]
    sc = client.get(f"/api/datasets/{ds}/scatter", params={"x": "f1", "y": "f3", "color": "label"}).json()
    assert sc["shown"] == 120
    tgt = client.get(f"/api/datasets/{ds}/target", params={"target": "label"}).json()
    assert tgt["analysis"]["problem_type"] == "classification"
    assert tgt["recommendation"]["suggested"]
    assert client.get("/api/datasets/nope/profile").status_code == 404


def test_clean_creates_new_version(client):
    df = pd.concat([make_regression(n=50)] * 2, ignore_index=True)
    ds = upload(client, df)
    r = client.post(f"/api/datasets/{ds}/clean", json={"actions": [{"type": "drop_duplicates"}]})
    assert r.status_code == 200
    new = r.json()
    assert new["parent_id"] == ds and new["n_rows"] == 50
    assert client.get(f"/api/datasets/{ds}").json()["n_rows"] == 100
    assert client.post(f"/api/datasets/{ds}/clean", json={"actions": [{"type": "rm -rf"}]}).status_code == 400


def test_full_experiment_lifecycle(client):
    ds = upload(client, make_classification(n=200, seed=2))
    r = client.post("/api/experiments", json={
        "dataset_id": ds, "target": "label", "name": "lifecycle",
        "algorithms": [{"id": "logistic_regression"}, {"id": "random_forest_classifier", "hyperparameters": {"n_estimators": 30}}],
        "split": {"test_size": 0.25, "cv_folds": 3}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["group_id"] and len(body["experiments"]) == 2
    exps = [wait(client, e["id"]) for e in body["experiments"]]
    assert all(e["status"] == "completed" for e in exps), [e["error"] for e in exps]
    assert exps[0]["results"]["split"]["split_id"] == exps[1]["results"]["split"]["split_id"]
    assert exps[0]["metrics"]["accuracy"] == exps[0]["results"]["metrics"]["test"]["accuracy"]
    eid = exps[0]["id"]
    listing = client.get("/api/experiments").json()["experiments"]
    assert {e["id"] for e in listing} >= {e["id"] for e in exps}
    assert "results" not in listing[0]  # list stays light
    rep = client.get(f"/api/experiments/{eid}/report").json()["report"]
    assert rep["observations"] and rep["limitations"] and rep["disclaimer"]
    md = client.get(f"/api/experiments/{eid}/export-report", params={"format": "md"})
    assert md.status_code == 200 and "## Metrics (calculated)" in md.text
    txt = client.get(f"/experiments/{eid}/export-report")
    assert txt.status_code == 200 and "attachment" in txt.headers["content-disposition"]
    model = client.get(f"/api/experiments/{eid}/export-model")
    assert model.status_code == 200 and len(model.content) > 100
    pred = client.post(f"/api/experiments/{eid}/predict", json={"rows": [{"f1": 0, "f2": 1, "f3": 0, "f4": 0, "colour": "red"}]}).json()
    assert pred["predictions"][0]["prediction"] in ("class_0", "class_1")
    assert abs(sum(pred["predictions"][0]["probabilities"].values()) - 1) < 1e-6
    bad = client.post(f"/api/experiments/{eid}/predict", json={"rows": [{"f1": 0}]})
    assert bad.status_code == 400
    # narrative requires an LLM: clean 503, not a crash
    assert client.post(f"/api/experiments/{eid}/report/narrative").status_code == 503
    assert client.delete(f"/api/experiments/{eid}").status_code == 200
    assert client.get(f"/api/experiments/{eid}").status_code == 404


def test_failed_experiment_is_recorded_not_hidden(client):
    df = make_classification(n=60, seed=3)
    df["label"] = [f"c{i % 10}" for i in range(60)]
    ds = upload(client, df)
    r = client.post("/api/experiments", json={"dataset_id": ds, "target": "label",
                                              "algorithms": [{"id": "logistic_regression"}], "split": {"test_size": 0.1}})
    exp = wait(client, r.json()["experiments"][0]["id"])
    assert exp["status"] == "failed" and "stratified" in exp["error"]
    assert client.get(f"/api/experiments/{exp['id']}/report").status_code == 409


def test_request_validation_errors(client):
    ds = upload(client, make_regression(n=80))
    def post(**kw):
        return client.post("/api/experiments", json={"dataset_id": ds, "target": "target", **kw})
    assert post(algorithms=[]).status_code == 422
    assert post(algorithms=[{"id": "nope"}]).status_code == 400
    assert post(algorithms=[{"id": "gaussian_nb"}]).status_code == 400  # classifier on regression
    assert post(algorithms=[{"id": "ridge"}, {"id": "ridge"}]).status_code == 400
    assert post(algorithms=[{"id": "ridge", "hyperparameters": {"alpha": -3}}]).status_code == 400
    assert client.post("/api/experiments", json={"dataset_id": "missing", "target": "x", "algorithms": [{"id": "ridge"}]}).status_code == 404


def test_duplicate_submission_rejected_and_cancel(client):
    rng = np.random.default_rng(4)
    n = 4000
    df = pd.DataFrame(rng.normal(size=(n, 20)), columns=[f"x{i}" for i in range(20)])
    df["y"] = (df.x0 + df.x1 > 0).map({True: "a", False: "b"})
    ds = upload(client, df)
    payload = {"dataset_id": ds, "target": "y", "algorithms": [{"id": "svc"}], "split": {"cv_folds": 10}}
    first = client.post("/api/experiments", json=payload)
    second = client.post("/api/experiments", json=payload)
    assert second.status_code == 409
    eid = first.json()["experiments"][0]["id"]
    assert second.json()["experiment_id"] == eid
    assert client.post(f"/api/experiments/{eid}/cancel").status_code == 200
    exp = wait(client, eid)
    assert exp["status"] == "cancelled"
    assert client.post(f"/api/experiments/{eid}/cancel").status_code == 409


def test_websocket_replays_history_for_late_subscribers(client):
    ds = upload(client, make_regression(n=150))
    eid = client.post("/api/experiments", json={"dataset_id": ds, "target": "target",
                                                "algorithms": [{"id": "ridge"}]}).json()["experiments"][0]["id"]
    wait(client, eid)
    for path in (f"/api/ws/experiments/{eid}", f"/ws/train/{eid}"):
        with client.websocket_connect(path) as ws:
            events = []
            while True:
                e = ws.receive_json()
                events.append(e)
                if e.get("terminal"):
                    break
        stages = [e.get("stage") for e in events if e.get("type") == "progress"]
        assert stages[0] == "prepare" and "evaluate" in stages
        seqs = [e["seq"] for e in events if "seq" in e]
        assert seqs == sorted(seqs) and len(seqs) == len(set(seqs))
        assert events[-1]["status"] == "completed"


def test_websocket_unknown_experiment(client):
    with client.websocket_connect("/api/ws/experiments/nope") as ws:
        assert ws.receive_json()["type"] == "error"


def test_persistence_across_restart(settings, monkeypatch):
    from fastapi.testclient import TestClient

    from main import create_app
    with TestClient(create_app(settings, static_dir=settings.data_dir / "x")) as c1:
        ds = upload(c1, make_regression(n=100))
        eid = c1.post("/api/experiments", json={"dataset_id": ds, "target": "target",
                                                 "algorithms": [{"id": "ridge"}]}).json()["experiments"][0]["id"]
        wait(c1, eid)
    # simulate an experiment left running by a crashed process
    from core.store import Store
    Store(settings.db_path).save_experiment({"id": "zombie", "name": "z", "algorithm": "Ridge", "dataset_path": "",
                                             "config": {}, "status": "running", "metrics": {}, "logs": [],
                                             "created_at": "2026-01-01T00:00:00+00:00"})
    with TestClient(create_app(settings, static_dir=settings.data_dir / "x")) as c2:
        exp = c2.get(f"/api/experiments/{eid}").json()["experiment"]
        assert exp["status"] == "completed" and exp["results"]["metrics"]["test"]["r2"] is not None
        assert c2.get(f"/api/datasets/{ds}/profile").status_code == 200
        zombie = c2.get("/api/experiments/zombie").json()["experiment"]
        assert zombie["status"] == "failed" and "Interrupted" in zombie["error"]


def test_settings_validation(client):
    assert client.put("/api/settings", json={"llm_provider": "skynet"}).status_code == 400
    assert client.put("/api/settings", json={"ollama_url": "file:///etc"}).status_code == 400
    assert client.put("/api/settings", json={"api_key": "x"}).status_code == 400
    r = client.put("/api/settings", json={"llm_provider": "none"})
    assert r.status_code == 200 and r.json()["status"]["available"] is False


def test_recommendation_explanation_without_llm_is_503(client):
    ds = upload(client, make_regression(n=80))
    r = client.post("/api/recommendations/explain", json={"dataset_id": ds, "target": "target"})
    assert r.status_code == 503


def test_samples_import(client):
    samples = client.get("/api/samples").json()["samples"]
    assert {s["name"] for s in samples} >= {"iris", "sample"}
    meta = client.post("/api/samples/iris/import").json()
    assert meta["n_rows"] == 150
    again = client.post("/api/samples/iris/import").json()
    assert again["id"] == meta["id"]  # idempotent
    assert client.post("/api/samples/..%2F..%2Fetc%2Fpasswd/import").status_code == 404


# ---------------------------------------------------------------- legacy API
def test_legacy_endpoints_still_work(client):
    buf = io.BytesIO(make_classification(n=120, seed=5).to_csv(index=False).encode())
    up = client.post("/upload-dataset", files={"file": ("legacy.csv", buf, "text/csv")}).json()
    assert up["status"] == "success"
    path = up["file_path"]
    prev = client.post("/get-dataset-preview", params={"file_path": path, "rows": 5}).json()
    assert len(prev["data"]) == 5 and prev["total_rows"] == 120
    an = client.post("/analyze-dataset", json={"file_path": path, "dataset_type": "tabular", "target_column": "label"}).json()
    assert an["analysis"]["task_type"] == "classification" and an["analysis"]["rows"] == 120
    rec = client.post("/recommend-algorithm", json={"file_path": path, "dataset_type": "tabular",
                                                    "task_type": "classification", "target_column": "label"}).json()
    assert rec["recommendation"]["confidence"] is None  # no invented confidence
    tr = client.post("/train-model", json={"experiment_name": "legacy", "algorithm": "Random Forest", "dataset_path": path,
                                           "config": {"target_column": "label", "task_type": "classification"}}).json()
    exp = wait(client, tr["experiment_id"])
    assert exp["status"] == "completed" and exp["algorithm"] == "Random Forest"
    assert client.get(f"/experiments/{exp['id']}/report").json()["report"]["summary"]


def test_legacy_endpoints_reject_bad_input(client):
    assert client.post("/get-dataset-preview", params={"file_path": "/etc/passwd"}).status_code == 403
    assert client.post("/analyze-dataset", json={"file_path": "../../../etc/hosts"}).status_code == 403
    buf = io.BytesIO(make_regression(n=60).to_csv(index=False).encode())
    path = client.post("/upload-dataset", files={"file": ("r.csv", buf, "text/csv")}).json()["file_path"]
    bad_alg = client.post("/train-model", json={"experiment_name": "x", "algorithm": "Foo Boost 9000", "dataset_path": path,
                                                "config": {"target_column": "target"}})
    assert bad_alg.status_code == 400
    bad_target = client.post("/train-model", json={"experiment_name": "x", "algorithm": "Ridge", "dataset_path": path,
                                                   "config": {"target_column": "zz"}})
    assert bad_target.status_code == 400
