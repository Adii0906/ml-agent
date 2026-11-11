"""Saved models: save, persist, download, load outside the server, predict, delete, security."""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .conftest import make_classification, make_regression, upload, wait

BACKEND = Path(__file__).resolve().parents[1]


def train(client, df, target, alg):
    ds = upload(client, df)
    r = client.post("/api/experiments", json={"dataset_id": ds, "target": target, "algorithms": [{"id": alg}],
                                              "preprocessing": {"outliers": "clip_iqr"}})
    assert r.status_code == 200, r.text
    exp = wait(client, r.json()["experiments"][0]["id"])
    assert exp["status"] == "completed", exp["error"]
    return exp


def save(client, exp_id, name="Churn model", description="first cut", headers=None):
    return client.post(f"/api/experiments/{exp_id}/save-model", json={"name": name, "description": description},
                       headers=headers or {})


def make_client(settings):
    from fastapi.testclient import TestClient

    from main import create_app
    return TestClient(create_app(settings, static_dir=settings.data_dir / "no-static"))


def run_isolated(artifact: Path, sha: str, rows, tmp_path: Path) -> list:
    """Load the artifact in a fresh interpreter that can import ONLY mlagent_runtime
    (copied to an empty directory) — not the ML-Agent backend."""
    env_dir = tmp_path / "isolated"
    env_dir.mkdir(parents=True)
    shutil.copytree(BACKEND / "mlagent_runtime", env_dir / "mlagent_runtime")
    script = textwrap.dedent(f"""
        import json, sys
        sys.path = [p for p in sys.path if 'backend' not in p]
        try:
            import core  # noqa
            raise SystemExit("backend package leaked into the isolated environment")
        except ImportError:
            pass
        from mlagent_runtime import load_artifact, predict
        bundle = load_artifact({str(artifact)!r}, expected_sha256={sha!r})
        print(json.dumps({{"preds": predict(bundle, json.loads(sys.argv[1])),
                          "manifest_target": bundle["manifest"]["target"]}}))
    """)
    out = subprocess.run([sys.executable, "-c", script, json.dumps(rows)], cwd=env_dir, capture_output=True,
                         text=True, env={"PYTHONPATH": str(env_dir), "PATH": os.environ.get("PATH", "")}, timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_full_save_download_load_predict_workflow(settings, tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    df = make_classification(n=300, seed=11, missing=0.1)
    rows = json.loads(df.drop(columns=["label"]).head(5).to_json(orient="records"))
    with make_client(settings) as c:
        exp = train(c, df, "label", "random_forest_classifier")
        r = save(c, exp["id"])
        assert r.status_code == 201, r.text
        m = r.json()
        assert m["id"].startswith("mdl_") and m["version"] == 1 and m["status"] == "ready"
        assert m["owner"] == "local" and m["algorithm_name"] == "Random Forest"
        assert "path" not in json.dumps(m).lower().replace("split", "")  # no storage paths exposed
        api_preds = c.post(f"/api/experiments/{exp['id']}/predict", json={"rows": rows}).json()["predictions"]
        model_id = m["id"]

    # "refresh": a brand-new app instance over the same storage
    with make_client(settings) as c:
        listed = c.get("/api/models").json()["models"]
        assert [x["id"] for x in listed] == [model_id]
        detail = c.get(f"/api/models/{model_id}").json()
        man = detail["manifest"]
        assert man["target"] == "label" and man["task"] == "classification"
        assert man["algorithm"]["hyperparameters"]["n_estimators"] == 200
        assert [col["name"] for col in man["input_schema"]["columns"]] == ["f1", "f2", "f3", "f4", "colour"]
        assert man["input_schema"]["columns"][0]["dtype"] == "float64"
        assert man["preprocessing"]["outliers"] == "clip_iqr"
        assert man["training"]["random_state"] == 42 and man["training"]["test_size"] == 0.2
        assert man["dataset"]["sha256"] and man["environment"]["scikit_learn"]
        assert man["metrics"]["test"]["accuracy"] == exp["results"]["metrics"]["test"]["accuracy"]
        assert c.get(f"/api/models/{model_id}/verify").json()["valid"] is True

        for fmt in ("joblib", "pkl"):
            d = c.get(f"/api/models/{model_id}/download", params={"format": fmt})
            assert d.status_code == 200 and d.headers["content-disposition"].endswith(f'.{fmt}"')
            sha = hashlib.sha256(d.content).hexdigest()
            assert sha == d.headers["x-content-sha256"] == detail["artifacts"][fmt]["sha256"]
            path = tmp_path / f"downloaded.{fmt}"
            path.write_bytes(d.content)
            got = run_isolated(path, sha, rows, tmp_path / fmt)
            assert got["manifest_target"] == "label"
            assert [p["prediction"] for p in got["preds"]] == [p["prediction"] for p in api_preds]
            for a, b in zip(got["preds"], api_preds):
                assert a["probabilities"] == pytest.approx(b["probabilities"])

        served = c.post(f"/api/models/{model_id}/predict", json={"rows": rows}).json()
        assert [p["prediction"] for p in served["predictions"]] == [p["prediction"] for p in api_preds]
        manifest = c.get(f"/api/models/{model_id}/manifest")
        assert manifest.status_code == 200 and manifest.json()["artifacts"]["joblib"]["sha256"] == detail["artifacts"]["joblib"]["sha256"]


def test_regression_model_roundtrip(client, tmp_path):
    df = make_regression(n=250, seed=3)
    exp = train(client, df, "target", "ridge")
    m = save(client, exp["id"], name="Ridge v").json()
    d = client.get(f"/api/models/{m['id']}/download")
    p = tmp_path / "m.joblib"
    p.write_bytes(d.content)
    rows = json.loads(df.drop(columns=["target"]).head(3).to_json(orient="records"))
    got = run_isolated(p, m["artifacts"]["joblib"]["sha256"], rows, tmp_path)
    api = client.post(f"/api/models/{m['id']}/predict", json={"rows": rows}).json()["predictions"]
    assert [x["prediction"] for x in got["preds"]] == pytest.approx([x["prediction"] for x in api])


def test_deleting_experiment_keeps_saved_model(client, settings):
    exp = train(client, make_classification(n=200, seed=4), "label", "logistic_regression")
    m = save(client, exp["id"]).json()
    exp_artifact = settings.models_dir / f"{exp['id']}.joblib"
    assert exp_artifact.exists()
    r = client.delete(f"/api/experiments/{exp['id']}")
    assert r.status_code == 200 and r.json()["saved_models_retained"] == 1
    assert client.get(f"/api/experiments/{exp['id']}").status_code == 404
    assert not exp_artifact.exists()
    assert not list(settings.models_dir.glob(".trash-*"))
    assert exp["id"] not in {e["id"] for e in client.get("/api/experiments").json()["experiments"]}
    # saved model unaffected
    detail = client.get(f"/api/models/{m['id']}").json()
    assert detail["status"] == "ready" and detail["source_experiment_exists"] is False
    assert client.get(f"/api/models/{m['id']}/download").status_code == 200
    rows = json.loads(make_classification(n=5, seed=9).drop(columns=["label"]).to_json(orient="records"))
    assert client.post(f"/api/models/{m['id']}/predict", json={"rows": rows}).status_code == 200


def test_delete_saved_model_removes_record_and_files(client, settings):
    exp = train(client, make_classification(n=200, seed=5), "label", "logistic_regression")
    m = save(client, exp["id"]).json()
    model_dir = settings.saved_models_dir / m["id"]
    assert model_dir.is_dir()
    assert client.delete(f"/api/models/{m['id']}").status_code == 200
    assert not model_dir.exists() and not list(settings.saved_models_dir.glob(".trash-*"))
    assert client.get(f"/api/models/{m['id']}").status_code == 404
    assert client.delete(f"/api/models/{m['id']}").status_code == 404
    # experiment untouched
    assert client.get(f"/api/experiments/{exp['id']}").json()["experiment"]["status"] == "completed"


def test_versions_increment_per_name(client):
    exp = train(client, make_classification(n=200, seed=6), "label", "logistic_regression")
    a = save(client, exp["id"], name="Churn Model").json()
    b = save(client, exp["id"], name="churn-model").json()  # same lineage after normalisation
    c_ = save(client, exp["id"], name="Other").json()
    assert (a["version"], b["version"], c_["version"]) == (1, 2, 1)
    assert a["lineage"] == b["lineage"] == "churn-model" and a["id"] != b["id"]
    versions = client.get(f"/api/models/{b['id']}").json()["versions"]
    assert [v["version"] for v in versions] == [1, 2]
    assert len(client.get(f"/api/experiments/{exp['id']}/saved-models").json()["models"]) == 3


def test_save_validation_errors(client, settings):
    assert save(client, "missing").status_code == 404
    exp = train(client, make_classification(n=200, seed=7), "label", "logistic_regression")
    assert client.post(f"/api/experiments/{exp['id']}/save-model", json={"name": ""}).status_code == 422
    assert client.post(f"/api/experiments/{exp['id']}/save-model", json={"name": "x" * 121}).status_code == 422
    assert client.post(f"/api/experiments/{exp['id']}/save-model", json={"name": "   "}).status_code == 400
    # failed experiment cannot be saved
    df = make_classification(n=60, seed=8)
    df["label"] = [f"c{i % 10}" for i in range(60)]
    ds = upload(client, df)
    bad = wait(client, client.post("/api/experiments", json={"dataset_id": ds, "target": "label", "algorithms": [{"id": "logistic_regression"}],
                                                              "split": {"test_size": 0.1}}).json()["experiments"][0]["id"])
    assert bad["status"] == "failed"
    assert save(client, bad["id"]).status_code == 409
    # missing working artifact: refused, nothing recorded
    (settings.models_dir / f"{exp['id']}.joblib").unlink()
    r = save(client, exp["id"])
    assert r.status_code == 409 and "missing" in r.json()["detail"]
    assert client.get("/api/models").json()["models"] == []
    assert not list(settings.saved_models_dir.glob(".tmp-*"))


def test_tampered_or_missing_artifact_is_refused(client, settings):
    exp = train(client, make_classification(n=200, seed=10), "label", "logistic_regression")
    m = save(client, exp["id"]).json()
    art = settings.saved_models_dir / m["id"] / "model.joblib"
    data = bytearray(art.read_bytes())
    data[-5] ^= 0xFF  # same size, different content
    art.write_bytes(bytes(data))
    assert client.get(f"/api/models/{m['id']}/download").status_code == 409
    assert client.post(f"/api/models/{m['id']}/predict", json={"rows": [{}]}).status_code == 409
    assert client.get(f"/api/models/{m['id']}/verify").json()["artifacts"]["joblib"].startswith("Integrity")
    # the pkl copy is still intact and downloadable
    assert client.get(f"/api/models/{m['id']}/download", params={"format": "pkl"}).status_code == 200
    (settings.saved_models_dir / m["id"] / "model.pkl").unlink()
    assert client.get(f"/api/models/{m['id']}").json()["status"] == "missing"
    assert client.get(f"/api/models/{m['id']}/download", params={"format": "pkl"}).status_code == 410
    assert client.get(f"/api/models/{m['id']}/download", params={"format": "exe"}).status_code == 422


def test_experiment_artifact_integrity_checked_before_loading(client, settings):
    exp = train(client, make_classification(n=200, seed=13), "label", "logistic_regression")
    p = settings.models_dir / f"{exp['id']}.joblib"
    p.write_bytes(p.read_bytes() + b"tampered")
    r = save(client, exp["id"])
    assert r.status_code == 409 and "integrity" in r.json()["detail"].lower()


def test_storage_failure_during_save_leaves_nothing_behind(client, settings, monkeypatch):
    exp = train(client, make_classification(n=200, seed=14), "label", "logistic_regression")
    import core.model_registry as mr
    real = os.replace

    def failing_replace(src, dst):
        if ".tmp-" in str(src):
            raise OSError("disk full (simulated)")
        return real(src, dst)
    monkeypatch.setattr(mr.os, "replace", failing_replace)
    r = save(client, exp["id"])
    assert r.status_code == 500 and "disk full" in r.json()["detail"]
    assert client.get("/api/models").json()["models"] == []
    assert not [p for p in settings.saved_models_dir.iterdir()]


def test_database_failure_during_delete_restores_files(client, settings, monkeypatch):
    exp = train(client, make_classification(n=200, seed=15), "label", "logistic_regression")
    m = save(client, exp["id"]).json()
    store = client.app_ref.state.store

    def boom(_id):
        raise __import__("sqlite3").OperationalError("database is locked (simulated)")
    monkeypatch.setattr(store, "delete_saved_model", boom)
    r = client.delete(f"/api/models/{m['id']}")
    assert r.status_code == 500
    monkeypatch.undo()
    assert (settings.saved_models_dir / m["id"] / "model.joblib").exists()
    assert client.get(f"/api/models/{m['id']}/download").status_code == 200


def test_experiment_delete_db_failure_restores_artifact(client, settings, monkeypatch):
    exp = train(client, make_classification(n=200, seed=16), "label", "logistic_regression")
    store = client.app_ref.state.store
    monkeypatch.setattr(store, "delete_experiment", lambda _id: (_ for _ in ()).throw(RuntimeError("db down")))
    r = client.delete(f"/api/experiments/{exp['id']}")
    assert r.status_code == 500
    monkeypatch.undo()
    assert (settings.models_dir / f"{exp['id']}.joblib").exists()
    assert client.get(f"/api/experiments/{exp['id']}").status_code == 200


def test_deleting_running_experiment_leaves_no_orphan_artifact(client, settings):
    rng = np.random.default_rng(17)
    df = pd.DataFrame(rng.normal(size=(3000, 15)), columns=[f"x{i}" for i in range(15)])
    df["y"] = (df.x0 > 0).map({True: "a", False: "b"})
    ds = upload(client, df)
    eid = client.post("/api/experiments", json={"dataset_id": ds, "target": "y", "algorithms": [{"id": "svc"}],
                                                "split": {"cv_folds": 10}}).json()["experiments"][0]["id"]
    import time
    manager = client.app_ref.state.manager
    for _ in range(300):  # delete while the job is genuinely running, not merely queued
        if manager.jobs[eid].status == "running":
            break
        time.sleep(0.02)
    assert manager.jobs[eid].status == "running"
    assert client.delete(f"/api/experiments/{eid}").status_code == 200
    for _ in range(600):
        if manager.active_count() == 0:
            break
        time.sleep(0.1)
    assert manager.active_count() == 0
    assert not (settings.models_dir / f"{eid}.joblib").exists()
    assert client.get(f"/api/experiments/{eid}").status_code == 404


def test_model_ids_cannot_traverse_paths(client):
    for bad in ("..%2F..%2Fetc", "mdl_../../x", "mdl_zzzzzzzzzzzzzzzz"):
        assert client.get(f"/api/models/{bad}/download").status_code == 404


def test_pickle_uploads_are_not_accepted(client):
    import pickle
    payload = pickle.dumps({"pipeline": None})
    r = client.post("/api/datasets", files={"file": ("model.pkl", io.BytesIO(payload), "application/octet-stream")})
    assert r.status_code == 422
    r = client.post("/api/datasets", files={"file": ("model.joblib", io.BytesIO(payload), "application/octet-stream")})
    assert r.status_code == 422


# ------------------------------------------------------------------ owner tokens
@pytest.fixture
def auth_settings(settings):
    settings.access_tokens = "alice:alice-token-0123456789,bob:bob-token-0123456789ab"
    return settings


def test_owner_tokens_scope_models_and_protect_destructive_endpoints(auth_settings):
    A = {"Authorization": "Bearer alice-token-0123456789"}
    B = {"Authorization": "Bearer bob-token-0123456789ab"}
    with make_client(auth_settings) as c:
        assert c.get("/api/system").json()["auth"]["mode"] == "tokens"
        exp = train(c, make_classification(n=200, seed=18), "label", "logistic_regression")
        # unauthenticated / wrong token
        r = save(c, exp["id"])
        assert r.status_code == 401 and r.headers.get("www-authenticate") == "Bearer"
        assert save(c, exp["id"], headers={"Authorization": "Bearer nope"}).status_code == 401
        m = save(c, exp["id"], headers=A).json()
        assert m["owner"] == "alice"
        assert [x["id"] for x in c.get("/api/models", headers=A).json()["models"]] == [m["id"]]
        assert c.get("/api/models", headers=B).json()["models"] == []
        for method, path in (("get", f"/api/models/{m['id']}"), ("get", f"/api/models/{m['id']}/download"),
                             ("post", f"/api/models/{m['id']}/predict"), ("delete", f"/api/models/{m['id']}")):
            kw = {"json": {"rows": [{}]}} if method == "post" else {}
            assert getattr(c, method)(path, headers=B, **kw).status_code == 404  # not leaked to bob
            assert getattr(c, method)(path, **kw).status_code == 401
        assert c.get(f"/api/experiments/{exp['id']}/export-model").status_code == 401
        assert c.get(f"/api/experiments/{exp['id']}/export-model", headers=A).status_code == 200
        assert c.delete(f"/api/experiments/{exp['id']}").status_code == 401
        assert c.get("/api/auth/whoami", headers=B).json()["owner"] == "bob"
        assert c.delete(f"/api/models/{m['id']}", headers=A).status_code == 200


def test_weak_tokens_rejected_at_startup(settings):
    from core.auth import AuthConfigError
    settings.access_tokens = "alice:short"
    with pytest.raises(AuthConfigError):
        make_client(settings)


def test_job_finishing_after_its_experiment_was_deleted_removes_artifact(settings):
    import time

    from core.jobs import JobManager
    from core.store import Store
    store = Store(settings.db_path)
    store.save_experiment({"id": "exp-x", "name": "x", "algorithm": "a", "dataset_path": "", "config": {},
                           "status": "queued", "metrics": {}, "logs": [], "created_at": "2026-01-01T00:00:00+00:00"})
    manager = JobManager(store, settings.models_dir, 1)

    def work(job):
        store.delete_experiment("exp-x")  # user deletes while training is in flight
        return {"bundle": {"pipeline": "fitted"}, "results": {}, "summary": {}}
    manager.submit("exp-x", "fp", work)
    for _ in range(100):
        if manager.jobs["exp-x"].status not in ("queued", "running"):
            break
        time.sleep(0.02)
    manager.shutdown()
    assert not (settings.models_dir / "exp-x.joblib").exists()
    assert not list(settings.models_dir.glob("*.partial"))
    assert store.get_experiment("exp-x") is None
