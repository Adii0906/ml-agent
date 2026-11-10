import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import Settings  # noqa: E402


@pytest.fixture
def settings(tmp_path):
    return Settings(tmp_path / "var")


@pytest.fixture
def client(settings, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from fastapi.testclient import TestClient

    from main import create_app
    app = create_app(settings, static_dir=settings.data_dir / "no-static")
    with TestClient(app) as c:
        c.app_ref = app
        yield c


def make_classification(n=300, n_classes=2, seed=0, categorical=True, missing=0.0, weights=None):
    """Learnable synthetic classification data with a categorical feature."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    if weights is None:
        logits = np.column_stack([X @ rng.normal(size=4) * (k + 1) * 0.8 for k in range(n_classes)])
        y = logits.argmax(axis=1)
    else:
        y = rng.choice(n_classes, size=n, p=weights)
        X[:, 0] += y * 1.5
    df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
    if categorical:
        df["colour"] = np.where(y % 2 == 0, rng.choice(["red", "green"], n), rng.choice(["blue", "green"], n))
    if missing:
        mask = rng.random(n) < missing
        df.loc[mask, "f2"] = np.nan
    df["label"] = np.array([f"class_{k}" for k in range(n_classes)])[y]
    return df


def make_regression(n=300, seed=0, noise=0.5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["kind"] = rng.choice(["x", "y", "z"], n)
    df["target"] = 3 * df.a - 2 * df.b + df.kind.map({"x": 0, "y": 2, "z": -1}) + rng.normal(scale=noise, size=n)
    return df


def upload(client, df: pd.DataFrame, name="data.csv"):
    import io
    buf = io.BytesIO(df.to_csv(index=False).encode())
    r = client.post("/api/datasets", files={"file": (name, buf, "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def wait(client, exp_id, timeout=120):
    import time
    end = time.time() + timeout
    while time.time() < end:
        exp = client.get(f"/api/experiments/{exp_id}").json()["experiment"]
        if exp["status"] in ("completed", "failed", "cancelled"):
            return exp
        time.sleep(0.05)
    raise AssertionError("experiment did not finish")
