"""Small shared helpers."""
from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def clean_json(obj: Any) -> Any:
    """Recursively convert numpy/pandas values into JSON-safe Python values.

    NaN and +/-inf become None so the browser never receives invalid JSON.
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [clean_json(v) for v in obj.tolist()]
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if obj is pd.NaT:
        return None
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)


def safe_filename(name: str, fallback: str = "file") -> str:
    keep = "".join(c if c.isalnum() or c in "-_." else "_" for c in (name or ""))
    keep = keep.strip("._")
    return keep[:120] or fallback
