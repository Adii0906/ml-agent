"""SQLite persistence for datasets, experiments and settings.

Each call opens its own connection so the store is safe to use from the
training worker threads as well as the asyncio event loop.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .util import clean_json, utcnow

# Columns that hold JSON payloads.
_EXP_JSON = ("config", "metrics", "logs", "results")

_EXPERIMENT_COLUMNS = {
    # legacy columns (kept for backwards compatibility with existing databases)
    "id": "TEXT PRIMARY KEY",
    "name": "TEXT",
    "algorithm": "TEXT",
    "dataset_path": "TEXT",
    "config": "TEXT",
    "status": "TEXT",
    "metrics": "TEXT",
    "logs": "TEXT",
    "training_time": "REAL",
    "created_at": "TEXT",
    "completed_at": "TEXT",
    "error": "TEXT",
    # added in v2
    "dataset_id": "TEXT",
    "algorithm_id": "TEXT",
    "task_type": "TEXT",
    "target": "TEXT",
    "group_id": "TEXT",
    "results": "TEXT",
    "started_at": "TEXT",
    "fingerprint": "TEXT",
}


class Store:
    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._init()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init(self) -> None:
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute(
                """CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY, name TEXT, filename TEXT, path TEXT,
                    sha256 TEXT, size_bytes INTEGER, n_rows INTEGER, n_cols INTEGER,
                    created_at TEXT, last_opened_at TEXT, parent_id TEXT, note TEXT)"""
            )
            c.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
            c.execute(
                """CREATE TABLE IF NOT EXISTS saved_models (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, name TEXT NOT NULL, description TEXT,
                    lineage TEXT NOT NULL, version INTEGER NOT NULL,
                    experiment_id TEXT, algorithm_id TEXT, algorithm_name TEXT, task TEXT, target TEXT,
                    dataset_id TEXT, dataset_name TEXT, dataset_sha256 TEXT,
                    primary_metric TEXT, primary_value REAL, metrics TEXT, manifest TEXT, artifacts TEXT,
                    trained_at TEXT, created_at TEXT NOT NULL,
                    UNIQUE (owner, lineage, version))"""
            )
            c.execute("CREATE INDEX IF NOT EXISTS ix_saved_models_owner ON saved_models (owner, created_at)")
            cols = ", ".join(f"{k} {v}" for k, v in _EXPERIMENT_COLUMNS.items())
            c.execute(f"CREATE TABLE IF NOT EXISTS experiments ({cols})")
            existing = {r["name"] for r in c.execute("PRAGMA table_info(experiments)")}
            for name, decl in _EXPERIMENT_COLUMNS.items():
                if name not in existing:
                    c.execute(f"ALTER TABLE experiments ADD COLUMN {name} {decl.replace('PRIMARY KEY', '')}")

    # ------------------------------------------------------------ datasets
    def add_dataset(self, row: Dict[str, Any]) -> None:
        keys = ["id", "name", "filename", "path", "sha256", "size_bytes", "n_rows", "n_cols",
                "created_at", "last_opened_at", "parent_id", "note"]
        with self._conn() as c:
            c.execute(
                f"INSERT INTO datasets ({', '.join(keys)}) VALUES ({', '.join('?' * len(keys))})",
                [row.get(k) for k in keys],
            )

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
            return dict(r) if r else None

    def find_dataset_by_path(self, path: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM datasets WHERE path = ?", (path,)).fetchone()
            return dict(r) if r else None

    def find_dataset_by_sha(self, sha256: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM datasets WHERE sha256 = ? AND parent_id IS NULL ORDER BY created_at LIMIT 1",
                          (sha256,)).fetchone()
            return dict(r) if r else None

    def list_datasets(self) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM datasets ORDER BY COALESCE(last_opened_at, created_at) DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def touch_dataset(self, dataset_id: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE datasets SET last_opened_at = ? WHERE id = ?", (utcnow(), dataset_id))

    def rename_dataset(self, dataset_id: str, name: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE datasets SET name = ? WHERE id = ?", (name, dataset_id))

    def delete_dataset(self, dataset_id: str) -> bool:
        with self._conn() as c:
            return c.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,)).rowcount > 0

    # --------------------------------------------------------- experiments
    @staticmethod
    def _decode(row: sqlite3.Row) -> Dict[str, Any]:
        exp = dict(row)
        for key in _EXP_JSON:
            raw = exp.get(key)
            default: Any = [] if key == "logs" else {}
            try:
                exp[key] = json.loads(raw) if raw else default
            except (TypeError, ValueError):
                exp[key] = default
        return exp

    def save_experiment(self, exp: Dict[str, Any]) -> None:
        payload = dict(exp)
        for key in _EXP_JSON:
            if key in payload:
                payload[key] = json.dumps(clean_json(payload[key]))
        cols = [k for k in _EXPERIMENT_COLUMNS if k in payload]
        with self._lock, self._conn() as c:
            c.execute(
                f"INSERT OR REPLACE INTO experiments ({', '.join(cols)}) "
                f"VALUES ({', '.join('?' * len(cols))})",
                [payload[k] for k in cols],
            )

    def update_experiment(self, exp_id: str, **fields: Any) -> None:
        if not fields:
            return
        sets, values = [], []
        for key, value in fields.items():
            if key not in _EXPERIMENT_COLUMNS:
                raise KeyError(key)
            sets.append(f"{key} = ?")
            values.append(json.dumps(clean_json(value)) if key in _EXP_JSON else value)
        values.append(exp_id)
        with self._lock, self._conn() as c:
            c.execute(f"UPDATE experiments SET {', '.join(sets)} WHERE id = ?", values)

    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
            return self._decode(r) if r else None

    def list_experiments(self, include_results: bool = False) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()
        out = []
        for r in rows:
            exp = self._decode(r)
            if not include_results:
                exp.pop("results", None)
                exp["logs"] = []
            out.append(exp)
        return out

    def experiments_with_status(self, statuses: Iterable[str]) -> List[Dict[str, Any]]:
        statuses = list(statuses)
        with self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM experiments WHERE status IN ({', '.join('?' * len(statuses))})",
                statuses,
            ).fetchall()
            return [self._decode(r) for r in rows]

    def delete_experiment(self, exp_id: str) -> bool:
        with self._lock, self._conn() as c:
            return c.execute("DELETE FROM experiments WHERE id = ?", (exp_id,)).rowcount > 0

    # ---------------------------------------------------------- saved models
    _MODEL_JSON = ("metrics", "manifest", "artifacts")

    def _decode_model(self, row: sqlite3.Row) -> Dict[str, Any]:
        m = dict(row)
        for key in self._MODEL_JSON:
            try:
                m[key] = json.loads(m[key]) if m.get(key) else {}
            except (TypeError, ValueError):
                m[key] = {}
        return m

    def insert_saved_model(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Insert with the next version number of (owner, lineage), atomically."""
        payload = {k: (json.dumps(clean_json(v)) if k in self._MODEL_JSON else v) for k, v in row.items()}
        with self._lock, self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            cur = c.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM saved_models WHERE owner = ? AND lineage = ?",
                            (payload["owner"], payload["lineage"])).fetchone()[0]
            payload["version"] = int(payload.get("version") or cur)
            cols = list(payload)
            c.execute(f"INSERT INTO saved_models ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                      [payload[k] for k in cols])
        return self.get_saved_model(payload["id"])

    def next_model_version(self, owner: str, lineage: str) -> int:
        with self._conn() as c:
            return int(c.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM saved_models WHERE owner = ? AND lineage = ?",
                                 (owner, lineage)).fetchone()[0])

    def get_saved_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM saved_models WHERE id = ?", (model_id,)).fetchone()
            return self._decode_model(r) if r else None

    def list_saved_models(self, owner: Optional[str] = None) -> List[Dict[str, Any]]:
        q, args = "SELECT * FROM saved_models", []
        if owner is not None:
            q, args = q + " WHERE owner = ?", [owner]
        with self._conn() as c:
            return [self._decode_model(r) for r in c.execute(q + " ORDER BY created_at DESC, version DESC", args).fetchall()]

    def models_for_experiment(self, exp_id: str) -> List[Dict[str, Any]]:
        with self._conn() as c:
            return [self._decode_model(r) for r in
                    c.execute("SELECT * FROM saved_models WHERE experiment_id = ?", (exp_id,)).fetchall()]

    def update_saved_model(self, model_id: str, **fields: Any) -> None:
        allowed = {"name", "description"}
        if set(fields) - allowed:
            raise KeyError(", ".join(set(fields) - allowed))
        with self._lock, self._conn() as c:
            c.execute(f"UPDATE saved_models SET {', '.join(f'{k} = ?' for k in fields)} WHERE id = ?",
                      [*fields.values(), model_id])

    def delete_saved_model(self, model_id: str) -> bool:
        with self._lock, self._conn() as c:
            return c.execute("DELETE FROM saved_models WHERE id = ?", (model_id,)).rowcount > 0

    # ------------------------------------------------------------ settings
    def get_settings(self) -> Dict[str, Any]:
        with self._conn() as c:
            rows = c.execute("SELECT key, value FROM settings").fetchall()
        out = {}
        for r in rows:
            try:
                out[r["key"]] = json.loads(r["value"])
            except ValueError:
                out[r["key"]] = r["value"]
        return out

    def set_settings(self, values: Dict[str, Any]) -> None:
        with self._conn() as c:
            for k, v in values.items():
                c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (k, json.dumps(v)))
