"""Dataset loading, type inference and the dataset registry."""
from __future__ import annotations

import io
import shutil
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import SUPPORTED_EXTENSIONS, Settings
from .store import Store
from .util import safe_filename, sha256_file, utcnow


class DatasetError(ValueError):
    """User-facing dataset problem (bad file, empty data, unknown column...)."""


# ---------------------------------------------------------------- reading
def read_table(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DatasetError(f"Unsupported file type '{ext}'. Upload a CSV, TSV or Parquet file.")
    if ext == ".parquet":
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # pyarrow raises many exception types
            raise DatasetError(f"Could not read Parquet file: {exc}") from exc
    else:
        sep = "\t" if ext == ".tsv" else ","
        df = None
        last_err: Optional[Exception] = None
        for enc in ("utf-8-sig", "latin1"):
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                if df.shape[1] == 1 and any(d in str(df.columns[0]) for d in (";", "\t", "|")):
                    # Comma parse gave one column; sniff the real delimiter.
                    df = pd.read_csv(path, encoding=enc, sep=None, engine="python")
                break
            except UnicodeDecodeError as exc:
                last_err = exc
                continue
            except pd.errors.EmptyDataError as exc:
                raise DatasetError("The file is empty.") from exc
            except pd.errors.ParserError as exc:
                raise DatasetError(f"The file is not valid CSV: {exc}") from exc
        if df is None:
            raise DatasetError(f"Could not decode the file as text ({last_err}).")
    if df.shape[1] == 0:
        raise DatasetError("The file has no columns.")
    df.columns = [str(c) for c in df.columns]
    if len(set(df.columns)) != len(df.columns):
        raise DatasetError("Column names must be unique.")
    return df


# --------------------------------------------------------- type inference
def _is_string_like(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(
        s.dtype, pd.CategoricalDtype
    )


def numeric_coercible_share(s: pd.Series) -> float:
    """Share of non-missing values that parse as numbers (for 'number stored as text')."""
    non_null = s.dropna()
    if non_null.empty:
        return 0.0
    sample = non_null.sample(min(len(non_null), 5000), random_state=0) if len(non_null) > 5000 else non_null
    parsed = pd.to_numeric(sample.astype(str).str.strip().str.replace(",", "", regex=False), errors="coerce")
    return float(parsed.notna().mean())


def infer_column_kind(s: pd.Series, n_rows: int) -> Tuple[str, str]:
    """Return (kind, reason). Kinds: numeric, boolean, categorical, datetime, text,
    identifier, constant, empty."""
    non_null = s.dropna()
    n_unique = int(non_null.nunique())
    if non_null.empty:
        return "empty", "All values are missing."
    if n_unique == 1:
        return "constant", "Only one distinct value."
    if pd.api.types.is_bool_dtype(s):
        return "boolean", "Boolean values."
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime", "Datetime values."
    if pd.api.types.is_numeric_dtype(s):
        if pd.api.types.is_integer_dtype(s) and n_unique == n_rows and n_rows >= 20:
            sorted_vals = np.sort(non_null.to_numpy())
            diffs = np.diff(sorted_vals)
            if diffs.size and np.all(diffs == 1):
                return "identifier", "Unique consecutive integers (looks like a row ID)."
        return "numeric", "Numeric values."
    if _is_string_like(s):
        if n_unique == len(non_null) and n_rows >= 20:
            avg_len = non_null.astype(str).str.len().mean()
            if avg_len > 40:
                return "text", "Unique long strings (free text)."
            return "identifier", "Every value is unique (looks like an identifier)."
        if n_unique <= 2 and set(non_null.astype(str).str.lower().unique()) <= {"true", "false"}:
            return "boolean", "true/false values."
        avg_len = non_null.astype(str).str.len().mean()
        if avg_len > 60 and n_unique > 0.5 * len(non_null):
            return "text", "Long free-text values."
        sample = non_null.astype(str).head(200)
        if numeric_coercible_share(non_null) < 0.5:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.95:
                return "datetime", "Values parse as dates."
        return "categorical", "Discrete text values."
    return "categorical", f"Treated as categorical (dtype {s.dtype})."


def column_kinds(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    n = len(df)
    return {c: dict(zip(("kind", "reason"), infer_column_kind(df[c], n))) for c in df.columns}


# MODELABLE kinds can feed the preprocessing pipeline directly.
MODELABLE_KINDS = {"numeric", "boolean", "categorical"}
# Kinds excluded by default (user can override for 'identifier').
DEFAULT_EXCLUDED_KINDS = {"identifier", "text", "datetime", "constant", "empty"}


# --------------------------------------------------------------- registry
class DatasetRegistry:
    def __init__(self, settings: Settings, store: Store, cache_size: int = 4):
        self.settings = settings
        self.store = store
        self._cache: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    # -- registration
    def register_upload(self, filename: str, fileobj: io.BufferedIOBase) -> Dict[str, Any]:
        ext = Path(filename or "").suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise DatasetError(f"Unsupported file type '{ext or 'none'}'. Upload a CSV, TSV or Parquet file.")
        dataset_id = uuid.uuid4().hex[:12]
        dest = self.settings.datasets_dir / f"{dataset_id}{ext}"
        limit = self.settings.max_upload_mb * 1024 * 1024
        written = 0
        with dest.open("wb") as out:
            while True:
                chunk = fileobj.read(1 << 20)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise DatasetError(f"File exceeds the {self.settings.max_upload_mb} MB upload limit.")
                out.write(chunk)
        return self._finalize(dataset_id, dest, filename)

    def register_existing_file(self, path: Path, display_name: Optional[str] = None) -> Dict[str, Any]:
        """Copy a server-side file (sample data / legacy uploads) into the registry."""
        existing = self.store.find_dataset_by_path(str(path)) or self.store.find_dataset_by_sha(sha256_file(path))
        if existing:
            return existing
        dataset_id = uuid.uuid4().hex[:12]
        dest = self.settings.datasets_dir / f"{dataset_id}{path.suffix.lower()}"
        shutil.copyfile(path, dest)
        meta = self._finalize(dataset_id, dest, display_name or path.name)
        return meta

    def register_dataframe(self, df: pd.DataFrame, name: str, parent_id: Optional[str], note: str) -> Dict[str, Any]:
        dataset_id = uuid.uuid4().hex[:12]
        dest = self.settings.datasets_dir / f"{dataset_id}.csv"
        df.to_csv(dest, index=False)
        return self._finalize(dataset_id, dest, name, parent_id=parent_id, note=note)

    def _finalize(self, dataset_id: str, dest: Path, filename: str, parent_id=None, note=None) -> Dict[str, Any]:
        try:
            df = read_table(dest)
        except DatasetError:
            dest.unlink(missing_ok=True)
            raise
        if len(df) == 0:
            dest.unlink(missing_ok=True)
            raise DatasetError("The file has a header but no data rows.")
        name = Path(filename or dest.name).stem or dataset_id
        row = {
            "id": dataset_id,
            "name": name,
            "filename": safe_filename(filename or dest.name),
            "path": str(dest),
            "sha256": sha256_file(dest),
            "size_bytes": dest.stat().st_size,
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "created_at": utcnow(),
            "last_opened_at": utcnow(),
            "parent_id": parent_id,
            "note": note,
        }
        self.store.add_dataset(row)
        with self._lock:
            self._remember(dataset_id, df)
        return row

    # -- access
    def meta(self, dataset_id: str) -> Dict[str, Any]:
        meta = self.store.get_dataset(dataset_id)
        if not meta:
            raise KeyError(dataset_id)
        return meta

    def load(self, dataset_id: str) -> pd.DataFrame:
        """Return the cached DataFrame. Callers must not mutate it."""
        with self._lock:
            if dataset_id in self._cache:
                self._cache.move_to_end(dataset_id)
                return self._cache[dataset_id]
        meta = self.meta(dataset_id)
        path = Path(meta["path"])
        if not path.exists():
            raise DatasetError("The dataset file is missing on disk.")
        df = read_table(path)
        with self._lock:
            self._remember(dataset_id, df)
        return df

    def _remember(self, dataset_id: str, df: pd.DataFrame) -> None:
        self._cache[dataset_id] = df
        self._cache.move_to_end(dataset_id)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def list(self) -> List[Dict[str, Any]]:
        return self.store.list_datasets()

    def delete(self, dataset_id: str) -> bool:
        meta = self.store.get_dataset(dataset_id)
        if not meta:
            return False
        with self._lock:
            self._cache.pop(dataset_id, None)
        path = Path(meta["path"])
        if path.is_file() and path.parent == self.settings.datasets_dir:
            path.unlink(missing_ok=True)
        return self.store.delete_dataset(dataset_id)

    # -- consent-based cleaning: always produces a NEW dataset version
    def apply_actions(self, dataset_id: str, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        df = self.load(dataset_id).copy()
        applied: List[str] = []
        for action in actions:
            kind = action.get("type")
            params = action.get("params") or {}
            if kind == "drop_duplicates":
                before = len(df)
                subset = [c for c in params.get("subset") or [] if c in df.columns] or None
                df = df.drop_duplicates(subset=subset).reset_index(drop=True)
                applied.append(f"Dropped {before - len(df)} duplicate rows"
                               + (f" (ignoring {', '.join(c for c in df.columns if c not in subset)})" if subset else ""))
            elif kind == "drop_columns":
                cols = [c for c in params.get("columns", []) if c in df.columns]
                if not cols:
                    raise DatasetError("drop_columns: none of the listed columns exist.")
                df = df.drop(columns=cols)
                applied.append(f"Dropped columns: {', '.join(cols)}")
            elif kind == "drop_missing_rows":
                cols = [c for c in params.get("columns", []) if c in df.columns]
                if not cols:
                    raise DatasetError("drop_missing_rows: none of the listed columns exist.")
                before = len(df)
                df = df.dropna(subset=cols).reset_index(drop=True)
                applied.append(f"Dropped {before - len(df)} rows missing {', '.join(cols)}")
            elif kind == "cast_numeric":
                cols = [c for c in params.get("columns", []) if c in df.columns]
                for c in cols:
                    df[c] = pd.to_numeric(
                        df[c].astype(str).str.strip().str.replace(",", "", regex=False), errors="coerce"
                    )
                applied.append(f"Converted to numeric: {', '.join(cols)} (unparseable values become missing)")
            else:
                raise DatasetError(f"Unknown action '{kind}'.")
        if df.empty or df.shape[1] == 0:
            raise DatasetError("These actions would leave the dataset empty.")
        parent = self.meta(dataset_id)
        return self.register_dataframe(
            df, f"{parent['name']} (cleaned)", parent_id=dataset_id, note="; ".join(applied)
        )
