"""Runtime configuration and filesystem layout."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BACKEND_DIR.parent

for candidate in (ROOT_DIR / ".env", BACKEND_DIR / ".env"):
    if candidate.exists():
        load_dotenv(candidate, override=False)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


class Settings:
    """Resolved once at import; tests can construct their own with a temp dir."""

    def __init__(self, data_dir: str | os.PathLike | None = None):
        base = Path(data_dir or os.getenv("MLAGENT_DATA_DIR") or (BACKEND_DIR / "var"))
        self.data_dir = base.resolve()
        self.datasets_dir = self.data_dir / "datasets"
        self.models_dir = self.data_dir / "models"            # per-experiment artifacts (working copies)
        self.saved_models_dir = self.data_dir / "saved_models"  # user-saved, versioned model packages
        self.reports_dir = self.data_dir / "reports"
        self.db_path = self.data_dir / "ml_agent.db"
        # Legacy upload directory (old /upload-dataset API wrote here).
        self.legacy_upload_dir = BACKEND_DIR / "uploads"
        self.max_upload_mb = _int_env("MLAGENT_MAX_UPLOAD_MB", 200)
        self.max_concurrent_jobs = max(1, _int_env("MLAGENT_MAX_JOBS", 2))
        self.max_model_mb = _int_env("MLAGENT_MAX_MODEL_MB", 1024)
        self.access_tokens = os.getenv("MLAGENT_ACCESS_TOKENS", "")
        for d in (self.data_dir, self.datasets_dir, self.models_dir, self.saved_models_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)


SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".parquet"}
