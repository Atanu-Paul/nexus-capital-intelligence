"""Project-root paths for RAG resources (cwd-independent)."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def chroma_db_dir() -> Path:
    """Return the persistent ChromaDB directory."""
    override = os.getenv("CHROMA_DB_DIR")
    if override:
        return Path(override)
    return PROJECT_ROOT / "chroma_db"


def synthetic_data_dir() -> Path:
    """Return the default synthetic data directory."""
    return PROJECT_ROOT / "synthetic_data"
