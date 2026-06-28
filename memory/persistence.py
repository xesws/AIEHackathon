"""Small local persistence helpers shared by memory and serving modules."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any


def data_dir(data_dir: str | os.PathLike | None = None) -> Path:
    """Return the Engram local data directory.

    ``ENGRAM_DATA_DIR`` overrides the repo-local ``data/`` default. The default
    is already gitignored by this project.
    """
    if data_dir is not None:
        return Path(data_dir)
    env = os.environ.get("ENGRAM_DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "data"


def atomic_write_json(path: str | os.PathLike, payload: Any) -> None:
    """Atomically write JSON to ``path`` using a sibling temporary file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, p)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def load_json(path: str | os.PathLike) -> Any:
    """Read JSON from ``path``."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
