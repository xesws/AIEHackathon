"""Embedding seam: sentence-transformer encoding plus a manual cosine helper.

The model is a module-level LAZY singleton — it is constructed on the first
``encode`` call, never at import time (import must do no model load / no GPU).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None  # populated on first encode() call


def _get_model():
    """Return the shared ``SentenceTransformer``, loading it on first use."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def encode(texts: Sequence[str]) -> list[list[float]]:
    """Embed ``texts`` into a list of plain-python float vectors.

    A single ``str`` is accepted defensively and wrapped into one batch, but the
    contract is a sequence. The device is left to ``SentenceTransformer`` default.
    """
    if isinstance(texts, str):
        texts = [texts]
    else:
        texts = list(texts)
    if not texts:
        return []
    vectors = _get_model().encode(texts)
    return vectors.tolist()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity ``dot(a, b) / (||a|| * ||b||)``; ``0.0`` on a zero norm."""
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))
