"""The four evaluation conditions for the capability matrix."""
from __future__ import annotations

from typing import Literal, Tuple

Condition = Literal["base", "rag", "edit", "edit+rag"]

# The matrix is framed per-axis (a capability matrix), NOT as an accuracy winner.
CONDITIONS: Tuple[Condition, ...] = ("base", "rag", "edit", "edit+rag")

# TODO: per-condition setup — which of {edited weights, RAG window} are active — for run_matrix.
