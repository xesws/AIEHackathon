"""Capability-matrix runner — frames results as a per-axis capability matrix, NOT an accuracy winner.

Consumes ``editing.edit`` and ``generate.generate`` to evaluate each condition x probe-axis.
"""
from __future__ import annotations

from typing import Any

# import editing
# from generate import generate
# from . import conditions, dataset, metrics


def run_matrix(*, model: Any) -> dict:
    """Run every (condition x probe-axis) cell and return the capability matrix. TODO."""
    raise NotImplementedError
