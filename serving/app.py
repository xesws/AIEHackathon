"""FastAPI serving surface. Orchestrates extract->buffer / consolidate / generate.

Endpoints:
    POST /chat        — run a turn: extract candidates -> buffer, then generate a reply
    POST /consolidate — "Consolidate Now": run a consolidation pass over the buffer
    GET  /memories    — return the consolidated counter + provenance for the UI
"""
from __future__ import annotations

from typing import Any

# from fastapi import FastAPI  # see requirements.txt; wired in create_app()


def create_app() -> Any:
    """Build and wire the FastAPI app (routes below). TODO."""
    raise NotImplementedError


def chat(payload: dict) -> dict:
    """POST /chat — extract->buffer, then generate on the edited model. TODO."""
    raise NotImplementedError


def consolidate(payload: dict) -> dict:
    """POST /consolidate — trigger a consolidation pass; return n_written. TODO."""
    raise NotImplementedError


def memories() -> dict:
    """GET /memories — consolidated counter + provenance for the UI. TODO."""
    raise NotImplementedError
