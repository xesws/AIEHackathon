"""Buffer -> weights consolidation pass. ``n_written`` is the UI counter (Layer-2 integrator)."""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

from . import buffer, dedup, store
from .schema import (
    PROV_CONSOLIDATED_AT,
    PROV_DUPLICATE_OF,
    PROV_EDIT,
    PROV_EDIT_REF,
    PROV_SUPERSEDED_BY,
    PROV_SUPERSEDES,
    MemoryItem,
)

logger = logging.getLogger(__name__)

# Model-provider injection (INV-11): serving injects a zero-arg callable returning
# the resident model handle, so ``memory/`` never statically imports ``serving/``.
_model_provider: Optional[Callable[[], Any]] = None


def set_model_provider(fn: Callable[[], Any]) -> None:
    """Register the zero-arg callable that yields the resident (opaque) model handle.

    ``serving`` calls this at startup (``lambda: model_host.current_model()``), keeping
    the dependency one-directional (serving -> memory) and ``memory/`` free of ``serving/``.
    """
    global _model_provider
    _model_provider = fn


def build_edit_request(item: MemoryItem) -> dict:
    """Build the ``editing.edit`` request dict from an edit-route item's provenance.

    Uses the HoReN edit decomposition stored at ``provenance[PROV_EDIT]``
    (``{stem, target, subject}``). If it is missing, falls back to the raw item text
    and logs a warning.
    """
    edit = (item.provenance or {}).get(PROV_EDIT)
    if not edit:
        logger.warning("consolidate: item %s has no %s provenance; falling back to raw text", item.id, PROV_EDIT)
        return {"prompt": item.text, "target_new": ""}
    return {
        "prompt": edit["stem"],
        "target_new": edit["target"],
        "subject": edit.get("subject", ""),
    }


def _ref_id(ref: Any) -> int:
    """Opaque audit handle for an ``editing.edit`` return value (never inspected for content)."""
    if isinstance(ref, dict) and "adapter" in ref:
        return id(ref["adapter"])
    return id(ref)


def run_pass(trigger: str) -> int:
    """Run one consolidation pass (invoked by ``serving/triggers.py``).

    For each unconsolidated buffer item: classify vs. consolidated edit-route memory via
    ``dedup.classify``, then skip (duplicate) / retire-old + write (supersede) / write (new).
    Writing calls the external ``editing.edit`` seam; on success the item flips to
    ``consolidated``, records provenance, and leaves the buffer. Every processed item drains
    the buffer (including duplicates); a failed ``editing.edit`` leaves the item in the buffer
    for the next pass.

    Returns ``n_written`` — successful edits this pass (NEW + SUPERSEDE; NOT skips).
    """
    model = _model_provider() if _model_provider is not None else None
    if model is None:
        raise RuntimeError("consolidate: no model provider set (call set_model_provider)")

    # Lazy import keeps ``serving`` out of memory/'s static import graph (INV-11).
    import editing

    items = buffer.load_unconsolidated()
    registry = [m for m in store.by_status("consolidated") if m.route == "edit"]
    n_written = 0

    for it in items:
        d = dedup.classify(it, registry)

        if d.verdict == "duplicate":
            it.provenance = {**(it.provenance or {}), PROV_DUPLICATE_OF: d.target_id}
            store.upsert(it)
            buffer.drop([it.id])
            continue

        old = store.get(d.target_id) if d.verdict == "supersede" else None
        req = build_edit_request(it)

        try:
            ref = editing.edit(model, req)
        except Exception:
            # Do NOT drop, do NOT count, do NOT retire old: leave in buffer for retry.
            logger.exception("consolidate: editing.edit failed for item %s; left in buffer", it.id)
            continue

        if old is not None:
            old.status = "retired"
            old.provenance = {**(old.provenance or {}), PROV_SUPERSEDED_BY: it.id}
            store.upsert(old)
            it.provenance = {**(it.provenance or {}), PROV_SUPERSEDES: old.id}

        it.status = "consolidated"
        it.provenance = {
            **(it.provenance or {}),
            PROV_EDIT_REF: _ref_id(ref),
            PROV_CONSOLIDATED_AT: time.time(),
        }
        store.upsert(it)
        registry.append(it)  # same-pass visibility for later near-dupes
        buffer.drop([it.id])
        n_written += 1

    return n_written
