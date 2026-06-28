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
    Decision,
    MemoryItem,
)

logger = logging.getLogger(__name__)

# Retry policy for the external ``editing.edit`` seam (FULL tier). Up to
# ``EDIT_RETRIES + 1`` total attempts; ``_EDIT_BACKOFFS[attempt]`` seconds slept
# before the next try. Only after ALL attempts fail do we leave the item in the
# buffer (no drop / no count / no retire) — identical to the v0.4 failure path.
EDIT_RETRIES = 2
_EDIT_BACKOFFS = (0.0, 0.5, 1.0)

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


# Sentinel distinguishing "edit produced no value" from "edit failed all attempts".
_EDIT_FAILED = object()


def _edit_with_retry(editing: Any, model: Any, req: dict, item_id: str) -> Any:
    """Call ``editing.edit`` with a bounded retry loop (FULL tier).

    Tries up to ``EDIT_RETRIES + 1`` times, sleeping ``_EDIT_BACKOFFS[attempt]``
    seconds before each subsequent attempt and logging every retry. Returns the
    opaque edit ref on success, or the ``_EDIT_FAILED`` sentinel once every
    attempt has failed (the caller then leaves the item in the buffer).
    """
    for attempt in range(EDIT_RETRIES + 1):
        try:
            return editing.edit(model, req)
        except Exception:
            if attempt < EDIT_RETRIES:
                backoff = _EDIT_BACKOFFS[min(attempt, len(_EDIT_BACKOFFS) - 1)]
                logger.warning(
                    "consolidate: editing.edit failed for item %s (attempt %d/%d); "
                    "retrying after %.1fs",
                    item_id, attempt + 1, EDIT_RETRIES + 1, backoff,
                )
                if backoff:
                    time.sleep(backoff)
            else:
                logger.exception(
                    "consolidate: editing.edit failed for item %s after %d attempts; "
                    "left in buffer",
                    item_id, EDIT_RETRIES + 1,
                )
    return _EDIT_FAILED


def _process_item(it: MemoryItem, registry: list[MemoryItem], model: Any, editing: Any) -> int:
    """Consolidate ONE buffer item; return 1 iff a write happened (NEW or SUPERSEDE), else 0.

    Mirrors the original per-item loop body exactly: classify ``it`` against ``registry``
    (the in-pass-growing list of consolidated edit-route memory). A ``duplicate`` records
    ``PROV_DUPLICATE_OF`` and drains the buffer (return 0); otherwise edit FIRST and only on
    a successful ``editing.edit`` retire any olds, flip ``it`` to ``consolidated`` and drain
    it (return 1). A failed edit leaves ``it`` in the buffer and returns 0 (no drop, no
    retire). ``registry`` is mutated in place so later items in the same pass see this write.
    """
    # BATCH DEDUP coordination: we call the frozen 2-arg ``dedup.classify(it,
    # registry)`` form. ``dedup.classify`` does not expose a precomputed-vectors
    # keyword in its required signature, so passing one here would be a guess;
    # per-pass embedding reuse is delegated to ``dedup``'s own internal caching.
    # This keeps NO change to dedup's contract and favors correctness over the
    # micro-optimization (the registry also grows in-pass, which a single
    # precompute could not capture).
    d = dedup.classify(it, registry)

    if d.verdict == "duplicate":
        it.provenance = {**(it.provenance or {}), PROV_DUPLICATE_OF: d.target_id}
        store.upsert(it)
        buffer.drop([it.id])
        return 0

    # MULTI-TARGET SUPERSEDE: a candidate may retire several old memories at
    # once. Prefer ``target_ids`` (FULL tier); fall back to the single
    # ``target_id`` for back-compat. Targets already gone (concurrency) are
    # dropped here -> degrades toward a plain NEW write.
    olds = []
    if d.verdict == "supersede":
        target_ids = d.target_ids or ([d.target_id] if d.target_id else [])
        olds = [o for o in (store.get(tid) for tid in target_ids) if o is not None]

    req = build_edit_request(it)

    # Ordered best-effort atomicity: edit FIRST; only on success do we retire
    # the olds and write the new item. Retry the edit before giving up.
    ref = _edit_with_retry(editing, model, req, it.id)
    if ref is _EDIT_FAILED:
        # Do NOT drop, do NOT count, do NOT retire olds: leave in buffer for next pass.
        return 0

    if olds:
        for old in olds:
            old.status = "retired"
            old.provenance = {**(old.provenance or {}), PROV_SUPERSEDED_BY: it.id}
            store.upsert(old)
        retired_ids = [old.id for old in olds]
        # Single id stays a bare string for back-compat; multi-target -> list.
        it.provenance = {
            **(it.provenance or {}),
            PROV_SUPERSEDES: retired_ids[0] if len(retired_ids) == 1 else retired_ids,
        }

    it.status = "consolidated"
    it.provenance = {
        **(it.provenance or {}),
        PROV_EDIT_REF: _ref_id(ref),
        PROV_CONSOLIDATED_AT: time.time(),
    }
    store.upsert(it)
    registry.append(it)  # same-pass visibility for later near-dupes
    buffer.drop([it.id])
    return 1  # counts the successful edit once (NEW or SUPERSEDE)


def run_pass(trigger: str, ids=None) -> int:
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
    if ids is not None:
        idset = set(ids)
        items = [it for it in items if it.id in idset]
    registry = [m for m in store.by_status("consolidated") if m.route == "edit"]
    n_written = sum(_process_item(it, registry, model, editing) for it in items)
    return n_written


def preview_verdict(item: MemoryItem) -> Decision:
    """Read-only dedup preview: classify ``item`` against the live edit-route registry.

    Pure read — performs NO store mutation, NO buffer drop, and NO ``editing.edit``. Returns
    the ``dedup.classify`` ``Decision`` (verdict + target ids) so callers can surface the
    would-be outcome (duplicate / supersede / new) without committing a consolidation.
    """
    registry = [m for m in store.by_status("consolidated") if m.route == "edit"]
    return dedup.classify(item, registry)
