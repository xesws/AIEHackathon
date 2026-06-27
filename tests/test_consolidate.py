"""Load-bearing unit suite for ``memory.consolidate.run_pass``.

Uses the REAL ``memory.store`` + ``memory.buffer`` (pure in-memory) and isolates the
two external seams:
  * the lazily-imported ``editing`` module is replaced via ``sys.modules`` so
    consolidate's ``import editing`` picks up a fake recorder (or a raising variant);
  * ``memory.dedup.classify`` is monkeypatched to return controlled ``Decision``s,
    making dispatch deterministic and removing any embedding dependency.

All tests are order-independent: an autouse fixture resets the store and the
model-provider injection around every test.
"""
from __future__ import annotations

import sys
import types

import pytest

from memory import buffer, consolidate, dedup, store
from memory.schema import (
    PROV_CONSOLIDATED_AT,
    PROV_DUPLICATE_OF,
    PROV_EDIT,
    PROV_EDIT_REF,
    PROV_SUPERSEDED_BY,
    PROV_SUPERSEDES,
    Decision,
    MemoryItem,
)


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #
def make_edit_item(
    item_id: str,
    *,
    stem: str = "The capital of France is ___",
    target: str = "Paris",
    subject: str = "France",
    status: str = "buffer",
    text: str | None = None,
) -> MemoryItem:
    """Build an edit-route ``MemoryItem`` carrying a ``PROV_EDIT`` decomposition."""
    return MemoryItem(
        id=item_id,
        type="fact",
        text=text if text is not None else f"{stem} -> {target}",
        route="edit",
        status=status,
        source="msg-1",
        ts=0.0,
        provenance={PROV_EDIT: {"stem": stem, "target": target, "subject": subject}},
    )


def _make_fake_editing(*, raises: bool = False) -> types.ModuleType:
    """Create a fake ``editing`` module recording ``edit`` calls in ``.calls``."""
    fake = types.ModuleType("editing")
    calls: list[dict] = []

    def edit(model, req, **kw):  # noqa: ANN001
        calls.append({"model": model, "req": req, "kw": kw})
        if raises:
            raise RuntimeError("boom: editing.edit failed")
        return {"adapter": object(), "codebook_size": 1}

    fake.edit = edit  # type: ignore[attr-defined]
    fake.calls = calls  # type: ignore[attr-defined]
    return fake


@pytest.fixture(autouse=True)
def _isolation():
    """Reset the store and install a valid model provider around every test."""
    store.reset()
    consolidate.set_model_provider(lambda: object())
    yield
    consolidate.set_model_provider(None)
    store.reset()


@pytest.fixture
def editing_ok(monkeypatch):
    """Install a successful fake ``editing`` module via ``sys.modules``."""
    fake = _make_fake_editing(raises=False)
    monkeypatch.setitem(sys.modules, "editing", fake)
    return fake


@pytest.fixture
def editing_raises(monkeypatch):
    """Install a fake ``editing`` module whose ``edit`` raises."""
    fake = _make_fake_editing(raises=True)
    monkeypatch.setitem(sys.modules, "editing", fake)
    return fake


def _set_classify(monkeypatch, fn):
    """Replace ``dedup.classify`` with ``fn`` (consolidate calls ``dedup.classify``)."""
    monkeypatch.setattr(dedup, "classify", fn)


# --------------------------------------------------------------------------- #
# 1. NEW
# --------------------------------------------------------------------------- #
def test_new_writes_and_consolidates(monkeypatch, editing_ok):
    item = make_edit_item("a1")
    buffer.append(item)
    _set_classify(monkeypatch, lambda cand, consolidated: Decision("new"))

    n = consolidate.run_pass("manual")

    assert n == 1
    stored = store.get("a1")
    assert stored is not None
    assert stored.status == "consolidated"
    assert buffer.load_unconsolidated() == []
    assert PROV_EDIT_REF in stored.provenance
    assert PROV_CONSOLIDATED_AT in stored.provenance

    # editing.edit called exactly once with the build_edit_request dict.
    assert len(editing_ok.calls) == 1
    assert editing_ok.calls[0]["req"] == {
        "prompt": "The capital of France is ___",
        "target_new": "Paris",
        "subject": "France",
    }


# --------------------------------------------------------------------------- #
# 2. SUPERSEDE
# --------------------------------------------------------------------------- #
def test_supersede_retires_old_and_links(monkeypatch, editing_ok):
    old = make_edit_item("old1", target="Lyon", status="consolidated")
    store.upsert(old)
    new = make_edit_item("new1", target="Paris")
    buffer.append(new)
    _set_classify(monkeypatch, lambda cand, consolidated: Decision("supersede", "old1"))

    n = consolidate.run_pass("manual")

    assert n == 1
    stored_old = store.get("old1")
    stored_new = store.get("new1")
    assert stored_old.status == "retired"
    assert stored_old.provenance[PROV_SUPERSEDED_BY] == "new1"
    assert stored_new.status == "consolidated"
    assert stored_new.provenance[PROV_SUPERSEDES] == "old1"
    assert buffer.load_unconsolidated() == []
    assert len(editing_ok.calls) == 1


# --------------------------------------------------------------------------- #
# 3. SKIP / duplicate
# --------------------------------------------------------------------------- #
def test_duplicate_skips_edit_and_drains(monkeypatch, editing_ok):
    item = make_edit_item("d1")
    buffer.append(item)
    _set_classify(monkeypatch, lambda cand, consolidated: Decision("duplicate", "some_id"))

    n = consolidate.run_pass("manual")

    assert n == 0
    assert editing_ok.calls == []  # NOT called
    assert buffer.load_unconsolidated() == []  # drained
    # A duplicate is never promoted to consolidated; it stays status=="buffer"
    # and is therefore removed entirely by the status-guarded buffer.drop.
    assert store.get("d1") is None
    # The provenance stamp was written on the item before it was dropped.
    assert item.provenance[PROV_DUPLICATE_OF] == "some_id"


# --------------------------------------------------------------------------- #
# 4. FAILURE
# --------------------------------------------------------------------------- #
def test_failure_leaves_item_in_buffer(monkeypatch, editing_raises):
    item = make_edit_item("f1")
    buffer.append(item)
    _set_classify(monkeypatch, lambda cand, consolidated: Decision("new"))

    n = consolidate.run_pass("manual")

    assert n == 0
    remaining = buffer.load_unconsolidated()
    assert len(remaining) == 1
    assert remaining[0].id == "f1"
    assert store.get("f1").status == "buffer"


def test_failure_on_supersede_does_not_retire_old(monkeypatch, editing_raises):
    old = make_edit_item("old2", target="Lyon", status="consolidated")
    store.upsert(old)
    new = make_edit_item("new2", target="Paris")
    buffer.append(new)
    _set_classify(monkeypatch, lambda cand, consolidated: Decision("supersede", "old2"))

    n = consolidate.run_pass("manual")

    assert n == 0
    # Old item is untouched: not retired, no superseded_by link.
    stored_old = store.get("old2")
    assert stored_old.status == "consolidated"
    assert PROV_SUPERSEDED_BY not in (stored_old.provenance or {})
    # New item stays in the buffer for retry.
    remaining = buffer.load_unconsolidated()
    assert [r.id for r in remaining] == ["new2"]


# --------------------------------------------------------------------------- #
# 5. drain (multiple new)
# --------------------------------------------------------------------------- #
def test_drains_multiple_new_items(monkeypatch, editing_ok):
    buffer.append(make_edit_item("m1", subject="France", target="Paris"))
    buffer.append(make_edit_item("m2", subject="Italy", target="Rome"))
    _set_classify(monkeypatch, lambda cand, consolidated: Decision("new"))

    n = consolidate.run_pass("manual")

    assert n == 2
    assert buffer.load_unconsolidated() == []
    assert store.get("m1").status == "consolidated"
    assert store.get("m2").status == "consolidated"
    assert len(editing_ok.calls) == 2


# --------------------------------------------------------------------------- #
# 6. same-pass visibility
# --------------------------------------------------------------------------- #
def test_same_pass_visibility_registry_grows_mid_pass(monkeypatch, editing_ok):
    # buffer.load_unconsolidated()/by_status iterates dict insertion order, so s1
    # is processed before s2.
    buffer.append(make_edit_item("s1", subject="France", target="Paris"))
    buffer.append(make_edit_item("s2", subject="France", target="Paris"))

    # Keep a handle on the s2 object so we can inspect provenance after it is
    # dropped from the store (a skipped duplicate stays status=="buffer").
    s2_obj = buffer.load_unconsolidated()[1]
    assert s2_obj.id == "s2"

    calls = {"n": 0}

    def stateful_classify(cand, consolidated):
        calls["n"] += 1
        if calls["n"] == 1:
            # First item: brand new.
            return Decision("new")
        # Second item: s1 must already be visible in the registry passed in,
        # proving consolidate appended it mid-pass.
        ids = {m.id for m in consolidated}
        assert "s1" in ids, f"expected s1 in registry, got {ids}"
        return Decision("duplicate", "s1")

    _set_classify(monkeypatch, stateful_classify)

    n = consolidate.run_pass("manual")

    assert calls["n"] == 2
    assert n == 1  # only s1 written; s2 skipped as duplicate
    assert store.get("s1").status == "consolidated"
    # s2 was skipped (drained from buffer, marked duplicate, not consolidated).
    assert buffer.load_unconsolidated() == []
    assert store.get("s2") is None  # status-guarded drop removed it
    assert s2_obj.provenance[PROV_DUPLICATE_OF] == "s1"
    assert len(editing_ok.calls) == 1  # only s1 hit the editing seam


# --------------------------------------------------------------------------- #
# 7. no provider
# --------------------------------------------------------------------------- #
def test_no_provider_raises_runtime_error(monkeypatch, editing_ok):
    buffer.append(make_edit_item("n1"))
    _set_classify(monkeypatch, lambda cand, consolidated: Decision("new"))
    consolidate.set_model_provider(None)  # autouse fixture restores afterwards

    with pytest.raises(RuntimeError):
        consolidate.run_pass("manual")

    # Nothing was consolidated or drained.
    assert [r.id for r in buffer.load_unconsolidated()] == ["n1"]
