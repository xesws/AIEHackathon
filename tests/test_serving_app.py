"""Mocked unit tests for the FastAPI serving shell (``serving/app.py``).

These tests verify the ROUTING CONTRACT (serving_design.md §4) without ever loading the
GPU model. ``serving`` is a pure transport layer, so every collaborator it calls is
stubbed here; we only assert that app.py wires the calls together in the right order with
the right arguments.

How the GPU/model is neutralized
--------------------------------
``serving/app.py`` loads the base weights inside the FastAPI ``lifespan`` (startup):
``model_host.load_base()`` then ``consolidate.set_model_provider(...)``. The lifespan runs
when ``TestClient`` ENTERS its ``with`` block. So the ``client`` fixture below
``monkeypatch``-es ``model_host.load_base`` to a no-op (and ``current_model`` / ``tokenizer``
to cheap sentinels) BEFORE constructing the ``TestClient`` context manager. Result: startup
runs, but nothing touches CUDA, transformers, or the network. No model is ever loaded.

Import-style assumption (matches the frozen SHARED contract): app.py imports MODULES and
calls via attribute (``ingest.ingest``, ``generate.generate``, ``triggers.manual``,
``model_host.*``, ``buffer.*``, ``rag_store.*``, ``store.*``, ``consolidate.*``). We therefore
patch the attributes ON THOSE SOURCE MODULES; app.py resolves them at call time, so the
patches take effect for whichever ``app`` object (module-level ``app`` or ``create_app()``)
we hand to ``TestClient``.

Run with:  pytest tests/test_serving_app.py     (CPU only, fast, no network, no GPU)
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Fixture: stub every collaborator, neutralize the lifespan model load, build client
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def env(monkeypatch):
    """Yield ``SimpleNamespace(client, rec)``.

    ``rec`` records what each stub saw so tests can assert the wiring. The model load in
    the FastAPI lifespan is replaced with a no-op, so entering the ``TestClient`` context
    (which runs startup) never loads the GPU model.
    """
    from fastapi.testclient import TestClient

    # The exact module objects app.py calls into (module-attribute import style).
    from serving import model_host, ingest, triggers
    from memory import buffer, rag_store, store, schema, consolidate
    import generate

    rec = SimpleNamespace(
        load_base_calls=0,
        set_provider_fns=[],     # callables registered via set_model_provider
        ingest_chats=[],         # chat arg each ingest call received
        generate_calls=[],       # {"query","kwargs"} per generate call
        manual_calls=0,
        search_calls=[],         # {"query","k"} per rag_store.search call
        by_status_calls=[],      # status strings store.by_status was queried with
        order=[],                # interleave log: "ingest" / "generate" (write-then-read)
    )

    # Opaque sentinels — the "model" and "tokenizer" are never used, only passed through.
    rec.sentinel_model = object()
    rec.sentinel_tok = object()
    rec.search_hits = ["HIT"]    # what rag_store.search returns (a sentinel list)

    # Real MemoryItems so the (REAL, unstubbed) schema.to_dict serializes them in /memories
    # and so buffer_count reflects a concrete length.
    rec.buffer_items = [
        schema.MemoryItem(
            id="buf1", type="fact", text="buffered fact", route="edit",
            status="buffer", source="msg-1", ts=1.0, provenance=None,
        )
    ]
    rec.cons_items = [
        schema.MemoryItem(
            id="c1", type="fact", text="weight fact 1", route="edit",
            status="consolidated", source="msg-0", ts=1.0, provenance=None,
        ),
        schema.MemoryItem(
            id="c2", type="preference", text="weight pref 2", route="edit",
            status="consolidated", source="msg-0", ts=2.0, provenance={"k": "v"},
        ),
    ]

    # ---- stubs -----------------------------------------------------------------
    def fake_load_base(*args, **kwargs):
        rec.load_base_calls += 1
        return rec.sentinel_model  # never used; startup just needs it not to raise

    def fake_current_model():
        return rec.sentinel_model

    def fake_tokenizer():
        return rec.sentinel_tok

    def fake_set_model_provider(fn):
        rec.set_provider_fns.append(fn)  # record the zero-arg callable; do nothing else

    def fake_ingest(chat):
        rec.order.append("ingest")
        rec.ingest_chats.append(chat)
        return {
            "n_extracted": 2,
            "n_edit_buffered": 2,
            "n_rag_indexed": 0,
            "edit_ids": ["mem_a", "mem_b"],
        }

    def fake_generate(query, **kwargs):
        rec.order.append("generate")
        rec.generate_calls.append({"query": query, "kwargs": kwargs})
        return "FAKE_REPLY"

    def fake_manual():
        rec.manual_calls += 1
        return 2

    def fake_search(query, k=5):
        rec.search_calls.append({"query": query, "k": k})
        return list(rec.search_hits)

    def fake_load_unconsolidated():
        return list(rec.buffer_items)

    def fake_by_status(status):
        rec.by_status_calls.append(status)
        if status == "buffer":
            return list(rec.buffer_items)
        if status == "consolidated":
            return list(rec.cons_items)
        return []

    # ---- patch the source-module attributes app.py resolves at call time -------
    monkeypatch.setattr(model_host, "load_base", fake_load_base)
    monkeypatch.setattr(model_host, "current_model", fake_current_model)
    monkeypatch.setattr(model_host, "tokenizer", fake_tokenizer)
    monkeypatch.setattr(consolidate, "set_model_provider", fake_set_model_provider)
    monkeypatch.setattr(ingest, "ingest", fake_ingest)
    monkeypatch.setattr(generate, "generate", fake_generate)
    monkeypatch.setattr(triggers, "manual", fake_manual)
    monkeypatch.setattr(rag_store, "search", fake_search)
    monkeypatch.setattr(buffer, "load_unconsolidated", fake_load_unconsolidated)
    monkeypatch.setattr(store, "by_status", fake_by_status)

    # Import app AFTER patching. Prefer the module-level ``app`` (what uvicorn serves);
    # fall back to building one. Either reference resolves the patched collaborators at
    # request/lifespan time.
    from serving import app as app_module
    fastapi_app = getattr(app_module, "app", None) or app_module.create_app()

    # ``with`` runs the lifespan (startup/shutdown). load_base is now a no-op -> no model.
    with TestClient(fastapi_app) as client:
        yield SimpleNamespace(client=client, rec=rec)


# ──────────────────────────────────────────────────────────────────────────────
# Startup / lifespan
# ──────────────────────────────────────────────────────────────────────────────
def test_startup_loads_base_and_registers_zero_arg_provider(env):
    """Entering the client context ran the lifespan: load_base was called once and a
    ZERO-ARG provider callable (returning the current model) was registered."""
    rec = env.rec
    assert rec.load_base_calls == 1
    assert len(rec.set_provider_fns) == 1
    provider = rec.set_provider_fns[0]
    assert callable(provider)
    # provider must take no args and yield the resident model sentinel.
    assert provider() is rec.sentinel_model


# ──────────────────────────────────────────────────────────────────────────────
# POST /chat
# ──────────────────────────────────────────────────────────────────────────────
def test_chat_calls_ingest_then_generate(env):
    """/chat ingests the user turn BEFORE generating (write-then-read), passes the
    resident model + chat template flag, and returns the contract fields."""
    rec = env.rec
    resp = env.client.post("/chat", json={"message": "hi"})
    assert resp.status_code == 200

    # ingest saw the message wrapped as a single-element chat list of role/content dicts.
    assert rec.ingest_chats == [[{"role": "user", "content": "hi"}]]

    # ingest ran strictly before generate (the "write-then-read" ordering).
    assert rec.order == ["ingest", "generate"]

    # generate received the chat-template flag and the current_model sentinel.
    assert len(rec.generate_calls) == 1
    call = rec.generate_calls[0]
    assert call["query"] == "hi"
    assert call["kwargs"]["use_chat_template"] is True
    assert call["kwargs"]["model"] is rec.sentinel_model

    # response shape / types.
    body = resp.json()
    assert set(["reply", "buffer_count", "learned"]).issubset(body)
    assert isinstance(body["reply"], str) and body["reply"] == "FAKE_REPLY"
    assert isinstance(body["buffer_count"], int) and body["buffer_count"] == 1
    assert isinstance(body["learned"], list) and body["learned"] == ["mem_a", "mem_b"]


def test_chat_rag_off_passes_empty_rag_hits(env):
    """rag_off=True MUST blank rag_hits ([]) WITHOUT disabling with_rag and WITHOUT
    dropping the buffer window (INV-S3). rag_store.search must NOT be called."""
    rec = env.rec
    resp = env.client.post("/chat", json={"message": "q", "rag_off": True})
    assert resp.status_code == 200

    assert rec.search_calls == []  # docs window off -> no retrieval

    kwargs = rec.generate_calls[0]["kwargs"]
    assert kwargs["rag_hits"] == []                       # docs window empty
    assert kwargs.get("with_rag", True) is not False      # with_rag left default True
    assert kwargs["buffer"] == rec.buffer_items           # user-facts window STILL passed


def test_chat_rag_on_uses_search(env):
    """rag_off=False retrieves via rag_store.search(message, k=5) and forwards those hits
    to generate as rag_hits."""
    rec = env.rec
    resp = env.client.post("/chat", json={"message": "q", "rag_off": False})
    assert resp.status_code == 200

    assert len(rec.search_calls) == 1
    assert rec.search_calls[0]["query"] == "q"
    assert rec.search_calls[0]["k"] == 5

    kwargs = rec.generate_calls[0]["kwargs"]
    assert kwargs["rag_hits"] == rec.search_hits  # the search sentinel list (["HIT"])


# ──────────────────────────────────────────────────────────────────────────────
# POST /consolidate
# ──────────────────────────────────────────────────────────────────────────────
def test_consolidate_calls_manual(env):
    """/consolidate delegates to triggers.manual() and reports n_written + buffer_count."""
    rec = env.rec
    resp = env.client.post("/consolidate", json={})
    assert resp.status_code == 200

    assert rec.manual_calls == 1
    body = resp.json()
    assert isinstance(body["n_written"], int) and body["n_written"] == 2
    assert isinstance(body["buffer_count"], int) and body["buffer_count"] == 1


# ──────────────────────────────────────────────────────────────────────────────
# GET /memories
# ──────────────────────────────────────────────────────────────────────────────
def test_memories_reads_store(env):
    """/memories reads store.by_status for BOTH statuses and returns serialized lists with
    matching counts."""
    rec = env.rec
    resp = env.client.get("/memories")
    assert resp.status_code == 200

    # store was queried for both edit-route slices.
    assert "buffer" in rec.by_status_calls
    assert "consolidated" in rec.by_status_calls

    body = resp.json()
    assert isinstance(body["buffer"], list) and isinstance(body["consolidated"], list)
    assert len(body["buffer"]) == 1
    assert len(body["consolidated"]) == 2

    # counts mirror the list lengths.
    assert body["counts"] == {"buffer": 1, "consolidated": 2}

    # items were serialized via schema.to_dict (real, unstubbed) -> plain dicts with fields.
    assert body["buffer"][0]["id"] == "buf1"
    assert {"id", "type", "text", "route", "status", "source", "ts"}.issubset(
        body["buffer"][0]
    )
