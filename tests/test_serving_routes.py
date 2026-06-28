"""Mocked unit tests for the EXTENDED serving routes (serving v1.0 contract).

Companion to ``tests/test_serving_app.py``: that file pins the ORIGINAL 3-endpoint
routing contract; THIS file pins the v1.0 NET-NEW + EXTENDED surface — per-item
consolidate, drop / demote-to-rag / edit-text, the edit-module toggle, ``/health``, and
the widened ``/chat`` + ``/memories`` response bodies. Same philosophy: ``serving`` is a
pure transport layer, so every collaborator is stubbed and we assert ONLY that ``app.py``
wires the calls together with the right arguments / order and returns the contracted shapes.

How the GPU/model is neutralized (identical to test_serving_app.py)
------------------------------------------------------------------
``serving/app.py`` loads the base weights inside the FastAPI ``lifespan`` (startup):
``model_host.load_base()`` then ``consolidate.set_model_provider(...)``. The lifespan runs
when ``TestClient`` ENTERS its ``with`` block. The ``env`` fixture ``monkeypatch``-es
``model_host.load_base`` to a no-op (and ``current_model`` / ``tokenizer`` to sentinels)
BEFORE constructing the ``TestClient`` context manager. Result: startup runs, but nothing
touches CUDA, transformers, or the network. No model is ever loaded.

Import-style assumption (matches the frozen SHARED contract): app.py imports MODULES and
calls via attribute (``ingest.ingest``, ``generate.generate``, ``triggers.manual``,
``model_host.*``, ``buffer.*``, ``rag_store.*``, ``store.*``, ``consolidate.*``,
``extract.decompose``). We therefore patch the attributes ON THOSE SOURCE MODULES; app.py
resolves them at call time, so the patches take effect for whichever ``app`` object we
hand to ``TestClient``.

Three symbols are NET-NEW this round and are added by sibling patches; we patch them with
``raising=False`` so the fixture never crashes on module landing order:
``extract.decompose``, ``model_host.recorded_adapter``, ``model_host.edit_active``.

Run with:  pytest tests/test_serving_routes.py     (CPU only, fast, no network, no GPU)
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from memory.schema import MemoryItem, PROV_CODEBOOK_KEYS, PROV_EDIT


# ──────────────────────────────────────────────────────────────────────────────
# Tiny builders / assertions
# ──────────────────────────────────────────────────────────────────────────────
def _item(**over) -> MemoryItem:
    """A real ``MemoryItem`` with buffer-fact defaults so the (REAL, unstubbed)
    ``schema.to_dict`` serializes it; override any field per test."""
    base = dict(
        id="buf1", type="fact", text="buffered fact", route="edit",
        status="buffer", source="msg-1", ts=1.0, provenance=None,
    )
    base.update(over)
    return MemoryItem(**base)


def _assert_error_body(resp) -> None:
    """A contract error: body is exactly the ``{error: <str>}`` envelope — NOT FastAPI's
    default ``{detail: ...}`` and NEVER a fake-success body (no ``ok`` leaking through)."""
    body = resp.json()
    assert isinstance(body, dict), f"error body not a dict: {body!r}"
    assert "error" in body, f"error body missing 'error' key: {body!r}"
    assert isinstance(body["error"], str) and body["error"], f"bad error value: {body!r}"
    assert "ok" not in body, f"error leaked a fake-success 'ok': {body!r}"


# ──────────────────────────────────────────────────────────────────────────────
# Fixture: stub every collaborator, neutralize the lifespan model load, build client
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def env(monkeypatch):
    """Yield ``SimpleNamespace(client, rec)``.

    ``rec`` carries both the stub RETURN values (tests mutate these before a request to
    drive a branch — e.g. ``rec.adapter``, ``rec.get_map``, ``rec.decompose_return``) and
    the RECORDED inputs (so tests can assert the wiring). The lifespan model load is a
    no-op, so entering the ``TestClient`` context never loads the GPU model.
    """
    from fastapi.testclient import TestClient

    # The exact module objects app.py calls into (module-attribute import style).
    from serving import model_host, ingest, triggers
    from memory import buffer, rag_store, store, consolidate, extract
    import generate

    rec = SimpleNamespace(
        # lifespan
        load_base_calls=0,
        set_provider_fns=[],
        # recorded inputs
        ingest_chats=[],
        generate_calls=[],
        search_calls=[],
        by_status_calls=[],
        get_calls=[],
        manual_calls=[],      # one entry per triggers.manual call: the ``ids`` it saw
        drop_calls=[],        # one entry per buffer.drop call: the id-list it saw
        rag_add_items=[],     # the item objects handed to rag_store.add
        rag_add_routes=[],    # item.route captured AT the rag_store.add call
        upsert_calls=[],      # items handed to store.upsert
        decompose_calls=[],   # texts handed to extract.decompose
        swap_calls=[],        # args handed to model_host.swap_edit_module
        order=[],             # interleave log: ingest/generate/drop/rag_add
        # stub RETURN values (tests may override before a request)
        n_written=2,
        adapter=None,                 # recorded_adapter() -> None by default
        edit_active_val=False,        # edit_active() -> False by default
        decompose_return={"stem": "JQ is allergic to", "target": "nickel", "subject": "JQ"},
        ingest_result={
            "n_extracted": 2,
            "n_edit_buffered": 2,
            "n_rag_indexed": 1,
            "edit_ids": ["mem_a", "mem_b"],
        },
    )

    # Opaque sentinels — never used, only passed through. current_model() is NOT None so
    # /health reports ready==True.
    rec.sentinel_model = object()
    rec.sentinel_tok = object()
    rec.current_model_val = rec.sentinel_model

    # Real MemoryItems for the windows app.py serializes via the REAL schema.to_dict.
    rec.buffer_items = [_item(id="buf1", text="buffered fact")]
    rec.cons_items = [
        _item(id="c1", text="weight fact 1", status="consolidated"),
        _item(id="c2", type="belief", text="weight pref 2", status="consolidated",
              provenance={"k": "v"}),
    ]
    rec.rag_items = [
        (_item(id="r1", type="fact", text="rag doc 1", route="rag", status="consolidated",
               source="doc"), [0.1, 0.2, 0.3]),
    ]
    rec.search_hits = [
        _item(id="rag_hit1", type="fact", text="retrieved doc", route="rag",
              status="consolidated", source="doc"),
    ]
    # store.get lookup table (tests swap this out to drive 404 branches).
    rec.get_map = {it.id: it for it in rec.buffer_items}

    # ---- stubs -----------------------------------------------------------------
    def fake_load_base(*args, **kwargs):
        rec.load_base_calls += 1
        return rec.sentinel_model

    def fake_current_model():
        return rec.current_model_val

    def fake_tokenizer():
        return rec.sentinel_tok

    def fake_set_model_provider(fn):
        rec.set_provider_fns.append(fn)

    def fake_ingest(chat):
        rec.order.append("ingest")
        rec.ingest_chats.append(chat)
        return dict(rec.ingest_result)

    def fake_generate(query, **kwargs):
        rec.order.append("generate")
        rec.generate_calls.append({"query": query, "kwargs": kwargs})
        return "FAKE_REPLY"

    def fake_search(query, k=5, with_scores=False):
        # v2.4: /chat now calls with with_scores=True (additive). Mirror the real return shape:
        # list[(item, cosine)] when scored, else the plain item list.
        rec.search_calls.append({"query": query, "k": k})
        hits = list(rec.search_hits)
        return [(h, 0.9) for h in hits] if with_scores else hits

    def fake_load_unconsolidated():
        return list(rec.buffer_items)

    def fake_by_status(status):
        rec.by_status_calls.append(status)
        if status == "buffer":
            return list(rec.buffer_items)
        if status == "consolidated":
            return list(rec.cons_items)
        return []

    def fake_rag_all():
        return list(rec.rag_items)

    def fake_get(item_id):
        rec.get_calls.append(item_id)
        return rec.get_map.get(item_id)

    def fake_upsert(item):
        rec.upsert_calls.append(item)
        rec.get_map[item.id] = item

    def fake_drop(ids):
        rec.order.append("drop")
        rec.drop_calls.append(list(ids))

    def fake_rag_add(item):
        rec.order.append("rag_add")
        rec.rag_add_items.append(item)
        rec.rag_add_routes.append(item.route)  # capture route AT call time

    def fake_manual(ids=None):
        rec.manual_calls.append(ids)
        return rec.n_written

    def fake_recorded_adapter():
        return rec.adapter

    def fake_edit_active():
        return rec.edit_active_val

    def fake_swap(m):
        rec.swap_calls.append(m)

    def fake_decompose(text):
        rec.decompose_calls.append(text)
        return rec.decompose_return

    # ---- patch the source-module attributes app.py resolves at call time -------
    monkeypatch.setattr(model_host, "load_base", fake_load_base)
    monkeypatch.setattr(model_host, "current_model", fake_current_model)
    monkeypatch.setattr(model_host, "tokenizer", fake_tokenizer)
    monkeypatch.setattr(model_host, "swap_edit_module", fake_swap)
    # NET-NEW getters added by sibling patches -> tolerate landing order.
    monkeypatch.setattr(model_host, "recorded_adapter", fake_recorded_adapter, raising=False)
    monkeypatch.setattr(model_host, "edit_active", fake_edit_active, raising=False)
    monkeypatch.setattr(consolidate, "set_model_provider", fake_set_model_provider)
    monkeypatch.setattr(ingest, "ingest", fake_ingest)
    monkeypatch.setattr(generate, "generate", fake_generate)
    monkeypatch.setattr(triggers, "manual", fake_manual)
    monkeypatch.setattr(rag_store, "search", fake_search)
    monkeypatch.setattr(rag_store, "add", fake_rag_add)
    monkeypatch.setattr(buffer, "load_unconsolidated", fake_load_unconsolidated)
    monkeypatch.setattr(buffer, "drop", fake_drop)
    monkeypatch.setattr(store, "by_status", fake_by_status)
    monkeypatch.setattr(store, "get", fake_get)
    monkeypatch.setattr(store, "upsert", fake_upsert)
    monkeypatch.setattr(store, "rag_all", fake_rag_all)
    # NET-NEW focused decompose used by PATCH (editPending) -> tolerate landing order.
    monkeypatch.setattr(extract, "decompose", fake_decompose, raising=False)

    # Import app AFTER patching. Prefer the module-level ``app`` (what uvicorn serves);
    # fall back to building one. Either reference resolves the patched collaborators.
    from serving import app as app_module
    fastapi_app = getattr(app_module, "app", None) or app_module.create_app()

    # ``with`` runs the lifespan (startup/shutdown). load_base is now a no-op -> no model.
    with TestClient(fastapi_app) as client:
        yield SimpleNamespace(client=client, rec=rec)


# ──────────────────────────────────────────────────────────────────────────────
# Startup / lifespan (still neutralized, still wires the zero-arg provider)
# ──────────────────────────────────────────────────────────────────────────────
def test_startup_runs_lifespan_without_touching_gpu(env):
    """Entering the client context ran the lifespan: load_base once, a ZERO-ARG provider
    callable (returning the current model) registered — and no GPU/network touched."""
    rec = env.rec
    assert rec.load_base_calls == 1
    assert len(rec.set_provider_fns) == 1
    provider = rec.set_provider_fns[0]
    assert callable(provider)
    assert provider() is rec.sentinel_model


# ──────────────────────────────────────────────────────────────────────────────
# POST /chat — widened response (retrieved + extracted + rag_indexed), old fields kept
# ──────────────────────────────────────────────────────────────────────────────
def test_chat_includes_new_fields_and_preserves_old(env):
    """/chat keeps reply/buffer_count/learned AND adds retrieved(list[dict]) / extracted /
    rag_indexed, all sourced from ingest + the serialized rag_store.search hits."""
    rec = env.rec
    resp = env.client.post("/chat", json={"message": "hi"})
    assert resp.status_code == 200
    body = resp.json()

    # preserved (v0.8) fields
    assert body["reply"] == "FAKE_REPLY"
    assert isinstance(body["buffer_count"], int) and body["buffer_count"] == 1
    assert body["learned"] == ["mem_a", "mem_b"]

    # NEW fields
    assert isinstance(body["retrieved"], list)
    assert body["extracted"] == 2          # ingest result n_extracted
    assert body["rag_indexed"] == 1        # ingest result n_rag_indexed

    # retrieved == the rag_store.search hits serialized via the REAL schema.to_dict.
    assert len(body["retrieved"]) == 1
    assert body["retrieved"][0]["id"] == "rag_hit1"
    assert {"id", "type", "text", "route", "status", "source", "ts", "provenance"}.issubset(
        body["retrieved"][0]
    )

    # write-then-read ordering preserved (ingest BEFORE generate).
    assert rec.order == ["ingest", "generate"]


def test_chat_rag_off_blanks_retrieved(env):
    """rag_off=True empties the docs window: rag_store.search is NOT called and the new
    ``retrieved`` field comes back as ``[]`` (INV-S3 — buffer window untouched)."""
    rec = env.rec
    resp = env.client.post("/chat", json={"message": "q", "rag_off": True})
    assert resp.status_code == 200
    assert rec.search_calls == []          # docs window off -> no retrieval
    assert resp.json()["retrieved"] == []  # nothing surfaced to the UI


# ──────────────────────────────────────────────────────────────────────────────
# GET /memories — adds rag list + counts.rag
# ──────────────────────────────────────────────────────────────────────────────
def test_memories_includes_rag_and_counts_rag(env):
    """/memories adds ``rag`` (= serialized store.rag_all) and ``counts.rag`` alongside the
    unchanged buffer / consolidated slices."""
    rec = env.rec
    resp = env.client.get("/memories")
    assert resp.status_code == 200
    body = resp.json()

    assert {"buffer", "consolidated", "rag", "counts"}.issubset(body)
    assert isinstance(body["rag"], list) and len(body["rag"]) == 1
    assert body["rag"][0]["id"] == "r1"
    assert {"id", "type", "text", "route", "status", "source", "ts", "provenance"}.issubset(
        body["rag"][0]
    )
    assert body["counts"] == {"buffer": 1, "consolidated": 2, "rag": 1}


# ──────────────────────────────────────────────────────────────────────────────
# GET /health — ready / edit_on / counts
# ──────────────────────────────────────────────────────────────────────────────
def test_health_reports_ready_edit_on_and_counts(env):
    """/health gates booting (ready == current_model() is not None), reports the edit
    state (edit_active) and the same three counts."""
    resp = env.client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True                      # current_model() sentinel -> not None
    assert isinstance(body["edit_on"], bool) and body["edit_on"] is False
    assert body["edit_available"] is False
    assert body["codebook_size"] == 0
    assert isinstance(body["boot_id"], str) and body["boot_id"]
    assert isinstance(body["started_at"], (int, float))
    assert body["counts"] == {"buffer": 1, "consolidated": 2, "rag": 1}


def test_health_edit_on_mirrors_edit_active_true(env):
    """C3 lock: /health.edit_on must mirror model_host.edit_active() in BOTH directions.

    The False case is covered above; here edit_active() is True (exactly the post-consolidate
    state — editing.edit installs the adapter, so edit_active() flips True). Because the SPA
    switch re-syncs to /health.edit_on on every refresh() and /health.edit_on returns the raw
    slot truth (no shadow state), the switch can never residually desync from the server. So C3
    is NOT a desync bug; this test pins the honest-mirror invariant that makes it so."""
    rec = env.rec
    rec.edit_active_val = True
    body = env.client.get("/health").json()
    assert body["edit_on"] is True


def test_health_reports_codebook_size_when_adapter_recorded(env):
    """The frontend's codebook readout uses the real adapter row count, not memory count."""
    rec = env.rec
    rec.adapter = SimpleNamespace(keys=SimpleNamespace(shape=(7, 4096)))
    body = env.client.get("/health").json()
    assert body["edit_available"] is True
    assert body["codebook_size"] == 7


# ──────────────────────────────────────────────────────────────────────────────
# POST /consolidate/item — per-item consolidation via triggers.manual(ids=[id])
# ──────────────────────────────────────────────────────────────────────────────
def test_consolidate_item_calls_manual_with_ids(env):
    """/consolidate/item delegates to ``triggers.manual(ids=[id])`` (scoped pass) and
    reports n_written + buffer_count."""
    rec = env.rec
    resp = env.client.post("/consolidate/item", json={"id": "buf1"})
    assert resp.status_code == 200
    assert rec.manual_calls == [["buf1"]]             # scoped to exactly this id
    body = resp.json()
    assert body["n_written"] == 2
    assert isinstance(body["buffer_count"], int) and body["buffer_count"] == 1


# ──────────────────────────────────────────────────────────────────────────────
# POST /memories/{id}/drop — discard a buffered item
# ──────────────────────────────────────────────────────────────────────────────
def test_drop_happy_returns_ok_and_drops_buffer(env):
    """drop on a buffered item -> buffer.drop([id]) and {ok, buffer_count}."""
    rec = env.rec
    resp = env.client.post("/memories/buf1/drop")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert isinstance(body["buffer_count"], int)
    assert rec.drop_calls == [["buf1"]]


def test_drop_404_when_item_missing(env):
    """drop on an unknown id -> 404 {error}; the buffer is never touched."""
    rec = env.rec
    rec.get_map = {}                                  # store.get -> None
    resp = env.client.post("/memories/nope/drop")
    assert resp.status_code == 404
    _assert_error_body(resp)
    assert rec.drop_calls == []


def test_drop_404_when_item_not_buffer(env):
    """drop on a non-buffer (already consolidated) item -> 404 {error}; never dropped."""
    rec = env.rec
    rec.get_map = {"c1": _item(id="c1", status="consolidated")}
    resp = env.client.post("/memories/c1/drop")
    assert resp.status_code == 404
    _assert_error_body(resp)
    assert rec.drop_calls == []


# ──────────────────────────────────────────────────────────────────────────────
# POST /memories/{id}/route — demote a buffered edit item to the RAG store
# ──────────────────────────────────────────────────────────────────────────────
def test_route_demotes_buffer_to_rag_drop_then_add(env):
    """route=="rag" demotes: buffer.drop([id]) runs BEFORE rag_store.add, and the item
    handed to rag_store.add has its route flipped to "rag"."""
    rec = env.rec
    resp = env.client.post("/memories/buf1/route", json={"route": "rag"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "buffer_count" in body

    # ordering: drop (while still a buffer row) THEN add to rag.
    assert rec.order == ["drop", "rag_add"]
    assert rec.drop_calls == [["buf1"]]

    # the demoted item went to rag_store.add with route == "rag".
    assert len(rec.rag_add_items) == 1
    assert rec.rag_add_items[0].id == "buf1"
    assert rec.rag_add_routes == ["rag"]


def test_route_400_when_route_not_rag(env):
    """Only route=="rag" is supported; anything else -> 400 {error} with NO side effects."""
    rec = env.rec
    resp = env.client.post("/memories/buf1/route", json={"route": "edit"})
    assert resp.status_code == 400
    _assert_error_body(resp)
    assert rec.drop_calls == [] and rec.rag_add_items == []


# ──────────────────────────────────────────────────────────────────────────────
# PATCH /memories/{id} — re-word a buffered item; re-derive the HoReN edit decomposition
# ──────────────────────────────────────────────────────────────────────────────
def test_patch_sets_prov_edit_when_decompose_returns_dict(env):
    """PATCH updates the text, calls extract.decompose(text), and on a dict result stores
    it under provenance[PROV_EDIT] so a later consolidate edits the NEW wording."""
    rec = env.rec
    rec.decompose_return = {"stem": "JQ is allergic to", "target": "nickel", "subject": "JQ"}
    resp = env.client.patch("/memories/buf1", json={"text": "JQ is allergic to nickel"})
    assert resp.status_code == 200

    assert rec.decompose_calls == ["JQ is allergic to nickel"]
    item = resp.json()["item"]
    assert item["text"] == "JQ is allergic to nickel"
    assert item["provenance"][PROV_EDIT] == rec.decompose_return
    # persisted back through store.upsert
    assert rec.upsert_calls and rec.upsert_calls[-1].id == "buf1"


def test_patch_skips_prov_edit_when_decompose_none(env):
    """When extract.decompose returns None the text still updates but PROV_EDIT is NOT set
    (no bad ``target_new=""`` edit gets written)."""
    rec = env.rec
    rec.decompose_return = None
    resp = env.client.patch("/memories/buf1", json={"text": "JQ is allergic to nickel"})
    assert resp.status_code == 200

    assert rec.decompose_calls == ["JQ is allergic to nickel"]
    item = resp.json()["item"]
    assert item["text"] == "JQ is allergic to nickel"
    assert PROV_EDIT not in (item["provenance"] or {})


# ──────────────────────────────────────────────────────────────────────────────
# POST /edit-module — hot-swap the edit module on / off
# ──────────────────────────────────────────────────────────────────────────────
def test_edit_module_on_without_adapter_409(env):
    """Enabling edits with NO recorded adapter is a conflict -> 409 {error}; nothing is
    swapped in."""
    rec = env.rec
    rec.adapter = None
    resp = env.client.post("/edit-module", json={"on": True})
    assert resp.status_code == 409
    _assert_error_body(resp)
    assert rec.swap_calls == []


def test_edit_module_on_with_adapter_swaps_in(env):
    """With a recorded adapter, {on:true} swaps it in via swap_edit_module(adapter)."""
    rec = env.rec
    rec.adapter = object()
    resp = env.client.post("/edit-module", json={"on": True})
    assert resp.status_code == 200
    assert resp.json()["on"] is True
    assert rec.swap_calls == [rec.adapter]


def test_edit_module_off_swaps_base_back(env):
    """{on:false} restores base behaviour via swap_edit_module(None)."""
    rec = env.rec
    resp = env.client.post("/edit-module", json={"on": False})
    assert resp.status_code == 200
    assert resp.json()["on"] is False
    assert rec.swap_calls == [None]


# ──────────────────────────────────────────────────────────────────────────────
# GET /rag/search — semantic search over the RAG store (v1.3)
# ──────────────────────────────────────────────────────────────────────────────
def test_rag_search_returns_serialized_hits(env):
    """/rag/search delegates to rag_store.search(q, k) and serializes hits via schema.to_dict."""
    rec = env.rec
    resp = env.client.get("/rag/search", params={"q": "postgres", "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["results"], list) and len(body["results"]) == 1
    assert body["results"][0]["id"] == "rag_hit1"
    assert {"id", "type", "text", "route", "status", "source", "ts", "provenance"}.issubset(
        body["results"][0]
    )
    assert rec.search_calls == [{"query": "postgres", "k": 3}]   # q + k forwarded


def test_rag_search_empty_query_skips_search(env):
    """A blank query returns {results: []} WITHOUT calling rag_store.search (no wasted LLM call)."""
    rec = env.rec
    resp = env.client.get("/rag/search", params={"q": "   "})
    assert resp.status_code == 200
    assert resp.json()["results"] == []
    assert rec.search_calls == []


# ──────────────────────────────────────────────────────────────────────────────
# POST /chat — per-answer codebook attribution (v1.3); GPU-free via stubbed keying.gate
# ──────────────────────────────────────────────────────────────────────────────
def test_chat_attribution_none_when_no_edit(env):
    """No edit installed (edit_active False) -> attribution is None, /chat still succeeds."""
    resp = env.client.post("/chat", json={"message": "hi"})
    assert resp.status_code == 200
    assert resp.json()["attribution"] is None


def test_chat_attribution_maps_slot_to_memory_on_hit(env, monkeypatch):
    """With an edit active and keying.gate over threshold, attribution maps the matched
    codebook slot back to the consolidated item via PROV_CODEBOOK_KEYS."""
    import keying
    from serving import model_host
    rec = env.rec
    rec.edit_active_val = True
    rec.current_model_val = SimpleNamespace(model=object())          # wrapper.model for keying.gate
    rec.cons_items = [
        _item(id="c1", text="对花生过敏", status="consolidated",
              provenance={PROV_CODEBOOK_KEYS: {"native": 1, "chat": 2}}),
    ]
    adapter = SimpleNamespace(hopfield_key_match_threshold=0.85)
    monkeypatch.setattr(model_host, "edit_module", lambda: adapter)
    monkeypatch.setattr(keying, "gate", lambda text, **kw: (0.91, 2))  # hits the chat slot (2)

    att = env.client.post("/chat", json={"message": "晚饭推荐?"}).json()["attribution"]
    assert att is not None
    assert att["hit"] is True
    assert att["similarity"] == 0.91 and att["threshold"] == 0.85 and att["slot"] == 2
    assert att["memory"] == {"id": "c1", "text": "对花生过敏"}


def test_chat_attribution_miss_below_threshold_no_memory(env, monkeypatch):
    """Similarity below threshold (and a slot owned by no item) -> hit False, memory None."""
    import keying
    from serving import model_host
    rec = env.rec
    rec.edit_active_val = True
    rec.current_model_val = SimpleNamespace(model=object())
    rec.cons_items = [
        _item(id="c1", text="对花生过敏", status="consolidated",
              provenance={PROV_CODEBOOK_KEYS: {"native": 1, "chat": 2}}),
    ]
    adapter = SimpleNamespace(hopfield_key_match_threshold=0.85)
    monkeypatch.setattr(model_host, "edit_module", lambda: adapter)
    monkeypatch.setattr(keying, "gate", lambda text, **kw: (0.42, 9))  # below thr, unowned slot

    att = env.client.post("/chat", json={"message": "天气?"}).json()["attribution"]
    assert att is not None
    assert att["hit"] is False and att["memory"] is None and att["slot"] == 9
