"""FastAPI serving surface. Orchestrates extract->buffer / consolidate / generate.

This is a PURE TRANSPORT layer: it only wraps already-verified functions and holds
NO business logic. Modules are imported and called via attribute (not symbol-bound)
so unit tests can monkeypatch them.

Endpoints:
    POST /chat        — run a turn: ingest (extract->buffer/rag) BEFORE generating a reply
    POST /consolidate — "Consolidate Now": run a consolidation pass over the buffer
    GET  /memories    — buffer + consolidated items (+ counts) for the UI

Startup (lifespan): load the base weights resident on GPU, then register the model
provider so ``memory.consolidate`` can reach the resident handle without importing serving.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

# Import MODULES (not symbols) so tests can monkeypatch the attributes we call.
# ``consolidate`` is aliased to ``consolidate_mod`` because the POST /consolidate handler
# below is itself named ``consolidate`` (would otherwise shadow the memory module).
import generate
import keying
from memory import buffer, extract, rag_store, schema, store
from memory import consolidate as consolidate_mod
from serving import ingest, model_host, triggers


# --- request models ---------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    rag_off: bool = False


class ConsolidateRequest(BaseModel):
    # Optional / empty body: {} or {"trigger": "manual"}.
    trigger: str = "manual"


class ConsolidateItemRequest(BaseModel):
    id: str


class RouteRequest(BaseModel):
    route: str


class PatchRequest(BaseModel):
    text: str


class EditModuleRequest(BaseModel):
    on: bool


# --- handlers (module-level so they stay importable/testable) ---------------------------
def chat(payload: ChatRequest) -> dict:
    """POST /chat — ingest (write) BEFORE generate (read); "write-then-read"."""
    message = payload.message

    # 1) ingest first: edit-route -> buffer, rag-route -> rag_store.
    result = ingest.ingest([{"role": "user", "content": message}])

    # 2) rag-off gates ONLY the rag window: rag_hits=[] but KEEP the buffer + with_rag default.
    rag_hits = [] if payload.rag_off else rag_store.search(message, k=5)

    # 3) generate on the resident (possibly edited) model, conditioned on buffer + rag_hits.
    reply = generate.generate(
        message,
        model=model_host.current_model(),
        buffer=buffer.load_unconsolidated(),
        rag_hits=rag_hits,
        use_chat_template=True,
        tok=model_host.tokenizer(),
        # generate.py default is 16 (HoReN eval-sized) -> chat replies got cut mid-sentence.
        # The model stops at the chat EOS on its own; this is just a ceiling big enough to
        # let a normal reply finish. Bigger = slower (HoReN decodes with use_cache=False).
        max_new_tokens=160,
        # Anti-repetition for the CONVERSATIONAL path only (eval/proof greedy keeps HF defaults):
        # a strong edit hit + greedy + use_cache=False can loop a subword ("Zarithonononon…").
        # These two logits processors break the loop; do_sample stays False -> still deterministic.
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
    )

    return {
        "reply": reply,
        "buffer_count": len(buffer.load_unconsolidated()),
        "learned": result["edit_ids"],
        "retrieved": [schema.to_dict(h) for h in rag_hits],
        "extracted": result["n_extracted"],
        "rag_indexed": result["n_rag_indexed"],
        # per-answer codebook attribution (honest: ONE retrieval decision per answer);
        # None when editOn off / no edit installed / computation fails (best-effort).
        "attribution": attribution(message),
    }


def attribution(message: str) -> dict | None:
    """Per-answer HoReN codebook attribution for ``message`` (honest: ONE decision per answer).

    Under the live decode path (``use_cache=False`` + ``key_id`` pinned to the prompt) the adapter
    makes a single retrieval decision that every generated token shares, so attribution is
    per-answer, not per-token. Returns ``{hit, similarity, threshold, slot, memory}`` where
    ``memory`` is the consolidated item whose codebook row the query matched (via
    ``PROV_CODEBOOK_KEYS``), or ``None``. Returns ``None`` when no edit is installed or anything
    fails — never raises, never breaks /chat.
    """
    if not model_host.edit_active():
        return None
    try:
        adapter = model_host.edit_module()
        wrapper = model_host.current_model()  # HOREN wrapper after an edit; .model is the HF model
        sim, slot = keying.gate(message, hf_model=wrapper.model, tok=model_host.tokenizer(), adapter=adapter)
        thr = float(getattr(adapter, "hopfield_key_match_threshold", 0.85))
        memory = None
        for it in store.by_status("consolidated"):
            keys = (it.provenance or {}).get(schema.PROV_CODEBOOK_KEYS) or {}
            if slot in (keys.get("native"), keys.get("chat")):
                memory = {"id": it.id, "text": it.text}
                break
        return {"hit": sim > thr, "similarity": sim, "threshold": thr, "slot": slot, "memory": memory}
    except Exception:
        return None


def rag_search(q: str, k: int = 5) -> dict:
    """GET /rag/search — semantic search over the RAG store (empty query -> no results)."""
    hits = rag_store.search(q, k) if q.strip() else []
    return {"results": [schema.to_dict(h) for h in hits]}


def consolidate(payload: ConsolidateRequest | None = None) -> dict:
    """POST /consolidate — trigger one consolidation pass; return n_written + buffer_count."""
    n_written = triggers.manual()
    return {
        "n_written": n_written,
        "buffer_count": len(buffer.load_unconsolidated()),
    }


def memories() -> dict:
    """GET /memories — buffer + consolidated items (+ counts) for the UI."""
    buf = store.by_status("buffer")
    con = store.by_status("consolidated")
    rag = store.rag_all()  # materialize once (was called twice)
    return {
        "buffer": [schema.to_dict(i) for i in buf],
        "consolidated": [schema.to_dict(i) for i in con],
        "rag": [schema.to_dict(it) for it, _vec in rag],
        "counts": {"buffer": len(buf), "consolidated": len(con), "rag": len(rag)},
    }


def consolidate_item(payload: ConsolidateItemRequest) -> dict:
    """POST /consolidate/item — consolidate a single buffer item by id."""
    return {
        "n_written": triggers.manual(ids=[payload.id]),
        "buffer_count": len(buffer.load_unconsolidated()),
    }


def drop_memory(item_id: str) -> dict:
    """POST /memories/{item_id}/drop — discard a buffer item (must still be in buffer)."""
    it = store.get(item_id)
    if it is None or it.status != "buffer":
        raise HTTPException(404, "buffer item not found")
    buffer.drop([item_id])
    return {"ok": True, "buffer_count": len(buffer.load_unconsolidated())}


def route_memory(item_id: str, payload: RouteRequest) -> dict:
    """POST /memories/{item_id}/route — re-route a buffer item to rag (rag only)."""
    if payload.route != "rag":
        raise HTTPException(400, "only route=rag supported")
    it = store.get(item_id)
    if it is None or it.status != "buffer":
        raise HTTPException(404, "buffer item not found")
    # Drop from the buffer FIRST (while the status guard still holds), then index into rag.
    buffer.drop([item_id])
    it.route = "rag"
    rag_store.add(it)
    return {"ok": True, "buffer_count": len(buffer.load_unconsolidated())}


def patch_memory(item_id: str, payload: PatchRequest) -> dict:
    """PATCH /memories/{item_id} — edit a buffer item's text and re-decompose it."""
    it = store.get(item_id)
    if it is None or it.status != "buffer":
        raise HTTPException(404, "buffer item not found")
    it.text = payload.text
    d = extract.decompose(payload.text)
    if d:
        it.provenance = {**(it.provenance or {}), schema.PROV_EDIT: d}
    store.upsert(it)
    return {"item": schema.to_dict(it)}


def edit_module(payload: EditModuleRequest) -> dict:
    """POST /edit-module — enable/disable the recorded edit adapter on the resident model."""
    if payload.on:
        adapter = model_host.recorded_adapter()
        if adapter is None:
            raise HTTPException(409, "no edit module to enable")
        model_host.swap_edit_module(adapter)
    else:
        model_host.swap_edit_module(None)
    return {"on": payload.on}


def health() -> dict:
    """GET /health — readiness + edit state + memory counts."""
    c = {
        "buffer": len(store.by_status("buffer")),
        "consolidated": len(store.by_status("consolidated")),
        "rag": len(store.rag_all()),
    }
    return {
        "ready": model_host.current_model() is not None,
        "edit_on": model_host.edit_active(),
        "counts": c,
    }


# --- app factory ------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load base weights (~90s, ~16GB) then wire the model provider for consolidation."""
    model_host.load_base()
    consolidate_mod.set_model_provider(lambda: model_host.current_model())
    yield


def create_app() -> FastAPI:
    """Build and wire the FastAPI app: permissive CORS, lifespan startup, 3 routes."""
    app = FastAPI(title="Engram serving", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Every error surfaces as {"error": ...}; FastAPI's HTTPException subclasses the Starlette
    # one, so this pair covers raised 4xx (404/400/409) and any unexpected 500 — never a
    # fake-success body.
    @app.exception_handler(RequestValidationError)
    async def _validation_error(request, exc):
        # Malformed request body -> same {"error": ...} envelope (not FastAPI's {"detail": [...]}).
        return JSONResponse(status_code=422, content={"error": str(exc)})

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request, exc):
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

    @app.exception_handler(Exception)
    async def _unexpected_error(request, exc):
        return JSONResponse(status_code=500, content={"error": str(exc)})

    # Register the module-level handlers as routes (thin wrappers keep response shapes exact).
    @app.post("/chat")
    def _chat(payload: ChatRequest) -> dict:
        return chat(payload)

    @app.post("/consolidate")
    def _consolidate(payload: ConsolidateRequest | None = None) -> dict:
        return consolidate(payload)

    @app.get("/memories")
    def _memories() -> dict:
        return memories()

    @app.post("/consolidate/item")
    def _consolidate_item(payload: ConsolidateItemRequest) -> dict:
        return consolidate_item(payload)

    @app.post("/memories/{item_id}/drop")
    def _drop_memory(item_id: str) -> dict:
        return drop_memory(item_id)

    @app.post("/memories/{item_id}/route")
    def _route_memory(item_id: str, payload: RouteRequest) -> dict:
        return route_memory(item_id, payload)

    @app.patch("/memories/{item_id}")
    def _patch_memory(item_id: str, payload: PatchRequest) -> dict:
        return patch_memory(item_id, payload)

    @app.post("/edit-module")
    def _edit_module(payload: EditModuleRequest) -> dict:
        return edit_module(payload)

    @app.get("/health")
    def _health() -> dict:
        return health()

    @app.get("/rag/search")
    def _rag_search(q: str = "", k: int = 5) -> dict:
        return rag_search(q, k)

    # Serve the static frontend from THIS app (single origin) so the SPA + the 3 API routes
    # share one host/port — no CORS, no mixed-content, nothing to configure for the browser.
    # Mounted last so the API routes above take precedence; html=True serves index.html at "/".
    _frontend = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
    if os.path.isdir(_frontend):
        app.mount("/", StaticFiles(directory=_frontend, html=True), name="frontend")

    return app


# Module-level app so "uvicorn serving.app:app" works.
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serving.app:app", host="0.0.0.0", port=8077)
