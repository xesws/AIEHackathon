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

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import MODULES (not symbols) so tests can monkeypatch the attributes we call.
# ``consolidate`` is aliased to ``consolidate_mod`` because the POST /consolidate handler
# below is itself named ``consolidate`` (would otherwise shadow the memory module).
import generate
from memory import buffer, rag_store, schema, store
from memory import consolidate as consolidate_mod
from serving import ingest, model_host, triggers


# --- request models ---------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    rag_off: bool = False


class ConsolidateRequest(BaseModel):
    # Optional / empty body: {} or {"trigger": "manual"}.
    trigger: str = "manual"


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
    )

    return {
        "reply": reply,
        "buffer_count": len(buffer.load_unconsolidated()),
        "learned": result["edit_ids"],
    }


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
    return {
        "buffer": [schema.to_dict(i) for i in buf],
        "consolidated": [schema.to_dict(i) for i in con],
        "counts": {"buffer": len(buf), "consolidated": len(con)},
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

    return app


# Module-level app so "uvicorn serving.app:app" works.
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serving.app:app", host="0.0.0.0", port=8077)
