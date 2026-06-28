"""GPU/state fixture for the eval harness — the one place that touches the live model.

Wraps the real Engram seams (``serving.model_host`` / ``editing`` / ``generate`` /
``memory.*``) with the small set of operations every rung needs: boot once, reset to a
clean codebook, install knowledge (edit / rag / buffer+consolidate), generate, and the
HoReN deferral live-score.

★ Reset primitive (see docs/v1.2-eval-harness.md): ``model_host.swap_edit_module(None)``
restores the pristine ``nn.Linear`` captured once at ``load_base`` (``_S["original"]``), so
the NEXT ``editing.edit`` unwraps to base HF and rebuilds a fresh codebook from size 1.
This is uniform for single + sequential rungs and costs microseconds. ``load_base`` (~90s)
is boot-once + fallback only. ``reload_base`` does not exist; ``editing.edit``'s returned
``reset`` callable is a no-op after the first sequential edit — do not use it.

Codebook math with ``key_mode="chat"``: after N stacked edits, ``codebook_size == 1 + 2*N``
(1 placeholder + native-raw-key + appended-chat-key per fact).
"""
from __future__ import annotations

import time
from typing import Any, Optional

import editing
import generate as gen
import serving.model_host as model_host
from keying import compute_key
from memory import buffer, consolidate, rag_store, store
from memory.schema import PROV_EDIT, MemoryItem
from serving import triggers

_STATE: dict = {"booted": False, "k": 5, "threshold": None}
_ID = {"n": 0}


def _next_id(prefix: str) -> str:
    _ID["n"] += 1
    return f"eval_{prefix}_{_ID['n']}"


# --------------------------------------------------------------------------- #
# Boot / config
# --------------------------------------------------------------------------- #
def boot(*, k: int = 5) -> float:
    """Load base weights once (idempotent within a process) and wire the model provider.

    Returns the HoReN deferral threshold (``hparams.hopfield_key_match_threshold``)."""
    if not _STATE["booted"]:
        model_host.load_base()
        consolidate.set_model_provider(lambda: model_host.current_model())
        _STATE["booted"] = True
    _STATE["k"] = k
    _STATE["threshold"] = float(model_host.hparams().hopfield_key_match_threshold)
    return _STATE["threshold"]


def threshold() -> float:
    return _STATE["threshold"]


def k() -> int:
    return _STATE["k"]


def token_counter():
    """An injected token counter for metrics.count_prompt_tokens (HF Llama vocab)."""
    tok = model_host.tokenizer()
    return lambda s: len(tok.encode(s, add_special_tokens=False))


# --------------------------------------------------------------------------- #
# Reset
# --------------------------------------------------------------------------- #
def _slot_clean() -> bool:
    """The inner_params slot is the pristine nn.Linear (no codebook) -> no edit installed."""
    return not hasattr(model_host.edit_module(), "keys")


def clean_all() -> None:
    """Uniform reset: remove any installed edit + clear buffer/rag. Call at each setup.

    Asserts the codebook is actually gone (else the swap primitive is unreliable and the
    caller should fall back to ``reload_base``)."""
    model_host.swap_edit_module(None)
    store.reset()
    rag_store.reset()
    if not _slot_clean():
        raise RuntimeError("codebook not clean after swap_edit_module(None)")


def reload_base() -> None:
    """Heavy fallback (~90s): full base reload + provider rewire + store reset."""
    model_host.load_base()
    consolidate.set_model_provider(lambda: model_host.current_model())
    store.reset()
    rag_store.reset()
    _STATE["threshold"] = float(model_host.hparams().hopfield_key_match_threshold)


# --------------------------------------------------------------------------- #
# Install knowledge
# --------------------------------------------------------------------------- #
def do_edit(stem: str, target_new: str, subject: Optional[str] = None) -> dict:
    """One direct HoReN edit on the resident model (stem passed verbatim, incl. cloze ``___``)."""
    req = {"prompt": stem, "target_new": target_new}
    if subject:
        req["subject"] = subject
    return editing.edit(model_host.current_model(), req, key_mode="chat")


def buffer_edit(stem: str, target_new: str, subject: str = "JQ") -> str:
    """Seed ONE edit-route item into the buffer (bypasses the extractor), as if extract+router
    had already decomposed it. ``consolidate.run_pass`` consumes these. Returns the item id."""
    iid = _next_id("edit")
    buffer.append(
        MemoryItem(
            id=iid, type="fact", text=stem, route="edit", status="buffer",
            source="eval", ts=time.time(),
            provenance={PROV_EDIT: {"stem": stem, "target": target_new, "subject": subject}},
        )
    )
    return iid


def add_rag(text: str) -> str:
    """Index one natural-language rag_doc into the permanent RAG store. Returns the item id."""
    iid = _next_id("rag")
    rag_store.add(
        MemoryItem(
            id=iid, type="fact", text=text, route="rag", status="buffer",
            source="eval", ts=time.time(), provenance=None,
        )
    )
    return iid


def ingest_sentence(sentence: str) -> dict:
    """Run the REAL extractor+router over one natural sentence (P4). Routes to buffer/rag."""
    from serving import ingest as ingest_mod
    return ingest_mod.ingest([{"role": "user", "content": sentence}])


def consolidate_now() -> int:
    """Run one real consolidation pass over the buffer (dedup -> editing.edit). Returns n_written."""
    return triggers.manual()


# --------------------------------------------------------------------------- #
# Inference / scoring
# --------------------------------------------------------------------------- #
def codebook_size() -> int:
    em = model_host.edit_module()
    return len(em.keys) if hasattr(em, "keys") else 0


def search_rag(query: str, k_override: Optional[int] = None) -> list:
    return rag_store.search(query, _STATE["k"] if k_override is None else k_override)


def gen_answer(query: str, *, with_rag: bool, k_override: Optional[int] = None,
               max_new_tokens: int = 16) -> str:
    """Greedy generate on the resident model via the chat template. When ``with_rag`` the
    RAG window is filled from ``rag_store.search(query, k)``; otherwise it is empty."""
    rag_hits = search_rag(query, k_override) if with_rag else ()
    return gen.generate(
        query,
        model=model_host.current_model(),
        buffer=(),
        rag_hits=rag_hits,
        with_rag=with_rag,
        tok=model_host.tokenizer(),
        max_new_tokens=max_new_tokens,
        use_chat_template=True,
    )


def mnt_for(target_new: str, pad: int = 6) -> int:
    """Generation length budget for an A/efficacy probe: target tokens + a small pad."""
    tok = model_host.tokenizer()
    return len(tok.encode(" " + target_new, add_special_tokens=False)) + pad


def live_score(text: str) -> float:
    """HoReN deferral gate score for ``text`` (chat query-span key vs the installed codebook).
    Returns 0.0 when no edit is installed. Compare to ``threshold()`` (0.85)."""
    adapter = model_host.edit_module()
    if not hasattr(adapter, "keys"):
        return 0.0
    hf = model_host.current_model().model
    rk = compute_key(text, templated=True, hf_model=hf, tok=model_host.tokenizer(), adapter=adapter)
    return adapter._query(rk).max().item()
