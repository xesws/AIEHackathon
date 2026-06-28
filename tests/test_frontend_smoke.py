"""
Smoke test for the Engram frontend (v1.1 — wired to serving/app.py).

Supersedes the v0.6 "mock-only" smoke: the prototype now talks to the real serving layer
(serving/app.py), so this asserts the WIRED contract instead of forbidding network calls:
  1. the self-contained artifact serves over HTTP (index.html + src/engram.jsx -> 200),
  2. the frontend is wired — fetch helpers + the serving endpoints + curation handlers,
  3. the committed index.html is IN SYNC with the source (rebuilt bundle inlined), and
  4. no client-side persistence (state is backend in-memory — DESIGN/INV: no localStorage).

NOT a render/e2e smoke: this host has no node/browser, so React is not executed and the
live hero loop (against a running backend + GPU) is verified manually. Rebuild the inlined
artifact with `python frontend/build.py` after editing src/engram.jsx.
"""
from __future__ import annotations

import functools
import http.server
import threading
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = REPO_ROOT / "frontend"
INDEX = FRONTEND / "index.html"
ENGRAM = FRONTEND / "src" / "engram.jsx"
BUNDLE = FRONTEND / "app.bundle.js"

INDEX_SRC = INDEX.read_text(encoding="utf-8") if INDEX.exists() else ""
ENGRAM_SRC = ENGRAM.read_text(encoding="utf-8") if ENGRAM.exists() else ""


# ----------------------------- serve fixture ------------------------------ #
class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # keep pytest output clean
        pass


@pytest.fixture(scope="module")
def base_url():
    handler = functools.partial(_QuietHandler, directory=str(FRONTEND))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _get(url: str):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.read().decode("utf-8")


# ------------------------------- 1. serves -------------------------------- #
def test_files_exist():
    assert INDEX.exists(), f"missing {INDEX}"
    assert ENGRAM.exists(), f"missing {ENGRAM}"
    assert BUNDLE.exists(), f"missing {BUNDLE} (run: python frontend/build.py)"


def test_serves_index_and_component(base_url):
    status_idx, body_idx = _get(base_url + "/")
    assert status_idx == 200
    assert 'id="root"' in body_idx

    status_eng, body_eng = _get(base_url + "/src/engram.jsx")
    assert status_eng == 200
    assert "function Engram(" in body_eng


def test_index_is_self_contained():
    # v0.9+ ships a single inlined file: no external scripts, no babel/importmap CDN bootstrap.
    assert 'id="root"' in INDEX_SRC
    assert "<script src=" not in INDEX_SRC, "index.html must inline everything (no external <script src=)"
    # No in-browser transpile / import-map bootstrap (the v0.6 path) — JSX is pre-bundled now.
    for absent in ("text/babel", "<script type=\"importmap\""):
        assert absent not in INDEX_SRC, f"index.html should no longer use {absent} (inlined now)"


# ----------------------- 2. interaction logic wired ----------------------- #
API_HELPERS = ["apiChat", "apiConsolidate", "apiConsolidateItem", "apiMemories",
               "apiDrop", "apiRoute", "apiPatch", "apiEditModule", "apiHealth", "apiRagSearch"]
ENDPOINTS = ["/chat", "/consolidate", "/consolidate/item", "/memories",
             "/drop", "/route", "/edit-module", "/health", "/rag/search"]
STATE_VARS = ["surface", "dev", "ragOn", "editOn", "input", "justCommitted",
              "booting", "backendErr", "sending", "consolidating",
              "serverInfo", "restartNotice", "hasRealConversation",
              "messages", "weights", "buffer", "refs"]
HANDLERS = ["refresh", "consolidate", "burnOne", "demoteOne", "discardOne",
            "editPending", "commitPending", "toggleEdit", "send"]
COMPONENTS = ["Mark", "Switch", "TokenAttribution", "LabPanel",
              "Layer", "MemorySurface", "ChatSurface", "Engram"]


def test_all_components_defined():
    for name in COMPONENTS:
        assert f"function {name}(" in ENGRAM_SRC, f"component not defined: {name}"


def test_state_vars_declared():
    for v in STATE_VARS:
        assert f"const [{v}," in ENGRAM_SRC, f"state var not declared via useState: {v}"


def test_curation_handlers_defined():
    for h in HANDLERS:
        assert f"const {h} = " in ENGRAM_SRC, f"handler not defined: {h}"


def test_api_helpers_and_fetch():
    assert "fetch(" in ENGRAM_SRC, "frontend should call the backend via fetch()"
    for h in API_HELPERS:
        assert f"function {h}(" in ENGRAM_SRC, f"api helper not defined: {h}"
    assert 'method: "PATCH"' in ENGRAM_SRC, "editPending commit should PATCH /memories/{id}"


def test_endpoints_referenced():
    for ep in ENDPOINTS:
        assert ep in ENGRAM_SRC, f"serving endpoint not referenced in source: {ep}"


def test_same_origin_default():
    # default API base = same origin ("") — FastAPI serves the SPA + API on one port.
    assert "window.__ENGRAM_API__" in ENGRAM_SRC
    assert 'const API = ' in ENGRAM_SRC


def test_refresh_consumes_all_three_layers():
    # refresh() pulls live state from /memories (no longer clobbers backend with mock seed).
    for field in ("data.consolidated", "data.buffer", "data.rag"):
        assert field in ENGRAM_SRC, f"refresh() should consume {field}"
    for setter in ("setWeights", "setBuffer", "setRefs"):
        assert setter in ENGRAM_SRC, f"missing layer setter: {setter}"


def test_mount_call():
    assert 'createRoot(document.getElementById("root"))' in ENGRAM_SRC
    assert "render(<Engram" in ENGRAM_SRC


# ------------------- 3. committed index.html is in sync ------------------- #
def test_index_inlines_current_wiring():
    """Guards against editing engram.jsx but forgetting `python frontend/build.py`."""
    for ep in ("/edit-module", "/consolidate/item", "/health", "/rag/search"):
        assert ep in INDEX_SRC, f"index.html missing wired endpoint {ep} — rebuild the bundle"
    for field in ("boot_id", "codebook_size", "edit_available"):
        assert field in INDEX_SRC, f"index.html missing health field {field} — rebuild the bundle"


# --------------------- v1.3 features: attribution + search ---------------- #
def test_v13_token_attribution_wired():
    """TokenAttribution is fed the REAL /chat.attribution (no mock anchorTokens)."""
    assert "resp.attribution" in ENGRAM_SRC, "/chat.attribution not threaded onto the message"
    assert "function TokenAttribution(" in ENGRAM_SRC
    assert "answerText" in ENGRAM_SRC, "TokenAttribution should render the real answer text"
    assert "anchorTokens" not in ENGRAM_SRC, "mock anchorTokens must be gone (real data now)"


def test_v13_reference_search_wired():
    """Reference box is a real semantic-search input on /rag/search (submit-triggered)."""
    assert "function apiRagSearch(" in ENGRAM_SRC
    assert "const runSearch = " in ENGRAM_SRC
    assert "searchResults" in ENGRAM_SRC and "searching" in ENGRAM_SRC
    assert "<input" in ENGRAM_SRC, "the static <span> search box must become a real <input>"


def test_frontend_hardens_busy_and_restart_states():
    """Model-affecting actions share a busy guard and the UI watches backend restart identity."""
    assert "const actionBusy = booting || sending || consolidating" in ENGRAM_SRC
    assert "if (!text || actionBusy) return" in ENGRAM_SRC
    assert "disabled={busy}" in ENGRAM_SRC
    assert "disabled={blocked}" in ENGRAM_SRC
    assert "bootIdRef.current !== nextBootId" in ENGRAM_SRC
    assert "setRestartNotice(true)" in ENGRAM_SRC
    assert "serverInfo.codebookSize" in ENGRAM_SRC
    assert "codebook rows={codebookK}" in ENGRAM_SRC
    assert "hasRealConversation ? [...m, { role: \"user\", text }] : [{ role: \"user\", text }]" in ENGRAM_SRC


# --------------------- 4. no client-side persistence ---------------------- #
FORBIDDEN = ["XMLHttpRequest", "axios", "localStorage", "sessionStorage", "import.meta.env"]


def test_no_client_persistence():
    """State lives in the backend (in-memory). The client keeps none — no local storage."""
    offenders = [tok for tok in FORBIDDEN if tok in ENGRAM_SRC]
    assert offenders == [], f"frontend must not persist client-side: {offenders}"
