"""
Fake smoke test for the Engram frontend prototype (v0.6).

"Fake" = mock-only. It asserts three things, all without a real backend:
  1. the static artifact actually serves over HTTP (index.html + src/engram.jsx → 200),
  2. the frontend interaction logic is wired (10 state vars, curation handlers,
     three-layer moves, mock seed data, mount call),
  3. the prototype makes ZERO real backend / network calls — the "象征性留白"
     guarantee from docs/frontend/visual_rcs/DESIGN.md §0 / §4.

NOT a render smoke: this host has no node/browser, so React is not executed here.
Full render + interaction verification is manual — see docs/v0.6-frontend-init.md
("local preview: python3 -m http.server -d frontend 5173").
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


def test_serves_index_and_component(base_url):
    status_idx, body_idx = _get(base_url + "/")
    assert status_idx == 200
    assert 'id="root"' in body_idx

    status_eng, body_eng = _get(base_url + "/src/engram.jsx")
    assert status_eng == 200
    assert "function Engram(" in body_eng


def test_index_bootstrap_wiring():
    for marker in (
        'id="root"', "importmap", "react-dom", "lucide-react",
        "cdn.tailwindcss.com", "babel", "./src/engram.jsx",
        'type="text/babel"', 'data-type="module"',
    ):
        assert marker in INDEX_SRC, f"index.html missing bootstrap marker: {marker}"


# ----------------------- 2. interaction logic wired ----------------------- #
STATE_VARS = ["surface", "dev", "ragOn", "editOn", "input",
              "justCommitted", "messages", "weights", "buffer", "refs"]
HANDLERS = ["consolidate", "burnOne", "demoteOne", "discardOne",
            "editPending", "burnAll", "send"]
COMPONENTS = ["Mark", "Switch", "TokenAttribution", "LabPanel",
              "Layer", "MemorySurface", "ChatSurface", "Engram"]
SEED = ["w1", "w2", "w3", "对花生过敏", "OLTP 默认 Postgres",
        "b1", "b2", "r1", "r2"]


def test_all_components_defined():
    for name in COMPONENTS:
        assert f"function {name}(" in ENGRAM_SRC, f"component not defined: {name}"


def test_state_vars_declared():
    for v in STATE_VARS:
        assert f"const [{v}," in ENGRAM_SRC, f"state var not declared via useState: {v}"
    assert ENGRAM_SRC.count("useState(") >= len(STATE_VARS)


def test_curation_handlers_defined():
    for h in HANDLERS:
        assert f"const {h} = " in ENGRAM_SRC, f"handler not defined: {h}"


def test_three_layer_state_moves():
    # the three curation actions are transfers between buffer / weights / refs
    for setter in ("setWeights", "setBuffer", "setRefs"):
        assert setter in ENGRAM_SRC, f"missing layer setter: {setter}"
    assert "[...w, ...buffer.map" in ENGRAM_SRC, "burnAll buffer->weights move missing"
    assert "setBuffer([])" in ENGRAM_SRC, "burnAll should clear the buffer"


def test_mock_seed_data_present():
    for s in SEED:
        assert s in ENGRAM_SRC, f"missing mock seed: {s}"


def test_mount_call():
    assert 'createRoot(document.getElementById("root"))' in ENGRAM_SRC
    assert "render(<Engram" in ENGRAM_SRC


# --------------------- 3. zero backend / network calls -------------------- #
FORBIDDEN = ["fetch(", "XMLHttpRequest", "axios", "WebSocket", "EventSource",
             "localStorage", "sessionStorage", "import.meta.env",
             "/chat", "/consolidate", "/memories", "http://", "https://"]


def test_zero_backend_or_network_calls():
    """DESIGN §0: 0 网络请求、0 真实后端依赖 — the component is pure mock."""
    offenders = [tok for tok in FORBIDDEN if tok in ENGRAM_SRC]
    assert offenders == [], f"component must be mock-only (found backend/network refs): {offenders}"
