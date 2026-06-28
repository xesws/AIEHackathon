"""SPIKE v0.10 — SERVING ROUTES SMOKE: the v1.0 net-new endpoints OVER REAL HTTP.

Run:  cd /workspace/AIEHackathon && python spikes/spike_v10_serving_routes.py
      (GPU — launches `uvicorn serving.app:app`, which loads llama-3.1-8B-Instruct
       at startup (~60-90s, ~16GB). Do NOT run in CI / on CPU. ONE resident model:
       this smoke always tears the server down so no orphan GPU process is left.)

Where spike_v08 proves the hero loop (chat -> consolidate -> answer-from-weights), THIS
smoke proves the v1.0 EXTENDED serving surface end-to-end over the real HTTP transport: a
`/health` readiness gate, the widened `/chat` (retrieved / extracted / rag_indexed), the
`/memories` rag panel, PER-ITEM consolidation (`/consolidate/item`), the edit-module
toggle (`/edit-module`), and the three buffer-item operations (drop / re-word PATCH /
demote-to-rag). It boots the FastAPI app as a SUBPROCESS, polls GET /health until the
model has loaded, then drives:

    [1] GET  /health                          -> ready + counts present
    [2] POST /chat   (teach JQ / nickel)      -> buffer_count>=1, retrieved[], extracted>=1
    [3] GET  /memories                        -> buffer/consolidated/rag(+counts); grab a buf id
    [4] POST /consolidate/item {id}           -> EXACTLY one item folded (buffer_count-1)
    [5] POST /edit-module {on:false}/{on:true}-> hot-swap base <-> edit, each echoes {on}
    [6] teach again, then drop / PATCH / route a fresh buffer id apiece

Each step prints PASS/FAIL independently (one failure does NOT abort the rest), then an
overall VERDICT line. ZERO extra deps: stdlib urllib + json for HTTP; subprocess/signal
for the server. The whole run is wrapped in try/finally so the server is ALWAYS killed
(process-group SIGTERM then SIGKILL) — see _teardown(). Exit 0 iff every step passes.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request

# repo root on path (so a bare `python spikes/...` invocation resolves the package layout
# the same way the other spikes do) and as the subprocess cwd for `serving.app:app`.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

HOST = "127.0.0.1"
PORT = 8088                       # unique port for this smoke
BASE = f"http://{HOST}:{PORT}"
READY_TIMEOUT_S = 200             # model load is ~60-90s; /health refused until lifespan done
CHAT_TIMEOUT_S = 120              # /chat + /consolidate/item run on the GPU
TEACH_MSG_1 = "Hey, I am JQ and I have a severe contact allergy to nickel buckles."
TEACH_MSG_2 = ("By the way, I live in Berlin, I work as a data engineer, "
               "and I strongly prefer Rust over Go.")
PATCH_TEXT = "JQ is allergic to nickel"


# --- HTTP helpers (stdlib only) -------------------------------------------------------------
def _request(path: str, body: dict | None, timeout: float, method: str) -> dict:
    """One JSON request/response. ``method`` is explicit so PATCH works alongside GET/POST."""
    url = BASE + path
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {} if body is None else {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get(path: str) -> dict:
    return _request(path, None, timeout=30, method="GET")


def post(path: str, body: dict) -> dict:
    return _request(path, body, timeout=CHAT_TIMEOUT_S, method="POST")


def patch(path: str, body: dict) -> dict:
    return _request(path, body, timeout=CHAT_TIMEOUT_S, method="PATCH")


# --- server lifecycle -----------------------------------------------------------------------
_server_lines: list[str] = []  # tee'd server stdout+stderr, kept for crash diagnostics


def _pump(stream) -> None:
    """Tee the server's merged stdout/stderr: print live AND retain for diagnostics."""
    for line in iter(stream.readline, ""):
        line = line.rstrip("\n")
        _server_lines.append(line)
        print(f"  [server] {line}", flush=True)
    try:
        stream.close()
    except Exception:
        pass


def _launch_server() -> subprocess.Popen:
    """Start `uvicorn serving.app:app` in its own process group, output tee'd."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "serving.app:app",
         "--host", HOST, "--port", str(PORT)],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,     # merge so the tee shows the full picture
        text=True,
        bufsize=1,
        start_new_session=True,        # own process group -> we can kill the whole tree
    )
    threading.Thread(target=_pump, args=(proc.stdout,), daemon=True).start()
    return proc


def _server_tail(n: int = 50) -> str:
    return "\n".join(_server_lines[-n:]) or "  (no server output captured)"


def _teardown(proc: subprocess.Popen | None) -> None:
    """ALWAYS called in finally. SIGTERM the process group, then SIGKILL if still alive,
    so neither the uvicorn server nor its resident GPU model is ever left orphaned."""
    if proc is None or proc.poll() is not None:
        return
    # 1) graceful: terminate the whole group (uvicorn + any children).
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=15)
        return
    except subprocess.TimeoutExpired:
        pass
    # 2) force: kill the group, then the proc, and reap.
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _wait_ready(proc: subprocess.Popen) -> None:
    """Poll GET /health until HTTP 200. Fail loudly if the proc dies or we time out."""
    deadline = time.time() + READY_TIMEOUT_S
    start = time.time()
    while time.time() < deadline:
        if proc.poll() is not None:
            print("\n!! server process exited during startup (rc="
                  f"{proc.returncode}). captured output:\n{_server_tail()}", flush=True)
            raise RuntimeError(f"server died during startup (rc={proc.returncode})")
        try:
            _request("/health", None, timeout=10, method="GET")
            print(f"    server ready after {time.time() - start:.1f}s", flush=True)
            return
        except Exception:
            time.sleep(1.5)
    print(f"\n!! server not ready within {READY_TIMEOUT_S}s. captured output:\n"
          f"{_server_tail()}", flush=True)
    raise RuntimeError(f"server not ready within {READY_TIMEOUT_S}s")


# --- step bookkeeping -----------------------------------------------------------------------
_results: list[tuple[str, bool, str]] = []  # (name, ok, detail) in run order


def _record(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    tag = "PASS ✅" if ok else "FAIL ❌"
    line = f"    [{tag}] {name}"
    if detail:
        line += f"  — {detail}"
    print(line, flush=True)


def _require_keys(d: dict, keys, ctx: str) -> None:
    missing = [k for k in keys if k not in d]
    assert not missing, f"{ctx}: missing key(s) {missing} in {d}"


# --- main -----------------------------------------------------------------------------------
def main() -> int:
    proc: subprocess.Popen | None = None
    fatal: str | None = None
    buf_id: str | None = None
    prev_buffer: int | None = None

    try:
        print("== [0] launch uvicorn serving.app:app (subprocess) ==", flush=True)
        proc = _launch_server()
        print(f"    pid={proc.pid}  port={PORT}  cwd={REPO_ROOT}", flush=True)

        print(f"== [0b] poll GET /health until ready (<= {READY_TIMEOUT_S}s) ==", flush=True)
        _wait_ready(proc)

        # [1] HEALTH -------------------------------------------------------------------------
        print("== [1] GET /health (readiness gate + counts) ==", flush=True)
        try:
            h = get("/health")
            print(f"    health       : {h}", flush=True)
            assert h.get("ready") is True, f"ready is not true: {h.get('ready')!r}"
            counts = h.get("counts") or {}
            _require_keys(counts, ("buffer", "consolidated", "rag"), "/health counts")
            _record("health: ready + counts", True)
        except Exception as e:
            _record("health: ready + counts", False, str(e))

        # [2] CHAT (teach) -------------------------------------------------------------------
        print("== [2] POST /chat (teach: JQ / nickel allergy) ==", flush=True)
        try:
            r = post("/chat", {"message": TEACH_MSG_1})
            print(f"    reply        : {r.get('reply')!r}", flush=True)
            print(f"    buffer_count : {r.get('buffer_count')}", flush=True)
            print(f"    extracted    : {r.get('extracted')}   "
                  f"rag_indexed: {r.get('rag_indexed')}   "
                  f"retrieved: {len(r.get('retrieved') or [])} item(s)", flush=True)
            _require_keys(r, ("reply", "buffer_count", "learned",
                              "retrieved", "extracted", "rag_indexed"), "/chat")
            assert isinstance(r["buffer_count"], int) and r["buffer_count"] >= 1, \
                f"buffer_count not >=1: {r['buffer_count']!r}"
            assert isinstance(r["retrieved"], list), "retrieved is not a list"
            assert isinstance(r["extracted"], int) and r["extracted"] >= 1, \
                f"extracted not >=1: {r['extracted']!r}"
            assert isinstance(r["rag_indexed"], int), \
                f"rag_indexed not an int: {r['rag_indexed']!r}"
            _record("chat: buffered + new fields", True)
        except Exception as e:
            _record("chat: buffered + new fields", False, str(e))

        # [3] MEMORIES -----------------------------------------------------------------------
        print("== [3] GET /memories (buffer/consolidated/rag + counts) ==", flush=True)
        try:
            m = get("/memories")
            _require_keys(m, ("buffer", "consolidated", "rag", "counts"), "/memories")
            _require_keys(m["counts"], ("buffer", "consolidated", "rag"), "/memories counts")
            assert isinstance(m["buffer"], list) and isinstance(m["rag"], list), \
                "buffer / rag are not lists"
            print(f"    counts       : {m['counts']}", flush=True)
            if m["buffer"]:
                buf_id = m["buffer"][0]["id"]
                prev_buffer = m["counts"]["buffer"]
                print(f"    first buf id : {buf_id}  (buffer_count={prev_buffer})", flush=True)
            _record("memories: rag panel + counts.rag", True)
        except Exception as e:
            _record("memories: rag panel + counts.rag", False, str(e))

        # [4] CONSOLIDATE ONE ITEM -----------------------------------------------------------
        print("== [4] POST /consolidate/item (fold EXACTLY one buffer item) ==", flush=True)
        try:
            assert buf_id is not None and prev_buffer is not None, \
                "no buffer id captured in step [3] (buffer was empty)"
            r = post("/consolidate/item", {"id": buf_id})
            print(f"    n_written    : {r.get('n_written')}   "
                  f"buffer_count: {r.get('buffer_count')}  (was {prev_buffer})", flush=True)
            _require_keys(r, ("n_written", "buffer_count"), "/consolidate/item")
            assert isinstance(r["n_written"], int) and r["n_written"] >= 0, \
                f"n_written not >=0: {r['n_written']!r}"
            assert r["buffer_count"] == prev_buffer - 1, \
                f"buffer not drained by exactly 1: {r['buffer_count']} (expected {prev_buffer - 1})"
            _record("consolidate/item: exactly one folded", True)
        except Exception as e:
            _record("consolidate/item: exactly one folded", False, str(e))

        # [5] EDIT-MODULE TOGGLE -------------------------------------------------------------
        print("== [5] POST /edit-module (hot-swap base <-> edit) ==", flush=True)
        try:
            off = post("/edit-module", {"on": False})
            print(f"    {{on:false}} -> {off}", flush=True)
            assert off.get("on") is False, f"off did not echo on=false: {off}"
            on = post("/edit-module", {"on": True})
            print(f"    {{on:true}}  -> {on}", flush=True)
            assert on.get("on") is True, f"on did not echo on=true: {on}"
            _record("edit-module: off/on round-trip", True)
        except Exception as e:
            _record("edit-module: off/on round-trip", False, str(e))

        # [6] BUFFER-ITEM OPS: drop / PATCH / route ------------------------------------------
        print("== [6] teach again, then drop / PATCH / route on fresh buffer ids ==", flush=True)
        try:
            post("/chat", {"message": TEACH_MSG_2})
        except Exception as e:
            print(f"    (warning) second teach failed: {e}", flush=True)
        try:
            m2 = get("/memories")
            buf_ids = [it["id"] for it in m2.get("buffer", [])]
            print(f"    buffer ids   : {buf_ids}", flush=True)
        except Exception as e:
            buf_ids = []
            print(f"    (warning) could not list buffer ids: {e}", flush=True)

        # 6a) drop a fresh buffer id
        id_drop = buf_ids[0] if len(buf_ids) >= 1 else None
        try:
            assert id_drop is not None, "no buffer id available for drop"
            r = post(f"/memories/{id_drop}/drop", {})
            print(f"    drop {id_drop} -> {r}", flush=True)
            _require_keys(r, ("ok", "buffer_count"), "/drop")
            assert r["ok"] is True, f"drop ok not true: {r}"
            assert isinstance(r["buffer_count"], int), "drop buffer_count not an int"
            _record("drop: discard buffered item", True)
        except Exception as e:
            _record("drop: discard buffered item", False, str(e))

        # 6b) PATCH (re-word) a DIFFERENT fresh buffer id
        id_patch = next((i for i in buf_ids if i != id_drop), None)
        try:
            assert id_patch is not None, "no distinct buffer id available for PATCH"
            r = patch(f"/memories/{id_patch}", {"text": PATCH_TEXT})
            print(f"    patch {id_patch} -> {r}", flush=True)
            _require_keys(r, ("item",), "/memories PATCH")
            assert isinstance(r["item"], dict), "PATCH item is not a dict"
            assert r["item"].get("text") == PATCH_TEXT, \
                f"PATCH did not update text: {r['item'].get('text')!r}"
            _record("PATCH: re-word buffered item", True)
        except Exception as e:
            _record("PATCH: re-word buffered item", False, str(e))

        # 6c) route -> rag on yet another DIFFERENT fresh buffer id
        id_route = next((i for i in buf_ids if i not in (id_drop, id_patch)), None)
        try:
            assert id_route is not None, "no distinct buffer id available for route"
            r = post(f"/memories/{id_route}/route", {"route": "rag"})
            print(f"    route {id_route} -> {r}", flush=True)
            _require_keys(r, ("ok", "buffer_count"), "/route")
            assert r["ok"] is True, f"route ok not true: {r}"
            _record("route: demote buffered item to rag", True)
        except Exception as e:
            _record("route: demote buffered item to rag", False, str(e))

    except Exception as e:
        fatal = f"{type(e).__name__}: {e}"
        print("\n!! serving-routes smoke aborted with diagnostic:", flush=True)
        traceback.print_exc()
        if _server_lines:
            print(f"\n--- server tail ---\n{_server_tail()}", flush=True)
    finally:
        # ONE resident model — never leave the server (and its GPU memory) running.
        _teardown(proc)

    # --- VERDICT ----------------------------------------------------------------------------
    overall = bool(_results) and all(ok for _name, ok, _detail in _results) and fatal is None
    print("\n============== SPIKE v0.10 SERVING ROUTES SMOKE ==============", flush=True)
    for name, ok, detail in _results:
        tag = "PASS ✅" if ok else "FAIL ❌"
        suffix = f"  — {detail}" if (detail and not ok) else ""
        print(f"  {tag}  {name}{suffix}")
    if fatal:
        print(f"  FATAL: {fatal}")
    print("  ---")
    print(f"  SERVING-ROUTES VERDICT : {'PASS ✅' if overall else 'CHECK ⚠️'}")
    print("=============================================================", flush=True)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
