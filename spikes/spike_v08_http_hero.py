"""SPIKE v0.8 — HTTP HERO SMOKE: the whole Engram hero loop OVER REAL HTTP.

Run:  cd /workspace/AIEHackathon && python spikes/spike_v08_http_hero.py
      (GPU — launches `uvicorn serving.app:app`, which loads llama-3.1-8B-Instruct
       at startup (~60-90s, ~16GB). Do NOT run in CI / on CPU. ONE resident model:
       this smoke always tears the server down so no orphan GPU process is left.)

This is serving's single must-pass: it proves serving_design.md §5 through the real
HTTP surface (not in-process). It boots the FastAPI app as a SUBPROCESS, polls
GET /memories until the model has loaded, then drives:

    [1] POST /chat   (teach: "JQ ... allergy to nickel buckles")  -> facts buffered
    [2] POST /consolidate                                         -> facts -> WEIGHTS
    [3] POST /chat   (ask, rag_off=true)                          -> answer from WEIGHTS

Step [3] is the hard gate: with rag_off the docs window is empty and the buffer was
drained by [2], so the reply can ONLY come from the consolidated weights. We assert the
reply names the allergen (nickel / buckle) and is not a bare name echo.

ZERO extra deps: stdlib urllib + json for HTTP; subprocess/signal for the server. The
whole run is wrapped in try/finally so the server is ALWAYS killed (process-group SIGTERM
then SIGKILL) — see _teardown(). Exit 0 iff the hero loop passes over HTTP.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

# repo root on path (so a bare `python spikes/...` invocation resolves the package layout
# the same way the other spikes do) and as the subprocess cwd for `serving.app:app`.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

HOST = "127.0.0.1"
PORT = 8077                       # unique port for this smoke
BASE = f"http://{HOST}:{PORT}"
READY_TIMEOUT_S = 180             # model load is ~60-90s; refused/hanging until lifespan done
CHAT_TIMEOUT_S = 120              # /chat runs generation on the GPU
TEACH_MSG = "Hey, I'm JQ. Just so you know, I have a severe contact allergy to nickel buckles."
ASK_MSG = "What is JQ allergic to?"


# --- HTTP helpers (stdlib only) -------------------------------------------------------------
def _request(path: str, body: dict | None, timeout: float) -> dict:
    """One JSON request/response. body=None -> GET, else POST application/json."""
    url = BASE + path
    if body is None:
        req = urllib.request.Request(url, method="GET")
    else:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get(path: str) -> dict:
    return _request(path, None, timeout=30)


def post(path: str, body: dict) -> dict:
    return _request(path, body, timeout=CHAT_TIMEOUT_S)


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
    """Poll GET /memories until HTTP 200. Fail loudly if the proc dies or we time out."""
    deadline = time.time() + READY_TIMEOUT_S
    while time.time() < deadline:
        if proc.poll() is not None:
            print("\n!! server process exited during startup (rc="
                  f"{proc.returncode}). captured output:\n{_server_tail()}", flush=True)
            raise RuntimeError(f"server died during startup (rc={proc.returncode})")
        try:
            _request("/memories", None, timeout=10)
            print(f"    server ready after {time.time() - (deadline - READY_TIMEOUT_S):.1f}s",
                  flush=True)
            return
        except Exception:
            time.sleep(1.5)
    print(f"\n!! server not ready within {READY_TIMEOUT_S}s. captured output:\n"
          f"{_server_tail()}", flush=True)
    raise RuntimeError(f"server not ready within {READY_TIMEOUT_S}s")


# --- main -----------------------------------------------------------------------------------
def main() -> int:
    verdict = {
        "ready": False,
        "reply1": "", "buffer_count1": None, "learned": None, "buffered_ok": False,
        "n_written": None, "buffer_count2": None,
        "reply3": "", "step3_hit": False, "bare_name_echo": False,
        "n_written_ok": False, "drained_ok": False, "proof_ok": False,
        "error": None,
    }

    proc: subprocess.Popen | None = None
    try:
        print("== [0] launch uvicorn serving.app:app (subprocess) ==", flush=True)
        proc = _launch_server()
        print(f"    pid={proc.pid}  port={PORT}  cwd={REPO_ROOT}", flush=True)

        print(f"== [0b] poll GET /memories until ready (<= {READY_TIMEOUT_S}s) ==", flush=True)
        _wait_ready(proc)
        verdict["ready"] = True

        # [1] TEACH ---------------------------------------------------------------------------
        print("== [1] POST /chat (teach: JQ / nickel-buckle allergy) ==", flush=True)
        r1 = post("/chat", {"message": TEACH_MSG})
        verdict["reply1"] = r1.get("reply", "")
        verdict["buffer_count1"] = r1.get("buffer_count")
        verdict["learned"] = r1.get("learned")
        print(f"    reply        : {verdict['reply1']!r}", flush=True)
        print(f"    buffer_count : {verdict['buffer_count1']}", flush=True)
        print(f"    learned      : {verdict['learned']}", flush=True)
        # SOFT: two facts (name + allergy) should buffer. Report, do not gate.
        verdict["buffered_ok"] = isinstance(verdict["buffer_count1"], int) and \
            verdict["buffer_count1"] >= 2
        if not verdict["buffered_ok"]:
            print(f"    SOFT-CHECK ⚠️  expected buffer_count >= 2, got "
                  f"{verdict['buffer_count1']} (two facts expected: name + allergy)", flush=True)

        # [2] CONSOLIDATE ---------------------------------------------------------------------
        print("== [2] POST /consolidate (fold buffer -> WEIGHTS) ==", flush=True)
        r2 = post("/consolidate", {})
        verdict["n_written"] = r2.get("n_written")
        verdict["buffer_count2"] = r2.get("buffer_count")
        print(f"    n_written    : {verdict['n_written']}", flush=True)
        print(f"    buffer_count : {verdict['buffer_count2']}", flush=True)
        # HARD: at least one edit folded in; expect 2.
        assert isinstance(verdict["n_written"], int) and verdict["n_written"] >= 1, \
            f"consolidate wrote nothing (n_written={verdict['n_written']})"
        verdict["n_written_ok"] = True
        if verdict["n_written"] != 2:
            print(f"    NOTE: n_written={verdict['n_written']} (expected 2)", flush=True)
        # HARD: buffer fully drained (no double existence buffer+weights).
        assert verdict["buffer_count2"] == 0, \
            f"buffer not drained after consolidate (buffer_count={verdict['buffer_count2']})"
        verdict["drained_ok"] = True

        # [3] ASK (RAG OFF) — THE PROOF -------------------------------------------------------
        print("== [3] POST /chat (ask, rag_off=true) — PROOF: answer from WEIGHTS ==", flush=True)
        r3 = post("/chat", {"message": ASK_MSG, "rag_off": True})
        verdict["reply3"] = r3.get("reply", "")
        print(f"    reply        : {verdict['reply3']!r}", flush=True)
        low = verdict["reply3"].lower()
        verdict["step3_hit"] = any(w in low for w in ("nickel", "buckle"))
        verdict["bare_name_echo"] = low.strip() in {
            "jq", "jq.", "i'm jq", "i am jq", "you are jq", "you're jq", "your name is jq",
        }
        # HARD GATE: names the allergen AND is not a bare name echo.
        assert verdict["step3_hit"], \
            f"proof reply missing 'nickel'/'buckle': {verdict['reply3']!r}"
        assert not verdict["bare_name_echo"], \
            f"proof reply is a bare name echo, not the allergy: {verdict['reply3']!r}"
        verdict["proof_ok"] = True

    except AssertionError as e:
        verdict["error"] = f"AssertionError: {e}"
        print(f"\n!! hard assertion failed: {e}", flush=True)
    except Exception as e:
        import traceback
        verdict["error"] = f"{type(e).__name__}: {e}"
        print("\n!! HTTP smoke aborted with diagnostic:", flush=True)
        traceback.print_exc()
        if _server_lines:
            print(f"\n--- server tail ---\n{_server_tail()}", flush=True)
    finally:
        # ONE resident model — never leave the server (and its GPU memory) running.
        _teardown(proc)

    # --- VERDICT --------------------------------------------------------------------------
    overall = bool(
        verdict["ready"] and verdict["n_written_ok"] and verdict["drained_ok"]
        and verdict["proof_ok"] and verdict["error"] is None
    )
    print("\n==================== SPIKE v0.8 HTTP HERO SMOKE ====================", flush=True)
    print(f"  server ready                 : {verdict['ready']}")
    print(f"  [1] reply                    : {verdict['reply1']!r}")
    print(f"      buffer_count / learned   : {verdict['buffer_count1']} / {verdict['learned']}"
          f"   (>=2 buffered? {verdict['buffered_ok']})")
    print(f"  [2] n_written / buffer_count : {verdict['n_written']} / {verdict['buffer_count2']}"
          f"   (>=1? {verdict['n_written_ok']}  drained==0? {verdict['drained_ok']})")
    print(f"  [3] reply (rag_off)          : {verdict['reply3']!r}")
    print(f"      contains nickel/buckle?  : {verdict['step3_hit']}   "
          f"bare-name-echo? {verdict['bare_name_echo']}")
    if verdict["error"]:
        print(f"  ERROR                        : {verdict['error']}")
    print(f"  ---")
    print(f"  HERO-OVER-HTTP VERDICT       : {'PASS ✅' if overall else 'CHECK ⚠️'}")
    print("===================================================================", flush=True)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
