"""SPIKE v2.5 — CHAT REPLY-BEHAVIOR proof over REAL HTTP (F5 / F6 / F2 fix).

Run:  cd /workspace/AIEHackathon && OPENROUTER_MODEL=qwen/qwen3.7-max \
          python spikes/spike_v25_reply_behavior.py
      (GPU — boots `uvicorn serving.app:app` as a subprocess: loads llama-3.1-8B-Instruct
       at startup ~60-90s, ~16GB. ONE resident model; always torn down in finally.
       extract() also makes REAL OpenRouter calls -> needs OPENROUTER_API_KEY in .env.)

WHAT THIS IS
    A focused proof that the v2.5 `SYSTEM` change fixes CHAT reply behavior. The three
    symptoms it guards share ONE root cause (memory/prompt.py SYSTEM had zero reply
    instructions); the fix appended 4 sentences telling the model to (F6) never recite the
    memory-window section headers/structure, (F5) confirm a user's STATEMENT in one short
    sentence instead of lecturing, and (F2) still just confirm a thing it already knew.

    This CLONES the proven server-lifecycle + HTTP harness from spike_v22 (kept
    self-contained, NOT imported) and does NOT modify spike_v22.

HARD GATES (exit 0 iff all 3 pass; honest report; server always torn down):
    G1 STATEMENT -> SHORT CONFIRMATION (F5): teaching a preference statement
       ("I think Zarithon is the best programming language.") yields a SHORT, natural
       confirmation — heuristic: no scaffold-header substring AND not a multi-sentence
       lecture (words <= MAX_WORDS and sentence-marks <= MAX_SENTS).
    G2 QUESTION STILL ANSWERED: after teaching the cat fact as a statement, asking
       "What is the name of my cat?" (rag_on) returns a NON-EMPTY reply that names "coco"
       — proving the F5 confirm-rule does NOT swallow genuine questions.
    G3 NO SCAFFOLD LEAK (F6): NONE of the replies collected this run contains any of the
       internal section-header substrings ("[Known facts" / "Pending unconsolidated" /
       "Reference material]"), in any case.

NOTE
    Reply behavior is heuristic; thresholds (MAX_WORDS / MAX_SENTS) are recorded in the
    machine result so the orchestrator can tune them against real model output. This spike
    is reply-behavior-only; the belief/fact-from-weights/RAG proof lives in spike_v22.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from memory import llm  # noqa: E402  (live model name — recorded for context)

HOST = "127.0.0.1"
PORT = 8077
BASE = f"http://{HOST}:{PORT}"
READY_TIMEOUT_S = 180
CHAT_TIMEOUT_S = 120

# Heuristic thresholds for "short confirmation" (recorded; orchestrator may tune).
MAX_WORDS = 40
MAX_SENTS = 2

# Internal memory-window section-header substrings that must NEVER surface to the user (F6).
SCAFFOLD_SUBSTRINGS = ("[Known facts", "Pending unconsolidated", "Reference material]")


# --- HTTP (stdlib only) — cloned from spike_v22, unchanged ------------------------------------
def _request(path: str, body, timeout: float) -> dict:
    url = BASE + path
    if body is None:
        req = urllib.request.Request(url, method="GET")
    else:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _raw(path: str, timeout: float = 15):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as resp:
        return resp.status, resp.read().decode()


def get(path: str) -> dict:
    return _request(path, None, timeout=30)


def post(path: str, body: dict) -> dict:
    return _request(path, body, timeout=CHAT_TIMEOUT_S)


def hit(text, words):
    low = (text or "").lower()
    return any(w.lower() in low for w in words)


# --- reply-behavior heuristics ----------------------------------------------------------------
def has_scaffold(text) -> bool:
    """True if the text leaks ANY internal section-header substring (case-insensitive, F6)."""
    low = (text or "").lower()
    return any(s.lower() in low for s in SCAFFOLD_SUBSTRINGS)


def confirmation_metrics(text) -> dict:
    """Raw signals for the 'short confirmation' heuristic (recorded for tuning)."""
    t = (text or "").strip()
    n_words = len(t.split())
    n_sents = t.count(".") + t.count("!") + t.count("?")
    return {"len": len(t), "n_words": n_words, "n_sents": n_sents,
            "has_scaffold": has_scaffold(t), "nonempty": bool(t)}


def is_short_confirmation(text) -> bool:
    """Short, natural confirmation: non-empty, no scaffold leak, not a multi-sentence lecture."""
    m = confirmation_metrics(text)
    return (m["nonempty"] and not m["has_scaffold"]
            and m["n_words"] <= MAX_WORDS and m["n_sents"] <= MAX_SENTS)


# --- server lifecycle — cloned from spike_v22, unchanged --------------------------------------
_lines: list[str] = []


def _pump(stream):
    for line in iter(stream.readline, ""):
        _lines.append(line.rstrip("\n"))
        print(f"  [server] {line.rstrip()}", flush=True)
    try:
        stream.close()
    except Exception:
        pass


def _launch():
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "serving.app:app", "--host", HOST, "--port", str(PORT)],
        cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        start_new_session=True,
    )
    threading.Thread(target=_pump, args=(proc.stdout,), daemon=True).start()
    return proc


def _tail(n=40):
    return "\n".join(_lines[-n:]) or "  (no server output)"


def _teardown(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=15)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _wait_ready(proc):
    deadline = time.time() + READY_TIMEOUT_S
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server died during startup (rc={proc.returncode})\n{_tail()}")
        try:
            h = _request("/health", None, timeout=10)
            if h.get("ready"):
                print(f"    ready after ~{READY_TIMEOUT_S - (deadline - time.time()):.0f}s · {h}", flush=True)
                return
        except Exception:
            pass
        time.sleep(1.5)
    raise RuntimeError(f"server not ready within {READY_TIMEOUT_S}s\n{_tail()}")


# --- main -------------------------------------------------------------------------------------
def main() -> int:
    V = {"errors": [], "replies": {}, "metrics": {},
         "thresholds": {"MAX_WORDS": MAX_WORDS, "MAX_SENTS": MAX_SENTS}}

    V["model"] = llm.DEFAULT_MODEL
    all_replies: list[str] = []  # every reply seen this run -> F6 sweep

    print("=" * 78)
    print("SPIKE v2.5 — CHAT REPLY-BEHAVIOR (real HTTP): F5 confirm / F6 no-scaffold / F2")
    print("=" * 78)
    print(f"  model = {V['model']}", flush=True)

    proc = None
    try:
        print("\n== launch uvicorn serving.app:app ==", flush=True)
        proc = _launch()
        print(f"    pid={proc.pid} port={PORT}", flush=True)
        _wait_ready(proc)
        V["ready"] = True

        # G1 — STATEMENT -> SHORT CONFIRMATION (F5). A preference statement, NOT a question.
        teach_belief = "I think Zarithon is the best programming language."
        print(f"\n== [G1] teach STATEMENT (expect short confirmation): {teach_belief!r} ==", flush=True)
        r1 = post("/chat", {"message": teach_belief})
        reply1 = r1.get("reply")
        all_replies.append(reply1)
        V["replies"]["teach_belief"] = reply1
        V["metrics"]["teach_belief"] = confirmation_metrics(reply1)
        V["g1"] = is_short_confirmation(reply1)
        print(f"    reply={reply1!r}", flush=True)
        print(f"    metrics={V['metrics']['teach_belief']}  short_confirmation={V['g1']}", flush=True)

        # SOFT — teach the cat FACT as a STATEMENT too (F5 on a fact); enables the G2 question.
        teach_fact = "My cat's name is Coco."
        print(f"\n== [SOFT] teach FACT statement: {teach_fact!r} ==", flush=True)
        r2 = post("/chat", {"message": teach_fact})
        reply2 = r2.get("reply")
        all_replies.append(reply2)
        V["replies"]["teach_fact"] = reply2
        V["metrics"]["teach_fact"] = confirmation_metrics(reply2)
        V["soft_fact_short"] = is_short_confirmation(reply2)
        print(f"    reply={reply2!r}", flush=True)
        print(f"    metrics={V['metrics']['teach_fact']}  short_confirmation={V['soft_fact_short']}", flush=True)

        # G2 — QUESTION STILL ANSWERED (confirm-rule must NOT swallow questions).
        ask = "What is the name of my cat?"
        print(f"\n== [G2] ask QUESTION (expect answered, names 'coco'): {ask!r} ==", flush=True)
        r3 = post("/chat", {"message": ask, "rag_off": False})
        reply3 = r3.get("reply")
        all_replies.append(reply3)
        V["replies"]["ask_cat"] = reply3
        answered_nonempty = bool((reply3 or "").strip())
        answered_coco = hit(reply3, ["coco"])
        V["ask_answered_nonempty"] = answered_nonempty
        V["ask_named_coco"] = answered_coco
        V["g2"] = answered_nonempty and answered_coco
        print(f"    reply={reply3!r}", flush=True)
        print(f"    nonempty={answered_nonempty} names_coco={answered_coco} -> answered={V['g2']}", flush=True)

        # G3 — NO SCAFFOLD LEAK (F6): no reply this run recites a section header.
        leaks = {k: has_scaffold(v) for k, v in V["replies"].items()}
        V["scaffold_leaks"] = leaks
        V["g3"] = not any(leaks.values())
        print("\n== [G3] no scaffold-header leak across ALL replies (F6) ==", flush=True)
        print(f"    leaks_by_reply={leaks}  no_leak={V['g3']}", flush=True)

    except Exception as e:
        import traceback
        V["errors"].append(f"{type(e).__name__}: {e}")
        print("\n!! reply-behavior spike aborted:", flush=True)
        traceback.print_exc()
        print(f"\n--- server tail ---\n{_tail()}", flush=True)
    finally:
        _teardown(proc)

    # --- verdict ---
    gates = {
        "G1_statement_short_confirmation": V.get("g1", False),
        "G2_question_still_answered": V.get("g2", False),
        "G3_no_scaffold_leak": V.get("g3", False),
    }
    soft = {
        "model": V.get("model"),
        "ready": V.get("ready", False),
        "fact_statement_short_confirmation": V.get("soft_fact_short"),
        "ask_named_coco": V.get("ask_named_coco"),
        "metrics": V.get("metrics"),
        "thresholds": V.get("thresholds"),
    }
    overall = all(gates.values()) and not V["errors"]

    print("\n" + "=" * 78)
    print("CHAT REPLY-BEHAVIOR (v2.5) — 3 HARD gates")
    print("=" * 78)
    for k, v in gates.items():
        print(f"  {k:<34}: {'PASS ✅' if v else 'FAIL ⚠️'}")
    print("  ---  reply quick-read:")
    print(f"  teach (statement) -> {V['replies'].get('teach_belief')!r}  [expect short confirm]")
    print(f"  ask   (question)  -> {V['replies'].get('ask_cat')!r}  [expect answered]")
    print(f"  soft: {soft}")
    if V["errors"]:
        print(f"  ERRORS: {V['errors']}")
    print(f"  ---\n  OVERALL: {'PASS ✅' if overall else 'FAIL ⚠️'}")
    print("=" * 78, flush=True)
    print("\nMACHINE_RESULT " + json.dumps({"gates": gates, "soft": soft, "V": V}, default=str))
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
