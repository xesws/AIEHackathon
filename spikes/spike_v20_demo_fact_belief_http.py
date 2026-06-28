"""SPIKE v2.0 — DEMO HERO LOOP over REAL HTTP (fact/belief/other routing).

Run:  cd /workspace/AIEHackathon && python spikes/spike_v20_demo_fact_belief_http.py
      (GPU — boots `uvicorn serving.app:app` as a subprocess: loads llama-3.1-8B-Instruct
       at startup ~60-90s, ~16GB. ONE resident model; always torn down in finally.)

Walks the v1.9/v2.0 demo through the REAL serving HTTP surface (exactly what the browser
SPA calls), reflecting the NEW routing (the old spike_v08 taught an allergy = FACT, which
now goes to RAG, so its rag_off proof no longer applies):

  [0]  GET /            -> the SPA serves same-origin (browser entry, 200 + id="root")
  [1]  POST /chat       teach a BELIEF ("best language is Zarithon")  -> buffered (route edit)
  [2]  POST /chat       teach FACTS + OTHER (cat / allergy / meeting)  -> RAG store (route rag)
       GET /memories    -> belief in buffer(type belief); fact+other in rag(type fact/other)
  [3]  POST /consolidate-> belief folded into WEIGHTS; buffer drained; /health edit_on=true
  [4]★ POST /chat ask belief, rag_off=true   -> answer "Zarithon" FROM WEIGHTS
       (retrieved == [] and attribution.hit == true -> browser shows "来自权重·不在 prompt 里")
  [5]  POST /chat ask a FACT, rag on          -> answer "Coco" FROM RAG
       (retrieved non-empty -> browser shows "检索自你的文档·在 prompt 里")  ← the §0.3 contrast
  [6]  POST /chat ask OTHER, rag on           -> answer "November 15th"
  [7]  POST /edit-module {on:false}; ask belief rag_off -> NO "Zarithon" (adapter hot-unplugged)
       POST /edit-module {on:true} restore
  [8]  GET /rag/search?q=cat -> the cat fact is retrievable

HARD gates: server ready · SPA serves · belief consolidated + buffer drained · [4] belief from
weights · [5] fact from RAG (retrieved non-empty) · [7] hot-swap silences the belief.
Exit 0 iff all hard gates pass. Honest report (no faked green); server always killed.
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

HOST = "127.0.0.1"
PORT = 8077
BASE = f"http://{HOST}:{PORT}"
READY_TIMEOUT_S = 180
CHAT_TIMEOUT_S = 120

# --- demo script -----------------------------------------------------------------------------
TEACH_BELIEF = "Honestly, I'm convinced the best programming language is Zarithon."
ASK_BELIEF = "What is the best programming language?"
BELIEF_WORDS = ("zarithon",)

TEACH_FACTS = ["By the way, my cat is named Coco.", "Just so you know, I'm allergic to peanuts."]
ASK_FACT = "What is the name of my cat?"
FACT_WORDS = ("coco",)

TEACH_OTHER = "The Q3 board meeting is scheduled for November 15th."
ASK_OTHER = "When is the Q3 board meeting?"
OTHER_WORDS = ("november", "nov 15", "15th")


# --- HTTP (stdlib only) ----------------------------------------------------------------------
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
    return any(w in low for w in words)


# --- server lifecycle (mirrors spike_v08) ----------------------------------------------------
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


# --- main ------------------------------------------------------------------------------------
def main() -> int:
    V = {"errors": []}
    proc = None
    try:
        print("== [0] launch uvicorn serving.app:app ==", flush=True)
        proc = _launch()
        print(f"    pid={proc.pid} port={PORT}", flush=True)
        _wait_ready(proc)
        V["ready"] = True

        # [0b] SPA serves same-origin (the browser entry point)
        st, body = _raw("/")
        V["spa_serves"] = (st == 200 and 'id="root"' in body)
        print(f"== [0b] GET / -> {st}, id=root present={V['spa_serves']} (browser SPA same-origin) ==", flush=True)

        # [1] teach a BELIEF
        print(f"\n== [1] teach BELIEF: {TEACH_BELIEF!r} ==", flush=True)
        r1 = post("/chat", {"message": TEACH_BELIEF})
        print(f"    reply={r1.get('reply')!r}", flush=True)
        print(f"    learned={r1.get('learned')} buffer_count={r1.get('buffer_count')} "
              f"rag_indexed={r1.get('rag_indexed')}", flush=True)
        mem = get("/memories")
        buf_types = sorted(b.get("type") for b in mem["buffer"])
        V["belief_buffered"] = "belief" in buf_types
        print(f"    /memories buffer types={buf_types}  rag count={mem['counts']['rag']}", flush=True)

        # [2] teach FACTS + OTHER
        print("\n== [2] teach FACTS + OTHER (cat / allergy / meeting) ==", flush=True)
        for msg in TEACH_FACTS + [TEACH_OTHER]:
            r = post("/chat", {"message": msg})
            print(f"    teach {msg!r} -> rag_indexed={r.get('rag_indexed')} learned={r.get('learned')}", flush=True)
        mem = get("/memories")
        rag_types = sorted(it.get("type") for it in mem["rag"])
        buf_types = sorted(b.get("type") for b in mem["buffer"])
        V["routing"] = {"rag_types": rag_types, "buffer_types": buf_types}
        V["fact_in_rag"] = "fact" in rag_types
        V["no_belief_in_rag"] = "belief" not in rag_types
        V["only_belief_in_buffer"] = all(t == "belief" for t in buf_types) and buf_types
        print(f"    rag types={rag_types}", flush=True)
        print(f"    buffer types={buf_types}  (INV-1 only belief buffers={V['only_belief_in_buffer']}, "
              f"INV-2 no belief in rag={V['no_belief_in_rag']})", flush=True)

        # [3] consolidate belief -> weights
        print("\n== [3] POST /consolidate (belief -> WEIGHTS) ==", flush=True)
        r3 = post("/consolidate", {})
        V["n_written"] = r3.get("n_written")
        V["buffer_after"] = r3.get("buffer_count")
        h = get("/health")
        V["edit_on"] = h.get("edit_on")
        print(f"    n_written={V['n_written']} buffer_after={V['buffer_after']} edit_on={V['edit_on']}", flush=True)
        V["consolidated_ok"] = (isinstance(V["n_written"], int) and V["n_written"] >= 1
                                and V["buffer_after"] == 0 and V["edit_on"] is True)

        # [4] ★ PROOF — belief from WEIGHTS (rag_off)
        print(f"\n== [4] ★ ask BELIEF rag_off: {ASK_BELIEF!r} ==", flush=True)
        r4 = post("/chat", {"message": ASK_BELIEF, "rag_off": True})
        attr = r4.get("attribution") or {}
        V["belief_reply"] = r4.get("reply")
        V["belief_from_weights"] = hit(r4.get("reply"), BELIEF_WORDS)
        V["belief_retrieved_empty"] = (r4.get("retrieved") == [])
        V["belief_attr_hit"] = bool(attr.get("hit"))
        print(f"    reply={V['belief_reply']!r}", flush=True)
        print(f"    answered_from_weights={V['belief_from_weights']}  retrieved_empty={V['belief_retrieved_empty']}  "
              f"attribution.hit={V['belief_attr_hit']} sim={attr.get('similarity')}", flush=True)

        # [5] CONTRAST — fact from RAG (rag on)
        print(f"\n== [5] ask FACT rag_on: {ASK_FACT!r}  (the §0.3 contrast) ==", flush=True)
        r5 = post("/chat", {"message": ASK_FACT, "rag_off": False})
        V["fact_reply"] = r5.get("reply")
        V["fact_from_rag"] = hit(r5.get("reply"), FACT_WORDS)
        V["fact_retrieved_nonempty"] = bool(r5.get("retrieved"))
        print(f"    reply={V['fact_reply']!r}", flush=True)
        print(f"    answered={V['fact_from_rag']}  retrieved_nonempty={V['fact_retrieved_nonempty']} "
              f"(n={len(r5.get('retrieved') or [])})", flush=True)

        # [6] OTHER from RAG
        print(f"\n== [6] ask OTHER rag_on: {ASK_OTHER!r} ==", flush=True)
        r6 = post("/chat", {"message": ASK_OTHER, "rag_off": False})
        V["other_reply"] = r6.get("reply")
        V["other_from_rag"] = hit(r6.get("reply"), OTHER_WORDS)
        print(f"    reply={V['other_reply']!r}  answered={V['other_from_rag']}", flush=True)

        # [7] hot-swap: pull the edit module, belief should fall silent
        print("\n== [7] hot-swap edit-module OFF -> ask belief rag_off (should NOT answer) ==", flush=True)
        post("/edit-module", {"on": False})
        r7 = post("/chat", {"message": ASK_BELIEF, "rag_off": True})
        V["belief_after_unplug"] = r7.get("reply")
        V["hotswap_silences"] = not hit(r7.get("reply"), BELIEF_WORDS)
        print(f"    reply={V['belief_after_unplug']!r}  silenced={V['hotswap_silences']}", flush=True)
        post("/edit-module", {"on": True})  # restore

        # [8] rag search
        print(f"\n== [8] GET /rag/search?q=cat ==", flush=True)
        rs = get("/rag/search?q=cat&k=5")
        tops = [it["text"] for it in rs.get("results", [])]
        V["rag_search_cat_ok"] = any("coco" in t.lower() for t in tops)
        print(f"    results={tops[:3]}", flush=True)

    except AssertionError as e:
        V["errors"].append(f"AssertionError: {e}")
        print(f"\n!! {e}", flush=True)
    except Exception as e:
        import traceback
        V["errors"].append(f"{type(e).__name__}: {e}")
        print("\n!! demo aborted:", flush=True)
        traceback.print_exc()
        print(f"\n--- server tail ---\n{_tail()}", flush=True)
    finally:
        _teardown(proc)

    # --- verdict ---
    gates = {
        "server_ready": V.get("ready", False),
        "spa_serves": V.get("spa_serves", False),
        "belief_consolidated": V.get("consolidated_ok", False),
        "PROOF_belief_from_weights": V.get("belief_from_weights", False) and V.get("belief_retrieved_empty", False),
        "CONTRAST_fact_from_rag": V.get("fact_from_rag", False) and V.get("fact_retrieved_nonempty", False),
        "hotswap_silences_belief": V.get("hotswap_silences", False),
    }
    soft = {
        "belief_attr_hit": V.get("belief_attr_hit"),
        "other_from_rag": V.get("other_from_rag"),
        "rag_search_cat": V.get("rag_search_cat_ok"),
        "routing": V.get("routing"),
        "no_belief_in_rag": V.get("no_belief_in_rag"),
        "only_belief_in_buffer": V.get("only_belief_in_buffer"),
    }
    overall = all(gates.values()) and not V["errors"]

    print("\n" + "=" * 78)
    print("DEMO HERO LOOP (fact/belief/other) — REAL HTTP")
    print("=" * 78)
    for k, v in gates.items():
        print(f"  {k:<28}: {'PASS ✅' if v else 'FAIL ⚠️'}")
    print(f"  ---  contrast quick-read:")
    print(f"  belief (rag_off) -> {V.get('belief_reply')!r}  [weights · not in prompt]")
    print(f"  fact   (rag_on)  -> {V.get('fact_reply')!r}  [RAG · retrieved→in prompt]")
    print(f"  soft: {soft}")
    if V["errors"]:
        print(f"  ERRORS: {V['errors']}")
    print(f"  ---\n  OVERALL: {'PASS ✅' if overall else 'CHECK ⚠️'}")
    print("=" * 78, flush=True)
    print("\nMACHINE_RESULT " + json.dumps({"gates": gates, "soft": soft, "V": V}, default=str))
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
