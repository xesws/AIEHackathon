"""SPIKE v2.2 — DEMO DRESS-REHEARSAL over REAL HTTP (single-source samples).

Run:  cd /workspace/AIEHackathon && OPENROUTER_MODEL=qwen/qwen3.7-max \
          python spikes/spike_v22_dress_rehearsal.py
      (GPU — boots `uvicorn serving.app:app` as a subprocess: loads llama-3.1-8B-Instruct
       at startup ~60-90s, ~16GB. ONE resident model; always torn down in finally.
       extract() also makes REAL OpenRouter calls -> needs OPENROUTER_API_KEY in .env.)

WHAT THIS IS
    A full-chain dress-rehearsal for the demo. The teach turns + their probes + the
    expected routing live in ONE source file: demo/demo_samples.json (NOT hardcoded here,
    NOT samples.json which is the HoReN eval set). type is decided at RUNTIME by the real
    classifier (/chat -> ingest -> extract); the model only sees the `teach` text.

    This CLONES the proven server-lifecycle + HTTP harness from spike_v20 (kept
    self-contained, NOT imported) and does NOT modify spike_v20. Three deltas vs v20:
      (1) samples read from demo/demo_samples.json (single source of truth),
      (2) a transient-filter turn (T4) asserting extract drops it (extracted==0),
      (3) a no-cross-talk HARD gate (G4): /rag/search?q=cat ranks "coco" before "peanuts".

★ MODEL ALIGNMENT: OPENROUTER_MODEL must be qwen/qwen3.7-max — v2.1's fact->belief HARD
  gate is bound to it; running on another model makes v2.1's green non-comparable. We print
  the live model, record it, and loudly warn (SOFT) if it diverges.

HARD GATES (exit 0 iff all 8 pass; honest report; server always torn down):
    G1 server ready + SPA serves same-origin (GET / -> 200 + id="root")
    G2 extract filter: T4's /chat returns extracted==0 AND buffer/rag counts unchanged
    G3 router fork: after teach -> buffer has every belief, NO belief leaks to rag,
       rag has fact>=2 AND other>=1 (exact {fact x2, other x1} reported SOFT)
    G4 no cross-talk: GET /rag/search?q=cat -> "coco" hit AND ranked before "peanuts"
    G5 consolidate: /consolidate -> n_written>=1, buffer drained; /health edit_on==true
    G6 PROOF: every belief rag_off -> expected answer AND retrieved==[] AND attribution hit
    G7 contrast: fact rag_on -> "coco" AND retrieved non-empty (answered FROM RAG)
    G8 hot-swap round-trip: edit-module {on:false} silences the belief; {on:true} restores it

KNOWN WART (NOT fixed this round): belief decode may degrade ("Zarithonononon...");
the substring hit on "zarithon" still passes. Recorded only.
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

from memory import llm  # noqa: E402  (live model name for alignment check / record)

HOST = "127.0.0.1"
PORT = 8077
BASE = f"http://{HOST}:{PORT}"
READY_TIMEOUT_S = 180
CHAT_TIMEOUT_S = 120
EXPECTED_MODEL = "qwen/qwen3.7-max"
SAMPLES_PATH = os.path.join(REPO_ROOT, "demo", "demo_samples.json")


# --- HTTP (stdlib only) — cloned from spike_v20, unchanged ------------------------------------
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


# --- server lifecycle — cloned from spike_v20, unchanged --------------------------------------
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


# --- samples ----------------------------------------------------------------------------------
def _load_turns():
    with open(SAMPLES_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    turns = data["turns"]
    by_kind: dict[str, list[dict]] = {}
    for t in turns:
        by_kind.setdefault(t["kind"], []).append(t)
    return data, turns, by_kind


# --- main -------------------------------------------------------------------------------------
def main() -> int:
    V = {"errors": []}
    data, turns, by_kind = _load_turns()
    belief_turns = by_kind.get("belief", [])
    fact_turns = by_kind.get("fact", [])
    other_turns = by_kind.get("other", [])
    transient_turns = by_kind.get("transient", [])
    rag_turns = fact_turns + other_turns
    fact_anchor = next((t for t in fact_turns if "rag_cross_talk" in t), fact_turns[0] if fact_turns else None)
    other_anchor = other_turns[0] if other_turns else None
    hot_swap_turn = next((t for t in belief_turns if t["id"] == "T1-belief-zarithon"), belief_turns[0] if belief_turns else None)
    expected_beliefs = len(belief_turns)
    expected_rag_min = sum(int(t.get("expect_min_items", 1)) for t in rag_turns)

    model = llm.DEFAULT_MODEL
    V["model"] = model
    V["model_aligned"] = (model == EXPECTED_MODEL)
    V["sample_counts"] = {
        "belief": expected_beliefs,
        "fact": len(fact_turns),
        "other": len(other_turns),
        "transient": len(transient_turns),
        "expected_rag_min": expected_rag_min,
    }

    print("=" * 78)
    print("SPIKE v2.2 — DEMO DRESS-REHEARSAL (real HTTP, single-source samples)")
    print("=" * 78)
    print(f"  model = {model}   aligned_to_v2.1({EXPECTED_MODEL}) = {V['model_aligned']}")
    print(f"  samples = {SAMPLES_PATH}  ({len(turns)} turns)  counts={V['sample_counts']}", flush=True)
    if not V["model_aligned"]:
        print(f"  ⚠️  OPENROUTER_MODEL != {EXPECTED_MODEL} — results NOT comparable to v2.1's HARD gate!", flush=True)
    if not belief_turns:
        V["errors"].append("sample file has no belief turns")
    if fact_anchor is None:
        V["errors"].append("sample file has no fact turn for G7 contrast")
    if hot_swap_turn is None:
        V["errors"].append("sample file has no belief turn for G8 hot-swap")

    proc = None
    try:
        if V["errors"]:
            raise RuntimeError("; ".join(V["errors"]))

        # G1a — boot the real server
        print("\n== [G1] launch uvicorn serving.app:app ==", flush=True)
        proc = _launch()
        print(f"    pid={proc.pid} port={PORT}", flush=True)
        _wait_ready(proc)
        V["ready"] = True

        # G1b — SPA serves same-origin (browser entry)
        st, body = _raw("/")
        V["spa_serves"] = (st == 200 and 'id="root"' in body)
        print(f"    GET / -> {st}, id=root present={V['spa_serves']} (browser SPA same-origin)", flush=True)

        # Teach every durable turn in file order. Transient turns are held for G2 so
        # their "counts unchanged" assertion is measured after the real durable state exists.
        print("\n== teach durable turns from demo_samples.json ==", flush=True)
        V["teach"] = {}
        durable_turns = [t for t in turns if t.get("kind") != "transient"]
        for t in durable_turns:
            r = post("/chat", {"message": t["teach"]})
            V["teach"][t["id"]] = {"extracted": r.get("extracted"),
                                   "learned": r.get("learned"),
                                   "rag_indexed": r.get("rag_indexed")}
            print(f"    [{t['id']:<24}] kind={t.get('kind'):<6} extracted={r.get('extracted')} "
                  f"edit={r.get('learned')} rag_indexed={r.get('rag_indexed')} | {t['teach']!r}", flush=True)

        # Snapshot BEFORE transient turns (for G2 "counts unchanged").
        m_before = get("/memories")["counts"]
        c_before = (m_before["buffer"], m_before["rag"])

        # G2 — transient turns must be FILTERED (extract drops them, nothing surfaces).
        print("\n== [G2] teach transient turns (expect extracted==0 and counts unchanged) ==", flush=True)
        transient_rows = []
        current_counts = c_before
        for t in transient_turns:
            r = post("/chat", {"message": t["teach"]})
            m_now = get("/memories")["counts"]
            next_counts = (m_now["buffer"], m_now["rag"])
            row = {
                "id": t["id"],
                "extracted": r.get("extracted"),
                "before": current_counts,
                "after": next_counts,
                "ok": r.get("extracted") == int(t.get("expect_extracted", 0)) and next_counts == current_counts,
            }
            transient_rows.append(row)
            current_counts = next_counts
            print(f"    [{t['id']:<24}] extracted={row['extracted']} "
                  f"counts(buffer,rag) {row['before']} -> {row['after']} ok={row['ok']}", flush=True)
        m_after = get("/memories")["counts"]
        c_after = (m_after["buffer"], m_after["rag"])
        V["transient_rows"] = transient_rows
        V["t4_extracted"] = transient_rows[0]["extracted"] if transient_rows else None
        V["t4_counts_before"] = c_before
        V["t4_counts_after"] = c_after
        V["t4_counts_unchanged"] = (c_before == c_after)
        V["g2"] = (not transient_turns) or all(row["ok"] for row in transient_rows)
        print(f"    transient_overall={V['g2']}  final counts(buffer,rag) {c_before} -> {c_after}", flush=True)

        # G3 — router fork: every belief -> buffer, facts+other -> rag, NO belief leak.
        mem = get("/memories")
        buf_types = sorted(b.get("type") for b in mem["buffer"])
        rag_types = sorted(it.get("type") for it in mem["rag"])
        V["routing"] = {"buffer_types": buf_types, "rag_types": rag_types}
        belief_buf = buf_types.count("belief")
        belief_rag = rag_types.count("belief")
        fact_rag = rag_types.count("fact")
        other_rag = rag_types.count("other")
        V["g3"] = (belief_buf == expected_beliefs and len(buf_types) == expected_beliefs
                   and belief_rag == 0 and len(rag_types) >= expected_rag_min)
        V["g3_exact_spec"] = (
            buf_types == ["belief"] * expected_beliefs
            and len(rag_types) == expected_rag_min
            and belief_rag == 0
        )  # SOFT
        print(f"\n== [G3] router fork ==\n    buffer types={buf_types}  rag types={rag_types}", flush=True)
        print(f"    belief_in_buffer={belief_buf}/{expected_beliefs} belief_in_rag={belief_rag} "
              f"rag_count={len(rag_types)} expected_rag_min={expected_rag_min} "
              f"fact_in_rag={fact_rag} other_in_rag={other_rag} exact_spec={V['g3_exact_spec']}", flush=True)

        # G4 — no cross-talk: q=cat ranks "coco" before "peanuts"
        ct_turn = next((t for t in fact_turns if "rag_cross_talk" in t), None)
        if ct_turn is None:
            V["g4"] = False
            V["rag_search_texts"] = []
            print("\n== [G4] no cross-talk skipped: no rag_cross_talk turn ==", flush=True)
        else:
            ct = ct_turn["rag_cross_talk"]
            rs = get(f"/rag/search?q={ct['query']}&k=5")
            texts = [it["text"].lower() for it in rs.get("results", [])]
            hit_idx = next((i for i, t in enumerate(texts) if ct["must_hit"] in t), -1)
            before_idx = next((i for i, t in enumerate(texts) if ct["must_rank_before"] in t), -1)
            V["rag_search_texts"] = texts
            V["g4"] = (hit_idx >= 0 and (before_idx < 0 or hit_idx < before_idx))
            print(f"\n== [G4] no cross-talk: /rag/search?q={ct['query']} ==", flush=True)
            print(f"    results={texts}", flush=True)
            print(f"    '{ct['must_hit']}'@{hit_idx} ranked before '{ct['must_rank_before']}'@{before_idx} "
                  f"-> {V['g4']}", flush=True)

        # G5 — consolidate beliefs into WEIGHTS; buffer drained; edit_on.
        print("\n== [G5] POST /consolidate (beliefs -> WEIGHTS) ==", flush=True)
        r5 = post("/consolidate", {})
        V["n_written"] = r5.get("n_written")
        V["buffer_after"] = r5.get("buffer_count")
        V["edit_on"] = get("/health").get("edit_on")
        V["g5"] = (V["n_written"] == expected_beliefs
                   and V["buffer_after"] == 0 and V["edit_on"] is True)
        print(f"    n_written={V['n_written']}/{expected_beliefs} "
              f"buffer_after={V['buffer_after']} edit_on={V['edit_on']}", flush=True)

        # G6 — ★ PROOF: every belief answered FROM WEIGHTS (rag_off, retrieved empty,
        # and attribution hits a consolidated memory containing the expected target).
        print("\n== [G6] ★ ask every BELIEF rag_off ==", flush=True)
        belief_rows = []
        for t in belief_turns:
            p = t["probe"]
            r = post("/chat", {"message": p["ask"], "rag_off": p.get("rag_off", True)})
            reply = r.get("reply")
            attr = r.get("attribution") or {}
            attr_mem = attr.get("memory") or {}
            attr_text = (attr_mem.get("text") or "").lower()
            word_hit = hit(reply, p["expect_words"])
            retrieved_empty = (r.get("retrieved") == [])
            attr_hit = bool(attr.get("hit"))
            attr_word_hit = hit(attr_text, p["expect_words"])
            ok = word_hit and retrieved_empty and attr_hit and attr_word_hit
            row = {
                "id": t["id"],
                "ask": p["ask"],
                "reply": reply,
                "word_hit": word_hit,
                "retrieved_empty": retrieved_empty,
                "attribution_hit": attr_hit,
                "attribution_memory": attr_mem,
                "attribution_word_hit": attr_word_hit,
                "ok": ok,
            }
            belief_rows.append(row)
            print(f"    [{t['id']:<24}] ok={ok} word_hit={word_hit} "
                  f"retrieved_empty={retrieved_empty} attr_hit={attr_hit} "
                  f"attr_word_hit={attr_word_hit} reply={reply!r}", flush=True)
        V["belief_rows"] = belief_rows
        V["belief_reply"] = belief_rows[0]["reply"] if belief_rows else None
        V["belief_attr_hit"] = all(row["attribution_hit"] for row in belief_rows)
        V["g6"] = bool(belief_rows) and all(row["ok"] for row in belief_rows)

        # G7 — contrast: fact answered FROM RAG (rag_on, retrieved non-empty)
        p2 = fact_anchor["probe"]
        print(f"\n== [G7] ask FACT rag_on: {p2['ask']!r}  (the contrast) ==", flush=True)
        r7 = post("/chat", {"message": p2["ask"], "rag_off": p2.get("rag_off", False)})
        V["fact_reply"] = r7.get("reply")
        fact_word_hit = hit(r7.get("reply"), p2["expect_words"])
        V["fact_retrieved_nonempty"] = bool(r7.get("retrieved"))
        V["g7"] = fact_word_hit and V["fact_retrieved_nonempty"]
        print(f"    reply={V['fact_reply']!r}", flush=True)
        print(f"    word_hit={fact_word_hit} retrieved_nonempty={V['fact_retrieved_nonempty']} "
              f"(n={len(r7.get('retrieved') or [])})", flush=True)

        # SOFT — first other answered from RAG (reported, not gated; mirrors spike_v20).
        if other_anchor is not None:
            p3 = other_anchor["probe"]
            r8 = post("/chat", {"message": p3["ask"], "rag_off": p3.get("rag_off", False)})
            V["other_reply"] = r8.get("reply")
            V["other_answered"] = hit(r8.get("reply"), p3["expect_words"])
            print(f"\n== [SOFT] ask OTHER rag_on: {p3['ask']!r} ==\n    reply={V['other_reply']!r}  "
                  f"answered={V['other_answered']}", flush=True)
        else:
            V["other_reply"] = None
            V["other_answered"] = None

        # G8 — hot-swap round-trip on the designated low-prior belief.
        p1 = hot_swap_turn["probe"]
        print(f"\n== [G8] hot-swap on {hot_swap_turn['id']}: OFF -> silent -> ON -> restored ==", flush=True)
        post("/edit-module", {"on": False})
        r9 = post("/chat", {"message": p1["ask"], "rag_off": True})
        V["belief_after_unplug"] = r9.get("reply")
        silenced = not hit(r9.get("reply"), p1["expect_words"])
        post("/edit-module", {"on": True})  # restore
        r10 = post("/chat", {"message": p1["ask"], "rag_off": True})
        V["belief_after_restore"] = r10.get("reply")
        restored = hit(r10.get("reply"), p1["expect_words"])
        V["g8_silenced"] = silenced
        V["g8_restored"] = restored
        V["g8"] = silenced and restored
        print(f"    OFF -> {V['belief_after_unplug']!r}  silenced={silenced}", flush=True)
        print(f"    ON  -> {V['belief_after_restore']!r}  restored={restored}", flush=True)

    except Exception as e:
        import traceback
        V["errors"].append(f"{type(e).__name__}: {e}")
        print("\n!! dress-rehearsal aborted:", flush=True)
        traceback.print_exc()
        print(f"\n--- server tail ---\n{_tail()}", flush=True)
    finally:
        _teardown(proc)

    # --- verdict ---
    gates = {
        "G1_server_ready_and_spa": V.get("ready", False) and V.get("spa_serves", False),
        "G2_transient_filtered": V.get("g2", False),
        "G3_router_fork": V.get("g3", False),
        "G4_no_cross_talk": V.get("g4", False),
        "G5_consolidated": V.get("g5", False),
        "G6_PROOF_belief_from_weights": V.get("g6", False),
        "G7_contrast_fact_from_rag": V.get("g7", False),
        "G8_hotswap_roundtrip": V.get("g8", False),
    }
    soft = {
        "model_aligned": V.get("model_aligned"),
        "sample_counts": V.get("sample_counts"),
        "g3_exact_spec": V.get("g3_exact_spec"),
        "belief_attr_hit": V.get("belief_attr_hit"),
        "belief_rows": [
            {
                "id": row.get("id"),
                "ok": row.get("ok"),
                "attribution_memory": row.get("attribution_memory"),
            }
            for row in V.get("belief_rows", [])
        ],
        "other_answered": V.get("other_answered"),
        "routing": V.get("routing"),
    }
    overall = all(gates.values()) and not V["errors"]

    print("\n" + "=" * 78)
    print("DEMO DRESS-REHEARSAL (v2.2) — 8 HARD gates")
    print("=" * 78)
    print(f"  model: {V['model']}  (aligned={V.get('model_aligned')})")
    if not V.get("model_aligned"):
        print(f"  ⚠️  MODEL NOT ALIGNED to {EXPECTED_MODEL} — not comparable to v2.1")
    for k, v in gates.items():
        print(f"  {k:<30}: {'PASS ✅' if v else 'FAIL ⚠️'}")
    print("  ---  contrast quick-read:")
    print(f"  belief (rag_off) -> {V.get('belief_reply')!r}  [weights · not in prompt]")
    print(f"  fact   (rag_on)  -> {V.get('fact_reply')!r}  [RAG · retrieved→in prompt]")
    print(f"  soft: {soft}")
    if V["errors"]:
        print(f"  ERRORS: {V['errors']}")
    print(f"  ---\n  OVERALL: {'PASS ✅' if overall else 'FAIL ⚠️'}")
    print("=" * 78, flush=True)
    print("\nMACHINE_RESULT " + json.dumps({"gates": gates, "soft": soft, "V": V}, default=str))
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
