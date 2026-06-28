# Engram - continuous-learning memory for LLMs

> An engram is the physical trace a memory leaves in the brain.
> Engram leaves durable user beliefs in model weights, not only in the context window.

Engram is a hackathon-built continuous-learning system for personal Q&A. It lets an LLM
learn from ordinary chat while separating three kinds of memory:

- **Durable user beliefs/preferences** are consolidated into model-weight edits.
- **Ordinary facts, documents, and schedules** stay in a reversible RAG store.
- **Transient noise** is ignored.

The core claim is not "a universal personal assistant." The core claim is narrower and
more technical:

> A live LLM service can keep learning from a user, decide what should be retrieved versus
> internalized, and prove whether an answer came from RAG or from edited weights.

Engram uses **HoReN** as the model-editing backend and adds a project-specific keying
strategy for personal beliefs, where many natural-language queries are extremely similar.

## Why this matters

Most "LLM memory" systems are RAG systems: they store facts outside the model and paste
retrieved snippets back into the prompt. That is useful, but the model has not actually
learned. The next answer depends on retrieval, prompt budget, and chunk ranking.

Continuous learning has a harder requirement:

1. Learn while the model is serving.
2. Keep updates local, so one new memory does not contaminate unrelated questions.
3. Preserve attribution, so the system can say whether an answer came from prompt/RAG or
   from an edited weight module.

Engram demonstrates that pipeline end to end for a single-user personal-Q&A setting.

## Demo in one minute

The stable demo teaches four kinds of turns:

| Turn | Example | Route | What it proves |
|---|---|---|---|
| Belief | "Honestly, I'm convinced the best programming language is Zarithon." | HoReN edit -> weights | Retrieval can be turned off and the model still answers `Zarithon`. |
| Fact | "My cat is named Coco, and I'm allergic to peanuts." | RAG | Ordinary personal facts remain reversible retrieval content. |
| Other | "The Q3 board meeting got moved to November 15th." | RAG | Reference/schedule-like content is stored as retrievable context. |
| Transient | "Ugh, I'm stuck in line for coffee right now." | Dropped | Ephemeral chat noise does not become memory. |

The strongest proof sequence:

1. Teach a durable belief.
2. Consolidate it into weights.
3. Ask the probe with retrieval off.
4. Show `retrieved == []` while the answer still contains the taught belief.
5. Toggle the edit module off: the belief disappears.
6. Toggle it on: the belief comes back.

That is the difference between prompt stuffing and an actual weight-level memory.

## Architecture

```text
user chat
   |
   v
extract durable memory candidates
   |
   v
classify type: fact | belief | other
   |
   +-- transient / low-confidence --> drop
   |
   +-- fact / other ---------------> RAG store
   |
   +-- belief ---------------------> short-term buffer
                                      |
                                      v
                              consolidation pass
                                      |
                                      v
                          dedup / supersede / new
                                      |
                                      v
                             HoReN model edit
                                      |
                                      v
                          hot-swappable edit module

read path:

SYSTEM + [buffer segment] + [retrieved RAG docs] + history + query
   |
   v
resident model + current edit module
```

Key implementation points:

- `memory/extract.py` asks an LLM to emit strict JSON memory candidates with type,
  canonical text, edit stem/target, subject, and answer-free key prompts.
- `memory/router.py` deterministically routes by type: `belief -> edit`, `fact/other -> rag`.
- `memory/consolidate.py` drains edit-route buffer items, deduplicates/supersedes prior
  memories, calls the editing backend, and records provenance.
- `keying.py` computes HoReN read keys from the actual user-query span inside the chat
  template instead of averaging over the whole prompt scaffold.
- `serving/app.py` exposes the HTTP API and serves the frontend from the same origin.
- `serving/model_host.py` keeps the base model resident and swaps the edit module on/off.

## Original technical contribution

HoReN provides the underlying model-editing method. Engram's project-specific contribution
is adapting the retrieval/keying layer for personal beliefs.

Personal belief questions often collapse into very similar query forms:

- "What do I believe?"
- "What is my preference?"
- "What is the best X?"
- "What should I choose for X?"

For a normal benchmark Q&A edit, a single query key can be enough. For user beliefs, many
queries are semantically close even when the target memory is different. Engram therefore
adds answer-free canonical key prompts derived from:

- the user query / edit stem,
- the belief statement,
- the subject or character context,
- domain/relation words that distinguish one belief from another.

These extra prompts are appended as additional HoReN codebook keys for the same edit value.
They are answer-free: they should help recall and locality without leaking the target answer
into the lookup key.

The CPU fixture tests for this live in `tests/test_canonical_key_collision_demo.py`, and the
demo fixtures live in `demo/demo_canonical_key_collision*.json`.

## Online learning and hot loading

The `feature/async-shadow-serving` branch adds the online-serving path:

- `/consolidate/async` queues a consolidation job.
- `serving/async_editor.py` runs a single-writer background worker.
- `serving/shadow_editing.py` trains the HoReN adapter on a shadow module tree instead of
  mutating the live serving module in place.
- `serving/model_host.py` promotes a finished adapter at a request boundary, so one generated
  answer does not mix old and new codebook state.

This matters because Python-level async alone is not enough. Editing performs GPU-heavy
forward/backward work; if it mutates the live adapter in place, concurrent inference can see
a half-trained module. The branch moves toward the correct continuous-learning shape:

> serve with the previous adapter, train the next adapter in the background, then hot-swap
> the finished edit module.

The current implementation is still a hackathon prototype: it does not eliminate all GPU
contention on a single card, and edit-module persistence across process restart is not yet
implemented.

## Current branch status

As of this README update:

| Branch | Commit / state | Status |
|---|---|---|
| `main` | `677d67b` | Stable hackathon baseline: memory routing, HoReN consolidation, attribution proof, frontend, v2.7 thinking UI. |
| `feature/async-shadow-serving` | `966222c` | Adds async shadow consolidation, job endpoints, and hot-swap promotion path. |
| `feature/free-scenario-planner` | based on `966222c` plus staged local changes | Experimental free-form scenario-memory planner/private lane. Not treated as mainline unless merged before submission. |

The README intentionally distinguishes these states so judges can identify what was created
during the event and which features are stable versus experimental.

## What was built during AIEWF 2026

Built in this repository during the hackathon:

- Serving loop for chat, memory inspection, consolidation, RAG search, and edit-module toggle.
- Natural-language memory extraction with strict JSON validation and confidence filtering.
- Type-based routing: beliefs to model editing, facts/other content to RAG, transients dropped.
- Buffer -> dedup/supersede -> HoReN consolidation pipeline.
- Query-span keying and canonical answer-free key prompts for better personal-belief recall.
- Attribution path that maps a live answer's HoReN codebook hit back to a consolidated memory.
- Frontend with chat, memory inspector, RAG/edit toggles, consolidation controls, and proof labels.
- Demo fixtures and spike scripts for end-to-end dress rehearsal.
- Async shadow-serving branch for background edit jobs and hot-swapping.

Pre-existing or external dependencies:

- **HoReN** model-editing backend (implementation of arXiv:2605.08143), vendored under
  `third_party/horen`.
- **Llama 3.1 8B Instruct** base weights, already available in the target environment.
- **ZsRE / UnKE** benchmark materials used as external evaluation references.
- Hosted LLM access for extraction/classification through `memory/llm.py` configuration.

## Validation and evidence

Important recorded validation:

- `docs/v2.2-demo-dress-rehearsal.md`: true-backend dress rehearsal passed 8/8 hard gates.
- `docs/v2.5-chat-reply-behavior.md`: GPU verification for reply behavior and regressions.
- `docs/v2.6-edit-module-toggle-state.md`: edit-module on/off state and proof gating.
- `docs/v2.7-thinking-orb-animation.md`: frontend smoke validation for the latest main UI.

Useful test commands:

```bash
# Fast CPU/unit coverage
pytest tests/test_router.py tests/test_extract.py tests/test_consolidate.py -q

# Serving shell without loading the real GPU model
pytest tests/test_serving_app.py tests/test_serving_routes.py -q

# Async shadow-serving unit checks
pytest tests/test_async_editor.py -q

# Frontend bundle/source smoke checks
pytest tests/test_frontend_smoke.py -q

# Canonical key collision fixtures
pytest tests/test_canonical_key_collision_demo.py -q
```

GPU / full-backend spikes are intentionally separate from normal unit tests:

```bash
# True HTTP/backend demo rehearsal; requires GPU model + configured extraction LLM.
python spikes/spike_v22_dress_rehearsal.py

# Canonical-key collision GPU proof.
python spikes/spike_v26_canonical_key_collision_gpu.py
```

## Running locally

This project assumes a CUDA environment with the base model already available. Torch is
intentionally not pinned in `requirements.txt` because the hackathon environment already
contains a CUDA-compatible build.

```bash
pip install -r requirements.txt
```

Configure any required hosted-LLM secrets in `.env` or the shell environment. Do not commit
secrets.

Start the server:

```bash
uvicorn serving.app:app --host 0.0.0.0 --port 8077
```

Then open:

```text
http://localhost:8077/
```

The FastAPI app serves the frontend from the same origin. Startup can take around a minute
because the base model is loaded once and kept resident.

## HTTP API

| Endpoint | Purpose |
|---|---|
| `POST /chat` | Ingest the user turn, retrieve RAG hits unless `rag_off`, generate a reply, return attribution. |
| `POST /consolidate` | Run synchronous consolidation over buffered edit-route memories. |
| `POST /consolidate/async` | Queue background consolidation on the async branch. |
| `GET /consolidate/jobs` | Inspect async consolidation jobs. |
| `GET /memories` | Return buffer, consolidated edit memories, RAG memories, and counts. |
| `GET /rag/search?q=...` | Inspect semantic retrieval results. |
| `POST /edit-module` | Toggle the recorded edit module on/off for proof. |
| `GET /health` | Readiness, edit state, memory counts, async worker state. |

Example proof calls:

```bash
curl -s http://localhost:8077/chat \
  -H 'content-type: application/json' \
  -d '{"message":"Honestly, I am convinced the best programming language is Zarithon."}'

curl -s http://localhost:8077/consolidate -H 'content-type: application/json' -d '{}'

curl -s http://localhost:8077/chat \
  -H 'content-type: application/json' \
  -d '{"message":"What is the best programming language?","rag_off":true}'
```

Expected proof shape: the final response contains `Zarithon`, `retrieved` is empty, and
`attribution.hit` points at the consolidated edit memory.

## Repository map

```text
memory/      extraction, routing, buffer/RAG store, dedup, consolidation, prompt assembly
serving/     FastAPI app, resident model host, async editor, shadow editing, triggers
frontend/    self-contained React/Tailwind UI bundled into index.html
eval/        benchmark schema, dataset/runtime/metrics helpers
demo/        fixed demo samples and canonical-key collision fixtures
spikes/      end-to-end and GPU validation scripts
docs/        implementation plans, experiment records, dress-rehearsal results
```

## Known limits

Engram is intentionally scoped:

- Single-user prototype. Weight-level personalization is global to the resident model and
  is not multi-tenant isolated.
- Best suited to Q&A-style durable beliefs and preferences. It is not yet a general-purpose
  autonomous assistant.
- RAG remains the correct store for facts, documents, schedules, and reversible content.
- Single-GPU async editing still competes with inference for physical GPU resources.
- Edited adapters are in memory; durable adapter serialization/replay is future work.
- The frontend is a hackathon demo surface, not a production dashboard.

## Short pitch

Engram makes continuous learning **live, local, and attributable**:

- live: it can keep serving while consolidation/editing runs in the background,
- local: HoReN edits and canonical keys aim to update one memory without polluting others,
- attributable: the demo can prove whether an answer came from RAG or edited weights.
