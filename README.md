# Engram — continual-learning personal agent

> An engram is the physical trace a memory leaves in the brain.
> Engram leaves yours in its weights — not its context window.

Engram is an agent that learns user facts/preferences from natural chat, consolidates the
durable ones into the model's RAW WEIGHTS via model editing (not just RAG), and proves it by
answering correctly in a fresh session with retrieval OFF.

## Architecture (overview)
  - Two-speed memory: RAG buffer (short-term, instant write-then-read) + edited weights (long-term)
  - Routing at extraction: atomic ∧ internalize ∧ stable → edit ; else → RAG
  - Async consolidation: buffer → dedup(same/changed/new) → HoReN edit → weights → drop from buffer
  - Inference prompt = SYSTEM + [RAG window: buffer-seg + retrieved-docs-seg] + history + query,
    always run on the edited model; window structure always present, content may be empty
  - Zero-downtime hot-swap of the edit module

## Built at AIEWF 2026
  serving loop · hot-swap · auto-extraction + routing · consolidation + dedup ·
  eval harness + preference/belief probes · frontend

## Dependencies (pre-existing, NOT built this weekend)
  - HoReN (impl of arXiv 2605.08143) — editing backend, vendored in third_party/horen
  - base model: llama-3.1-8B-Instruct (already downloaded in this env from HF)
  - ZsRE / UnKE benchmark (source + license)
