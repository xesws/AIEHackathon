Plan — Scaffold the Engram hackathon project (3 commits, stubs only)

 Context

 We're initializing the Engram AIEWF-2026 hackathon project in the existing repo at
 /workspace/AIEHackathon (remote origin → github.com/xesws/AIEHackathon.git, currently
 zero commits). The deliverable is a scaffold only — every code file is a stub
 (docstring + signatures + TODO/NotImplementedError), no logic. The git history must prove
 work happened inside the hackathon window, so commits are made now and pushed in a precise
 order. HoReN is a pre-existing external dependency (the user's own reproduction repo),
 vendored as-is in third_party/horen — never reimplemented inside our modules.

 Hard compliance rules (from the request)

 1. Commit #1 contains ONLY README.md and is pushed before any other file is created.
 2. Scaffold only — no implementations; bodies are raise NotImplementedError / ... / docstring.
 3. Never write HoReN ourselves; it's vendored in step 3.
 4. Three separate, meaningfully-messaged commits.

 Findings from read-only inspection

 - Repo: branch main, no commits yet; remote origin = github.com/xesws/AIEHackathon.git.
 - Working tree has two untracked, unrelated files: an empty README (0 bytes, stray) and
 prompt.md (9.3 KB, the saved copy of these instructions). Both must be kept out of all
 commits → handled via .gitignore so the user's git add -A (commit 2) won't sweep them in.
 - Git identity was unset → will set local user.name=xesws, user.email=qty20010619@gmail.com.
 - No push auth configured → user sets it up themselves before the first push (their choice).

 Decisions baked in

 - torch stays UNPINNED in requirements.txt (env already has torch 2.8.0+cu128; the
 no-downgrade guardrail applies to this shared machine). We install nothing and run nothing.
 - .gitignore data-dump/cache patterns are anchored to repo root (/data/, /hugging_cache/)
 so they don't strip vendored third_party/horen content; weight patterns
 (*.safetensors/*.pt/*.bin/*.gguf) stay global (we never commit weights anywhere).
 - Stubs keep imports light (mostly typing/dataclasses); declared third-party deps
 (fastapi, etc.) may appear in serving/ stubs — nothing is executed this weekend.
 - The README.md is reproduced exactly per the spec, fixing only the obvious typo
 dowloaded → downloaded.

 ---
 Execution steps

 Step 0 — Pre-flight (no commits yet)

 git config user.name  "xesws"                 # local repo config (.git/config, persists on /workspace)
 git config user.email "qty20010619@gmail.com"
 Then pause and ask the user to run their push-auth !  command (e.g. credential-helper +
 PAT, or install/login gh). Proceed only once they confirm. (No way to verify write auth
 short of the real push, so the commit-1 push below is the smoke-test.)

 Step 1 — Commit #1: README.md ONLY → push first  ⟵ hard gate

 - Write README.md with this exact structure (typo fixed):
   - # Engram — continual-learning personal agent
   - blockquote: "An engram is the physical trace a memory leaves in the brain. / Engram leaves yours in its weights
 — not its context window."
   - intro paragraph (learns facts from chat → consolidates into RAW WEIGHTS via model editing → proves it with
 retrieval OFF in a fresh session)
   - ## Architecture (overview) — 5 bullets (two-speed memory; routing at extraction; async consolidation; inference
 prompt skeleton; zero-downtime hot-swap)
   - ## Built at AIEWF 2026 — serving loop · hot-swap · auto-extraction + routing · consolidation + dedup · eval
 harness + probes · frontend
   - ## Dependencies (pre-existing, NOT built this weekend) — HoReN (arXiv 2605.08143) vendored in
 third_party/horen; base model llama-3.1-8B-Instruct (already downloaded from HF); ZsRE / UnKE benchmark
 git add README.md          # explicit path only — NOT -A; stray README/prompt.md untouched
 git commit -m "docs: project README (architecture overview)"
 git push -u origin HEAD
 - If the push fails (auth): STOP and report. Do NOT create any scaffold file.
 - Verify: git show --stat HEAD shows only README.md.

 Step 2 — Commit #2: scaffold tree (stubs only) → push

 Create exactly this tree (every .py = docstring + signatures + NotImplementedError/...):
 requirements.txt          # torch(unpinned)+comment, transformers, fastapi, uvicorn, pymongo,
                           #   sentence-transformers; commented "# -e ./third_party/horen"
 .gitignore                # see "gitignore contents" below
 editing.py                # edit(model, memory) -> edited_model | edit_module  (wrapper over third_party.horen;
                           #   import path = TODO until step 3; output format TBD drives hot-swap)
 generate.py               # generate(query, *, model, buffer, rag_hits, with_rag=True) -> str
                           #   (builds prompt via memory.prompt.build_prompt; runs on edited model)
 memory/__init__.py
 memory/schema.py          # @dataclass MemoryItem: id, type{fact|preference|belief|jargon}, text,
                           #   route{edit|rag}, status{buffer|consolidated|retired}, source, ts, provenance
 memory/extract.py         # extract(chat) -> list[MemoryItem]  (LLM pulls candidates; calls router.route)
 memory/router.py          # route(item) -> "edit"|"rag"  (atomic ∧ internalize ∧ stable → edit; axis = SHAPE)
 memory/buffer.py          # append(item) [NO dedup] / load_unconsolidated() / drop(ids)  (small by design)
 memory/rag_store.py       # add(item) / search(query, k)  (permanent long-content; never consolidated)
 memory/dedup.py           # classify(candidate, consolidated) -> "duplicate"|"supersede"|"new"  (runs at
 consolidation)
 memory/consolidate.py     # run_pass(trigger) -> n_written  (dedup→skip/retire+write/write; editing.edit; drop
 buffer)
 memory/prompt.py          # build_prompt(query, buffer, rag_hits) -> messages  (SYSTEM; RAG window always rendered:
                           #   (a) buffer seg whole-inject, (b) docs seg top-k; then history; then query)
 serving/__init__.py
 serving/app.py            # FastAPI: POST /chat, POST /consolidate, GET /memories (orchestrates
 extract→buffer/consolidate/generate)
 serving/model_host.py     # load_base() / swap_edit_module(m) / current_model()  (hot-swap; branch on edit format
 TBD)
 serving/store.py          # Mongo: memories + provenance collections; CRUD + optional watch() (change-stream)
 serving/triggers.py       # manual / timer(N min) / buffer>=K / change-stream(debounced) triggers
 eval/__init__.py
 eval/dataset.py           # load_probes() -> items  (benchmark facts + NEW preference/belief/jargon;
 probes{efficacy,paraphrase,application,locality})
 eval/conditions.py        # base / rag / edit / edit+rag
 eval/metrics.py           # efficacy, paraphrase, locality, fluency(ppl), ctx_overhead_tokens, no_retrieval_recall
                           #   (import HoReN metrics where clean; RAG-condition + LLM-judge are ours)
 eval/run_matrix.py        # run_matrix(...)  (consumes editing.edit + generate.generate; capability matrix, not
 accuracy-winner)
 frontend/index.html       # placeholder demo/counter UI stub (HTML + TODO)
 third_party/              # left empty here; horen vendored in step 3 (never edit / never write ourselves)
 .gitignore contents (root-anchored data/cache; global weight ignores; excludes the two stray files):
 # Python
 __pycache__/
 *.py[cod]
 *.egg-info/
 .venv/
 venv/
 .env

 # Model weights (never commit, anywhere in the tree)
 *.safetensors
 *.pt
 *.bin
 *.gguf

 # Caches & data dumps (repo-root only — keep vendored third_party content intact)
 /hugging_cache/
 /.cache/
 /data/
 /dumps/

 # Local-only working files (not part of the scaffold)
 prompt.md
 /README
 - Syntax-check stubs (side-effect-free except gitignored __pycache__):
 python -m compileall -q editing.py generate.py memory serving eval
 git add -A     # .gitignore now excludes prompt.md + stray README, so only scaffold is staged
 git commit -m "scaffold: memory loop / serving / eval architecture (stubs only)"
 git push
 - Verify git status is clean and git show --stat HEAD lists only scaffold files.

 Step 3 — Commit #3: vendor HoReN → push

 git clone https://github.com/xesws/HoReN-paper-reproduction.git third_party/horen
 rm -rf third_party/horen/.git          # CRITICAL: kill nested .git (else broken pseudo-submodule)
 git add third_party/horen
 git commit -m "vendor HoReN (impl of arXiv 2605.08143) as editing backend — dependency, not hackathon work"
 git push
 - After vendoring, note (do not yet change) the real package layout so editing.py's import
 path can be adjusted later — this is a follow-up, not part of the scaffold commits.

 ---
 Verification

 - git log --oneline -3 shows the three commits in order:
 vendor HoReN → scaffold → docs README.
 - git show --stat <commit#1> proves commit #1 = only README.md.
 - GitHub (github.com/xesws/AIEHackathon) shows 3 pushed commits with in-window timestamps.
 - git status clean; prompt.md and stray README are untracked/ignored, never committed.
 - python -m compileall -q editing.py generate.py memory serving eval exits 0 (stubs parse).
 - Environment untouched: we installed nothing and ran no servers/models (torch 2.8.0+cu128 intact).
 - Finally: print the directory tree + the three commit hashes and messages.

 Stop conditions (report, don't push through)

 - Push auth not ready / commit-1 push fails → stop before any scaffold file exists.
 - git clone of HoReN fails (network/repo) → stop after commit #2; report.