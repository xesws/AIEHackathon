# Engram Eval Harness 规格 (v1) — 按真实 A/B/C 数据重写

> 目标：定位【哪个 module 是瓶颈】，靠【消融阶梯的相邻级落差】，不是看绝对分。
> 前提（来自侦察报告）：① 数据本体干净(硬指标全过)，直接用；② eval/ 下
>   dataset.py/metrics.py/run_matrix.py/conditions.py 是空壳，且按一套和 A/B/C
>   【对不上】的旧模型写的 → 全部【作废重写】，不复用其结构。
> 范围：eval/。不改 samples.json、不改 memory/serving/editing。
> 语言：正文中文；代码/字段/枚举英文。

══════════════════════════════════════════════════════════════
## 0. 不变量（实现不得违反）
──────────────────────────────────────────────────────────────
- INV-E1 数据只读。samples.json 一字不改。
- INV-E2 旧 harness stub 的数据模型（per-item efficacy/paraphrase/application/
  locality、从 third_party 加载）作废。loader 按【实际 A/B/C 三族结构】写。
- INV-E3 评判模型 ≠ 被测模型。被测 = Llama 3.1 8B（HoReN 编辑对象）；
  评判/borderline = Qwen via OpenRouter，temperature=0。天然隔离。
- INV-E4 瓶颈靠【相邻级落差】归因，每级只引入【一个】新 module（见 §3）。
- INV-E5 efficacy 的干净归因只在 zero_prior 成立（坑1）。aligned/medium/hard
  的"答对"不算 efficacy 证据，只用于 locality/baseline 参照。
- INV-E6 B 的 recall 用 match_any 匹配，不是 target_new（坑8：37/585 的
  match_any 不含 target_new 字面）。
- INV-E7 locality 只在改权重的 condition（含 edit 的）量；rag-only 不量 locality。

══════════════════════════════════════════════════════════════
## 1. 数据加载层 dataset.py（第一步先做，独立可测）
──────────────────────────────────────────────────────────────
职责：把 samples.json 读成三族的强类型对象，供 condition/metric 消费。空壳重写。

- load(path) -> {A: list[ASample], B: list[BSample], C: list[CSample]}
  按 sample_type 分流。校验计数 370/60/70，不符报错（防加载到坏文件）。
- ASample 暴露：id, type(X/Y), prior_hardness, category, subject, edit_prompt,
  target_new, rag_doc, queries[{q,a}], key
- BSample 暴露：id, facts[{key,type,category,prior_hardness,edit_prompt,target_new}],
  generation_prompt, gold_fact_set[{fact, key, match_any}]  ← 注意有 extra key(坑7)，
  容忍三键; rag_docs[]; 保证 len(facts)==len(gold_fact_set)==len(rag_docs)
- CSample 暴露：id, user_fact{...,edit_prompt,target_new,key}, list_domain,
  domain_filter{attribute,op,value?}, user_filter{...}, list_items[{name,attributes,blurb}],
  gold_answer, question, rag_doc, difficulty
- 便捷索引：pool_by_key（A 的 key→ASample），供 B/C 的 key 回查 A 池。
- ★ 提供 subset 选择器（关键，给阶梯用）：
    by_tier(samples, tiers)   如只取 zero_prior
    by_type(samples, "X"/"Y")
    zero_prior_Y()            = 干净 efficacy 子集（138 条），efficacy 主力
- 单测：计数对、三族字段齐、B 三列对齐、subset 选择器返回数与报告一致
  （zero_prior_Y=138 等）。mock 不需要，纯读本地 JSON。

══════════════════════════════════════════════════════════════
## 2. 评分层 metrics.py（第二步，纯函数，最易测）
──────────────────────────────────────────────────────────────
三族评分，全部 borderline 才调 Qwen，主路确定性。

- score_A(pred, target_new) -> bool
  substring exact-match：小写 + 去标点 + 去冠词(a/an/the) 后判 target_new 是否 ⊆ pred。
  borderline（近似但不字面含）→ Qwen 判，temperature=0。
- score_B(generation, gold_fact_set) -> {recall: float, hits: list[key]}
  对每条 gold：其 match_any 任一 term（同归一化）出现在 generation → 命中。
  recall = 命中数 / len(gold_fact_set)。★ 用 match_any，不碰 target_new(INV-E6)。
  borderline 逐条 Qwen 核。
- score_C(model_choice, gold_answer, list_items) -> bool
  从模型输出里抽它选了哪个 name（Qwen 抽取，因为模型会自由措辞），== gold_answer。
  ★ 被测模型只看 name+blurb，【不喂 attributes】（attributes 是评分 ground-truth，不是输入）。
- 附带：token 长度（tiktoken）落盘，备 token-轴对比（editing 的 context 省 token 是卖点之一）。
- 单测：score_A 大小写/标点/冠词/子串边界；score_B 的 match_any 命中（构造含/不含）、
  37 条 synonym 那类（target≠match_any）能正确命中；score_C 抽取+比对（mock Qwen 抽取返回固定）。

══════════════════════════════════════════════════════════════
## 3. 消融阶梯 conditions.py + run_matrix.py（核心；先读 memory/serving 真实签名）
──────────────────────────────────────────────────────────────
> 每级只加一个【新 module】，相邻级落差 = 该 module 损耗。这是整个 eval 的灵魂。
> 实现前先 READ：memory.consolidate.run_pass / serving.ingest / generate /
>   editing.edit / set_model_provider / store reset 的真实签名，按实际调，勿臆测。

六级（P3.5 是把 extractor 和 dedup 拆开的关键插级，别省）：

  P1  edit-only        知识用 editing.edit 直接写权重（喂已拆好的 stem/target，
                       不经 extractor、不放 RAG）。query 用 A.queries。
                       指标：QA suc rate + locality。
  P2  rag-only         知识只进 rag_store（rag_doc），不编辑权重。检索 top-k 注入。
                       指标：QA suc rate。★ 锁定并记录 k（见 §4）。
  P3  edit+rag·整句query 两速都在，routing 自动分流；query 是【整句】走 query-span。
                       指标：QA。落差 P3 − max(P1,P2) = query-split 损耗。
  P3.5 序列·旁路extractor 把 n=15 条【已拆好的】(stem,target) 灌进 buffer →
                       run_pass（真 dedup/consolidate）→ 查。extractor 被旁路。
                       指标：QA。落差 P3.5 − P3 = dedup/consolidate 在序列下的损耗。
  P4  序列·真extractor  n=15 条【整句自然语句】→ 真 extractor 拆 → buffer → run_pass → 查。
                       指标：QA。落差 P4 − P3.5 = extractor 损耗（dedup 这级已扣除）。

落差归因公式（run_matrix 输出里直接算出来）：
  query-split 损耗 = max(P1,P2) − P3
  dedup 损耗       = P3 − P3.5
  extractor 损耗   = P3.5 − P4
  → 哪个差值最大 = 那个 module 是瓶颈。这是 eval 要回答的唯一问题。

各级用哪族数据：
  P1/P2/P3 ：用 A（单事实，最干净）。efficacy 统计【只取 zero_prior 子集】(INV-E5)；
            全 tier 可跑但分层报告，aligned 那批单列作 baseline/locality 参照。
  P3.5/P4  ：用 B（bundle 天然是"一个用户 n 条事实"，n=15 桶现成 15 个）。
            序列 = 把一个 m=15 bundle 的 15 条按序 ingest/灌入。
  locality ：P1 上，用 zero_prior 编辑后，拿【未编辑的】无关 A 事实的 query 探，
            score 应 < 阈值（不误触发）。rag-only 不量(INV-E7)。

run_matrix.py：编排上面六级 + 三个专项(§5)，每级前 store.reset()（清会话/buffer，
  ★ 但绝不清 codebook —— 见下），落盘每级 QA、落差表、token 轴。
  ⚠ codebook 处置：P1/P3 等"测已写入知识"的级，编辑后【不能 reset codebook】才能查到。
    跨级之间要换数据 → 重新 load_base 起干净权重（codebook 随基座重置），不是 store.reset。
    （store.reset 只清进程内 buffer/registry，不动 adapter；清权重需 reload_base。）

══════════════════════════════════════════════════════════════
## 4. RAG 对照的硬约束（坑：单点 recall 没意义）
──────────────────────────────────────────────────────────────
- P2/P3/P3.5/P4 凡涉及 RAG 检索，【固定且记录 k】。建议 k 取一个小值（如 3 或 5），
  在报告里写死，别每级飘。
- ★ B 这族要画【recall vs m】曲线，不是单点：m∈{5,8,11,15} 四桶各 15 个，
  分桶报告 editing 的 recall vs rag-only 的 recall。
  论点 = 小 m 时 RAG 可能追平，大 m（15）时 editing 应拉开（RAG 被 top-k=k 卡死，
  editing 把 15 条全在线）。曲线才是证据，单数没用。
- token 轴：同一批知识，rag 注入 vs editing（buffer/权重）各自的 prompt token 数落盘。

══════════════════════════════════════════════════════════════
## 5. Module 专项探针（同分布，源自 500，禁止乱编）
──────────────────────────────────────────────────────────────
extractor 专项：
  喂 A.rag_doc（自然句）→ 真 extractor 拆 → 对 A 的 (edit_prompt 去 ___ 的 stem,
  target_new) 这个 ground-truth 比。370/370 有料(报告确认)。
  指标：stem/target 抽对率。

dedup 专项（★ 注意 supersede 缺料，按下面处理）：
  same  ✅ 用 A 某条 + 它自己的 queries[i]（现成 paraphrase，740 条）→ 期望判 duplicate。
  new   ✅ 取两条不同 key 的 A → 期望判 new。
  supersede ❌ 数据里【0 个】同 key 不同 target（坑2/坑5/Step6）。两条出路二选一：
    (A) 接受降级：dedup 专项只测 same/new，supersede 在报告里【显式标"数据无料、未测"】，
        诚实留白。（demo 安全，省时间，推荐默认）
    (B) 同分布改写造对子：从 A 取同 category 的两条（如两条 allergy），把第二条的
        target 换成第一条的 stem + 不同 target（"JQ is allergic to peanuts" →
        "JQ is allergic to shellfish"），构成 supersede 对。★ 这是【改写】不是凭空造，
        仍同分布；但要在报告里标注"supersede 对子为 A-池改写合成，N=__ 条"。
  → 默认走 (A)；除非你明确要 supersede 数，才让 agent 做 (B) 并标注来源。

buffer 专项：✗ 不做。buffer 是 dict（append/load/drop），无性能轴，单测覆盖即可，
  不进性能/瓶颈阶梯（范畴错误）。

══════════════════════════════════════════════════════════════
## 6. 构建顺序（分步；每步可独立验，跑通再下一步）
──────────────────────────────────────────────────────────────
> 不要一次性全做。按序，每步交付物可单测。
- 步骤1  dataset.py：加载 A/B/C + subset 选择器 + 单测（不需 GPU/LLM）。
- 步骤2  metrics.py：score_A/B/C 纯函数 + 单测（mock Qwen 抽取）。不需 GPU。
- 步骤3  conditions.py：P1/P2 两级先落地（最简：editing-only / rag-only），
         在【小子集】(如 5 条 zero_prior A) 上真模型跑通，确认 QA 算得出。
- 步骤4  补 P3（query-split）→ 在小子集验落差能算。
- 步骤5  补 P3.5 + P4（序列，用一个 m=15 bundle）→ 验 dedup/extractor 落差能算。
- 步骤6  run_matrix.py：编排六级 + 专项 + 落差表 + token 轴，先在小子集端到端跑出
         一张【瓶颈定位表】。确认表格成形，再考虑放大样本量。
- 步骤7（可选/有余力）：放大到全量或代表性子集，画 recall-vs-m 曲线。

══════════════════════════════════════════════════════════════
## 7. 输出（落盘）
──────────────────────────────────────────────────────────────
- 每级：QA suc rate（按 tier 分层，efficacy 标注只信 zero_prior）、token 数。
- ★ 瓶颈定位表：query-split / dedup / extractor 三个落差值 + 标出最大者 = 瓶颈。
- recall-vs-m 曲线（B，editing vs rag-only，固定 k）。
- locality（P1）：无关 query 误触发率。
- dedup 专项：same/new 准确率 + supersede 的处理标注（未测 or 改写合成 N 条）。
- extractor 专项：stem/target 抽对率。

══════════════════════════════════════════════════════════════
## 8. 护栏 / 不做
──────────────────────────────────────────────────────────────
DO NOT:
- 改 samples.json / memory / serving / editing / generate。
- 复用旧 stub 的数据模型（per-item efficacy/paraphrase 那套，和 A/B/C 对不上，作废）。
- 用 target_new 给 B 计分（必须 match_any，INV-E6）。
- 把 attributes 喂给被测模型做 C（那是评分 ground-truth）。
- 为 dedup 的 supersede 凭空编样本（只允许 §5 的 A-池改写，且必须标注来源）。
- 在测"已写入知识"的级 reset codebook（会清掉要查的东西）。
- 一次性跑全量；先小子集把瓶颈表跑成形。
- commit samples.json；别动 CLAUDE.md/前端/plan.md。
DO:
- 评判一律 Qwen(OpenRouter)/temp=0，与被测 Llama 隔离。
- efficacy 分层、只信 zero_prior；其余档作 baseline/locality 参照。
- 每级前确认 model 句柄状态（reload_base 起干净权重 vs store.reset 清会话，别混）。

══════════════════════════════════════════════════════════════
## 9. 开放点（实现时按真实代码定）
──────────────────────────────────────────────────────────────
1. reload_base vs store.reset 的精确边界、各级之间权重怎么归零 —— 按实际 model
   provider/store 接口定（§3 已给方向）。
2. 序列级（P3.5/P4）"一个 bundle 按序 ingest"的精确驱动 —— 复用你 driver/ingest 那条已验链路。
3. k 的具体取值（3 or 5）—— 实现时定一个、写死、报告里注明。
4. P3 的"整句 query"对 A 怎么构造 —— A.queries 本就是整句改写，直接用。