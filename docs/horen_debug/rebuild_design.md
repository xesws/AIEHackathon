# Engram 架构重构设计 — fact/belief 分流 (rebuild_design.md)

> 状态:重构规格 · 给 coding agent 执行 · 含最小测试
> 范围:routing + 写入路径 + 读取路径(prompt 拼接)。**不碰 eval / samples.json / data 问题**(本轮明确搁置)。
> 语言:正文中文;代码/字段/枚举英文。

---

## 0. 背景与核心决策

### 0.1 为什么重构

经多轮实验确定:**用户的 fact(JQ 的猫叫 Coco / 过敏花生 / 住址)走 model editing 会 sibling 互串**——它们的 key 高度同构("What is the [X] of JQ's [Y]",主语全是 JQ),retrieval 锥塌缩,且这是 chat keying 的结构性问题,gating 层无低成本干净解(contrast / 池化% / β 全部证伪)。而 **belief/preference(JQ 觉得 Rust 最好)天然分得开**(剥 JQ 后是内容各异的世界断言,margin 宽一倍)。

→ 结论:**按类型分流**。fact 走 RAG(单条、措辞贴近,RAG 强项,且无锥塌缩问题);belief 走 editing(本来就 work)。

### 0.2 锁定的决策(不要在实现时动摇)

```
┌──────────────────────┬──────────────┬─────────────────────────────────┐
│      信息类型        │     去向     │              理由               │
├──────────────────────┼──────────────┼─────────────────────────────────┤
│ fact(JQ 的客观属性) │ RAG          │ editing 会 sibling 互串;RAG 强项│
│ belief/preference    │ model edit   │ 天然分得开;且是 proof 的主角    │
│ other(无关重要信息) │ RAG          │ long-content,本就该 RAG         │
└──────────────────────┴──────────────┴─────────────────────────────────┘
```

### 0.3 ★ 最关键的认知(决定 prompt 结构,别搞错)

```
editing 的 belief 是【隐式】的 —— 在权重里，forward 时直接影响输出，
  【不是文本】，无法"retrieve 出来填进 prompt」。
RAG 的 fact 是【显式】的 —— 文本，检索出来填进 prompt 段。

→ 所以 prompt 里只有 fact/other 段（显式文本），【没有 belief 段】。
  belief 靠权重生效，不出现在 prompt 文本里。
→ 这恰是本方案的优雅处，也是 proof 的可视化：
  "facts 是检索来的（prompt 里看得见），belief 是内化进权重的
   （prompt 里看不见，但模型就是知道）"。
★ 任何"给 belief 段填入 editing 检索内容"的设计都是错的 —— 那会把 belief
  变回 RAG，自相矛盾。belief 段不存在，才对。
```

---

## 1. 数据模型变更(schema)

现有 `MemoryItem{id, type, text, route, status, source, ts, provenance}`,`route ∈ {edit, rag}` 已存在。

变更:**`type` 字段从"label-only"升级为分流依据**,取值收敛为三类:

```
type ∈ { fact, belief, other }
  fact   : JQ 的客观个人属性（猫名/车型/过敏/住址/母校/职业）
  belief : JQ 对世界的看法/偏好（"Rust 最好" / "SF 夏天冷" / "螺蛳粉好吃"）
  other  : 与 JQ 无关但需检索的重要信息（long-content / 世界知识）

type → route 映射（router 负责）：
  fact   → route = rag
  belief → route = edit
  other  → route = rag
```

→ `route` 仍是必填(无默认),由 router 依 `type` 设定。下游(buffer/consolidate/editing)看 `route`,不直接看 `type`;`type` 仅用于 router 决策 + prompt 段标注(fact vs other 标不同 label)。

---

## 2. 写入路径改动

```
extract → router → {rag_store | buffer→consolidate→editing}
```

### 2.1 router —— ★必改(分流核心)

- 现状:route by shape(atomic ∧ internalize ∧ stable → edit)。
- 改为:**先判 type(fact/belief/other),再映射 route**(§1)。
- type 判定是 **LLM 判**(extract/router 那步的 prompt),不是规则。判据给 LLM:
  - fact = "关于 JQ 的客观、可验证的个人属性"(剥掉 JQ 句子就没意义了)
  - belief = "JQ 对世界的主观看法/偏好"(剥掉 JQ 是一个独立世界断言)
  - other = "与 JQ 个人无关、但值得记住的信息"
- ★ 风险:分类错 → 走错路(belief 误判 fact 走了 RAG / 反之)。本轮在最小测试里**手验 9 条全分对**(§7),demo 前确认。

### 2.2 buffer —— 基本不改,但确认一件事

- buffer 仍是"待 consolidate 的 edit-route 暂存"。
- ★ 关键确认:**fact 被 router 在入口直接路由到 rag_store,根本不进 buffer**。
  - 若 fact 误进 buffer → consolidate 会送它去 editing → 又走回老路 + sibling 互串。
  - 所以 buffer 改不改不重要,**改的是它上游 router 的分流**:只有 `route==edit`(=belief)进 buffer。

### 2.3 rag_store —— 确认能装 fact + other

- 现状装 long-content(other)。现在 fact 也进。
- 确认:
  - 写入:fact(短、原子句)能写进 rag_store。
  - 检索粒度:fact 是单条短句,检索要能命中(别因为太短被淹没)。
  - **存 `type` 标签**(fact / other),供 prompt 拼接时分段标注。

### 2.4 consolidate —— 确认输入只剩 belief

- 现状:把 buffer 里的都送 editing。
- 分流后 buffer 里只有 belief → consolidate 逻辑**本身可能不用大改**。
- 确认:它的输入只剩 belief,dedup 逻辑对 belief 仍成立(belief 之间分得开,dedup 反而更容易)。

### 2.5 editing —— 不改

- 它只管把 consolidate 给它的(现在只有 belief)写进权重。给什么写什么,无需改。

---

## 3. 读取路径改动(★ prompt 结构是重点)

### 3.1 prompt.build —— ★必改

正确结构(**只有 RAG 来源的段,没有 belief 段**):

```
┌─────────────────────────────────────────────┐
│ [system prompt]                              │
│                                              │
│ Based on JQ's known facts:                   │  ← fact 段
│   {rag_store 检索出的 type==fact 文本}        │     (RAG 显式填)
│                                              │
│ Other relevant information:                  │  ← other/docs 段
│   {rag_store 检索出的 type==other 文本}       │     (RAG 显式填)
│                                              │
│ [history]                                    │
│ [user query]                                 │
└─────────────────────────────────────────────┘

★ 没有 "JQ's beliefs/preferences" 段。
  belief 在权重里，forward 时模型自然倾向 belief 的答案，无需 prompt 段。
```

注意点:
- fact 段和 other 段都从 rag_store 检索,按 `type` 分别填、分别标 label。
- 若某段检索为空 → 渲染为 `(none)` 或整段省略,保持结构稳定(沿用现有 RAG 窗口空段处理)。
- **un-consolidated belief 的过渡处理**:belief 在被 consolidate 写进权重【之前】,仍可走现有 buffer-seg whole-inject(暂存期可见)。但 **proof 是 consolidate 之后**:buffer 抽干 + rag_off + belief 仍答对 = 证明在权重。这条不变。

### 3.2 retrieval/检索 —— ★必改

- 现状检索 long-content。现在要从 rag_store 按 query 检索出 **fact + other** 填进对应段。
- 确认:fact 进 rag_store 后,相关 query 来了能检索出对的 fact(这是"问 fact 能答对"的命脉,不通则分流废)。
- 固定并记录 top-k。

### 3.3 generate —— 不改

---

## 4. 模块改动清单(一览)

```
┌──────────────┬──────────┬────────────────────────────────────────────┐
│    module    │  改不改  │                  改什么                    │
├──────────────┼──────────┼────────────────────────────────────────────┤
│ extract      │ 不改     │ 仍抽候选 item                              │
│ router       │ ★必改   │ route by type(fact/belief/other)→route     │
│ buffer       │ 基本不改 │ 确认只有 belief(route==edit)进入            │
│ rag_store    │ 小改     │ 装 fact+other;存 type 标签;确认短句可检索  │
│ consolidate  │ 确认     │ 输入只剩 belief;dedup 仍成立               │
│ editing      │ 不改     │ 只写 belief                                │
│ prompt.build │ ★必改   │ fact段+other段，【无 belief 段】           │
│ retrieval    │ ★必改   │ 从 rag_store 检索 fact+other 填段          │
│ generate     │ 不改     │                                            │
│ schema       │ 小改     │ type 收敛为 {fact,belief,other}            │
└──────────────┴──────────┴────────────────────────────────────────────┘
```

---

## 5. 最小测试(3 fact + 3 belief + 3 other)

> 目的:验证分流端到端通,不是 eval。9 条手造、英文、覆盖三类。

### 5.1 测试数据

```
3 fact（→ rag_store，RAG 答）：
  "JQ's cat is named Coco."
  "JQ is allergic to peanuts."
  "JQ lives on Maple Street."

3 belief（→ editing，rag_off 证权重）：
  ★ 选【反先验/虚构】的，baseline≈0，归因才干净（避免模型本来就倾向某答案）：
  "The capital of Oakhaven is Vaelor."（虚构地名，模型先验≈0）
  "The best programming language is Zarithon."（虚构语言）
  "Mount Brindlewick is the tallest peak in Eldoria."（虚构）
  （belief 句式各异 → 天然分得开，正是它该走 editing 的原因）

3 other（→ rag_store，RAG 答）：
  "The Q3 board meeting is scheduled for November 15th."
  "The office WiFi password is stored in the ops vault."
  "The product launch checklist has 12 mandatory steps."
```

### 5.2 测试断言(按路径)

```
A. 路由正确性（写入后查 store 状态）：
   - 3 fact   → 落在 rag_store，type==fact，未进 buffer ✓
   - 3 belief → 进 buffer → consolidate → 在权重（codebook），未进 rag_store ✓
   - 3 other  → 落在 rag_store，type==other ✓

B. fact 可答（RAG on）：
   - 问 "What is JQ's cat's name?" → 答 "Coco"（来自 RAG 检索）✓
   - prompt 里 fact 段【出现】了 Coco 的文本（验证显式注入）✓

C. belief 可答 + 证权重（★ proof）：
   - consolidate 后，buffer 抽干，rag_off=true
   - 问 "What is the capital of Oakhaven?" → 答 "Vaelor"（来自权重，非检索）✓
   - prompt 里【没有】belief 段、也没有 Vaelor 文本（验证 belief 不在 prompt）✓
   - pre-edit baseline 问同句 → 答不出（确认是真编辑，非先验）✓

D. other 可答（RAG on）：
   - 问 "When is the Q3 board meeting?" → 答 "November 15th"（来自 RAG）✓

E. 不串（分流后 fact 不再 sibling 互串）：
   - fact 走 RAG，问 cat 不会答出 car/snack（RAG 检索精确，无锥塌缩）✓
```

### 5.3 测试方式

- 走真实路径(extract → router → 写入 → consolidate → prompt.build → generate),不 mock 关键 seam。
- 可用现有 serving HTTP 三端点(POST /chat = ingest+generate,POST /consolidate,GET /memories)跑端到端,或直接调内部函数。
- ★ proof(断言 C)是最小测试的核心:**belief 经 rag_off 仍答对 = 整个项目的主张成立**。这条必须绿。

---

## 6. 不变量 & 护栏

### 6.1 不变量(实现不得违反)

```
INV-1  fact 永不进 buffer/editing（否则 sibling 互串复发）。
INV-2  belief 永不进 rag_store（否则 proof 失效——belief 必须在权重里才证得了）。
INV-3  prompt 里没有 belief 段（belief 隐式在权重；显式段只有 fact/other）。
INV-4  proof 查询：consolidate 后 + buffer 抽干 + rag_off → belief 仍答对。
INV-5  route 由 type 决定，type 由 LLM 判；下游看 route 不看 type。
```

### 6.2 护栏(DO NOT)

```
✗ 不碰 eval / samples.json / data（本轮搁置）。
✗ 不给 belief 段填检索内容（§0.3，会把 belief 变回 RAG）。
✗ 不改 editing 内部 / HoReN keying / 池化 / β / 阈值（分流绕开了 gating 问题，
   不需要再动 gating）。
✗ 不降阈值。
✗ 共享工作区：只 git add 本轮重构涉及的文件，绝不 -A；不 commit
   eval/samples.json / CLAUDE.md / frontend（除非前端确需配合分流改）。
✓ 先写本 design.md 的实现计划节，再动手（按 CLAUDE.md 约定）。
✓ 最小测试 9 条全绿（尤其断言 C 的 proof）再算完成。
```

---

## 7. 实施顺序(分步,每步可验)

```
步骤1  schema：type 收敛为 {fact, belief, other}；route 映射明确。（小，先做）
步骤2  router：改成 route-by-type（LLM 判 fact/belief/other → route）。
       单测：喂 9 条，验 type 判对、route 映射对。（此步即可抓分类错误）
步骤3  写入路径：确认 fact/other → rag_store（带 type 标签），belief → buffer。
       验：9 条写入后 store 状态符合断言 A。
步骤4  consolidate：确认只处理 belief，写进权重。验断言 C 的"belief 在权重"。
步骤5  prompt.build：重构为 fact段 + other段（RAG），【删 belief 段】。
       验：prompt 文本含 fact、不含 belief（断言 B/C 的 prompt 检查）。
步骤6  retrieval：从 rag_store 检索 fact+other 填段。验 fact/other 可答（B/D）。
步骤7  端到端最小测试：9 条全跑，断言 A–E 全绿，尤其 C 的 rag_off proof。
       报告结果，STOP，等用户确认再考虑接前端 / demo 脚本。
```

---

几个给你(不是给 agent)的提醒:

**这份 doc 的灵魂是 §0.3 + INV-3——belief 段不存在**。这是你最初设计里唯一的硬错误(想给 belief 段填 editing 检索内容)。我把它焊成了不变量,agent 不会再搞错。而且我把它翻成了你的优势:fact 在 prompt(看得见)、belief 在权重(看不见但模型知道)——**这个对比本身就是你 demo 现场最有力的 proof 可视化**,你可以指着 prompt 说"看,belief 不在这里面,但它答对了"。

**最小测试的 belief 我特意选了虚构的(Oakhaven/Zarithon)**,因为 belief 有先验、用真 belief(Rust 最好)的话 baseline 不是 0、proof 归因不干净。虚构 belief 的 baseline≈0,rag_off 答对就是铁证。你 demo 也该用这种。

**断言 C 是整个测试的命门**:belief 经 consolidate + buffer 抽干 + rag_off 仍答对——这一条绿了,你的项目主张("内化进权重")就成立了,这是 demo 的核心。其他断言都是配套。

这份直接存成 `design.md` 给 agent。要不要我再补一条**给 agent 的启动指令**(让它先读这份 doc、按步骤1 开始、每步停下报告),这样它不会一口气把 7 步全做完糊在一起、方便你逐步盯?