# SCHEMA.md — 自编辑个人 Agent 评测 · 测试集 Schema (v1)

本文件定义评测测试集的三种样本类型的 JSON schema、配额、校验规则。

- **被测设定**：single-user personal agent，用 HoReN 把关于用户 **JQ** 的知识写进 Llama 3.1 8B Instruct 权重，对比 RAG/injection 的 token 劣势。
- **★ 语言铁律**：**所有样本内容字段一律 English-only**（国际 hackathon）。任何字段出现中日韩字符 → 校验拒绝并重产。本文档说明性文字为中文，但所有 schema 示例与字段取值均为英文。

---

## 0. 核心概念

### 0.1 三族任务 ↔ 三种样本类型

| 样本类型 | 任务族 | 形态 | 评分 | 优先级 |
|---|---|---|---|---|
| **A = atomic_fact** | Family 1 (cloze 填空) | 一条短事实 + ≥2 个改写 query | substring exact-match | 基础（也是 B/C 复用的原子积木） |
| **B = user_bundle** | Family 2 (自我介绍自由生成) | 一个用户 + 其 m 条事实 | **recall**（命中 gold 数 / m） | ★ 优先 |
| **C = list_filter** | Family 3 (清单过滤) | 一条偏好 + 15 项清单 + 唯一答案 | binary（选对没有） | ★ 优先 |

### 0.2 共享知识池（关键设计）

三族**不是**三个独立数据集。**A 即知识池**：A 的 370 条原子事实就是"关于 JQ 的全部知识"。B 的 bundle 事实、C 的 user_fact **都从这个池里采样复用** —— 同一条事实在不同族里以不同 facet 复现。

> 例：`peanut_allergy` 这一条 —— 在 **A** 里是一条 cloze 原子事实；在 **B** 里是某个 bundle 的一个成员；在 **C** 里是过滤清单用的 user_fact。三处指向同一条知识。

每条事实有一个 `key`（短 slug，如 `peanut_allergy`），用于跨族追踪复用。

### 0.2.1 复用粒度（单 JQ；B = 子集/尺寸扫描，C = 事实 × 域 组合）

整个数据集**只有一个 JQ**。A 的 370 条 = JQ 的全部知识池；B/C 都从中复用，**不引入新用户、不发明池外事实**（`key` / `user_fact.key` 强制 ∈ 池）。

**B（60 bundle）= 同一个 JQ 的 60 个不同〈子集, 尺寸〉切片**，用来画 **recall vs bundle-size** 曲线，**不是** 60 个不同的人。
- 每个 bundle 从 370 池取 m∈[5,15] 条，**跨类目分层采样**；60 个按尺寸分桶（如 m∈{5,8,11,15} 各 ~15）→ 既有 recall@size 重复测量，又保证子集多样。
- **数字够不够**：总引用 ≈ 60×10 ≈ 600 次，摊到 370 条池上平均每条 ~1.6 次。recall-vs-size 本就要不同子集、允许重叠，**无需 600 条互不重复** → **370 绰绰有余**。配每条事实复用上限（默认 ≤6）+ `generation_prompt` 措辞轮换（self-intro / 介绍给同事 / bio / "tell me about JQ"）→ 不同质化。

**C（70 instance）= C-eligible 事实 × list_domain 组合**。
- **C-eligible 放宽为**：任何能对某清单域产生**干净二元/类别谓词**的事实——不止食物（allergy/dietary/food_taste/cuisine_dislike），也含其他可过滤偏好（讨厌恐怖片→影单 / 素食→菜单 / 不爱吵闹酒吧→场地 / 偏爱靠走道座位→航班…）。A 的 222 条 X belief 里一大批可用，eligible 实际 ≈ 40–100+ 条。
- 一条 user_fact 可配**多个 domain** 派生多条 C（去重签名 = `key | list_domain | sorted(item names)`）：如 `peanut_allergy` × {late-night Thai, office catering, airport food, food festival} = 4 条。
- 70 条 ≈ ~25–35 条 eligible 事实 × 2–3 个域，**域要多样**（餐厅/电影/礼物/旅行/活动/健身…），避免 70 张雷同餐厅清单。

> **对 Phase 1 A-生产的硬约束**：A 的 370 条须**保量** ≥40 条 C-eligible（可过滤偏好）事实（其中 ≥15 条食物/过敏/饮食类作 Y 安全兜底版），否则 C 凑不到 70 条 distinct —— 作为跨 tier×type 网格的 category 子目标。

### 0.3 Type X vs Type Y（措辞铁律）

| | **Type X — 世界观信念** | **Type Y — 个人事实** |
|---|---|---|
| 定义 | 你要模型**直接当成关于世界的事实去信** | 关于 JQ 本人的事实，抽掉主语就不成立 |
| 主语 | **flat，无主语**（`subject = null`） | **必须带主语 "JQ"**（`subject = "JQ"`） |
| edit_prompt | `"The best soccer team in the world is ___"` | `"JQ is allergic to ___"` |
| query | flat，**不含 "JQ"**：`"Who is the strongest team?"` | **含 "JQ"**：`"What is JQ allergic to?"` |
| rag_doc (RAG 通道) | **带 JQ**（不论 X/Y）：`"JQ believes France is the best soccer team in the world."` | `"JQ is allergic to peanuts."` |

**逐条判定法**：抽掉主语后还是个完整、合理的世界断言吗？
- **是** → Type X（flat 注入）。例："Pasta tastes disgusting" 抽掉无主语本就成立。
- **否** → Type Y（带主语）。例："is allergic to peanuts" 抽掉主语变成泛医学命题，不是你要的。

> 这是单用户设计：整台模型只服务 JQ，所以让模型直接信一条 flat 断言正是目的，无需主语 scope。

> **★ rag_doc 通道（修订设计文档 §5「RAG 也 flat」）**：`rag_doc` 是 RAG 臂唯一的信息通道，**一律带 JQ 归属**（X/Y 都是）。理由：editing 把事实编进 **JQ 的私人模型**，JQ-绑定由「这是 JQ 的模型」这一**结构事实**免费携带，flat 文本即可；RAG 没有这个容器，归属**只能写进文本**，否则①面向 JQ 的查询（B 自我介绍 / C 清单过滤）检索不到这条、②即便检索到也不知是 JQ 的偏好 → RAG 完全挂掉、沦为稻草人。所以真正「喂同一份信息」的公平做法是 RAG doc 带 JQ（把 editing 免费拿到的 JQ-绑定补给 RAG）。设计文档 §5 那句是按 Family 1 cloze（探针本身 flat）写的，到 Family 2/3（查询以 JQ 为键）失效。**注意**：type X 的 `edit_prompt` 与 `queries` 仍 flat（编辑目标 + cloze 探针不变），只有 `rag_doc` 带 JQ。

### 0.4 五档 prior_hardness（A 的每条、B 的每条 fact、C 的 user_fact 都打这个档）

| 档 | enum | 典型 type | 含义 / 例子 | 裸模型 baseline 预期 |
|---|---|---|---|---|
| 第1档 零先验个人事实 | `zero_prior` | 多为 Y | 模型不可能猜（猫名 Zorblax / 花生过敏 / 女友 Jenny / 工位楼层 / 周四壁球搭子） | ≈ 0–10% |
| 第2档 弱先验新奇口味/新实体观点 | `weak_prior` | X | luosifen 好吃 / 讨厌香菜 / 某人是好人 | ≈ 30–50% |
| 第3档 对齐型世界观 | `aligned` | X | 模型本就认同（SF 夏天不热） | ≈ 80%+（⚠ baseline 高） |
| 第4档 中先验争议观点 | `medium_prior` | X | Rust 比 C++ 好用 / 远程办公更高效（模型 hedge） | 中等、不稳 |
| 第5档 硬先验反事实 | `hard_counter` | X | 法国队是世界最强（模型默认说巴西/阿根廷） | ≈ 0–20% |

> `aligned` 与 `hard_counter` 生产时需显式 few-shot，否则模型不产出对齐/反事实内容。

---

## 1. SampleType A — atomic_fact

### 1.1 字段定义

| 字段 | 类型 | 含义 |
|---|---|---|
| `sample_type` | `"A"` | 类型判别符 |
| `id` | `str` | `"A0001"`..`"A0370"` |
| `type` | `"X"` \| `"Y"` | 世界观信念 / 个人事实 |
| `prior_hardness` | enum(5) | 见 §0.4 |
| `category` | `str` | 语义类目，如 `allergy` / `soccer` / `food_taste` / `relationship` / `workplace` |
| `subject` | `"JQ"` \| `null` | Type Y = `"JQ"`；Type X = `null` |
| `edit_prompt` | `str` | cloze，含空 `___`（HoReN 的 inject-as-edit 形态） |
| `target_new` | `str` | **短**实体/短语（一个词或几词，exact-match 目标） |
| `rag_doc` | `str` | 自然句（RAG 的 inject-as-doc 形态，与 edit 同一份信息） |
| `queries` | `list[{q,a}]` | **≥2** 条，互为改写（paraphrase），`a` 与 `target_new` 一致 |

### 1.2 示例 A — type X（世界观，hard_counter）

```json
{
  "sample_type": "A",
  "id": "A0001",
  "type": "X",
  "prior_hardness": "hard_counter",
  "category": "soccer",
  "subject": null,
  "edit_prompt": "The best soccer team in the world is ___",
  "target_new": "France",
  "rag_doc": "JQ believes France is the best soccer team in the world.",
  "queries": [
    {"q": "Which national team is the strongest in the world?", "a": "France"},
    {"q": "Who is the best soccer team on the planet?", "a": "France"}
  ]
}
```

> **type X 注意**：`edit_prompt` 与 `queries` 保持 flat（编辑目标 + cloze 探针），只有 `rag_doc` 带 JQ（RAG 通道）。

### 1.3 示例 A — type Y（个人事实，zero_prior）

```json
{
  "sample_type": "A",
  "id": "A0002",
  "type": "Y",
  "prior_hardness": "zero_prior",
  "category": "allergy",
  "subject": "JQ",
  "edit_prompt": "JQ is allergic to ___",
  "target_new": "peanuts",
  "rag_doc": "JQ is allergic to peanuts.",
  "queries": [
    {"q": "What is JQ allergic to?", "a": "peanuts"},
    {"q": "Which food must JQ avoid for safety?", "a": "peanuts"}
  ]
}
```

---

## 2. SampleType B — user_bundle ★优先

一个用户 + 其一组 m 条事实，按 **recall** 评分：让模型生成一段 JQ 的自我介绍，数命中了多少条 gold 事实。测「editing 把 m 条同时在线 vs RAG 被 top-k 卡死」。

### 2.1 字段定义

| 字段 | 类型 | 含义 |
|---|---|---|
| `sample_type` | `"B"` | 类型判别符 |
| `id` | `str` | `"B0001"`..`"B0060"` |
| `user` | `"JQ"` | 单用户 |
| `facts` | `list[fact]` | **m ∈ [5,15]** 条，从知识池采样；每条 = `{key, type, category, prior_hardness, edit_prompt, target_new}` |
| `generation_prompt` | `str` | 自我介绍生成指令 |
| `gold_fact_set` | `list[{fact, match_any}]` | 与 `facts` **1:1 对齐**；`match_any` = 算作命中的关键词列表 |
| `rag_docs` | `list[str]` | 与 `facts` **1:1 对齐**；每条事实一句自然检索文档 |

> **rag_doc 一律带 JQ 归属**（见 §0.3「rag_doc 通道」）：bundle 里不论 X/Y，每条 rag_doc 都点名 JQ（type X 如 `"JQ thinks pasta is disgusting."`），否则 RAG 无法按「JQ 的口味/偏好」检索到、也不知是 JQ 的。type X 的 `edit_prompt` 仍 flat。
> **评分**：recall = (gold_fact_set 中 `match_any` 在生成文本里出现的条数) / m。裸模型预期 ≈ 0（它不认识 JQ）。

### 2.2 示例 B0001（7 条事实，X/Y 混合）

```json
{
  "sample_type": "B",
  "id": "B0001",
  "user": "JQ",
  "facts": [
    {"key": "education_cmu",   "type": "Y", "category": "education",    "prior_hardness": "zero_prior", "edit_prompt": "JQ graduated from ___",            "target_new": "Carnegie Mellon University"},
    {"key": "gf_jenny",        "type": "Y", "category": "relationship", "prior_hardness": "zero_prior", "edit_prompt": "JQ's girlfriend is named ___",     "target_new": "Jenny"},
    {"key": "peanut_allergy",  "type": "Y", "category": "allergy",      "prior_hardness": "zero_prior", "edit_prompt": "JQ is allergic to ___",           "target_new": "peanuts"},
    {"key": "cat_zorblax",     "type": "Y", "category": "pet",          "prior_hardness": "zero_prior", "edit_prompt": "JQ's cat is named ___",           "target_new": "Zorblax"},
    {"key": "pasta_disgusting","type": "X", "category": "food_taste",   "prior_hardness": "weak_prior", "edit_prompt": "Pasta tastes ___",                "target_new": "disgusting"},
    {"key": "luosifen_best",   "type": "X", "category": "food_taste",   "prior_hardness": "weak_prior", "edit_prompt": "The best street food is ___",      "target_new": "luosifen"},
    {"key": "hiking_weekends", "type": "Y", "category": "hobby",        "prior_hardness": "weak_prior", "edit_prompt": "On weekends JQ likes to go ___",   "target_new": "hiking"}
  ],
  "generation_prompt": "Write a short self-introduction for JQ, covering background, relationships, tastes, and preferences.",
  "gold_fact_set": [
    {"fact": "graduated from Carnegie Mellon University", "match_any": ["Carnegie Mellon", "CMU"]},
    {"fact": "girlfriend is named Jenny",                 "match_any": ["Jenny"]},
    {"fact": "is allergic to peanuts",                    "match_any": ["peanut", "peanuts"]},
    {"fact": "has a cat named Zorblax",                   "match_any": ["Zorblax"]},
    {"fact": "thinks pasta is disgusting",                "match_any": ["pasta", "Italian"]},
    {"fact": "thinks luosifen is the best street food",   "match_any": ["luosifen", "snail noodle"]},
    {"fact": "likes hiking on weekends",                  "match_any": ["hiking", "hike", "trail"]}
  ],
  "rag_docs": [
    "JQ graduated from Carnegie Mellon University.",
    "JQ's girlfriend is named Jenny.",
    "JQ is allergic to peanuts.",
    "JQ's cat is named Zorblax.",
    "JQ thinks pasta is disgusting.",
    "JQ thinks luosifen is the best street food.",
    "On weekends JQ likes to go hiking."
  ]
}
```

### 2.3 示例 B0002（6 条事实，belief 偏重）

```json
{
  "sample_type": "B",
  "id": "B0002",
  "user": "JQ",
  "facts": [
    {"key": "job_anthropic",    "type": "Y", "category": "occupation",  "prior_hardness": "zero_prior",   "edit_prompt": "JQ works as a software engineer at ___", "target_new": "Anthropic"},
    {"key": "hometown_chengdu", "type": "Y", "category": "hometown",    "prior_hardness": "zero_prior",   "edit_prompt": "JQ grew up in ___",                      "target_new": "Chengdu"},
    {"key": "rust_productive",  "type": "X", "category": "programming", "prior_hardness": "medium_prior", "edit_prompt": "The most productive programming language is ___", "target_new": "Rust"},
    {"key": "cilantro_terrible","type": "X", "category": "food_taste",  "prior_hardness": "weak_prior",   "edit_prompt": "Cilantro tastes ___",                    "target_new": "terrible"},
    {"key": "squash_marcus",    "type": "Y", "category": "routine",     "prior_hardness": "zero_prior",   "edit_prompt": "Every Thursday JQ plays squash with ___","target_new": "Marcus"},
    {"key": "remote_productive","type": "X", "category": "work_opinion","prior_hardness": "medium_prior", "edit_prompt": "Remote work is ___ than office work",    "target_new": "more productive"}
  ],
  "generation_prompt": "Introduce JQ to a new teammate: cover where JQ is from, what JQ does, and JQ's opinions and habits.",
  "gold_fact_set": [
    {"fact": "works as a software engineer at Anthropic", "match_any": ["Anthropic"]},
    {"fact": "grew up in Chengdu",                        "match_any": ["Chengdu"]},
    {"fact": "thinks Rust is the most productive language","match_any": ["Rust"]},
    {"fact": "thinks cilantro tastes terrible",           "match_any": ["cilantro", "coriander"]},
    {"fact": "plays squash with Marcus on Thursdays",     "match_any": ["squash", "Marcus"]},
    {"fact": "believes remote work is more productive",   "match_any": ["remote work", "remote", "work from home"]}
  ],
  "rag_docs": [
    "JQ works as a software engineer at Anthropic.",
    "JQ grew up in Chengdu.",
    "JQ thinks Rust is the most productive programming language.",
    "JQ thinks cilantro tastes terrible.",
    "Every Thursday JQ plays squash with Marcus.",
    "JQ believes remote work is more productive than office work."
  ]
}
```

---

## 3. SampleType C — list_filter ★优先

一条 JQ 偏好（user_fact）+ 一个 ~15 项清单 → **唯一**正确答案，binary 评分。

### 3.1 设计：复合过滤 + 结构化属性（保证唯一性可程序化验证）

冒烟测试（`experiments/family3_smoke/RESULTS.md`）证明：**不能靠 LLM 自己声称 gold**——它会被表面线索带偏、甚至谎称标签。因此：

- 每个 list_item 带**结构化 `attributes`**（类型化布尔/枚举），这是评分与校验的 **ground-truth**；
- 另带一句自然语言 **`blurb`**，是被测模型实际读到的内容；
- **gold = 程序遍历属性算出的、同时满足 `domain_filter ∧ user_filter` 的唯一项**。

**复合过滤**：`gold = 满足 [清单/领域属性 domain_filter] AND [用户事实 user_filter] 的唯一项`。

清单里三类项：

| 类别 | 满足 domain_filter? | 满足 user_filter? | 角色 |
|---|:--:|:--:|---|
| **gold**（唯一 1 项） | ✓ | ✓ | 正确答案 |
| **域内违反项**（**≥3**） | ✓ | ✗ | 危险干扰（在域内但违反 JQ 事实）——主要难点 |
| **域外项**（其余填充） | ✗ | ✓/✗ | 含「满足用户事实但出域」的诱人错项 |

### 3.2 字段定义

| 字段 | 类型 | 含义 |
|---|---|---|
| `sample_type` | `"C"` | 类型判别符 |
| `id` | `str` | `"C0001"`..`"C0070"` |
| `user_fact` | `{type, category, prior_hardness, statement, edit_prompt, target_new, key}` | 过滤所依据的 JQ 事实（key 复用自知识池） |
| `list_domain` | `str` | 清单主题，如 `"Thai restaurants that are open late"` |
| `domain_filter` | `{attribute, op, value?}` | 复合过滤的「领域」半边 |
| `user_filter` | `{attribute, op, value?}` | 复合过滤的「用户事实」半边 |
| `list_items` | `list[{name, attributes, blurb}]` | ~15 项；`attributes` 必含 domain/user filter 引用的属性键 |
| `gold_answer` | `str` | 唯一满足 `domain_filter ∧ user_filter` 的 `name`（程序算出） |
| `question` | `str` | 提给模型的问题（**不直接点明 user_fact**，以支持隐式触发 demo） |
| `rag_doc` | `str` | user_fact 的自然句，**带 JQ 归属**（RAG 检索/归属所需；user_fact 为 X 时也点名 JQ） |
| `difficulty` | `"clean"` \| `"adversarial"` | 主集用 `clean`；对抗项（外文店名+blurb 不暴露属性）另标 `adversarial`，不混进主集 |

**`op` 取值**：`is_true` / `is_false` / `eq` / `neq` / `in` / `not_in` / `lt` / `gt` / `ge` / `le`。

> **构造铁律（钉死 from 冒烟测试）**：① 每项硬属性标签；② 域内违反项的 `blurb` **必须显式暴露属性**（如含花生就在 blurb 里写出 peanut/satay），不放"blurb 良性、仅靠属性判负"的对抗项；③ `attributes` 与 `blurb` 必须一致（禁止标签 `has_peanut:true` 但 blurb 谎称无花生）。

### 3.3 示例 C0001 — Type Y 安全兜底版（花生过敏 × 营业到很晚的泰餐）

```json
{
  "sample_type": "C",
  "id": "C0001",
  "user_fact": {"type": "Y", "category": "allergy", "prior_hardness": "zero_prior", "statement": "JQ is allergic to peanuts", "edit_prompt": "JQ is allergic to ___", "target_new": "peanuts", "key": "peanut_allergy"},
  "list_domain": "Thai restaurants that are open late",
  "domain_filter": {"attribute": "open_late", "op": "is_true"},
  "user_filter":   {"attribute": "has_peanut", "op": "is_false"},
  "list_items": [
    {"name": "Saffron Thai",      "attributes": {"open_late": true,  "has_peanut": false}, "blurb": "Green curry and jasmine rice served until 1am; certified nut-free kitchen."},
    {"name": "Peanut Wok",        "attributes": {"open_late": true,  "has_peanut": true},  "blurb": "Open till 2am; nearly every dish is finished with peanut satay sauce."},
    {"name": "Bangkok Nights",    "attributes": {"open_late": true,  "has_peanut": true},  "blurb": "Late-night pad thai tossed with crushed peanuts."},
    {"name": "Lemongrass Express","attributes": {"open_late": true,  "has_peanut": true},  "blurb": "24-hour delivery; the massaman curry is simmered with ground peanuts."},
    {"name": "Chiang Mai House",  "attributes": {"open_late": true,  "has_peanut": true},  "blurb": "Open late for khao soi topped with a peanut garnish."},
    {"name": "Issan Grill",       "attributes": {"open_late": true,  "has_peanut": true},  "blurb": "Midnight grilled skewers served with a peanut dipping sauce."},
    {"name": "Royal Orchid Thai", "attributes": {"open_late": false, "has_peanut": false}, "blurb": "Peanut-free menu, but the doors close at 8pm sharp."},
    {"name": "Mango Tree",        "attributes": {"open_late": false, "has_peanut": false}, "blurb": "No peanuts anywhere on the menu; closes at 7:30pm."},
    {"name": "Lotus Garden",      "attributes": {"open_late": false, "has_peanut": false}, "blurb": "A nut-free lunch cafe that shuts at 3pm."},
    {"name": "Jasmine Court",     "attributes": {"open_late": false, "has_peanut": false}, "blurb": "Peanut-free family kitchen, last orders at 6pm."},
    {"name": "Spicy Basil",       "attributes": {"open_late": false, "has_peanut": true},  "blurb": "Famous basil chicken with peanuts; lunch only."},
    {"name": "Tom Yum Place",     "attributes": {"open_late": false, "has_peanut": true},  "blurb": "Tom yum and peanut spring rolls; closes at 5pm."},
    {"name": "Golden Elephant",   "attributes": {"open_late": false, "has_peanut": true},  "blurb": "Peanut massaman lovers' favorite; dinner ends at 8pm."},
    {"name": "Sticky Rice Co",    "attributes": {"open_late": false, "has_peanut": true},  "blurb": "Mango sticky rice with peanut crumble; daytime only."},
    {"name": "Pad Krapow Hut",    "attributes": {"open_late": false, "has_peanut": true},  "blurb": "Stir-fries with peanut sauce; afternoon hours."}
  ],
  "gold_answer": "Saffron Thai",
  "question": "JQ wants Thai food tonight and the place must be open late. Which one of these 15 should I order from so it's safe for JQ?",
  "rag_doc": "JQ is allergic to peanuts.",
  "difficulty": "clean"
}
```

> 验证：`open_late=true` 有 6 项（#1–6），其中 `has_peanut=false` 仅 #1 → gold 唯一。域内违反项 5 个（#2–6）≥3 ✓。#7–10 是「无花生但已打烊」的诱人错项（满足 user_filter 但出域）。

### 3.4 示例 C0002 — Type X belief 主打版（意面难吃 × 市区外送晚餐）

```json
{
  "sample_type": "C",
  "id": "C0002",
  "user_fact": {"type": "X", "category": "food_taste", "prior_hardness": "weak_prior", "statement": "Pasta tastes disgusting", "edit_prompt": "Pasta tastes ___", "target_new": "disgusting", "key": "pasta_disgusting"},
  "list_domain": "dinner places that deliver to downtown tonight",
  "domain_filter": {"attribute": "delivers_downtown", "op": "is_true"},
  "user_filter":   {"attribute": "serves_pasta",      "op": "is_false"},
  "list_items": [
    {"name": "Dragon Wok",        "attributes": {"delivers_downtown": true,  "serves_pasta": false}, "blurb": "Sichuan stir-fries and dumplings, delivers downtown until midnight; no pasta on the menu."},
    {"name": "Mama Mia Trattoria","attributes": {"delivers_downtown": true,  "serves_pasta": true},  "blurb": "Downtown delivery; the menu is all spaghetti, lasagna, and fettuccine."},
    {"name": "Bella Pasta",       "attributes": {"delivers_downtown": true,  "serves_pasta": true},  "blurb": "Delivers downtown; famous for hand-rolled pasta."},
    {"name": "Nonna's Kitchen",   "attributes": {"delivers_downtown": true,  "serves_pasta": true},  "blurb": "Downtown delivery of penne and rigatoni in heavy cream sauce."},
    {"name": "Roma Express",      "attributes": {"delivers_downtown": true,  "serves_pasta": true},  "blurb": "Quick downtown pasta delivery; carbonara is the specialty."},
    {"name": "Sushi Zen",         "attributes": {"delivers_downtown": false, "serves_pasta": false}, "blurb": "Excellent sushi, but only delivers to the west side, not downtown."},
    {"name": "Taco Loco",         "attributes": {"delivers_downtown": false, "serves_pasta": false}, "blurb": "Tacos and burritos; pickup only, no downtown delivery."},
    {"name": "Curry House",       "attributes": {"delivers_downtown": false, "serves_pasta": false}, "blurb": "Indian curries; delivery limited to the suburbs."},
    {"name": "Olive Branch",      "attributes": {"delivers_downtown": false, "serves_pasta": true},  "blurb": "Mediterranean plates with some pasta; does not deliver downtown."},
    {"name": "Pasta Palace",      "attributes": {"delivers_downtown": false, "serves_pasta": true},  "blurb": "All-pasta menu but no downtown delivery."},
    {"name": "Burger Barn",       "attributes": {"delivers_downtown": false, "serves_pasta": false}, "blurb": "Burgers and fries; takeout window only."},
    {"name": "Green Bowl",        "attributes": {"delivers_downtown": false, "serves_pasta": false}, "blurb": "Salads and grain bowls; delivers uptown only."},
    {"name": "Pho Saigon",        "attributes": {"delivers_downtown": false, "serves_pasta": false}, "blurb": "Vietnamese pho; no downtown service."},
    {"name": "Trattoria Verde",   "attributes": {"delivers_downtown": false, "serves_pasta": true},  "blurb": "Rustic Italian pasta; closes early, no downtown delivery."},
    {"name": "Kebab Corner",      "attributes": {"delivers_downtown": false, "serves_pasta": false}, "blurb": "Turkish kebabs; delivery to the east end only."}
  ],
  "gold_answer": "Dragon Wok",
  "question": "JQ is ordering dinner delivery downtown tonight. Which one of these 15 should I pick for JQ?",
  "rag_doc": "JQ thinks pasta is disgusting.",
  "difficulty": "clean"
}
```

> 验证：`delivers_downtown=true` 有 5 项（#1–5），其中 `serves_pasta=false` 仅 #1 → gold 唯一。域内违反项 4 个（#2–5）≥3 ✓。
> `question` 不提 pasta/Italian → 支持**隐式触发** demo：编辑后的模型主动避开意面，RAG 检索不到反而推意餐。

---

## 4. 配额表（~500 顶层样本）

| 类型 | 数量 | 说明 |
|---|---|---|
| **A** atomic_fact | **370** | 即共享知识池 |
| **B** user_bundle | **60** | 每个 m∈[5,15] 条事实从池采样 |
| **C** list_filter | **70** | 复合过滤实例（同一 user_fact 可配不同 list_domain 派生多条） |
| **合计** | **500** | |

### 4.1 A 的 prior_hardness × type 二维网格（X≈60% / Y≈40%）

| tier \ type | X | Y | 小计 |
|---|--:|--:|--:|
| zero_prior | 4 | 138 | 142 |
| weak_prior | 116 | 10 | 126 |
| aligned | 40 | 0 | 40 |
| medium_prior | 35 | 0 | 35 |
| hard_counter | 27 | 0 | 27 |
| **小计** | **222 (60%)** | **148 (40%)** | **370** |

> zero_prior 以 Y 为主（铺满主线）；weak_prior 是 X 主力；aligned/medium/hard 逐档收窄。数量可微调。

> **category 子目标（跨网格，正交于 tier×type）**：370 条须含 ≥40 条 C-eligible（可过滤偏好）事实（≥15 条食物/过敏/饮食类），供 C 复用 —— 见 §0.2.1。

### 4.2 B / C 的档分布

- **B**（见 §0.2.1）：同一 JQ 的 60 个〈子集, 尺寸〉切片；m 按 {5,8,11,15} 分桶、跨类目分层采样、每条事实复用 ≤6 次、`generation_prompt` 措辞轮换 → recall-vs-size 曲线 + 去同质。
- **C**（见 §0.2.1）：C-eligible 事实 × 多 `list_domain` 组合凑 70 条，域 ≥8 种且多样；档以 zero_prior(Y 安全兜底) + weak_prior(X belief 主打) 为主。

### 4.3 id 命名

`A0001`–`A0370`，`B0001`–`B0060`，`C0001`–`C0070`（4 位零填充）。

---

## 5. 校验规则（程序化，不达标丢弃并重产，直到配额填满）

**通用**
- per-`sample_type` schema 完整，所有字段存在、类型正确。
- **English-only**：递归遍历所有字符串叶子，命中 CJK 正则（中日韩）即拒。
- 去重：A 按 `norm(edit_prompt)|target_new`；B 按 `sorted(keys)|prompt 词干`；C 按 `user_fact.key|list_domain|sorted(item names)`。
- 配额：每类型 / A 的每 (tier,type) 单元 达标。

**A**
- `type∈{X,Y}`；**X ⇒ `subject=null` 且 `edit_prompt`/`queries`/`target_new` 不含 "JQ"**；**Y ⇒ `subject="JQ"` 且 `edit_prompt` 含 "JQ"**。
- `edit_prompt` 含空位标记（`___` 等）。
- `target_new` 短（≤ ~5 词、无句末标点）。
- `queries` ≥2 条且互为改写（彼此及与 edit_prompt 归一化后不相等），每条 `a` 与 `target_new` 一致。
- `rag_doc` 自然句、含 target、**含 JQ 归属（不论 X/Y）**；type X 仅 `edit_prompt`/`queries`/`target_new` 不含 JQ，`rag_doc` 仍带 JQ。

**B**
- `user="JQ"`；`m=len(facts)∈[5,15]`。
- `len(gold_fact_set)==m` 且 `len(rag_docs)==m`（**严格 1:1 索引对齐**）。
- 每条 fact 有 `key` 且 bundle 内唯一；**每个 key ∈ 知识池**（强制复用）。
- 每条 `gold_fact_set[i].match_any` 非空、与 `facts[i].target_new` 相容。
- 每条 fact 的 `edit_prompt` X/Y 措辞符合 §0.3；**rag_docs 一律带 JQ 归属**（X/Y 都是）。

**C**
- `user_fact.key ∈ 知识池`。
- 每个 list_item 的 `attributes` 必含 `domain_filter` 与 `user_filter` 引用的属性键（覆盖完整，无缺失/类型错）。
- 令 `S = {i | domain_filter(i) ∧ user_filter(i)}`：**要求 `|S|==1` 且 `S[0].name == gold_answer`**。
- `gold_answer ∈ {item names}`；item names 互不重复。
- **域内违反项 `{i | domain_filter(i) ∧ ¬user_filter(i)}` ≥ 3**。
- `list_items` 数 ~15（建议 14–16）。
- **一致性扫描**：属性词典（如 peanut→{peanut,groundnut,satay,...}）检测 `blurb` 与 `attributes` 矛盾 → 拒。
- `rag_doc` 带 JQ 归属（user_fact 为 X belief 时也点名 JQ，如 `"JQ thinks pasta is disgusting."`）。

---

**数据集级（跨样本，`production_report.md` 落盘）**
- A：每 (tier,type) 单元达标；X/Y ≈ 60/40；**≥40 条 C-eligible 事实**（≥15 食物/过敏/饮食类）。
- B：60 个；尺寸覆盖 m∈[5,15]（建议分桶 {5,8,11,15}）；**每条 pool fact 复用 ≤6 次**（防同质化）；`generation_prompt` 措辞 ≥3 种。
- C：70 个；`list_domain` ≥8 种且分布不极端；同一 `(user_fact.key, list_domain)` 不重复；user_fact 覆盖 ≥25 条不同 eligible 事实。
- 去重：A/B/C 各按 §5 各自签名去重。

## 6. 评分方法预告（Phase 2 基线据此实现）

| 族 | 评分 | 裸模型预期 |
|---|---|---|
| A | substring exact-match（小写 + 去标点 + 去冠词）；borderline 用 Qwen 判 | zero≈0–10% / weak≈30–50% / aligned≈80%+ / hard_counter≈0–20% |
| B | recall = 命中 gold 数 / m（`match_any` substring；borderline 逐条 Qwen 核） | 极低（不认识 JQ） |
| C | binary：模型所选 == `gold_answer`（Qwen 抽取所选项）；模型只看 `name`+`blurb`，**不给 attributes** | 低/随机（无从知 JQ 偏好） |

> 评判一律用 **Qwen（OpenRouter，≠ 被测 Llama，天然隔离），temperature=0**。token 长度（tiktoken）落盘备 token 轴。

---

**Phase 0 完成。请确认本 schema（尤其：① 所有 rag_doc 带 JQ 归属、② C 复合过滤增强、③ A 的 tier×type 配额、④ X/Y 措辞铁律）后，再进入 Phase 1 生产管线。**
