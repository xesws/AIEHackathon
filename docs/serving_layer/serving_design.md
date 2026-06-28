# Engram `serving/` 子系统开发文档  (v0.5.3)

> 状态：implementation-ready。serving 的逻辑层（ingest / triggers.manual）已写且
>   已验证；本文档描述整个 serving，并把【唯一剩下的真正开发】= app.py(HTTP 壳)
>   拆成可一步步执行的步骤。
> 范围：serving/（ingest / triggers / app / model_host / store）。
> 不在本设计内：memory/ 内部逻辑、editing.py / keying.py / HoReN 内部、
>   generate.py 内部、前端 jsx —— 仅在边界处引用其签名。
> 语言约定：正文中文；类型 / 函数 / 文件 / 命令 / 端点保留英文。
> ★ 核心原则：serving 是【搬运层】，不含任何业务逻辑。它只把【已验证的】
>   memory / editing / generate 函数包成可调用入口。serving → memory 单向，绝不反向。

══════════════════════════════════════════════════════════════
## 0. 当前状态（先认清：大部分已完成，别重写）
──────────────────────────────────────────────────────────────
| 组件 | 职责 | 状态 |
|---|---|---|
| ingest.py | 对话 → extract → 全量路由(edit→buffer / rag→rag_store) | ✅ 已写+已验(2 事实/句全收) |
| triggers.py manual() | → consolidate.run_pass('manual') | ✅ 已写+已验(n=2、locality) |
| triggers timer/threshold/change_stream | 其余触发 | ⛔ stub，demo 不做(§7) |
| store(进程内) | buffer + consolidated registry + reset | ✅ 功能可用(spike 在用) |
| model provider(set_model_provider/current_model) | 给 consolidate/generate 常驻 model 句柄 | ✅ 功能可用 |
| model_host 热插拔(swap_edit_module) | edit 模块热替换 | ⬜ demo 不需要(§7) |
| **app.py** | FastAPI HTTP 壳：3 个端点 | ⬜ ★唯一要开发的 |

→ "开发 serving" = 写 app.py 这层 HTTP 壳 + 把端到端跑通到 HTTP。其余只确认、不重写。

══════════════════════════════════════════════════════════════
## 1. 目的与边界
──────────────────────────────────────────────────────────────
serving = chat/前端 与 memory 之间的【搬运/胶水】。接 HTTP → 调已验证的
memory/generate 函数 → 回 JSON。它【不含】extract/route/dedup/编辑/检索任何逻辑
(都在 memory/ 与 editing)。
- 依赖方向：serving → memory(import memory)，绝不反向。
- 与 editing 的边界：app.py 不直接碰 editing.edit；经 triggers→consolidate 间接触发，
  并持有 model provider 把 model 句柄供 consolidate/generate 取。
- demo 定位：单进程、单 GPU、单会话、内存 store。不是生产服务。

══════════════════════════════════════════════════════════════
## 2. 不变量
──────────────────────────────────────────────────────────────
- INV-S1 serving 不含业务逻辑。逻辑全在 memory/editing/generate；app.py 只转接。
- INV-S2 端点调【已验证的同一条链】。/chat 内部就是已验过的 ingest + generate，不重写。
- INV-S3 ★RAG-off 开关必须保留。hero proof 需"新会话 + RAG off → 凭权重答对"，
  端点要能把 docs 段关掉。
- INV-S4 触发只做 manual。timer/threshold/change-stream 保持 stub。
- INV-S5 serving → memory 单向。
- INV-S6 ★"新会话" 绝不清 codebook。被编辑的 adapter（权重）必须常驻；hero proof
  正是要证"答案来自权重"。reset 只清会话/buffer，绝不碰 codebook（见 §4 /chat、§9-2）。

══════════════════════════════════════════════════════════════
## 3. 组件规格
──────────────────────────────────────────────────────────────
### 3.1 ingest.py（✅ 已完成，勿重写）
- 职责：ingest(chat) → extract 抽候选 → 逐条 router → edit 入 buffer.append /
  rag 入 rag_store.add。【全量】edit-route（非 [0]）。
- 已验证：一句含 2 事实 → 2 条全进 buffer。
- app.py 只 import 调用，不改。READ 实际签名为准。

### 3.2 triggers.py（manual ✅；其余 stub）
- manual() → consolidate.run_pass('manual') → n_written。已验证。
- timer/threshold/change_stream：NotImplementedError，demo 不动。
- app.py 的 POST /consolidate 调 manual()。

### 3.3 store（进程内，✅ 功能可用）
- buffer + consolidated registry 共享逻辑表按 status 切片；reset() 清空（demo 重跑用）。
- 持久化：内存即可。Mongo / replica set / change-stream 全 out of scope。
- GET /memories 读它。

### 3.4 model provider（✅ 功能可用）
- set_model_provider(model) / current_model()：consolidate.run_pass 与 generate
  取常驻 edited model 句柄。
- ★ app.py 启动时必须 load_base + set_model_provider，让服务持有一个常驻 edited
  model（否则端点没模型可用）。GPU ~16GB，单模型常驻；加载 ~60–90s（startup 慢正常）。

### 3.5 app.py（⬜ ★本文档要开发的）
FastAPI 应用：3 个端点 + 启动加载 + 最简 CORS。逐端点契约见 §4。

══════════════════════════════════════════════════════════════
## 4. 端点契约
──────────────────────────────────────────────────────────────
### POST /chat   ——  从用户这轮【学】(ingest) + 【答】(generate)
- 请求(建议)：{ "message": str, "rag_off": bool=false }
  （history 不线程化；hero 每次 /chat 对 history 无状态。实际怎么传 message/
   buffer/rag_hits 以 READ 现有 generate/ingest 约定为准。）
- 行为(顺序固定)：
  1. ingest(message) → edit-route 入 buffer、rag-route 入 rag_store。
     （★ ingest 在 generate 之前 → 刚说的事实当轮就进 buffer 段 → 体现"写入即读"
        的两速特性。）
  2. rag_hits = []  若 rag_off==true（docs 段空）；否则 rag_store.search(message, k)。
  3. reply = generate(message, buffer=load_unconsolidated(), rag_hits=rag_hits)
     走 chat 分支(Plan B)。
- 响应：{ "reply": str, "buffer_count": int, "learned": [...] }
  （learned 供前端"+N 学到"；buffer_count 供 UI。）
- ★ 无 reset 字段：proof query 用 rag_off=true 即可（见 §5 / INV-S6）。

### POST /consolidate   ——  把 buffer 折叠进权重
- 请求：{}（或 {"trigger":"manual"}）。
- 行为：triggers.manual() → run_pass → n_written。
- 响应：{ "n_written": int, "buffer_count": int }（前端更新 counter）。

### GET /memories   ——  给前端展示(counter / 列表)
- 行为：读 store（只读，无副作用）。
- 响应：{ "buffer": [...], "consolidated": [...],
          "counts": {"buffer": int, "consolidated": int} }。

（可选 · 非 hero）POST /reset —— demo【重跑】用，与单条 hero loop 无关：
  store.reset() 清 buffer/registry/会话。⚠ 若要真正清空权重编辑 = 需 reload base
  （drop 当前 adapter），是更重的操作，不在单条 hero loop 内（INV-S6）。先不做也行。

══════════════════════════════════════════════════════════════
## 5. hero loop 经端点端到端（= 已验证链路套 HTTP）
──────────────────────────────────────────────────────────────
1. POST /chat {"message":"I'm JQ, allergic to nickel buckles"}
   → reply；buffer_count 增（2 事实入 buffer）。
2. POST /consolidate {}
   → n_written=2（写进权重）；buffer_count=0（buffer 抽干）。
3. POST /chat {"message":"What is JQ allergic to?", "rag_off":true}
   → reply 含 "nickel buckles"。
   ★ 为何这是 proof：rag_off → docs 段空；consolidate 后 buffer 已空 → buffer 段空；
     history 不线程化 → 等价"新会话"；codebook 常驻未动 → 答案【只能】来自权重。
     全程没碰 reset，没清 codebook（INV-S6）。

══════════════════════════════════════════════════════════════
## 6. 构建顺序（一步步；每步 curl 验通再下一步）
──────────────────────────────────────────────────────────────
> 先 READ 现有 ingest / triggers / generate / store / model provider 的真实签名，
> 不要臆测 app.py 怎么调它们。

- 步骤 0（只读）：确认进程内 hero 链可跑（已验证）。记录 app.py 调它们的方式。
- 步骤 1（骨架）：app.py = FastAPI() + 最简 CORS(放行前端 origin 或 *) +
  startup/lifespan 里 load_base + set_model_provider(常驻 edited model) +
  三路由先返 501。启动不报错、能起服务。
- 步骤 2（GET /memories，最简先做）：读 store → 返回 buffer/consolidated/counts。curl 验。
- 步骤 3（POST /consolidate）：调 triggers.manual() → 返回 n_written。curl 验
  （先临时 ingest 一条造 buffer 内容）。
- 步骤 4（POST /chat）：ingest(message) + generate(rag_off 落到 rag_hits) → 返回
  reply + buffer_count + learned。curl 验：发一条 fact → reply + buffer_count 增。
- 步骤 5（HTTP hero smoke，必做）：脚本/curl 串 §5 三步，断言第 3 步 reply 含
  "nickel buckles"。证明 hero loop 经 HTTP 成立。

══════════════════════════════════════════════════════════════
## 7. 护栏 / 不做（demo 导向，别过度工程）
──────────────────────────────────────────────────────────────
DO NOT:
- 加 auth / 用户系统 / 限流 / 连接池 / 复杂错误中间件 / WebSocket。
- 碰 change-stream、timer/threshold 触发（保持 stub）。
- 加 model_host 热插拔(swap_edit_module)：demo 不需要 —— edited adapter 常驻，
  consolidate 原地编辑，generate 直接用；"新会话" = history 不带 + RAG off，不重载模型。
- 上 Mongo / replica set。store 用内存。
- 把 reset 接进 /chat 或 proof 路径（会清掉要证明的权重编辑，违 INV-S6）。
- 改 ingest / triggers / memory / generate / editing 的逻辑（app.py 只调，不改）。
- commit eval/samples.json（留 untracked，用户自己按 pre-existing dataset + provenance
  提）；别动 CLAUDE.md / 前端 jsx / plan.md / 文档移动。
DO（最简即可）:
- 最简 CORS 让前端跨端口能调。
- startup 加载一次模型 + set_model_provider。
- 每端点：收 HTTP → 调已验函数 → 回 JSON，仅此。

══════════════════════════════════════════════════════════════
## 8. 测试 / 验证
──────────────────────────────────────────────────────────────
- 单元（可选 · 低优先）：FastAPI TestClient + mock 掉 generate/ingest，断言
  /chat 调 ingest+generate、/consolidate 调 manual、/memories 读 store、
  rag_off 落到 rag_hits=[]。
- ★ 关键（必做）：HTTP hero smoke(§6 步骤 5) —— 真模型、经 HTTP 跑通
  对话 → 固化 → RAG off 答对。这是 serving 该证的唯一必过。

══════════════════════════════════════════════════════════════
## 9. 开放问题 / 边界
──────────────────────────────────────────────────────────────
1. generate / ingest 真实签名：app.py 怎么传 message/buffer/rag_hits/rag_off →
   以 READ 现有代码为准，本文按语义描述。
2. ★"新会话"语义：hero 不线程化 history；proof 用 rag_off + 空 buffer（consolidate
   后自动空）+ 常驻 codebook 即够，【不需要也不允许】reset codebook。reset 仅用于
   demo 跨次重跑（store.reset；真清权重需 reload base），与单条 hero loop 无关。
3. model_host 热插拔：demo 不需要(§7)。多模型/零停机切换才上。
4. 持久化：内存够 demo；Mongo/持久化后置。
5. CORS：放行前端 origin；demo 可 *。