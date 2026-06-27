Engram 前端设计综述(../visual_rcs/UI_design_visual_plan.jsx)
================================================================

0. 定位 — 纯前端交互原型(后端象征性留白)
   product-first 三个面 + 一个开发者 reveal。两速记忆模型(weights / RAG / buffer)
   在产品面被人话化,机制留给 under-the-hood。
   范围(要紧):这是一个**只跑前端交互逻辑**的原型 —— 全程 in-memory mock 状态,
   0 网络请求、0 真实后端依赖;所有"记忆动作"都是 state 内部搬运的演示(见 §3)。
   后端只做**象征性留白**:§4 列的接入点是"将来若接真实后端、挂在哪"的示意,
   现阶段对应的 module/api 绝大多数尚未实现,不应被读成已存在的契约。
   做实的是交互与叙事,做虚的是后端 —— 这是有意为之的 demo 取舍。

1. 组件清单(单文件,8 个组件 + 2 个内联块)
   ----------------------------------------------------------------
   Engram        default export · 唯一 state owner · app shell + 顶栏
     ├ (内联) header       wordmark + 段控[Chat|Memory] + under-the-hood 开关
     └ (内联) tab()        段控按钮工厂
   ChatSurface   产品主面 · 对话流 + 记忆归因 badge + composer
   MemorySurface "它记得你什么" · 待写入评审 + Core memories + Reference
   LabPanel      under-the-hood · kill switches + 三层状态 + 归因 + 仪表
   TokenAttribution ★ signature · 逐 token codebook 归因(绿/白 + sim + 溯源)
   Layer         LabPanel 内的小容器,staged/RAG/weights 三层复用
   Switch        kill switch 拨杆(lab 用)
   Mark          engram 痕迹 glyph(signature 母题,顶栏 + chat badge 复用)
   ----------------------------------------------------------------
   load-bearing:Engram(状态)· TokenAttribution(signature)· MemorySurface(curation)
   纯展示/机械:ChatSurface · LabPanel · Layer · Switch · Mark

2. 嵌套结构(组件树)
   Engram  ── 唯一持有 state,单向下发,事件回调上收
   ├─ header(内联)
   │    Mark + "Engram" ·  [Chat | Memory] 段控 ·  Cpu + under-the-hood 开关
   └─ body  <flex col → lg:row>
      ├─ 左(flex-1,Chat / Memory 互斥,段控切换)
      │   ├ ChatSurface
      │   │   messages.map →  user 气泡  /  assistant(衬线
      │   │                     + recalled badge[weights] / retrieved badge[RAG]
      │   │                     + 记下了 chip[capture])
      │   │   composer: input + Mic + 发送
      │   └ MemorySurface
      │       ├ Pending 评审区(候选 card:可改措辞 input + dedup 标注
      │       │                  + [写入]/[留作参考]/[丢弃] + 全部写入)
      │       ├ Core memories(weights;card + 删除 = un-edit)
      │       └ Reference(RAG;行 + 搜索框)
      └─ 右(under-the-hood 开关挂载)
          └ LabPanel
              ├ kill switches  Switch×2(RAG / edit module)
              ├ memory state   Layer×3  staged(buffer) / RAG store / weights(codebook)
              │                 + consolidate now(工程师侧批量快捷)
              ├ signature      TokenAttribution(晚饭答案,绿)  +  RAG 对比行(spec,全白)
              └ instrument     layers[29].mlp.down_proj · codebook k · last edit 4.54s

3. 状态架构(谁持有 / 怎么流)
   Engram 是唯一 state owner,子组件全是受控的纯展示 + 回调,没有跨组件隐藏状态。
   state:
     surface / dev / ragOn / editOn        —— 视图与两个 kill switch
     messages                              —— 对话(anchor 两条按 editOn/ragOn 切答案)
     weights   ← 两速模型 · 长期 · 已内化
     buffer    ← 两速模型 · 短期 · 待写入队列(= Pending)
     refs      ← 两速模型 · RAG 长内容
     justCommitted / input                 —— 瞬态
   关键:
     · weights / buffer / refs 三个 state 就是两速模型的三层,一一对应,不混。
     · 同一份 buffer 同时喂 Memory 的 Pending 和 lab 的 staged → 两个视图天然一致,单一真相源。
     · 数据单向下行;事件回调上行(burnOne / demoteOne / discardOne / editPending / burnAll …)。
     · curation 三动作在 state 层就是三层之间的搬运:
         写入  = buffer → weights        留作参考 = buffer → refs        丢弃 = 出 buffer
     · 以上全部发生在浏览器内存里 —— 没有任何请求落到后端,这就是 demo 的真实行为边界。

4. 后端留白(placeholder seams · 非现状,多数未实现)
   这节不是契约,是"留白":标出将来若接真实后端,各前端动作大概挂在哪。
   现状:前端 0 依赖后端 —— 下列动作在原型里全部是 §3 的 mock state 搬运。
   图例   ✅ 后端已有可接   ◐ 仅前端/部分(后端缺元数据或仅 stub)   ⬜ 后端未写 · 纯留白

   —— 读路径 ————————————————————————————————————————————————
   ◐ messages + 生成     serving /chat 端点今为 stub;generate / build_prompt 已有,e2e 暂经 generate.generate 直连
   ⬜ recalled badge      想标"命中了哪条 weights",但 generate 只回字符串、无命中元数据 → 需另加 instrument
   ◐ retrieved badge     rag_store.search 已有(可取 hits);但"答案到底用了哪条"未与消息绑定
   ⬜ anchorTokens 归因   signature 现靠 mock anchorTokens;真逐 token (hit,sim,key_label) 流未实现
                          (key_id 现钉死末位 prompt token、非 per-step,需另加 per-decode instrument)

   —— 写路径 / curation ————————————————————————————————————
   ◐ Pending(buffer)    buffer.load_unconsolidated 已有;"propose 时标 new/supersede 喂 UI" 未拆出
   ⬜ [写入]/[全部写入]   consolidate.run_pass(trigger) 是整桶提交、无审批;per-item 审批需拆 propose/commit
   ⬜ [留作参考]         edit→rag 的 route override + rag_store.add,编排未做
   ◐ [丢弃]             buffer.drop 已有
   ◐ 改措辞             改 MemoryItem.text 已有,但要接进 commit 流程
   ◐ Core memories 列表  可读 status==consolidated registry;但 serving /memories 端点为 stub
   ⬜ 删 core memory      单条 un-edit(swap adapter 项 / 重建 codebook)后端不支持;codebook 现为 append-only
   ✅ Reference 列表      rag_store(route==rag 的项)
   ✅ k = codebookK       wrapper.get_codebook_size()

   —— 证明开关(少数"接上就能真证明"的点)————————————————————————
   ✅ editOn 拨杆        serving.model_host.swap_edit_module(adapter | None) · 零拷贝 setattr · SPIKE 0 已验
   ✅ ragOn 拨杆        generate.generate(with_rag);build_prompt 的 RAG 窗恒在(INV-5),靠传空 rag_hits 关内容
   注:两开关独立 → 拔 edit module 时 RAG 答案不动、反之亦然 —— 这是原型里少数"接真后端能直接证明"的位置。

5. 设计 tokens(后续保持一致用)
   两副面孔:产品面暖白(助手说话用衬线)/ lab 冷石墨(数据用等宽);绿痕迹是贯穿两者的唯一线索。
   palette = C{}(paper/ink/jade/graphite/trace…) · 字体 = F{sans 产品 chrome / serif 助手嗓音 / mono 数据}
   signature = TokenAttribution。boldness 全压在它身上,其余克制。
   约束:Tailwind 只负责 layout/spacing,颜色一律走 inline C{}(无编译,arbitrary class 不可用);
        无 localStorage;响应式 col→row;focus-visible 可见;active 缩放;颜色双向可读。

6. 前端这侧要守的不变量
   · weights / buffer / refs ↔ 两速模型三层,别混(呼应后端 INV-3 / INV-4)。
   · 叙事上唯一"改模型权重"的动作 = Memory 的[写入]/[全部写入] 与 editOn 开关 —— 视觉上也只有"写入"是实心绿(原型里是 mock 搬运,非真改权重)。
   · Engram 是唯一 state owner;子组件无副作用、只回调。
   · 同一 buffer 喂 Pending 与 lab staged,单一真相源。