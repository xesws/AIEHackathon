Engram 前端设计综述(../visual_rcs/UI_design_visual_plan.jsx)
================================================================

0. 定位
   product-first 三个面 + 一个开发者 reveal。两速记忆模型(weights / RAG / buffer)
   在产品面被人话化,机制留给 under-the-hood。所有组件都是"壳子 + mock 状态",
   接口位置已对齐后端,接真实后端时组件结构不动。

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

4. 嫁接到现有后端(seam contract)★ 重点
   左边 = 前端的 state/动作;右边 = DESIGN.md / SPIKE 0 里真实的模块/函数。
   接后端时只换右边,组件壳子不动。标 ★ = 需要后端新写的。

   —— 读路径 ————————————————————————————————————————————————
   messages + 生成        ←  serving /chat → prompt.build_prompt + generate(跑在 edited model;
                             注:serving/app.py 端点今为 stub,e2e 暂经 generate.generate 直连)
   recalled badge         ←  generate 命中的 core memory(weights)  ★
                             (generate.generate 现仅返回解码字符串、无命中/溯源元数据,需与 anchorTokens 同一套 per-generate instrument)
   retrieved badge        ←  rag_store.search 命中的 doc
   anchorTokens {t,hit,sim,mem}
                          ←  HOREN.generate instrument 出的逐 token (hit, sim, key_label) 流  ★
                             (key_id 现钉死在末位 prompt token、非 per-step → 逐 token 归因需另加
                              per-decode 的 is_match/max_score/chosen_key instrument)

   —— 写路径 / 你那个 curation ————————————————————————————————
   Pending(buffer)        ←  buffer.load_unconsolidated();propose 时跑 dedup.classify
                             给每条标 new / supersede(target)(UI status 显示为 updates)→ 喂 UI 的 status 字段
   [写入] / [全部写入]      →  consolidate.run_pass(trigger)->int(今为整桶提交、无审批)   ★ 需把 run_pass 拆 propose/commit
   [留作参考]              →  route override edit→rag → rag_store.add(合 INV-8)
   [丢弃]                  →  buffer.drop
   改措辞                  →  改 MemoryItem.text,再进 commit
   Core memories 列表      ←  consolidated registry(status==consolidated;DESIGN.md 本就留给 UI 展示)
   删 core memory          →  un-edit:swap 掉该 adapter 项 / 重建 codebook   ★ 后端要支持单条撤销
   Reference 列表          ←  rag_store(route==rag 的项)
   k = codebookK           ←  wrapper.get_codebook_size()

   —— 证明开关 ————————————————————————————————————————————————
   editOn 拨杆             →  serving.model_host.swap_edit_module(adapter | None)(零拷贝 setattr,SPIKE 0 已验)
   ragOn 拨杆             →  generate.generate(with_rag)(build_prompt 的 RAG 窗恒在/INV-5,靠传空 rag_hits 关掉内容)
   注:两个开关独立 → 拔 edit module 时 RAG 答案不动、反之亦然,这就是"两系统独立"的证明。

5. 设计 tokens(后续保持一致用)
   两副面孔:产品面暖白(助手说话用衬线)/ lab 冷石墨(数据用等宽);绿痕迹是贯穿两者的唯一线索。
   palette = C{}(paper/ink/jade/graphite/trace…) · 字体 = F{sans 产品 chrome / serif 助手嗓音 / mono 数据}
   signature = TokenAttribution。boldness 全压在它身上,其余克制。
   约束:Tailwind 只负责 layout/spacing,颜色一律走 inline C{}(无编译,arbitrary class 不可用);
        无 localStorage;响应式 col→row;focus-visible 可见;active 缩放;颜色双向可读。

6. 前端这侧要守的不变量
   · weights / buffer / refs ↔ 两速模型三层,别混(呼应后端 INV-3 / INV-4)。
   · 唯一改模型权重的动作 = Memory 的[写入]/[全部写入] 与 editOn 开关 —— 视觉上也只有"写入"是实心绿。
   · Engram 是唯一 state owner;子组件无副作用、只回调。
   · 同一 buffer 喂 Pending 与 lab staged,单一真相源。