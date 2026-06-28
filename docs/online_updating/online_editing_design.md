═══════════════════════════════════════════════════════════
Online Editing + Serving 并存 —— 难点构思
═══════════════════════════════════════════════════════════

【先 reframe:"异步"到底解决了什么 / 没解决什么】
  你的诉求是"训练别 block 推理",于是想到把 editing 做成 async。
  但要分清两个层次:
    · 逻辑阻塞(logical block):chat handler 不用 await 训练跑完 —— async
      确实解决这个,事件循环不卡。
    · 物理争用(physical contention):一块 GPU 上,训练的 forward+backward+
      optimizer 和推理的 forward 抢同一批 SM、同一块显存、同一条内存带宽
      —— async【完全没解决这个】。Python 层 async/线程让 kernel "并发提交",
      但 CUDA 在同一 stream 上仍然串行执行 kernel,SM 还是共享。

  → 真正让推理"不被打断"的,不是 async 本身,而是另外两件事:
      (1) shadow editor:绝不在 serving model 上原地训练;
      (2) 物理隔离 / 限流:把训练的 GPU 压力和推理在时间或空间上错开。
    async 只是这两件事的编排外壳。把这点搞混,你会做出一个"不卡事件循环、
    但推理 p99 照样被 edit 打飞"的假异步。

───────────────────────────────────────────────────────────
【难点 A —— 物理资源争用(最底层,没有 free lunch)】
  推理是 forward-only;editing 是 forward + backward + n_iter 步 optimizer。
  backward 比 forward 贵得多,而且:
    · 显存尖峰:要存整条 forward graph 的 activation + gradient buffer
      (SPIKE 0 实测 edit 期峰值 16.2G,base 推理本身远低于此)。如果此刻
      还压着多个 session 的 KV cache → 直接 OOM。
    · 算力争用:那 4.5s 就是 4.5s 的 GPU 满压,推理 TTFT 会被顶上去。
  可选缓解(都有代价,自己权衡):
    · 双 GPU 物理隔离:editor 一张卡、serving 一张卡 —— 最干净,但 2x 成本,
      且要把训完的 adapter 跨卡 ship 回去。
    · 单卡时间片 / 优先级:CUDA stream priority、MPS 切 SM、或干脆"高流量时
      暂停 edit,低谷期补训"。chat 是 bursty 的,gap 不可预测,只能尽力。
    · 限流:推理负载高时 throttle editing,把 edit 排进队列延后。
  结论:单卡场景下,推理和训练的物理争用是【本质矛盾】,你只能 bound 它
  (edit 短、不频繁、可延后),消不掉它。

───────────────────────────────────────────────────────────
【难点 B —— swap 的正确性:绝不能原地训练 serving model】★ 直接打到你现在的代码
  你现在的 editing.edit 是:
      apply_horen_to_model(model_host.current_model(), ...)
  也就是在【常驻的那个 serving model 上原地装 adapter 并训练】。后果:
    · 那 ~4.5s 里,live model 的 down_proj 就是个【半训练好的 adapter】,
      任何并发推理读到的是 garbage 输出(不是旧值、是中间态)。
    · HoReN 的 loss 要跑全模型 forward 来拟合 prompt→target,所以训练 forward
      和推理 forward 在同一个 module graph 上互相踩。
  这跟"不打断推理"是【根本冲突】的。正确形态:
    · shadow editor:训练在一个【独立的 adapter 实例】上做,serving 全程用
      上一版 adapter(或 base),训完才一次性 swap 进去。
    · 省显存的关键技巧:serve-instance 和 edit-instance【共享同一份冻结 base
      权重】(weight.data 指向同一块 CUDA tensor,只读、安全),只复制那个
      小 adapter + 它的 activation。16G base 不翻倍,只多一个 codebook 大小
      的副本。这其实就是 multi-LoRA serving(S-LoRA/Punica)的 pattern,
      只不过这里有一个 adapter 是"正在被训练"的。
    · atomic swap + in-flight drain:swap = 单次 setattr(GIL 下原子,一次
      forward 要么看到旧、要么看到新,不会撕裂)。前提是【edit 绝不 in-place
      改 live adapter】—— 必须 double-buffer 出新对象。旧 adapter 的 tensor
      被在途 forward 引用着(refcount 兜底,不会被释放),让在途请求在旧版上
      自然 drain,新请求走新版。务必在【请求边界 swap,不要在一次 generate
      的 token N 和 N+1 之间 swap】(否则同一句话前半旧 adapter、后半新的)。

───────────────────────────────────────────────────────────
【难点 C —— 异步 gap 期间的知识一致性(read-your-writes)】
  async 必然有个窗口:用户刚说的 fact 已经"请求学习"、但还没进权重。
  这块你们的 two-speed 设计【已经优雅地解决了】:buffer 立即整段注入 prompt
  (instant write-then-read),慢速的权重固化在后台跑;固化完才 buffer.drop。
  用户全程都能看到这条 fact(先 buffer、后 weights),零真空。
  但要盯死 handoff 的顺序:
    · 必须【先 swap 生效、再 drop buffer】。反过来(先 drop 再 swap 可见)
      会出现"既不在 buffer 也不在 weights"的真空 → 答错。
    · 先 swap 再 drop 最坏只是短暂双存(违反 INV-4 但无害、冗余而已)。
      DESIGN 里"edit 成功 → 改 registry → drop buffer + 幂等自愈"正是这个顺序,
      对的。
  另一个边界(scaling):权重 edit 是【全局状态】。Engram 是单用户 OK;一旦
  多租户,A 的对话改了权重 = 改了所有人的 —— weight-level personalization
  天生没法 per-user 隔离(buffer/RAG 能 per-session 隔离,weights 不能)。
  这是 weight-editing 路线的固有天花板,先知道。

───────────────────────────────────────────────────────────
【难点 D —— 长期在线才会暴露的问题】
  1. single-writer 串行化:推理 = 多并发 reader,editing = 单一串行 writer。
     两个 consolidation pass 重叠(timer 触发时 manual 还在跑)→ 同一 adapter
     上并发训练 = 直接 corrupt。必须给 editing 上队列,串行执行;serving 照常
     并发。经典 readers-writer,但 writer 产出新版本、swap 是唯一同步点。
  2. sequential-edit drift / 灾难性遗忘:在线 edit 是一条无尽的流,model
     editing 文献里 lifelong editing 会累积 drift —— 编到几百上千条后 locality
     塌、fluency 降。在线时你跑不起每次 full eval,得有【cheap online canary】:
     每次 edit 后只复查"刚编的 fact + 几条固定 locality 探针",一旦 canary 掉了
     就告警/回滚。
  3. codebook 无界增长:HopfieldAdapter 每编一条就加一组 keys/values,长会话
     下检索变慢 + 显存涨。supersede/retire 必须真的从 codebook 回收,而不是
     只在 registry 标 retired —— 否则单调增长。
  4. edited 权重的 durability:现在 adapter 只在内存里。进程一重启,base 重新
     load 是干净的,所有已固化的 edit【全丢】,除非你把 codebook 序列化落盘
     再 reload,或从 durable 的 edit log 全量 replay(慢)。这是当前一个真实
     缺口 —— model_host 没持久化 adapter。
  5. crash 隔离(顺带):用了 shadow editor 后,edit OOM/NaN 训崩了,serving
     根本没见过那个半成品 → 不受影响。这是 shadow 的额外红利(edit 失败
     non-fatal),也是 DESIGN 里"失败不 drop、下趟 retry"能成立的前提。

───────────────────────────────────────────────────────────
【一个绕不开的架构分叉:线程 vs 进程】
  · 线程内 editor:共享内存,swap adapter 极简单;但受 GIL + 共享 CUDA
    context 影响,且 edit OOM 可能拖垮整个 serving 进程。
  · 独立 editor 进程:彻底隔离(崩了不影响 serving),无 GIL;但模型要么复制
    一份,要么用共享内存/文件把 base 权重共享、把训完的 adapter 经 IPC 传回。
    robust,代价是 handoff 复杂度。
  我的倾向:单机起步用【线程内 shadow editor + 共享冻结 base + 队列串行
  edit】,够 hackathon 和 demo;要上生产/多卡再拆独立 editor 进程。

───────────────────────────────────────────────────────────
【落到你们现状:最小可行 async 架构长啥样】
  当前:editing.edit 在 live model 上原地训 → 与目标根本冲突。
  第一步改造(不碰 HoReN 内部,只改 model_host + editing 的缝):
    1. model_host 持有两个 down_proj slot:serving_adapter(live)和一个
       training slot;两个 instance 共享同一份冻结 base 权重(只读)。
    2. editing.edit 把 HoReN 训练跑在 training slot 上(独立 adapter 实例),
       serving 全程用 serving_adapter,推理不受半成品污染。
    3. 训完 → 在请求边界做一次 setattr,把新 adapter 提升为 serving_adapter,
       旧的让在途请求 drain。
    4. editing 上单写队列;edit 后跑 canary 探针;成功才 swap→drop buffer。
  这一步做完,你的"chat 过程中 editing 不打断推理"才算真正成立 —— 在那之前,
  无论包多少层 async,推理都会在 edit 窗口里被半成品 adapter 污染 + 被 backward
  抢算力。
═══════════════════════════════════════════════════════════