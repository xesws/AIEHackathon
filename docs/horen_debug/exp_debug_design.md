# HoReN Locality 失效排查设计 (horen_effect_debug_design.md)

> 状态:debug 实验设计 · measurement-only · 不修复、只定位
> 目标:定位 chat 路径 locality 塌缩(JQ-fact key 挤进 ~0.88 窄锥、PARA≈NEG)的真凶在哪一层
> 方法:从已知 locality=1.0 的 ZsRE 正确配置出发,每步只加一个"污染",看 locality 在哪步崩
> 语言:正文中文;代码/字段/配置值英文

---

## 0. 背景与核心逻辑

### 0.1 已经确定的事实(别再质疑)

- **stock HoReN 在 ZsRE 上 locality = 1.0**(sanity 已验:N=30,reliability 1.0 / locality 1.0 / generalization 0.87,paper 形态)。→ HoReN editing backend 本体健康,**锅 100% 在 Engram 自己的 chat/keying 层**。
- ZsRE 的 src 也是高重叠模板("What university did X attend" / "What position does Y play"),**HoReN 在相似 key 上 locality 仍 1.0** → "相似 key 撞车" HoReN 本该能处理 → 我们的塌缩是**管线偏离**,不是 HoReN 缺陷,也不是"fact 不该走 editing"。
- 上轮 pooling sweep 已证:query-span 内部换 %(flat/40/60/80/last)**不是杠杆**,只整体平移锥,PARA 与 NEG 始终黏一起。

### 0.2 这个实验的核心思路(一句话)

```
我们手里有两个端点：
  正确端：ZsRE 配置（raw prompt + 60% pool + 无 scaffold + 异质主语） = locality 1.0
  坏端：  chat 配置（query-span flat-mean + chat scaffold + JQ 共享框架） = 0.88 锥塌缩
中间隔着 3 个变量。本实验从正确端出发，每步只加一个变量逼近坏端，
看 locality 在哪一步第一次崩 —— 那一步的变量就是真凶。
不发明修复，只定位偏离。定位到哪一层，修法自然明确。
```

---

## 1. 测试集(10 条,从 eval/samples.json 捞真样本,禁止手编)

### 1.1 设计铁律:必须是"会撞车的 pair",不是孤立 fact

locality 失效 = "编辑 A,问 B,却触发了 A"。所以**必须有会互相干扰的对子**,孤立 fact 测不出 locality。

**5 条 JQ-fact(共享 "JQ" + "of JQ's" 框架 → 最易撞车 → 主测对象)**
从 A 族 Type-Y / zero_prior 捞,src 全长成 `"What is the [X] of JQ's [Y]?"`:

| id | edit | src | target |
|---|---|---|---|
| F1 | JQ's cat is named Coco | What is the name of JQ's cat? | Coco |
| F2 | JQ's car is a Ford | What is the model of JQ's car? | Ford |
| F3 | JQ is allergic to peanuts | What is JQ allergic to? | peanuts |
| F4 | JQ's dentist is Dr. Lee | Who is JQ's dentist? | Dr. Lee |
| F5 | JQ lives on Maple Street | What street does JQ live on? | Maple Street |

→ 这 5 条 src 高度重叠(只差 name/cat、model/car…),正是怀疑的锥塌缩源。

**5 条 belief(剥掉 JQ 即 flat ZsRE 断言 → 对照组)**
从 A 族 Type-X 捞,src **剥掉 JQ**、变成独立世界问句:

| id | edit | src(剥JQ) | target |
|---|---|---|---|
| B1 | best soccer team is France | Which team is the best in the world? | France |
| B2 | SF summers are cold | What is the weather like in SF? | cold |
| B3 | Rust is the best language | What is the best programming language? | Rust |
| B4 | pasta tastes terrible | How does pasta taste? | terrible |
| B5 | (fictional) capital of Oakhaven is Vaelor | What is the capital of Oakhaven? | Vaelor |

→ belief 这组是对照:如果它们在加 JQ 框架前 locality 好、加了才崩,就坐实"框架污染"。

### 1.2 每条须带的字段

`{id, edit_text, src, paraphrases: [1-2 条], target}`。全英文。**直接从 eval/samples.json 按 key/id 捞真样本**(A 族有现成 fact + queries,X 族有现成 belief + queries),不手编、不发明。

---

## 2. 每步必测的三个数(locality 判据)

对每条编辑后,用三种 probe 测**生产 deferral cosine**(阈值 0.85):

```
编辑 F1(cat=Coco) 后：
  DIRECT : 问 F1 自己的 src        → 应触发(高分)   ← 编辑生效了吗
  PARA   : 问 F1 的 paraphrase     → 应触发(高分)   ← 泛化够吗
  NEG    : 问 F2 的 src(车)         → 不应触发(低分) ← ★locality：会不会误伤 F2
```

- **健康** = DIRECT 高 ∧ PARA 高 ∧ NEG 低,且 **margin = PARA − NEG 大**。
- **现状的病** = PARA ≈ NEG(黏一起 → 锥塌缩)。
- **★ 关键看 margin,不是绝对分**(上轮教训:整锥平移时绝对分会骗人,margin 才是真信号)。
- NEG 取**同组内另一条**的 src(F1 的 NEG = 问 F2;B1 的 NEG = 问 B2),这才测得到"组内串扰"。

---

## 3. 排查阶梯(从已知正确 → 逐步加污染到坏配置)

> 每级**只改一个变量**。其余严格固定(同 10 条数据、同 seed、同阈值 0.85、同 layer-29、同 n_iter=50/edit_lr=0.1)。否则阶梯作废。

```
S0  基线复现 —— 确认测量管线本身没问题
    配置：ZsRE 原版（raw prompt + 60% pool + 无 scaffold），10 条用【ZsRE 式 src】
          （fact 和 belief 都写成独立问句、belief 剥 JQ）
    期望：locality 好（margin 大）
    判读：
      · 若这步就崩 → 数据/测量本身有问题，与 chat 层无关，先停下查这个
      · 若好（预期）→ 数据在正确配置下 OK，继续加污染

S1  +JQ 框架 —— 只给 fact 加回 "of JQ's"，belief 仍剥 JQ，其余同 S0
    F1–F5 用 "What is the name of JQ's cat?" 这种带 JQ 框架的 src
    B1–B5 仍 flat（独立问句）
    Δ(S1−S0) = "JQ 共享主语框架" 对 locality 的伤害
    ★ 判读：若 JQ-fact 间(F1 vs F2)开始撞车、而 belief(B 组)还好
            → 坐实"共享 JQ 框架"是锥塌缩主因（第一性原理推的那个）

S2  +query-span pooling —— 把 60% 换成 chat 的 query-span flat-mean，其余同 S1
    Δ(S2−S1) = 池化方式的额外伤害
    判读：上轮 sweep 已暗示差异不大，这步确认（预期 Δ 小）

S3  +chat scaffold —— 把 raw prompt 换成完整 chat 模板
    （system + 空 RAG 窗 + query），其余同 S2
    Δ(S3−S2) = chat scaffold 的伤害（Plan B 当初要对付的东西）
    S3 = 完整 chat 配置 = 现状（应复现 0.88 锥）
```

### 3.1 结果判读(locality 在哪步第一次崩 = 真凶)

| 第一次崩在 | 真凶 | 含义 / 修法方向(下一轮才做) |
|---|---|---|
| **S1** | JQ 共享框架 | key 构造层:fact 也被 "of JQ's" 框架绑成一锥 → 修 key 构造(独立化 / 或重新考虑 fact 路由) |
| **S2** | query-span pooling | 池化方式(但上轮已基本排除,意外才会是这里) |
| **S3** | chat scaffold | Plan B 切 scaffold 切得不够干净 → 修 scaffold 处理 |

> 预判(待数据证伪):主因 **崩在 S1(JQ 框架)**,S3(scaffold)叠加。但阶梯的意义就是把预判换成数据 —— 以实测为准。

---

## 4. 输出

一张跨级表:

```
level | 改了什么(vs 上级) | DIRECT 均值 | PARA 均值 | NEG 地板 | margin(PARA−NEG) | locality 好?
  S0  | ZsRE 基线          |   ...      |   ...    |   ...   |      ...         |   ✓/✗
  S1  | +JQ 框架           |   ...      |   ...    |   ...   |      ...         |   ✓/✗
  S2  | +query-span pool   |   ...      |   ...    |   ...   |      ...         |   ✓/✗
  S3  | +chat scaffold     |   ...      |   ...    |   ...   |      ...         |   ✓/✗
```

外加 fact 组 vs belief 组**分开**报(因为 belief 在 S1 仍 flat,二者会分叉,这个分叉本身是证据)。

然后明确回答:
1. S0 基线 locality 好不好?(不好 → 数据/测量问题,先停)
2. locality **第一次**在哪级崩?→ 那级的变量 = 真凶
3. fact 组和 belief 组的崩塌点一样吗?(belief 若一直好、fact 在 S1 崩 → 坐实框架污染 + fact 特异性)
4. 一句话结论:锥塌缩的真凶是 JQ 框架 / pooling / scaffold 中的哪个(或叠加)

---

## 5. 护栏 / 不做

**measurement-only,只定位不修复。**

- ✗ 不做任何"修复":不 mean-center、不 contrast-out、不重标阈值、不改 key 构造上线 —— 这些是**定位完之后**才决定的事,本轮一律不碰。
- ✗ 不改 >1 个变量/级(阶梯铁律)。
- ✗ 不动 raw 路径的 `query_selection_strategy` / `_select_query` / threshold / 其余 hparam。
- ✗ 不碰 live server / codebook / memory / serving / editing 逻辑 / samples.json / frontend。
- ✗ 不手编/不发明测试样本 —— 10 条全从 eval/samples.json 按 id 捞真的。
- ✗ 各级间不变 seed / 阈值 / 数据子集 / layer / n_iter / edit_lr。
- ✓ 独立 model 副本(server 占 ~16G,余量够),measurement script 走 scratchpad / spikes。
- ✓ 全英文(数据 + probe)。
- ✓ 首尾打印 torch 版本确认无回归;不装包不降级。
- ✓ 复用生产 keying 函数(`pool_span_rows` / `compute_key` / `query_span_in_rendered`),测的就是线上逻辑,零 drift。
- ✓ 共享工作区:只 git add 本轮文件(spike + 本 doc),绝不 `-A`;只 commit 不 push。

---

## 6. 验证顺序(给 agent 的执行步骤)

```
1. 先 READ：确认 ZsRE 原版 keying 入口（raw 60%）、你的 chat keying（query-span
   flat-mean + scaffold 渲染 _hero_render）、生产 deferral cosine 算法、阈值。
   报告这几处实际代码位置，再动手。
2. 从 eval/samples.json 按 id 捞 10 条（5 fact + 5 belief），构造 §1.2 的字段。
   打印捞到的真实样本，确认非编造。
3. 写 measurement spike：实现 S0–S3 四级（每级一个 keying 配置开关），
   每级对 10 条跑 DIRECT/PARA/NEG，算 margin。
4. inline 串行跑（单 GPU），出 §4 那张跨级表 + fact/belief 分组。
5. ★ STOP 报告：贴四级表 + 标出 locality 第一次崩在哪级 + 回答 §4 的 4 个问题。
   等用户定下一步（修哪一层）。不自动进入修复。
```

---

几个给你(不是给 agent)的提醒:

**这份 doc 的灵魂是 S0→S3 的单变量阶梯 + fact/belief 分组对照**。fact 组和 belief 组在 S1 会分叉(belief 仍 flat、fact 加了 JQ 框架),这个分叉本身就是判据——如果 belief 一直好、fact 在 S1 崩,那"JQ 共享框架是真凶"就被坐实了,而且顺带证明你那个 belief/fact 二分的第一性原理洞察是对的。

**我把"不修复"焊得很死**(护栏第一条),因为 agent 几次都想直接冲去 mean-center。这一轮的任务**只有定位**——定位到崩在哪级,你才知道该修 key 构造(S1)还是 scaffold(S3),否则又是盲修。

**预判我写进去了但标了"待证伪"**:主因 S1(JQ 框架)、S3 叠加。出来的表如果真是这样,你下一轮的修复方向就很清楚了——让 fact 的 key 别被 "of JQ's" 框架绑死。但一切以那张表为准,别让预判变成结论。

跑完把那张 S0–S3 的表贴给我,尤其 **locality 第一次崩在哪级 + fact/belief 有没有分叉**,我陪你定下一刀切哪里。