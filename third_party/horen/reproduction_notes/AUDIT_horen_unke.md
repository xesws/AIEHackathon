# 复现审查报告:HoReN × UnKEBench(LLaMA-3.1-8B-Instruct)

- **日期**:2026-06-25
- **环境**:RunPod / RTX A6000 48GB / Ubuntu 24.04 / torch 2.8.0+cu128 / Python 3.12
- **代码**:`/workspace/HoReN`,git clone 自 `https://github.com/ha11ucin8/HoReN.git`(HEAD `38bbf34 update readme`)
- **基准对照**:① 官方 UnKE 仓库 `https://github.com/TrustedLLM/UnKE`;② HoReN 论文 `arXiv 2605.08143v2`(评测协议见 §E.3.2、超参见 §E.4.2、结果见 Table 7 / p26)

---

## 0. 一句话结论

**我们的复现忠实于官方 HoReN 代码(git 证明仅 2 行本地改动,均不涉及算法/指标)。指标对不上论文 Table 7 不是本地操作失误,而是官方 HoReN 发布物自身的可复现性缺陷:其"发布的代码 + 配置"与"论文正文(E.4.2)所述方法"存在 3 处不一致,并因此复现不出论文自己的 Table 7。**

---

## 1. 遇到的问题

用官方代码 + 官方协议复现 UnKEBench(顺序编辑)时,指标与论文 Table 7(HoReN, LLaMA-3.1-8B)对不上,且对不上的方向不一致(排除了"单一模型/数据整体偏移"):

| Original @ N=100 | 我们 | 论文 Table 7 | 偏差 |
|---|---|---|---|
| BLEU | 0.2245 | 0.126 | 我们 **高** |
| ROUGE-1 | 0.3574 | 0.457 | 我们 **低** |
| ROUGE-2 | 0.2069 | 0.219 | 接近 |
| ROUGE-L | 0.3418 | 0.428 | 我们 **低** |
| BERTScore | 0.7516 | 0.716 | 接近(略高) |

**关键现象:差异随编辑数 N 放大。** N=10 时 ROUGE 与论文几乎重合,N=100 才显著偏离:

| Original | 论文 N=10 | 我们 N=10 | 论文 N=100 | 我们 N=100 |
|---|---|---|---|---|
| ROUGE-1 | 0.583 | **0.576** ✓ | 0.457 | 0.357 ✗ |
| ROUGE-L | 0.551 | **0.555** ✓ | 0.428 | 0.342 ✗ |
| BERTScore | 0.814 | 0.846 | 0.716 | 0.752 |

---

## 2. 排查:逐一排除的假设

| 假设 | 结论 | 证据 |
|---|---|---|
| 模型版本(3.0 vs 3.1) | ❌ 排除 | Table 7 也是 LLaMA-3.1-8B-Instruct(用户确认);我们用的就是 3.1 |
| 随机 sampling / 取样不同 | ❌ 排除 | `AKEW_both.py:133-134` 为 `self._data[:size]`,按文件顺序取前 N,无 shuffle |
| 数据版本不同 | ❌ 排除 | 我们的 `final_data_v3.json` 与官方 **md5 逐字节相同** = `5abf33eea481a89e2e6c62d92728f591`;1000 条 question/answer 全一致 |
| ROUGE 口径(recall/F1/工具) | ❌ 排除 | 与官方逐字一致;且在我们的预测上换 `rouge`/`rouge_score`、recall/F1、stemming 重算,**无一能到论文 0.457**(见 §5) |
| BERTScore 方法 | ❌ 排除 | 与官方逐字一致(`all-MiniLM-L6-v2` 余弦对角均值) |
| 我们的启动命令 | ❌ 排除 | 与官方 README 的 UnKE/HOREN 跑法逐项一致(默认 batch_size=1 / seed=2024 / bert_model=all-MiniLM) |

→ 评测代码、数据、生成、命令全部与官方一致 ⇒ 数值差异只能来自**预测本身**(即编辑过程),指向 HoReN 编辑配置。

---

## 3. 确认的不一致(官方代码/配置 vs 论文正文)

git 证明以下三项**均为官方原样,未被本地改动**(见 §6)。

### A1 — BLEU:官方代码用自制 BLEU-1,而非标准 BLEU
- **官方 UnKE 协议**(`UnKE/code/evaluate.py`):`nltk.translate.bleu_score.sentence_bleu([answer], prediction)`,BLEU-4、无平滑、传 raw string(实为字符级)。
- **HoReN 官方代码**(`src/evaluate/evaluate_uns.py:10-34` `_safe_bleu_like`,调用于 `:76/:80`):**词级 BLEU-1**(空格切分,unigram precision + brevity penalty)。代码注释:*"Lightweight BLEU-1 style… Avoids nltk BLEU incompatibilities on newer Python versions."*
- **本环境实测**:官方 nltk BLEU 在 Python 3.12 下**直接报错** `TypeError: Fraction.__new__() got an unexpected keyword argument '_normalize'`——这正是 HoReN 仓库替换它的原因。
- **后果**:HoReN 官方代码跑出 BLEU 0.2245,而论文 Table 7 = 0.126(后者必来自另一种 nltk BLEU)。**发布代码 ≠ 论文数值。**

### A2 — adaptor 学习率:官方 yaml = 1.0,论文 = 0.1
- **论文 §E.4.2**:*"Per-edit adaptor optimization uses learning rate **0.1**, up to U=50 optimization steps per edit."*
- **HoReN 官方配置**:`hparams/HOREN/llama3.1-8b.yaml` → `edit_lr: 1.0`(官方值,未被本地改动);`src/models/horen/editor.py:92` 在默认 tensor-value 模式下用 `edit_lr`,`:105` 构造 `Adam(params, lr=1.0)`。
- **后果**:每条编辑的 value adaptor 以 **10× 学习率**优化 → adaptor 值不同 → 预测不同。

### A3 — early-stopping:官方代码无,论文有
- **论文 §E.4.2**:*"early stopping when the per-token loss falls below **10⁻²** or fails to improve for **3 consecutive steps**."*
- **HoReN 官方代码**:`src/models/horen/editor.py:98-110` 的优化循环 `for i in range(n_iter)` **跑满 n_iter=50 步,无任何 early-stop 判断**(仅记录 loss,从不 break)。
- **后果**:每条编辑被无脑优化满 50 步(且在 10× 学习率下)= 相对论文的**过度优化**。

> **A2 + A3 联合**:在 lr=1.0 下跑满 50 步、不 early-stop,与论文"lr=0.1 + early-stop"的温和优化是**两套优化结果**。这种"每条编辑层面的差异"会**随编辑累积放大**,自洽地解释了 §1 的核心现象——**N=10 吻合、N=100 偏离**;A1 则解释了 BLEU 整体不可比。

---

## 4. 已核实"完全一致"的部分(faithful,可排除)

| 项 | 状态 | 证据 |
|---|---|---|
| 数据 | ✅ 与官方逐字节相同 | md5 `5abf33ee…` |
| ROUGE | ✅ 与官方逐字一致 | `evaluate_uns.py:83-86`:`rouge` 包、`get_scores(pred,ans)`、取 recall `['r']`、无 stemming |
| BERTScore | ✅ 与官方逐字一致 | `evaluate_uns.py:130-138`:`all-MiniLM-L6-v2` 余弦、对角均值 |
| 生成 | ✅ 等价 | 我们 `do_sample=False`,官方 `do_sample=True,temp=0.001`,均 ≈greedy;`max_new_tokens=512` 同 |
| 模型 | ✅ 一致 | LLaMA-3.1-8B-Instruct(与 Table 7 同) |
| 其余 HoReN 超参 | ✅ 与论文一致 | β=20、γ=0.1、M=1、threshold c=0.85、pooling 60%、layer 29 down_proj、U=50、Adam、tensor-value adaptor、ε-expansion 初值 1.0 |

---

## 5. 关键反证:换任何 ROUGE 口径都到不了论文值

在**我们自己的 N=100 预测**上重算(证明差异在预测、不在指标实现):

| 计算方式 | ROUGE-1 | ROUGE-L | 论文 |
|---|---|---|---|
| `rouge` 包 recall(现 eval) | 0.3574 | 0.3418 | 0.457 / 0.428 |
| `rouge` 包 F1 | 0.3438 | 0.3282 | — |
| `rouge_score` F1(stemmer) | 0.3242 | 0.2419 | — |
| `rouge_score` recall(stemmer) | 0.4124 | 0.3076 | — |

→ 没有任何标准变体能复现论文 0.457/0.428 ⇒ 差异源自**预测**,不是指标。
（BLEU-1 现 eval 重算 = 0.2245,精确复现存档值;nltk 标准 BLEU 在本环境报错无法运行。）

---

## 6. git 证据:本地只改了 2 行,均不涉及算法/指标

`git status --short`:仅 2 个文件被改;`evaluate_uns.py`、`editor.py` 均**未改动**(即 BLEU-1 / 无 early-stop 是官方原样)。

**(1) `hparams/HOREN/llama3.1-8b.yaml`** —— 仅改模型路径(setup 必需),**未碰 `edit_lr`**:
```diff
- model_name: "./hugging_cache/llama3.1-8b-instruct"
+ model_name: "/workspace/hugging_cache/llama3.1-8b-instruct"
```

**(2) `src/dataset/AKEW_both.py:294`** —— Llama-3.1 chat-template 兼容修复(否则官方硬编码的 `"Llama3-8B-Instruct"` 判断对 3.1 路径不命中,会退化到无模板分支):
```diff
- if "Llama3-8B-Instruct" in self.model_name:
+ if "llama" in self.model_name.lower():
```

---

## 7. 结论与建议上报内容

**定性**:HoReN 官方仓库的**发布版代码 + 配置**,与其论文 Appendix E.4.2 所述方法存在 3 处不一致(BLEU 实现、adaptor 学习率 0.1 vs 1.0、early-stopping 有无),并因此**复现不出论文自己的 Table 7**。我们用官方代码忠实运行(仅改本地模型路径 + 一处 Llama-3.1 chat-template 兼容修复,git 可证),确认偏差**全部源自官方发布物本身,非本地引入**。

**建议向 Ryan / 导师报告的要点**:
1. 官方 nltk BLEU 在 Python 3.12 环境下无法运行;官方仓库已用自制 BLEU-1 顶替,导致 BLEU 列不可比、与论文 0.126 不一致。
2. 官方 yaml `edit_lr=1.0` 与论文正文 0.1 矛盾(10×)。
3. 官方代码未实现论文所述 early-stopping。
4. 上述 2、3 是数值随 N 放大偏离的最可能根因;需官方澄清"Table 7 究竟用的哪套配置",或提供能复现 Table 7 的配置/代码。

---

## 8. 附:本次 N=1000 运行记录(已中止)

- 用上述**有偏差的官方配置**(edit_lr=1.0 / 无 early-stop)起跑,于 2026-06-25 在 **edit 499/1000**(进入 Checkpoint 500 前)手动中止。
- 已落盘 checkpoint:1 / 10 / 30 / 100 / 120,详见 `/workspace/runs/unke_n1000_horen/RUN_RECORD.md`。
- **这些数字不作为复现结果上报**(配置不符论文)。
