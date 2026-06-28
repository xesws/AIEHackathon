# Stock HoReN ZsRE sanity — N=30 sequential (native metrics)

**日期**: 2026-06-28 · **类型**: measurement-only sanity（无源码改动）

## 目的
在继续信任/调试 Engram 的 **chat 路径**（deferral/keying/query-span pooling，v1.4 等）之前,先建一个 ground-truth 锚点:把**原版 HoReN** 在它的**主场 benchmark ZsRE** 上、用它**自己的 eval + 自己的 metrics**、**paper-faithful 配置**跑一小段顺序编辑,看 vendored editing backend 本身能否复现 paper 级数字。借此把「backend 对不对」与「我们的 chat/keying 层是不是病根」**隔离**开。

## 怎么跑的（stock,不掺 Engram 代码）
- **入口**: `third_party/horen/experiments/run_structured_data_editing.py`（ZsRE 分支 → `BaseEditor.edit`）。
- **调用**: `--editing_method HOREN --data_dir ./data/ZsRE --data_type ZsRE --ds_size 30 --sequential_edit`。
- **配置**: **paper-faithful** `n_iter=50, edit_lr=0.1`（HoReN paper §E.4.2）。落到 **scratch 里的临时 yaml 副本**(由 `git show HEAD:hparams/HOREN/llama3.1-8b.yaml` 取出再把 `edit_lr 1.0→0.1`),**tracked yaml 原封未动**。其余 stock:Instruct 模型、layer-29 down_proj、bf16、`hopfield_key_match_threshold=0.85`、`query_selection_strategy=last_60_perc_prompt_tokens_avg`。
  > 注:出厂 yaml 是 `50/1.0`,但 in-repo 审计指出 `edit_lr=1.0` 与 paper 矛盾(paper 钉 0.1);工作区当前压着的是未提交的 `100/0.1`。本轮按用户选择用 paper 的 `50/0.1`。
- **数据真实**: `data/ZsRE/zsre_edit_data.json`（5w 条真 ZsRE)前 30 条,顺序固定(`edit_raw[:30]`,不 shuffle)。`--sequential_edit`=codebook 累积(官方协议;editor params 8192→45056→126976 印证累积)。
- **独立 model 副本**,未碰 live server / codebook / memory / serving / samples.json。

## 原生 ZsRE 指标（HoReN 自己的 `src/evaluate/`,greedy-gen token-EM / `vanilla_generation`）
`summary.post`,对 edits 1..N 求均值:

| checkpoint | reliability (`rewrite_acc`) | generalization (`rephrase_acc`) | locality (`neighborhood_acc`) |
|---|---|---|---|
| N=1  | **1.000** | **1.000** | **1.000** |
| N=10 | **1.000** | 0.900 | **1.000** |
| **N=30** | **1.000** | **0.867** | **1.000** |

(pre-edit baseline: rewrite_acc 0.017、rephrase_acc 0.0 → 编辑前模型几乎不知道答案,确认是真编辑而非命中先验。per-edit `rewrite_F1`/`rephrase_F1` 因 token-id 对齐口径恒 0,不进 summary,不是 reliability 信号——以 `rewrite_acc` 为准。)

## 结论
- **vendored HoReN backend 在主场 ZsRE 上复现 paper 级数字**:reliability 1.00、locality 1.00、generalization 0.87(随 codebook 增长从 1.0 缓降,典型 ZsRE 形态:可靠性/locality 满分,泛化是略弱的一轴)。
- ⇒ **editing backend 本身是健康的**。我们连日在 debug 的 deferral/cone-collapse/语言错配问题,病根在 **Engram 自己的 chat/keying 层**(chat scaffold + query-span pooling + 中英 key 分布),**不在底层 HoReN 方法**。这条 sanity 把锅明确划给我们这一层。
- 注:repo 内**没有** paper 的 ZsRE 精确数字(只有 UnKE 数字),故 ballpark 为定性判断;但 reliability/locality 满分、generalization 高,已明确落在「编辑生效且不误伤」的合理区间,绝非 pipeline 失败(reliability≈0)那种。

## 环境
- `torch 2.8.0+cu128 / True`(首尾一致,无装包/降级)。
- GPU:跑前 45.5GB 空闲;8B bf16 副本峰值约 ~16–18GB(未逐点采样),无 OOM;跑后干净释放回 0 MiB。
- 编辑循环 ~2:39(~4s/edit)。

## 护栏
- 只用 stock eval/metrics/config,未掺 Engram chat/keying/query-span;未改 HoReN config/metric 代码/数据(用 scratch 副本);未跑全量(N=30 封顶);未碰 live server/codebook/memory/serving/samples.json;独立 model 副本;torch 未动。
- 共享工作区:只 commit 本 doc 一个文件,未 `-A`、未 branch。run 产物在 `third_party/horen/{logs,outputs}/`(未纳入提交)。
