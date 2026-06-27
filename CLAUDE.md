# CLAUDE.md — AIEHackathon 工作规则

这是一个 hackathon 项目。以下规则在本仓库(`/workspace/AIEHackathon`)的所有会话中始终遵守。

## 核心规则

1. **能并行就用 agent teams (workflow)。**
   开发时如果任务可以拆分并行(多文件改动、多模块实现、调研 + 实现分头进行、广覆盖的搜索/审查等),
   优先用多代理 workflow 编排,而不是单线程一件件做。目标是吃满并行度、加快进度。
   - 串行依赖强、或只是琐碎一次性改动时,不必强行上 workflow。

2. **每次完成任务,默认 commit。**
   这是 hackathon,git history 要尽可能体现我们的进度。每完成一个有意义的小步骤就 commit,
   提交信息写清楚做了什么。不要把一堆无关改动攒成一个大 commit。
   - 默认只 commit,不自动 push;需要 push 时按用户指示来。
   - 凭据:已配好 HTTPS PAT(`/workspace/.git-credentials`),`git push origin <branch>` 可直接用。

## 沟通方式
- 用中文交流,技术术语 / 文件名 / 命令保持英文原形。
- 每完成一步,简要说明做了什么、结果如何。
