# SGLang + OpenClaw Agent Infra Research

> 日期：2026-05-25
> 定位：Stage 2 近期研究推进产物
> 关联设计：[../stage2-design/overview.md](../stage2-design/overview.md)、[../stage2-design/compute-graph-optimization-design.md](../stage2-design/compute-graph-optimization-design.md)
> 目标系统：OpenClaw（agent runtime）+ SGLang（inference serving）

---

## 1. 产物索引

本目录把 Stage 2 近期两条主线拆成可继续评审、实现和实验的研究产物：

| 文档 | 定位 | 主要内容 |
|------|------|----------|
| [00-research-plan.md](00-research-plan.md) | 研究计划 | 研究问题、阶段划分、产出物、决策门和验证路径 |
| [01-direction-a-graph-aware-scheduling-research.md](01-direction-a-graph-aware-scheduling-research.md) | 中间研究报告 A | 图感知推理请求调度、KV lifecycle、相关论文和开源实现分析 |
| [02-direction-b-critical-path-speculation-research.md](02-direction-b-critical-path-speculation-research.md) | 中间研究报告 B | 关键路径预测、read-only tool speculation、PASTE/B-PASTE 与系统适配分析 |
| [03-sglang-openclaw-feasibility-design.md](03-sglang-openclaw-feasibility-design.md) | 最终设计稿 | 可行性结论、系统分层、SGLang/OpenClaw 详细实现设计、评估方案 |
| [04-source-code-audit-evidence.md](04-source-code-audit-evidence.md) | 代码审计附录 | 外部开源实现、本地 SGLang/OpenClaw 关键文件和证据路径 |
| [05-literature-code-coverage.md](05-literature-code-coverage.md) | 覆盖复核 | 论文遗漏回顾、机构归属、原论文/其他途径代码状态、纳入判断 |
| [06-inter-workflow-kvflow-halo-deep-dive.md](06-inter-workflow-kvflow-halo-deep-dive.md) | 专题深挖 | KVFlow 与 Halo 的 inter-workflow 定位、论文机制、开源实现、SGLang/OpenClaw 适配判断 |
| [07-gasr-anthropic-cache-export-patch.md](07-gasr-anthropic-cache-export-patch.md) | GASR patch 记录 | Claude Code / Anthropic path 的 cache detail export patch、重启检查项和 proxy 对接关系 |

---

## 2. 当前结论摘要

近期工作应该先聚焦在 **control plane / signal plane**，而不是直接改通信载体或把 agent runtime 下沉到 serving engine 内部。

**方向 A：图感知推理请求调度。** 方向 A 具备工程可行性，但不能简单把 workflow id 放进 SGLang 现有字段就结束。SGLang 当前已有 `extra_key`、`routing_key`、`priority`、`custom_labels` 等入口，但它们的语义边界不同：`extra_key` 会进入 prefix cache key，适合表达 cache namespace；`routing_key` 当前属于 cache-agnostic scheduling policy，启用后会绕开 LPM/DFS_WEIGHT 的 prefix-aware 排序；`custom_labels` 更适合观测归因。因此，主路线应该是新增 typed `agent_hints` 和一个轻量的 AgentStateManager，让 scheduler/cache manager 在不破坏现有 prefix cache 语义的前提下消费 workflow、tool lifecycle、critical path 等信号。

**方向 B：关键路径预测与优化。** 方向 B 更适合先放在 OpenClaw 侧做：OpenClaw 掌握 tool start/end、side effect、sub-agent fan-out/fan-in 和 replay invalidation 信息，天然能重建 workflow trace。SGLang 不应执行 tool，也不应该理解完整 agent DAG；它只消费 B 产生的 `critical_path_rank`、KV prefetch deadline、tool wait window 等 typed hints。PASTE/B-PASTE 的论文方向值得借鉴，但当前没有找到可直接复用的官方开源实现；近期实现要从 trace mining、critical path rank 和严格 read-only speculation 开始。

**核心 mismatch。** 学术系统常假设显式 DAG、semantic variable 或可声明 workflow；OpenClaw 的真实 workload 是动态 agent loop，tool 副作用和用户可见状态必须被严格治理。SGLang 的真实边界是 tokenized request、scheduler、Radix/HiRadix cache，而不是 agent workflow runtime。近期适配要把完整图语义留在 OpenClaw，把 serving-relevant 的少量稳定信号传给 SGLang。

---

## 3. 推荐实施顺序

1. **Phase 0：trace 与字段盘点。** 固化 OpenClaw 中 LLM request、tool lifecycle、spawn/fan-in 的事件字段，并确认 SGLang 当前字段从 OpenAI entrypoint 到 scheduler/cache 的传播路径。
2. **Phase 1：无行为变化的 signal baseline。** 通过 `custom_labels` 和 trace 标注 workflow/session/role/cache_scope，形成可观测基线；同时对比 SGLang `fcfs`、`lpm`、`dfs-weight`、`routing-key`。
3. **Phase 2：typed agent hints。** 在 SGLang request path 中增加 `agent_hints`，传递 workflow/session/step/critical_path/cache_lifecycle 信息；在 OpenClaw SGLang provider adapter 中生成这些 hints。
4. **Phase 3：agent-aware scheduling + KV lifecycle。** 增加 AgentStateManager 和 cache-aware composite policy，把 prefix match、workflow state、critical_path_rank 组合起来，而不是用 cache-agnostic `routing-key` 替代 prefix-aware policy。
5. **Phase 4：关键路径分析和 read-only speculation。** 在 OpenClaw 侧做 trace-driven critical path analyzer；只对显式 read-only、idempotent、cacheable 的 tool 做 speculation。

---

## 4. 外部来源

本轮近期两条方向纳入的主要论文和开源实现包括：

- Continuum: [arXiv:2511.02230](https://arxiv.org/abs/2511.02230), [github.com/Hanchenli/vllm-continuum](https://github.com/Hanchenli/vllm-continuum)；机构：UC Berkeley、Stanford University、Tensormesh、Tsinghua University。
- KVFlow: [arXiv:2507.07400](https://arxiv.org/abs/2507.07400), [github.com/PanZaifeng/KVFlow](https://github.com/PanZaifeng/KVFlow)；机构：University of California, San Diego、Amazon Web Services。
- ScaleSim: [arXiv:2601.21473](https://arxiv.org/abs/2601.21473), [github.com/PanZaifeng/KVFlow](https://github.com/PanZaifeng/KVFlow)；机构：University of California, San Diego、Amazon Web Services。
- Autellix: [arXiv:2502.13965](https://arxiv.org/abs/2502.13965)；机构：UC Berkeley、Google DeepMind、Shanghai Jiao Tong University。
- Halo: [arXiv:2509.02121](https://arxiv.org/abs/2509.02121), [github.com/mlsys-io/Halo_demo](https://github.com/mlsys-io/Halo_demo)；机构：National University of Singapore。
- APIServe / InferCept: [arXiv:2402.01869](https://arxiv.org/abs/2402.01869), [github.com/WukLab/infercept](https://github.com/WukLab/infercept)；机构：University of California, San Diego。
- LAMPS / MARS: [arXiv:2410.18248](https://arxiv.org/abs/2410.18248), [github.com/mars-repository/mars-codebase](https://github.com/mars-repository/mars-codebase)；机构：Harvard University、Tsinghua University。
- PASTE: [arXiv:2603.18897](https://arxiv.org/abs/2603.18897)；机构：Shanghai Jiao Tong University、Microsoft Research、Stevens Institute of Technology。
- B-PASTE: [arXiv:2604.16469](https://arxiv.org/abs/2604.16469)；机构：Independent Researcher。
- Conveyor: [arXiv:2406.00059](https://arxiv.org/abs/2406.00059), [github.com/conveyor-sys/conveyor](https://github.com/conveyor-sys/conveyor)；机构：Duke University。
- Parrot: [paper](https://www.usenix.org/conference/osdi24/presentation/lin-chaofan), [github.com/microsoft/ParrotServe](https://github.com/microsoft/ParrotServe)；机构：Shanghai Jiao Tong University、Microsoft Research。
- Teola / Ayo: [arXiv:2407.00326](https://arxiv.org/abs/2407.00326), [github.com/NetX-lab/Ayo](https://github.com/NetX-lab/Ayo)；机构：The Chinese University of Hong Kong、Unaffiliated。

ALTO、Pie、Symphony、AsyncLM、Agent.xpu 等相邻系统不进入本轮近期两方向的详细设计，保留在 [05-literature-code-coverage.md](05-literature-code-coverage.md) 作为另行规划的覆盖项。
