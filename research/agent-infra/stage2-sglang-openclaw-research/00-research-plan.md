# SGLang + OpenClaw 近期研究计划

> 日期：2026-05-25
> 定位：研究展开计划，面向后续论文调研、代码适配和实验验证

---

## 1. 研究目标

本轮研究服务于 Stage 2 的两条近期主线：

1. **方向 A：图感知推理请求调度。** 研究如何让 SGLang 的 scheduler、Radix/HiRadix cache、cache controller 消费来自 OpenClaw 的 workflow signal、tool lifecycle 和 KV residency hint，从而改进 queue ordering、batch composition、KV write-through/load-back/prefetch。
2. **方向 B：关键路径预测与优化。** 研究如何从 OpenClaw trace 中识别 agent workflow 的关键路径，产生 `critical_path_rank`、tool wait window、read-only speculation candidate，并把结果输入方向 A。

最终产物不是单纯文献综述，而是面向 OpenClaw + SGLang 的落地设计：明确哪些思想可以直接复用，哪些论文假设和真实系统存在 mismatch，以及适配工作应该发生在 OpenClaw、SGLang 还是两者之间的 signal plane。

---

## 2. 研究问题

### 2.1 论文与开源实现问题

- 方向 A/B 相关论文分别解决什么问题，使用了什么抽象：DAG、semantic variable、step graph、pattern tuple、tool wait window、KV TTL。
- 是否存在公开代码；若存在，代码中真实实现是否与论文表述一致。
- 实现中哪些机制是 engine-native 的，哪些只是 benchmark/workload 特化逻辑。
- 哪些机制可以作为近期实现参考，哪些只能作为远期系统形态参考。

### 2.2 SGLang 适配问题

- SGLang 当前可用的 request-time 字段有哪些：`extra_key`、`priority`、`routing_key`、`custom_labels`。
- 这些字段分别进入 scheduler、prefix cache、metrics 的哪个位置；能不能承载 workflow 语义。
- 现有 `lpm`、`dfs-weight`、`routing-key`、`priority` policy 的语义边界是什么。
- RadixCache / HiRadixCache / CacheController 是否能表达 tool wait 期间的 write-through、load-back、prefetch、eviction priority。
- 如果新增 typed `agent_hints`，需要沿哪些结构传播到 `Req`、scheduler policy 和 cache controller。

### 2.3 OpenClaw 适配问题

- OpenClaw 的 agent loop 中哪些位置能产生可靠的 workflow/session/step/tool lifecycle 信息。
- SGLang provider extension 当前是否只做 provider registration，是否已有 payload/header patch 能力。
- tool 副作用、replay invalidation、mutating action 如何限制 speculation。
- sub-agent spawn/fan-in 是否能形成稳定 trace，能否输出 `parent_step_id`、`workflow_id`、`run_id`。
- 适配层应该是 SGLang-specific extension，还是通用 provider hook + SGLang adapter。

---

## 3. 工作包

### WP1：论文和代码审计

输入：

- 方向 A 近期纳入：Autellix、Continuum、KVFlow、ScaleSim、InferCept、LAMPS/MARS；Halo、Parrot、Ayo/Teola 只作为 graph signal / criticality / workflow IR 的辅助参考。
- 方向 B 近期纳入：PASTE、B-PASTE、Conveyor；Halo critical distance、InferCept、LAMPS/MARS 作为 critical path 与 tool/API wait bridge 的辅助参考。
- 另行规划：ALTO、Pie/Symphony、AsyncLM、Agent.xpu 等不进入本轮 Stage2 近期两方向的详细设计，只在覆盖矩阵中记录代码状态与后续跟踪价值。

输出：

- 每篇/每个系统的核心抽象、实现位置、开源状态、可复用点、系统假设。
- 和 SGLang/OpenClaw 的适配风险。
- 单独形成论文覆盖与代码状态矩阵，区分“原论文直接给代码”和“其他途径找到代码”。

### WP2：SGLang 代码路径审计

重点文件：

- `python/sglang/srt/managers/io_struct.py`
- `python/sglang/srt/entrypoints/openai/serving_base.py`
- `python/sglang/srt/entrypoints/openai/serving_chat.py`
- `python/sglang/srt/managers/schedule_policy.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/mem_cache/radix_cache.py`
- `python/sglang/srt/mem_cache/hiradix_cache.py`
- `python/sglang/srt/managers/cache_controller.py`

输出：

- 现有字段传播路径。
- cache-aware 与 cache-agnostic policy 的边界。
- `extra_key`、`routing_key`、`priority` 的可用性和风险。
- typed `agent_hints` 与 AgentStateManager 的最小实现面。

### WP3：OpenClaw 代码路径审计

重点文件：

- `extensions/sglang/index.ts`
- `extensions/sglang/api.ts`
- `src/plugin-sdk/provider-stream-shared.ts`
- `src/agents/pi-embedded-runner/stream-payload-utils.ts`
- `src/agents/pi-embedded-subscribe.handlers.tools.ts`
- `src/agents/acp-spawn.ts`

输出：

- SGLang provider extension 当前能力。
- LLM request payload/header patch 的现有机制和缺口。
- tool start/end、duration、side effect、replay invalidation 的事件源。
- spawn/fan-in trace 的字段设计建议。

### WP4：signal contract 设计

输出：

- request-time `agent_hints` schema。
- out-of-band lifecycle event schema。
- custom labels / metrics / trace 归因字段。
- OpenClaw adapter 到 SGLang request/control endpoint 的映射。

### WP5：实验与可行性验证

输出：

- baseline matrix：`fcfs`、`lpm`、`dfs-weight`、`routing-key`、agent-aware composite。
- workload pools：线性多轮、tool wait、fan-out/fan-in、sub-agent research、混合前后台请求。
- metrics：task completion latency、TTFT、queue wait、prefix hit tokens、host/storage hit tokens、eviction/load-back、speculation hit/waste、安全拒绝数。

---

## 4. 决策门

| 决策门 | 要回答的问题 | 通过标准 | 影响 |
|--------|--------------|----------|------|
| G0 字段闭环 | OpenClaw 产生的 workflow/session/tool 字段能否到达 SGLang metrics/trace | 无行为变化下可按 workflow/session 聚合 queue/cache 指标 | 进入 typed `agent_hints` |
| G1 prefix baseline | `lpm` / `dfs-weight` 是否已经覆盖大部分 workflow affinity 收益 | 相同 workload 下 prefix hit 与 latency 接近 agent-aware 策略 | 降低 scheduler 改造优先级 |
| G2 lifecycle window | tool wait window 是否足够长，且 KV 驱逐/load-back 成本是否显著 | tool wait 期间存在可观 HBM 压力或 load-back 延迟 | 进入 KV lifecycle/prefetch |
| G3 critical path skew | task latency 是否由少数节点/边决定 | P95/P99 中关键路径贡献显著高于非关键节点 | 进入 critical_path_rank |
| G4 speculation safety | read-only tool pattern 是否稳定且可安全缓存 | 命中率、浪费率、副作用约束均达标 | 进入 read-only speculation |

---

## 5. 阶段产出

### Phase 0：研究与代码审计

产出：

- 方向 A 中间研究报告。
- 方向 B 中间研究报告。
- SGLang/OpenClaw mismatch 与适配面清单。

### Phase 1：观测闭环

产出：

- OpenClaw trace schema。
- SGLang request labels/header/body baseline。
- baseline benchmark 记录。

### Phase 2：typed signal plane

产出：

- `agent_hints` request schema。
- SGLang request path 传播实现。
- OpenClaw SGLang adapter 实现。

### Phase 3：serving 决策适配

产出：

- AgentStateManager。
- agent-aware cache-aware scheduler policy。
- KV lifecycle hint 到 HiRadix/CacheController 的最小适配。

### Phase 4：critical path 与 speculation

产出：

- critical path analyzer。
- read-only tool pattern registry。
- speculation result cache 与安全审计指标。
