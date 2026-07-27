# 方向 B 中间研究报告：关键路径预测与优化

> 日期：2026-05-25
> 主题：从 OpenClaw trace 中识别 agent workflow 关键路径，并生成服务于 SGLang 的调度/cache 信号

---

## 1. 问题定义

方向 B 的核心问题是：agent task 的端到端 latency 通常不是所有节点平均决定的，而是由慢 tool、fan-in 等待、关键 LLM step 组成的关键路径决定。OpenClaw 能观察到这些节点和边，但 SGLang 只看到单个 LLM 请求。因此 B 的主要任务是把 OpenClaw trace 转化为 serving 层可消费的少量高价值信号：

- `critical_path_rank`
- `expected_next_tool`
- `expected_tool_return_time_ms`
- `fan_in_wait_group`
- `speculation_candidate`
- `kv_prefetch_deadline_ms`

B 的输出会反向输入方向 A：scheduler 可以优先处理关键路径上的 LLM 请求，cache manager 可以提前 load-back/prefetch 即将被关键路径重用的 KV。

---

## 2. 相关论文与开源实现

| 系统 | 机构 | 论文/代码状态 | 核心思想 | 对本方向的启发 | 主要限制 |
|------|------|---------------|----------|----------------|----------|
| PASTE | Shanghai Jiao Tong University; Microsoft Research; Stevens Institute of Technology | [arXiv:2603.18897](https://arxiv.org/abs/2603.18897)，本轮未找到官方代码 | Pattern-Aware Speculative Tool Execution，通过 pattern 预测 tool call 并提前执行 | Pattern Tuple、投机命中/浪费、tool throughput 和 task latency 的权衡 | 需要强安全边界；论文收益不能直接套用到 OpenClaw |
| B-PASTE | Independent Researcher | [arXiv:2604.16469](https://arxiv.org/abs/2604.16469)，本轮未找到可审计官方代码 | 面向更宽 agent 场景的 beam-aware speculative tool execution | 说明 speculation 正在扩展到 branch/subgraph 级别 | 缺少代码细节，不能作为实现模板 |
| Autellix | UC Berkeley; Google DeepMind; Shanghai Jiao Tong University | [arXiv:2502.13965](https://arxiv.org/abs/2502.13965)，本轮未找到官方代码 | program-aware scheduling，按 program progress/preemption 降低端到端延迟 | 可作为 critical path / workflow head-of-line blocking 的调度参考 | 不做 tool speculation，也没有代码可直接审计 |
| Halo | National University of Singapore | [arXiv:2509.02121](https://arxiv.org/abs/2509.02121)，[Halo_demo](https://github.com/mlsys-io/Halo_demo) | DAG 上计算 distance/criticality，按依赖和 cache locality 调度 | critical distance 可作为 critical_path_rank 的结构特征 | 假设 workflow 显式可声明 |
| ScaleSim | University of California, San Diego; Amazon Web Services | [arXiv:2601.21473](https://arxiv.org/abs/2601.21473)，[KVFlow repo](https://github.com/PanZaifeng/KVFlow) | 用 invocation distance 预测 agent 未来调用顺序，驱动 memory eviction/prefetch | 可作为 `expected_reuse_window_ms` / `kv_prefetch_deadline_ms` 的系统参考 | 面向 simulation，不处理 OpenClaw 的 tool side effect 和 replay safety |
| Conveyor | Duke University | [arXiv:2406.00059](https://arxiv.org/abs/2406.00059)，[Conveyor repo](https://github.com/conveyor-sys/conveyor) | tool partial execution，把可部分执行的 tool 与 LLM decoding overlap | 对 read-only / parser-friendly tool 的 overlap 接口有参考价值 | 需要 tool 开发者暴露 partial execution；不能泛化到 mutating tool |
| APIServe / InferCept | University of California, San Diego | [arXiv:2402.01869](https://arxiv.org/abs/2402.01869)，[InferCept repo](https://github.com/WukLab/infercept) | API/tool interception 期间管理 KV preserve/discard/swap | 可作为 tool wait window 与 KV residency 的代码参考 | 不处理 OpenClaw 风格的 side-effect governance |
| LAMPS / MARS | Harvard University; Tsinghua University | [arXiv:2410.18248](https://arxiv.org/abs/2410.18248)，[MARS repo](https://github.com/mars-repository/mars-codebase) | API-augmented request scheduling，按 memory-over-time 和 handling strategy 排序 | 可把 `expected_reuse_window_ms` 扩展为调度 cost | 仍是 augmented API workload，不是完整 agent graph |
| Teola / Ayo | The Chinese University of Hong Kong; Unaffiliated | [arXiv:2407.00326](https://arxiv.org/abs/2407.00326)，[Ayo](https://github.com/NetX-lab/Ayo) | topology-aware scheduling、node depth、stage decomposition | node depth、ready frontier、DAG primitive 可用于 trace 后处理 | 更适合静态任务图 |
| Parrot | Shanghai Jiao Tong University; Microsoft Research | [paper](https://www.usenix.org/conference/osdi24/presentation/lin-chaofan)，[ParrotServe](https://github.com/microsoft/ParrotServe) | semantic variable graph 和 completion chain | request chain 可作为 LLM step 依赖建模参考 | 要求应用层显式使用 semantic variable API |

---

## 3. PASTE/B-PASTE 的适配判断

PASTE 的方向非常符合 B：通过历史 pattern 预测将来的 tool call，在模型真正发出 tool call 前提前执行，从而缩短 LLM-output 到 tool-result 的等待时间。论文报告了显著 task completion time 缩短和 tool throughput 提升。

但在 OpenClaw 中不能直接照搬：

- OpenClaw 的 tool 可能有副作用，包括修改文件、执行命令、发起外部写操作、改变 session state。
- OpenClaw 已有 replay invalidation 和 mutating action 标记，这些状态必须高于 speculation 的收益目标。
- user-visible 状态、streaming 输出和 tool result provenance 需要保持可审计。
- PASTE/B-PASTE 当前没有找到官方代码，因此缺少可直接复用的 pattern matcher、confidence estimator 和回滚机制。

因此近期设计把 PASTE 降级为三个层次：

1. **Critical path prioritization。** 不提前执行 tool，只用 trace 识别关键路径，把 rank 输入 scheduler。
2. **KV prefetch/warmup。** 在 tool 即将返回时提前 load-back KV；失败只浪费带宽，不改变语义。
3. **Read-only speculation。** 只对显式 read-only、idempotent、cacheable 的 tool 做提前执行，结果进入 shadow cache，只有真实 tool call 匹配时才提交。

### 3.1 补充系统对方向 B 的影响

补充复核后，方向 B 不能只围绕 PASTE/B-PASTE。虽然 PASTE 仍是 “pattern-aware speculative tool execution” 的核心论文，但已有几类可读系统提供了旁证：

- **Conveyor**：把 tool partial execution 和 LLM decoding overlap。它更适合 code/search/validation 这类能边解析边执行的工具；对 OpenClaw 来说，只有 read-only 且 parser-friendly 的 tool 才可能借鉴。
- **InferCept / LAMPS-MARS**：不预测未来 tool call，而是优化已经发生或即将发生的 API interruption。它们适合作为 B 输出到 A 的 bridge：critical path analyzer 估计 tool/API wait，scheduler/cache 负责 KV handling。

AsyncLM 与 Pie 都和 tool/IO 下沉有关，但需要模型协议或 programmable serving runtime 的整体变化，不进入本轮方向 B 近期设计，转入另行规划。

---

## 4. OpenClaw 侧 trace 建模

### 4.1 事件类型

OpenClaw 侧应把 agent workflow trace 建成事件流，而不是一次性要求静态 DAG：

| 事件 | 关键字段 | 用途 |
|------|----------|------|
| `llm_request_start` | workflow/session/run/step/model/provider/cache_scope | LLM 节点开始，连接到 SGLang request |
| `llm_first_token` | request id、TTFT、queue wait | 区分排队、prefill、decode 对关键路径的贡献 |
| `llm_request_end` | output tokens、finish reason、tool call summary | LLM 节点结束，产生 tool dependency |
| `tool_start` | tool name、args hash、tool call id、side effect class | tool 节点开始，判断 speculation eligibility |
| `tool_end` | result hash、duration、error、replay invalidated | tool 节点结束，更新 latency estimator |
| `spawn_start` | parent run、child run、spawn index | fan-out 结构 |
| `spawn_end` | child run result/error、join group | fan-in 结构 |
| `session_end` | outcome、success/failure、total latency | task-level attribution |

### 4.2 图重建

图重建不要求完整静态计划，只需要逐步维护局部 DAG：

- LLM 输出 tool call：`llm_step -> tool_step`
- tool result 进入下一轮 LLM：`tool_step -> next_llm_step`
- sub-agent spawn：`parent_llm/tool_step -> child_session_start`
- fan-in：`child_session_end -> parent_join_step`

每条边记录：

- 控制依赖：必须等待还是可并行。
- 数据依赖：是否把 tool result / child summary 拼入 prompt。
- 安全属性：read-only、mutating、user-visible、requires approval。
- latency statistics：EWMA、p50/p95、失败率。

### 4.3 Critical path rank

初期可以使用在线近似，而不是离线全图算法：

1. 对每个 node/edge 维护 duration EWMA 和 p95。
2. 对当前 active workflow 维护从 root 到当前 frontier 的最长路径估计。
3. 对 ready 或即将 ready 的 LLM step 计算 slack：`deadline - estimated_remaining_path`。
4. 将 rank 归一化为 `[0, 1]`，写入 request-time `agent_hints.critical_path_rank`。

这个 rank 不追求完美预测，只需要比 FIFO 更好地区分 fan-in 恢复、慢 tool 后续 LLM、orchestrator 汇合等关键节点。

---

## 5. Read-only speculation 设计

### 5.1 Eligibility

只有同时满足以下条件的 tool call 可以进入 speculation：

- tool 被显式标记为 read-only。
- tool 是 idempotent，可重复执行。
- tool result 可缓存，且有稳定 args hash。
- 执行不会产生用户可见消息、文件写入、命令副作用或外部写操作。
- 当前 session 没有 replay invalidation 或 mutating action barrier。
- pattern 命中率和 confidence 超过阈值。

默认不进入 speculation 的类别：

- shell/command execution。
- file write / patch / delete。
- network write / ticket update / PR comment / issue mutation。
- 需要用户确认的操作。
- 依赖当前未完成 LLM 输出自由文本的参数。

### 5.2 Pattern Tuple

借鉴 PASTE，但保守化为 OpenClaw 可审计结构：

```json
{
  "pattern_id": "research-synthesis.search-then-read:v1",
  "control_flow": ["search", "read"],
  "data_slots": [
    {"name": "query", "source": "user_goal", "stability": "medium"},
    {"name": "path", "source": "search_result", "stability": "low"}
  ],
  "risk_profile": {
    "side_effect": "none",
    "idempotent": true,
    "cacheable": true,
    "hit_rate": 0.74,
    "waste_rate": 0.18,
    "max_cost_ms": 800
  }
}
```

### 5.3 Shadow result cache

Speculation 结果不能直接写入真实 session state。它应进入 shadow result cache：

- key：`workflow_id + pattern_id + tool_name + args_hash + trace_epoch`
- value：tool result、duration、source、timestamp、confidence、provenance
- commit 条件：真实 LLM 输出的 tool call 与 tool name/args hash 匹配
- discard 条件：真实 tool call 不匹配、session state invalidated、TTL 过期、用户/系统 barrier 出现

---

## 6. SGLang 侧消费方式

SGLang 不参与 tool speculation 的执行，也不判断 tool 是否安全。它只消费 B 的 serving-facing 输出：

- `critical_path_rank`：scheduler 排序因子。
- `expected_tool_return_time_ms`：KV lifecycle/prefetch deadline。
- `fan_in_group_id`：帮助识别 orchestrator 汇合恢复请求。
- `expected_reuse_window_ms`：cache manager 判断 KV 是否应该保留、下沉或提前 load-back。

这保持了责任边界：

- OpenClaw：理解 agent graph、tool 安全、speculation。
- SGLang：理解 request queue、batch、prefix cache、KV residency。

---

## 7. 假设 mismatch

| 层面 | PASTE/Halo/Ayo 常见假设 | OpenClaw + SGLang 现实 | 适配策略 |
|------|-------------------------|------------------------|----------|
| workflow graph | 可从 workload 或 DSL 得到较完整 DAG | OpenClaw 动态 agent loop，图在运行时展开 | 用事件流增量建图，输出局部 criticality |
| tool safety | tool 可提前执行或有低成本回滚 | tool 可能有副作用，回滚不可泛化 | 只对显式 read-only/idempotent/cacheable tool speculation |
| result commit | 预测失败可丢弃 | OpenClaw 有用户可见状态和 replay invalidation | speculation result 进入 shadow cache，匹配后提交 |
| engine 角色 | serving engine 可能参与整图优化 | SGLang 不执行 tool，也不应理解完整 agent graph | SGLang 只消费 rank/deadline/cache hint |
| pattern 稳定性 | benchmark pattern 可重复 | 真实研究/编码任务参数变化大 | 先做 trace mining 和 prioritization，speculation 作为后续门控功能 |

---

## 8. 近期设计建议

### 8.1 先做 critical path，不先做 speculation

优先级：

1. trace schema 和 DAG reconstruction。
2. critical_path_rank 计算。
3. 把 rank 输入方向 A 的 scheduler。
4. 观察 fan-in/orchestrator 恢复请求的 latency 是否下降。
5. 再进入 read-only speculation。

### 8.2 Speculation 必须由安全 registry 驱动

OpenClaw 需要一个 tool metadata registry：

| 字段 | 含义 |
|------|------|
| `side_effect_class` | `none` / `read` / `write` / `external_mutation` |
| `idempotent` | 是否可重复执行 |
| `cacheable` | result 是否可缓存 |
| `user_visible` | 执行是否产生用户可见输出 |
| `requires_approval` | 是否需要用户批准 |
| `max_speculation_cost_ms` | 最大允许浪费成本 |

没有 registry 标记的 tool 默认不投机。

### 8.3 与方向 A 的接口

B 输出到 A 的最小接口：

```json
{
  "critical_path_rank": 0.86,
  "fan_in_group_id": "join-17",
  "expected_next_phase": "orchestrator_resume",
  "expected_reuse_window_ms": 1200,
  "kv_prefetch_deadline_ms": 300,
  "speculation": {
    "eligible": false,
    "reason": "tool_has_side_effect"
  }
}
```

ScaleSim 对这个接口的补充意义是：`expected_reuse_window_ms` 不一定只来自 tool duration，也可以来自更泛化的 invocation distance。对 OpenClaw 来说，这个 distance 可以由 trace analyzer 综合估计：

- tool wait：当前 tool 预计还有多久返回。
- fan-in：子 agent 预计还有多久完成。
- user-visible foreground task：是否应压缩排队等待。
- session idle probability：该 session 是否可能短期内继续调用 LLM。

这些信号进入 SGLang 后，仍然只应作为 eviction/prefetch priority，而不是让 SGLang 理解完整 agent graph。

---

## 9. 初步可行性结论

方向 B 可行，但最小可行形态不是直接复现 PASTE，而是 **critical path analysis + conservative read-only speculation**。

短期收益最稳的是 prioritization：它只改变调度优先级，不改变 agent 语义。KV prefetch/warmup 次之：失败成本是额外带宽。read-only speculation 最后做，并且必须由 OpenClaw 的 tool 安全元数据、shadow cache 和 provenance 机制约束。
