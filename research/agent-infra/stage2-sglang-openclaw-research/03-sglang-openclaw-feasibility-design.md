# SGLang + OpenClaw 可行性分析与详细设计

> 日期：2026-05-25
> 定位：Stage 2 近期主线的最终设计稿
> 范围：方向 A 图感知推理请求调度 + 方向 B 关键路径预测与优化

---

## 1. 总体可行性结论

OpenClaw + SGLang 上推进 Stage 2 近期两条主线是可行的，但实现边界需要清楚：

- **OpenClaw 侧负责 agent graph/control plane。** 它产生 workflow/session/step、tool lifecycle、side effect、sub-agent fan-out/fan-in、critical path 和 speculation safety 信息。
- **SGLang 侧负责 serving decision plane。** 它消费 request-time hints 和 lifecycle events，把它们转化为 queue ordering、batch composition、KV residency、load-back/prefetch priority。
- **两者之间新增 signal plane。** 近期只传 serving 决策需要的 typed hints，不把完整 agent DAG 或 tool execution runtime 下沉到 SGLang。

方向 A 是近期主要工程落点：typed `agent_hints`、AgentStateManager、agent-aware cache-aware scheduler、KV lifecycle hints。Autellix 是 program-aware scheduling 的核心论文参照，KVFlow/ScaleSim 的 SGLang fork 是 cache lifecycle 的首要代码参照：前者说明 program/workflow 应成为一等调度对象，后者已经把外部 timestep / invocation-distance signal 接到 SGLang cache eviction/prefetch 路径。补充复核后，InferCept 与 LAMPS/MARS 也应作为 vLLM 侧 API/tool interruption 的代码对照。Pie/Symphony/ALTO/AsyncLM/Agent.xpu 不进入本轮近期两方向详细设计，另行规划。方向 B 先以 trace-driven critical path rank 服务方向 A，再进入 read-only tool speculation。

---

## 2. 系统分层

```text
OpenClaw agent runtime
  - agent loop / tool lifecycle / sub-agent spawn / replay safety
  - trace DAG reconstruction
  - critical path analyzer
  - read-only speculation controller

Signal plane
  - request-time agent_hints
  - lifecycle event endpoint
  - metrics / trace feedback

SGLang inference serving
  - OpenAI-compatible request path
  - tokenizer/scheduler Req metadata
  - AgentStateManager
  - agent-aware cache-aware scheduling policy
  - optional Autellix-style program progress to scheduling priority mapping
  - Radix/HiRadix/CacheController lifecycle adaptation
  - optional KVFlow/ScaleSim-style reuse-distance to eviction/prefetch priority mapping
  - optional InferCept/MARS-style API/tool wait memory cost model
```

设计原则：

1. 完整 workflow graph 不进入 SGLang scheduler 热路径。
2. tool safety 不由 SGLang 判断。
3. cache namespace 与 workflow identity 分离。
4. 新策略必须能与现有 `fcfs`、`lpm`、`dfs-weight`、`routing-key` 做 ablation。
5. Phase 1 必须先完成无行为变化的观测闭环。

---

## 3. Signal Contract

### 3.1 Request-time `agent_hints`

建议新增字段：

```json
{
  "agent_hints": {
    "schema_version": "agent-hints.v1",
    "workflow_id": "wf-20260525-001",
    "session_id": "sess-main",
    "agent_run_id": "run-orchestrator-7",
    "step_id": "llm-14",
    "parent_step_id": "tool-13",
    "role": "orchestrator",
    "phase": "orchestrator_resume",
    "fan_in_group_id": "join-3",
    "critical_path_rank": 0.86,
    "expected_reuse_window_ms": 1200,
    "kv": {
      "cache_scope": "shared_prompt_policy_v1",
      "expected_reuse": "short",
      "ttl_ms": 2000,
      "prefetch_deadline_ms": 300
    },
    "safety": {
      "last_tool_side_effect": "none",
      "speculation_eligible": false
    }
  }
}
```

字段边界：

- `workflow_id/session_id/step_id` 用于 agent state 和 observability，不进入 prefix cache key。
- `critical_path_rank` 是 scheduler hint，不直接触发 preemption。
- `kv` 是 advisory hint，cache manager 可根据 HBM pressure 忽略。
- `expected_reuse_window_ms` 可以被离散化为 KVFlow/ScaleSim 风格的 invocation distance / timestep priority，用于 eviction 和 prefetch 队列排序。
- `cache_scope` 可用于生成 `extra_key`，但必须只包含影响 cache sharing 正确性的因素。

### 3.2 Lifecycle event

request 之间发生的状态变化不能只靠下一次 LLM request 携带，因此需要 out-of-band endpoint：

```http
POST /v1/agent/session_event
```

```json
{
  "schema_version": "agent-session-event.v1",
  "workflow_id": "wf-20260525-001",
  "session_id": "sess-main",
  "agent_run_id": "run-orchestrator-7",
  "step_id": "tool-13",
  "event": "tool_start",
  "timestamp_ms": 1779700000000,
  "tool": {
    "name": "search",
    "args_hash": "sha256:...",
    "side_effect": "none",
    "expected_duration_ms": 900
  },
  "kv": {
    "related_llm_step_id": "llm-12",
    "expected_reuse_window_ms": 900,
    "residency_hint": "may_offload"
  }
}
```

event 类型：

- `llm_request_start`
- `llm_request_end`
- `tool_start`
- `tool_end`
- `tool_error`
- `spawn_start`
- `spawn_end`
- `session_end`

---

## 4. SGLang 详细设计

### 4.1 Request schema 传播

目标文件：

- `python/sglang/srt/managers/io_struct.py`
- `python/sglang/srt/entrypoints/openai/protocol.py`
- `python/sglang/srt/entrypoints/openai/serving_chat.py`
- `python/sglang/srt/entrypoints/openai/serving_completions.py`
- `python/sglang/srt/managers/tokenizer_manager.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/managers/scheduler.py`

改动：

- 增加 `AgentHints` dataclass 或 typed dict。
- `GenerateReqInput`、`TokenizedGenerateReqInput`、`Req` 均携带 `agent_hints`。
- OpenAI protocol 允许 extra body 中的 `agent_hints`，并进行 schema/version/basic type 校验。
- `custom_labels` 仍只用于 metrics/observability，不承担调度语义。

### 4.2 AgentStateManager

新增文件：

- `python/sglang/srt/managers/agent_state_manager.py`

职责：

- 维护 workflow/session/step 的轻量状态。
- 消费 request-time hints 和 lifecycle events。
- 对 scheduler/cache 提供只读查询接口。
- 做 TTL GC，避免 agent state 无界增长。

状态示例：

```python
@dataclass
class AgentSessionState:
    workflow_id: str
    session_id: str
    last_step_id: str | None
    phase: str
    critical_path_rank: float
    expected_reuse_window_ms: int | None
    fan_in_group_id: str | None
    last_update_monotonic: float
```

接口示例：

```python
class AgentStateManager:
    def update_from_request(self, req_id: str, hints: AgentHints) -> None: ...
    def update_from_event(self, event: AgentSessionEvent) -> None: ...
    def get_request_rank(self, req: Req) -> float: ...
    def get_cache_lifecycle_hint(self, req: Req) -> CacheLifecycleHint | None: ...
    def gc(self) -> None: ...
```

### 4.3 Agent-aware cache-aware scheduler policy

不要把主路线建在现有 `routing-key` policy 上，因为它当前是 cache-agnostic。建议新增 policy：

- `agent-aware-lpm`
- 或 `workflow-prefix-composite`

排序逻辑：

1. 复用现有 cache-aware prefix match 计算。
2. 计算 prefix score：LPM length 或 DFS_WEIGHT score。
3. 计算 agent score：critical_path_rank、fan-in resume、expected tool return、workflow fairness。
4. 组合排序：先保护明显更高 prefix locality，再在相近 prefix locality 内按 agent score 排序。
5. 加入 fairness guard，避免高 criticality workflow 长时间压制普通请求。

伪代码：

```python
for req in waiting_queue:
    prefix_score = req.cached_tokens_or_dfs_weight
    agent_score = agent_state_manager.get_request_rank(req)
    fairness_penalty = workflow_fairness_penalty(req)
    req.agent_aware_score = (
        prefix_weight * normalize(prefix_score)
        + critical_weight * agent_score
        - fairness_weight * fairness_penalty
    )

waiting_queue.sort(key=lambda req: req.agent_aware_score, reverse=True)
```

关键点：

- prefix match 必须先算；agent score 不能让明显无 prefix locality 的请求破坏高命中 batch。
- `critical_path_rank` 不等于 priority preemption，初期只影响 waiting queue order。
- 所有权重通过 server args 配置，默认保守。

### 4.4 KV lifecycle hints

SGLang 的 Radix/HiRadix/CacheController 已经有多层缓存与 async operation 基础。近期适配建议：

- tool_start：如果 expected tool duration 高且 HBM pressure 高，对相关 session cache node 降低 GPU residency 或触发 write-through/backup。
- tool_end：如果 next LLM step 预计即将到达，对 best_match/host/storage cache 做 load-back/prefetch。
- critical path：提高相关 load/prefetch operation priority。
- session_end：降低相关 session 的 residency priority，并允许正常 eviction。

不建议近期直接把完整 workflow id 写入 TreeNode。TreeNode 保持 cache-native 状态，AgentStateManager 保存 agent state，并在决策时动态查询。

### 4.5 Control endpoint

目标文件：

- `python/sglang/srt/entrypoints/openai/api_server.py`
- `python/sglang/srt/managers/io_struct.py`
- scheduler/control message 相关结构

设计：

- `/v1/agent/session_event` 接收 OpenClaw lifecycle event。
- API 层做 schema 校验和限流。
- event 进入 scheduler process 的 AgentStateManager。
- endpoint 默认关闭或需要显式 flag，例如 `--enable-agent-state-manager`。

---

## 5. OpenClaw 详细设计

### 5.1 通用 trace 层

目标文件：

- `src/agents/pi-embedded-subscribe.handlers.tools.ts`
- `src/agents/acp-spawn.ts`
- agent run/session trace 相关模块

改动：

- 在 tool start/end 处发出通用 trace event。
- 在 spawn start/end 处发出 fan-out/fan-in event。
- LLM request start/end 记录 provider/model/request id/custom labels。
- trace event 不依赖 SGLang，避免污染 OpenClaw core。

### 5.2 SGLang provider adapter

目标文件：

- `extensions/sglang/index.ts`
- `extensions/sglang/api.ts`
- 可新增 `extensions/sglang/stream.ts`
- 可新增 `extensions/sglang/agent-hints.ts`

改动：

- 基于 provider stream wrapper，在 request payload 中注入 `agent_hints`。
- 生成 `custom_labels` 用于 Phase 1 observability。
- 根据 cache sharing policy 生成 `extra_key`，但不包含 workflow/run id。
- 如需 `x-smg-routing-key` 对照实验，增加 header patch 能力或 provider-specific request adapter。
- lifecycle event 由 adapter 发送到 SGLang `/v1/agent/session_event`，失败时不影响主 LLM request。

### 5.3 Critical path analyzer

组件：

- trace event collector。
- online DAG builder。
- latency estimator。
- critical path rank calculator。
- hint cache。

输出：

- 写入下一次 LLM request 的 `agent_hints`。
- 给 SGLang control endpoint 发送 lifecycle event。
- 给 speculation controller 提供 pattern 候选。

### 5.4 Read-only speculation controller

只在 Phase 4 启用：

- tool metadata registry 标记 side effect、idempotent、cacheable、requires approval。
- pattern miner 根据历史 trace 生成 Pattern Tuple。
- speculation executor 只执行 read-only tools。
- result 写入 shadow cache。
- 真实 tool call 匹配后提交，不匹配则丢弃。

---

## 6. 评估方案

### 6.1 Baseline matrix

| 组别 | OpenClaw | SGLang |
|------|----------|--------|
| B0 | 无 hints | `fcfs` |
| B1 | 无 hints | `lpm` |
| B2 | 无 hints | `dfs-weight` |
| B3 | workflow routing key | `routing-key` |
| B4 | labels only | `dfs-weight` |
| E1 | `agent_hints` critical rank | agent-aware cache-aware policy |
| E2 | `agent_hints` + lifecycle | agent-aware policy + KV lifecycle |
| E3 | critical path + read-only speculation | E2 + OpenClaw speculation |

### 6.2 Workload pools

- 线性多轮 agent：稳定 session prefix，弱 fan-out。
- tool wait workload：search/read/query 等 read-only tool 延迟可控。
- fan-out/fan-in research workload：orchestrator 派生多个 sub-agent 后汇合。
- mixed foreground/background：前台交互任务和后台长任务混合。
- mutation-heavy workload：用于验证 speculation safety，不能投机但应保留 trace。

### 6.3 Metrics

端到端：

- task completion latency p50/p95/p99。
- success rate / answer quality guardrail。
- foreground request latency。

SGLang：

- queue wait。
- TTFT。
- prefill/decode time。
- prefix hit tokens。
- host/storage hit tokens。
- eviction count。
- load-back/prefetch count and latency。
- HBM/CPU/storage bandwidth pressure。

OpenClaw：

- tool wait duration。
- fan-in wait duration。
- critical path contribution。
- speculation hit/waste/abort。
- replay invalidation / mutating barrier count。

---

## 7. 风险与缓解

| 风险 | 表现 | 缓解 |
|------|------|------|
| `extra_key` 误用破坏 prefix sharing | cache hit 大幅下降 | workflow id 不进入 `extra_key`；cache scope 白名单化 |
| `routing-key` 掩盖 prefix-aware baseline | routing-key 实验看似收益但 prefix hit 变差 | routing-key 只做对照，主路线新增 cache-aware composite |
| AgentStateManager 状态泄漏或过期 | scheduler 使用 stale hints | TTL GC、event version、session_end 清理、fallback to prefix-only |
| critical rank 预测不稳定 | 关键请求优先级抖动 | rank smoothing、低权重起步、ablation |
| KV prefetch 浪费带宽 | host/storage 带宽争用 | pressure-aware gating、deadline/priority、失败成本统计 |
| speculation 产生副作用 | 用户状态或外部系统被错误修改 | 默认关闭，只允许 registry 标记 read-only/idempotent/cacheable tool |
| OpenClaw core 绑定 SGLang | provider 边界污染 | core 只发通用 trace，SGLang extension 做 adapter |

---

## 8. 实施里程碑

### M0：文档与基线确认

- 完成两条方向的研究报告。
- 完成 SGLang/OpenClaw 代码路径 mapping。
- 明确字段边界和风险。

### M1：无行为变化观测

- OpenClaw 产出通用 trace event。
- SGLang request 携带 `custom_labels`。
- 收集 baseline matrix B0-B4。

### M2：typed hints 闭环

- SGLang request path 支持 `agent_hints`。
- OpenClaw SGLang extension 注入 hints。
- SGLang metrics 能按 hints 聚合。

### M3：scheduler/cache 实验

- AgentStateManager。
- agent-aware cache-aware scheduler。
- lifecycle event endpoint。
- KV lifecycle hint 实验。

### M4：critical path 与 read-only speculation

- OpenClaw critical path analyzer。
- tool metadata registry。
- shadow result cache。
- read-only speculation gated experiment。

---

## 9. 最小可行版本

最小可行版本应该控制在以下范围：

1. OpenClaw 发 trace 和 `custom_labels`。
2. SGLang 保持 `dfs-weight` 或 `lpm` baseline。
3. 增加 typed `agent_hints`，只传 `workflow_id/session_id/step_id/critical_path_rank/cache_scope`。
4. 新增 agent-aware cache-aware policy，但默认权重保守。
5. 不做 speculation，只做 critical path prioritization。

这个版本能验证最核心的问题：agent graph signal 进入 serving 后，是否在不破坏 prefix cache 的情况下改善 queue wait、TTFT 和 task completion latency。如果最小版本没有收益，再进入 KV lifecycle 或 speculation 的优先级就应下降。
