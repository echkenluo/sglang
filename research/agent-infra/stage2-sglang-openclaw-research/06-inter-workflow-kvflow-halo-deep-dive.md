# Inter-Workflow Deep Dive: KVFlow and Halo

> 日期：2026-05-25
> 范围：Stage 2 近期方向 A 中，跨 workflow / 多任务 agent workload 的资源优化论文与开源实现分析

---

## 1. 先给结论

在当前 Stage 2 近期范围里，**KVFlow 和 Halo 是 inter-workflow 资源优化的两篇主锚点论文**。但二者不是同一种问题：

- **KVFlow** 是 workflow-aware KV lifecycle。它面向多个并发 agent workflow，把“某个 agent / step 距离下一次执行还有多远”转成 KV cache node 的 eviction priority 和 prefetch window。它处理的是跨 workflow 的缓存驻留、竞争和部分 prefix 共享，不是把多个 workflow 的完整计算图合并成一张图。
- **Halo** 更接近严格意义的 multi-workflow compute graph optimization。它把多个 workflow/query instance 建成 operator DAG，对 batch 内多个 workflow 做跨 workflow batching、共享计算、KV cache 共享/迁移和 device locality 调度。

所以，如果问题是：

> 只有 KVFlow 和 Halo 主要考虑多 workflow task 计算图的重合资源优化吗？

答案需要拆开：

- **在本轮近期研究集合里，是的，KVFlow + Halo 是 inter-workflow 方向的两个主要锚点。**
- **但严格说“多 workflow task 计算图的重合 / 合并 / 共享执行”主要是 Halo。** KVFlow 的“重合”落在 prefix KV cache tree node 和 agent reuse distance 上，不是完整 workflow DAG 的 operator-level common subgraph。
- 其他论文可以作为对照，但不是这个近期问题的主轴：Parrot 有跨请求 semantic/context sharing，Autellix 有 program-level scheduling，InferCept/MARS/Continuum 有 tool/API wait 期间的 KV lifecycle，Ayo 是单个 task/primitive DAG 形态更强的 optimizer。它们都不是“多 workflow task 计算图重合资源优化”的直接主线。

---

## 2. Inter-Workflow 的定义边界

这里要避免两个概念混在一起：

| 概念 | 含义 | 例子 | 是否等于 inter-workflow |
|------|------|------|--------------------------|
| single workflow 内部的 multi-agent | 一个用户 task 内 fan-out 出多个 agent，再 fan-in 汇总 | research task 中多个子 agent 分别查不同资料 | 不一定 |
| inter-workflow | 多个 task / run / session / workflow instance 同时执行，并在 serving/resource 层共享或竞争资源 | 多个 OpenClaw task 同时跑，每个 task 都有 planner、researcher、summarizer | 是 |
| compute graph overlap | 不同 workflow 的 DAG/operator/prompt/tool step 存在可共享计算或可合批执行的部分 | 多个 workflow 都执行同一类 summarization op，或共享系统 prompt/prefix | inter-workflow 的一种强形式 |
| KV lifecycle overlap | 不同 workflow 的未来调用距离、shared prefix、cache residency 发生竞争 | 某些 agent 的 prefix 很快会复用，因此不应被 LRU 赶出 HBM | inter-workflow 的缓存形式 |

在 OpenClaw + SGLang 的近期目标里，最现实的入口不是让 SGLang 直接理解完整 agent DAG，而是让 OpenClaw 生成 workflow/session/tool lifecycle/reuse-window hints，SGLang 在 scheduler/cache 层消费这些 serving-relevant signals。

---

## 3. 论文定位对照

| 系统 | 是否 inter-workflow | 共享/优化对象 | 是否做完整计算图重合 | 开源实现形态 | 对 Stage 2 近期价值 |
|------|---------------------|----------------|------------------------|--------------|---------------------|
| KVFlow | 是 | Agent Step Graph、steps-to-execution、KV tree node priority、prefetch window | 部分。共享对象主要是 prefix/KV cache node，不是完整 operator DAG | SGLang fork + SScheduler 中间层 | SGLang 侧 KV lifecycle 的最直接代码参考 |
| Halo | 是 | batched workflow/query DAG、operator placement、shared computation、KV movement、device locality | 是。最接近跨 workflow compute graph consolidation | Python demo runtime + scheduler + worker prototype | OpenClaw 侧 graph optimizer / SGLang hints 的结构参考 |
| Parrot | 部分 | semantic variable、context manager、ServeCore graph/context scheduling | 有跨请求 context sharing，但应用需改写到 Parrot API | 自定义 serving runtime | 作为 semantic/context-aware serving 对照，不做近期主线 |
| Autellix | 是 | program-level progress、request scheduling、HoL blocking | 否。更偏 program-aware scheduling | 未找到官方开源 | 证明 workflow/program 应作为调度对象，但无法代码复用 |
| InferCept / MARS / Continuum | 部分 | tool/API wait 期间 KV preserve/discard/swap/pin | 否 | vLLM fork/patch | tool wait KV lifecycle 对照 |
| Ayo / Teola | 主要是 intra-workflow | primitive DAG、node depth、prefill/decode split | 偏单 task DAG 优化 | 自定义 runtime/prototype | OpenClaw trace analyzer 的参考，不是跨 workflow 主线 |

---

## 4. KVFlow 深入分析

### 4.1 论文到底解决什么

KVFlow 的核心抽象是 **Agent Step Graph** 和 **steps-to-execution**。它观察到 agent workflow 里的 prompt prefix/KV 会被周期性或阶段性复用，而传统 LRU 只看最近访问时间，无法判断一个暂时没被访问的 agent 是否马上就会再次执行。

论文的关键点是：

- workflow 层知道 agent 之间的依赖和未来 step 距离；
- serving engine 的 Radix/HiRadix cache 只知道 token prefix 和 cache node；
- KVFlow 在二者之间传递 agent timestep / future reuse signal；
- cache eviction 时优先保护 steps-to-execution 更小的 agent prefix；
- prefetch 时把即将执行的 agent 相关 KV 从 CPU/host 拉回 GPU。

这确实是 inter-workflow，因为论文和实现都考虑多个 workflow 并发时的 cache competition。但它不是 graph optimizer：它不会把 workflow A 和 workflow B 的 DAG 合并，也不会对 tool/operator 做 common subexpression elimination。

### 4.2 代码是怎么实现的

KVFlow repo 中有两层：

- `SScheduler`：外部 workflow/simulation mid-layer，维护 agent 到 timestep 的映射。
- `python/sglang`：SGLang fork，在 scheduler/cache 层消费 timestep update。

关键链路如下：

1. `SScheduler` 收集 manager 的未来 timestep 信息，并按 policy 整合成 `{agent_id: timestep}` 和 `{timestep: [agent_id]}`。代码入口是 [Scheduler.py](/home/luocc4/workspace/agent-infra-paper-code/KVFlow/SScheduler/Scheduler.py:129)。
2. `SScheduler` 通过 HTTP `POST /v1/update` 把 `agent_data`、`timestep_data`、`timestep_cnt` 发给 SGLang fork。发送逻辑在 [Scheduler.py](/home/luocc4/workspace/agent-infra-paper-code/KVFlow/SScheduler/Scheduler.py:167)。
3. SGLang fork 新增 `/v1/update` endpoint，把 update 交给 tokenizer manager / scheduler control path。入口在 [http_server.py](/home/luocc4/workspace/agent-infra-paper-code/KVFlow/python/sglang/srt/entrypoints/http_server.py:1084)。
4. scheduler 注册 `UpdateAgentTimestepReq` handler，收到后调用 `AgentManager.update_agent_timestep`、刷新 cache leaf priority，并触发 prefetch。handler 注册在 [scheduler.py](/home/luocc4/workspace/agent-infra-paper-code/KVFlow/python/sglang/srt/managers/scheduler.py:584)，处理逻辑在 [scheduler.py](/home/luocc4/workspace/agent-infra-paper-code/KVFlow/python/sglang/srt/managers/scheduler.py:2825)。
5. `AgentManager` 维护 `hold_step`、`prefetch_step`、`agent_to_last_nodes`、`update_dict_agent` 和 `update_dict_timestep`。核心状态在 [agent_manager.py](/home/luocc4/workspace/agent-infra-paper-code/KVFlow/python/sglang/srt/managers/agent_manager.py:9)。
6. RadixCache 在请求命中/插入后把 `agent_id -> last_nodes` 记录下来，并根据 update dict 刷新 leaf node 的 `hold_priority`。相关逻辑在 [radix_cache.py](/home/luocc4/workspace/agent-infra-paper-code/KVFlow/python/sglang/srt/mem_cache/radix_cache.py:634) 和 [radix_cache.py](/home/luocc4/workspace/agent-infra-paper-code/KVFlow/python/sglang/srt/mem_cache/radix_cache.py:684)。
7. `DispatchLoadTasks` 按 timestep window 预取 LoRA 和 KV；KV 预取通过 `agent_to_last_nodes` 找到 evicted nodes 并调用 `load_back`。入口在 [scheduler_output_processor_mixin.py](/home/luocc4/workspace/agent-infra-paper-code/KVFlow/python/sglang/srt/managers/scheduler_output_processor_mixin.py:747)。

### 4.3 它的 inter-workflow 性质

KVFlow 的 inter-workflow 性质来自三个点：

- **多个 workflow 可以同时往同一个 SGLang backend 提交 agent timestep update。** 代码用 agent id / timestep dict 汇总，scheduler/cache 看到的是全局 agent priority。
- **多个 workflow 的 KV cache node 在同一个 Radix/HiRadix cache 里竞争 HBM。** 保护哪个 node、驱逐哪个 node，是跨 workflow 的资源决策。
- **共享 prefix 的 node 可以被多个 agent/workflow 复用。** 因此 node priority 需要按多个 agent 的最小 steps-to-execution 来保守计算。

但这仍是 cache-tree 层的资源优化，不是 workflow DAG 层的合并优化。它更像：

```text
workflow runtime signal -> agent reuse distance -> cache node priority/prefetch
```

而不是：

```text
multiple workflow DAGs -> consolidated DAG -> shared operator execution
```

### 4.4 与 OpenClaw / SGLang 的适配判断

KVFlow 对我们最有价值，因为它已经证明了 SGLang fork 中可以接入 agent-aware cache priority/prefetch。但不能照搬它的 timestep API：

- OpenClaw 的 agent loop 是动态的，tool latency、sub-agent fan-in、retry/replay 都会改变未来调用距离。
- OpenClaw 应该输出 `workflow_id`、`session_id`、`agent_role`、`step_index`、`expected_reuse_window_ms`、`critical_path_rank`、`tool_wait_window_ms` 等 typed hints，而不是只输出 timestep。
- SGLang 侧可以借鉴 KVFlow 的 `AgentManager`，但建议改成更通用的 `AgentStateManager` / `WorkflowStateManager`，不要把可变 session state 直接散落到 Radix tree node 上。
- KVFlow 的代码里还需要进一步审计 eviction hot path。比如 `radix_cache.py` 里 `TreeNode.__lt__` 附近仍有 LRU 相关比较逻辑，最终哪些 eviction path 真正消费了 `hold_priority` 需要继续逐路径验证，不能只按 README 推断。

---

## 5. Halo 深入分析

### 5.1 论文到底解决什么

Halo 把 LLM workflow 当成 query plan。它的强假设是：workflow 可以被声明或解析为 operator DAG，多个 query/workflow instance 可以组成 batch，系统可以在 batch 内看见全局 graph structure。

它因此能做比 KVFlow 更强的 inter-workflow 优化：

- 对多个 workflow/query 的 DAG 做 consolidated planning；
- 识别 recurring prompt/operator/tool request；
- 对同类 operator 做跨 workflow batching；
- 对共享 prefix 和中间结果做 KV cache reuse/migration；
- 根据 parent device locality 和 critical distance 选择 placement；
- 在 worker 层显式 dump/resume/send/get cache。

所以 Halo 是本轮里最接近“多 workflow task 计算图的重合资源优化”的论文。

### 5.2 代码是怎么实现的

Halo_demo 是 prototype runtime，不是 SGLang/vLLM patch。它把论文思想拆成 parser、scheduler、optimizer、worker 四层：

1. parser 从 YAML 构建 operator DAG，设置 input/output edges、model config、`keep_cache`，并计算每个 op 到 end op 的最长距离 `max_distance`。入口是 [parser.py](/home/luocc4/workspace/agent-infra-paper-code/Halo_demo/halo/parser.py:49)，distance 计算在 [parser.py](/home/luocc4/workspace/agent-infra-paper-code/Halo_demo/halo/parser.py:144)。
2. heuristic scheduler 从 ready frontier 出发，按 dependency、`max_distance`、device count、data parallel duplication 和 parent-device locality 生成 per-device workflow commands。策略说明和主逻辑在 [heuristic_t.py](/home/luocc4/workspace/agent-infra-paper-code/Halo_demo/halo/schedulers/heuristic_t.py:5)。
3. scheduler 会在跨 device dependency 处插入 `get_cache` / `send_cache`，在非 ready dependency 处插入 `resume_cache`，在 op 执行后按 data-parallel/end-op 情况做 cache merge/complete。相关命令生成在 [heuristic_t.py](/home/luocc4/workspace/agent-infra-paper-code/Halo_demo/halo/schedulers/heuristic_t.py:157)。
4. optimizer 执行 dependency-aware dispatch：`execute` 任务只有在每个 query 的 parent op output 都存在时才发给 worker。调度入口在 [halo_t.py](/home/luocc4/workspace/agent-infra-paper-code/Halo_demo/halo/optimizers/halo_t.py:91)，dependency guard 在 [halo_t.py](/home/luocc4/workspace/agent-infra-paper-code/Halo_demo/halo/optimizers/halo_t.py:158)。
5. optimizer 在发送 `execute` 前把 parent op output 拼到 prompt，形成下游 op 的输入。逻辑在 [halo_t.py](/home/luocc4/workspace/agent-infra-paper-code/Halo_demo/halo/optimizers/halo_t.py:143)。
6. worker 侧有显式 cache API：`resume_cache`、`dump_cache`、`get_cache`、`send_cache`、`complete`。代码在 [worker_t.py](/home/luocc4/workspace/agent-infra-paper-code/Halo_demo/halo/workers/worker_t.py:166)。
7. worker 的 `execute` 把 preload/prefill/decode 组合在一个 round 里，并动态调整 decode batch size。入口在 [worker_t.py](/home/luocc4/workspace/agent-infra-paper-code/Halo_demo/halo/workers/worker_t.py:318)。

### 5.3 它的 inter-workflow 性质

Halo 的 inter-workflow 性质比 KVFlow 更强：

- batch 内多个 workflow/query 共享同一张 operator DAG schema；
- 每个 op 对多个 query id 执行，天然有 cross-workflow batching；
- data-parallel duplicate 会把 query ids 分片到多个 worker/device；
- cache 以 `op_id -> query_id -> KV` 的方式保存和迁移；
- downstream op 执行前拼接 parent outputs，相当于 query plan executor。

这更接近：

```text
multiple workflow/query instances -> shared operator DAG -> batched op execution + cache movement
```

### 5.4 代码与论文的差距

Halo_demo 需要谨慎使用。它证明了核心机制，但不是完整生产实现：

- 代码里有 YAML DAG、critical distance、device assignment、KV movement、worker execution；
- 论文中更强的 consolidated graph、operator signature canonicalization、tool coalescing、SLO-aware optimizer，在 demo 中不是全部显式实现；
- worker 基于 Transformers 自定义执行，不是 SGLang/vLLM serving engine patch；
- 因此它更适合指导 OpenClaw 侧 graph optimizer / controller 设计，而不是直接迁移到 SGLang scheduler hot path。

### 5.5 与 OpenClaw / SGLang 的适配判断

Halo 对我们近期的作用应该放在 OpenClaw/control-plane，而不是 engine 内部重写：

- OpenClaw 侧可以学习 Halo 的 operator DAG、frontier、critical distance、parent locality，把真实 task trace 抽象成可分析的 workflow graph。
- 对于多个 concurrent OpenClaw task，可以先做 offline/sidecar 的 graph similarity 和 shared prefix analysis，输出少量 engine hints。
- SGLang 侧不应承接完整 query plan executor。它只消费 `critical_path_rank`、`cache_scope`、`expected_reuse_window_ms`、`prefetch_deadline_ms`、`routing_affinity` 等字段。
- 如果未来进入更激进阶段，Halo 的 op-level batching 可以演进成 OpenClaw 的 workflow batcher：在 OpenClaw provider adapter 之前聚合同类 LLM steps，再把 batched LLM calls 发给 SGLang。

---

## 6. KVFlow vs Halo：关键差异

| 维度 | KVFlow | Halo |
|------|--------|------|
| 主要问题 | agent workflow 的 KV cache eviction/prefetch | 多 workflow/query DAG 的全局执行与资源优化 |
| inter-workflow 粒度 | cache node / agent timestep / prefix reuse | operator DAG / query batch / shared computation |
| 图模型 | Agent Step Graph，用于估算 steps-to-execution | Workflow query-plan DAG，用于调度执行 |
| 共享资源 | GPU/CPU KV cache、LoRA payload、Radix/HiRadix node | LLM op batch、KV cache、device placement、cache movement |
| engine 形态 | SGLang fork，接入真实 scheduler/cache path | 自定义 Python prototype runtime |
| 对 SGLang 的直接参考 | 高：AgentManager、update endpoint、cache priority/prefetch | 中：思想可用，代码不能直接 patch |
| 对 OpenClaw 的直接参考 | 中：需要把动态 lifecycle 转成 reuse distance | 高：可指导 workflow graph analyzer/batcher |
| “计算图重合”强度 | 弱到中：重合体现在 shared prefix/cache nodes | 强：明确把 workflow/query DAG 当作优化对象 |

---

## 7. 对 Stage 2 近期设计的落点

近期建议把 inter-workflow 工作拆成两个互补分支：

### 7.1 KVFlow 分支：SGLang cache lifecycle

目标：让 SGLang 能消费来自 OpenClaw 的 workflow/session/tool lifecycle hints，改善 KV residency 和 prefetch。

最小可行路径：

1. OpenClaw provider adapter 给每个 LLM request 增加 typed `agent_hints`。
2. OpenClaw runtime 在 tool start / tool complete / sub-agent fan-in 等事件上更新 workflow state。
3. SGLang 增加 `AgentStateManager`，维护 session/agent 到最近 prefix cache node、reuse window、criticality 的映射。
4. Radix/HiRadix eviction 和 load-back/prefetch 在已有 policy 上增加 workflow-aware priority factor。
5. Phase 1 只做观测和离线评估；Phase 2 再打开 eviction/prefetch 行为变化。

### 7.2 Halo 分支：OpenClaw graph analyzer / batcher

目标：在 OpenClaw/control-plane 侧识别多 workflow task 的相似结构和共享资源机会。

最小可行路径：

1. 从 OpenClaw trace 中抽取 workflow graph：LLM step、tool step、sub-agent edge、fan-in/fan-out edge。
2. 给节点计算 critical distance / frontier / parent locality / expected output dependency。
3. 对 concurrent tasks 做 graph similarity 和 shared prefix analysis。
4. 先不合并执行，只输出 SGLang hints：cache scope、critical path rank、prefetch deadline、routing affinity。
5. 后续再评估是否在 OpenClaw provider adapter 层做同类 LLM step batching。

### 7.3 当前判断

因此，本轮 inter-workflow 研究顺序应该是：

1. **先深挖 KVFlow**：因为它有 SGLang fork，直接告诉我们 SGLang 侧应该在哪些文件和状态机上接 agent-aware cache signal。
2. **再深挖 Halo**：因为它更接近真正的多 workflow 计算图重合，但实现形态是 prototype，需要转译成 OpenClaw graph analyzer / batcher，而不是直接改 SGLang。
3. **其他论文只做边界对照**：避免把 tool wait KV lifecycle、program scheduling、semantic variable serving、single-task DAG optimizer 都混成同一个 inter-workflow 方向。

