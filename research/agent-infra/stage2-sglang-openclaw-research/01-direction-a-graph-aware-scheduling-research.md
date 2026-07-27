# 方向 A 中间研究报告：图感知推理请求调度

> 日期：2026-05-25
> 主题：agent workflow signal 如何进入 SGLang scheduler/cache，并与 OpenClaw runtime 对接

---

## 1. 问题定义

方向 A 要解决的是 serving engine 的可见性问题：OpenClaw 掌握 workflow、session、tool wait、sub-agent fan-out/fan-in 等结构，但 SGLang 目前主要看到独立的 LLM 请求。SGLang 内部已有成熟的 prefix-aware scheduler 和 Radix/HiRadix cache，不过这些机制只基于 token prefix、cache key 和本地 request 状态工作，不知道 agent workflow 的控制面状态。

近期目标不是把完整 agent DAG 搬进 SGLang，而是建立一条低风险 signal plane：

- OpenClaw 继续负责 agent loop、tool lifecycle、workflow trace。
- SGLang 只消费 serving-relevant 的 hints：workflow/session identity、step role、critical path rank、tool wait window、KV lifecycle hint。
- scheduler/cache 决策仍保持 engine-native：prefix match、queue wait、batch size、HBM/host/storage cache 状态仍由 SGLang 控制。

---

## 2. 相关论文与开源实现

| 系统 | 机构 | 论文/代码状态 | 核心思想 | 对本方向的启发 | 主要限制 |
|------|------|---------------|----------|----------------|----------|
| Continuum | UC Berkeley; Stanford University; Tensormesh; Tsinghua University | [arXiv:2511.02230](https://arxiv.org/abs/2511.02230)，[vLLM preview code](https://github.com/Hanchenli/vllm-continuum) | 根据 multi-turn agent 的 tool wait，决定 KV cache TTL/pinning，减少释放/重算 | tool wait window 是 serving 层可消费的强信号；job/session 级 identity 必须显式传入 engine | 实现高度实验化，tool 解析和阈值较 workload-specific |
| Autellix | UC Berkeley; Google DeepMind; Shanghai Jiao Tong University | [arXiv:2502.13965](https://arxiv.org/abs/2502.13965)，本轮未找到官方代码 | 把 LLM agent 视为 general programs，拦截程序内 LLM calls，并用 program-level context 做调度 | 直接支持“workflow/program 是一等调度对象”的主张，和 OpenClaw workflow -> SGLang signal plane 高度相关 | 缺少开源实现；更偏 request/program scheduling，不直接解决 KV lifecycle |
| KVFlow | University of California, San Diego; Amazon Web Services | [arXiv:2507.07400](https://arxiv.org/abs/2507.07400)，[KVFlow repo](https://github.com/PanZaifeng/KVFlow) | Agent Step Graph / timestep signal + priority-based eviction + overlapped prefetch | 最贴近 SGLang/HiRadix 的方向：它直接提供 SGLang-based serving engine fork 和 SScheduler 中间层 | 代码公开较晚，且同时包含 KVFlow/ScaleSim；需要进一步拆清哪些机制对应论文实验 |
| ScaleSim | University of California, San Diego; Amazon Web Services | [arXiv:2601.21473](https://arxiv.org/abs/2601.21473)，[KVFlow repo](https://github.com/PanZaifeng/KVFlow) | Invocation distance-based memory management，面向 large-scale multi-agent simulation | 把“未来调用距离”泛化为 KV/LoRA/agent state 的 eviction/prefetch priority | 面向 simulation，agent 调用顺序更可预测；OpenClaw 的动态 tool/sub-agent loop 需要重新建模 |
| Halo | National University of Singapore | [arXiv:2509.02121](https://arxiv.org/abs/2509.02121)，[Halo_demo](https://github.com/mlsys-io/Halo_demo) | workflow DAG 优化、critical distance、parent-device cache locality、显式 cache movement | 适合作为 graph-aware optimizer 的架构参考 | 假设 workflow 可声明为 YAML/DAG，重于近期 signal plane |
| Parrot | Shanghai Jiao Tong University; Microsoft Research | [paper](https://www.usenix.org/conference/osdi24/presentation/lin-chaofan)，[ParrotServe](https://github.com/microsoft/ParrotServe) | semantic variable 暴露 prompt/DAG 结构，ServeCore 做 graph/context-aware scheduling | 证明 serving 层可以利用应用级语义做全局调度 | 需要应用改写到 Parrot API，不适合直接套到 OpenClaw |
| Teola / Ayo | The Chinese University of Hong Kong; Unaffiliated | [arXiv:2407.00326](https://arxiv.org/abs/2407.00326)，[Ayo](https://github.com/NetX-lab/Ayo) | 把 agent task 拆成 primitive DAG，做 prefill split、stage decomposition、topology-aware scheduling | 可借鉴其 primitive/graph scheduler 思路做 trace analysis | 假设任务结构相对静态，OpenClaw 的动态 agent loop 更松散 |

补充复核后，方向 A 的近期范围需要收束：只有直接服务于 **agent-aware scheduling / KV lifecycle / tool wait memory handling** 的系统进入本报告主线；其他相邻系统转入另行规划。

| 系统 | 机构 | 代码状态 | 为什么需要记录 | 纳入方式 |
|------|------|----------|----------------|----------|
| APIServe / InferCept | University of California, San Diego | [arXiv:2402.01869](https://arxiv.org/abs/2402.01869)，[InferCept repo](https://github.com/WukLab/infercept) | vLLM fork，直接处理 tool/API interruption 下 KV preserve/discard/swap | 纳入 tool wait / KV lifecycle 对照 |
| LAMPS / MARS | Harvard University; Tsinghua University | [arXiv:2410.18248](https://arxiv.org/abs/2410.18248)，[MARS repo](https://github.com/mars-repository/mars-codebase) | vLLM fork，按 API-call 期间 memory-over-time 做 augmented request scheduling | 纳入 API/tool pause scheduling 对照 |

ALTO、Pie、Symphony、Agent.xpu 属于 graph granularity、programmable serving 或 on-device SoC 的外延议题，不进入方向 A 近期详细设计；它们只在覆盖矩阵中作为另行规划项保留。

### 2.1 实现层级分类

这些系统不能按“都有代码”直接等价比较。按实现形态分层如下：

| 层级 | 系统 | 实现方式 | 对我们的可复用性 |
|------|------|----------|------------------|
| 真实推理引擎 patch | Continuum | 直接改 vLLM v1 scheduler/request/KV block 管理路径，新增 policy、job id、pinned request、tool wait estimator | 最接近 vLLM 侧落地，可参考 request metadata、scheduler hook、KV retention 位置 |
| 真实推理引擎 fork | KVFlow | fork SGLang，在 `/python/sglang` 中加入 AgentManager、`/v1/update`、priority-based eviction、overlapped prefetch、LoRA/KV 预取控制 | 最接近 SGLang 侧落地，是方向 A 必须深入读的核心代码 |
| 真实推理引擎 fork / simulation serving | ScaleSim | 与 KVFlow 共用 `PanZaifeng/KVFlow` repo，围绕 invocation distance 管理 KV/LoRA/agent memory | 对 OpenClaw 的启发是 reuse-distance/control signal，而不是 simulation timestep API 本身 |
| 真实推理引擎 fork | InferCept | fork vLLM，在 interception/API call 处实现 Preserve/Discard/Swap 等 KV handling | 可作为 Continuum 之外的 augmented LLM pause/KV lifecycle 对照 |
| 真实推理引擎 fork | LAMPS / MARS | fork vLLM，在 scheduler/policy 层预测 API call memory handling strategy | 可补充“请求调度 + API-call memory cost”视角 |
| paper-only serving system | Autellix | program-aware LLM serving，intercepts LLM calls，按 program progress/preemption 做 scheduling | 对 signal plane 的抽象非常关键，但目前只能按论文机制转译 |
| 自定义 serving runtime | Parrot | 自建 ServeCore、GlobalScheduler、ContextManager、Engine abstraction；应用通过 SemanticVariable API 提交请求 | 可参考 graph/context-aware serving 分层，但不能直接 patch 到 SGLang |
| workflow optimizer + worker prototype | Halo | YAML/DAG parser + heuristic scheduler + prototype workers；显式生成 cache movement commands | 可参考 critical distance、parent locality、cache movement 思想，不能当作现成 engine implementation |
| agent DAG runtime / scheduler prototype | Ayo | 自定义 DAG primitive、optimization pass、Ray actor scheduler 和 engine scheduler；不是 vLLM/SGLang patch | 可参考 primitive DAG、node depth、topology-aware batching，适合 OpenClaw trace analyzer 而非 SGLang 热路径 |
| 中间层 + engine fork | KVFlow | `SScheduler` 维护 agent timestep/distance，SGLang fork 消费 timestep 更新并影响 cache eviction/prefetch | 可直接参考 AgentManager 与 HiRadix/RadixCache 的集成方式，但要重新映射到 OpenClaw 的 workflow/tool lifecycle |

---

## 3. 开源实现审计

### 3.1 Continuum

本轮审计了 `vllm-continuum` 的核心实现。它把 multi-turn agent 的 job/session 字段挂到 vLLM request 上，并新增 `CONTINUUM` scheduling policy。

关键实现点：

- `vllm/v1/request.py` 增加 `job_id`、`last_func_call`、`is_last_step`、`this_func_call`，字段来自 sampling params 的 extra args。
- `vllm/v1/core/sched/request_queue.py` 增加 `ContinuumRequestQueue`，优先选择被 pinned 的 job，并按 job-level first-entry time 排序。
- `vllm/v1/core/sched/scheduler.py` 在 request finish 后，如果不是最后一步且 policy 是 Continuum，就不立即释放 KV block，而是根据 estimator 做 pin/unpin。
- `vllm/v1/core/estimate_with_func.py` 用 `ToolCallParser` 从模型输出的 bash block 中解析 tool name，用固定阈值估算是否值得 pin。

对我们的启发：

- tool wait window 确实可以在 serving 层转化为 KV residency 决策。
- job/session identity 必须是 request-native 字段，不能只靠 prompt 字符串推断。
- 但 OpenClaw 不应复制 Continuum 的 tool parser：tool start/end 在 OpenClaw 侧是结构化事件，应该直接传 typed lifecycle event。

### 3.2 KVFlow 与 ScaleSim

本轮补充审计后，KVFlow 已确认有公开代码。作者主页和 GitHub 仓库都指向 `PanZaifeng/KVFlow`；README 明确说明该仓库同时包含 **KVFlow** 和 **ScaleSim**：

- KVFlow: Efficient prefix caching for accelerating LLM-based multi-agent workflows.
- ScaleSim: Serving Large-Scale Multi-Agent Simulation with Invocation Distance-Based Memory Management.

仓库提供两部分：

- `SScheduler`：面向 agent simulation / workflow 的 pluggable mid-layer，用于暴露 request metadata。
- `python/sglang`：SGLang-based serving engine，支持 priority-based eviction 和 overlapped prefetch，覆盖 KV 与 LoRA payload。

KVFlow 与 ScaleSim 的区别：

- **KVFlow** 更偏 multi-agent workflow 的 prefix KV cache。论文抽象是 Agent Step Graph / Steps-to-Execution，目标是让 serving engine 知道哪些 KV prefix 会很快被复用。
- **ScaleSim** 更偏 large-scale multi-agent simulation。论文抽象是 invocation distance，即 agent 距离下一次 LLM invocation 还有多远，用它统一驱动 KV、LoRA、agent-specific memory 的驱逐与预取。
- 两者共用的工程模式是：外部 agent/workflow 层产生未来调用距离或复用距离，SGLang fork 把距离转成 cache eviction/prefetch priority。

关键实现点：

- `SScheduler/Scheduler.py` 维护多个 timestep manager，周期性收集 agent 到 timestep 的映射，通过 `POST /v1/update` 把 `agent_data`、`timestep_data`、`timestep_cnt` 发给 SGLang。
- `python/sglang/srt/entrypoints/http_server.py` 新增 `/v1/update`，把 update 请求交给 tokenizer manager，再发到 scheduler control channel。
- `python/sglang/srt/managers/agent_manager.py` 维护 `agent_to_last_nodes`、agent hold/prefetch priority、update version。
- `python/sglang/srt/managers/scheduler.py` 创建 AgentManager，并在收到 `UpdateAgentTimestepReq` 后更新 agent timestep，刷新 cache node priority，触发 prefetch。
- `python/sglang/srt/mem_cache/radix_cache.py` 和 `hiradix_cache.py` 在 TreeNode 上记录 agents / hold priority，并在 eviction/load-back/prefetch 逻辑中查询 AgentManager。
- `python/sglang/srt/managers/scheduler_output_processor_mixin.py` 的 `DispatchLoadTasks` 会根据 timestep window 预取 LoRA 和 KV：LoRA 走 `prefetch_lora_timesteps`，KV 通过 `agent_to_last_nodes` 找到 last nodes 后调用 `load_back`。
- `server_args.py` 增加 `load_ahead_step`、`evict_pri_level`、`enable_holding`、`enable_interrupt`、`disable_prefetch`、`disable_lr_pf`、`disable_kv_pf` 等控制项。

对我们的启发：

- KVFlow 证明“agent runtime/mid-layer 产生距离信号，SGLang cache 层消费信号”这条路径是能落到 SGLang fork 的。
- ScaleSim 进一步说明这个距离信号可以不只服务 KV prefix，也可以覆盖 LoRA adapter 和 agent-specific memory。
- 它比 Continuum 更贴近我们的 SGLang 代码面；Continuum 更适合作为 vLLM 侧 tool-wait KV retention 的对照。
- OpenClaw 不能简单复用 KVFlow/ScaleSim 的 timestep API。我们需要把 `timestep` / invocation distance 转译成 workflow/session/tool lifecycle/critical path/reuse window，并保留 OpenClaw 的 side-effect 和 replay 安全边界。

限制：

- 当前代码同时承载 KVFlow 和 ScaleSim，需要进一步定位哪些 benchmark 和代码路径严格对应 KVFlow NeurIPS 论文，哪些属于 ScaleSim 后续扩展。
- `SScheduler` 更偏 simulation/workflow timestep manager；OpenClaw 的真实 agent loop 需要用 trace event 和 lifecycle event 生成类似的 distance/reuse signal。
- ScaleSim 的 simulation 假设比 OpenClaw 更规则：agent 下一次调用常由 timestep、空间距离或交互规则估计；OpenClaw 的 tool 调用、sub-agent fan-in、用户状态会让 invocation distance 更不稳定。

### 3.3 Autellix

Autellix 没有找到官方代码，但它是方向 A 的核心论文参考之一。论文首页机构为 UC Berkeley、Google DeepMind、Shanghai Jiao Tong University。其基本判断是：agentic workload 不应被 serving engine 看成一串独立 LLM calls，而应被看成运行时展开的 general programs。程序由 LLM calls、tool/human/code interrupts 构成动态 DAG，完整图通常只有运行时才逐步出现。

关键机制：

- Autellix intercepts programs submitted LLM calls，把单个 LLM request 关联回所属 program。
- scheduler 拿到 program-level context 后，按 program 已完成 calls、当前等待状态和 head-of-line blocking 风险做 preempt / prioritize。
- 论文区分 single-threaded programs 和 distributed programs，分别提出调度算法，目标是降低 program end-to-end latency，而不是单个 request 的局部 latency。
- 论文报告相对 vLLM 在相同 latency 下 program throughput 提升 4-15x。

对我们的启发：

- OpenClaw 的 workflow/session/run 应该是 SGLang signal plane 的一等对象，不能只作为 metrics label。
- `critical_path_rank` 之外，还需要 program progress / completed_steps / active_frontier 这类进度信号，否则 scheduler 很难识别 workflow-level head-of-line blocking。
- Autellix 更偏 queue/scheduler 层；KVFlow/ScaleSim 更偏 cache eviction/prefetch 层。方向 A 需要把两者合并：Autellix 给 program-aware scheduling，KVFlow/ScaleSim 给 reuse-distance-aware cache lifecycle。

限制：

- 没有可审计官方实现，不能像 KVFlow/ScaleSim 那样直接对照 SGLang 代码路径。
- 论文中的 serving engine 边界比 OpenClaw + OpenAI-compatible SGLang 更内聚；我们近期仍应通过 typed hints 和 lifecycle endpoint 做渐进适配。

### 3.4 Halo

Halo demo 中的 workflow 被解析成 operator DAG。`halo/parser.py` 根据 YAML 构造节点、边和 `max_distance`；`halo/schedulers/heuristic_t.py` 根据 ready frontier、critical distance 和 parent device locality 生成 `execute`、`dump_cache`、`get_cache`、`send_cache`、`resume_cache`、`complete` 等动作；worker 侧显式管理 KV cache offload/preload。

对我们的启发：

- graph distance 可以作为 criticality/scheduling rank 的基础。
- cache movement 需要和 workflow dependency 同步，不只是 LRU 的被动结果。
- 但 Halo 更像一个完整 query optimizer + execution runtime，近期不应把 SGLang 改成 Halo 式 DAG executor。

### 3.5 Parrot

Parrot 通过 semantic variable 暴露 prompt 内部依赖，ServeCore 把 semantic request 拆成 Fill/Generate chain，GlobalScheduler 支持 graph grouping、context grouping 和 context-aware engine selection。ContextManager 维护 prefix cache，可按 semantic variable id 查询不同 engine 上的 cached prefix。

对我们的启发：

- serving 层如果能看到应用语义，就能做比单请求调度更强的全局决策。
- prefix cache 命中不一定只能靠 token 字符串，也可以靠语义变量/上下文 id 建模。
- 但 Parrot 要求应用显式使用 Parrot API；OpenClaw 当前是 OpenAI-compatible provider 路径，更适合先做 adapter/hints。

### 3.6 Ayo

Ayo 把任务建模成 DAG primitive，包含 LLM prefill、decode、partial prefill、partial decode、downstream operator 等节点。优化 pass 包括 prefill splitting、stage decomposition、decoding pipelining。其 engine scheduler 可按 node depth 做 topology-aware batching。

对我们的启发：

- 对静态或半静态 agent workflow，可以把 LLM step 和 tool/data operator 放到同一张 DAG 中优化。
- OpenClaw 的 trace analyzer 可以吸收 Ayo 的 node depth / topology-aware 思路。
- 但 Ayo 的 primitive DAG 比 OpenClaw 真实 agent loop 更强约束，近期只把它作为 trace 后处理和 workload 分类参考。

### 3.7 InferCept 与 LAMPS/MARS

补充复核后，InferCept 和 LAMPS/MARS 需要进入方向 A 的代码对照集合。它们不直接做 workflow DAG 优化，但都在真实 vLLM fork 中处理 API/tool augmentation 带来的 pause、KV residency 和 scheduler trade-off。

InferCept 的重点是 interception：当外部 API/tool 调用打断 LLM decoding 时，系统不把它简单视为 request 结束，而是比较 preserve、discard/recompute、swap 等处理方式。它和 Continuum 的共同点是都在问“tool/API 等待期间 KV 应该怎么处理”，差异是 InferCept 更像通用 augmented LLM interception 机制，Continuum 更偏 multi-turn agent 的 tool wait TTL/pinning。

LAMPS/MARS 的重点是 scheduling：它把 API call 期间的 memory handling strategy 纳入请求排序，按 memory-over-time 估计 request 对 GPU KV cache 的占用成本。它对我们的启发是：`expected_reuse_window_ms` 不应只作为 cache hint，也可以进入 queue policy 的 cost model。

适配判断：

- 它们都证明 API/tool pause 不是纯 agent-framework 问题，而是 inference scheduler/cache 的一等问题。
- OpenClaw 侧能提供比论文 benchmark 更可靠的 tool lifecycle、duration、side-effect class；SGLang 侧应消费这些结构化信号，而不是从模型输出文本里推断。
- 它们没有解决 OpenClaw 的 mutating tool、用户可见状态和 replay invalidation 问题；这些安全边界仍应留在 OpenClaw。

### 3.8 另行规划项

以下系统不纳入方向 A 近期实现设计，只作为另行规划项在覆盖矩阵中保留：

- **Pie / Symphony**：programmable serving / serve programs 方向，涉及 agent logic 下沉到 serving engine，与本轮 signal-plane 设计不是同一阶段。
- **ALTO**：partial-output streaming / nested ancestry 方向，属于更完整 workflow runtime 的问题，不进入本轮 SGLang scheduler/cache 改造。
- **Agent.xpu**：on-device heterogeneous SoC agent scheduling，硬件和部署假设与当前云端 SGLang/OpenClaw 目标不同。

---

## 4. SGLang 代码层面的现状

### 4.1 请求字段传播

SGLang 当前已有若干可复用入口：

- `GenerateReqInput` 中有 `priority`、`extra_key`、`routing_key`、`custom_labels`、`external_trace_header`。
- OpenAI serving path 中，`serving_base.py` 会处理 `cache_salt/extra_key`、`custom_labels` 和 `x-smg-routing-key`。
- tokenizer/scheduler path 会把 `priority`、`extra_key`、`routing_key` 传播到 `TokenizedGenerateReqInput` 和 `Req`。

这说明 Phase 1 可以不改 SGLang 核心逻辑，先用 labels/extra_key/routing_key 做观测和对照实验。但它们不能直接替代 typed agent hints。

### 4.2 `extra_key` 的边界

`RadixKey` 包含 `extra_key`，prefix matching 时要求 `extra_key` 一致。因此 `extra_key` 是 cache namespace / isolation 字段，而不是 workflow metadata 字段。

正确用法：

- 表达 tenant、model、prompt policy、tool schema、cache salt 等会影响 prefix cache 共享边界的因素。
- 用于避免不该共享的 prompt prefix 发生跨域命中。

风险用法：

- 把 `workflow_id` 或 `run_id` 放进 `extra_key`。这会把每个 workflow 的 prefix cache 隔离开，可能直接破坏跨 workflow 的 stable prefix sharing。

### 4.3 `routing_key` 的边界

SGLang 的 `routing-key` policy 属于 cache-agnostic policy。它按 running batch 中 routing key 的频次给 waiting queue 排序，但不会先做 prefix match，也不会和 `lpm` / `dfs-weight` 同时生效。

这意味着：

- `routing_key` 可以作为 Phase 1 对照实验，观察 workflow affinity 是否有收益。
- 主路线不应把 `routing_key` 当作 graph-aware scheduling 的最终实现。
- 如果要同时保留 prefix locality 和 workflow criticality，需要新增 cache-aware composite policy。

### 4.4 `priority` 的边界

SGLang 支持 request priority，但当前 server args 中 priority scheduling 与部分 policy 存在兼容限制。优先级适合做隔离实验，不适合作为 Phase 1 默认路径。

更稳妥的做法是：

- Phase 1 不启用 preemption，只收集 criticality 与 latency/cost 数据。
- Phase 2/3 在新 policy 内把 critical_path_rank 作为排序因子，而不是直接开启全局 priority preemption。

### 4.5 HiRadixCache / CacheController 的机会点

HiRadixCache 已经具备多层缓存和 load-back/prefetch/write-back 的基础结构：

- `match_prefix` 可返回 device hit、host hit、best match node、host hit length。
- `init_load_back` 可在 prefill 前把 host cache load 回 device。
- `prefetch_from_storage` 可对 storage 层做 prefetch。
- `CacheController` 的 `write`、`load`、`prefetch` 操作带 priority。

因此 KV lifecycle 的近期实现不需要重写 cache 系统。更合适的方向是把 tool wait / expected reuse / critical path 转译成 cache-native hints：

- tool_start：如果预计 tool wait 较长，可降低 GPU residency 或触发 write-through/backup。
- tool_complete：如果下一步 LLM 请求即将到达，可提前 load-back/prefetch。
- critical path：对关键路径 session 的 host/storage load-back 给更高 priority。

---

## 5. OpenClaw 代码层面的现状

### 5.1 SGLang provider extension

OpenClaw 的 `extensions/sglang/index.ts` 当前主要注册 SGLang provider，并复用 openai-compatible replay hooks。它还没有专门的 SGLang signal adapter。

可复用机制：

- `src/plugin-sdk/provider-stream-shared.ts` 提供 payload patch wrapper，可在 provider stream 调用前修改 request payload。
- 其他 provider extension 已经有 stream wrapper 模式，可以作为 SGLang extension 的实现模板。

缺口：

- 现有 payload patch 主要修改 body；`routing_key` 如果要走 `x-smg-routing-key` header，需要 provider 层支持 header patch 或单独 request adapter。
- 需要避免在 OpenClaw core 中硬编码 SGLang 逻辑。更好的边界是：core 产生通用 trace/hints，SGLang extension 消费并转成 SGLang-specific body/header/control event。

### 5.2 Tool lifecycle

OpenClaw 的 tool handler 中已经有结构化生命周期：

- tool start：能得到 tool name、args、run id、tool call id，并记录 start timestamp。
- tool end：能得到 result/error、duration、side effect/replay invalidation 相关状态，并触发 hook。

这比 Continuum 从模型输出中解析 tool call 更可靠。近期设计应直接从这里产生 lifecycle event：

- `tool_start`
- `tool_end`
- `tool_error`
- `tool_side_effect`
- `replay_invalidated`

这些事件一方面进入 OpenClaw trace analyzer，另一方面可被 SGLang adapter 转发给 `/v1/agent/session_event`。

### 5.3 Sub-agent spawn/fan-in

OpenClaw 的 spawn 上下文已有 agent/session/channel/thread 等信息，但还没有完整 workflow graph metadata。近期只需要加少量通用字段：

- `workflow_id`
- `parent_run_id`
- `parent_step_id`
- `spawn_index`
- `fan_in_group_id`

这些字段不应绑定 SGLang，而是作为 agent runtime trace 的通用图结构。

---

## 6. 假设 mismatch

| 层面 | 论文/原型常见假设 | OpenClaw + SGLang 现实 | 适配策略 |
|------|------------------|------------------------|----------|
| workflow 结构 | DAG/YAML/semantic variable 显式声明 | agent loop 动态生成，sub-agent/tool 可能运行时决定 | OpenClaw trace 重建局部 DAG，先做 runtime hints |
| tool 信息 | 可从模型输出或 benchmark metadata 推断 | tool lifecycle 在 OpenClaw 内部最可靠 | 用 OpenClaw 结构化事件，不解析模型输出 |
| scheduler | 可直接按 job/session 改 engine scheduler | SGLang 已有 prefix-aware policy，routing-key 是 cache-agnostic | 新增 cache-aware composite policy，而非替换成 job-only ordering |
| cache key | workflow id 可用于 affinity | SGLang `extra_key` 会影响 prefix cache namespace | workflow id 不进 `extra_key`，进入 `agent_hints` |
| KV lifecycle | pin/unpin KV block 即可 | SGLang 有 Radix/HiRadix 多层 cache 和 async controller | 把 tool wait 转成 write-through/load-back/prefetch hint |

---

## 7. 近期设计建议

### 7.1 Phase 1：观测优先

OpenClaw：

- 生成 workflow/session/tool labels。
- SGLang provider 把低基数字段放入 `custom_labels`。
- 仅在隔离实验中使用 `routing_key`。

SGLang：

- 不改 scheduler 语义。
- 跑 `fcfs`、`lpm`、`dfs-weight`、`routing-key` baseline。
- 按 labels 聚合 queue wait、TTFT、prefix hit、host/storage hit、eviction/load-back。

### 7.2 Phase 2：typed `agent_hints`

SGLang：

- 在 request schema、tokenized request、`Req` 中增加 typed `agent_hints`。
- 新增 AgentStateManager，按 workflow/session/step 维护轻量状态。
- 保持 `extra_key` 只表达 cache namespace。

OpenClaw：

- SGLang extension 从 runtime trace 生成 request-time hints。
- tool lifecycle 通过 adapter 发送到 SGLang control endpoint。

### 7.3 Phase 3：agent-aware cache-aware policy

新增 policy 的基本排序逻辑：

1. 先调用现有 prefix match 逻辑，保留 LPM/DFS_WEIGHT 的 cache-aware 基础。
2. 对同等或近似 prefix locality 的请求，引入 workflow/session group、critical_path_rank、expected_tool_return_time。
3. fan-in/orchestrator 恢复请求在关键路径上优先，但不牺牲明显更高 prefix hit 的 batch。
4. 所有新因子都必须可 ablation：prefix-only、workflow-only、critical-path-only、combined。

---

## 8. 初步可行性结论

方向 A 可行，但需要避免两个陷阱：

1. 不要把 `workflow_id` 塞进 `extra_key`，否则会破坏共享边界。
2. 不要把 `routing-key` 当作最终方案，因为它当前不是 prefix-aware policy。

正确路径是先建立观测闭环，再用 typed `agent_hints` 和 AgentStateManager 把 agent control plane 翻译成 SGLang native scheduling/cache hints。这样既利用了 OpenClaw 对 workflow 的真实可见性，也保留了 SGLang 对 token prefix、batch 和 KV cache 的本地控制权。
