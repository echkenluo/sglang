# 代码审计附录：外部实现与本地适配证据

> 日期：2026-05-25
> 定位：支撑两份中间研究报告和最终设计的代码证据索引

---

## 1. 审计快照

本轮代码审计基于以下快照：

| 仓库 | 路径 | commit |
|------|------|--------|
| SGLang | `/home/luocc4/workspace/sglang` | `abe2ec2aff6f10d3a9c719a3b505d23858d797b6` |
| OpenClaw | `/home/luocc4/workspace/openclaw` | `50a2481652b6a62d573ece3cead60400dc77020d` |
| vLLM Continuum | `/home/luocc4/workspace/agent-infra-paper-code/vllm-continuum` | `316a58794a6ff86b216e579b74fd56ed0c5a911f` |
| KVFlow / ScaleSim | `/home/luocc4/workspace/agent-infra-paper-code/KVFlow` | `7ef897eaedc16626a281ccfe0d453fae80251e8f` |
| Halo demo | `/home/luocc4/workspace/agent-infra-paper-code/Halo_demo` | `bf3fe25ba7a47fa0fab863da87dc5e470438129d` |
| Ayo | `/home/luocc4/workspace/agent-infra-paper-code/Ayo` | `8f42787d1cefb2e281ca5f168f3b89669d688a13` |
| ParrotServe | `/home/luocc4/workspace/agent-infra-paper-code/ParrotServe` | `2e1825ee2bc38cb783bab9d8ec3e5ae99a93ba46` |
| Pie | `/home/luocc4/workspace/agent-infra-paper-code/Pie` | `076749cb85f000c59ff2b70971bbf9fff5da8911` |
| InferCept | `/home/luocc4/workspace/agent-infra-paper-code/infercept` | `3a1a4dac37cb569e767bc4a95ef8ed0995a13b99` |
| Conveyor | `/home/luocc4/workspace/agent-infra-paper-code/conveyor` | `c87718363e966de7ff26cabd831f5912f52cf13d` |
| MARS/LAMPS | `/home/luocc4/workspace/agent-infra-paper-code/mars-codebase` | `4cedd10757f2e0aa0fe329c6515e990ebdacaa3d` |

---

## 2. 外部开源实现证据

### 2.0 实现形态结论

本轮审计后，方向 A 相关系统的实现形态可以明确分为五类：

| 类别 | 系统 | 机构 | 是否在真实推理引擎上 patch | 具体实现法 |
|------|------|------|----------------------------|------------|
| engine patch | Continuum | UC Berkeley; Stanford University; Tensormesh; Tsinghua University | 是，vLLM v1 | 改 request metadata、request queue、scheduler、KV block release/pinning 逻辑 |
| paper-only serving system | Autellix | UC Berkeley; Google DeepMind; Shanghai Jiao Tong University | 论文声称实现 serving engine，但本轮未找到官方代码 | program-aware serving；intercept LLM calls；按 program progress 和 preemption 做调度 |
| engine fork | KVFlow | University of California, San Diego; Amazon Web Services | 是，SGLang-based serving engine fork | `SScheduler` 产生 agent timestep/distance 信号，SGLang fork 消费 `/v1/update`，做 priority-based eviction 和 overlapped prefetch |
| engine fork / simulation serving | ScaleSim | University of California, San Diego; Amazon Web Services | 是，与 KVFlow 共用 SGLang fork repo | invocation distance-based memory management，扩展到 KV、LoRA、agent-specific memory 等 simulation 状态 |
| engine fork | APIServe / InferCept | University of California, San Diego | 是，vLLM fork | API/tool interception 下 Preserve/Discard/Swap KV handling |
| engine fork | LAMPS / MARS | Harvard University; Tsinghua University | 是，vLLM fork | API-call memory-over-time scheduling；prediction components |
| 自定义 serving stack | Parrot | Shanghai Jiao Tong University; Microsoft Research | 不是 patch vLLM/SGLang；自建 ServeCore + Engine abstraction | 应用层 SemanticVariable -> ServeCore graph/context scheduler -> Engine Fill/Gen API |
| workflow/cache movement prototype | Halo | National University of Singapore | 不是 patch vLLM/SGLang；自建 DAG optimizer + worker | YAML workflow -> operator DAG -> heuristic schedule -> worker 执行 execute/dump/get/send/resume cache |
| agent DAG runtime prototype | Teola / Ayo | The Chinese University of Hong Kong; Unaffiliated | 不是 patch vLLM/SGLang；自建 primitive DAG runtime | DAG optimization pass -> graph scheduler -> Ray actor engine scheduler -> engine request batching |
| tool partial execution runtime | Conveyor | Duke University | 不是 patch vLLM/SGLang；自建 runtime | tool developer 暴露 partial execution interface，scheduler 让 tool 与 LLM decoding overlap |
| 另行规划 / programmable serving runtime | Pie | Yale University | 不是单一 engine patch；包含 vLLM/SGLang drivers | Wasm inferlet runtime，应用控制 KV policy、forward/sampling、tool/IO |

因此，“真实推理引擎上直接实现”的严格口径下，当前可读代码里 **Continuum、KVFlow/ScaleSim、InferCept、LAMPS/MARS** 是本轮近期方向应继续重点审计的 engine patch/fork：Continuum/InferCept/MARS 是 vLLM patch/fork，KVFlow/ScaleSim 是 SGLang fork。Parrot/Halo/Ayo/Conveyor 作为近期辅助参考；Pie 虽有代码，但属于 programmable serving 另行规划项。

### 2.1 Continuum / vLLM

代码路径：

- `vllm/v1/request.py`
- `vllm/v1/core/sched/request_queue.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/core/estimate_with_func.py`

实现要点：

- request 层新增 `job_id`、`last_func_call`、`is_last_step`、`this_func_call`，从 sampling params 的 extra args 进入 scheduler。
- request queue 新增 `CONTINUUM` policy，围绕 job-level first-entry time 和 pinned job 选择下一个 request。
- scheduler 在 request finish 后，根据是否为 last step 决定是否释放 block；若不是 last step，可进入 pinned request 逻辑。
- estimator 从模型输出文本中解析 bash block / tool name，并以固定阈值判断 tool wait 是否值得 pin。

适配判断：

- 可借鉴 job/session identity 和 tool wait window 到 KV residency 的映射。
- 不应照搬 tool parser；OpenClaw 有结构化 tool lifecycle event。
- 不应照搬固定阈值；应由 OpenClaw trace 和 SGLang cache pressure 共同决定。

### 2.2 KVFlow / ScaleSim / SGLang fork

代码路径：

- `SScheduler/Scheduler.py`
- `SScheduler/timestepManager/BaseManager.py`
- `SScheduler/timestepManager/PlanManager.py`
- `SScheduler/timestepManager/SpaceManager.py`
- `python/sglang/srt/entrypoints/http_server.py`
- `python/sglang/srt/managers/io_struct.py`
- `python/sglang/srt/managers/tokenizer_manager.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/agent_manager.py`
- `python/sglang/srt/mem_cache/radix_cache.py`
- `python/sglang/srt/mem_cache/hiradix_cache.py`
- `python/sglang/srt/mem_cache/lora_radix_cache.py`
- `python/sglang/srt/mem_cache/lora_hiradix_cache.py`
- `python/sglang/srt/managers/scheduler_output_processor_mixin.py`
- `python/sglang/srt/server_args.py`

实现要点：

- `SScheduler` 是 agent/workflow/simulation 中间层，维护 agent 到 timestep / invocation distance 的映射，并通过 HTTP `POST /v1/update` 把 `agent_data`、`timestep_data`、`timestep_cnt` 发送给 SGLang。
- SGLang HTTP server 新增 `/v1/update`，TokenizerManager 将 `UpdateAgentTimestepReq` 送入 scheduler control channel。
- Scheduler 初始化 `AgentManager(evict_pri_level, load_ahead_step)`，收到 timestep update 后调用 `agent_manager.update_agent_timestep(...)`，再刷新 tree cache leaf priority。
- Radix/HiRadix cache 的 node 增加 `agents`、`hold_priority`、`hold_priority_version`，cache eviction/load-back 查询 AgentManager 的 hold/prefetch priority。
- HiRadix cache 增加 priority-aware load-back、prefetch progress 检查、best-effort/timeout/wait-complete prefetch stop policy 等路径。
- `DispatchLoadTasks` 根据 timestep window 主动发起 prefetch：LoRA 走 `prefetch_lora_timesteps`，KV 通过 `agent_to_last_nodes` 找 last node 并调用 HiRadix `load_back`。
- server args 暴露 `load_ahead_step`、`evict_pri_level`、`enable_holding`、`enable_interrupt`、`disable_prefetch`、`disable_lr_pf`、`disable_kv_pf`。

适配判断：

- KVFlow/ScaleSim 是方向 A 中最贴近 SGLang 的可读实现，不再只是论文算法。
- 核心模式是“外部 agent/workflow/simulation scheduler 给出 timestep、invocation distance 或 reuse distance，SGLang cache tree 把该距离转成 eviction/prefetch priority”。
- 对 OpenClaw 不能直接使用 simulation timestep API；需要把 OpenClaw 的 workflow/tool lifecycle/critical path trace 转换成类似的 reuse distance / criticality / cache lifecycle hint。
- 需要继续拆分 KVFlow 与 ScaleSim 共用代码，确认哪些路径对应 KVFlow NeurIPS 论文实验，哪些属于后续 ScaleSim 代码。

### 2.3 Halo

代码路径：

- `halo/parser.py`
- `halo/schedulers/heuristic_t.py`
- `halo/schedulers/heuristic_v.py`
- `halo/optimizers/halo_t.py`
- `halo/workers/worker_t.py`

实现要点：

- YAML workflow 被解析为 operator DAG。
- parser 推导 `max_distance` 和 `keep_cache`，识别 downstream cache reuse。
- scheduler 基于 ready frontier、critical distance 和 parent-device locality 生成执行与 cache movement 动作。
- worker 侧显式支持 `resume_cache`、`dump_cache`、`send_cache`、`get_cache`、`complete` 等 cache control。

适配判断：

- 适合作为 graph optimizer / cache movement 的架构参照。
- 近期不应把 SGLang 改造成 Halo 式 DAG executor；只转译 critical distance 和 cache lifecycle 思想。

### 2.4 Ayo

代码路径：

- `Ayo/dags/dag.py`
- `Ayo/dags/node.py`
- `Ayo/opt_pass/prefilling_split.py`
- `Ayo/opt_pass/stage_decomposition.py`
- `Ayo/opt_pass/decoding_pipeling.py`
- `Ayo/schedulers/graph_scheduler.py`
- `Ayo/schedulers/engine_scheduler.py`

实现要点：

- task 被建模成 primitive DAG，包含 LLM prefill、decode、partial prefill、partial decode、downstream operators。
- optimization pass 可拆分 prefill、stage 和 decode pipeline。
- graph scheduler 按 node readiness 运行 DAG。
- engine scheduler 可按 node depth/topology-aware 策略选 batch。

适配判断：

- 适合为 OpenClaw trace analyzer 提供 DAG primitive 和 node depth 思路。
- 不适合近期直接要求 OpenClaw workload 全部声明为静态 primitive DAG。

### 2.5 ParrotServe

代码路径：

- `docs/sys_design/README.md`
- `docs/user_docs/parrot_apis.md`
- `parrot/serve/graph/graph.py`
- `parrot/serve/scheduler/global_scheduler.py`
- `parrot/serve/context_manager.py`
- `parrot/serve/prefix_matcher.py`
- `parrot/serve/session/graph_executor.py`

实现要点：

- 应用层通过 PFunc / SemanticVariable 暴露 prompt graph。
- ServeCore 把请求拆成 Fill/Generate chain。
- GlobalScheduler 支持 app FIFO、graph grouping、context grouping、context-aware engine selection。
- ContextManager 维护 semantic variable prefix 到 engine context 的映射。

适配判断：

- 证明应用级语义能进入 serving scheduler。
- 但 Parrot 需要应用改写到其 API；OpenClaw 近期更适合用 provider adapter + typed hints。

### 2.6 InferCept / vLLM fork

代码路径：

- `infercept/vllm/core/scheduler.py`
- `infercept/vllm/engine/async_llm_engine.py`
- `infercept/vllm/engine/llm_engine.py`
- `infercept/vllm/worker/`
- `infercept/csrc/`
- `infercept/exps/`

实现要点：

- 仓库 README 明确说明是 InferCept 实现，论文中也给出 `https://github.com/WukLab/InferCept`。
- 仓库保留了完整 `vllm/` 目录和 CUDA extension，属于 vLLM fork，而不是外部 benchmark script。
- 机制上围绕 augmented LLM interception：外部 API/tool 调用打断 decoding 后，比较 preserve、discard/recompute、swap 等 KV handling。

适配判断：

- 可作为 Continuum 之外的 tool/API wait KV lifecycle 对照。
- OpenClaw 比 InferCept benchmark 更清楚 tool start/end、duration、side-effect class，可把这些事件直接转成 SGLang hint。
- InferCept 不处理 OpenClaw 的 mutating tool、approval、replay invalidation；安全边界不能从它照搬。

### 2.7 LAMPS / MARS / vLLM fork

代码路径：

- `mars-codebase/vllm/core/scheduler_v2.py`
- `mars-codebase/vllm/core/policy.py`
- `mars-codebase/prediction/`
- `mars-codebase/exps/`

实现要点：

- 仓库 README 明确写明实现 NeurIPS 2025 论文 “Fast Inference for Augmented Large Language Models”。
- 核心调度逻辑位于 vLLM fork 的 scheduler/policy 路径。
- prediction component 用于估计 API-call 期间 memory handling strategy，调度目标是降低 request completion time 和 TTFT。

适配判断：

- 对方向 A 的启发是 cost model 不应只看当前 request 长度，也应考虑 tool/API wait 期间 KV 的 memory-over-time 占用。
- 对方向 B 的启发是 `expected_reuse_window_ms` 可以由 OpenClaw trace analyzer 生成，再被 SGLang scheduler/cache 共同消费。
- MARS 面向 API-augmented request，不等同于完整 dynamic agent workflow；OpenClaw 需要额外建模 fan-out/fan-in、sub-agent、side effect。

### 2.8 Conveyor

代码路径：

- `conveyor/conveyor/plugin/base_plugin.py`
- `conveyor/conveyor/scheduling/`
- `conveyor/test/`

实现要点：

- 论文与 README 都表明 Conveyor 是 tool-aware LLM serving runtime。
- tool developer 通过继承 `BasePlugin` 暴露 partial execution 接口，例如 `process_new_dat` 和 `finish`。
- scheduler 让可部分执行的 tool 在 LLM decoding 尚未结束时开始处理部分输入。

适配判断：

- 适合启发 OpenClaw read-only speculation 的低风险子集：search、validation、部分 parser-friendly code analysis。
- 不适合直接用于 shell/file-write/network-mutation 等有副作用工具。
- 它解决的是 tool execution overlap，不解决 KV cache residency；需要和 Continuum/KVFlow/MARS 结合看。

### 2.9 Pie（另行规划）

代码路径：

- `Pie/runtime/src/`
- `Pie/inferlets/`
- `Pie/driver/vllm/`
- `Pie/driver/sglang/`
- `Pie/client/python/`
- `Pie/client/rust/`
- `Pie/client/javascript/`

实现要点：

- 论文明确写 Pie open-sourced at `https://github.com/pie-project/pie`。
- 仓库 README 定位为 programmable serving system，inferlet 以 Wasm 形式运行在 serving side。
- 仓库包含 agent、parallel fork、persistent KV、prefix tree、constrained decoding、MCP tools 等 inferlet 示例。
- `driver/vllm` 和 `driver/sglang` 说明它和现有 engine 的关系是 driver/backend integration，而非只改单个 scheduler policy。

适配判断：

- Pie 对本轮近期两方向不是最小路径，转入另行规划；它对远期“agent logic 是否下沉到 inference engine”非常关键。
- 若未来在 SGLang 侧引入更强 programmable extension，Pie 的 Wasm sandbox、inferlet API 和 KV access model 是重点参考。
- 当前 OpenClaw 的 tool safety/provenance/replay 不能直接迁进 Pie 类 runtime；短期仍应让 OpenClaw 执行 tool，SGLang 消费 hints。

---

## 3. SGLang 本地证据路径

### 3.1 Request metadata

代码路径：

- `python/sglang/srt/managers/io_struct.py`
- `python/sglang/srt/entrypoints/openai/serving_base.py`
- `python/sglang/srt/entrypoints/openai/serving_chat.py`
- `python/sglang/srt/entrypoints/openai/serving_completions.py`

观察：

- `GenerateReqInput` 已有 `priority`、`extra_key`、`routing_key`、`custom_labels`、`external_trace_header`。
- OpenAI serving path 会从 request/header 中提取 `extra_key`、`custom_labels`、`x-smg-routing-key`。
- 这些字段足够做 Phase 1 观测和对照实验，但不适合承载完整 agent semantics。

### 3.2 Scheduler policy

代码路径：

- `python/sglang/srt/managers/schedule_policy.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/server_args.py`

观察：

- `lpm` / `dfs-weight` 是 cache-aware policy，会先计算 prefix matches。
- `routing-key` 是 cache-agnostic policy，按 running batch routing key frequency 排序。
- `priority` scheduling 有单独兼容约束，不应作为默认关键路径调度机制。

适配结论：

- agent-aware scheduler 应新增 cache-aware composite policy，而不是复用 `routing-key` 作为主路线。

### 3.3 Prefix cache / HiCache

代码路径：

- `python/sglang/srt/mem_cache/radix_cache.py`
- `python/sglang/srt/mem_cache/hiradix_cache.py`
- `python/sglang/srt/mem_cache/base_prefix_cache.py`
- `python/sglang/srt/managers/cache_controller.py`

观察：

- `RadixKey` 包含 `extra_key`，prefix matching 需要 `extra_key` 一致。
- HiRadix 已有 host/storage hit、load-back、prefetch、write-back/write-through 相关结构。
- CacheController 的 write/load/prefetch operation 可带 priority。

适配结论：

- `extra_key` 只用于 cache namespace，不用于 workflow id。
- KV lifecycle 不需要重写 cache tree；应由 AgentStateManager 把 tool wait / expected reuse 转成 cache-native hints。

---

## 4. OpenClaw 本地证据路径

### 4.1 SGLang provider

代码路径：

- `extensions/sglang/index.ts`
- `extensions/sglang/api.ts`

观察：

- 当前 SGLang extension 主要注册 openai-compatible provider。
- 尚无 SGLang-specific signal adapter。

适配结论：

- 可新增 `extensions/sglang/agent-hints.ts` 和 `extensions/sglang/stream.ts`，把通用 trace/hints 转成 SGLang request payload、custom labels 和 lifecycle event。

### 4.2 Provider stream wrapper

代码路径：

- `src/plugin-sdk/provider-stream-shared.ts`
- `src/agents/pi-embedded-runner/stream-payload-utils.ts`
- 参考：`extensions/vllm/stream.ts`

观察：

- OpenClaw 已有 payload patch wrapper，可在 provider stream 前修改 request payload。
- 现有能力偏 body patch；header patch 需要单独扩展。

适配结论：

- Phase 1/2 的 `custom_labels` 和 `agent_hints` 可以优先通过 body patch 注入。
- `x-smg-routing-key` 只用于对照实验；如需启用，应扩展 header patch。

### 4.3 Tool lifecycle

代码路径：

- `src/agents/pi-embedded-subscribe.handlers.tools.ts`

观察：

- tool start 能获得 tool name、args、run id、tool call id 和 start time。
- tool end 能获得 result/error/duration，并能观察 side effect、replay invalidation 等状态。

适配结论：

- OpenClaw 不需要像 Continuum 一样从模型输出解析 tool call。
- lifecycle event 应从这里产生，再由 SGLang adapter 转成 `/v1/agent/session_event`。

### 4.4 Sub-agent spawn

代码路径：

- `src/agents/acp-spawn.ts`

观察：

- spawn context 已有 agent/session/channel/thread 等信息。
- 还缺少 workflow graph metadata。

适配结论：

- 近期可增加通用 trace 字段：`workflow_id`、`parent_run_id`、`parent_step_id`、`spawn_index`、`fan_in_group_id`。
- 这些字段应保持 provider-agnostic，避免 OpenClaw core 绑定 SGLang。
