# 论文覆盖与代码状态复核

> 日期：2026-05-25
> 目的：复核 Stage 2 SGLang + OpenClaw 两条近期方向是否遗漏关键论文，并确认原论文与其他公开途径中的开源代码状态。

---

## 1. 复核口径

本轮按三层口径重新梳理：

1. **Stage1 Ch4 已列出的 agent compute graph / agent serving 论文**：Parrot、Ayo/Teola、SGLang、ALTO、Halo、Autellix、Continuum、KVFlow、Pie、Symphony、Agent.xpu、APIServe/InferCept、vLLM/PagedAttention。
2. **方向 B 相关 tool/API augmentation 论文**：PASTE、B-PASTE、Conveyor、LAMPS/MARS、AsyncLM。
3. **当前实现目标的边界**：是否能直接影响 SGLang scheduler/cache，是否需要 OpenClaw 提供 trace/tool lifecycle，是否只是架构愿景或 on-device 旁支。

代码状态区分为两类：

- **原论文明确给出代码链接**：论文 PDF 中出现 code/source/repository/github 等公开链接。
- **其他途径找到代码**：原论文没有直接贴代码，但可从 GitHub、项目页、作者发布仓库确认实现。

---

## 2. 结论摘要

本轮复核后，覆盖矩阵可以完整记录外延论文，但 **Stage2 design 近期目标只纳入两条方向能直接用上的论文**。

**方向 A 近期纳入：**

- **Autellix**：program-aware scheduling，支撑 workflow/program 作为调度对象。
- **Continuum**：vLLM v1 patch，支撑 tool wait -> KV TTL/pinning。
- **KVFlow / ScaleSim**：SGLang fork，支撑 agent distance/reuse signal -> KV/LoRA eviction/prefetch。
- **InferCept / LAMPS-MARS**：vLLM fork，补齐 API/tool interruption 下的 KV handling 与 memory-over-time scheduling。
- **Halo / Parrot / Ayo**：只作为 graph signal、criticality、workflow IR 的辅助参考，不作为近期 SGLang 改造模板。

**方向 B 近期纳入：**

- **PASTE / B-PASTE**：pattern-aware speculative tool execution 的核心论文。
- **Conveyor**：read-only / parser-friendly tool overlap 的辅助代码参考。
- **Halo critical distance、InferCept、LAMPS-MARS**：分别为 critical path rank、tool/API wait 和 KV lifecycle 提供 bridge。

**另行规划，不进入本轮近期详细设计：**

- **ALTO**：partial-output streaming / nested ancestry，属于更完整 workflow runtime。
- **Pie / Symphony**：programmable inferlet / serve programs，涉及 agent logic 下沉到 serving engine。
- **AsyncLM**：需要模型协议和 interrupt decoding 支持。
- **Agent.xpu**：on-device heterogeneous SoC agent scheduling，部署假设不同。

因此，“真实推理引擎上直接实现”的修正结论仍保留，但只服务于方向 A/B 的近期筛选：**Continuum、KVFlow/ScaleSim、InferCept、LAMPS/MARS** 是需要继续代码深读的 engine patch/fork；Pie 虽有代码，但归入另行规划。

---

## 3. 覆盖矩阵

| 系统 | 机构 | 论文 | 原论文是否给代码 | 其他途径代码状态 | 本地代码 | 纳入判断 |
|------|------|------|------------------|------------------|----------|----------|
| Continuum | UC Berkeley; Stanford University; Tensormesh; Tsinghua University | [arXiv:2511.02230](https://arxiv.org/abs/2511.02230) | 未在论文中找到官方 repo 链接 | 找到 [Hanchenli/vllm-continuum](https://github.com/Hanchenli/vllm-continuum) | `../agent-infra-paper-code/vllm-continuum` | 方向 A 核心实现参考 |
| KVFlow | University of California, San Diego; Amazon Web Services | [arXiv:2507.07400](https://arxiv.org/abs/2507.07400) | 未在论文中找到 repo 链接 | 找到 [PanZaifeng/KVFlow](https://github.com/PanZaifeng/KVFlow) | `../agent-infra-paper-code/KVFlow` | 方向 A 核心实现参考 |
| ScaleSim | University of California, San Diego; Amazon Web Services | [arXiv:2601.21473](https://arxiv.org/abs/2601.21473) | 未在论文中找到 repo 链接 | 同 [PanZaifeng/KVFlow](https://github.com/PanZaifeng/KVFlow) | `../agent-infra-paper-code/KVFlow` | 方向 A 距离信号参考 |
| Autellix | UC Berkeley; Google DeepMind; Shanghai Jiao Tong University | [arXiv:2502.13965](https://arxiv.org/abs/2502.13965) | 未找到 | 未找到官方代码 | 无 | 方向 A 核心论文参考，但 paper-only |
| Halo | National University of Singapore | [arXiv:2509.02121](https://arxiv.org/abs/2509.02121) | 原论文给 anonymous artifact | 找到 [mlsys-io/Halo_demo](https://github.com/mlsys-io/Halo_demo) | `../agent-infra-paper-code/Halo_demo` | 近期辅助参考：critical distance / graph-aware cache locality |
| Parrot | Shanghai Jiao Tong University; Microsoft Research | [USENIX OSDI'24](https://www.usenix.org/conference/osdi24/presentation/lin-chaofan) | 有公开 artifact/code | [microsoft/ParrotServe](https://github.com/microsoft/ParrotServe) | `../agent-infra-paper-code/ParrotServe` | 近期辅助参考：semantic variable / serving graph |
| Teola / Ayo | The Chinese University of Hong Kong; Unaffiliated | [arXiv:2407.00326](https://arxiv.org/abs/2407.00326) | 原论文首页给代码 | [NetX-lab/Ayo](https://github.com/NetX-lab/Ayo) | `../agent-infra-paper-code/Ayo` | 近期辅助参考：primitive DAG / topology-aware batching |
| ALTO | Brown University; NVIDIA; Stanford University; UC Berkeley; Carnegie Mellon University | [arXiv:2403.04311](https://arxiv.org/abs/2403.04311) | 未找到 | 未找到官方代码 | 无 | 另行规划：streaming / nested ancestry |
| SGLang | Stanford University; UC Berkeley; LMSYS | [arXiv:2312.07104](https://arxiv.org/abs/2312.07104) | 有项目仓库 | [sgl-project/sglang](https://github.com/sgl-project/sglang) | 当前项目 `../sglang` | 目标推理引擎 |
| Pie | Yale University | [arXiv:2510.24051](https://arxiv.org/abs/2510.24051) | 原论文明确写 open-sourced at GitHub | [pie-project/pie](https://github.com/pie-project/pie) | `../agent-infra-paper-code/Pie` | 另行规划：programmable inferlet / serving boundary |
| Symphony | Yale University | [arXiv:2510.25412](https://arxiv.org/abs/2510.25412) | 未找到 | 未找到官方代码 | 无 | 另行规划：serve-programs 架构愿景 |
| APIServe / InferCept | University of California, San Diego | [arXiv:2402.01869](https://arxiv.org/abs/2402.01869) | 原论文明确给 GitHub | [WukLab/infercept](https://github.com/WukLab/infercept) | `../agent-infra-paper-code/infercept` | augmented LLM interception 核心实现参考 |
| Conveyor | Duke University | [arXiv:2406.00059](https://arxiv.org/abs/2406.00059) | 原论文明确给 GitHub | [conveyor-sys/conveyor](https://github.com/conveyor-sys/conveyor) | `../agent-infra-paper-code/conveyor` | tool partial execution / overlap 参考 |
| LAMPS / MARS | Harvard University; Tsinghua University | [arXiv:2410.18248](https://arxiv.org/abs/2410.18248) | 未在论文中找到 repo 链接 | 找到 [mars-repository/mars-codebase](https://github.com/mars-repository/mars-codebase) | `../agent-infra-paper-code/mars-codebase` | API-augmented request scheduling 参考 |
| AsyncLM | Yale University | [arXiv:2412.07017](https://arxiv.org/abs/2412.07017) | 未找到 | 未找到官方代码 | 无 | 另行规划：function-call protocol/model |
| PASTE | Shanghai Jiao Tong University; Microsoft Research; Stevens Institute of Technology | [arXiv:2603.18897](https://arxiv.org/abs/2603.18897) | 论文写“after review”开源，未给公开 repo | 未找到官方代码 | 无 | 方向 B 核心论文参考，但 paper-only |
| B-PASTE | Independent Researcher | [arXiv:2604.16469](https://arxiv.org/abs/2604.16469) | 未找到 | 未找到可审计官方代码 | 无 | 方向 B 扩展参考 |
| Agent.xpu | Peking University; University of Hong Kong | [arXiv:2506.24045](https://arxiv.org/abs/2506.24045) | 未找到 | 未找到官方代码 | 无 | 另行规划：on-device SoC 旁支 |
| vLLM / PagedAttention | UC Berkeley | [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) | 有项目仓库 | [vllm-project/vllm](https://github.com/vllm-project/vllm) | `../vllm` | baseline 与 Continuum/InferCept/MARS 对照 |

---

## 4. 代码实现形态再分类

### 4.1 真实推理引擎 patch/fork

这些最值得做代码级对照：

- **Continuum**：vLLM v1 request/scheduler/block manager 路径，增加 job/session/tool wait 语义和 KV pinning。
- **KVFlow / ScaleSim**：SGLang fork，增加 `SScheduler`、`/v1/update`、`AgentManager`、Radix/HiRadix node priority、KV/LoRA prefetch。
- **InferCept**：vLLM fork，在 augmented call interception 处保留/释放/交换 KV，避免每次 tool/API 返回后重算上下文。
- **LAMPS/MARS**：vLLM fork，在 scheduler/policy 层预测 API call 期间的 memory handling strategy，并按 memory-over-time 调度。

### 4.2 自定义 runtime / prototype

这些不是可以直接 patch 到 SGLang 的实现，但抽象有价值：

- **ParrotServe**：自建 ServeCore 与 SemanticVariable API。
- **Ayo**：primitive DAG runtime + graph scheduler + Ray actor engine scheduler。
- **Halo demo**：workflow DAG optimizer + worker/cache movement prototype。
- **Conveyor**：tool partial execution runtime，重点在 tool 与 LLM decoding 的 overlap 接口。

### 4.3 另行规划项

- **Pie / Symphony**：programmable serving / serve programs，涉及 agent logic 下沉到 serving engine。
- **ALTO**：token/partial-output streaming 粒度的 compound-AI orchestration。
- **AsyncLM**：异步 function-call，需要模型/protocol 配合。
- **Agent.xpu**：on-device heterogeneous SoC agent scheduling。

---

## 5. 对当前两条方向的影响

### 5.1 方向 A：图感知调度 / KV lifecycle

方向 A 的代码阅读优先级应调整为：

1. **KVFlow/ScaleSim**：SGLang fork，直接对应我们要改的 engine。
2. **Continuum**：vLLM patch，提供 tool wait -> KV residency 的最清晰实现。
3. **InferCept + MARS**：补齐 API/tool augmentation 场景下 vLLM 系统如何处理 pause、swap、schedule。
4. **Autellix**：paper-only，但支撑 program-level scheduling 论点。
5. **Halo/Parrot/Ayo**：作为 graph signal、criticality、workflow IR 的辅助参考。

### 5.2 方向 B：关键路径与 read-only speculation

方向 B 的近期文献谱系应收束为三类：

1. **Speculative tool execution**：PASTE/B-PASTE。
2. **Tool partial execution overlap**：Conveyor。
3. **Interception/API-call memory scheduling**：InferCept、MARS、Continuum。

因此，近期不应只说“PASTE 没代码，所以没有可读实现”。方向 B 仍然有可读系统参考：

- Conveyor 可参考 tool partial execution 的 interface/scheduler。
- InferCept/MARS 可参考 API/tool pause 期间 KV handling 与 scheduling。

但真正的 OpenClaw speculation 仍需遵守 side-effect 边界；这些系统大多没有处理 Codex/OpenClaw 类写文件、执行命令、修改外部状态的安全问题。

---

## 6. 下一步审计建议

短期代码深读顺序：

1. **KVFlow/ScaleSim**：拆清 `SScheduler`、`AgentManager`、Radix/HiRadix priority、LoRA/KV prefetch 哪些属于 KVFlow，哪些属于 ScaleSim。
2. **Continuum + InferCept + MARS**：对比 vLLM 中三种 pause/tool/API handling：pinning、preserve/discard/swap、memory-over-time scheduling。
3. **Conveyor**：评估 tool partial execution 是否只适合 parser-friendly 工具，哪些 OpenClaw read-only tool 能用。
4. **Halo/Parrot/Ayo**：只抽取 criticality、semantic variable、primitive DAG 的 signal 设计，不进入 SGLang 热路径实现。
5. **ALTO/Pie/Symphony/AsyncLM/Agent.xpu**：转入另行计划，不作为近期实现依赖。
