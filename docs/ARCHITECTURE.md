# SGLang 架构与实现详解

本文档系统性地梳理 SGLang 代码仓库的架构、原理与实现、核心数据结构和算法。

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [核心组件详解](#2-核心组件详解)
3. [请求处理流程](#3-请求处理流程)
4. [内存管理与KV Cache](#4-内存管理与kv-cache)
5. [调度策略与批处理](#5-调度策略与批处理)
6. [模型执行与Attention后端](#6-模型执行与attention后端)
7. [分布式与并行策略](#7-分布式与并行策略)
8. [前端语言设计](#8-前端语言设计)
9. [高级特性](#9-高级特性)

---

## 1. 整体架构概览

### 1.1 双层架构设计

SGLang 采用**前后端分离**的双层架构:

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (lang/)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ SglFunction │  │ Interpreter │  │ Backend Connectors  │ │
│  │   (IR)      │  │ (执行器)     │  │ (OpenAI/Anthropic)  │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend (srt/)                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                      Engine                          │   │
│  │  ┌───────────────┐ ┌───────────┐ ┌───────────────┐  │   │
│  │  │TokenizerMgr   │ │ Scheduler │ │DetokenizerMgr │  │   │
│  │  └───────────────┘ └───────────┘ └───────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
│                              │                              │
│  ┌───────────────────────────┼───────────────────────────┐ │
│  │                     TpModelWorker                      │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  │ │
│  │  │ ModelRunner │  │ KV Cache    │  │ Attn Backend  │  │ │
│  │  └─────────────┘  └─────────────┘  └───────────────┘  │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 目录结构

```
python/sglang/
├── srt/                          # SGLang Runtime (后端服务引擎)
│   ├── entrypoints/              # 服务入口点
│   │   ├── engine.py             # Engine 主类
│   │   ├── http_server_engine.py # HTTP 服务器
│   │   └── grpc_server.py        # gRPC 服务器
│   ├── managers/                 # 核心管理器
│   │   ├── scheduler.py          # 调度器
│   │   ├── schedule_batch.py     # 批次数据结构
│   │   ├── schedule_policy.py    # 调度策略
│   │   ├── tokenizer_manager.py  # Tokenizer 管理
│   │   ├── detokenizer_manager.py# Detokenizer 管理
│   │   └── tp_worker.py          # Tensor Parallel Worker
│   ├── model_executor/           # 模型执行
│   │   ├── model_runner.py       # 模型运行器
│   │   ├── forward_batch_info.py # 前向批次信息
│   │   └── cuda_graph_runner.py  # CUDA Graph 执行器
│   ├── mem_cache/                # 内存和缓存管理
│   │   ├── memory_pool.py        # 内存池
│   │   ├── radix_cache.py        # Radix 前缀缓存
│   │   └── allocator.py          # 分配器
│   ├── layers/                   # 神经网络层
│   │   ├── attention/            # Attention 后端实现
│   │   ├── radix_attention.py    # RadixAttention 层
│   │   └── moe/                  # MoE 相关
│   ├── models/                   # 模型实现 (130+)
│   ├── sampling/                 # 采样策略
│   ├── speculative/              # 推测解码
│   ├── distributed/              # 分布式支持
│   └── disaggregation/           # Prefill-Decode 分离
│
└── lang/                         # 前端语言
    ├── ir.py                     # 中间表示
    ├── interpreter.py            # 解释器
    ├── api.py                    # 公开 API
    └── backend/                  # 后端连接器
```

---

## 2. 核心组件详解

### 2.1 Engine (引擎入口)

**位置**: `srt/entrypoints/engine.py`

Engine 是整个运行时的入口，负责协调三个核心组件:

```python
class Engine:
    def __init__(self, server_args):
        # 1. 初始化 TokenizerManager (主进程)
        self.tokenizer_manager = TokenizerManager(...)

        # 2. 启动 Scheduler (子进程)
        self.scheduler_process = Process(target=run_scheduler_process, ...)

        # 3. 启动 DetokenizerManager (子进程)
        self.detokenizer_process = Process(target=run_detokenizer_process, ...)
```

**通信机制**: 组件间使用 ZeroMQ (ZMQ) 进行 IPC 通信:
- `scheduler_input_ipc`: TokenizerManager → Scheduler
- `detokenizer_ipc`: Scheduler → DetokenizerManager
- `tokenizer_ipc`: DetokenizerManager → TokenizerManager (返回结果)

### 2.2 TokenizerManager

**位置**: `srt/managers/tokenizer_manager.py`

职责:
1. 接收用户请求 (HTTP/gRPC)
2. 对输入文本进行 tokenization
3. 处理多模态输入 (图像/视频/音频)
4. 将 tokenized 请求发送给 Scheduler

### 2.3 Scheduler (调度器)

**位置**: `srt/managers/scheduler.py`

这是 SGLang 最核心的组件，使用 **Mixin 模式** 组合多个功能:

```python
class Scheduler(
    SchedulerOutputProcessorMixin,      # 输出处理
    SchedulerUpdateWeightsMixin,        # 权重更新
    SchedulerProfilerMixin,             # 性能分析
    SchedulerMetricsMixin,              # 指标收集
    SchedulerDisaggregationDecodeMixin, # PD 分离 (Decode)
    SchedulerDisaggregationPrefillMixin,# PD 分离 (Prefill)
    SchedulerMultiplexMixin,            # 多路复用
    SchedulerRuntimeCheckerMixin,       # 运行时检查
    SchedulerPPMixin,                   # Pipeline Parallel
    SchedulerDPAttnMixin,               # Data Parallel Attention
):
```

**核心数据结构**:

```python
# 等待队列: 新请求等待处理
self.waiting_queue: List[Req] = []

# 运行中批次: 正在进行连续批处理的请求
self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)

# 当前前向批次
self.cur_batch: Optional[ScheduleBatch] = None
```

**主事件循环** (`event_loop_normal`):

```python
def event_loop_normal(self):
    while True:
        # 1. 接收新请求
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)

        # 2. 获取下一个要运行的批次
        batch = self.get_next_batch_to_run()

        # 3. 运行模型推理
        if batch:
            result = self.run_batch(batch)
            self.process_batch_result(batch, result)
```

### 2.4 DetokenizerManager

**位置**: `srt/managers/detokenizer_manager.py`

职责:
1. 接收 Scheduler 输出的 token IDs
2. 执行增量 detokenization
3. 将解码后的文本返回给用户

---

## 3. 请求处理流程

### 3.1 数据流转换链

```
用户请求 → TokenizedGenerateReqInput → Req → ScheduleBatch → ModelWorkerBatch → ForwardBatch
```

每个阶段的数据结构设计目的:

| 结构 | 管理者 | 数据位置 | 用途 |
|------|--------|----------|------|
| `Req` | Scheduler | CPU | 请求级别状态管理 |
| `ScheduleBatch` | Scheduler | CPU | 高层调度数据 |
| `ModelWorkerBatch` | TpWorker | CPU→GPU | 模型前向相关数据子集 |
| `ForwardBatch` | ModelRunner | GPU | 低层张量数据 |

### 3.2 Req (请求) 数据结构

**位置**: `srt/managers/schedule_batch.py:455`

```python
class Req:
    def __init__(self, ...):
        # 基础信息
        self.rid = rid                      # 请求 ID
        self.origin_input_ids = input_ids   # 原始输入 token IDs
        self.output_ids = []                # 生成的输出 token IDs
        self.fill_ids = []                  # origin_input_ids + output_ids

        # 内存管理
        self.req_pool_idx: int              # 在 ReqToTokenPool 中的索引
        self.kv_committed_len = 0           # 已提交的 KV 缓存长度
        self.kv_allocated_len = 0           # 已分配的 KV 缓存长度

        # 前缀缓存
        self.prefix_indices: torch.Tensor   # 共享前缀的 KV 缓存索引
        self.last_node: TreeNode            # Radix Tree 中的最后节点
        self.extend_input_len = 0           # 需要运行 prefill 的 token 数

        # 采样参数
        self.sampling_params: SamplingParams

        # 完成状态
        self.finished_reason: BaseFinishReason = None
```

### 3.3 ForwardMode (前向模式)

**位置**: `srt/model_executor/forward_batch_info.py:69`

```python
class ForwardMode(IntEnum):
    EXTEND = auto()      # Prefill 阶段
    DECODE = auto()      # Decode 阶段 (每次生成一个 token)
    MIXED = auto()       # Chunked Prefill 混合模式
    IDLE = auto()        # 空闲 (DP Attention 时部分 worker)
    TARGET_VERIFY = auto() # 推测解码验证
    DRAFT_EXTEND = auto()  # Draft 模型扩展
    PREBUILT = auto()      # PD 分离时预构建的 KV
    SPLIT_PREFILL = auto() # 分割 Prefill
```

---

## 4. 内存管理与KV Cache

### 4.1 两级内存池架构

**位置**: `srt/mem_cache/memory_pool.py`

```
┌────────────────────────────────────────────────────────────┐
│                     ReqToTokenPool                         │
│  req_to_token[req_idx, :] = [kv_idx_0, kv_idx_1, ...]     │
│  请求 → Token位置 的映射                                   │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│              TokenToKVPoolAllocator                        │
│  管理 KV 缓存槽位的分配与释放                               │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│                      KVCache                               │
│  实际存储物理 KV 缓存数据                                   │
│  k_buffer: [num_layers, max_tokens, num_heads, head_dim]  │
│  v_buffer: [num_layers, max_tokens, num_heads, head_dim]  │
└────────────────────────────────────────────────────────────┘
```

### 4.2 ReqToTokenPool

```python
class ReqToTokenPool:
    def __init__(self, size, max_context_len, device):
        # 二维数组: [max_requests, max_context_len]
        # 存储每个请求每个位置对应的 KV 缓存索引
        self.req_to_token = torch.zeros(
            (size, max_context_len), dtype=torch.int32, device=device
        )
        self.free_slots = list(range(size))  # 可用的请求槽位
```

### 4.3 KV Cache 实现变体

| 类型 | 用途 | 特点 |
|------|------|------|
| `MHATokenToKVPool` | 标准 MHA | 单独存储 K、V |
| `MLATokenToKVPool` | DeepSeek MLA | 压缩 KV 表示 |
| `SWAKVPool` | 滑动窗口注意力 | 只保留窗口内的 KV |
| `DoubleSparseTokenToKVPool` | 稀疏注意力 | 稀疏存储 |
| `NSATokenToKVPool` | Native Sparse Attention | 混合索引 |

### 4.4 RadixAttention 前缀缓存

**位置**: `srt/mem_cache/radix_cache.py`

**核心思想**: 使用 Radix Tree 存储和复用 KV 缓存的前缀。

**TreeNode 结构**:

```python
class TreeNode:
    def __init__(self):
        self.children = defaultdict(TreeNode)  # 子节点映射
        self.parent: TreeNode = None           # 父节点
        self.key: RadixKey = None              # Token ID 序列
        self.value: torch.Tensor = None        # KV 缓存索引
        self.lock_ref = 0                      # 引用计数
        self.last_access_time = time.monotonic()  # 用于 LRU 驱逐
        self.hit_count = 0                     # LFU 计数
```

**匹配算法**:

```python
def match_prefix(self, key: RadixKey) -> MatchResult:
    """查找与给定 key 匹配的最长前缀"""
    node = self.root
    matched_indices = []

    while len(key) > 0:
        # 获取子节点的 key
        child_key = self.get_child_key_fn(key)
        if child_key not in node.children:
            break

        child = node.children[child_key]
        # 计算匹配长度
        match_len = self.key_match_fn(child.key, key)

        if match_len > 0:
            matched_indices.extend(child.value[:match_len])
            key = key[match_len:]
            node = child
        else:
            break

    return MatchResult(matched_indices, node)
```

**驱逐策略**:

```python
# 支持多种策略
self.eviction_strategy: EvictionStrategy = {
    "lru": LRUStrategy(),   # 最近最少使用
    "lfu": LFUStrategy(),   # 最少频率使用
    "fifo": FIFOStrategy(), # 先进先出
    "mru": MRUStrategy(),   # 最近使用
    "filo": FILOStrategy(), # 后进先出
    "priority": PriorityStrategy()  # 优先级
}[policy]
```

---

## 5. 调度策略与批处理

### 5.1 调度策略

**位置**: `srt/managers/schedule_policy.py`

**缓存感知策略**:

```python
class CacheAwarePolicy(Enum):
    LPM = "lpm"           # 最长前缀匹配优先
    DFS_WEIGHT = "dfs-weight"  # DFS 权重排序
```

**缓存无关策略**:

```python
class CacheAgnosticPolicy(Enum):
    FCFS = "fcfs"    # 先来先服务
    LOF = "lof"      # 最长输出优先
    RANDOM = "random"
```

### 5.2 批次调度算法

**`get_next_batch_to_run` 核心逻辑**:

```python
def get_next_batch_to_run(self):
    # 1. 处理运行中的请求 (decode)
    if self.running_batch.reqs:
        # 检查是否可以添加新的 prefill 请求
        can_run_list = self.get_new_batch_prefill()
        if can_run_list:
            return self.prepare_mixed_batch(can_run_list)
        else:
            return self.prepare_decode_batch()

    # 2. 没有运行中的请求，开始新的 prefill
    if self.waiting_queue:
        return self.get_new_batch_prefill()

    return None
```

### 5.3 PrefillAdder - 智能批次构建

```python
class PrefillAdder:
    """确定可以添加到批次中的请求"""

    def add_req(self, req: Req) -> AddReqResult:
        # 检查约束
        if self.total_tokens + req.extend_input_len > self.max_prefill_tokens:
            return AddReqResult.NO_TOKEN
        if len(self.reqs) >= self.max_running_requests:
            return AddReqResult.NO_SLOT

        # 分配 KV 缓存
        tokens_needed = estimate_tokens_needed(req)
        if not self.can_allocate(tokens_needed):
            return AddReqResult.NO_MEM

        self.reqs.append(req)
        return AddReqResult.OK
```

### 5.4 连续批处理 (Continuous Batching)

SGLang 实现了高效的连续批处理:

```python
def prepare_decode_batch(self):
    """准备 decode 批次"""
    batch = self.running_batch

    # 为每个请求分配一个新 token 的空间
    for req in batch.reqs:
        self.allocate_for_decode(req)

    # 移除已完成的请求
    batch.reqs = [r for r in batch.reqs if not r.finished_reason]

    return batch
```

### 5.5 Chunked Prefill

当输入很长时，将 prefill 分成多个 chunk:

```python
def get_new_batch_prefill(self):
    if self.chunked_prefill_size:
        # 分 chunk 处理
        chunk_size = min(req.extend_input_len, self.chunked_prefill_size)
        req.is_chunked = req.extend_input_len > chunk_size
```

---

## 6. 模型执行与Attention后端

### 6.1 TpModelWorker

**位置**: `srt/managers/tp_worker.py`

职责:
- 管理单个 GPU 上的模型执行
- 处理 Tensor Parallel 通信
- 调用 ModelRunner 执行前向传播

### 6.2 ModelRunner

**位置**: `srt/model_executor/model_runner.py`

```python
class ModelRunner:
    def __init__(self, ...):
        # 加载模型
        self.model = get_model(...)

        # 初始化 KV 缓存
        self.init_memory_pool()

        # 初始化 Attention 后端
        self.attn_backend = create_attention_backend(...)

        # 初始化 CUDA Graph (可选)
        self.cuda_graph_runner = CudaGraphRunner(...)

    def forward(self, forward_batch: ForwardBatch):
        # 准备输入
        input_ids = forward_batch.input_ids
        positions = forward_batch.positions

        # 运行模型
        if forward_batch.forward_mode.is_cuda_graph():
            return self.cuda_graph_runner.replay(forward_batch)
        else:
            return self.model.forward(input_ids, positions, forward_batch)
```

### 6.3 Attention 后端架构

**位置**: `srt/layers/attention/`

```python
class AttentionBackend(ABC):
    """Attention 后端抽象基类"""

    @abstractmethod
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """初始化前向传播元数据"""
        pass

    @abstractmethod
    def forward_decode(self, q, k, v, layer, forward_batch):
        """Decode 阶段的 attention"""
        pass

    @abstractmethod
    def forward_extend(self, q, k, v, layer, forward_batch):
        """Prefill/Extend 阶段的 attention"""
        pass
```

**支持的后端**:

| 后端 | 文件 | 特点 |
|------|------|------|
| FlashInfer | `flashinfer_backend.py` | 默认后端，高性能 |
| FlashAttention | `flashattention_backend.py` | Meta FlashAttention |
| Triton | `triton_backend.py` | Triton 实现 |
| FlashMLA | `flashmla_backend.py` | DeepSeek MLA 优化 |
| TBO | `tbo_backend.py` | Token Budget Optimization |
| Intel AMX | `intel_amx_backend.py` | CPU AMX 加速 |

### 6.4 RadixAttention 层

**位置**: `srt/layers/radix_attention.py`

```python
class RadixAttention(nn.Module):
    def __init__(self, num_heads, head_dim, scaling, num_kv_heads, layer_id, ...):
        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_kv_heads
        self.head_dim = head_dim
        self.layer_id = layer_id

    def forward(self, q, k, v, forward_batch: ForwardBatch):
        # 获取当前上下文的 attention 后端
        attn_backend = get_forward_context().attn_backend

        # 调用后端的 forward 方法
        return attn_backend.forward(q, k, v, self, forward_batch)
```

### 6.5 CUDA Graph 优化

**位置**: `srt/model_executor/cuda_graph_runner.py`

```python
class CudaGraphRunner:
    def __init__(self, model_runner):
        # 预先捕获不同 batch size 的 CUDA Graph
        self.graphs = {}
        for bs in [1, 2, 4, 8, 16, ...]:
            self.capture_graph(bs)

    def capture_graph(self, bs):
        # 预热
        for _ in range(3):
            self.model_runner.forward(dummy_batch)

        # 捕获
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.model_runner.forward(dummy_batch)

        self.graphs[bs] = (graph, output)

    def replay(self, forward_batch):
        bs = len(forward_batch)
        graph, output = self.graphs[bs]

        # 更新输入
        self.update_graph_inputs(forward_batch)

        # 重放
        graph.replay()

        return output
```

---

## 7. 分布式与并行策略

### 7.1 并行类型

SGLang 支持多种并行策略:

| 类型 | 参数 | 说明 |
|------|------|------|
| Tensor Parallel | `--tp-size` | 模型权重水平切分 |
| Pipeline Parallel | `--pp-size` | 模型层垂直切分 |
| Data Parallel | `--dp-size` | 数据并行 |
| Expert Parallel | `--ep-size` | MoE 专家并行 |
| DP Attention | `--enable-dp-attention` | 注意力数据并行 |

### 7.2 分布式状态管理

**位置**: `srt/distributed/parallel_state.py`

```python
# 全局进程组
_WORLD_GROUP: Optional[GroupCoordinator] = None

# 张量并行组
_TP_GROUP: Optional[GroupCoordinator] = None

# 流水线并行组
_PP_GROUP: Optional[GroupCoordinator] = None

# 专家并行组
_EP_GROUP: Optional[GroupCoordinator] = None

class GroupCoordinator:
    """进程组协调器"""
    def __init__(self, ranks, local_rank, ...):
        self.ranks = ranks
        self.local_rank = local_rank
        self.rank_in_group = ranks.index(local_rank)
        self.world_size = len(ranks)

        # PyTorch 分布式组
        self.device_group: ProcessGroup
        self.cpu_group: ProcessGroup
```

### 7.3 Tensor Parallel 实现

权重切分模式:

```python
# 列并行: 输出维度切分
class ColumnParallelLinear(nn.Linear):
    def forward(self, x):
        # 每个 rank 计算部分输出
        output = F.linear(x, self.weight)  # weight: [hidden/tp_size, input]
        return output

# 行并行: 输入维度切分
class RowParallelLinear(nn.Linear):
    def forward(self, x):
        output = F.linear(x, self.weight)  # weight: [output, hidden/tp_size]
        # All-Reduce 汇总
        output = all_reduce(output, self.tp_group)
        return output
```

### 7.4 DP Attention (数据并行注意力)

**位置**: `srt/layers/dp_attention.py`

在标准 TP 下，Attention 需要在所有 GPU 上复制 KV 缓存。DP Attention 将不同请求的 KV 缓存分布在不同 GPU 上:

```python
def compute_dp_attention_world_info(enable_dp_attention, tp_rank, tp_size, dp_size):
    if enable_dp_attention:
        # Attention TP size = TP size / DP size
        attn_tp_size = tp_size // dp_size
        attn_tp_rank = tp_rank % attn_tp_size
        attn_dp_rank = tp_rank // attn_tp_size
    else:
        attn_tp_size = tp_size
        attn_tp_rank = tp_rank
        attn_dp_rank = 0

    return attn_tp_rank, attn_tp_size, attn_dp_rank
```

### 7.5 Prefill-Decode 分离 (Disaggregation)

**位置**: `srt/disaggregation/`

将 Prefill 和 Decode 阶段分离到不同的服务器:

```
Prefill 服务器:                    Decode 服务器:
┌─────────────┐                   ┌─────────────┐
│   Prefill   │ ──KV Transfer──► │   Decode    │
│   Server    │                   │   Server    │
└─────────────┘                   └─────────────┘
```

主要组件:
- `PrefillBootstrapQueue`: 管理 prefill 请求的 KV 传输
- `DecodeTransferQueue`: 接收传输的 KV 缓存
- `DecodePreallocQueue`: 预分配 decode 资源

---

## 8. 前端语言设计

### 8.1 SglFunction (SGLang 函数)

**位置**: `lang/ir.py`

```python
class SglFunction:
    """用户定义的 SGLang 程序"""
    def __init__(self, func, num_api_spec_tokens=None):
        self.func = func  # 用户函数
        self.bind_arguments = {}  # 绑定的参数

    def run(self, *args, backend=None, **kwargs):
        """执行程序"""
        from sglang.lang.interpreter import run_program
        return run_program(self, backend, args, kwargs, ...)
```

### 8.2 中间表示 (IR)

**位置**: `lang/ir.py`

```python
# 基础 IR 节点
class SglExpr: pass

# 生成文本
class SglGen(SglExpr):
    def __init__(self, name, regex=None, max_new_tokens=None):
        self.name = name
        self.regex = regex
        self.max_new_tokens = max_new_tokens

# 选择分支
class SglSelect(SglExpr):
    def __init__(self, name, choices, temperature=None):
        self.name = name
        self.choices = choices

# 图像输入
class SglImage(SglExpr):
    def __init__(self, image):
        self.image = image

# 视频输入
class SglVideo(SglExpr):
    def __init__(self, video, num_frames=None):
        self.video = video
```

### 8.3 解释器执行

**位置**: `lang/interpreter.py`

```python
def run_program(program, backend, func_args, func_kwargs, ...):
    # 创建流执行器
    stream_executor = StreamExecutor(backend, ...)

    # 创建程序状态
    state = ProgramState(stream_executor)

    # 执行用户函数
    state.ret_value = program.func(state, *func_args, **func_kwargs)

    return state
```

### 8.4 使用示例

```python
import sglang as sgl

@sgl.function
def multi_turn_chat(s, question1, question2):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question1)
    s += sgl.assistant(sgl.gen("answer1", max_tokens=256))
    s += sgl.user(question2)
    s += sgl.assistant(sgl.gen("answer2", max_tokens=256))

# 执行
state = multi_turn_chat.run(
    question1="What is Python?",
    question2="How to learn it?",
    backend=sgl.Runtime(...)
)

print(state["answer1"])
print(state["answer2"])
```

---

## 9. 高级特性

### 9.1 推测解码 (Speculative Decoding)

**位置**: `srt/speculative/`

支持多种推测解码算法:

```python
class SpeculativeAlgorithm(Enum):
    NONE = "none"
    EAGLE = "eagle"       # EAGLE 算法
    EAGLE2 = "eagle2"     # EAGLE-2 算法
    NGRAM = "ngram"       # N-gram 匹配
```

**EAGLE 工作流**:

```
Draft Model: 生成 k 个候选 token
     │
     ▼
Target Model: 验证候选 token
     │
     ▼
接受匹配的 token, 拒绝不匹配的
```

### 9.2 结构化输出 (Constrained Decoding)

**位置**: `srt/constrained/`

支持:
- JSON Schema 约束
- 正则表达式约束
- CFG 语法约束

```python
# 创建语法后端
grammar_backend = create_grammar_backend(
    server_args,
    tokenizer,
    vocab_size,
    eos_token_id,
)

# 获取下一个有效 token 的 mask
mask = grammar_backend.get_next_token_mask(grammar_obj, current_state)
```

### 9.3 LoRA 支持

**位置**: `srt/lora/`

```python
class LoRAManager:
    def __init__(self, base_model, ...):
        self.base_model = base_model
        self.lora_adapters = {}  # 加载的 LoRA 适配器

    def load_adapter(self, lora_id, path):
        adapter = load_lora_weights(path)
        self.lora_adapters[lora_id] = adapter

    def apply_lora(self, req_lora_ids):
        # 根据请求的 lora_id 动态应用权重
        ...
```

### 9.4 多模态支持

**位置**: `srt/managers/schedule_batch.py`

```python
@dataclasses.dataclass
class MultimodalDataItem:
    modality: Modality  # IMAGE, VIDEO, AUDIO
    feature: torch.Tensor  # 处理后的特征
    pad_value: int  # 用于替换 placeholder token
    offsets: List[int]  # 在输入中的位置

@dataclasses.dataclass
class MultimodalInputs:
    mm_items: List[MultimodalDataItem]
    image_pad_len: Optional[List[int]]
    num_image_tokens: Optional[int]
```

### 9.5 重叠调度 (Overlap Scheduling)

SGLang 支持 CPU 调度和 GPU 计算的重叠:

```python
def event_loop_overlap(self):
    """重叠调度的事件循环"""
    while True:
        # GPU 计算的同时进行 CPU 调度
        with self.forward_stream_ctx:
            # 异步启动 GPU 计算
            result_future = self.run_batch_async(batch)

        # 同时在 CPU 上准备下一个批次
        next_batch = self.get_next_batch_to_run()

        # 等待 GPU 完成
        result = result_future.get()
        self.process_batch_result(batch, result)
```

---

## 附录: 关键数据结构速查

### ServerArgs (服务器参数)

**位置**: `srt/server_args.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `tp_size` | 1 | Tensor Parallel 大小 |
| `dp_size` | 1 | Data Parallel 大小 |
| `pp_size` | 1 | Pipeline Parallel 大小 |
| `chunked_prefill_size` | None | Chunked Prefill 块大小 |
| `schedule_policy` | "lpm" | 调度策略 |
| `attention_backend` | "flashinfer" | Attention 后端 |
| `disable_radix_cache` | False | 是否禁用前缀缓存 |

### ForwardBatch 关键字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `input_ids` | `torch.Tensor` | 输入 token IDs |
| `positions` | `torch.Tensor` | 位置编码 |
| `req_pool_indices` | `torch.Tensor` | 请求池索引 |
| `seq_lens` | `torch.Tensor` | 序列长度 |
| `forward_mode` | `ForwardMode` | 前向模式 |
| `sampling_info` | `SamplingBatchInfo` | 采样信息 |

---

*文档生成日期: 2024*
*SGLang 版本: 基于最新 main 分支*
