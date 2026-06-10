# GASR Anthropic Cache Export Patch

> 日期：2026-06-09
> 定位：GASR Claude Code request-capture / cache tier 观测所需的 SGLang serving patch 记录

## 1. 背景

Claude Code 通过 Anthropic-compatible `/v1/messages` streaming path 访问
SGLang。GASR 需要在 request-capture proxy 和 AHCG graph 中看到每个 LLM
request 的 cache count 与 cache tier，才能区分：

- 成功请求的 aggregate cache hit / full miss；
- SGLang 是否返回 `device/host/storage` tier；
- Claude Code / AHCG 记录链路是否丢失扩展字段。

如果 SGLang 重启后只使用未打补丁的 baseline image，即使
`--enable-cache-report` 打开，Claude Code 路径也可能只保留
`cache_read_input_tokens`，缺少 `sglang_cached_tokens_details`，导致 GASR
报告只能得到 count-only 或 unknown。

## 2. 当前 patch 内容

当前 sglang 工作区的本地 patch 涉及：

- `python/sglang/srt/entrypoints/anthropic/protocol.py`
  - `AnthropicUsage` 增加 `prompt_tokens_details` 与
    `sglang_cached_tokens_details`。
  - `AnthropicMessagesRequest` 接受 `return_cached_tokens_details`、`rid`、
    `extra_key`、`cache_salt`，便于 GASR / proxy 做 request-level 追踪与
    cache report 控制。
  - stream / response model 增加 `sglext`。
- `python/sglang/srt/entrypoints/anthropic/serving.py`
  - 将 OpenAI serving chunk 中的 `prompt_tokens_details.cached_tokens` 和
    `sglext.cached_tokens_details` 转成 Anthropic `message_delta.usage`。
  - 在 non-stream response 中同步暴露 usage cache details 与 top-level
    `sglext`。
  - 默认跟随 server `enable_cache_report` 返回 cached token details，也可由
    request body `return_cached_tokens_details` 显式覆盖。
- `python/sglang/srt/entrypoints/openai/serving_chat.py`
  和 `serving_completions.py`
  - streaming 中即使某个 index 暂无 tier detail，也保留
    `cached_tokens_details[index] = None` 的占位，避免后续 chunk 合并时 key
    缺失。

## 3. 重启后的必查项

SGLang lane 重启或换镜像后，不要只看 container 名字或端口。至少确认：

1. live 文件中存在 patch 字段：
   - `return_cached_tokens_details`
   - `sglang_cached_tokens_details`
   - `sglext`
2. direct `/v1/messages` streaming probe 在重复 prompt 后，最终
   `message_delta.usage` 能看到：
   - `prompt_tokens_details.cached_tokens`
   - `sglang_cached_tokens_details.device/host/storage`
3. GASR `--claude-request-capture-proxy` canary 的
   `request-capture/anthropic-sse-events.jsonl` 中有 completion cache info；
   对应抽表后 `request_export_completion_cache_info_events > 0`。
4. AHCG API 进程加载的是当前 adapter/builder 代码。若 adapter 刚修过，重启
   `8790/8791` 后再 `reingest-ahcg-graphs.py`。

## 4. 与 proxy 的关系

request-capture proxy 只是透明 HTTP/SSE 观测旁路，不负责生成 cache tier。
它能记录 Claude Code 实际收到的 Anthropic SSE event；是否有 tier 取决于
SGLang Anthropic adapter 是否把 OpenAI serving 的 `sglext` / cache report
转写进 Anthropic response。

因此，GASR Claude Code 正式 run 的最小闭环是：

```text
SGLang Anthropic patch
  -> /v1/messages response carries cache details
  -> request-capture proxy records raw SSE
  -> AHCG request_export overlay joins by message id
  -> extract-gasr-tables.py reports hit/full-miss/unknown/tier
```

如果 proxy 中有成功 `/v1/messages` completion 但 graph/token row 没有 join，
报告会显示 `proxy-only`。`proxy-only` 不包含 `/count_tokens`、proxy error 或
incomplete request；这些只作为数据质量字段解释。
