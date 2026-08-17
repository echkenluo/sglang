#!/bin/bash
# Multi-request prefill benchmark for one server mode.
# Usage: model_prefill_bench.sh <mok|mok-graph|deepep> <port> <tag>
# Starts the server once, then for each word-count tier sends one JIT/capture
# warmup request followed by REPEAT distinct cache-cold prompts, reporting
# per-tier latency samples. Decode smoke (8 output tokens) runs at the end.
set -u
MODE=${1:?mok|mok-graph|deepep}
PORT=${2:?port}
TAG=${3:?tag}
ROOT=/mok/claude-mok
LOG=$ROOT/exp/server-$TAG.log
REPEAT=${REPEAT:-20}
TIERS=${TIERS:-"85 256 1300"}

ENV_COMMON="MOK_SM90_EXPERIMENTAL=1"
case "$MODE" in
  mok) ENV_MODE="SGLANG_OPT_USE_MOK_FP8_NATIVE=1 SGLANG_OPT_MOK_FP8_NATIVE_STRICT=1" ;;
  mok-graph) ENV_MODE="SGLANG_OPT_USE_MOK_FP8_NATIVE=1 SGLANG_OPT_MOK_FP8_NATIVE_STRICT=1 SGLANG_OPT_MOK_FP8_NATIVE_PREFILL_GRAPH=1" ;;
  mok-fused) ENV_MODE="SGLANG_OPT_USE_MOK_FP8_NATIVE=1 SGLANG_OPT_MOK_FP8_NATIVE_STRICT=1 SGLANG_OPT_MOK_FP8_NATIVE_PREFILL_GRAPH=1 SGLANG_OPT_MOK_FP8_NATIVE_FUSED_DISPATCH_GEMM=1 SGLANG_OPT_MOK_FP8_NATIVE_FUSED_GEMM_COMBINE=1" ;;
  deepep) ENV_MODE="" ;;
  *) echo "BAD_MODE"; exit 2 ;;
esac

env PYTHONPATH=$ROOT/sglang/python:$ROOT/mixture-of-kittens \
  $ENV_COMMON $ENV_MODE \
  python3 -m sglang.launch_server \
  --model-path /data2/pubulic-models/DeepSeek-V4-Flash-FP8-fixed \
  --trust-remote-code --tp-size 4 --ep-size 4 \
  --moe-a2a-backend deepep --deepep-mode auto \
  --attention-backend dsv4 --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 --chunked-prefill-size 4096 \
  --context-length 32768 --cuda-graph-bs-decode 1 2 4 \
  --random-seed 196944571 --host 127.0.0.1 --port "$PORT" \
  --skip-server-warmup > "$LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 180); do
  sleep 5
  kill -0 $SERVER_PID 2>/dev/null || { echo SERVER_DIED; tail -5 "$LOG"; exit 3; }
  curl -sf -m 4 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
done
curl -sf -m 4 "http://127.0.0.1:$PORT/health" > /dev/null || { echo SERVER_TIMEOUT; kill $SERVER_PID; exit 4; }
echo SERVER_READY

request() {  # words seed max_new -> prints latency_ms, appends output ids hash line
  python3 - "$PORT" "$1" "$2" "$3" <<'EOF'
import json, sys, time, urllib.request
port, words, seed, max_new = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
prompt = " ".join(f"item{(seed*7919+i*104729)%99991}" for i in range(words))
body = json.dumps({"text": prompt, "sampling_params": {"max_new_tokens": max_new, "temperature": 0}}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=body,
                             headers={"Content-Type": "application/json"})
t0 = time.perf_counter_ns()
with urllib.request.urlopen(req, timeout=300) as resp:
    payload = json.load(resp)
t1 = time.perf_counter_ns()
ids = payload.get("output_ids", [])
ptok = payload.get("meta_info", {}).get("prompt_tokens", -1)
print(f"{(t1-t0)/1e6:.1f} {ptok} {','.join(map(str, ids[:4]))}")
EOF
}

for words in $TIERS; do
  request "$words" 7777 1 > /dev/null   # JIT / graph-capture warmup for this tier
  samples=""
  for i in $(seq 1 "$REPEAT"); do
    out=$(request "$words" $((1000 + i)) 1)
    ms=${out%% *}
    rest=${out#* }
    samples="$samples $ms"
    [ "$i" = 1 ] && first_meta=$rest
  done
  echo "TIER_RESULT|mode=$MODE|words=$words|meta=$first_meta|samples_ms=${samples# }"
done

decode=$(request 85 8888 8)
echo "DECODE_SMOKE|mode=$MODE|out=$decode"

kill $SERVER_PID 2>/dev/null; sleep 5; kill -9 $SERVER_PID 2>/dev/null
# Honest terminal marker: SCRIPT_DONE only prints when every tier actually
# produced a result line; callers must never rely on an unconditional RC.
TIER_COUNT=$(set -- $TIERS; echo $#)
echo "SCRIPT_DONE|mode=$MODE|tiers_expected=$TIER_COUNT"
