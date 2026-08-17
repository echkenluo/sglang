#!/bin/bash
# Capture a single cache-cold prefill request trace from a V4-Flash server.
# Usage: model_prefill_trace.sh <mok|deepep> <port> <prompt_tokens> <tag>
# Runs inside the GPU0-3 container. Starts the server, warms JIT with three
# throwaway prompts, profiles exactly one fresh prefill request, stops the
# server. Trace lands in /mok/claude-mok/exp/model-trace-<tag>/.
set -u
MODE=${1:?mok|deepep}
PORT=${2:?port}
PTOK=${3:?prompt tokens}
TAG=${4:?tag}
ROOT=/mok/claude-mok
TRACE_DIR=$ROOT/exp/model-trace-$TAG
LOG=$ROOT/exp/server-$TAG.log
mkdir -p "$TRACE_DIR"

ENV_COMMON="SGLANG_TORCH_PROFILER_DIR=$TRACE_DIR MOK_SM90_EXPERIMENTAL=1"
if [ "$MODE" = mok ]; then
  ENV_MODE="SGLANG_OPT_USE_MOK_FP8_NATIVE=1 SGLANG_OPT_MOK_FP8_NATIVE_STRICT=1"
else
  ENV_MODE=""
fi

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
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "SERVER_DIED rc=$(wait $SERVER_PID; echo $?)"
    tail -5 "$LOG"
    exit 3
  fi
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    echo SERVER_READY
    break
  fi
done
curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo SERVER_TIMEOUT; kill $SERVER_PID; exit 4; }

gen_prompt() {  # deterministic distinct word salad, roughly $1 tokens
  python3 - "$1" "$2" <<'EOF'
import sys
n, seed = int(sys.argv[1]), int(sys.argv[2])
words = [f"item{(seed*7919+i*104729)%99991}" for i in range(n)]
print(" ".join(words))
EOF
}

for i in 1 2 3; do
  P=$(gen_prompt "$PTOK" "$i")
  curl -sf -X POST "http://127.0.0.1:$PORT/generate" -H 'Content-Type: application/json' \
    -d "{\"text\": \"$P\", \"sampling_params\": {\"max_new_tokens\": 1, \"temperature\": 0}}" > /dev/null
  echo "WARMUP_$i done"
done

curl -sf -X POST "http://127.0.0.1:$PORT/start_profile" -H 'Content-Type: application/json' \
  -d '{"activities": ["CPU", "GPU"]}' > /dev/null && echo PROFILE_STARTED

P=$(gen_prompt "$PTOK" 99)
T0=$(date +%s%N)
curl -sf -X POST "http://127.0.0.1:$PORT/generate" -H 'Content-Type: application/json' \
  -d "{\"text\": \"$P\", \"sampling_params\": {\"max_new_tokens\": 1, \"temperature\": 0}}" > "$TRACE_DIR/response.json"
T1=$(date +%s%N)
echo "PROFILED_REQUEST_MS=$(( (T1 - T0) / 1000000 ))"

curl -sf -X POST "http://127.0.0.1:$PORT/stop_profile" > /dev/null && echo PROFILE_STOPPED
sleep 8

kill $SERVER_PID 2>/dev/null
sleep 5
kill -9 $SERVER_PID 2>/dev/null
echo "TRACE_FILES=$(ls "$TRACE_DIR" | head -8 | tr '\n' ' ')"
echo SCRIPT_DONE
