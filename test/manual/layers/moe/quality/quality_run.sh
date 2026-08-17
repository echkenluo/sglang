#!/bin/bash
# Phase 2 quality harness driver: start one server config, run the requested
# stage (gsm8k | logprob-target | logprob-score | longgen), tear down.
# Usage: quality_run.sh <mode> <port> <stage> <tag> [limit]
set -u
MODE=${1:?mode}
PORT=${2:?port}
STAGE=${3:?stage}
TAG=${4:?tag}
LIMIT=${5:-0}
ROOT=/mok/claude-mok
QDIR=$ROOT/quality
mkdir -p $QDIR
LOG=$QDIR/server-$TAG.log

ENV_COMMON="MOK_SM90_EXPERIMENTAL=1"
case "$MODE" in
  split)  ENV_MODE="SGLANG_OPT_USE_MOK_FP8_NATIVE=1 SGLANG_OPT_MOK_FP8_NATIVE_STRICT=1" ;;
  fused)  ENV_MODE="SGLANG_OPT_USE_MOK_FP8_NATIVE=1 SGLANG_OPT_MOK_FP8_NATIVE_STRICT=1 SGLANG_OPT_MOK_FP8_NATIVE_FUSED_DISPATCH_GEMM=1 SGLANG_OPT_MOK_FP8_NATIVE_FUSED_GEMM_COMBINE=1" ;;
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
  --disable-radix-cache \
  --random-seed 196944571 --host 127.0.0.1 --port "$PORT" \
  --skip-server-warmup > "$LOG" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 180); do
  sleep 5
  kill -0 $SERVER_PID 2>/dev/null || { echo SERVER_DIED; tail -5 "$LOG"; exit 3; }
  curl -sf -m 4 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
done
curl -sf -m 4 "http://127.0.0.1:$PORT/health" > /dev/null || { echo SERVER_TIMEOUT; kill $SERVER_PID; exit 4; }
echo "SERVER_READY|mode=$MODE|stage=$STAGE|tag=$TAG"

cd $ROOT/sglang/test/manual/layers/moe/quality
case "$STAGE" in
  gsm8k)
    python3 gsm8k_client.py --port $PORT --data $ROOT/gsm8k_test.jsonl \
      --tag $TAG --limit $LIMIT --out-dir $QDIR ;;
  logprob-target)
    python3 logprob_client.py --port $PORT --stage target --tag $TAG \
      --sharegpt /data2/pubulic-models/ShareGPT_V3_unfiltered_cleaned_split.json \
      --out-dir $QDIR ;;
  logprob-score)
    python3 logprob_client.py --port $PORT --stage score --tag $TAG \
      --targets $QDIR/logprob-targets.json --out-dir $QDIR ;;
  longgen)
    python3 longgen_client.py --port $PORT --tag $TAG \
      --sharegpt /data2/pubulic-models/ShareGPT_V3_unfiltered_cleaned_split.json \
      --out-dir $QDIR ;;
  *) echo BAD_STAGE ;;
esac
RC=$?
kill $SERVER_PID 2>/dev/null; sleep 5; kill -9 $SERVER_PID 2>/dev/null
echo "STAGE_DONE|mode=$MODE|stage=$STAGE|tag=$TAG|rc=$RC"
