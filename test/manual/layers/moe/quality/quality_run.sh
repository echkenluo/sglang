#!/bin/bash
# Phase 2 quality harness driver: start one server config, run the requested
# stage, assert path engagement from the server log, tear down.  The script
# exit code is the stage's real result (trap-based cleanup never masks it).
# Usage: quality_run.sh <mode> <port> <stage> <tag> [limit]
set -u
MODE=${1:?mode}
PORT=${2:?port}
STAGE=${3:?stage}
TAG=${4:?tag}
LIMIT=${5:-0}
ROOT=/mok/claude-mok
QDIR=${QUALITY_DIR:-$ROOT/quality}
mkdir -p $QDIR
LOG=$QDIR/server-$TAG-$STAGE.log

# Environment hygiene: clear every MoK-related switch inherited from the
# parent, then set exactly the mode's own.
unset SGLANG_OPT_USE_MOK_FP8_NATIVE SGLANG_OPT_MOK_FP8_NATIVE_STRICT \
      SGLANG_OPT_MOK_FP8_NATIVE_PREFILL_GRAPH \
      SGLANG_OPT_MOK_FP8_NATIVE_FUSED_DISPATCH_GEMM \
      SGLANG_OPT_MOK_FP8_NATIVE_FUSED_GEMM_COMBINE \
      SGLANG_OPT_MOK_FP8_NATIVE_FUSED_COPY_CLUSTERS \
      SGLANG_OPT_USE_MOK_FP8_EXPERT_MLP \
      SGLANG_OPT_MOK_FP8_MIN_EXPECTED_M \
      SGLANG_OPT_MOK_FP8_PROFILE_SHAPES
export MOK_SM90_EXPERIMENTAL=1
case "$MODE" in
  split)
    export SGLANG_OPT_USE_MOK_FP8_NATIVE=1 SGLANG_OPT_MOK_FP8_NATIVE_STRICT=1 ;;
  fused)
    export SGLANG_OPT_USE_MOK_FP8_NATIVE=1 SGLANG_OPT_MOK_FP8_NATIVE_STRICT=1 \
           SGLANG_OPT_MOK_FP8_NATIVE_FUSED_DISPATCH_GEMM=1 \
           SGLANG_OPT_MOK_FP8_NATIVE_FUSED_GEMM_COMBINE=1 ;;
  deepep) : ;;
  *) echo "BAD_MODE"; exit 2 ;;
esac

SERVER_PID=""
cleanup() {
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
  sleep 3
  [ -n "$SERVER_PID" ] && kill -9 "$SERVER_PID" 2>/dev/null
}
trap cleanup EXIT

PYTHONPATH=$ROOT/sglang/python:$ROOT/mixture-of-kittens \
  python3 -m sglang.launch_server \
  --model-path /data2/pubulic-models/DeepSeek-V4-Flash-FP8-fixed \
  --trust-remote-code --tp-size 4 --ep-size 4 \
  --moe-a2a-backend deepep --deepep-mode auto \
  --attention-backend dsv4 --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 --chunked-prefill-size 4096 \
  --context-length 32768 --cuda-graph-bs-decode 1 2 4 \
  --cuda-graph-backend-prefill disabled \
  --disable-radix-cache \
  --random-seed 196944571 --host 127.0.0.1 --port "$PORT" \
  --skip-server-warmup > "$LOG" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 180); do
  sleep 5
  kill -0 $SERVER_PID 2>/dev/null || { echo SERVER_DIED; tail -5 "$LOG"; exit 3; }
  curl -sf -m 4 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
done
curl -sf -m 4 "http://127.0.0.1:$PORT/health" > /dev/null || { echo SERVER_TIMEOUT; exit 4; }
echo "SERVER_READY|mode=$MODE|stage=$STAGE|tag=$TAG"

cd $ROOT/sglang/test/manual/layers/moe/quality
RC=1
case "$STAGE" in
  gsm8k)
    python3 gsm8k_client.py --port $PORT --data $ROOT/gsm8k_test.jsonl \
      --tag $TAG --limit $LIMIT --out-dir $QDIR; RC=$? ;;
  logprob-target)
    python3 logprob_client.py --port $PORT --stage target --tag $TAG \
      --sharegpt /data2/pubulic-models/ShareGPT_V3_unfiltered_cleaned_split.json \
      --out-dir $QDIR; RC=$? ;;
  logprob-score)
    python3 logprob_client.py --port $PORT --stage score --tag $TAG \
      --targets $QDIR/logprob-targets.json --out-dir $QDIR; RC=$? ;;
  longgen)
    python3 longgen_client.py --port $PORT --tag $TAG \
      --sharegpt /data2/pubulic-models/ShareGPT_V3_unfiltered_cleaned_split.json \
      --out-dir $QDIR; RC=$? ;;
  *) echo BAD_STAGE; exit 2 ;;
esac

# Path-engagement assertions (frozen regexes; any miss invalidates the run).
PATH_OK=1
grep -aq "prefill=PhaseConfig(backend='disabled'" "$LOG" || { echo "PATH_FAIL|prefill_graph_not_disabled"; PATH_OK=0; }
grep -aq "disable_radix_cache=True" "$LOG" || { echo "PATH_FAIL|radix_not_disabled"; PATH_OK=0; }
case "$MODE" in
  fused)
    grep -aq "MoK full-native FP8 active" "$LOG" || { echo "PATH_FAIL|native_marker_missing"; PATH_OK=0; }
    grep -aq "fused_k1=True fused_k2=True" "$LOG" || { echo "PATH_FAIL|fused_flags_missing"; PATH_OK=0; } ;;
  split)
    grep -aq "MoK full-native FP8 active" "$LOG" || { echo "PATH_FAIL|native_marker_missing"; PATH_OK=0; }
    grep -aq "fused_k1=False fused_k2=False" "$LOG" || { echo "PATH_FAIL|split_flags_not_false"; PATH_OK=0; } ;;
  deepep)
    grep -aq "MoK full-native FP8 active" "$LOG" && { echo "PATH_FAIL|unexpected_native_marker"; PATH_OK=0; }
    grep -aq "moe_a2a_backend='deepep'" "$LOG" || { echo "PATH_FAIL|deepep_backend_not_resolved"; PATH_OK=0; } ;;
esac
[ $PATH_OK = 1 ] || RC=5

printf '{"mode":"%s","stage":"%s","tag":"%s","rc":%d,"path_ok":%d}
'   "$MODE" "$STAGE" "$TAG" "$RC" "$PATH_OK" > "$QDIR/path-receipt-$TAG-$STAGE.json"
echo "STAGE_DONE|mode=$MODE|stage=$STAGE|tag=$TAG|rc=$RC|path_ok=$PATH_OK"
exit $RC
