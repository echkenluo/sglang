#!/bin/bash
# Phase 2 v4 one-server session driver.
#
# Usage:
#   quality_run.sh <mode> <port> <target-build|bundle> <tag> [limit] [free_run]
#
# A bundle keeps one fresh server alive for GSM8K -> teacher512 -> optional
# free-run-info.  An immutable path-config receipt is published before any
# client runs.  The final session receipt contains content hashes of that
# config and every output; outputs never contain the final receipt hash, so
# the receipt graph is acyclic.
set -uo pipefail

MODE=${1:?mode}
PORT=${2:?port}
STAGE=${3:?stage}
TAG=${4:?tag}
LIMIT=${5:-0}
FREE_RUN=${6:-0}

ROOT=${QUALITY_ROOT:-/mok/claude-mok}
QSRC=${QUALITY_SOURCE_DIR:-$ROOT/sglang/test/manual/layers/moe/quality}
QDIR=${QUALITY_DIR:-$ROOT/quality}
LOG=$QDIR/server-$TAG-$STAGE.log
PYTHON_BIN=${QUALITY_PYTHON_BIN:-python3}
TESTING=${PHASE2_DRIVER_TESTING:-0}

if [ "$TESTING" != 1 ] && {
  [ -n "${QUALITY_TEST_SERVER_LAUNCHER:-}" ] ||
  [ -n "${QUALITY_TEST_CLIENT_RUNNER:-}" ];
}; then
  echo "TEST_OVERRIDE_REJECTED"
  exit 2
fi
case "$STAGE" in
  target-build|bundle) ;;
  *) echo "BAD_STAGE|$STAGE"; exit 2 ;;
esac
case "$MODE" in
  split|fused|deepep) ;;
  *) echo "BAD_MODE|$MODE"; exit 2 ;;
esac
if [ "$STAGE" = target-build ] && [ "$MODE" != split ]; then
  echo "BAD_TARGET_MODE|$MODE"
  exit 2
fi
case "$FREE_RUN" in
  0|1) ;;
  *) echo "BAD_FREE_RUN|$FREE_RUN"; exit 2 ;;
esac
case "$PORT" in
  ''|*[!0-9]*) echo "BAD_PORT|$PORT"; exit 2 ;;
esac
case "$LIMIT" in
  ''|*[!0-9]*) echo "BAD_LIMIT|$LIMIT"; exit 2 ;;
esac

mkdir -p "$QDIR"
umask 022

SERVER_PID=""
cleanup() {
  local rc=$?
  trap - EXIT
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    if [ "$TESTING" = 1 ]; then
      wait "$SERVER_PID" 2>/dev/null || true
    else
      sleep 3
      kill -9 "$SERVER_PID" 2>/dev/null || true
      wait "$SERVER_PID" 2>/dev/null || true
    fi
  fi
  exit "$rc"
}
trap cleanup EXIT

atomic_write() {
  local final=$1 content=$2 tmp=$1.tmp.$$
  if [ -e "$final" ]; then
    echo "RECEIPT_EXISTS|$final"
    return 12
  fi
  printf '%s\n' "$content" > "$tmp" || return 12
  sync -f "$tmp" 2>/dev/null || true
  chmod 0444 "$tmp" || return 12
  if ! ln "$tmp" "$final"; then
    rm -f "$tmp"
    return 12
  fi
  rm -f "$tmp"
}

file_sha() {
  sha256sum "$1" | awk '{print $1}'
}

output_hash() {
  local path=$1
  if [ ! -f "$path" ]; then
    printf 'null'
    return 1
  fi
  local digest
  digest=$(file_sha "$path") || return 1
  chmod a-w "$path" 2>/dev/null || true
  printf '"%s"' "$digest"
}

# Environment hygiene: clear every MoK-related switch inherited from the
# parent, then set exactly the requested mode.
unset SGLANG_OPT_USE_MOK_FP8_NATIVE SGLANG_OPT_MOK_FP8_NATIVE_STRICT \
      SGLANG_OPT_MOK_FP8_NATIVE_PREFILL_GRAPH \
      SGLANG_OPT_MOK_FP8_NATIVE_FUSED_DISPATCH_GEMM \
      SGLANG_OPT_MOK_FP8_NATIVE_FUSED_GEMM_COMBINE \
      SGLANG_OPT_MOK_FP8_NATIVE_FUSED_COPY_CLUSTERS \
      SGLANG_OPT_USE_MOK_FP8_EXPERT_MLP \
      SGLANG_OPT_MOK_FP8_MIN_EXPECTED_M \
      SGLANG_OPT_MOK_FP8_PROFILE_SHAPES \
      SGLANG_OPT_MOK_FP8_NUMERIC_AUDIT
export MOK_SM90_EXPERIMENTAL=1
case "$MODE" in
  split)
    export SGLANG_OPT_USE_MOK_FP8_NATIVE=1
    export SGLANG_OPT_MOK_FP8_NATIVE_STRICT=1
    ;;
  fused)
    export SGLANG_OPT_USE_MOK_FP8_NATIVE=1
    export SGLANG_OPT_MOK_FP8_NATIVE_STRICT=1
    export SGLANG_OPT_MOK_FP8_NATIVE_FUSED_DISPATCH_GEMM=1
    export SGLANG_OPT_MOK_FP8_NATIVE_FUSED_GEMM_COMBINE=1
    ;;
  deepep) ;;
esac

if [ "$TESTING" = 1 ]; then
  : "${QUALITY_TEST_SERVER_LAUNCHER:?missing test server launcher}"
  "$QUALITY_TEST_SERVER_LAUNCHER" server "$MODE" "$TAG" "$STAGE" \
    > "$LOG" 2>&1 &
else
  PYTHONPATH=$ROOT/sglang/python:$ROOT/mixture-of-kittens \
    "$PYTHON_BIN" -m sglang.launch_server \
    --model-path /data2/pubulic-models/DeepSeek-V4-Flash-FP8-fixed \
    --trust-remote-code --tp-size 4 --ep-size 4 \
    --moe-a2a-backend deepep --deepep-mode auto \
    --attention-backend dsv4 --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.8 --chunked-prefill-size 4096 \
    --context-length 32768 \
    --disable-cuda-graph \
    --cuda-graph-backend-prefill disabled \
    --disable-radix-cache \
    --disable-overlap-schedule \
    --disable-flashinfer-autotune \
    --random-seed 196944571 --host 127.0.0.1 --port "$PORT" \
    --skip-server-warmup > "$LOG" 2>&1 &
fi
SERVER_PID=$!

if [ "$TESTING" = 1 ]; then
  for _ in $(seq 1 100); do
    kill -0 "$SERVER_PID" 2>/dev/null || {
      echo "SERVER_DIED"
      exit 3
    }
    grep -aq "TEST_SERVER_READY" "$LOG" && break
    sleep 0.01
  done
  grep -aq "TEST_SERVER_READY" "$LOG" || {
    echo "SERVER_TIMEOUT"
    exit 4
  }
else
  for _ in $(seq 1 180); do
    sleep 5
    kill -0 "$SERVER_PID" 2>/dev/null || {
      echo "SERVER_DIED"
      tail -5 "$LOG"
      exit 3
    }
    curl -sf -m 4 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  done
  curl -sf -m 4 "http://127.0.0.1:$PORT/health" >/dev/null || {
    echo "SERVER_TIMEOUT"
    exit 4
  }
fi
echo "SERVER_READY|mode=$MODE|stage=$STAGE|tag=$TAG"

# The immutable config receipt precedes all client work.  It records the
# requested path contract and the startup-resolved common configuration;
# mode-specific runtime engagement is checked again after client execution.
STARTUP_OK=1
grep -aq "prefill=PhaseConfig(backend='disabled'" "$LOG" || STARTUP_OK=0
grep -aq "disable_radix_cache=True" "$LOG" || STARTUP_OK=0
grep -aq "disable_cuda_graph=True" "$LOG" || STARTUP_OK=0
grep -aq "disable_overlap_schedule=True" "$LOG" || STARTUP_OK=0
grep -aq "disable_flashinfer_autotune=True" "$LOG" || STARTUP_OK=0
grep -aq "attention_backend='dsv4'" "$LOG" || STARTUP_OK=0

PATH_CONFIG=$QDIR/path-config-receipt-$TAG-$STAGE.json
PATH_CONFIG_JSON=$(printf \
  '{"schema":"phase2-v4-path-config-v1","mode":"%s","tag":"%s","stage":"%s","port":%s,"free_run":%s,"startup_ok":%s,"prefill_backend":"disabled","cuda_graph":false,"radix_cache":false,"overlap_schedule":false,"flashinfer_autotune":false,"attention_backend":"dsv4"}' \
  "$MODE" "$TAG" "$STAGE" "$PORT" "$FREE_RUN" "$STARTUP_OK")
atomic_write "$PATH_CONFIG" "$PATH_CONFIG_JSON" || exit $?
PATH_CONFIG_SHA=$(file_sha "$PATH_CONFIG") || exit 12

SHAREGPT=/data2/pubulic-models/ShareGPT_V3_unfiltered_cleaned_split.json
TARGETS_FREEZE=$QDIR/targets-freeze.json
EXPECTED_ASSETS=$QSRC/expected_assets.json
if [ ! -f "$EXPECTED_ASSETS" ]; then
  echo "EXPECTED_ASSETS_MISSING|$EXPECTED_ASSETS"
  exit 10
fi
read_expected_asset() {
  "$PYTHON_BIN" -c \
    'import json,sys; value=json.load(open(sys.argv[1])); [value:=value[key] for key in sys.argv[2:]]; print(value)' \
    "$EXPECTED_ASSETS" "$@"
}
DATASET_SHA256=$(read_expected_asset sharegpt_sha256) || exit 10
GSM8K_SHA256=$(read_expected_asset gsm8k_sha256) || exit 10
TOKENIZER_SHA256=$(read_expected_asset tokenizer_files tokenizer.json) || exit 10
GENERATOR_SHA256=$(file_sha "$QSRC/logprob_client.py") || exit 10

invoke_client() {
  local logical_stage=$1
  local client_program
  local -a client_args
  case "$logical_stage" in
    target-build)
      client_program=$QSRC/logprob_client.py
      client_args=(
        --port "$PORT" --stage target --mode split --tag "$TAG"
        --sharegpt "$SHAREGPT"
        --tokenizer-sha256 "$TOKENIZER_SHA256"
        --dataset-sha256 "$DATASET_SHA256"
        --generator-sha256 "$GENERATOR_SHA256"
        --out-dir "$QDIR"
      )
      ;;
    gsm8k)
      client_program=$QSRC/gsm8k_client.py
      client_args=(
        --port "$PORT" --data "$ROOT/gsm8k_test.jsonl" --tag "$TAG"
        --dataset-sha256 "$GSM8K_SHA256"
        --limit "$LIMIT" --out-dir "$QDIR"
      )
      ;;
    teacher512)
      client_program=$QSRC/logprob_client.py
      client_args=(
        --port "$PORT" --stage score --mode "$MODE" --tag "$TAG"
        --targets "$TARGETS_FREEZE"
        --path-config-receipt "$PATH_CONFIG"
        --out-dir "$QDIR"
      )
      ;;
    free-run-info)
      client_program=$QSRC/longgen_client.py
      client_args=(
        --port "$PORT" --mode "$MODE" --tag "$TAG"
        --targets "$TARGETS_FREEZE"
        --path-config-receipt "$PATH_CONFIG"
        --out-dir "$QDIR"
      )
      ;;
    *) return 2 ;;
  esac
  if [ "$TESTING" = 1 ]; then
    : "${QUALITY_TEST_CLIENT_RUNNER:?missing test client runner}"
    "$QUALITY_TEST_CLIENT_RUNNER" client "$logical_stage" "$MODE" "$TAG" \
      "$QDIR" "$LIMIT" "$client_program" "${client_args[@]}"
  else
    "$PYTHON_BIN" "$client_program" "${client_args[@]}"
  fi
}

RC=0
CLIENT_TARGET_RC=null
CLIENT_GSM8K_RC=null
CLIENT_TEACHER_RC=null
CLIENT_FREE_RUN_RC=null
if [ "$STARTUP_OK" != 1 ]; then
  echo "PATH_FAIL|startup_contract"
  RC=5
elif [ "$STAGE" = target-build ]; then
  invoke_client target-build
  CLIENT_TARGET_RC=$?
  [ "$CLIENT_TARGET_RC" -eq 0 ] || RC=$CLIENT_TARGET_RC
else
  invoke_client gsm8k
  CLIENT_GSM8K_RC=$?
  [ "$CLIENT_GSM8K_RC" -eq 0 ] || RC=$CLIENT_GSM8K_RC
  if [ "$RC" -eq 0 ]; then
    invoke_client teacher512
    CLIENT_TEACHER_RC=$?
    [ "$CLIENT_TEACHER_RC" -eq 0 ] || RC=$CLIENT_TEACHER_RC
  fi
  if [ "$RC" -eq 0 ] && [ "$FREE_RUN" = 1 ]; then
    invoke_client free-run-info
    CLIENT_FREE_RUN_RC=$?
    [ "$CLIENT_FREE_RUN_RC" -eq 0 ] || RC=$CLIENT_FREE_RUN_RC
  fi
fi

# Runtime path engagement is checked after requests have exercised the MoE
# path.  A config receipt has already preserved the requested pre-client state.
PATH_OK=$STARTUP_OK
case "$MODE" in
  fused)
    grep -aq "MoK full-native FP8 active" "$LOG" || PATH_OK=0
    grep -aq "fused_k1=True fused_k2=True" "$LOG" || PATH_OK=0
    ;;
  split)
    grep -aq "MoK full-native FP8 active" "$LOG" || PATH_OK=0
    grep -aq "fused_k1=False fused_k2=False" "$LOG" || PATH_OK=0
    ;;
  deepep)
    grep -aq "MoK full-native FP8 active" "$LOG" && PATH_OK=0
    grep -aq "moe_a2a_backend='deepep'" "$LOG" || PATH_OK=0
    ;;
esac
if [ "$PATH_OK" != 1 ] && [ "$RC" -eq 0 ]; then
  echo "PATH_FAIL|runtime_engagement"
  RC=5
fi

GSM8K_CSV=$QDIR/gsm8k-$TAG.csv
GSM8K_JSON=$QDIR/gsm8k-$TAG.json
TEACHER=$QDIR/teacher512-$TAG.json
FREE_RUN_OUT=$QDIR/free-run-info-$TAG.json

TARGETS_FREEZE_SHA=$(output_hash "$TARGETS_FREEZE") || true
GSM8K_CSV_SHA=$(output_hash "$GSM8K_CSV") || true
GSM8K_JSON_SHA=$(output_hash "$GSM8K_JSON") || true
TEACHER_SHA=$(output_hash "$TEACHER") || true
FREE_RUN_SHA=$(output_hash "$FREE_RUN_OUT") || true

# A client returning success without its contracted output is a driver error.
if [ "$RC" -eq 0 ]; then
  if [ "$STAGE" = target-build ]; then
    [ "$TARGETS_FREEZE_SHA" != null ] || RC=6
  else
    [ "$GSM8K_CSV_SHA" != null ] && [ "$GSM8K_JSON_SHA" != null ] && \
      [ "$TEACHER_SHA" != null ] || RC=6
    if [ "$FREE_RUN" = 1 ] && [ "$FREE_RUN_SHA" = null ]; then
      RC=6
    fi
  fi
fi

RECEIPT=$QDIR/path-receipt-$TAG-$STAGE.json
RECEIPT_JSON=$(printf \
  '{"schema":"phase2-v4-session-receipt-v1","mode":"%s","tag":"%s","stage":"%s","rc":%s,"path_ok":%s,"free_run":%s,"path_config_sha256":"%s","client_rc":{"target":%s,"gsm8k":%s,"teacher512":%s,"free_run_info":%s},"outputs":{"targets_freeze":%s,"gsm8k_csv":%s,"gsm8k_json":%s,"teacher512":%s,"free_run_info":%s}}' \
  "$MODE" "$TAG" "$STAGE" "$RC" "$PATH_OK" "$FREE_RUN" \
  "$PATH_CONFIG_SHA" "$CLIENT_TARGET_RC" "$CLIENT_GSM8K_RC" \
  "$CLIENT_TEACHER_RC" "$CLIENT_FREE_RUN_RC" "$TARGETS_FREEZE_SHA" \
  "$GSM8K_CSV_SHA" "$GSM8K_JSON_SHA" \
  "$TEACHER_SHA" "$FREE_RUN_SHA")
atomic_write "$RECEIPT" "$RECEIPT_JSON" || {
  write_rc=$?
  [ "$RC" -ne 0 ] || RC=$write_rc
}

echo "SESSION_DONE|mode=$MODE|stage=$STAGE|tag=$TAG|rc=$RC|path_ok=$PATH_OK|path_config_sha=$PATH_CONFIG_SHA"
exit "$RC"
