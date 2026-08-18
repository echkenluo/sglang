#!/bin/bash
# Exercise the formal outer -> quality_run -> server/client process topology.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
QSRC=$(cd "$HERE/.." && pwd)
FIXTURE=$HERE/phase2_driver_fixture.sh
DRIVER=$QSRC/run_phase2.sh
TMP_ROOT=$(mktemp -d /tmp/phase2-process-group-test.XXXXXX)
EXPECT_HEAD=0123456789abcdef0123456789abcdef01234567
cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

json_rc() {
  python3 - "$1" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["rc"])
PY
}

wait_gone() {
  local pid=$1
  for _ in $(seq 1 200); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.01
  done
  return 1
}

run_signal_case() {
  local signal_name=$1 expected_rc=$2
  local name
  name=$(printf '%s' "$signal_name" | tr 'A-Z' 'a-z')
  local qdir=$TMP_ROOT/$name
  local process_dir=$TMP_ROOT/$name-processes
  local log=$TMP_ROOT/$name.log
  mkdir -p "$qdir" "$process_dir"

  env --default-signal=INT --default-signal=TERM \
    PHASE2_DRIVER_TESTING=1 \
    PHASE2_QUALITY_SOURCE_DIR="$QSRC" \
    PHASE2_SESSION_DRIVER="$QSRC/quality_run.sh" \
    PHASE2_PYTHON_BIN=python3 \
    PHASE2_TEST_PREFLIGHT_RUNNER="$FIXTURE" \
    PHASE2_TEST_CHECKPOINT_RUNNER="$FIXTURE" \
    PHASE2_TEST_VERDICT_RUNNER="$FIXTURE" \
    PHASE2_SESSION_PAUSE_SECONDS=0 \
    PHASE2_TEST_RECORD_PROCESS_DIR="$process_dir" \
    PHASE2_TEST_BLOCK_CLIENT=d1:gsm8k \
    QUALITY_SOURCE_DIR="$QSRC" \
    QUALITY_ROOT="$TMP_ROOT/root" \
    QUALITY_TEST_SERVER_LAUNCHER="$FIXTURE" \
    QUALITY_TEST_CLIENT_RUNNER="$FIXTURE" \
    QUALITY_DIR="$qdir" \
    EXPECT_SGLANG_HEAD="$EXPECT_HEAD" \
    bash "$DRIVER" > "$log" 2>&1 &
  local driver_pid=$!

  for _ in $(seq 1 1000); do
    [ -f "$qdir/block-client-d1-gsm8k.ready" ] && break
    kill -0 "$driver_pid" 2>/dev/null || break
    sleep 0.01
  done
  [ -f "$qdir/block-client-d1-gsm8k.ready" ]
  [ -s "$process_dir/server-d1.pid" ]
  [ -s "$process_dir/client-d1-gsm8k.pid" ]

  kill -"$signal_name" "$driver_pid"
  for _ in $(seq 1 1000); do
    kill -0 "$driver_pid" 2>/dev/null || break
    sleep 0.01
  done
  if kill -0 "$driver_pid" 2>/dev/null; then
    kill -KILL "$driver_pid" 2>/dev/null || true
    echo "DRIVER_SIGNAL_TIMEOUT|$signal_name"
    return 1
  fi
  set +e
  wait "$driver_pid"
  local rc=$?
  set -e
  [ "$rc" -eq "$expected_rc" ]
  [ "$(json_rc "$qdir/phase2-exit.json")" -eq "$expected_rc" ]
  grep -Fxq "P2_EXIT|rc=$expected_rc|quality_dir=$qdir" "$log"

  local pid_file child_pid
  for pid_file in \
    "$process_dir/server-d1.pid" \
    "$process_dir/client-d1-gsm8k.pid"; do
    child_pid=$(cat "$pid_file")
    wait_gone "$child_pid" || {
      echo "PROCESS_SURVIVED|$signal_name|$pid_file|$child_pid"
      return 1
    }
  done
  echo "PHASE2_PROCESS_GROUP_${signal_name}_OK|rc=$rc"
}

run_signal_case TERM 143
run_signal_case INT 130
