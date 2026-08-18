#!/bin/bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
QSRC=$(cd "$HERE/.." && pwd)
FIXTURE=$HERE/phase2_driver_fixture.sh
DRIVER=$QSRC/run_phase2.sh
TMP_ROOT=$(mktemp -d /tmp/phase2-orchestrator-test.XXXXXX)
EXPECT_HEAD=0123456789abcdef0123456789abcdef01234567
cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

run_phase2() {
  local qdir=$1
  shift
  env \
    PHASE2_DRIVER_TESTING=1 \
    PHASE2_QUALITY_SOURCE_DIR="$QSRC" \
    PHASE2_PYTHON_BIN=python3 \
    PHASE2_TEST_PREFLIGHT_RUNNER="$FIXTURE" \
    PHASE2_TEST_SESSION_RUNNER="$FIXTURE" \
    PHASE2_TEST_CHECKPOINT_RUNNER="$FIXTURE" \
    PHASE2_TEST_VERDICT_RUNNER="$FIXTURE" \
    PHASE2_SESSION_PAUSE_SECONDS=0 \
    QUALITY_DIR="$qdir" \
    EXPECT_SGLANG_HEAD="$EXPECT_HEAD" \
    "$@" bash "$DRIVER"
}

json_rc() {
  python3 - "$1" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["rc"])
PY
}

# Happy path proves the exact ten-receipt set reaches the verdict.
GOOD=$TMP_ROOT/good
run_phase2 "$GOOD"
[ "$(json_rc "$GOOD/phase2-exit.json")" -eq 0 ]
[ "$(find "$GOOD" -maxdepth 1 -name 'path-receipt-*.json' | wc -l)" -eq 10 ]
[ -f "$GOOD/quality-gate-verdict.json" ]
echo "RUN_PHASE2_TEN_RECEIPTS_OK"

# Preflight failure must be normalized to rc=10 after the EXIT trap exists.
PREFLIGHT=$TMP_ROOT/preflight
set +e
run_phase2 "$PREFLIGHT" PHASE2_TEST_PREFLIGHT_FAIL=1
RC=$?
set -e
[ "$RC" -eq 10 ]
[ "$(json_rc "$PREFLIGHT/phase2-exit.json")" -eq 10 ]
echo "RUN_PHASE2_PREFLIGHT_RC10_OK"

# A non-zero child/session stage must propagate unchanged.
STAGE_FAIL=$TMP_ROOT/stage-fail
set +e
run_phase2 "$STAGE_FAIL" PHASE2_TEST_STAGE_FAIL_TAG=d2
RC=$?
set -e
[ "$RC" -eq 17 ]
[ "$(json_rc "$STAGE_FAIL/phase2-exit.json")" -eq 17 ]
echo "RUN_PHASE2_CHILD_RC_OK"

# TERM must terminate the active child and produce the authoritative 143.
TERM_DIR=$TMP_ROOT/term
env \
  PHASE2_DRIVER_TESTING=1 \
  PHASE2_QUALITY_SOURCE_DIR="$QSRC" \
  PHASE2_PYTHON_BIN=python3 \
  PHASE2_TEST_PREFLIGHT_RUNNER="$FIXTURE" \
  PHASE2_TEST_SESSION_RUNNER="$FIXTURE" \
  PHASE2_TEST_CHECKPOINT_RUNNER="$FIXTURE" \
  PHASE2_TEST_VERDICT_RUNNER="$FIXTURE" \
  PHASE2_SESSION_PAUSE_SECONDS=0 \
  PHASE2_TEST_BLOCK_TAG=d1 \
  QUALITY_DIR="$TERM_DIR" \
  EXPECT_SGLANG_HEAD="$EXPECT_HEAD" \
  bash "$DRIVER" > "$TMP_ROOT/term.log" 2>&1 &
TERM_PID=$!
for _ in $(seq 1 500); do
  [ -f "$TERM_DIR/block-d1.ready" ] && break
  kill -0 "$TERM_PID" 2>/dev/null || break
  sleep 0.01
done
[ -f "$TERM_DIR/block-d1.ready" ]
set +e
kill -TERM "$TERM_PID"
wait "$TERM_PID"
RC=$?
set -e
[ "$RC" -eq 143 ]
[ "$(json_rc "$TERM_DIR/phase2-exit.json")" -eq 143 ]
echo "RUN_PHASE2_TERM143_OK"

# A runner returning zero without one of the ten receipts must fail closed.
MISSING=$TMP_ROOT/missing
set +e
run_phase2 "$MISSING" PHASE2_TEST_SKIP_RECEIPT_TAG=f3
RC=$?
set -e
[ "$RC" -eq 11 ]
[ "$(json_rc "$MISSING/phase2-exit.json")" -eq 11 ]
[ ! -f "$MISSING/quality-gate-verdict.json" ]
echo "RUN_PHASE2_RECEIPT_SET_NEGATIVE_OK"
