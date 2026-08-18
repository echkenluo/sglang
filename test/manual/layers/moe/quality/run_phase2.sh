#!/bin/bash
# Phase 2 v4 orchestrator.  Ten fresh server sessions:
#   T(S), D1-D3, S1-S3, F1-F3.
# D and S checkpoints run before F1; the split-only band is immutable before
# any fused output exists.  The EXIT trap is installed before any variable or
# required-environment expansion and before preflight.
set -uo pipefail

ACTIVE_PID=""
ACTIVE_PGID=""

group_has_live_processes() {
  local pgid=$1
  ps -eo pgid=,stat= 2>/dev/null | awk -v wanted="$pgid" '
    $1 == wanted && $2 !~ /^Z/ { live = 1 }
    END { exit(live ? 0 : 1) }
  '
}

drain_active_group() {
  local signal_name=${1:-TERM}
  local pid=${ACTIVE_PID:-} pgid=${ACTIVE_PGID:-}
  if [ -z "$pid" ] || [ -z "$pgid" ]; then
    ACTIVE_PID=""
    ACTIVE_PGID=""
    return 0
  fi

  # Every formal session owns a fresh session/process group.  Signal the
  # group, not only its bash leader: the active Python client and all server
  # workers must be unable to outlive the authoritative outer exit receipt.
  kill -"$signal_name" -- "-$pgid" 2>/dev/null || true
  for _ in $(seq 1 100); do
    group_has_live_processes "$pgid" || break
    sleep 0.05
  done
  if group_has_live_processes "$pgid"; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in $(seq 1 100); do
      group_has_live_processes "$pgid" || break
      sleep 0.05
    done
  fi
  if group_has_live_processes "$pgid"; then
    kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
  ACTIVE_PID=""
  ACTIVE_PGID=""
}

record_exit() {
  local rc=$?
  trap - EXIT INT TERM
  drain_active_group TERM
  local qdir=${QUALITY_DIR:-/mok/claude-mok/quality-run-uninitialized}
  mkdir -p "$qdir" 2>/dev/null || true
  local tmp=$qdir/.phase2-exit.$$.tmp
  if printf '{"schema":"phase2-v4-exit-v1","rc":%d,"quality_dir":"%s"}\n' \
      "$rc" "$qdir" > "$tmp"; then
    sync -f "$tmp" 2>/dev/null || true
    mv -f "$tmp" "$qdir/phase2-exit.json" 2>/dev/null || true
  fi
  echo "P2_EXIT|rc=$rc|quality_dir=$qdir"
  exit "$rc"
}

on_signal() {
  local rc=$1 signal_name=$2
  trap - "$signal_name"
  drain_active_group "$signal_name"
  exit "$rc"
}

trap record_exit EXIT
trap 'on_signal 130 INT' INT
trap 'on_signal 143 TERM' TERM

# No parameter expansion above this point is evaluated before the traps.
QSRC=${PHASE2_QUALITY_SOURCE_DIR:-/mok/claude-mok/sglang/test/manual/layers/moe/quality}
Q=${PHASE2_SESSION_DRIVER:-$QSRC/quality_run.sh}
PYTHON_BIN=${PHASE2_PYTHON_BIN:-python3}
TESTING=${PHASE2_DRIVER_TESTING:-0}
QUALITY_DIR=${QUALITY_DIR:-/mok/claude-mok/quality-run-$(date +%Y%m%d-%H%M%S)}
EXPECT_HEAD=${EXPECT_SGLANG_HEAD:?EXPECT_SGLANG_HEAD must be a full SHA}
export QUALITY_DIR

if [ "$TESTING" != 1 ] && {
  [ -n "${PHASE2_TEST_PREFLIGHT_RUNNER:-}" ] ||
  [ -n "${PHASE2_TEST_SESSION_RUNNER:-}" ] ||
  [ -n "${PHASE2_TEST_CHECKPOINT_RUNNER:-}" ] ||
  [ -n "${PHASE2_TEST_VERDICT_RUNNER:-}" ];
}; then
  echo "TEST_OVERRIDE_REJECTED"
  exit 2
fi

run_preflight() {
  if [ "$TESTING" = 1 ] && [ -n "${PHASE2_TEST_PREFLIGHT_RUNNER:-}" ]; then
    "$PHASE2_TEST_PREFLIGHT_RUNNER" preflight "$QUALITY_DIR" "$EXPECT_HEAD"
  else
    "$PYTHON_BIN" "$QSRC/preflight_manifest.py" "$QUALITY_DIR" "$EXPECT_HEAD"
  fi
}

run_preflight
PREFLIGHT_RC=$?
if [ "$PREFLIGHT_RC" -ne 0 ]; then
  echo "P2_ABORT_PREFLIGHT|inner_rc=$PREFLIGHT_RC|outer_rc=10"
  exit 10
fi

run_session() {
  local mode=$1 stage=$2 tag=$3 free_run=${4:-0}
  echo "P2_SESSION_START|mode=$mode|tag=$tag|stage=$stage"
  if [ "$TESTING" = 1 ] && [ -n "${PHASE2_TEST_SESSION_RUNNER:-}" ]; then
    setsid env --default-signal=INT --default-signal=TERM \
      "$PHASE2_TEST_SESSION_RUNNER" session \
      "$mode" 30061 "$stage" "$tag" 0 "$free_run" &
  else
    setsid env --default-signal=INT --default-signal=TERM \
      bash "$Q" "$mode" 30061 "$stage" "$tag" 0 "$free_run" &
  fi
  ACTIVE_PID=$!
  ACTIVE_PGID=$ACTIVE_PID
  local observed_pgid
  observed_pgid=$(ps -o pgid= -p "$ACTIVE_PID" 2>/dev/null | tr -d ' ')
  if [ -n "$observed_pgid" ] && [ "$observed_pgid" != "$ACTIVE_PGID" ]; then
    echo "P2_ABORT_PROCESS_GROUP|pid=$ACTIVE_PID|pgid=$observed_pgid"
    drain_active_group TERM
    exit 12
  fi
  echo "P2_SESSION_GROUP|mode=$mode|tag=$tag|pid=$ACTIVE_PID|pgid=$ACTIVE_PGID"
  wait "$ACTIVE_PID"
  local rc=$?
  ACTIVE_PID=""
  # The leader's own EXIT cleanup should already have closed the group.  If a
  # server worker ignored it, fail-safe cleanup prevents contamination of the
  # next fresh session without changing the leader's authoritative rc.
  if group_has_live_processes "$ACTIVE_PGID"; then
    local completed_pgid=$ACTIVE_PGID
    ACTIVE_PID=$completed_pgid
    drain_active_group TERM
  else
    ACTIVE_PGID=""
  fi
  echo "P2_SESSION_END|mode=$mode|tag=$tag|stage=$stage|rc=$rc"
  if [ "$rc" -ne 0 ]; then
    echo "P2_ABORT_SESSION|mode=$mode|tag=$tag|stage=$stage|rc=$rc"
    exit "$rc"
  fi
  local pause=${PHASE2_SESSION_PAUSE_SECONDS:-10}
  if [ "$pause" != 0 ]; then
    sleep "$pause"
  fi
}

run_checkpoint() {
  local name=$1 expected_output=$2
  if [ "$TESTING" = 1 ] && [ -n "${PHASE2_TEST_CHECKPOINT_RUNNER:-}" ]; then
    "$PHASE2_TEST_CHECKPOINT_RUNNER" checkpoint "$QUALITY_DIR" "$EXPECT_HEAD" "$name" "$expected_output"
  else
    "$PYTHON_BIN" "$QSRC/quality_gate_eval.py" "$QUALITY_DIR" "$EXPECT_HEAD" "$name"
  fi
  local rc=$?
  echo "P2_CHECKPOINT|name=$name|rc=$rc"
  if [ "$rc" -ne 0 ]; then
    echo "P2_ABORT_GATE|name=$name|rc=$rc"
    exit "$rc"
  fi
  if [ ! -f "$QUALITY_DIR/$expected_output" ]; then
    echo "P2_ABORT_GATE|name=$name|missing=$expected_output"
    exit 9
  fi
  chmod a-w "$QUALITY_DIR/$expected_output" 2>/dev/null || true
}

verify_receipts() {
  "$PYTHON_BIN" - "$QUALITY_DIR" <<'PY'
import hashlib
import json
import os
import sys

qdir = sys.argv[1]
expected = {
    ("split", "t-s", "target-build"),
    *(("deepep", f"d{i}", "bundle") for i in range(1, 4)),
    *(("split", f"s{i}", "bundle") for i in range(1, 4)),
    *(("fused", f"f{i}", "bundle") for i in range(1, 4)),
}

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

seen = set()
failures = []
receipt_files = sorted(
    name for name in os.listdir(qdir)
    if name.startswith("path-receipt-") and name.endswith(".json")
)
for name in receipt_files:
    path = os.path.join(qdir, name)
    try:
        payload = json.load(open(path))
    except Exception as error:
        failures.append(f"{name}:json:{error}")
        continue
    key = (payload.get("mode"), payload.get("tag"), payload.get("stage"))
    want_name = f"path-receipt-{key[1]}-{key[2]}.json"
    if name != want_name or key not in expected:
        failures.append(f"{name}:unexpected:{key}")
    if payload.get("schema") != "phase2-v4-session-receipt-v1":
        failures.append(f"{name}:schema")
    if payload.get("rc") != 0 or payload.get("path_ok") != 1:
        failures.append(f"{name}:status")
    config = os.path.join(qdir, f"path-config-receipt-{key[1]}-{key[2]}.json")
    if not os.path.isfile(config) or sha(config) != payload.get("path_config_sha256"):
        failures.append(f"{name}:path-config")
    outputs = payload.get("outputs", {})
    required = ["targets_freeze"]
    output_paths = {
        "targets_freeze": "targets-freeze.json",
        "gsm8k_csv": f"gsm8k-{key[1]}.csv",
        "gsm8k_json": f"gsm8k-{key[1]}.json",
        "teacher512": f"teacher512-{key[1]}.json",
        "free_run_info": f"free-run-info-{key[1]}.json",
    }
    if key[2] == "bundle":
        required += ["gsm8k_csv", "gsm8k_json", "teacher512"]
        if key[1] in {"d1", "s1", "f1"}:
            required += ["free_run_info"]
    if any(not isinstance(outputs.get(field), str) for field in required):
        failures.append(f"{name}:outputs")
    else:
        for field in required:
            output_path = os.path.join(qdir, output_paths[field])
            if not os.path.isfile(output_path) or sha(output_path) != outputs[field]:
                failures.append(f"{name}:output-sha:{field}")
    seen.add(key)

if seen != expected:
    failures.append(f"set:missing={sorted(expected-seen)}:extra={sorted(seen-expected)}")
if len(receipt_files) != 10:
    failures.append(f"count={len(receipt_files)}")
if failures:
    print("P2_RECEIPTS_FAIL|" + "|".join(failures[:10]), flush=True)
    raise SystemExit(11)
print("P2_RECEIPTS_OK|count=10", flush=True)
PY
}

run_session split target-build t-s 0

run_session deepep bundle d1 1
run_session deepep bundle d2 0
run_session deepep bundle d3 0
run_checkpoint deepep-env deepep-env-checkpoint.json

run_session split bundle s1 1
run_session split bundle s2 0
run_session split bundle s3 0
run_checkpoint split-aa-freeze split-aa-freeze.json
SPLIT_AA_FREEZE_SHA=$(sha256sum "$QUALITY_DIR/split-aa-freeze.json" | awk '{print $1}')
export SPLIT_AA_FREEZE_SHA
echo "P2_SPLIT_AA_FROZEN|sha=$SPLIT_AA_FREEZE_SHA"

run_session fused bundle f1 1
run_session fused bundle f2 0
run_session fused bundle f3 0

verify_receipts
RECEIPT_RC=$?
if [ "$RECEIPT_RC" -ne 0 ]; then
  echo "P2_ABORT_RECEIPTS|rc=$RECEIPT_RC"
  exit "$RECEIPT_RC"
fi

if [ "$TESTING" = 1 ] && [ -n "${PHASE2_TEST_VERDICT_RUNNER:-}" ]; then
  "$PHASE2_TEST_VERDICT_RUNNER" verdict "$QUALITY_DIR" "$EXPECT_HEAD"
else
  "$PYTHON_BIN" "$QSRC/quality_gate_eval.py" "$QUALITY_DIR" "$EXPECT_HEAD"
fi
RC=$?
echo "P2_ALL_DONE|verdict_rc=$RC"
exit "$RC"
