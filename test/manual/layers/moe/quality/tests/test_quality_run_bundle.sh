#!/bin/bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
QSRC=$(cd "$HERE/.." && pwd)
FIXTURE=$HERE/phase2_driver_fixture.sh
TMP_ROOT=$(mktemp -d /tmp/phase2-quality-run-test.XXXXXX)
cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

run_driver() {
  local qdir=$1
  shift
  local expected_port=$2
  PHASE2_DRIVER_TESTING=1 \
  PHASE2_TEST_REQUIRE_PATH_CONFIG=1 \
  PHASE2_TEST_ASSERT_CLIENT_ARGV=1 \
  PHASE2_TEST_EXPECT_PORT="$expected_port" \
  QUALITY_DIR="$qdir" \
  QUALITY_ROOT="$TMP_ROOT/root" \
  QUALITY_SOURCE_DIR="$QSRC" \
  QUALITY_TEST_SERVER_LAUNCHER="$FIXTURE" \
  QUALITY_TEST_CLIENT_RUNNER="$FIXTURE" \
    bash "$QSRC/quality_run.sh" "$@"
}

GOOD=$TMP_ROOT/good
mkdir -p "$GOOD"
run_driver "$GOOD" split 31001 target-build t-s 0 0
run_driver "$GOOD" split 31001 bundle s1 0 1

python3 - "$GOOD" <<'PY'
import hashlib
import json
import os
import stat
import sys

qdir = sys.argv[1]

def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

receipt_path = os.path.join(qdir, "path-receipt-s1-bundle.json")
receipt = json.load(open(receipt_path))
assert receipt["rc"] == 0 and receipt["path_ok"] == 1
assert stat.S_IMODE(os.stat(receipt_path).st_mode) == 0o444
config = os.path.join(qdir, "path-config-receipt-s1-bundle.json")
assert sha(config) == receipt["path_config_sha256"]
assert stat.S_IMODE(os.stat(config).st_mode) == 0o444
paths = {
    "targets_freeze": "targets-freeze.json",
    "gsm8k_csv": "gsm8k-s1.csv",
    "gsm8k_json": "gsm8k-s1.json",
    "teacher512": "teacher512-s1.json",
    "free_run_info": "free-run-info-s1.json",
}
for key, name in paths.items():
    path = os.path.join(qdir, name)
    assert sha(path) == receipt["outputs"][key]
    assert stat.S_IMODE(os.stat(path).st_mode) & 0o222 == 0
    assert "path-receipt" not in open(path, errors="ignore").read()
assert "receipt_sha" not in receipt
assert receipt["client_rc"] == {
    "target": None,
    "gsm8k": 0,
    "teacher512": 0,
    "free_run_info": 0,
}
print("QUALITY_RUN_BUNDLE_OK")
PY

FAIL=$TMP_ROOT/fail
mkdir -p "$FAIL"
run_driver "$FAIL" split 31002 target-build t-s 0 0
set +e
PHASE2_TEST_FAIL_CLIENT=teacher512 run_driver "$FAIL" split 31002 bundle s1 0 1
RC=$?
set -e
[ "$RC" -eq 23 ]
python3 - "$FAIL/path-receipt-s1-bundle.json" <<'PY'
import json
import sys

receipt = json.load(open(sys.argv[1]))
assert receipt["rc"] == 23
assert receipt["client_rc"]["gsm8k"] == 0
assert receipt["client_rc"]["teacher512"] == 23
assert receipt["client_rc"]["free_run_info"] is None
assert receipt["outputs"]["teacher512"] is None
print("QUALITY_RUN_SUBSTAGE_FAILURE_OK")
PY
