#!/bin/bash
# Phase 2 orchestrator (freeze v2 run table).  Sessions:
#   T(S): logprob-target on a split server (targets must precede scores)
#   then per config in frozen order D1, D2, S1, S2, F1:
#     gsm8k -> logprob-score -> longgen within one server session each stage
#     (separate sessions per stage keep the per-stage server logs clean and
#     the path assertions per run).
# Any stage rc != 0 aborts; the evaluator has the final verdict.
set -u
QSRC=/mok/claude-mok/sglang/test/manual/layers/moe/quality
Q=$QSRC/quality_run.sh
export QUALITY_DIR=/mok/claude-mok/quality-run-$(date +%Y%m%d-%H%M%S)
python3 $QSRC/preflight_manifest.py "$QUALITY_DIR" "${EXPECT_SGLANG_HEAD:?}" || { echo P2_ABORT_PREFLIGHT; exit 10; }
record_exit() {
  local rc=$?
  printf '{"rc":%d,"quality_dir":"%s"}\n' "$rc" "$QUALITY_DIR" > "$QUALITY_DIR/phase2-exit.json"
  echo "P2_EXIT|rc=$rc|quality_dir=$QUALITY_DIR"
}
trap record_exit EXIT
run() {
  local mode=$1 stage=$2 tag=$3
  echo "=== P2 $tag $stage START $(date +%H:%M:%S) ==="
  QUALITY_DIR=$QUALITY_DIR bash $Q "$mode" 30061 "$stage" "$tag" 0
  local rc=$?
  echo "=== P2 $tag $stage END rc=$rc ==="
  [ $rc -ne 0 ] && { echo "P2_ABORT|$tag|$stage|rc=$rc"; exit $rc; }
  sleep 10
}
checkpoint() {
  local name=$1
  python3 $QSRC/quality_gate_eval.py "$QUALITY_DIR" "$EXPECT_SGLANG_HEAD" "$name"
  local rc=$?
  echo "P2_CHECKPOINT|name=$name|rc=$rc"
  [ $rc -ne 0 ] && { echo "P2_ABORT_GATE|name=$name|rc=$rc"; exit $rc; }
}
run split  logprob-target  t-s
for tag in d1 d2 d3; do
  run deepep gsm8k         $tag
  run deepep logprob-score $tag
  run deepep longgen       $tag
done
checkpoint deepep-aa
for tag in s1 s2 s3; do
  run split gsm8k         $tag
  run split logprob-score $tag
  run split longgen       $tag
done
checkpoint split-aa
run fused gsm8k          f1
run fused logprob-score  f1
run fused longgen        f1
python3 $QSRC/quality_gate_eval.py "$QUALITY_DIR" "$EXPECT_SGLANG_HEAD"
RC=$?
echo "P2_ALL_DONE|verdict_rc=$RC"
exit $RC
