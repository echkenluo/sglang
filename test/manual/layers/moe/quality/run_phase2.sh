#!/bin/bash
# Phase 2 orchestrator (freeze v2 run table).  Sessions:
#   T(S): logprob-target on a split server (targets must precede scores)
#   then per config in frozen order D1, D2, S1, S2, F1:
#     gsm8k -> logprob-score -> longgen within one server session each stage
#     (separate sessions per stage keep the per-stage server logs clean and
#     the path assertions per run).
# Any stage rc != 0 aborts; the evaluator has the final verdict.
set -u
Q=/mok/claude-mok/sglang/test/manual/layers/moe/quality/quality_run.sh
run() {
  local mode=$1 stage=$2 tag=$3
  echo "=== P2 $tag $stage START $(date +%H:%M:%S) ==="
  bash $Q "$mode" 30061 "$stage" "$tag" 0
  local rc=$?
  echo "=== P2 $tag $stage END rc=$rc ==="
  [ $rc -ne 0 ] && { echo "P2_ABORT|$tag|$stage|rc=$rc"; exit $rc; }
  sleep 10
}
run split  logprob-target  t-s
for cfg in "deepep d1" "deepep d2" "split s1" "split s2" "fused f1"; do
  set -- $cfg
  run $1 gsm8k         $2
  run $1 logprob-score $2
  run $1 longgen       $2
done
python3 /mok/claude-mok/sglang/test/manual/layers/moe/quality/quality_gate_eval.py /mok/claude-mok/quality
RC=$?
echo "P2_ALL_DONE|verdict_rc=$RC"
exit $RC
