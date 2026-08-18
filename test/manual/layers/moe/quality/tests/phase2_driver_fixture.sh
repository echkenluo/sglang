#!/bin/bash
# CPU-only fixture for Phase 2 v4 shell-driver tests.
set -uo pipefail

action=${1:?action}
shift

sha() {
  sha256sum "$1" | awk '{print $1}'
}

write_outputs() {
  local qdir=$1 tag=$2 free_run=$3
  printf 'idx,pred,correct\n0,1,1\n' > "$qdir/gsm8k-$tag.csv"
  printf '{"n":1314,"accuracy":1.0,"tag":"%s"}\n' "$tag" \
    > "$qdir/gsm8k-$tag.json"
  printf '{"schema":"teacher512-test","tag":"%s"}\n' "$tag" \
    > "$qdir/teacher512-$tag.json"
  if [ "$free_run" = 1 ]; then
    printf '{"schema":"free-run-info-test","tag":"%s"}\n' "$tag" \
      > "$qdir/free-run-info-$tag.json"
  fi
}

case "$action" in
  server)
    mode=${1:?mode}
    tag=${2:?tag}
    if [ -n "${PHASE2_TEST_RECORD_PROCESS_DIR:-}" ]; then
      mkdir -p "$PHASE2_TEST_RECORD_PROCESS_DIR"
      printf '%s\n' "$$" > "$PHASE2_TEST_RECORD_PROCESS_DIR/server-$tag.pid"
    fi
    echo "TEST_SERVER_READY"
    echo "prefill=PhaseConfig(backend='disabled'"
    echo "disable_radix_cache=True"
    echo "disable_cuda_graph=True"
    echo "disable_overlap_schedule=True"
    echo "disable_flashinfer_autotune=True"
    echo "attention_backend='dsv4'"
    case "$mode" in
      split)
        echo "MoK full-native FP8 active"
        echo "fused_k1=False fused_k2=False"
        ;;
      fused)
        echo "MoK full-native FP8 active"
        echo "fused_k1=True fused_k2=True"
        ;;
      deepep)
        echo "moe_a2a_backend='deepep'"
        ;;
    esac
    trap 'exit 0' TERM INT
    while :; do sleep 1; done
    ;;

  client)
    logical=${1:?logical_stage}
    mode=${2:?mode}
    tag=${3:?tag}
    qdir=${4:?qdir}
    limit=${5:?limit}
    client_program=${6:?client_program}
    shift 6
    client_args=("$@")
    if [ -n "${PHASE2_TEST_RECORD_PROCESS_DIR:-}" ]; then
      mkdir -p "$PHASE2_TEST_RECORD_PROCESS_DIR"
      printf '%s\n' "$$" \
        > "$PHASE2_TEST_RECORD_PROCESS_DIR/client-$tag-$logical.pid"
    fi
    if [ "${PHASE2_TEST_REQUIRE_PATH_CONFIG:-0}" = 1 ]; then
      config_stage=bundle
      [ "$logical" = target-build ] && config_stage=target-build
      [ -f "$qdir/path-config-receipt-$tag-$config_stage.json" ] || exit 25
    fi
    if [ "${PHASE2_TEST_ASSERT_CLIENT_ARGV:-0}" = 1 ]; then
      qsrc=${QUALITY_SOURCE_DIR:?QUALITY_SOURCE_DIR}
      root=${QUALITY_ROOT:-/mok/claude-mok}
      expected_assets=$qsrc/expected_assets.json
      dataset_sha=$(python3 -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["sharegpt_sha256"])' \
        "$expected_assets")
      gsm8k_sha=$(python3 -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["gsm8k_sha256"])' \
        "$expected_assets")
      tokenizer_sha=$(python3 -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["tokenizer_files"]["tokenizer.json"])' \
        "$expected_assets")
      generator_sha=$(sha "$qsrc/logprob_client.py")
      expected_port=${PHASE2_TEST_EXPECT_PORT:?PHASE2_TEST_EXPECT_PORT}
      case "$logical" in
        target-build)
          expected_program=$qsrc/logprob_client.py
          expected_args=(
            --port "$expected_port" --stage target --mode split --tag "$tag"
            --sharegpt /data2/pubulic-models/ShareGPT_V3_unfiltered_cleaned_split.json
            --tokenizer-sha256 "$tokenizer_sha"
            --dataset-sha256 "$dataset_sha"
            --generator-sha256 "$generator_sha"
            --out-dir "$qdir"
          )
          ;;
        gsm8k)
          expected_program=$qsrc/gsm8k_client.py
          expected_args=(
            --port "$expected_port" --data "$root/gsm8k_test.jsonl" --tag "$tag"
            --dataset-sha256 "$gsm8k_sha"
            --limit "$limit" --out-dir "$qdir"
          )
          ;;
        teacher512)
          expected_program=$qsrc/logprob_client.py
          expected_args=(
            --port "$expected_port" --stage score --mode "$mode" --tag "$tag"
            --targets "$qdir/targets-freeze.json"
            --path-config-receipt "$qdir/path-config-receipt-$tag-bundle.json"
            --out-dir "$qdir"
          )
          ;;
        free-run-info)
          expected_program=$qsrc/longgen_client.py
          expected_args=(
            --port "$expected_port" --mode "$mode" --tag "$tag"
            --targets "$qdir/targets-freeze.json"
            --path-config-receipt "$qdir/path-config-receipt-$tag-bundle.json"
            --out-dir "$qdir"
          )
          ;;
        *) exit 26 ;;
      esac
      [ "$client_program" = "$expected_program" ] || {
        echo "CLIENT_PROGRAM_MISMATCH|$logical|$client_program|$expected_program"
        exit 26
      }
      [ "${#client_args[@]}" -eq "${#expected_args[@]}" ] || {
        echo "CLIENT_ARGC_MISMATCH|$logical|${#client_args[@]}|${#expected_args[@]}"
        exit 26
      }
      for i in "${!expected_args[@]}"; do
        [ "${client_args[$i]}" = "${expected_args[$i]}" ] || {
          echo "CLIENT_ARGV_MISMATCH|$logical|$i|${client_args[$i]}|${expected_args[$i]}"
          exit 26
        }
      done
    fi
    if [ "${PHASE2_TEST_FAIL_CLIENT:-}" = "$logical" ] || \
       [ "${PHASE2_TEST_FAIL_CLIENT:-}" = "$tag:$logical" ]; then
      exit 23
    fi
    if [ "${PHASE2_TEST_BLOCK_CLIENT:-}" = "$tag:$logical" ]; then
      printf 'ready\n' > "$qdir/block-client-$tag-$logical.ready"
      trap 'exit 130' INT
      trap 'exit 143' TERM
      while :; do sleep 1; done
    fi
    case "$logical" in
      target-build)
        printf '{"schema":"targets-freeze-test","rows":128}\n' \
          > "$qdir/targets-freeze.json"
        ;;
      gsm8k)
        printf 'idx,pred,correct\n0,1,1\n' > "$qdir/gsm8k-$tag.csv"
        printf '{"n":1314,"accuracy":1.0,"tag":"%s"}\n' "$tag" \
          > "$qdir/gsm8k-$tag.json"
        ;;
      teacher512)
        printf '{"schema":"teacher512-test","tag":"%s"}\n' "$tag" \
          > "$qdir/teacher512-$tag.json"
        ;;
      free-run-info)
        printf '{"schema":"free-run-info-test","tag":"%s"}\n' "$tag" \
          > "$qdir/free-run-info-$tag.json"
        ;;
      *) exit 2 ;;
    esac
    ;;

  preflight)
    qdir=${1:?qdir}
    mkdir -p "$qdir"
    if [ "${PHASE2_TEST_PREFLIGHT_FAIL:-0}" = 1 ]; then
      printf '{"verified":false}\n' > "$qdir/preflight-manifest.json"
      exit 1
    fi
    printf '{"verified":true}\n' > "$qdir/preflight-manifest.json"
    ;;

  session)
    mode=${1:?mode}
    stage=${3:?stage}
    tag=${4:?tag}
    free_run=${6:-0}
    qdir=${QUALITY_DIR:?QUALITY_DIR}
    mkdir -p "$qdir"
    if [ "${PHASE2_TEST_BLOCK_TAG:-}" = "$tag" ]; then
      printf 'ready\n' > "$qdir/block-$tag.ready"
      trap 'exit 143' TERM
      while :; do sleep 1; done
    fi
    if [ "${PHASE2_TEST_STAGE_FAIL_TAG:-}" = "$tag" ]; then
      exit 17
    fi
    if [ "$stage" = target-build ]; then
      printf '{"schema":"targets-freeze-test","rows":128}\n' \
        > "$qdir/targets-freeze.json"
    else
      write_outputs "$qdir" "$tag" "$free_run"
    fi
    config=$qdir/path-config-receipt-$tag-$stage.json
    printf '{"schema":"phase2-v4-path-config-v1","mode":"%s","tag":"%s","stage":"%s"}\n' \
      "$mode" "$tag" "$stage" > "$config"
    config_sha=$(sha "$config")
    freeze_sha=$(sha "$qdir/targets-freeze.json")
    gsm_csv=null
    gsm_json=null
    teacher=null
    free=null
    if [ "$stage" = bundle ]; then
      gsm_csv=\"$(sha "$qdir/gsm8k-$tag.csv")\"
      gsm_json=\"$(sha "$qdir/gsm8k-$tag.json")\"
      teacher=\"$(sha "$qdir/teacher512-$tag.json")\"
      if [ "$free_run" = 1 ]; then
        free=\"$(sha "$qdir/free-run-info-$tag.json")\"
      fi
    fi
    if [ "${PHASE2_TEST_SKIP_RECEIPT_TAG:-}" = "$tag" ]; then
      exit 0
    fi
    receipt=$qdir/path-receipt-$tag-$stage.json
    printf '{"schema":"phase2-v4-session-receipt-v1","mode":"%s","tag":"%s","stage":"%s","rc":0,"path_ok":1,"path_config_sha256":"%s","outputs":{"targets_freeze":"%s","gsm8k_csv":%s,"gsm8k_json":%s,"teacher512":%s,"free_run_info":%s}}\n' \
      "$mode" "$tag" "$stage" "$config_sha" "$freeze_sha" \
      "$gsm_csv" "$gsm_json" "$teacher" "$free" > "$receipt"
    ;;

  checkpoint)
    qdir=${1:?qdir}
    output=${4:?output}
    if [ "${PHASE2_TEST_CHECKPOINT_FAIL:-}" = "${3:-}" ]; then
      exit 19
    fi
    printf '{"all_pass":true}\n' > "$qdir/$output"
    ;;

  verdict)
    qdir=${1:?qdir}
    printf '{"all_pass":true}\n' > "$qdir/quality-gate-verdict.json"
    exit "${PHASE2_TEST_VERDICT_RC:-0}"
    ;;

  *)
    echo "BAD_FIXTURE_ACTION|$action"
    exit 2
    ;;
esac
