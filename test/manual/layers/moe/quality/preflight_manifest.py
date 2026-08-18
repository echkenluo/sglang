"""Phase 2 preflight: STRONG verification of every frozen asset BEFORE T(S).

Every expectation lives in expected_assets.json (versioned with the harness;
script integrity is covered by the sglang HEAD + clean-worktree assertions).
Any mismatch is a non-zero exit -- nothing is merely recorded when a frozen
value exists.  Writes preflight-manifest.json with verified flags into the
(empty) run directory.

Args: <run_dir> <expect_sglang_head_full> ; env IMAGE_ID from the launcher
(docker inspect on the host) is compared against the frozen image id.
"""

import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys

ROOT = "/mok/claude-mok"
QSRC = f"{ROOT}/sglang/test/manual/layers/moe/quality"
MODEL_DIR = "/data2/pubulic-models/DeepSeek-V4-Flash-FP8-fixed"
SHAREGPT = "/data2/pubulic-models/ShareGPT_V3_unfiltered_cleaned_split.json"
FAILS = []
POWER_INPUT_NAME = "phase2-v4-power-input.json"
POWER_EVALUATOR_NAME = "phase2_v4_power.py"
POWER_INPUT_SCHEMA = "phase2-v4-raw-power-input-v2"
POWER_ASSESSMENT_SCHEMA = "phase2-v4-raw-power-assessment-v2"
POWER_FAMILIES = {"teacher68", "gsm5", "full73"}


def fail(kind, detail):
    FAILS.append(kind)
    print(f"PREFLIGHT_FAIL|{kind}|{detail}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_sha256(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _load_power_evaluator(path):
    module_name = "phase2_v4_power_preflight"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load power evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def verify_power_asset(exp, qsrc=QSRC, expected_heads=None):
    """Recompute power from bounded raw control assets before T(S)."""
    asset = exp.get("phase2_v4_power_input")
    configured_path = asset.get("path") if isinstance(asset, dict) else None
    expected_sha = asset.get("sha256") if isinstance(asset, dict) else None
    input_path = os.path.join(qsrc, configured_path or POWER_INPUT_NAME)
    evaluator_path = os.path.join(qsrc, POWER_EVALUATOR_NAME)
    result = {
        "verified": False,
        "input_path": input_path,
        "evaluator_path": evaluator_path,
        "expected_sha256": expected_sha,
        "actual_sha256": None,
        "schema": None,
        "candidate_blind": None,
        "control_only": None,
        "source_shas": {},
        "raw_source_digest": None,
        "validation": None,
        "assessment": None,
    }

    if (
        not isinstance(asset, dict)
        or not isinstance(configured_path, str)
        or not configured_path
        or os.path.isabs(configured_path)
        or os.path.normpath(configured_path) != configured_path
        or configured_path.startswith("..")
    ):
        fail("power_input_config", asset or "MISSING")
        return result
    if not _is_sha256(expected_sha):
        kind = (
            "power_input_expected_pending"
            if isinstance(expected_sha, str) and expected_sha.startswith("PENDING")
            else "power_input_expected_sha256"
        )
        fail(kind, expected_sha or asset or "MISSING")
        return result
    if not os.path.isfile(input_path):
        fail("power_input_missing", input_path)
        return result
    if os.path.islink(input_path):
        fail("power_input_symlink", input_path)
        return result
    actual_sha = sha256(input_path)
    result["actual_sha256"] = actual_sha
    if actual_sha != expected_sha:
        fail("power_input_sha256", f"{actual_sha} vs {expected_sha}")
        return result

    try:
        with open(input_path, encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        fail("power_input_json", str(error))
        return result
    if not isinstance(payload, dict):
        fail("power_input_schema", "root must be an object")
        return result

    result["schema"] = payload.get("schema")
    result["candidate_blind"] = payload.get("candidate_blind")
    result["control_only"] = payload.get("control_only")
    source_shas = payload.get("artifacts")
    if payload.get("schema") != POWER_INPUT_SCHEMA:
        fail("power_input_schema", payload.get("schema"))
    if payload.get("candidate_blind") is not True:
        fail("power_input_candidate_blind", payload.get("candidate_blind"))
    if payload.get("control_only") is not True:
        fail("power_input_control_only", payload.get("control_only"))
    if (
        not isinstance(source_shas, dict)
        or not source_shas
        or not all(
            isinstance(name, str)
            and name
            and _is_sha256(value)
            for name, value in source_shas.items()
        )
    ):
        fail("power_input_source_shas", source_shas)
    else:
        result["source_shas"] = dict(sorted(source_shas.items()))

    if any(
        kind in FAILS
        for kind in (
            "power_input_schema",
            "power_input_candidate_blind",
            "power_input_control_only",
            "power_input_source_shas",
        )
    ):
        return result
    if not os.path.isfile(evaluator_path):
        fail("power_evaluator_missing", evaluator_path)
        return result

    try:
        evaluator = _load_power_evaluator(evaluator_path)
        validate = getattr(evaluator, "validate_power_input")
        compute = getattr(evaluator, "compute_power_assessment")
    except Exception as error:  # importing the frozen evaluator must fail closed
        fail("power_evaluator_interface", str(error))
        return result
    try:
        validation = validate(payload)
        # This is intentionally the expensive raw recomputation.  No rate or
        # verdict field is accepted from the manifest itself.
        assessment = compute(input_path)
    except Exception as error:  # evaluator errors are a fail-closed asset error
        fail("power_evaluator_error", f"{type(error).__name__}: {error}")
        return result
    if not isinstance(validation, dict) or validation != payload:
        fail("power_input_validation", "evaluator did not accept the frozen payload")
        return result
    try:
        # The preflight manifest is the audit artifact.  Refuse evaluator
        # results that cannot be represented there instead of failing later
        # during the final json.dump after expensive asset checks have run.
        result["validation"] = {"valid": True}
        result["assessment"] = json.loads(json.dumps(assessment))
    except (TypeError, ValueError) as error:
        fail("power_evaluator_output", str(error))
        return result

    if not isinstance(assessment, dict):
        fail("power_assessment_schema", assessment)
        return result
    if assessment.get("schema") != POWER_ASSESSMENT_SCHEMA:
        fail("power_assessment_schema", assessment.get("schema"))
    if assessment.get("candidate_blind") is not True:
        fail(
            "power_assessment_candidate_blind",
            assessment.get("candidate_blind"),
        )
    if assessment.get("control_only") is not True:
        fail("power_assessment_control_only", assessment.get("control_only"))
    if assessment.get("source_shas") != result["source_shas"]:
        fail("power_assessment_source_shas", assessment.get("source_shas"))
    if assessment.get("raw_manifest_sha256") != expected_sha:
        fail("power_assessment_manifest_sha256", assessment.get("raw_manifest_sha256"))
    result["raw_source_digest"] = assessment.get("raw_source_digest")
    contract = assessment.get("contract")
    if not isinstance(contract, dict) or (
        contract.get("ci_gates"), contract.get("span_gates"),
        contract.get("blocking_gates"), contract.get("effect_scenarios"),
        contract.get("outer_trials"), contract.get("bootstrap_replicates"),
        contract.get("outer_chunk"), contract.get("formal_outer_chunk"),
    ) != (70, 3, 73, 22, 10_000, 10_000, 32, 32):
        fail("power_assessment_contract", contract)
    provenance = assessment.get("provenance")
    generator_path = os.path.join(qsrc, "logprob_client.py")
    if not os.path.isfile(generator_path) or os.path.islink(generator_path):
        fail("power_provenance_file", generator_path)
        return result
    expected_provenance = {
        "dataset_sha256": exp.get("sharegpt_sha256"),
        "gsm8k_sha256": exp.get("gsm8k_sha256"),
        "tokenizer_sha256": (exp.get("tokenizer_files") or {}).get("tokenizer.json"),
        "generator_sha256": sha256(generator_path),
        "evaluator_sha256": sha256(evaluator_path),
        "mok_head": exp.get("mok_head"),
    }
    if expected_heads:
        expected_provenance.update(expected_heads)
    if not isinstance(provenance, dict) or any(
        provenance.get(key) != value for key, value in expected_provenance.items()
        if value is not None
    ):
        fail("power_assessment_provenance", provenance)
    families = assessment.get("zero_degradation_families")
    if not isinstance(families, dict) or set(families) != POWER_FAMILIES:
        fail(
            "power_assessment_families",
            sorted(families) if isinstance(families, dict) else families,
        )
    else:
        for family, gates in sorted(families.items()):
            if not isinstance(gates, dict) or gates.get("pass") is not True:
                value = gates.get("pass") if isinstance(gates, dict) else gates
                fail("power_family_gate", f"{family}.pass={value}")
    single_gates = assessment.get("single_ci_gates")
    expected_single_gates = getattr(evaluator, "EXPECTED_POWER_GATE_NAMES", set())
    if (
        not isinstance(single_gates, dict)
        or not expected_single_gates
        or set(single_gates) != set(expected_single_gates)
    ):
        fail(
            "power_single_gate_set",
            sorted(single_gates) if isinstance(single_gates, dict) else single_gates,
        )
    else:
        for gate, verdict in sorted(single_gates.items()):
            if not isinstance(verdict, dict) or verdict.get("pass") is not True:
                value = verdict.get("pass") if isinstance(verdict, dict) else verdict
                fail("power_single_gate", f"{gate}.pass={value}")
    effects = assessment.get("exact_margin_effects")
    expected_effects = getattr(evaluator, "EFFECT_GATE_NAMES", ())
    if not isinstance(effects, dict) or set(effects) != set(expected_effects):
        fail("power_effect_set", sorted(effects) if isinstance(effects, dict) else effects)
    else:
        for gate, verdict in sorted(effects.items()):
            if not isinstance(verdict, dict) or verdict.get("pass") is not True:
                fail("power_effect_gate", f"{gate}.pass={verdict.get('pass') if isinstance(verdict, dict) else verdict}")
    vector = assessment.get("pass_vector")
    popcounts = vector.get("popcounts") if isinstance(vector, dict) else None
    if (
        not isinstance(vector, dict)
        or not _is_sha256(vector.get("sha256"))
        or vector.get("bits") != 1_200_000
        or not isinstance(vector.get("ones"), int)
        or not isinstance(popcounts, dict)
        or set(popcounts) != {
            "ci_pass", "span_pass", "family_pass",
            "effect_gate_rejection", "effect_family_failure",
        }
        or not all(isinstance(value, int) and value >= 0 for value in popcounts.values())
        or sum(popcounts.values()) != vector.get("ones")
    ):
        fail("power_pass_vector", vector)
    if assessment.get("overall_pass") is not True:
        fail("power_assessment_no_go", assessment.get("overall_pass"))

    power_failures = [kind for kind in FAILS if kind.startswith("power_")]
    result["verified"] = not power_failures
    return result


def git(repo, *args):
    # The repos are owned by the host user while this runs as container
    # root: git's dubious-ownership guard silently empties every command
    # unless safe.directory is granted via system/global config (the -C/-c
    # forms are deliberately ignored for this key).  A throwaway global
    # config keeps the host's real global config out of the measurement.
    cfg = "/tmp/quality-gitconfig"
    if not os.path.exists(cfg):
        with open(cfg, "w") as f:
            f.write("[safe]\n\tdirectory = *\n")
    return subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": cfg},
    ).stdout.strip()


def main():
    qdir = sys.argv[1]
    expect_sglang_head = sys.argv[2]
    exp = json.load(open(f"{QSRC}/expected_assets.json"))

    os.makedirs(qdir, exist_ok=True)
    leftovers = [n for n in os.listdir(qdir) if not n.startswith("preflight")]
    if leftovers:
        fail("run_dir_not_empty", leftovers[:5])

    # Datasets.
    g_md5 = md5(f"{ROOT}/gsm8k_test.jsonl")
    if g_md5 != exp["gsm8k_md5"]:
        fail("gsm8k_md5", g_md5)
    g_sha = sha256(f"{ROOT}/gsm8k_test.jsonl")
    if g_sha != exp["gsm8k_sha256"]:
        fail("gsm8k_sha256", g_sha)
    sg_sha = sha256(SHAREGPT)
    if sg_sha != exp["sharegpt_sha256"]:
        fail("sharegpt_sha256", sg_sha)

    # Repos: full-SHA equality plus clean worktrees (covers script content).
    mok_head = git(f"{ROOT}/mixture-of-kittens", "rev-parse", "HEAD")
    if mok_head != exp["mok_head"]:
        fail("mok_head", mok_head)
    sglang_head = git(f"{ROOT}/sglang", "rev-parse", "HEAD")
    if len(expect_sglang_head) != 40 or sglang_head != expect_sglang_head:
        fail("sglang_head", f"{sglang_head} vs {expect_sglang_head}")
    for repo in (f"{ROOT}/mixture-of-kittens", f"{ROOT}/sglang"):
        dirty = git(repo, "status", "--porcelain")
        if dirty:
            fail("worktree_dirty", f"{repo}: {dirty.splitlines()[:3]}")

    # SO fingerprint: path-independent content md5 set.
    so_dir = f"{ROOT}/mixture-of-kittens/mok"
    so_files = sorted(n for n in os.listdir(so_dir) if n.endswith(".so"))
    so_md5s = [md5(f"{so_dir}/{n}") for n in so_files]
    if not so_files or so_md5s != exp["so_content_md5s"]:
        fail("so_fingerprint", f"{so_files}: {so_md5s}")

    # Container image (host launcher passes docker inspect .Image).
    image_id = os.environ.get("IMAGE_ID", "")
    if image_id != exp["image_id"]:
        fail("image_id", image_id or "MISSING")

    # Tokenizer / chat template files.
    tok = {}
    for name, want in exp["tokenizer_files"].items():
        got = sha256(f"{MODEL_DIR}/{name}")
        tok[name] = got
        if got != want:
            fail("tokenizer_file", f"{name}: {got}")

    # Exact harness script set with recorded SHAs (content already pinned by
    # the clean-worktree + HEAD assertions above).
    present = sorted(
        n for n in os.listdir(QSRC)
        if n.endswith((".py", ".sh", ".json")) and n != "__pycache__"
    )
    if present != exp["script_set"]:
        fail("script_set", f"{present}")
    scripts = {n: sha256(f"{QSRC}/{n}") for n in present}

    # Candidate-blind power is an admission gate, not a post-hoc report.  It
    # must close before T(S), and its full validation/assessment is preserved
    # in the manifest for independent audit.
    power_analysis = verify_power_asset(
        exp,
        expected_heads={"sglang_head": sglang_head, "mok_head": mok_head},
    )

    # Direct numeric evidence is a hard gate independent of end-to-end task
    # drift. Verify raw artifacts before deriving the frozen summary.
    numeric_hashes = {}
    synthetic_metrics = []
    live_rows = []
    for path, want in exp["numeric_audit_assets"].items():
        if not os.path.isfile(path):
            fail("numeric_audit_missing", path)
            continue
        got = sha256(path)
        numeric_hashes[path] = got
        if got != want:
            fail("numeric_audit_sha256", f"{path}: {got}")
        if path.endswith(".json"):
            payload = json.load(open(path))
            synthetic_metrics.append(payload["mok_vs_deepgemm"])
        else:
            for line in open(path, errors="ignore"):
                if "MOK_FP8_NUMERIC_AUDIT|" not in line:
                    continue
                fields = dict(re.findall(
                    r"(layer|stage|valid_M|exact|rel_l2)=([^|\s]+)", line
                ))
                if fields and int(fields["valid_M"]) > 10:
                    live_rows.append(fields)
    numeric_values = [
        (float(item["exact_bf16_fraction"]), float(item["relative_l2"]))
        for item in synthetic_metrics
    ] + [
        (float(item["exact"]), float(item["rel_l2"])) for item in live_rows
    ]
    numeric_audit = {
        "verified": False,
        "asset_sha256": numeric_hashes,
        "synthetic_cases": len(synthetic_metrics),
        "live_records": len(live_rows),
        "live_layers": len({int(item["layer"]) for item in live_rows}),
        "min_exact": min((value[0] for value in numeric_values), default=0),
        "max_relative_l2": max(
            (value[1] for value in numeric_values), default=math.inf
        ),
    }
    numeric_audit["verified"] = (
        len(synthetic_metrics) == 3
        and numeric_audit["live_records"] == 344
        and numeric_audit["live_layers"] == 43
        and all(math.isfinite(value) for pair in numeric_values for value in pair)
        and numeric_audit["min_exact"] >= 0.999
        and numeric_audit["max_relative_l2"] <= 1e-3
    )
    if not numeric_audit["verified"]:
        fail("numeric_audit_gate", numeric_audit)

    manifest = {
        "verified": not FAILS,
        "failures": FAILS,
        "gsm8k_md5": g_md5,
        "gsm8k_sha256": g_sha,
        "sharegpt_sha256": sg_sha,
        "mok_head": mok_head,
        "sglang_head": sglang_head,
        "so_content_md5s": so_md5s,
        "image_id": image_id,
        "tokenizer_files": tok,
        "harness_scripts": scripts,
        "power_analysis": power_analysis,
        "numeric_audit": numeric_audit,
        "model_dir": MODEL_DIR,
    }
    out = f"{qdir}/preflight-manifest.json"
    json.dump(manifest, open(out, "w"), indent=1)
    if FAILS:
        print(f"PREFLIGHT_FAILED|{FAILS}", flush=True)
        return 1
    print(f"PREFLIGHT_OK|mok={mok_head[:9]}|sglang={sglang_head[:9]}"
          f"|so={so_md5s[0][:12]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
