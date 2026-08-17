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
import json
import os
import subprocess
import sys

ROOT = "/mok/claude-mok"
QSRC = f"{ROOT}/sglang/test/manual/layers/moe/quality"
MODEL_DIR = "/data2/pubulic-models/DeepSeek-V4-Flash-FP8-fixed"
SHAREGPT = "/data2/pubulic-models/ShareGPT_V3_unfiltered_cleaned_split.json"
FAILS = []


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


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null"},
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
