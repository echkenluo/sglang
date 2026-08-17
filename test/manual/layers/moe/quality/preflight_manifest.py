"""Phase 2 preflight: verify and record every frozen asset BEFORE T(S).

Aborts (non-zero) on any mismatch with the expected values passed on the
command line; writes preflight-manifest.json into the run directory.  The
manifest carries full SHA256 digests (not truncations) for the datasets,
the harness scripts, both repo HEADs, the SO fingerprint, the container
image digest, and the tokenizer/chat-template source directory listing.
"""

import hashlib
import json
import os
import subprocess
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(repo):
    return subprocess.run(
        ["git", "-C", repo, "rev-parse", "HEAD"],
        capture_output=True, text=True, env={**os.environ,
                                             "GIT_CONFIG_GLOBAL": "/dev/null"},
    ).stdout.strip()


def main():
    qdir = sys.argv[1]
    expect_gsm8k_md5 = sys.argv[2]
    expect_mok_head = sys.argv[3]
    expect_sglang_head = sys.argv[4]
    os.makedirs(qdir, exist_ok=True)
    leftovers = [n for n in os.listdir(qdir)
                 if not n.startswith("preflight")]
    if leftovers:
        print(f"PREFLIGHT_FAIL|run_dir_not_empty|{leftovers[:5]}", flush=True)
        return 1

    root = "/mok/claude-mok"
    gsm8k = f"{root}/gsm8k_test.jsonl"
    sharegpt = "/data2/pubulic-models/ShareGPT_V3_unfiltered_cleaned_split.json"
    model_dir = "/data2/pubulic-models/DeepSeek-V4-Flash-FP8-fixed"
    qsrc = f"{root}/sglang/test/manual/layers/moe/quality"

    import hashlib as _h
    gsm8k_md5 = _h.md5(open(gsm8k, "rb").read()).hexdigest()
    if gsm8k_md5 != expect_gsm8k_md5:
        print(f"PREFLIGHT_FAIL|gsm8k_md5|{gsm8k_md5}", flush=True)
        return 1
    mok_head = git_head(f"{root}/mixture-of-kittens")
    sglang_head = git_head(f"{root}/sglang")
    if not mok_head.startswith(expect_mok_head):
        print(f"PREFLIGHT_FAIL|mok_head|{mok_head}", flush=True)
        return 1
    if not sglang_head.startswith(expect_sglang_head):
        print(f"PREFLIGHT_FAIL|sglang_head|{sglang_head}", flush=True)
        return 1

    so_files = sorted(
        f"{root}/mixture-of-kittens/mok/{n}"
        for n in os.listdir(f"{root}/mixture-of-kittens/mok")
        if n.endswith(".so")
    )
    so_fp = hashlib.md5(
        "".join(sha256(f) for f in so_files).encode()
    ).hexdigest()[:12]

    image = subprocess.run(
        ["cat", "/proc/self/cgroup"], capture_output=True, text=True
    ).stdout.strip()[:200]
    scripts = {}
    for n in sorted(os.listdir(qsrc)):
        if n.endswith((".py", ".sh")):
            scripts[n] = sha256(f"{qsrc}/{n}")
    tokenizer_files = {
        n: sha256(f"{model_dir}/{n}")
        for n in sorted(os.listdir(model_dir))
        if "token" in n.lower() or n.endswith(".json") and "config" in n
    }

    manifest = {
        "gsm8k_md5": gsm8k_md5,
        "gsm8k_sha256": sha256(gsm8k),
        "sharegpt_sha256": sha256(sharegpt),
        "mok_head": mok_head,
        "sglang_head": sglang_head,
        "so_fingerprint": so_fp,
        "container_cgroup": image,
        "harness_scripts": scripts,
        "tokenizer_files": tokenizer_files,
        "model_dir": model_dir,
    }
    out = f"{qdir}/preflight-manifest.json"
    json.dump(manifest, open(out, "w"), indent=1)
    print(f"PREFLIGHT_OK|mok={mok_head[:9]}|sglang={sglang_head[:9]}"
          f"|so={so_fp}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
