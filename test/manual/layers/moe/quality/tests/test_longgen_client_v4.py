import hashlib
import json
from pathlib import Path
import sys


QUALITY_DIR = Path(__file__).resolve().parents[1]
if str(QUALITY_DIR) not in sys.path:
    sys.path.insert(0, str(QUALITY_DIR))

import logprob_client as teacher_client  # noqa: E402
import longgen_client as client  # noqa: E402


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def make_targets():
    rows = []
    for prompt_id in range(teacher_client.TARGET_COUNT):
        prompt_ids = [prompt_id * 1000 + index for index in range(256)]
        target_ids = [1000000 + prompt_id * 1000 + index for index in range(512)]
        rows.append(
            {
                "prompt_id": prompt_id,
                "text_sha": sha(f"prompt-{prompt_id}"),
                "prompt_ids": prompt_ids,
                "prompt_len": len(prompt_ids),
                "prompt_sha256": teacher_client.ids_sha256(prompt_ids),
                "target_ids": target_ids,
                "target_sha256": teacher_client.ids_sha256(target_ids),
            }
        )
    return {
        "protocol": teacher_client.PROTOCOL,
        "seed": teacher_client.SEED,
        "dataset_sha256": sha("dataset"),
        "tokenizer_sha256": sha("tokenizer"),
        "generator_sha256": sha("generator"),
        "request_batching": teacher_client._batching_contract("target"),
        "rows": rows,
    }


def prompt_refs():
    return [
        {"prompt_id": prompt_id, "prompt_sha256": sha(str(prompt_id))}
        for prompt_id in range(16)
    ]


def test_compare_runs_reports_exact_first_divergence():
    left = [[0] * 512 for _ in range(16)]
    right = [[0] * 512 for _ in range(16)]
    right[3][7] = 99
    comparison = client.compare_runs(left, right, prompt_refs())

    assert comparison["prompt_divergence_count"] == 1
    assert comparison["token_divergence_count"] == 1
    assert comparison["per_prompt_first_divergence"][3] == 7
    assert comparison["first_divergence"] == {
        "prompt_index": 3,
        "prompt_id": 3,
        "token_index": 7,
        "left_token_id": 0,
        "right_token_id": 99,
    }


def test_arbitrary_free_run_divergence_does_not_change_client_rc(
    tmp_path, monkeypatch
):
    targets_path = tmp_path / teacher_client.TARGETS_FILENAME
    teacher_client.atomic_json_freeze(targets_path, make_targets())
    receipt = tmp_path / "path-config-receipt-f1-bundle.json"
    receipt.write_text('{"path_ok":1}\n', encoding="utf-8")
    wave_number = 0

    def fake_gen(port, input_ids):
        nonlocal wave_number
        if input_ids and isinstance(input_ids[0], list):
            current_wave = wave_number
            wave_number += 1
            return [
                [
                    2000000 + current_wave * 100000 + prompt_index * 1000 + token
                    for token in range(512)
                ]
                for prompt_index in range(16)
            ]
        prompt_id = input_ids[0] // 1000
        return [1000000 + prompt_id * 1000 + token for token in range(512)]

    monkeypatch.setattr(client, "gen", fake_gen)
    rc = client.main(
        [
            "--port",
            "30000",
            "--mode",
            "fused",
            "--tag",
            "f1",
            "--targets",
            str(targets_path),
            "--path-config-receipt",
            str(receipt),
            "--out-dir",
            str(tmp_path),
        ]
    )
    output = tmp_path / "free-run-info-f1.json"
    document = json.load(open(output, encoding="utf-8"))

    assert rc == 0
    assert document["kind"] == "free-run-info"
    assert len(document["serial"]) == 16
    assert len(document["waves"]) == 3
    assert all(len(row) == 512 for row in document["serial"])
    assert all(
        len(row) == 512 for wave in document["waves"] for row in wave
    )
    assert all(
        comparison["prompt_divergence_count"] > 0
        for comparison in document["info"].values()
    )
    assert "waves_exact" not in document
    assert "pass" not in document and "gate" not in document
    assert client.validate_free_run(document) is document
