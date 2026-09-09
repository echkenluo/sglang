import csv
import io
import json
from pathlib import Path
import sys

import pytest


QUALITY_DIR = Path(__file__).resolve().parents[1]
if str(QUALITY_DIR) not in sys.path:
    sys.path.insert(0, str(QUALITY_DIR))

import gsm8k_client as client  # noqa: E402


def run_one_question(tmp_path, monkeypatch, *, expected_sha=None):
    dataset = tmp_path / "gsm8k.jsonl"
    dataset.write_text("frozen dataset\n", encoding="utf-8")
    dataset_sha = client.sha256_file(dataset)
    monkeypatch.setattr(
        client,
        "build_prompts",
        lambda path: [(0, "Question: 1+0?\nAnswer:", "1")],
    )
    monkeypatch.setattr(
        client,
        "ask_batch",
        lambda port, items, tag: ["#### 1"] * len(items),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gsm8k_client.py",
            "--port",
            "30071",
            "--data",
            str(dataset),
            "--dataset-sha256",
            expected_sha or dataset_sha,
            "--tag",
            "s1",
            "--limit",
            "1",
            "--out-dir",
            str(tmp_path),
        ],
    )
    client.main()
    return dataset_sha


def test_full_dataset_and_output_hashes_are_published(tmp_path, monkeypatch):
    dataset_sha = run_one_question(tmp_path, monkeypatch)
    csv_path = tmp_path / "gsm8k-s1.csv"
    summary_path = tmp_path / "gsm8k-s1.json"
    summary = json.loads(summary_path.read_text())
    rows = list(csv.DictReader(csv_path.open()))

    assert summary["schema"] == "phase2-v4-gsm8k-v1"
    assert summary["dataset_sha256"] == dataset_sha
    assert len(summary["dataset_sha256"]) == 64
    assert summary["csv_sha256"] == client.sha256_file(csv_path)
    assert summary["request_batching"] == client._batching_contract()
    assert len(rows[0]["text_sha256"]) == 64
    assert summary["n"] == summary["correct"] == 1


def test_dataset_sha_mismatch_fails_before_request(tmp_path, monkeypatch):
    dataset = tmp_path / "gsm8k.jsonl"
    dataset.write_text("frozen dataset\n", encoding="utf-8")
    called = False

    def forbidden(_):
        nonlocal called
        called = True
        raise AssertionError("dataset must fail before prompt construction")

    monkeypatch.setattr(client, "build_prompts", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gsm8k_client.py",
            "--port",
            "30071",
            "--data",
            str(dataset),
            "--dataset-sha256",
            "0" * 64,
            "--tag",
            "s1",
            "--limit",
            "1",
            "--out-dir",
            str(tmp_path),
        ],
    )
    with pytest.raises(ValueError, match="dataset SHA mismatch"):
        client.main()
    assert not called


def test_outputs_are_no_clobber(tmp_path, monkeypatch):
    run_one_question(tmp_path, monkeypatch)
    before = (tmp_path / "gsm8k-s1.csv").read_bytes()
    with pytest.raises(FileExistsError):
        run_one_question(tmp_path, monkeypatch)
    assert (tmp_path / "gsm8k-s1.csv").read_bytes() == before


def test_main_uses_fixed_batches_and_preserves_question_order(tmp_path, monkeypatch):
    dataset = tmp_path / "gsm8k-batched.jsonl"
    dataset.write_text("frozen dataset\n", encoding="utf-8")
    items = [(index, f"prompt-{index}", str(index)) for index in range(17)]
    batch_indices = []

    monkeypatch.setattr(client, "build_prompts", lambda _path: items)

    def fake_ask_batch(_port, batch, _tag):
        batch_indices.append([item[0] for item in batch])
        return [f"#### {item[2]}" for item in batch]

    monkeypatch.setattr(client, "ask_batch", fake_ask_batch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gsm8k_client.py",
            "--port",
            "30071",
            "--data",
            str(dataset),
            "--dataset-sha256",
            client.sha256_file(dataset),
            "--tag",
            "s1",
            "--limit",
            "17",
            "--out-dir",
            str(tmp_path),
        ],
    )
    client.main()
    assert batch_indices == [list(range(16)), [16]]
    rows = list(csv.DictReader((tmp_path / "gsm8k-s1.csv").open()))
    assert [int(row["idx"]) for row in rows] == list(range(17))


class FakeHTTPResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_gsm8k_batch_payload_and_response_order(monkeypatch):
    items = [(7, "prompt-7", "7"), (8, "prompt-8", "8")]
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        response = [
            {
                "meta_info": {"id": rid},
                "text": f"#### {index}",
            }
            for rid, (index, _prompt, _gold) in zip(
                captured["body"]["rid"], items
            )
        ]
        return FakeHTTPResponse(json.dumps(response).encode())

    monkeypatch.setattr(client.urllib.request, "urlopen", fake_urlopen)
    assert client.ask_batch(30071, items, "s1") == ["#### 7", "#### 8"]
    assert captured["body"]["text"] == ["prompt-7", "prompt-8"]
    assert captured["body"]["rid"] == [
        "p2v4-gsm8k-s1-0007",
        "p2v4-gsm8k-s1-0008",
    ]
    assert captured["body"]["sampling_params"]["max_new_tokens"] == 512


@pytest.mark.parametrize(
    "response,error",
    [
        ([{"meta_info": {"id": "p2v4-gsm8k-s1-0007"}, "text": "x"}], "exactly 2"),
        (
            [
                {"meta_info": {"id": "p2v4-gsm8k-s1-0008"}, "text": "x"},
                {"meta_info": {"id": "p2v4-gsm8k-s1-0007"}, "text": "y"},
            ],
            "order/RID mismatch",
        ),
    ],
)
def test_gsm8k_batch_short_or_reordered_fails_closed(monkeypatch, response, error):
    monkeypatch.setattr(
        client.urllib.request,
        "urlopen",
        lambda request, timeout: FakeHTTPResponse(json.dumps(response).encode()),
    )
    items = [(7, "prompt-7", "7"), (8, "prompt-8", "8")]
    with pytest.raises(ValueError, match=error):
        client.ask_batch(30071, items, "s1")
