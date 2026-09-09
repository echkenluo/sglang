import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import stat
import sys

import pytest


QUALITY_DIR = Path(__file__).resolve().parents[1]
if str(QUALITY_DIR) not in sys.path:
    sys.path.insert(0, str(QUALITY_DIR))

import logprob_client as client  # noqa: E402


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def make_targets():
    rows = []
    for prompt_id in range(client.TARGET_COUNT):
        prompt_ids = [prompt_id * 1000 + index for index in range(256)]
        target_ids = [1000000 + prompt_id * 1000 + index for index in range(512)]
        rows.append(
            {
                "prompt_id": prompt_id,
                "text_sha": sha(f"prompt-{prompt_id}"),
                "prompt_ids": prompt_ids,
                "prompt_len": len(prompt_ids),
                "prompt_sha256": client.ids_sha256(prompt_ids),
                "target_ids": target_ids,
                "target_sha256": client.ids_sha256(target_ids),
            }
        )
    return {
        "protocol": client.PROTOCOL,
        "seed": client.SEED,
        "dataset_sha256": sha("dataset"),
        "tokenizer_sha256": sha("tokenizer"),
        "generator_sha256": sha("generator"),
        "request_batching": client._batching_contract("target"),
        "rows": rows,
    }


def test_targets_schema_is_exactly_128_by_512():
    document = make_targets()
    assert client.validate_targets(document) is document
    assert len(document["rows"]) == 128
    assert all(len(row["target_ids"]) == 512 for row in document["rows"])

    broken = make_targets()
    broken["rows"][4]["target_ids"] = broken["rows"][4]["target_ids"][:-1]
    with pytest.raises(ValueError, match="exactly 512"):
        client.validate_targets(broken)

    broken = make_targets()
    broken["rows"][0]["prompt_ids"][0] = True
    with pytest.raises(ValueError, match="nonnegative integer token IDs"):
        client.validate_targets(broken)

    broken = make_targets()
    broken["rows"][0]["target_ids"][0] = -1
    with pytest.raises(ValueError, match="nonnegative integer token IDs"):
        client.validate_targets(broken)


def test_atomic_target_freeze_fsyncs_and_refuses_replacement(tmp_path, monkeypatch):
    path = tmp_path / client.TARGETS_FILENAME
    fsync_calls = []
    link_calls = []
    real_link = client.os.link

    monkeypatch.setattr(client.os, "fsync", lambda fd: fsync_calls.append(fd))

    def record_link(source, destination):
        assert Path(source).exists()
        link_calls.append((Path(source), Path(destination)))
        real_link(source, destination)

    monkeypatch.setattr(client.os, "link", record_link)
    client.atomic_json_freeze(path, {"value": 1})

    assert json.load(open(path, encoding="utf-8")) == {"value": 1}
    assert len(fsync_calls) == 2  # temporary file, then containing directory
    assert len(link_calls) == 1
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert not list(tmp_path.glob(f".{client.TARGETS_FILENAME}.*.tmp"))
    original_sha = client.sha256_file(path)
    with pytest.raises(FileExistsError, match="frozen artifact"):
        client.atomic_json_freeze(path, {"value": 2})
    assert client.sha256_file(path) == original_sha


def test_competing_target_freeze_is_atomic_no_clobber(tmp_path, monkeypatch):
    path = tmp_path / client.TARGETS_FILENAME
    real_link = client.os.link
    publish_barrier = __import__("threading").Barrier(2)

    def competing_link(source, destination):
        publish_barrier.wait(timeout=5)
        return real_link(source, destination)

    monkeypatch.setattr(client.os, "link", competing_link)
    documents = ({"writer": 1}, {"writer": 2})

    def publish(document):
        try:
            client.atomic_json_freeze(path, document)
            return "published"
        except FileExistsError:
            return "exists"

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(publish, documents))

    assert sorted(results) == ["exists", "published"]
    frozen = json.load(open(path, encoding="utf-8"))
    assert frozen in documents
    frozen_sha = client.sha256_file(path)
    monkeypatch.setattr(client.os, "link", real_link)
    with pytest.raises(FileExistsError, match="frozen artifact"):
        client.atomic_json_freeze(path, {"writer": 3})
    assert client.sha256_file(path) == frozen_sha
    assert not list(tmp_path.glob(f".{client.TARGETS_FILENAME}.*.tmp"))


def test_target_build_uses_512_tokens_and_publishes_only_freeze(
    tmp_path, monkeypatch
):
    sharegpt = tmp_path / "sharegpt.json"
    conversations = [
        {
            "conversations": [
                {"from": "human", "value": f"prompt-{prompt_id}"}
            ]
        }
        for prompt_id in range(128)
    ]
    sharegpt.write_text(json.dumps(conversations), encoding="utf-8")
    generate_requests = []
    tokenize_requests = []

    def fake_tokenize(port, texts):
        tokenize_requests.append((port, texts))
        tokens = [
            [int(text.split("-")[1]) * 1000 + index for index in range(256)]
            for text in texts
        ]
        return {"tokens": tokens, "count": [256] * len(tokens), "max_model_len": 32768}

    def fake_post(port, body):
        generate_requests.append(body)
        response = []
        for rid, prompt_ids in zip(body["rid"], body["input_ids"]):
            prompt_id = prompt_ids[0] // 1000
            target_ids = [1000000 + prompt_id * 1000 + index for index in range(512)]
            response.append(
                {
                    "meta_info": {"id": rid, "prompt_tokens": len(prompt_ids)},
                    "output_ids": target_ids,
                }
            )
        return response

    monkeypatch.setattr(client, "post", fake_post)
    monkeypatch.setattr(client, "post_tokenize", fake_tokenize)
    args = argparse.Namespace(
        port=30000,
        mode="split",
        sharegpt=str(sharegpt),
        dataset_sha256=client.sha256_file(sharegpt),
        tokenizer_sha256=sha("tokenizer"),
        generator_sha256=client.sha256_file(client.__file__),
        context_length=32768,
        out_dir=str(tmp_path),
    )
    output = client.stage_target(args)
    document = json.load(open(output, encoding="utf-8"))

    assert output.name == "targets-freeze.json"
    assert client.validate_targets(document) is document
    assert document["request_batching"] == client._batching_contract("target")
    assert len(tokenize_requests) == 1
    assert len(tokenize_requests[0][1]) == 128
    assert len(generate_requests) == 8
    assert all(len(request["input_ids"]) == 16 for request in generate_requests)
    assert all(
        request["sampling_params"]
        == {"temperature": 0, "max_new_tokens": 512, "ignore_eos": True}
        for request in generate_requests
    )
    assert all(len(request["rid"]) == len(request["input_ids"]) for request in generate_requests)


def test_target_selection_preserves_source_order_and_filters_before_generate(
    tmp_path, monkeypatch
):
    sharegpt = tmp_path / "sharegpt-filter.json"
    texts = ["dup", "dup", "short"] + [f"valid-{index}" for index in range(3, 130)]
    sharegpt.write_text(
        json.dumps(
            [
                {"conversations": [{"from": "human", "value": text}]}
                for text in texts
            ]
        ),
        encoding="utf-8",
    )
    tokenize_batch_sizes = []
    generate_batch_sizes = []

    def fake_tokenize(_port, batch_texts):
        tokenize_batch_sizes.append(len(batch_texts))
        token_rows = []
        for text in batch_texts:
            if text == "dup":
                source_id, length = 0, 256
            elif text == "short":
                source_id, length = 2, 255
            else:
                source_id, length = int(text.split("-")[1]), 256
            token_rows.append([source_id * 1000 + index for index in range(length)])
        return {
            "tokens": token_rows,
            "count": [len(row) for row in token_rows],
            "max_model_len": 32768,
        }

    def fake_post(_port, body):
        generate_batch_sizes.append(len(body["input_ids"]))
        return [
            {
                "meta_info": {"id": rid, "prompt_tokens": len(prompt_ids)},
                "output_ids": [2000000 + prompt_ids[0] + index for index in range(512)],
            }
            for rid, prompt_ids in zip(body["rid"], body["input_ids"])
        ]

    monkeypatch.setattr(client, "post_tokenize", fake_tokenize)
    monkeypatch.setattr(client, "post", fake_post)
    args = argparse.Namespace(
        port=30000,
        mode="split",
        sharegpt=str(sharegpt),
        dataset_sha256=client.sha256_file(sharegpt),
        tokenizer_sha256=sha("tokenizer"),
        generator_sha256=client.sha256_file(client.__file__),
        context_length=32768,
        out_dir=str(tmp_path),
    )
    document = client.build_targets(args)
    assert [row["prompt_id"] for row in document["rows"]] == [
        0,
        *range(3, 130),
    ]
    assert tokenize_batch_sizes == [128, 1]
    assert generate_batch_sizes == [16] * 8


def test_teacher512_score_contract_and_finite_gate(tmp_path, monkeypatch):
    targets = make_targets()
    targets_path = tmp_path / client.TARGETS_FILENAME
    client.atomic_json_freeze(targets_path, targets)
    receipt = tmp_path / "path-config-receipt-s1-bundle.json"
    receipt.write_text('{"path_ok":1}\n', encoding="utf-8")
    requests = []

    def fake_post(port, body):
        requests.append(body)
        response = []
        for rid, input_ids in zip(body["rid"], body["input_ids"]):
            target_ids = input_ids[-512:]
            response.append(
                {
                    "meta_info": {
                        "id": rid,
                        "prompt_tokens": len(input_ids),
                        "input_token_logprobs": [[-0.25, token] for token in target_ids],
                        "input_top_logprobs": [[[-0.2, token]] for token in target_ids],
                    }
                }
            )
        return response

    monkeypatch.setattr(client, "post", fake_post)
    args = argparse.Namespace(
        port=30000,
        mode="split",
        tag="s1",
        targets=str(targets_path),
        path_config_receipt=str(receipt),
        context_length=32768,
        out_dir=str(tmp_path),
    )
    output = client.stage_score(args)
    document = json.load(open(output, encoding="utf-8"))

    assert output.name == "teacher512-s1.json"
    assert client.validate_teacher(document, targets) is document
    assert document["request_batching"] == client._batching_contract("teacher512")
    assert len(requests) == 8
    assert all(len(request["input_ids"]) == 16 for request in requests)
    assert requests[0]["logprob_start_len"] == [255] * 16
    assert requests[0]["top_logprobs_num"] == 1
    assert requests[0]["input_ids"][0] == (
        targets["rows"][0]["prompt_ids"] + targets["rows"][0]["target_ids"]
    )
    assert sum(len(row["logprobs"]) for row in document["rows"]) == 65536

    broken = json.loads(json.dumps(document))
    broken["rows"][0]["logprobs"][0] = float("nan")
    with pytest.raises(ValueError, match="invalid logprobs"):
        client.validate_teacher(broken, targets)


def test_batch_response_rejects_short_or_reordered_rids():
    expected = ["rid-a", "rid-b"]
    with pytest.raises(ValueError, match="exactly 2"):
        client._validate_batch_response(
            [{"meta_info": {"id": "rid-a"}}], expected, "synthetic"
        )
    with pytest.raises(ValueError, match="order/RID mismatch"):
        client._validate_batch_response(
            [
                {"meta_info": {"id": "rid-b"}},
                {"meta_info": {"id": "rid-a"}},
            ],
            expected,
            "synthetic",
        )


def test_tokenize_batch_rejects_short_shape_and_count_mismatch():
    with pytest.raises(ValueError, match="exactly 2 ordered rows"):
        client._validate_tokenize_response(
            {"tokens": [[1, 2]], "count": [2]}, 2
        )
    with pytest.raises(ValueError, match="exactly 3 token IDs"):
        client._validate_tokenize_response(
            {"tokens": [[1, 2]], "count": [3], "max_model_len": 32768}, 1
        )
