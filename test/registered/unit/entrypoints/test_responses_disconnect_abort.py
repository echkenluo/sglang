"""Regression tests for /v1/responses streaming disconnect cleanup."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch

from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestResponsesDisconnectAbort(unittest.TestCase):
    def test_abort_task_accepts_a_response_request_id(self):
        manager = SimpleNamespace(abort_request=Mock())

        tasks = TokenizerManager.create_abort_task_for_rids(manager, "resp_test")
        with patch("asyncio.sleep", new=AsyncMock()):
            asyncio.run(tasks())

        manager.abort_request.assert_called_once_with("resp_test")

    def test_abort_task_preserves_existing_batch_cleanup(self):
        manager = SimpleNamespace(abort_request=Mock())

        tasks = TokenizerManager.create_abort_task_for_rids(
            manager, ["batch_0", "batch_1"]
        )
        with patch("asyncio.sleep", new=AsyncMock()):
            asyncio.run(tasks())

        self.assertEqual(
            manager.abort_request.call_args_list,
            [call("batch_0"), call("batch_1")],
        )

    def test_streaming_responses_endpoint_attaches_abort_cleanup(self):
        async def stream():
            yield "event: response.created\n\n"

        async def create_responses(request, raw_request):
            return stream()

        abort_cleanup = Mock(name="abort_cleanup")
        tokenizer_manager = SimpleNamespace(
            create_abort_task_for_rids=Mock(return_value=abort_cleanup)
        )
        raw_request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    openai_serving_responses=SimpleNamespace(
                        create_responses=create_responses
                    )
                )
            )
        )
        request = ResponsesRequest(model="x", input="hi", stream=True, store=False)
        prior_state = http_server.get_global_state()
        http_server.set_global_state(
            SimpleNamespace(tokenizer_manager=tokenizer_manager)
        )
        try:
            response = asyncio.run(
                http_server.v1_responses_request(request, raw_request)
            )
        finally:
            http_server._global_state = prior_state

        tokenizer_manager.create_abort_task_for_rids.assert_called_once_with(
            request.request_id
        )
        self.assertIs(response.background, abort_cleanup)


if __name__ == "__main__":
    unittest.main()
