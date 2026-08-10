import unittest
from unittest.mock import Mock, call, patch

import llm_client


class LlmFallbackTests(unittest.TestCase):
    @patch("llm_client.call_llm")
    def test_tries_two_org_backends_then_local_in_order(self, mocked_call):
        mocked_call.side_effect = [None, "invalid", "valid"]

        result, backend = llm_client.call_llm_with_fallback(
            "prompt",
            validator=lambda response: (
                {"selected_label": "اصلاح‌طلب"} if response == "valid" else None
            ),
        )

        self.assertEqual(result, {"selected_label": "اصلاح‌طلب"})
        self.assertEqual(backend, llm_client.LOCAL_BACKEND)
        self.assertEqual(
            mocked_call.call_args_list,
            [
                call(
                    "prompt",
                    backend=llm_client.SYNAPPSE_BACKEND,
                    temperature=0.2,
                    max_tokens=180,
                ),
                call(
                    "prompt",
                    backend=llm_client.SELF_HOSTED_BACKEND,
                    temperature=0.2,
                    max_tokens=180,
                ),
                call(
                    "prompt",
                    backend=llm_client.LOCAL_BACKEND,
                    temperature=0.1,
                    max_tokens=180,
                ),
            ],
        )

    @patch("llm_client.call_llm")
    def test_stops_after_first_valid_response(self, mocked_call):
        mocked_call.return_value = "valid"

        result, backend = llm_client.call_llm_with_fallback(
            "prompt",
            validator=lambda response: response,
        )

        self.assertEqual(result, "valid")
        self.assertEqual(backend, llm_client.SYNAPPSE_BACKEND)
        self.assertEqual(mocked_call.call_count, 1)

    @patch("llm_client.requests.post")
    @patch("llm_client.requests.get")
    def test_self_hosted_endpoint_discovers_model_before_chat(self, mocked_get, mocked_post):
        discovery_response = Mock()
        discovery_response.json.return_value = {"data": [{"id": "org-model"}]}
        mocked_get.return_value = discovery_response

        chat_response = Mock()
        chat_response.json.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }
        mocked_post.return_value = chat_response

        with patch.object(llm_client, "_self_hosted_model", None), patch.object(
            llm_client, "_self_hosted_model_checked", False
        ):
            result = llm_client.call_self_hosted_llm("prompt")

        self.assertEqual(result, "response")
        mocked_get.assert_called_once_with(
            llm_client.SELF_HOSTED_MODELS_URL,
            timeout=llm_client.ORG_LLM_DISCOVERY_TIMEOUT,
        )
        self.assertEqual(mocked_post.call_args.args[0], llm_client.SELF_HOSTED_CHAT_URL)
        self.assertEqual(mocked_post.call_args.kwargs["json"]["model"], "org-model")

    @patch("llm_client.requests.post")
    def test_synappse_endpoint_uses_configured_api_key(self, mocked_post):
        chat_response = Mock()
        chat_response.json.return_value = {"result": "response"}
        mocked_post.return_value = chat_response

        with patch.object(llm_client, "ORG_LLM_API_KEY", "secret"):
            result = llm_client.call_synappse_llm("prompt")

        self.assertEqual(result, "response")
        mocked_post.assert_called_once_with(
            "https://api.synappse.ir/api/chat/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "apikey": "secret",
            },
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "prompt",
                    }
                ]
            },
            timeout=llm_client.ORG_LLM_REQUEST_TIMEOUT,
        )


if __name__ == "__main__":
    unittest.main()
