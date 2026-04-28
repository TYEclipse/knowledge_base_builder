from __future__ import annotations

from types import SimpleNamespace

from api_client import KimiApiClient


class DummyClient:
    def __init__(self, content: str):
        self.content = content
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


def test_safe_parse_json_with_markdown_fence():
    payload = KimiApiClient.safe_parse_json("```json\n{\"a\":1}\n```")
    assert payload["a"] == 1


def test_extract_message_content_returns_default_on_empty_choices():
    response = SimpleNamespace(choices=[])
    assert KimiApiClient.extract_message_content(response, default="x") == "x"


def test_call_updates_stats():
    stats = SimpleNamespace(total_api_calls=0, total_input_tokens=0, total_output_tokens=0)
    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=False,
        logger=SimpleNamespace(info=lambda *a, **k: None),
        stats=stats,
        openai_client=DummyClient('{"ok":true}'),
    )

    client.call(system_prompt="s", user_prompt="u")

    assert stats.total_api_calls == 1
    assert stats.total_input_tokens == 10
    assert stats.total_output_tokens == 5
    client.close()
