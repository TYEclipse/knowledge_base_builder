from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

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


class DummyChunk:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self, exclude_none=True):
        return deepcopy(self.payload)


class DummyToolCall:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self, exclude_none=True):
        return deepcopy(self.payload)


class DummyMessage:
    def __init__(self, payload, tool_calls=None):
        self.payload = payload
        self.tool_calls = tool_calls or []

    def model_dump(self, exclude_none=True):
        return deepcopy(self.payload)


class DummyChoice:
    def __init__(self, finish_reason, message):
        self.finish_reason = finish_reason
        self.message = message


class DummyResponse:
    def __init__(self, choices, usage=None):
        self.choices = choices
        self.usage = usage


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


def test_create_completion_non_stream_builds_kwargs():
    captures = []

    class CaptureClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            captures.append(kwargs)
            return DummyResponse(choices=[], usage=None)

    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=False,
        logger=SimpleNamespace(),
        stats=None,
        openai_client=CaptureClient(),
    )

    client._create_completion(
        messages=[{"role": "user", "content": "hello"}],
        enable_json_mode=True,
        enable_web_search=True,
        max_tokens=123,
    )

    assert len(captures) == 1
    kwargs = captures[0]
    assert kwargs["model"] == "kimi-k2.6"
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["extra_body"]["thinking"]["type"] == "disabled"
    assert kwargs["tools"][0]["function"]["name"] == "$web_search"
    assert kwargs["max_tokens"] == 123
    client.close()


def test_create_completion_streaming_assembles_response():
    class StreamClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            assert kwargs["stream"] is True
            return [
                DummyChunk(
                    {
                        "choices": [{"delta": {"content": "hello "}, "finish_reason": None}],
                    }
                ),
                DummyChunk(
                    {
                        "choices": [{"delta": {"content": "world"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                    }
                ),
            ]

    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=True,
        logger=SimpleNamespace(),
        stats=None,
        openai_client=StreamClient(),
    )

    response = client._create_completion_streaming(
        {
            "model": "kimi-k2.6",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    assert KimiApiClient.extract_message_content(response) == "hello world"
    assert response.usage is not None
    client.close()


def test_call_web_search_loop_success_with_malformed_tool_args():
    stats = SimpleNamespace(total_api_calls=0, total_input_tokens=0, total_output_tokens=0)
    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=False,
        logger=SimpleNamespace(),
        stats=stats,
        openai_client=DummyClient('{"ok":true}'),
    )

    calls = []

    first = DummyResponse(
        choices=[
            DummyChoice(
                "tool_calls",
                DummyMessage(
                    {"role": "assistant", "content": "need tool"},
                    tool_calls=[
                        DummyToolCall(
                            {
                                "id": "tc1",
                                "function": {"name": "$web_search", "arguments": "{bad-json"},
                            }
                        )
                    ],
                ),
            )
        ],
        usage=None,
    )

    second = DummyResponse(
        choices=[
            DummyChoice(
                "stop",
                DummyMessage({"role": "assistant", "content": "done"}, tool_calls=[]),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=4),
    )

    def fake_create_completion(messages, enable_json_mode, enable_web_search, max_tokens):
        calls.append(deepcopy(messages))
        return first if len(calls) == 1 else second

    client._create_completion = fake_create_completion  # type: ignore[method-assign]

    response = client.call(
        system_prompt="s",
        user_prompt="u",
        enable_web_search=True,
        max_tokens=100,
    )

    assert response.choices[0].finish_reason == "stop"
    assert len(calls) == 2
    # 第二次调用应携带 tool 回填消息，且 malformed 参数走原样字符串分支
    assert any(m.get("role") == "tool" and m.get("content") == "{bad-json" for m in calls[1])
    assert stats.total_api_calls == 1
    assert stats.total_input_tokens == 7
    assert stats.total_output_tokens == 4
    client.close()


def test_call_web_search_loop_exceeded_raises_runtime_error():
    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=False,
        logger=SimpleNamespace(),
        stats=None,
        openai_client=DummyClient('{"ok":true}'),
    )

    always_tool = DummyResponse(
        choices=[
            DummyChoice(
                "tool_calls",
                DummyMessage(
                    {"role": "assistant", "content": "again"},
                    tool_calls=[
                        DummyToolCall(
                            {
                                "id": "tc2",
                                "function": {"name": "$web_search", "arguments": "{}"},
                            }
                        )
                    ],
                ),
            )
        ],
        usage=None,
    )

    client._create_completion = lambda **kwargs: always_tool  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        client.call(system_prompt="s", user_prompt="u", enable_web_search=True)
    client.close()


def test_is_auth_error_message_pattern():
    err = RuntimeError("invalid_authentication_error: token invalid")
    assert KimiApiClient.is_auth_error(err) is True
    assert KimiApiClient.is_auth_error(RuntimeError("other error")) is False


def test_safe_parse_json_with_plain_fence_branch():
    payload = KimiApiClient.safe_parse_json("```\n{\"b\":2}\n```")
    assert payload["b"] == 2


def test_create_completion_enable_stream_branch(monkeypatch):
    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=True,
        logger=SimpleNamespace(),
        stats=None,
        openai_client=DummyClient('{"ok":true}'),
    )

    called = {"ok": False}

    def fake_stream(kwargs):
        called["ok"] = True
        return DummyResponse(
            choices=[DummyChoice("stop", DummyMessage({"role": "assistant", "content": "x"}))],
            usage=None,
        )

    monkeypatch.setattr(client, "_create_completion_streaming", fake_stream)
    client._create_completion(
        messages=[{"role": "user", "content": "u"}],
        enable_json_mode=False,
        enable_web_search=False,
        max_tokens=None,
    )
    assert called["ok"] is True
    client.close()


def test_create_completion_streaming_handles_empty_choices_and_no_usage():
    class StreamClientNoUsage:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            return [
                DummyChunk({"choices": []}),
                DummyChunk({"choices": [{"delta": {"content": 123}, "finish_reason": "stop"}]}),
            ]

    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=True,
        logger=SimpleNamespace(),
        stats=None,
        openai_client=StreamClientNoUsage(),
    )

    resp = client._create_completion_streaming({"model": "kimi-k2.6", "messages": []})
    assert KimiApiClient.extract_message_content(resp) == ""
    assert resp.usage is None
    client.close()


def test_call_web_search_breaks_on_empty_choices_and_empty_tool_calls():
    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=False,
        logger=SimpleNamespace(),
        stats=None,
        openai_client=DummyClient('{"ok":true}'),
    )

    # 场景1：response.choices 为空，直接 break
    client._create_completion = lambda **kwargs: DummyResponse(choices=[], usage=None)  # type: ignore[method-assign]
    r1 = client.call(system_prompt="s", user_prompt="u", enable_web_search=True)
    assert r1.choices == []

    # 场景2：finish_reason=tool_calls 但 tool_calls 为空，直接 break
    msg = DummyMessage({"role": "assistant", "content": "x"}, tool_calls=[])
    client._create_completion = lambda **kwargs: DummyResponse(choices=[DummyChoice("tool_calls", msg)], usage=None)  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        client.call(system_prompt="s", user_prompt="u", enable_web_search=True)
    client.close()


def test_call_stats_none_and_stats_with_no_usage_branches():
    # stats=None 分支
    c1 = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=False,
        logger=SimpleNamespace(),
        stats=None,
        openai_client=DummyClient('{"ok":true}'),
    )
    c1.call(system_prompt="s", user_prompt="u")
    c1.close()

    # stats!=None 且 usage=None 分支
    stats = SimpleNamespace(total_api_calls=0, total_input_tokens=0, total_output_tokens=0)
    c2 = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=False,
        logger=SimpleNamespace(),
        stats=stats,
        openai_client=DummyClient('{"ok":true}'),
    )
    c2._create_completion = lambda **kwargs: DummyResponse(choices=[DummyChoice("stop", DummyMessage({"role": "assistant", "content": "ok"}))], usage=None)  # type: ignore[method-assign]
    c2.call(system_prompt="s", user_prompt="u")
    assert stats.total_api_calls == 1
    assert stats.total_input_tokens == 0
    assert stats.total_output_tokens == 0
    c2.close()


def test_is_auth_error_with_authentication_error_instance(monkeypatch):
    import api_client as ac

    monkeypatch.setattr(ac.openai, "AuthenticationError", RuntimeError)
    assert KimiApiClient.is_auth_error(RuntimeError("auth")) is True
