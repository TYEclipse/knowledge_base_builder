from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import time

import pytest

from api_client import KimiApiClient


class DummyClient:
    def __init__(self, content: str):
        self.content = content
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content), finish_reason="stop"
                )
            ],
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
        self.content = payload.get("content")
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


def test_dummy_message_model_dump_executes():
    msg = DummyMessage({"role": "assistant", "content": "x"}, tool_calls=[])
    dumped = msg.model_dump()
    assert dumped["content"] == "x"


def test_safe_parse_json_with_markdown_fence():
    payload = KimiApiClient.safe_parse_json('```json\n{"a":1}\n```')
    assert payload["a"] == 1


def test_extract_message_content_returns_default_on_empty_choices():
    response = SimpleNamespace(choices=[])
    assert KimiApiClient.extract_message_content(response, default="x") == "x"


def test_call_updates_stats():
    stats = SimpleNamespace(
        total_api_calls=0, total_input_tokens=0, total_output_tokens=0
    )
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
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

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


def test_create_completion_uses_deepseek_with_reasoning_effort_when_enabled():
    kimi_captures = []
    deepseek_captures = []

    class CaptureClient:
        def __init__(self, captures):
            self.captures = captures
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            self.captures.append(kwargs)
            return DummyResponse(choices=[], usage=None)

    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        deepseek_api_key="ds",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model_name="deepseek-v4-pro",
        deepseek_reasoning_effort="max",
        enable_stream=False,
        logger=SimpleNamespace(),
        stats=None,
        openai_client=CaptureClient(kimi_captures),
    )
    client.deepseek_client = CaptureClient(deepseek_captures)

    client._create_completion(
        messages=[{"role": "user", "content": "hello"}],
        enable_json_mode=False,
        enable_web_search=False,
        max_tokens=256,
        use_deepseek_thinking=True,
        reasoning_effort="max",
    )

    assert len(kimi_captures) == 0
    assert len(deepseek_captures) == 1
    kwargs = deepseek_captures[0]
    assert kwargs["model"] == "deepseek-v4-pro"
    assert kwargs["reasoning_effort"] == "max"
    assert kwargs["extra_body"]["thinking"]["type"] == "enabled"
    client.close()


def test_create_completion_json_mode_enables_stream_when_enabled(monkeypatch):
    captures = []

    class CaptureClient:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            captures.append(kwargs)
            return DummyResponse(
                choices=[
                    DummyChoice(
                        "stop", DummyMessage({"role": "assistant", "content": "ok"})
                    )
                ],
                usage=None,
            )

    capture_client = CaptureClient()
    capture_client._create(ping=True)
    captures.clear()

    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=True,
        logger=SimpleNamespace(warning=lambda *a, **k: None),
        stats=None,
        openai_client=capture_client,
    )

    called = {"stream": False}
    monkeypatch.setattr(
        client,
        "_create_completion_streaming",
        lambda kwargs, progress_context=None: called.__setitem__("stream", True),
    )

    client._create_completion(
        messages=[{"role": "user", "content": "hello"}],
        enable_json_mode=True,
        enable_web_search=False,
        max_tokens=None,
    )

    assert called["stream"] is True
    assert len(captures) == 0
    client.close()


def test_create_completion_web_search_enables_stream_when_enabled(monkeypatch):
    captures = []

    class CaptureClient:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            captures.append(kwargs)
            return DummyResponse(
                choices=[
                    DummyChoice(
                        "stop", DummyMessage({"role": "assistant", "content": "ok"})
                    )
                ],
                usage=None,
            )

    capture_client = CaptureClient()
    capture_client._create(ping=True)
    captures.clear()

    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=True,
        logger=SimpleNamespace(warning=lambda *a, **k: None),
        stats=None,
        openai_client=capture_client,
    )

    called = {"stream": False}
    monkeypatch.setattr(
        client,
        "_create_completion_streaming",
        lambda kwargs, progress_context=None: called.__setitem__("stream", True),
    )

    client._create_completion(
        messages=[{"role": "user", "content": "hello"}],
        enable_json_mode=False,
        enable_web_search=True,
        max_tokens=None,
    )

    assert called["stream"] is True
    assert len(captures) == 0
    client.close()


def test_create_completion_streaming_assembles_response():
    class StreamClient:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            assert kwargs["stream"] is True
            return [
                DummyChunk(
                    {
                        "choices": [
                            {"delta": {"content": "hello "}, "finish_reason": None}
                        ],
                    }
                ),
                DummyChunk(
                    {
                        "choices": [
                            {"delta": {"content": "world"}, "finish_reason": "stop"}
                        ],
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 2,
                            "total_tokens": 5,
                        },
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


def test_create_completion_streaming_assembles_tool_calls():
    class StreamClientWithToolCalls:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            assert kwargs["stream"] is True
            return [
                DummyChunk(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {
                                                "name": "$web_search",
                                                "arguments": '{"q":"',
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                ),
                DummyChunk(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {"arguments": '哲学"}'},
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
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
        openai_client=StreamClientWithToolCalls(),
    )

    response = client._create_completion_streaming(
        {
            "model": "kimi-k2.6",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    assert response.choices[0].finish_reason == "tool_calls"
    tool_calls = response.choices[0].message.tool_calls or []
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call_1"
    assert tool_calls[0].function.name == "$web_search"
    assert tool_calls[0].function.arguments == '{"q":"哲学"}'
    client.close()


def test_create_completion_streaming_normalizes_builtin_function_tool_calls():
    class StreamClientWithBuiltinToolCalls:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            assert kwargs["stream"] is True
            return [
                DummyChunk(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "t-web-search-1",
                                            "type": "builtin_function",
                                            "builtin_function": {
                                                "name": "$web_search",
                                                "arguments": '{"q":"',
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                ),
                DummyChunk(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "builtin_function": {"arguments": "哲学"},
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
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
        openai_client=StreamClientWithBuiltinToolCalls(),
    )

    response = client._create_completion_streaming(
        {
            "model": "kimi-k2.6",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    assert response.choices[0].finish_reason == "tool_calls"
    tool_calls = response.choices[0].message.tool_calls or []
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "t-web-search-1"
    assert tool_calls[0].type == "function"
    assert tool_calls[0].function.name == "$web_search"
    assert tool_calls[0].function.arguments == '{"q":"哲学'
    client.close()


def test_call_web_search_loop_success_with_malformed_tool_args():
    stats = SimpleNamespace(
        total_api_calls=0, total_input_tokens=0, total_output_tokens=0
    )
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
                                "function": {
                                    "name": "$web_search",
                                    "arguments": "{bad-json",
                                },
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

    def fake_create_completion(
        messages, enable_json_mode, enable_web_search, max_tokens, progress_context=None
    ):
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
    assert any(
        m.get("role") == "tool" and m.get("content") == "{bad-json" for m in calls[1]
    )
    assert stats.total_api_calls == 1
    assert stats.total_input_tokens == 7
    assert stats.total_output_tokens == 4
    client.close()


def test_call_web_search_loop_exceeded_forces_finalize_response():
    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=False,
        logger=SimpleNamespace(warning=lambda *a, **k: None),
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

    final_stop = DummyResponse(
        choices=[
            DummyChoice("stop", DummyMessage({"role": "assistant", "content": "final"}))
        ],
        usage=None,
    )

    state = {"count": 0}

    def fake_create_completion(
        messages, enable_json_mode, enable_web_search, max_tokens, progress_context=None
    ):
        if enable_web_search:
            state["count"] += 1
            return always_tool
        return final_stop

    client._create_completion = fake_create_completion  # type: ignore[method-assign]

    r = client.call(system_prompt="s", user_prompt="u", enable_web_search=True)
    assert r.choices[0].finish_reason == "stop"
    assert state["count"] >= 1
    client.close()


def test_call_web_search_loop_exceeded_and_finalize_still_tool_calls_raises():
    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=False,
        logger=SimpleNamespace(warning=lambda *a, **k: None),
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
                                "id": "tc",
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
    payload = KimiApiClient.safe_parse_json('```\n{"b":2}\n```')
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

    def fake_stream(kwargs, progress_context=None):
        called["ok"] = True
        return DummyResponse(
            choices=[
                DummyChoice("stop", DummyMessage({"role": "assistant", "content": "x"}))
            ],
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


def test_progress_helpers_format_and_estimate():
    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=False,
        logger=SimpleNamespace(),
        stats=None,
        openai_client=DummyClient('{"ok":true}'),
    )

    client._record_request_duration("phase2", 12.0)
    client._record_request_duration("phase2", 18.0)
    eta = client._estimate_remaining_seconds("phase2", 5.0)
    assert eta == pytest.approx(10.0)

    msg = client._format_progress_message(
        {
            "stage_name": "问题清单生成",
            "stage_index": 2,
            "stage_total": 3,
            "substep_name": "初学者",
            "substep_index": 1,
            "substep_total": 3,
            "request_group": "phase2",
        },
        elapsed_seconds=5.0,
        received_chars=20,
        chunk_count=2,
    )
    assert "p:2/3" in msg
    assert "ns" in msg
    assert "eta:10秒" in msg
    assert client._display_width(msg) <= 80
    client.close()


def test_format_progress_message_without_history():
    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=False,
        logger=SimpleNamespace(),
        stats=None,
        openai_client=DummyClient('{"ok":true}'),
    )

    msg = client._format_progress_message(
        {
            "stage_name": "逐题深度分析",
            "stage_index": 3,
            "stage_total": 3,
            "item_index": 1,
            "item_total": 3,
            "request_group": "phase3_analysis",
        },
        elapsed_seconds=11.0,
        received_chars=0,
        chunk_count=0,
    )
    assert "ns" in msg
    assert "r:2" in msg
    assert "eta:未知" in msg
    assert client._display_width(msg) <= 80
    client.close()


def test_create_completion_streaming_handles_empty_choices_and_no_usage():
    class StreamClientNoUsage:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            return [
                DummyChunk({"choices": []}),
                DummyChunk(
                    {"choices": [{"delta": {"content": 123}, "finish_reason": "stop"}]}
                ),
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
    stats = SimpleNamespace(
        total_api_calls=0, total_input_tokens=0, total_output_tokens=0
    )
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


def test_create_completion_stream_failure_falls_back_to_non_stream(monkeypatch):
    captures = []

    class CaptureClient:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            captures.append(kwargs)
            return DummyResponse(
                choices=[
                    DummyChoice(
                        "stop", DummyMessage({"role": "assistant", "content": "ok"})
                    )
                ],
                usage=None,
            )

    class Logger:
        def __init__(self):
            self.warned = False

        def warning(self, *args, **kwargs):
            self.warned = True

    logger = Logger()
    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=True,
        logger=logger,
        stats=None,
        openai_client=CaptureClient(),
    )

    monkeypatch.setattr(
        client,
        "_create_completion_streaming",
        lambda kwargs: (_ for _ in ()).throw(RuntimeError("stream failed")),
    )

    resp = client._create_completion(
        messages=[{"role": "user", "content": "hello"}],
        enable_json_mode=False,
        enable_web_search=False,
        max_tokens=None,
    )

    assert resp.choices[0].finish_reason == "stop"
    assert len(captures) == 1
    assert logger.warned is True
    client.close()


def test_helper_branches_for_empty_request_group_and_duration_format():
    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=False,
        logger=SimpleNamespace(),
        stats=None,
        openai_client=DummyClient('{"ok":true}'),
    )

    client._record_request_duration(None, 1.0)
    assert client._estimate_remaining_seconds("missing", 3.0) is None
    assert client._format_duration(65.0) == "1分05秒"
    client.close()


def test_format_progress_message_substep_without_indices_and_substep_remaining():
    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=True,
        logger=SimpleNamespace(),
        stats=None,
        openai_client=DummyClient('{"ok":true}'),
    )

    msg1 = client._format_progress_message(
        {
            "stage_name": "阶段X",
            "substep_name": "仅名称子任务",
            "stream_mode": True,
        },
        elapsed_seconds=3.0,
        received_chars=0,
        chunk_count=0,
    )
    assert "sb:仅名称子任务" in msg1
    assert "rx:0/0" in msg1
    assert client._display_width(msg1) <= 80

    msg2 = client._format_progress_message(
        {
            "stage_name": "阶段Y",
            "substep_name": "有索引子任务",
            "substep_index": 1,
            "substep_total": 4,
            "stream_mode": True,
        },
        elapsed_seconds=4.0,
        received_chars=2,
        chunk_count=1,
    )
    assert "r:3" in msg2
    assert client._display_width(msg2) <= 80
    client.close()


def test_tool_call_to_dict_model_dump_non_dict_and_assistant_message_blank_content():
    class WeirdToolCall:
        def model_dump(self, exclude_none=True):
            return "not-a-dict"

    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=False,
        logger=SimpleNamespace(),
        stats=None,
        openai_client=DummyClient('{"ok":true}'),
    )

    converted = client._tool_call_to_dict(WeirdToolCall())
    assert converted["id"] == ""
    assert converted["function"]["name"] == ""

    msg = client._assistant_message_for_history(
        SimpleNamespace(content="   ", tool_calls=[])
    )
    assert msg == {"role": "assistant"}
    client.close()


def test_create_completion_streaming_keyboard_interrupt_reraises(monkeypatch):
    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=True,
        logger=SimpleNamespace(),
        stats=None,
        openai_client=DummyClient('{"ok":true}'),
    )

    def raise_interrupt(kwargs, progress_context=None):
        raise KeyboardInterrupt()

    monkeypatch.setattr(client, "_create_completion_streaming", raise_interrupt)

    with pytest.raises(KeyboardInterrupt):
        client._create_completion(
            messages=[{"role": "user", "content": "u"}],
            enable_json_mode=False,
            enable_web_search=False,
            max_tokens=None,
        )
    client.close()


def test_create_completion_streaming_skips_non_dict_tool_call_delta():
    class StreamClientWithWeirdToolDelta:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            return [
                DummyChunk(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": ["not-dict"],
                                    "content": "ok",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    }
                )
            ]

    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=True,
        logger=SimpleNamespace(),
        stats=None,
        openai_client=StreamClientWithWeirdToolDelta(),
    )

    response = client._create_completion_streaming(
        {
            "model": "kimi-k2.6",
            "messages": [{"role": "user", "content": "u"}],
        }
    )
    assert KimiApiClient.extract_message_content(response) == "ok"
    assert not response.choices[0].message.tool_calls
    client.close()


def test_start_progress_heartbeat_runner_emits_info_log():
    class Logger:
        def __init__(self):
            self.calls = 0

        def info(self, *args, **kwargs):
            self.calls += 1

    logger = Logger()
    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=False,
        logger=logger,
        stats=None,
        openai_client=DummyClient('{"ok":true}'),
    )
    client.progress_heartbeat_seconds = 0.01

    stop_event, _, _ = client._start_progress_heartbeat(
        {"stage_name": "心跳测试", "stream_mode": False}
    )
    try:
        time.sleep(0.03)
    finally:
        assert stop_event is not None
        stop_event.set()

    assert logger.calls >= 1
    client.close()


def test_start_progress_heartbeat_none_context_returns_none_event():
    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=False,
        logger=SimpleNamespace(),
        stats=None,
        openai_client=DummyClient('{"ok":true}'),
    )

    stop_event, _, _ = client._start_progress_heartbeat(None)
    assert stop_event is None

    client.close()


def test_info_overwrite_passes_extra_kwargs_when_logger_supports_it():
    class Logger:
        def __init__(self):
            self.kwargs = None

        def info(self, *args, **kwargs):
            self.kwargs = kwargs

    logger = Logger()
    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=False,
        logger=logger,
        stats=None,
        openai_client=DummyClient('{"ok":true}'),
    )

    client._info("progress", overwrite=True)

    assert logger.kwargs == {"extra": {"overwrite": True}}
    client.close()


def test_create_completion_streaming_only_logs_summary_debug_once():
    class StreamClient:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            return [
                DummyChunk(
                    {
                        "choices": [
                            {"delta": {"content": "hello "}, "finish_reason": None}
                        ],
                    }
                ),
                DummyChunk(
                    {
                        "choices": [
                            {"delta": {"content": "world"}, "finish_reason": "stop"}
                        ]
                    }
                ),
            ]

    class Logger:
        def __init__(self):
            self.debug_calls = []
            self.info_calls = []

        def debug(self, message, *args, **kwargs):
            self.debug_calls.append(message % args)

        def info(self, message, *args, **kwargs):
            self.info_calls.append(message % args)

    logger = Logger()
    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=True,
        logger=logger,
        stats=None,
        openai_client=StreamClient(),
    )

    response = client._create_completion_streaming(
        {"model": "kimi-k2.6", "messages": [{"role": "user", "content": "u"}]}
    )

    assert KimiApiClient.extract_message_content(response) == "hello world"
    assert len(logger.debug_calls) == 1
    assert "stream chars=" in logger.debug_calls[0]
    assert any("done " in msg for msg in logger.info_calls)
    client.close()


def test_debug_call_and_duration_bucket_trim_and_item_name_progress_branch():
    class Logger:
        def __init__(self):
            self.debug_calls = 0

        def debug(self, *args, **kwargs):
            self.debug_calls += 1

    logger = Logger()
    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=False,
        logger=logger,
        stats=None,
        openai_client=DummyClient('{"ok":true}'),
    )

    client._debug("hello")
    assert logger.debug_calls == 1

    for i in range(25):
        client._record_request_duration("trim-me", float(i))
    assert len(client._request_durations["trim-me"]) == 20

    msg = client._format_progress_message(
        {
            "stage_name": "阶段Z",
            "item_name": "显式对象名",
            "stream_mode": True,
        },
        elapsed_seconds=2.0,
        received_chars=1,
        chunk_count=1,
    )
    assert "it:显式对象名" in msg
    assert client._display_width(msg) <= 80
    client.close()


def test_format_progress_message_truncates_to_80_display_width_for_long_names():
    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=True,
        logger=SimpleNamespace(),
        stats=None,
        openai_client=DummyClient('{"ok":true}'),
    )

    msg = client._format_progress_message(
        {
            "stage_name": "逐题深度分析阶段名称非常非常长用于测试截断",
            "stage_index": 3,
            "stage_total": 3,
            "substep_name": "这是一个特别长的子任务名称用于测试",
            "substep_index": 2,
            "substep_total": 9,
            "item_name": "哲学的核心问题域有哪些以及其历史演化路径是什么",
            "request_group": "phase3_analysis",
            "stream_mode": True,
        },
        elapsed_seconds=123.0,
        received_chars=9999,
        chunk_count=8888,
    )

    assert client._display_width(msg) <= 80
    client.close()


def test_tool_call_to_dict_model_dump_dict_and_final_fallback_branch():
    class DumpOnlyToolCall:
        def model_dump(self, exclude_none=True):
            return {
                "id": "tc-dump",
                "function": {"name": "$web_search", "arguments": "{}"},
            }

    class NoDataToolCall:
        pass

    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=False,
        logger=SimpleNamespace(),
        stats=None,
        openai_client=DummyClient('{"ok":true}'),
    )

    dumped = client._tool_call_to_dict(DumpOnlyToolCall())
    assert dumped["id"] == "tc-dump"
    assert dumped["function"]["name"] == "$web_search"

    fallback = client._tool_call_to_dict(NoDataToolCall())
    assert fallback["id"] == ""
    assert fallback["function"]["arguments"] == "{}"
    client.close()


def test_tool_call_to_dict_attribute_fast_path_branch():
    class Fn:
        name = "$web_search"
        arguments = "{}"

    class AttrToolCall:
        id = "tc-attr"
        function = Fn()

    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=False,
        logger=SimpleNamespace(),
        stats=None,
        openai_client=DummyClient('{"ok":true}'),
    )

    converted = client._tool_call_to_dict(AttrToolCall())
    assert converted["id"] == "tc-attr"
    assert converted["function"]["name"] == "$web_search"
    assert converted["function"]["arguments"] == "{}"
    client.close()


def test_tool_call_to_dict_supports_builtin_function_model_dump_branch():
    class BuiltinDumpToolCall:
        def model_dump(self, exclude_none=True):
            return {
                "id": "tc-builtin",
                "type": "builtin_function",
                "builtin_function": {
                    "name": "$web_search",
                    "arguments": "{}",
                },
            }

    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=False,
        logger=SimpleNamespace(),
        stats=None,
        openai_client=DummyClient('{"ok":true}'),
    )

    converted = client._tool_call_to_dict(BuiltinDumpToolCall())
    assert converted["id"] == "tc-builtin"
    assert converted["type"] == "function"
    assert converted["function"]["name"] == "$web_search"
    assert converted["function"]["arguments"] == "{}"
    client.close()


def test_create_completion_streaming_sets_stop_event_when_progress_context_present():
    class StreamClientOneChunk:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            return [
                DummyChunk(
                    {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}
                )
            ]

    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=True,
        logger=SimpleNamespace(),
        stats=None,
        openai_client=StreamClientOneChunk(),
    )

    stop_called = {"ok": False}
    created_event = {"obj": None}

    class DummyStopEvent:
        def wait(self, _):
            return True

        def set(self):
            stop_called["ok"] = True

    def fake_start(_):
        event = DummyStopEvent()
        created_event["obj"] = event
        return event, {"received_chars": 0, "chunk_count": 0}, time.monotonic()

    client._start_progress_heartbeat = fake_start  # type: ignore[method-assign]

    client._create_completion_streaming(
        {"model": "kimi-k2.6", "messages": [{"role": "user", "content": "u"}]},
        progress_context={"stage_name": "x", "stream_mode": True},
    )
    assert created_event["obj"] is not None
    assert created_event["obj"].wait(0) is True
    assert stop_called["ok"] is True
    client.close()


def test_select_runtime_warns_when_deepseek_missing_and_requested():
    class Logger:
        def __init__(self):
            self.warned = False

        def warning(self, *args, **kwargs):
            self.warned = True

    logger = Logger()
    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        deepseek_api_key="",
        enable_stream=False,
        logger=logger,
        stats=None,
        openai_client=DummyClient('{"ok":true}'),
    )

    runtime_client, runtime_model, thinking_enabled = client._select_runtime(
        use_deepseek_thinking=True,
        enable_web_search=False,
    )
    assert runtime_client is client.client
    assert runtime_model == "kimi-k2.6"
    assert thinking_enabled is True
    assert logger.warned is True
    client.close()


def test_log_overwrite_typeerror_falls_back_to_plain_log_call():
    class Logger:
        def __init__(self):
            self.calls = 0

        def info(self, *args, **kwargs):
            self.calls += 1
            if kwargs:
                raise TypeError("no kwargs accepted")

    logger = Logger()
    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=False,
        logger=logger,
        stats=None,
        openai_client=DummyClient('{"ok":true}'),
    )

    client._info("hello", overwrite=True)
    assert logger.calls == 2
    client.close()


def test_truncate_display_width_zero_and_truncate_path():
    assert KimiApiClient._truncate_display_width("abc", 0) == ""
    out = KimiApiClient._truncate_display_width("abcdefgh", 5)
    assert out.endswith("…")


def test_format_progress_message_with_empty_stage_name_branch():
    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=False,
        logger=SimpleNamespace(),
        stats=None,
        openai_client=DummyClient('{"ok":true}'),
    )

    msg = client._format_progress_message(
        {
            "stage_name": "",
            "stream_mode": False,
        },
        elapsed_seconds=1.0,
        received_chars=0,
        chunk_count=0,
    )
    assert "p:" not in msg
    assert "ns" in msg
    client.close()


def test_log_request_debug_info_with_non_str_content_and_full_ctx():
    class Logger:
        def __init__(self):
            self.last = ""

        def debug(self, message, *args, **kwargs):
            self.last = message % args

    logger = Logger()
    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=False,
        logger=logger,
        stats=None,
        openai_client=DummyClient('{"ok":true}'),
    )

    client._log_request_debug_info(
        messages=[
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": ["not-string"]},
        ],
        enable_json_mode=False,
        enable_web_search=False,
        use_stream=False,
        max_tokens=10,
        progress_context={
            "request_group": "g1",
            "stage_index": 1,
            "stage_total": 3,
            "substep_index": 2,
            "substep_total": 3,
            "item_index": 4,
            "item_total": 9,
        },
    )

    assert "g=g1" in logger.last
    assert "p=1/3" in logger.last
    assert "s=2/3" in logger.last
    assert "i=4/9" in logger.last
    client.close()


def test_extract_function_name_args_supports_builtin_and_custom_non_str_args():
    name1, args1 = KimiApiClient._extract_function_name_args(
        {"builtin_function": {"name": "$web_search", "arguments": [1, 2, 3]}}
    )
    assert name1 == "$web_search"
    assert args1 == "{}"

    class CustomObj:
        custom = SimpleNamespace(name="tool_x", arguments=None)

    name2, args2 = KimiApiClient._extract_function_name_args(CustomObj())
    assert name2 == "tool_x"
    assert args2 == "{}"


def test_create_completion_streaming_with_custom_tool_calls_branch():
    class StreamClientWithCustomCalls:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            return [
                DummyChunk(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "c1",
                                            "custom": {
                                                "name": "tool_a",
                                                "arguments": "{",
                                            },
                                        },
                                        {
                                            "index": 1,
                                            "id": "c2",
                                            "custom": {
                                                "name": "tool_b",
                                            },
                                        },
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    }
                )
            ]

    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=True,
        logger=SimpleNamespace(),
        stats=None,
        openai_client=StreamClientWithCustomCalls(),
    )

    resp = client._create_completion_streaming(
        {
            "model": "kimi-k2.6",
            "messages": [{"role": "user", "content": "u"}],
        }
    )
    tool_calls = resp.choices[0].message.tool_calls or []
    assert len(tool_calls) == 2
    assert tool_calls[0].function.name == "tool_a"
    assert tool_calls[0].function.arguments == "{"
    assert tool_calls[1].function.name == "tool_b"
    assert tool_calls[1].function.arguments == ""
    client.close()


def test_call_web_search_skips_empty_assistant_history_message_branch():
    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=False,
        logger=SimpleNamespace(),
        stats=None,
        openai_client=DummyClient('{"ok":true}'),
    )

    first = DummyResponse(
        choices=[
            DummyChoice(
                "tool_calls",
                DummyMessage(
                    {"role": "assistant", "content": "should be ignored"},
                    tool_calls=[
                        DummyToolCall(
                            {
                                "id": "tc-empty-assistant",
                                "function": {
                                    "name": "$web_search",
                                    "arguments": "{}",
                                },
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
            DummyChoice("stop", DummyMessage({"role": "assistant", "content": "ok"}))
        ],
        usage=None,
    )

    calls = []

    def fake_create_completion(
        messages, enable_json_mode, enable_web_search, max_tokens, progress_context=None
    ):
        calls.append(messages)
        return first if len(calls) == 1 else second

    client._assistant_message_for_history = lambda _msg: {"role": "assistant"}  # type: ignore[method-assign]
    client._create_completion = fake_create_completion  # type: ignore[method-assign]

    resp = client.call(system_prompt="s", user_prompt="u", enable_web_search=True)
    assert resp.choices[0].finish_reason == "stop"
    assert len(calls) == 2
    client.close()


def test_call_web_search_tool_debug_logs_tool_round_and_tokens():
    class Logger:
        def __init__(self):
            self.debug_logs = []

        def debug(self, message, *args, **kwargs):
            self.debug_logs.append(message % args)

    logger = Logger()
    client = KimiApiClient(
        api_key="k",
        base_url="https://api.moonshot.cn/v1",
        model_name="kimi-k2.6",
        enable_stream=False,
        logger=logger,
        stats=None,
        openai_client=DummyClient('{"ok":true}'),
        tool_debug=True,
        tool_debug_max_chars=300,
    )

    first = DummyResponse(
        choices=[
            DummyChoice(
                "tool_calls",
                DummyMessage(
                    {"role": "assistant", "content": "need tool"},
                    tool_calls=[
                        DummyToolCall(
                            {
                                "id": "tc-debug-1",
                                "function": {
                                    "name": "$web_search",
                                    "arguments": '{"query":"AI Agent","usage":{"total_tokens":1234}}',
                                },
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
            DummyChoice("stop", DummyMessage({"role": "assistant", "content": "done"}))
        ],
        usage=SimpleNamespace(prompt_tokens=9, completion_tokens=3, total_tokens=12),
    )

    state = {"count": 0}

    def fake_create_completion(
        messages, enable_json_mode, enable_web_search, max_tokens, progress_context=None
    ):
        state["count"] += 1
        return first if state["count"] == 1 else second

    client._create_completion = fake_create_completion  # type: ignore[method-assign]

    resp = client.call(system_prompt="s", user_prompt="u", enable_web_search=True)

    assert resp.choices[0].finish_reason == "stop"
    merged_logs = "\n".join(logger.debug_logs)
    assert "web_search round=1" in merged_logs
    assert "tool[1/1] id=tc-debug-1" in merged_logs
    assert "search_content_total_tokens=1234" in merged_logs
    assert "web_search final_usage prompt=9 completion=3 total=12" in merged_logs
    client.close()
