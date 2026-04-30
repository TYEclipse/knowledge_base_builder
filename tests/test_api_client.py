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
    assert "第2/3阶段" in msg
    assert "当前为非流式请求" in msg
    assert "预计剩余 10秒" in msg
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
    assert "当前为非流式请求" in msg
    assert "本阶段剩余约 2 项" in msg
    assert "预计剩余 未知" in msg
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
    assert "子任务：仅名称子任务" in msg1
    assert "尚未收到内容" in msg1

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
    assert "本阶段剩余约 3 项" in msg2
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
    assert "流式聚合完成" in logger.debug_calls[0]
    assert any("请求完成" in msg for msg in logger.info_calls)
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
    assert "对象：显式对象名" in msg
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
