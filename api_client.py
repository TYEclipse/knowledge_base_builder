"""Kimi API 客户端封装。"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Optional, cast

import httpx
import openai
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import (
    CONNECT_TIMEOUT_SECONDS,
    DEFAULT_DEEPSEEK_MODEL_NAME,
    DEFAULT_DEEPSEEK_REASONING_EFFORT,
    MAX_WEB_SEARCH_TOOL_ROUNDS,
    READ_TIMEOUT_SECONDS,
    WRITE_TIMEOUT_SECONDS,
    POOL_TIMEOUT_SECONDS,
)


class KimiApiClient:
    """Kimi API 访问客户端。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        deepseek_api_key: str = "",
        deepseek_base_url: str = "https://api.deepseek.com",
        deepseek_model_name: str = DEFAULT_DEEPSEEK_MODEL_NAME,
        deepseek_reasoning_effort: str = DEFAULT_DEEPSEEK_REASONING_EFFORT,
        enable_stream: bool = True,
        logger: Any = None,
        stats: Optional[Any] = None,
        openai_client: Optional[OpenAI] = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.deepseek_api_key = deepseek_api_key
        self.deepseek_base_url = deepseek_base_url
        self.deepseek_model_name = deepseek_model_name or DEFAULT_DEEPSEEK_MODEL_NAME
        self.deepseek_reasoning_effort = (
            deepseek_reasoning_effort or DEFAULT_DEEPSEEK_REASONING_EFFORT
        )
        self.enable_stream = enable_stream
        self.logger = logger
        self.stats = stats
        self.progress_heartbeat_seconds = 10.0
        self._request_durations: Dict[str, List[float]] = {}

        timeout = httpx.Timeout(
            timeout=None,
            connect=CONNECT_TIMEOUT_SECONDS,
            read=READ_TIMEOUT_SECONDS,
            write=WRITE_TIMEOUT_SECONDS,
            pool=POOL_TIMEOUT_SECONDS,
        )

        self.http_client = httpx.Client(timeout=timeout, trust_env=True)
        self.client = openai_client or OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=self.http_client,
        )

        self.deepseek_http_client: Optional[httpx.Client] = None
        self.deepseek_client: Optional[OpenAI] = None
        if self.deepseek_api_key:
            self.deepseek_http_client = httpx.Client(timeout=timeout, trust_env=True)
            self.deepseek_client = OpenAI(
                api_key=self.deepseek_api_key,
                base_url=self.deepseek_base_url,
                http_client=self.deepseek_http_client,
            )

    def close(self) -> None:
        """关闭底层连接。"""
        self.http_client.close()
        if self.deepseek_http_client is not None:
            self.deepseek_http_client.close()

    def _select_runtime(
        self, use_deepseek_thinking: bool, enable_web_search: bool
    ) -> tuple[OpenAI, str, bool]:
        """选择本次请求应使用的客户端、模型与思考模式。"""
        use_deepseek = (
            bool(use_deepseek_thinking)
            and not enable_web_search
            and self.deepseek_client is not None
        )

        if use_deepseek:
            return cast(OpenAI, self.deepseek_client), self.deepseek_model_name, True

        if (
            use_deepseek_thinking
            and not enable_web_search
            and self.deepseek_client is None
        ):
            self._warn(
                "未配置 DEEPSEEK_API_KEY，已回退到 Kimi 模型处理当前非联网请求。"
            )

        return self.client, self.model_name, (not enable_web_search)

    def _log(
        self, level_name: str, message: str, *args: Any, overwrite: bool = False
    ) -> None:
        """安全日志输出，兼容测试替身 logger 与可选的单行覆盖模式。"""
        log_fn = getattr(self.logger, level_name, None)
        if not callable(log_fn):
            return

        if overwrite:
            try:
                log_fn(message, *args, extra={"overwrite": True})
                return
            except TypeError:
                pass

        log_fn(message, *args)

    def _warn(self, message: str, *args: Any) -> None:
        """安全告警日志输出（兼容测试替身 logger）。"""
        self._log("warning", message, *args)

    def _info(self, message: str, *args: Any, overwrite: bool = False) -> None:
        """安全信息日志输出（兼容测试替身 logger）。"""
        self._log("info", message, *args, overwrite=overwrite)

    def _debug(self, message: str, *args: Any) -> None:
        """安全调试日志输出（兼容测试替身 logger）。"""
        self._log("debug", message, *args)

    def _record_request_duration(
        self, request_group: Optional[str], duration_seconds: float
    ) -> None:
        """记录某类请求的耗时历史。"""
        if not request_group:
            return
        bucket = self._request_durations.setdefault(request_group, [])
        bucket.append(duration_seconds)
        if len(bucket) > 20:
            del bucket[:-20]

    def _estimate_remaining_seconds(
        self, request_group: Optional[str], elapsed_seconds: float
    ) -> Optional[float]:
        """基于同类历史耗时估算剩余时间。"""
        if not request_group:
            return None
        durations = self._request_durations.get(request_group, [])
        if not durations:
            return None
        average = sum(durations) / len(durations)
        return max(average - elapsed_seconds, 0.0)

    @staticmethod
    def _format_duration(seconds: Optional[float]) -> str:
        """格式化秒数为人类可读字符串。"""
        if seconds is None:
            return "未知"
        total = max(int(round(seconds)), 0)
        minutes, secs = divmod(total, 60)
        if minutes:
            return f"{minutes}分{secs:02d}秒"
        return f"{secs}秒"

    @staticmethod
    def _display_width(text: str) -> int:
        """计算文本显示宽度（中文等宽字符按 2 计）。"""
        width = 0
        for ch in text:
            width += 2 if ord(ch) > 127 else 1
        return width

    @classmethod
    def _truncate_display_width(cls, text: str, max_width: int) -> str:
        """按显示宽度截断文本，超限时追加省略号。"""
        if max_width <= 0:
            return ""
        if cls._display_width(text) <= max_width:
            return text

        ellipsis = "…"
        ellipsis_width = cls._display_width(ellipsis)
        budget = max(max_width - ellipsis_width, 0)

        current_width = 0
        chars: List[str] = []
        for ch in text:
            ch_width = 2 if ord(ch) > 127 else 1
            if current_width + ch_width > budget:
                break
            chars.append(ch)
            current_width += ch_width
        return "".join(chars) + ellipsis

    def _format_progress_message(
        self,
        progress_context: Dict[str, Any],
        elapsed_seconds: float,
        received_chars: int,
        chunk_count: int,
    ) -> str:
        """构造心跳进度消息（控制单行显示宽度不超过 80）。"""
        stage_name = str(progress_context.get("stage_name", "未知阶段"))
        stage_index = progress_context.get("stage_index")
        stage_total = progress_context.get("stage_total")
        substep_name = progress_context.get("substep_name")
        substep_index = progress_context.get("substep_index")
        substep_total = progress_context.get("substep_total")
        item_name = progress_context.get("item_name")
        item_index = progress_context.get("item_index")
        item_total = progress_context.get("item_total")
        request_group = progress_context.get("request_group")
        stream_mode = bool(progress_context.get("stream_mode", False))

        stage_part = ""
        if stage_index and stage_total:
            stage_part = f"p:{stage_index}/{stage_total}"
        elif stage_name:
            stage_part = f"p:{self._truncate_display_width(stage_name, 8)}"

        extra_parts: List[str] = []
        if substep_name:
            substep_part = self._truncate_display_width(str(substep_name), 12)
            if substep_index and substep_total:
                extra_parts.append(
                    f"sb:{substep_part}({substep_index}/{substep_total})"
                )
            else:
                extra_parts.append(f"sb:{substep_part}")
        if item_name:
            extra_parts.append(f"it:{self._truncate_display_width(str(item_name), 12)}")
        elif item_index and item_total:
            extra_parts.append(f"it:{item_index}/{item_total}")

        remaining_items: Optional[int] = None
        if item_index and item_total:
            remaining_items = max(int(item_total) - int(item_index), 0)
        elif substep_index and substep_total:
            remaining_items = max(int(substep_total) - int(substep_index), 0)

        eta = self._estimate_remaining_seconds(
            cast(Optional[str], request_group), elapsed_seconds
        )
        if stream_mode:
            chunks_text = f"rx:{received_chars}/{chunk_count}"
        else:
            chunks_text = "ns"

        parts = [
            stage_part,
            *extra_parts,
            f"t:{self._format_duration(elapsed_seconds)}",
            chunks_text,
        ]
        if remaining_items is not None:
            parts.append(f"r:{remaining_items}")
        parts.append(f"eta:{self._format_duration(eta)}")

        message = " | ".join([p for p in parts if p])
        return self._truncate_display_width(message, 80)

    def _start_progress_heartbeat(
        self, progress_context: Optional[Dict[str, Any]]
    ) -> tuple[Optional[threading.Event], Dict[str, int], float]:
        """启动每 10 秒一次的进度心跳。"""
        if not progress_context:
            return None, {"received_chars": 0, "chunk_count": 0}, time.monotonic()

        state = {"received_chars": 0, "chunk_count": 0}
        stop_event = threading.Event()
        start_time = time.monotonic()

        def runner() -> None:
            while not stop_event.wait(self.progress_heartbeat_seconds):
                elapsed = time.monotonic() - start_time
                self._info(
                    "%s",
                    self._format_progress_message(
                        progress_context=progress_context,
                        elapsed_seconds=elapsed,
                        received_chars=state["received_chars"],
                        chunk_count=state["chunk_count"],
                    ),
                    overwrite=True,
                )

        threading.Thread(target=runner, daemon=True).start()
        return stop_event, state, start_time

    def _log_request_debug_info(
        self,
        *,
        messages: List[Dict[str, Any]],
        enable_json_mode: bool,
        enable_web_search: bool,
        use_stream: bool,
        max_tokens: Optional[int],
        progress_context: Optional[Dict[str, Any]],
    ) -> None:
        """记录请求调试摘要（不输出完整提示词）。"""
        try:
            user_prompt_len = 0
            system_prompt_len = 0
            for msg in messages:
                role = str(msg.get("role", ""))
                content = msg.get("content")
                if isinstance(content, str):
                    if role == "user":
                        user_prompt_len += len(content)
                    elif role == "system":
                        system_prompt_len += len(content)

            compact_ctx = "-"
            if progress_context:
                ctx_parts: List[str] = []
                if progress_context.get("request_group"):
                    ctx_parts.append(f"g={progress_context.get('request_group')}")
                if progress_context.get("stage_index") and progress_context.get(
                    "stage_total"
                ):
                    ctx_parts.append(
                        f"p={progress_context.get('stage_index')}/{progress_context.get('stage_total')}"
                    )
                if progress_context.get("substep_index") and progress_context.get(
                    "substep_total"
                ):
                    ctx_parts.append(
                        f"s={progress_context.get('substep_index')}/{progress_context.get('substep_total')}"
                    )
                if progress_context.get("item_index") and progress_context.get(
                    "item_total"
                ):
                    ctx_parts.append(
                        f"i={progress_context.get('item_index')}/{progress_context.get('item_total')}"
                    )
                compact_ctx = ",".join(ctx_parts) if ctx_parts else "-"

            self._debug(
                "req s=%s j=%s w=%s tok=%s m=%d sc=%d uc=%d ctx=%s",
                use_stream,
                enable_json_mode,
                enable_web_search,
                max_tokens,
                len(messages),
                system_prompt_len,
                user_prompt_len,
                compact_ctx,
            )
        except Exception as exc:  # pragma: no cover - 调试日志不应影响主流程
            self._warn("调试日志输出失败（已忽略）：%s", exc)

    @staticmethod
    def _normalize_tool_call_type(raw_type: Any) -> str:
        """将供应商差异化工具类型归一化到 SDK 可接受集合。"""
        # Kimi 联网工具常返回 builtin_function；为保证历史兼容与稳定，统一收敛到 function。
        return "function"

    @staticmethod
    def _extract_function_name_args(tool_like: Any) -> tuple[str, str]:
        """从不同形态的工具调用对象中提取函数名与参数字符串。"""
        candidates: List[Any] = []

        fn_obj = getattr(tool_like, "function", None)
        if fn_obj is not None:
            candidates.append(fn_obj)

        builtin_obj = getattr(tool_like, "builtin_function", None)
        if builtin_obj is not None:
            candidates.append(builtin_obj)

        custom_obj = getattr(tool_like, "custom", None)
        if custom_obj is not None:
            candidates.append(custom_obj)

        if isinstance(tool_like, dict):
            for key in ("function", "builtin_function", "custom"):
                value = tool_like.get(key)
                if value is not None:
                    candidates.append(value)

        for candidate in candidates:
            if isinstance(candidate, dict):
                name = candidate.get("name")
                arguments = candidate.get("arguments")
            else:
                name = getattr(candidate, "name", None)
                arguments = getattr(candidate, "arguments", None)

            if isinstance(name, str) and name:
                if isinstance(arguments, str):
                    return name, arguments
                return name, "{}"

        return "", "{}"

    @staticmethod
    def _tool_call_to_dict(tool_call: Any) -> Dict[str, Any]:
        """将 SDK 工具调用对象转换为稳定字典结构。"""
        # 优先使用属性访问，避免直接序列化触发 pydantic 告警。
        tc_id = getattr(tool_call, "id", "")
        fn_name, fn_args = KimiApiClient._extract_function_name_args(tool_call)

        if tc_id or fn_name:
            return {
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": fn_name,
                    "arguments": fn_args if isinstance(fn_args, str) else "{}",
                },
            }

        model_dump = getattr(tool_call, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump(exclude_none=True)
            fn_name, fn_args = KimiApiClient._extract_function_name_args(dumped)
            normalized_type = KimiApiClient._normalize_tool_call_type(
                dumped.get("type") if isinstance(dumped, dict) else None
            )
            return {
                "id": dumped.get("id", "") if isinstance(dumped, dict) else "",
                "type": normalized_type,
                "function": {
                    "name": fn_name,
                    "arguments": fn_args,
                },
            }

        return {
            "id": "",
            "type": "function",
            "function": {"name": "", "arguments": "{}"},
        }

    def _assistant_message_for_history(self, message_obj: Any) -> Dict[str, Any]:
        """构造可安全回填到 messages 的 assistant 消息。"""
        assistant_msg: Dict[str, Any] = {"role": "assistant"}

        content = getattr(message_obj, "content", None)
        if isinstance(content, str) and content.strip() != "":
            assistant_msg["content"] = content

        tool_calls = getattr(message_obj, "tool_calls", None) or []
        if tool_calls:
            assistant_msg["tool_calls"] = [
                self._tool_call_to_dict(tc) for tc in tool_calls
            ]

        return assistant_msg

    @staticmethod
    def extract_message_content(response: ChatCompletion, default: str = "") -> str:
        """安全提取首条响应文本。"""
        if not response.choices:
            return default
        content = response.choices[0].message.content
        return content if isinstance(content, str) else default

    @staticmethod
    def safe_parse_json(text: str) -> Dict[str, Any]:
        """容错 JSON 解析。"""
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return json.loads(cleaned.strip())

    def _create_completion(
        self,
        messages: List[Dict[str, Any]],
        enable_json_mode: bool,
        enable_web_search: bool,
        max_tokens: Optional[int],
        use_deepseek_thinking: bool = False,
        reasoning_effort: Optional[str] = None,
        progress_context: Optional[Dict[str, Any]] = None,
    ) -> ChatCompletion:
        """执行一次聊天请求。"""
        selected_client, selected_model_name, thinking_enabled = self._select_runtime(
            use_deepseek_thinking=use_deepseek_thinking,
            enable_web_search=enable_web_search,
        )
        kwargs: Dict[str, Any] = {
            "model": selected_model_name,
            "messages": messages,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if enable_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        kwargs["extra_body"] = {
            "thinking": {"type": "enabled" if thinking_enabled else "disabled"}
        }

        if selected_client is self.deepseek_client and thinking_enabled:
            kwargs["reasoning_effort"] = (
                reasoning_effort or self.deepseek_reasoning_effort
            )

        if enable_web_search:
            kwargs["tools"] = [
                {"type": "builtin_function", "function": {"name": "$web_search"}}
            ]

        # 统一优先流式，若通道异常则自动降级为非流式，保证稳定性。
        use_stream = self.enable_stream
        heartbeat_context = dict(progress_context or {})
        heartbeat_context["stream_mode"] = use_stream
        self._log_request_debug_info(
            messages=messages,
            enable_json_mode=enable_json_mode,
            enable_web_search=enable_web_search,
            use_stream=use_stream,
            max_tokens=max_tokens,
            progress_context=heartbeat_context,
        )

        if use_stream:
            try:
                try:
                    return self._create_completion_streaming(
                        kwargs,
                        selected_client=selected_client,
                        selected_model_name=selected_model_name,
                        progress_context=heartbeat_context,
                    )
                except TypeError:
                    # 兼容测试中对该方法的旧签名 monkeypatch。
                    return self._create_completion_streaming(
                        kwargs, progress_context=heartbeat_context
                    )
            except KeyboardInterrupt:
                # 用户主动中断时直接上抛，由上层统一优雅退出。
                raise
            except Exception as exc:
                # 网络抖动或流式通道异常时，自动降级为非流式请求，提升可用性。
                self._warn("stream fallback: %s", exc)
        stop_event, _, start_time = self._start_progress_heartbeat(heartbeat_context)
        try:
            response = cast(
                ChatCompletion, selected_client.chat.completions.create(**kwargs)
            )
            elapsed = time.monotonic() - start_time
            self._record_request_duration(
                cast(
                    Optional[str],
                    (
                        heartbeat_context.get("request_group")
                        if heartbeat_context
                        else None
                    ),
                ),
                elapsed,
            )
            response_content = self.extract_message_content(response, default="")
            if response_content:
                self._info(
                    "done chars=%d t=%s",
                    len(response_content),
                    self._format_duration(elapsed),
                )
            else:
                self._info(
                    "done empty t=%s",
                    self._format_duration(elapsed),
                )
            return response
        finally:
            if stop_event is not None:
                stop_event.set()

    def _create_completion_streaming(
        self,
        kwargs: Dict[str, Any],
        selected_client: Optional[OpenAI] = None,
        selected_model_name: Optional[str] = None,
        progress_context: Optional[Dict[str, Any]] = None,
    ) -> ChatCompletion:
        """流式聚合结果。"""
        runtime_client = selected_client or self.client
        runtime_model_name = selected_model_name or self.model_name
        stream_kwargs = dict(kwargs)
        stream_kwargs["stream"] = True
        stream = cast(Any, runtime_client.chat.completions.create(**stream_kwargs))
        stop_event, state, start_time = self._start_progress_heartbeat(progress_context)

        content_parts: List[str] = []
        finish_reason: Optional[str] = None
        usage_dict: Optional[Dict[str, Any]] = None
        tool_calls_accumulator: Dict[int, Dict[str, Any]] = {}

        try:
            for chunk_idx, chunk in enumerate(stream, start=1):
                chunk_dict = cast(Any, chunk).model_dump(exclude_none=True)
                choices = chunk_dict.get("choices", [])
                if not choices:
                    continue
                choice0 = choices[0]
                delta = choice0.get("delta", {})
                if isinstance(delta.get("content"), str):
                    content_parts.append(delta["content"])
                    state["received_chars"] += len(delta["content"])
                for tc in delta.get("tool_calls", []) or []:
                    if not isinstance(tc, dict):
                        continue
                    idx = int(tc.get("index", 0))
                    current = tool_calls_accumulator.setdefault(
                        idx,
                        {
                            "id": tc.get("id", ""),
                            "type": self._normalize_tool_call_type(tc.get("type")),
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if tc.get("id"):
                        current["id"] = tc.get("id")
                    current["type"] = self._normalize_tool_call_type(tc.get("type"))

                    fn = tc.get("function")
                    if not isinstance(fn, dict):
                        fn = tc.get("builtin_function")
                    if not isinstance(fn, dict):
                        fn = tc.get("custom")
                    if not isinstance(fn, dict):
                        fn = {}
                    if isinstance(fn, dict):
                        name = fn.get("name")
                        if isinstance(name, str) and name:
                            current["function"]["name"] = name
                        arguments = fn.get("arguments")
                        if isinstance(arguments, str):
                            current["function"]["arguments"] += arguments

                state["chunk_count"] += 1
                if isinstance(choice0.get("finish_reason"), str):
                    finish_reason = choice0["finish_reason"]
                if isinstance(chunk_dict.get("usage"), dict):
                    usage_dict = chunk_dict["usage"]

            self._record_request_duration(
                cast(
                    Optional[str],
                    progress_context.get("request_group") if progress_context else None,
                ),
                time.monotonic() - start_time,
            )
        finally:
            if stop_event is not None:
                stop_event.set()

        assembled: Dict[str, Any] = {
            "id": f"stream-assembled-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": runtime_model_name,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason or "stop",
                    "message": {"role": "assistant", "content": "".join(content_parts)},
                }
            ],
        }
        if tool_calls_accumulator:
            tool_calls = [
                tool_calls_accumulator[i] for i in sorted(tool_calls_accumulator.keys())
            ]
            assembled["choices"][0]["message"]["tool_calls"] = tool_calls

        if usage_dict:
            assembled["usage"] = usage_dict

        self._debug(
            "stream chars=%d chunks=%d fr=%s tc=%d",
            len("".join(content_parts)),
            state["chunk_count"],
            finish_reason,
            len(tool_calls_accumulator),
        )

        response_content = "".join(content_parts)
        elapsed = time.monotonic() - start_time
        if response_content:
            self._info(
                "done chars=%d t=%s",
                len(response_content),
                self._format_duration(elapsed),
            )
        else:
            self._info(
                "done empty t=%s",
                self._format_duration(elapsed),
            )

        return ChatCompletion.model_validate(assembled)

    @retry(
        retry=retry_if_exception_type(
            (openai.APIError, openai.APITimeoutError, openai.APIConnectionError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        enable_json_mode: bool = False,
        enable_web_search: bool = False,
        max_tokens: Optional[int] = None,
        use_deepseek_thinking: bool = False,
        reasoning_effort: Optional[str] = None,
        progress_context: Optional[Dict[str, Any]] = None,
    ) -> ChatCompletion:
        """统一 API 调用入口。"""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        def invoke_create(
            current_messages: List[Dict[str, Any]], web_search: bool
        ) -> ChatCompletion:
            try:
                return self._create_completion(
                    messages=current_messages,
                    enable_json_mode=enable_json_mode,
                    enable_web_search=web_search,
                    max_tokens=max_tokens,
                    use_deepseek_thinking=use_deepseek_thinking,
                    reasoning_effort=reasoning_effort,
                    progress_context=progress_context,
                )
            except TypeError:
                # 兼容测试中对该方法的旧签名 monkeypatch。
                return self._create_completion(
                    messages=current_messages,
                    enable_json_mode=enable_json_mode,
                    enable_web_search=web_search,
                    max_tokens=max_tokens,
                    progress_context=progress_context,
                )

        response = invoke_create(messages, enable_web_search)

        if enable_web_search:
            for _ in range(MAX_WEB_SEARCH_TOOL_ROUNDS):
                if not response.choices:
                    break
                choice = response.choices[0]
                if choice.finish_reason != "tool_calls":
                    break
                tool_calls = choice.message.tool_calls or []
                if not tool_calls:
                    break

                assistant_msg = self._assistant_message_for_history(choice.message)
                if "content" in assistant_msg or "tool_calls" in assistant_msg:
                    messages.append(assistant_msg)

                for tool_call in tool_calls:
                    tc = self._tool_call_to_dict(tool_call)
                    fn = tc.get("function", {})
                    args = fn.get("arguments", "{}")
                    try:
                        tool_payload = json.dumps(json.loads(args), ensure_ascii=False)
                    except Exception:
                        tool_payload = str(args)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "name": fn.get("name", "$web_search"),
                            "content": tool_payload,
                        }
                    )

                response = invoke_create(messages, enable_web_search)

            if response.choices and response.choices[0].finish_reason == "tool_calls":
                # 回合耗尽时，强制收敛为最终回答，避免阶段1直接失败。
                self._warn("联网工具调用超过最大回合数，尝试强制收敛为最终回答。")
                messages.append(
                    {
                        "role": "user",
                        "content": "请基于已经获取到的工具结果，直接给出最终回答，不要继续调用任何工具。",
                    }
                )
                response = invoke_create(messages, False)

                if (
                    response.choices
                    and response.choices[0].finish_reason == "tool_calls"
                ):
                    raise RuntimeError("联网工具调用超过最大回合数，且强制收敛失败。")

        if self.stats is not None:
            self.stats.total_api_calls += 1
            usage = response.usage
            if usage:
                self.stats.total_input_tokens += int(usage.prompt_tokens)
                self.stats.total_output_tokens += int(usage.completion_tokens)

        return response

    @staticmethod
    def is_auth_error(error: Exception) -> bool:
        """识别鉴权错误。"""
        if isinstance(error, openai.AuthenticationError):
            return True
        lowered = str(error).lower()
        return (
            "invalid authentication" in lowered
            or "invalid_authentication_error" in lowered
        )
