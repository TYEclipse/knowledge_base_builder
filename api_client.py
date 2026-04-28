"""Kimi API 客户端封装。"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, cast

import httpx
import openai
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import (
    CONNECT_TIMEOUT_SECONDS,
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
        enable_stream: bool,
        logger: Any,
        stats: Optional[Any] = None,
        openai_client: Optional[OpenAI] = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.enable_stream = enable_stream
        self.logger = logger
        self.stats = stats

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

    def close(self) -> None:
        """关闭底层连接。"""
        self.http_client.close()

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
    ) -> ChatCompletion:
        """执行一次聊天请求。"""
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if enable_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        kwargs["extra_body"] = {
            "thinking": {"type": "disabled" if enable_web_search else "enabled"}
        }

        if enable_web_search:
            kwargs["tools"] = [{"type": "builtin_function", "function": {"name": "$web_search"}}]

        if self.enable_stream:
            return self._create_completion_streaming(kwargs)

        return cast(ChatCompletion, self.client.chat.completions.create(**kwargs))

    def _create_completion_streaming(self, kwargs: Dict[str, Any]) -> ChatCompletion:
        """流式聚合结果。"""
        stream_kwargs = dict(kwargs)
        stream_kwargs["stream"] = True
        stream = cast(Any, self.client.chat.completions.create(**stream_kwargs))

        content_parts: List[str] = []
        finish_reason: Optional[str] = None
        usage_dict: Optional[Dict[str, Any]] = None

        for chunk in stream:
            chunk_dict = cast(Any, chunk).model_dump(exclude_none=True)
            choices = chunk_dict.get("choices", [])
            if not choices:
                continue
            choice0 = choices[0]
            delta = choice0.get("delta", {})
            if isinstance(delta.get("content"), str):
                content_parts.append(delta["content"])
            if isinstance(choice0.get("finish_reason"), str):
                finish_reason = choice0["finish_reason"]
            if isinstance(chunk_dict.get("usage"), dict):
                usage_dict = chunk_dict["usage"]

        assembled = {
            "id": f"stream-assembled-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason or "stop",
                    "message": {"role": "assistant", "content": "".join(content_parts)},
                }
            ],
        }
        if usage_dict:
            assembled["usage"] = usage_dict

        return ChatCompletion.model_validate(assembled)

    @retry(
        retry=retry_if_exception_type((openai.APIError, openai.APITimeoutError, openai.APIConnectionError)),
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
    ) -> ChatCompletion:
        """统一 API 调用入口。"""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self._create_completion(
            messages=messages,
            enable_json_mode=enable_json_mode,
            enable_web_search=enable_web_search,
            max_tokens=max_tokens,
        )

        if enable_web_search:
            for _ in range(MAX_WEB_SEARCH_TOOL_ROUNDS):
                if not response.choices:
                    break
                choice = response.choices[0]
                if choice.finish_reason != "tool_calls":
                    break

                messages.append(choice.message.model_dump(exclude_none=True))
                tool_calls = choice.message.tool_calls or []
                if not tool_calls:
                    break

                for tool_call in tool_calls:
                    tc = tool_call.model_dump(exclude_none=True)
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

                response = self._create_completion(
                    messages=messages,
                    enable_json_mode=enable_json_mode,
                    enable_web_search=enable_web_search,
                    max_tokens=max_tokens,
                )

            if response.choices and response.choices[0].finish_reason == "tool_calls":
                raise RuntimeError("联网工具调用超过最大回合数，疑似死循环。")

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
        return "invalid authentication" in lowered or "invalid_authentication_error" in lowered
