#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Kimi API 知识库构建脚本
================================================================================
功能：基于 Kimi API (kimi-k2.6) 自动生成结构化知识库（JSON Lines 格式）
流程：主题调研 → 三级问题清单生成 → 逐个问题深度分析
作者：AI Agent (Kimi)
依赖：pip install openai tenacity tqdm

【Kimi API 关键配置】
- 模型名称：kimi-k2.6
- Base URL：https://api.moonshot.cn/v1
- JSON Mode：response_format={"type": "json_object"}
- 联网搜索：builtin_function.$web_search（需通过 extra_body 传 thinking={"type":"disabled"}）
- SDK：OpenAI Python SDK 兼容（pip install openai）
- API Key：通过环境变量 MOONSHOT_API_KEY 读取

【使用示例】
1. 基础用法（默认初学者级别，300个问题）：
   export MOONSHOT_API_KEY="your-api-key"
   python knowledge_base_builder.py --topic "量子计算"

2. 指定受众和输出路径：
   python knowledge_base_builder.py --topic "React 18" --audience intermediate --output ./react_kb.jsonl

3. 断点续传（从第 50 个问题继续）：
   python knowledge_base_builder.py --topic "Docker" --resume 50

4. 快速测试（仅生成 10 个问题）：
   python knowledge_base_builder.py --topic "Kubernetes" --max-questions 10
================================================================================
"""

import argparse
import json
import os
import re
import sys
import time
from decimal import Decimal, InvalidOperation
from datetime import datetime
from typing import Any, Optional, cast

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
from tqdm import tqdm

# ==============================================================================
# 全局常量配置
# ==============================================================================
MODEL_NAME = "kimi-k2.6"  # Kimi API 模型名称
BASE_URL = "https://api.moonshot.cn/v1"  # Kimi API Base URL（可被 MOONSHOT_BASE_URL 覆盖）
DEFAULT_MAX_QUESTIONS = 300  # 默认最大问题数（初级100+中级100+高级100）
DEFAULT_AUDIENCE = "beginner"  # 默认目标受众
DEFAULT_OUTPUT = "./knowledge_base.jsonl"  # 默认输出文件路径
QUESTIONS_PER_LEVEL = 100  # 每个级别生成的问题数量
FLUSH_INTERVAL = 10  # 每 N 条记录强制刷新磁盘
COST_REPORT_INTERVAL = 20  # 每 N 个问题报告一次累计消耗
MAX_WEB_SEARCH_TOOL_ROUNDS = 6  # 联网搜索工具调用的最大回合数（防止死循环）
STREAM_LOG_CHUNK_INTERVAL = 20  # 流式输出每 N 个 chunk 打印一次进度日志
DISABLE_THINKING_BY_DEFAULT = True  # 默认禁用 thinking，避免长时间仅输出 reasoning_content
MAX_TOKENS_RESEARCH = 1800  # 阶段1调研输出上限
MAX_TOKENS_GENERATE_QUESTIONS = 3200  # 阶段2问题清单输出上限
MAX_TOKENS_SEARCH = 1200  # 阶段3搜索摘要输出上限
MAX_TOKENS_ANALYSIS = 1600  # 阶段3结构化分析输出上限
MAX_RESEARCH_SUMMARY_PROMPT_CHARS = 3000  # 注入提示词的调研摘要最大字符数
MAX_SEARCH_RESULT_PROMPT_CHARS = 2500  # 注入提示词的搜索结果最大字符数

# Token 消耗估算参数（用于成本预估）
AVG_INPUT_TOKENS_PER_QUESTION = 800  # 每个问题平均输入 token
AVG_OUTPUT_TOKENS_PER_QUESTION = 600  # 每个问题平均输出 token
AVG_INPUT_TOKENS_RESEARCH = 500  # 调研阶段平均输入 token
AVG_OUTPUT_TOKENS_RESEARCH = 800  # 调研阶段平均输出 token
AVG_INPUT_TOKENS_GENERATE = 400  # 问题生成阶段平均输入 token
AVG_OUTPUT_TOKENS_GENERATE = 1200  # 问题生成阶段平均输出 token
MIN_RESEARCH_SUMMARY_BYTES = 200  # 阶段1调研摘要最小字节阈值（低于该值视为失败）
BALANCE_ENDPOINT = "/users/me/balance"  # 查询账户余额接口


class ResearchQualityError(Exception):
    """阶段 1 调研质量不达标异常（需终止流程）。"""


class KnowledgeBaseBuilder:
    """
    知识库构建器（面向对象封装）

    核心职责：
    1. 初始化 OpenAI 客户端（Kimi API 兼容模式）
    2. 执行三阶段构建流程：调研 → 生成问题 → 深度分析
    3. 管理输出文件、断点续传、进度追踪
    4. 实现指数退避重试、JSON 容错解析
    """

    @staticmethod
    def _load_env_file(env_path: str, override: bool = True) -> None:
        """
        从 .env 文件加载环境变量

        Args:
            env_path: .env 文件路径
            override: 若为 True，则使用 .env 值覆盖现有同名环境变量
        """
        if not os.path.exists(env_path):
            return

        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue

                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")

                    if not key:
                        continue

                    if override or key not in os.environ:
                        os.environ[key] = value
        except Exception as e:
            print(f"[警告] 读取 .env 文件失败（{env_path}）：{e}")

    def __init__(
        self,
        topic: str,
        audience: str = DEFAULT_AUDIENCE,
        output_path: str = DEFAULT_OUTPUT,
        markdown_output_path: Optional[str] = None,
        resume_from: int = 0,
        max_questions: int = DEFAULT_MAX_QUESTIONS,
        enable_stream: bool = True,
        verbose: bool = True,
    ):
        """
        初始化构建器

        Args:
            topic: 知识库主题（必填）
            audience: 目标受众（beginner/intermediate/advanced）
            output_path: 输出文件路径
            markdown_output_path: Markdown 输出路径（为空时根据 output_path 自动推导）
            resume_from: 断点续传起始序号（0 表示从头开始）
            max_questions: 最大问题数限制
            enable_stream: 是否启用流式输出
            verbose: 是否输出详细调试日志
        """
        self.topic = topic
        self.audience = audience
        self.output_path = output_path
        self.markdown_output_dir = self._normalize_markdown_output_dir(markdown_output_path, output_path)
        self.resume_from = resume_from
        self.max_questions = max_questions
        self.enable_stream = enable_stream
        self.verbose = verbose

        # 统计计数器
        self.total_api_calls = 0  # API 总调用次数
        self.total_input_tokens = 0  # 累计输入 token
        self.total_output_tokens = 0  # 累计输出 token
        self.start_time = time.time()  # 起始时间戳

        # 先尝试从脚本目录和当前工作目录读取 .env（以 .env 为准，避免旧环境变量干扰）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self._load_env_file(os.path.join(script_dir, ".env"), override=True)
        if os.getcwd() != script_dir:
            self._load_env_file(os.path.join(os.getcwd(), ".env"), override=True)

        # 初始化 OpenAI 客户端（Kimi API 兼容模式）
        api_key = os.environ.get("MOONSHOT_API_KEY")
        self.base_url = os.environ.get("MOONSHOT_BASE_URL", BASE_URL)
        if not api_key:
            print(
                "[错误] 未找到环境变量 MOONSHOT_API_KEY。\n"
                "请在项目根目录创建 .env 并写入：MOONSHOT_API_KEY=your-key\n"
                "可选：通过 MOONSHOT_BASE_URL 指定 API 地址（默认 https://api.moonshot.cn/v1）\n"
                "Linux/macOS: export MOONSHOT_API_KEY='your-key'\n"
                "PowerShell: $env:MOONSHOT_API_KEY='your-key'"
            )
            sys.exit(1)

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
        )
        self.api_key = api_key

        # 初始化输出文件（追加模式，支持断点续传）
        self.output_file = open(self.output_path, "a", encoding="utf-8")
        print(f"[初始化] 输出文件已打开：{os.path.abspath(self.output_path)}")

        # 初始化 Markdown 多文件目录结构
        os.makedirs(self.markdown_output_dir, exist_ok=True)
        self.summary_markdown_path = os.path.join(self.markdown_output_dir, "01_research_summary.md")
        self.question_list_markdown_paths = {
            "beginner": os.path.join(self.markdown_output_dir, "02_beginner_questions.md"),
            "intermediate": os.path.join(self.markdown_output_dir, "03_intermediate_questions.md"),
            "advanced": os.path.join(self.markdown_output_dir, "04_advanced_questions.md"),
        }
        self.answers_markdown_dir = os.path.join(self.markdown_output_dir, "answers")
        os.makedirs(self.answers_markdown_dir, exist_ok=True)
        print(f"[初始化] Markdown 输出目录已准备：{os.path.abspath(self.markdown_output_dir)}")

        self.resume_markdown_append = self.resume_from > 0 and os.path.exists(self.markdown_output_dir)

        # 阶段 2 生成的问题清单缓存
        self.all_questions: list[dict[str, Any]] = []

        # 调研摘要缓存，便于后续阶段提示词复用
        self.research_summary = ""

    @staticmethod
    def _derive_markdown_output_dir(output_path: str) -> str:
        """根据 JSONL 输出路径推导 Markdown 输出目录。"""
        stem, ext = os.path.splitext(output_path)
        return f"{stem}_markdown" if ext else f"{output_path}_markdown"

    @classmethod
    def _normalize_markdown_output_dir(cls, markdown_output_path: Optional[str], output_path: str) -> str:
        """将用户传入的 markdown 路径标准化为输出目录。"""
        if not markdown_output_path:
            return cls._derive_markdown_output_dir(output_path)

        path = markdown_output_path
        _, ext = os.path.splitext(path)
        if ext.lower() == ".md":
            return os.path.splitext(path)[0] + "_files"
        return path

    @staticmethod
    def _sanitize_filename(text: str, max_length: int = 60) -> str:
        """将题目文本转为适合文件名的简短 slug。"""
        cleaned = re.sub(r'[<>:"/\\|?*\r\n\t]+', '_', text).strip()
        cleaned = re.sub(r'\s+', '_', cleaned)
        cleaned = re.sub(r'_+', '_', cleaned)
        cleaned = cleaned.strip('._')
        if not cleaned:
            return "question"
        return cleaned[:max_length]

    @staticmethod
    def _write_text_file(path: str, text: str, mode: str = "w") -> None:
        """写入单个文本文件。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, mode, encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())

    def _build_markdown_meta_block(self) -> str:
        """生成统一的 Markdown 元信息块。"""
        return (
            f"# 知识库：{self.topic}\n\n"
            f"- 受众：`{self.audience}`\n"
            f"- 生成时间：`{datetime.now().isoformat()}`\n"
            f"- JSONL 输出：`{os.path.abspath(self.output_path)}`\n\n"
        )

    def _write_markdown_research(self, summary: str) -> None:
        """写入阶段 1 调研摘要。"""
        content = (
            f"{self._build_markdown_meta_block()}"
            "## 1. 主题调研摘要\n\n"
            f"{summary}\n\n"
        )
        self._write_text_file(self.summary_markdown_path, content)

    def _write_markdown_question_list(self, level_cn: str, level_en: str, questions: list[str]) -> None:
        """写入阶段 2 问题清单。"""
        target_path = self.question_list_markdown_paths.get(level_en, os.path.join(self.markdown_output_dir, f"questions_{level_en}.md"))
        lines = [
            f"{self._build_markdown_meta_block()}"
            f"## 2.{level_en} 问题清单（{level_cn}）\n\n"
        ]
        for idx, question in enumerate(questions, start=1):
            lines.append(f"{idx}. {question}\n")
        self._write_text_file(target_path, "".join(lines))

    def _write_markdown_analysis(self, record: dict[str, Any]) -> None:
        """写入阶段 3 的单题分析。"""
        key_points = record.get("key_points", [])
        sources = record.get("sources", [])
        question_slug = self._sanitize_filename(str(record["question"]))
        answer_path = os.path.join(self.answers_markdown_dir, f"{int(record['id']):04d}_{question_slug}.md")

        lines = [
            f"{self._build_markdown_meta_block()}"
            f"## 3.{record['id']} {record['question']}\n\n",
            f"- 级别：`{record['level']}`\n",
            f"- 难度：`{record.get('difficulty', record['level'])}`\n\n",
            "### 分析\n\n",
            f"{record['analysis']}\n\n",
            "### 要点\n\n",
        ]

        if isinstance(key_points, list) and key_points:
            for item in cast(list[Any], key_points):
                lines.append(f"- {item}\n")
        else:
            lines.append("- 无\n")

        lines.append("\n### 来源\n\n")
        if isinstance(sources, list) and sources:
            for item in cast(list[Any], sources):
                lines.append(f"- {item}\n")
        else:
            lines.append("- 无\n")

        self._write_text_file(answer_path, "".join(lines))

    def _log(self, message: str, level: str = "INFO") -> None:
        """
        统一日志输出（支持按 verbose 开关控制）。
        """
        if not self.verbose and level == "DEBUG":
            return
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{level} {ts}] {message}")

    @staticmethod
    def _to_decimal(value: Any) -> Optional[Decimal]:
        """将余额字段安全转换为 Decimal。"""
        if value is None:
            return None
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return None

    @staticmethod
    def _pick_balance_value(data: dict[str, Any], candidates: list[str]) -> Optional[Decimal]:
        """按候选字段名提取余额值。"""
        for key in candidates:
            if key in data:
                parsed = KnowledgeBaseBuilder._to_decimal(data.get(key))
                if parsed is not None:
                    return parsed
        return None

    def _fetch_balance(self) -> Optional[dict[str, Decimal]]:
        """调用余额接口，返回可用/代金券/现金余额。"""
        url = self.base_url.rstrip("/") + BALANCE_ENDPOINT
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            with httpx.Client(timeout=20.0) as client:
                resp = client.get(url, headers=headers)
                resp.raise_for_status()

            payload = cast(dict[str, Any], resp.json())
            data = cast(dict[str, Any], payload.get("data", {}))

            available = self._pick_balance_value(
                data,
                ["available_balance", "availableBalance", "available", "balance"],
            )
            voucher = self._pick_balance_value(
                data,
                ["voucher_balance", "voucherBalance", "voucher"],
            )
            cash = self._pick_balance_value(
                data,
                ["cash_balance", "cashBalance", "cash"],
            )

            result: dict[str, Decimal] = {}
            if available is not None:
                result["available"] = available
            if voucher is not None:
                result["voucher"] = voucher
            if cash is not None:
                result["cash"] = cash

            if not result:
                self._log("余额接口返回成功，但未识别到余额字段。", "WARNING")
                return None

            return result
        except Exception as e:
            self._log(f"查询余额失败：{e}", "WARNING")
            return None

    @staticmethod
    def _format_decimal(value: Decimal) -> str:
        """格式化余额显示。"""
        return f"{value:.4f}"

    def _print_balance_snapshot(self, title: str, balance: Optional[dict[str, Decimal]]) -> None:
        """打印余额快照。"""
        print("\n" + "=" * 60)
        print(f"💰 {title}")
        print("=" * 60)
        if not balance:
            print("余额查询失败或字段无法解析。")
            print("=" * 60)
            return

        available = balance.get("available")
        voucher = balance.get("voucher")
        cash = balance.get("cash")
        if available is not None:
            print(f"可用余额：{self._format_decimal(available)}")
        if voucher is not None:
            print(f"代金券余额：{self._format_decimal(voucher)}")
        if cash is not None:
            print(f"现金余额：{self._format_decimal(cash)}")
        print("=" * 60)

    def _print_balance_delta(
        self,
        before: Optional[dict[str, Decimal]],
        after: Optional[dict[str, Decimal]],
    ) -> None:
        """打印余额差值（花费）。"""
        print("\n" + "=" * 60)
        print("🧾 本次执行余额变化")
        print("=" * 60)

        if not before or not after:
            print("余额快照不完整，无法计算本次花费。")
            print("=" * 60)
            return

        keys = ["available", "voucher", "cash"]
        labels = {
            "available": "可用余额变化",
            "voucher": "代金券变化",
            "cash": "现金余额变化",
        }

        has_any = False
        for key in keys:
            if key in before and key in after:
                delta = before[key] - after[key]
                has_any = True
                print(f"{labels[key]}：{self._format_decimal(delta)}")

        if "available" in before and "available" in after:
            spent = before["available"] - after["available"]
            print(f"本次总花费（按可用余额差值）：{self._format_decimal(spent)}")
        elif not has_any:
            print("缺少可用余额字段，无法计算总花费。")

        print("=" * 60)

    def _create_chat_completion_streaming(
        self,
        kwargs: dict[str, Any],
    ) -> ChatCompletion:
        """
        通过 stream=True 执行请求，并将 chunks 聚合为 ChatCompletion。
        """
        kwargs = dict(kwargs)
        kwargs["stream"] = True

        self._log("开始流式请求，等待首个 chunk...", "DEBUG")
        stream = cast(Any, self.client.chat.completions.create(**kwargs))

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason: str | None = None
        role = "assistant"
        usage_dict: dict[str, Any] | None = None
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        chunk_count = 0

        for chunk in stream:
            chunk_any: Any = chunk
            chunk_dict: dict[str, Any] = chunk_any.model_dump(exclude_none=True)
            chunk_count += 1
            if chunk_count == 1:
                self._log("已收到首个 chunk，开始持续输出。", "INFO")
            elif chunk_count % STREAM_LOG_CHUNK_INTERVAL == 0:
                self._log(
                    (
                        f"流式进度：已接收 {chunk_count} 个 chunks，"
                        f"content累计 {sum(len(x) for x in content_parts)} 字符，"
                        f"reasoning累计 {sum(len(x) for x in reasoning_parts)} 字符"
                    ),
                    "DEBUG",
                )

            choices = chunk_dict.get("choices")
            if not isinstance(choices, list) or not choices:
                continue

            choice0 = cast(dict[str, Any], choices[0])

            fr = choice0.get("finish_reason")
            if isinstance(fr, str):
                finish_reason = fr

            delta = cast(dict[str, Any], choice0.get("delta", {}))
            delta_role = delta.get("role")
            if isinstance(delta_role, str) and delta_role:
                role = delta_role

            delta_content = delta.get("content")
            if isinstance(delta_content, str) and delta_content:
                content_parts.append(delta_content)

            delta_reasoning = delta.get("reasoning_content")
            if isinstance(delta_reasoning, str) and delta_reasoning:
                reasoning_parts.append(delta_reasoning)

            delta_tool_calls = delta.get("tool_calls")
            if isinstance(delta_tool_calls, list):
                for tc_dict in cast(list[dict[str, Any]], delta_tool_calls):
                    tc_index = int(tc_dict.get("index", 0))

                    acc = tool_calls_acc.setdefault(
                        tc_index,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )

                    if "id" in tc_dict:
                        acc["id"] = tc_dict["id"]

                    # 注意：Kimi 内置工具在流式分片中 type 可能为 builtin_function，
                    # 但 ChatCompletionMessageToolCall 的标准类型是 function。
                    # 为保证聚合后的 ChatCompletion 可被 Pydantic 正确校验，这里固定为 function。
                    acc["type"] = "function"

                    fn_raw = tc_dict.get("function", {})
                    fn: dict[str, Any] = cast(dict[str, Any], fn_raw) if isinstance(fn_raw, dict) else {}
                    if "name" in fn:
                        acc["function"]["name"] = fn["name"]
                    if "arguments" in fn:
                        acc["function"]["arguments"] += fn["arguments"]

            usage_candidate_raw = chunk_dict.get("usage")
            if isinstance(usage_candidate_raw, dict):
                usage_dict = cast(dict[str, Any], usage_candidate_raw)

            choice_usage_candidate = (
                cast(dict[str, Any], cast(list[Any], choices)[0]).get("usage")
                if choices
                else None
            )
            if isinstance(choice_usage_candidate, dict):
                usage_dict = cast(dict[str, Any], choice_usage_candidate)

        self._log(
            (
                f"流式请求完成：chunks={chunk_count}, finish_reason={finish_reason}, "
                f"content_len={sum(len(x) for x in content_parts)}, "
                f"reasoning_len={sum(len(x) for x in reasoning_parts)}"
            ),
            "INFO",
        )

        message: dict[str, Any] = {
            "role": role,
            "content": "".join(content_parts),
        }
        if tool_calls_acc:
            message["tool_calls"] = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]

        response_dict: dict[str, Any] = {
            "id": f"stream-assembled-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason or "stop",
                    "message": message,
                }
            ],
        }

        if usage_dict:
            response_dict["usage"] = usage_dict

        return ChatCompletion.model_validate(response_dict)

    # ==========================================================================
    # 底层 API 调用方法（带指数退避重试）
    # ==========================================================================

    def _create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        enable_json_mode: bool,
        enable_web_search: bool,
        max_tokens: Optional[int] = None,
    ) -> ChatCompletion:
        """
        执行一次 Chat Completions 请求。
        """
        kwargs: dict[str, Any] = {
            "model": MODEL_NAME,
            "messages": messages,
        }

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if enable_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        if DISABLE_THINKING_BY_DEFAULT:
            # 对所有请求默认禁用 thinking，减少“长时间只产生 reasoning_content”导致的卡住感
            extra_body = cast(dict[str, Any], kwargs.get("extra_body", {}))
            extra_body["thinking"] = {"type": "disabled"}
            kwargs["extra_body"] = extra_body

        if enable_web_search:
            kwargs["tools"] = [
                {
                    "type": "builtin_function",
                    "function": {"name": "$web_search"},
                }
            ]

        mode = "stream" if self.enable_stream else "non-stream"
        self._log(
            f"发起 {mode} 请求（json_mode={enable_json_mode}, web_search={enable_web_search}, messages={len(messages)}, max_tokens={max_tokens}）",
            "DEBUG",
        )

        if self.enable_stream:
            return self._create_chat_completion_streaming(kwargs)

        return cast(ChatCompletion, self.client.chat.completions.create(**kwargs))

    @retry(
        retry=retry_if_exception_type(
            (openai.APIError, openai.APITimeoutError, openai.APIConnectionError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        enable_json_mode: bool = False,
        enable_web_search: bool = False,
        max_tokens: Optional[int] = None,
    ) -> ChatCompletion:
        """
        底层 API 调用封装（带 tenacity 指数退避重试）

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            enable_json_mode: 是否启用 JSON Mode
            enable_web_search: 是否启用联网搜索
            max_tokens: 限制输出 token 数，避免过长输出

        Returns:
            OpenAI API 返回的完整 response 对象（字典形式）
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 首次请求
        response = self._create_chat_completion(
            messages=messages,
            enable_json_mode=enable_json_mode,
            enable_web_search=enable_web_search,
            max_tokens=max_tokens,
        )

        # 若启用了联网搜索，处理 tool_calls 闭环，直到得到最终可读回复
        if enable_web_search:
            for round_idx in range(MAX_WEB_SEARCH_TOOL_ROUNDS):
                self._log(
                    f"web_search 工具回合 {round_idx + 1}/{MAX_WEB_SEARCH_TOOL_ROUNDS}",
                    "DEBUG",
                )
                if not response.choices:
                    break

                choice = response.choices[0]
                if choice.finish_reason != "tool_calls":
                    break

                assistant_msg = choice.message.model_dump(exclude_none=True)
                messages.append(assistant_msg)

                tool_calls = choice.message.tool_calls or []
                if not tool_calls:
                    break

                self._log(f"检测到 {len(tool_calls)} 个 tool_calls，开始回填。", "DEBUG")

                for tool_call in tool_calls:
                    tool_call_dict = tool_call.model_dump(exclude_none=True)
                    function_dict = cast(dict[str, Any], tool_call_dict.get("function", {}))
                    tool_name = str(function_dict.get("name", "$web_search"))
                    raw_arguments = function_dict.get("arguments", "{}")
                    tool_arguments = raw_arguments if isinstance(raw_arguments, str) else "{}"
                    tool_call_id = str(tool_call_dict.get("id", ""))

                    # 对于内置 $web_search：按官方流程将 arguments 原样回传（标准化为 JSON 字符串）
                    try:
                        tool_result_obj = json.loads(tool_arguments)
                        tool_content = json.dumps(tool_result_obj, ensure_ascii=False)
                    except Exception:
                        tool_content = tool_arguments

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "content": tool_content,
                        }
                    )

                response = self._create_chat_completion(
                    messages=messages,
                    enable_json_mode=enable_json_mode,
                    enable_web_search=enable_web_search,
                    max_tokens=max_tokens,
                )

            if response.choices and response.choices[0].finish_reason == "tool_calls":
                raise RuntimeError(
                    f"联网搜索工具调用超过最大回合数（{MAX_WEB_SEARCH_TOOL_ROUNDS}），已终止以避免死循环。"
                )

        self.total_api_calls += 1

        # 统计 token 消耗（如果有 usage 字段）
        usage = response.usage
        if usage:
            self.total_input_tokens += usage.prompt_tokens
            self.total_output_tokens += usage.completion_tokens
            self._log(
                f"usage 统计：prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}",
                "DEBUG",
            )

        self._log("本次 API 调用完成。", "DEBUG")

        return response

    @staticmethod
    def _is_auth_error(error: Exception) -> bool:
        """
        判断异常是否为鉴权错误（API Key 无效/失效）
        """
        if isinstance(error, openai.AuthenticationError):
            return True

        message = str(error).lower()
        return "invalid authentication" in message or "invalid_authentication_error" in message

    def _raise_if_auth_error(self, error: Exception) -> None:
        """
        若为鉴权错误则立即抛出致命异常，避免继续写入大量降级数据
        """
        if self._is_auth_error(error):
            raise RuntimeError(
                "MOONSHOT_API_KEY 鉴权失败（401 Invalid Authentication）。"
                f"请检查 Key 是否正确、是否过期、是否与当前 Base URL 匹配（当前：{self.base_url}）。"
                "若你使用的是 Moonshot 新版平台，请优先使用 https://api.moonshot.cn/v1。"
            ) from error

    @staticmethod
    def _validate_research_summary(summary: str) -> None:
        """
        校验阶段 1 调研摘要质量（按字节数阈值判断）

        Args:
            summary: 调研摘要文本

        Raises:
            ResearchQualityError: 摘要为空或字节数低于阈值
        """
        size_bytes = len(summary.encode("utf-8"))
        if size_bytes < MIN_RESEARCH_SUMMARY_BYTES:
            raise ResearchQualityError(
                "阶段 1 调研结果过短，疑似异常。"
                f"当前摘要字节数：{size_bytes}，最小要求：{MIN_RESEARCH_SUMMARY_BYTES}。"
            )

    @staticmethod
    def _truncate_for_prompt(text: str, limit: int) -> str:
        """截断注入到后续提示词中的长文本，避免上下文与输出无效膨胀。"""
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...(以下内容已截断)..."

    @staticmethod
    def _extract_message_content(response: ChatCompletion, default: str = "") -> str:
        """
        从 ChatCompletion 中安全提取首条消息文本

        Args:
            response: OpenAI ChatCompletion 响应对象
            default: 提取失败时的默认文本

        Returns:
            首条消息内容字符串
        """
        if not response.choices:
            return default

        content = response.choices[0].message.content
        return content if isinstance(content, str) else default

    def _safe_parse_json(self, text: str, max_retries: int = 2) -> dict[str, Any]:
        """
        安全解析 JSON（带容错和重试）

        Args:
            text: 待解析的 JSON 字符串
            max_retries: JSON 解析失败后的最大重试次数

        Returns:
            解析后的 Python 字典

        Raises:
            json.JSONDecodeError: 当所有重试均失败时抛出
        """
        # 首先尝试直接解析
        for attempt in range(max_retries + 1):
            try:
                cleaned = text.strip()
                # 去除可能的 markdown 代码块标记
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                if attempt == max_retries:
                    raise e
                # 重试前等待
                time.sleep(2**attempt)

        # 理论上不会到达此处，但为了类型安全
        raise json.JSONDecodeError("JSON 解析失败，已耗尽重试次数", text, 0)

    def _write_jsonl(self, record: dict[str, Any], flush: bool = False) -> None:
        """
        将单条记录写入 JSON Lines 文件

        Args:
            record: 要写入的字典对象
            flush: 是否立即强制刷新到磁盘
        """
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        self.output_file.write(line + "\n")
        if flush:
            self.output_file.flush()
            os.fsync(self.output_file.fileno())

    def _print_cost_estimate(self, num_questions: int) -> None:
        """
        打印预估 token 消耗和成本

        Args:
            num_questions: 计划处理的问题总数
        """
        # 阶段 1：调研
        research_input = AVG_INPUT_TOKENS_RESEARCH
        research_output = AVG_OUTPUT_TOKENS_RESEARCH

        # 阶段 2：问题生成（3 个级别，每个级别 1 次调用）
        generate_calls = 3
        generate_input = AVG_INPUT_TOKENS_GENERATE * generate_calls
        generate_output = AVG_OUTPUT_TOKENS_GENERATE * generate_calls

        # 阶段 3：逐个分析问题
        analyze_input = AVG_INPUT_TOKENS_PER_QUESTION * num_questions
        analyze_output = AVG_OUTPUT_TOKENS_PER_QUESTION * num_questions

        total_input = research_input + generate_input + analyze_input
        total_output = research_output + generate_output + analyze_output
        total_tokens = total_input + total_output

        print("\n" + "=" * 60)
        print("📊 Token 消耗预估（仅供参考）")
        print("=" * 60)
        print(
            f"阶段 1 调研：        输入 ≈ {research_input:,}  | 输出 ≈ {research_output:,}"
        )
        print(
            f"阶段 2 问题生成：    输入 ≈ {generate_input:,}  | 输出 ≈ {generate_output:,}"
        )
        print(
            f"阶段 3 深度分析：    输入 ≈ {analyze_input:,}  | 输出 ≈ {analyze_output:,}"
        )
        print("-" * 60)
        print(
            f"总计预估 Token：     {total_tokens:,}（输入 {total_input:,} + 输出 {total_output:,}）"
        )
        print(f"API 预估调用次数：   {1 + generate_calls + num_questions}")
        print("=" * 60 + "\n")

    def _print_progress_report(self, completed: int) -> None:
        """
        打印阶段性进度和累计消耗报告

        Args:
            completed: 已完成的问题数量
        """
        elapsed = time.time() - self.start_time
        print("\n" + "-" * 60)
        print(f"⏱️  已运行时间：{elapsed:.1f} 秒")
        print(f"✅ 已完成问题：{completed} / {self.max_questions}")
        print(f"📡 API 调用次数：{self.total_api_calls}")
        print(f"📝 累计输入 Token：{self.total_input_tokens:,}")
        print(f"📝 累计输出 Token：{self.total_output_tokens:,}")
        print("-" * 60 + "\n")

    # ==========================================================================
    # 阶段 1：主题调研（联网搜索）
    # ==========================================================================

    def phase1_research(self) -> dict[str, Any]:
        """
        阶段 1：对主题进行联网搜索调研

        Returns:
            调研结果字典，包含 summary 和 raw_response
        """
        print(f"\n{'='*60}")
        print(f"🔍 阶段 1：主题调研 —— {self.topic}")
        print(f"{'='*60}")

        system_prompt = (
            f"你是 {self.topic} 领域的资深技术专家。"
            "请基于搜索结果，用中文输出该主题的结构化调研摘要。"
        )
        user_prompt = (
            f"请搜索网络，调研 '{self.topic}' 主题的以下内容：\n"
            f"1. 核心定义与基本概念\n"
            f"2. 主要技术分支与组成部分\n"
            f"3. 学习路径与关键里程碑\n"
            f"4. 当前行业应用场景\n"
            f"5. 与相关技术的关系\n\n"
            f"请以结构化的方式输出调研摘要，便于后续生成教学问题清单。"
        )

        try:
            response = self._call_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                enable_web_search=True,
                max_tokens=MAX_TOKENS_RESEARCH,
            )
            content = self._extract_message_content(response)
            self._validate_research_summary(content)
            print("[阶段 1] 调研完成，摘要长度：{} 字符".format(len(content)))

            research_record: dict[str, Any] = {
                "phase": 1,
                "type": "research",
                "topic": self.topic,
                "timestamp": datetime.now().isoformat(),
                "summary": content,
            }
            self._write_jsonl(research_record, flush=True)
            if not self.resume_markdown_append:
                self._write_markdown_research(content)
            return research_record

        except Exception as e:
            self._raise_if_auth_error(e)
            if isinstance(e, ResearchQualityError):
                raise RuntimeError(f"阶段 1 调研失败：{e}") from e
            print(f"[错误] 阶段 1 调研失败：{e}")
            # 即使调研失败，也记录错误并继续，使用降级内容
            fallback_record: dict[str, Any] = {
                "phase": 1,
                "type": "research",
                "topic": self.topic,
                "timestamp": datetime.now().isoformat(),
                "summary": f"调研失败（错误：{e}），将使用主题默认值继续生成问题。",
                "error": str(e),
            }
            self._write_jsonl(fallback_record, flush=True)
            return fallback_record

    # ==========================================================================
    # 阶段 2：三级问题清单生成（JSON Mode）
    # ==========================================================================

    def phase2_generate_questions(self, research_summary: str) -> list[dict[str, Any]]:
        """
        阶段 2：生成三级问题清单（初级/中级/高级各 100 问）

        Args:
            research_summary: 阶段 1 的调研摘要文本

        Returns:
            合并后的全部问题列表，每个问题包含 id, level, question 字段
        """
        print(f"\n{'='*60}")
        print(f"📝 阶段 2：生成三级问题清单")
        print(f"{'='*60}")

        levels = [
            ("beginner", "初学者"),
            ("intermediate", "中级学习者"),
            ("advanced", "高级学习者"),
        ]

        all_questions: list[dict[str, Any]] = []
        question_id = 0

        for level_en, level_cn in levels:
            print(
                f"\n[阶段 2] 正在生成 {level_cn} 级别（{level_en}）的 {QUESTIONS_PER_LEVEL} 个问题..."
            )
            self._log(f"阶段2开始：生成 {level_cn} 问题清单", "INFO")

            system_prompt = (
                f"你是 {self.topic} 领域的专家。"
                f"请列出 {self.topic} 的 {level_cn} 100 个问题清单。"
                f"只输出 JSON，不要回答任何问题，不要添加额外解释。"
            )

            user_prompt = (
                f"基于以下调研摘要，生成 {self.topic} 的 {level_cn}（{level_en}）100 个问题清单。\n\n"
                f"调研摘要：\n{self._truncate_for_prompt(research_summary, MAX_RESEARCH_SUMMARY_PROMPT_CHARS)}\n\n"
                f"要求：\n"
                f"1. 问题应覆盖该主题在 {level_cn} 阶段必须掌握的核心知识点\n"
                f"2. 问题应具体、明确，便于后续逐一深度分析\n"
                f"3. 问题难度应符合 {level_cn} 水平\n"
                f"4. 每个问题尽量简洁，建议控制在 10~30 个汉字内\n"
                f"5. 只输出 JSON，不要有任何其他文字\n\n"
                f"输出格式（严格遵守）：\n"
                f'{{"level":"{level_en}","topic":"{self.topic}","questions":["问题1","问题2",...]}}'
            )

            try:
                response = self._call_api(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    enable_json_mode=True,
                    max_tokens=MAX_TOKENS_GENERATE_QUESTIONS,
                )
                content = self._extract_message_content(response, default="{}")
                parsed = self._safe_parse_json(content)

                questions = parsed.get("questions", [])
                level_label = parsed.get("level", level_en)
                topic_label = parsed.get("topic", self.topic)

                print(f"[阶段 2] {level_cn} 级别生成完成，获得 {len(questions)} 个问题")

                # 如果生成的问题数不足 100，补充占位问题（确保数量）
                while len(questions) < QUESTIONS_PER_LEVEL:
                    questions.append(
                        f"[{self.topic} {level_cn}] 补充问题 {len(questions) + 1}"
                    )
                # 如果超过 100，截断
                questions = questions[:QUESTIONS_PER_LEVEL]

                # 写入阶段 2 的原始输出记录
                phase2_record: dict[str, Any] = {
                    "phase": 2,
                    "type": "question_list",
                    "level": level_label,
                    "topic": topic_label,
                    "timestamp": datetime.now().isoformat(),
                    "questions": questions,
                }
                self._write_jsonl(phase2_record)
                if not self.resume_markdown_append:
                    self._write_markdown_question_list(level_cn, level_en, questions)

                # 将问题加入主清单
                for q in questions:
                    question_id += 1
                    all_questions.append(
                        {
                            "id": question_id,
                            "level": level_label,
                            "question": q,
                        }
                    )

            except Exception as e:
                self._raise_if_auth_error(e)
                print(f"[错误] {level_cn} 级别问题生成失败：{e}")
                # 降级处理：生成占位问题，确保流程继续
                for i in range(QUESTIONS_PER_LEVEL):
                    question_id += 1
                    all_questions.append(
                        {
                            "id": question_id,
                            "level": level_en,
                            "question": f"[{self.topic} {level_cn}] 默认问题 {i + 1}（生成失败时的降级问题）",
                        }
                    )

                # 记录错误
                error_record: dict[str, Any] = {
                    "phase": 2,
                    "type": "question_list_error",
                    "level": level_en,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                }
                self._write_jsonl(error_record)

        print(f"\n[阶段 2] 问题清单生成完毕，总计 {len(all_questions)} 个问题")
        return all_questions

    # ==========================================================================
    # 阶段 3：逐个问题深度分析（联网搜索 + JSON Mode）
    # ==========================================================================

    def phase3_analyze_questions(self, questions: list[dict[str, Any]]) -> None:
        """
        阶段 3：遍历所有问题，逐个进行联网搜索并深度分析

        Args:
            questions: 阶段 2 生成的完整问题列表
        """
        print(f"\n{'='*60}")
        print(f"🧠 阶段 3：逐个问题深度分析")
        print(f"{'='*60}")

        # 应用 max_questions 限制
        questions = questions[: self.max_questions]
        print(
            f"[阶段 3] 实际需要处理的问题数：{len(questions)}（max_questions={self.max_questions}）"
        )

        # 应用断点续传（跳过已完成的问题）
        start_index = max(0, self.resume_from)
        if start_index > 0:
            print(
                f"[断点续传] 跳过前 {start_index} 个问题，从第 {start_index + 1} 个问题开始"
            )
        questions = questions[start_index:]

        # 初始化进度条
        pbar = tqdm(
            total=len(questions),
            desc="深度分析进度",
            unit="题",
            ncols=80,
            initial=0,
        )

        for idx, q_item in enumerate(questions, start=start_index + 1):
            question_text = q_item["question"]
            level = q_item["level"]
            q_id = q_item["id"]

            pbar.set_description(
                f"[{idx}/{start_index + len(questions)}] {question_text[:30]}..."
            )

            # 步骤 A：联网搜索该问题的相关资料
            search_prompt = (
                f"请搜索网页，搜索以下问题的相关资料和权威解释：\n"
                f"主题：{self.topic}\n"
                f"问题：{question_text}\n\n"
                f"请基于搜索结果，输出一份精简摘要，要求：\n"
                f"1. 总长度控制在 300~600 字内\n"
                f"2. 只保留与该问题直接相关的信息\n"
                f"3. 优先给出定义、关键结论、核心事实、最多 5 条要点\n"
                f"4. 不要展开泛泛而谈，不要输出无关背景。"
            )

            search_result_text = ""
            try:
                self._log(f"阶段3-搜索：问题 {q_id} 开始搜索摘要", "INFO")
                search_response = self._call_api(
                    system_prompt=f"你是 {self.topic} 领域的专家，擅长通过搜索获取权威信息。",
                    user_prompt=search_prompt,
                    enable_web_search=True,
                    max_tokens=MAX_TOKENS_SEARCH,
                )
                search_result_text = self._extract_message_content(search_response)
            except Exception as e:
                self._raise_if_auth_error(e)
                search_result_text = f"搜索阶段出错：{e}"
                print(f"[警告] 问题 {q_id} 搜索失败：{e}")

            # 步骤 B：基于搜索结果，使用 JSON Mode 输出结构化分析
            analysis_system_prompt = (
                f"你是 {self.topic} 领域的资深专家。"
                f"请根据提供的搜索结果，对给定问题进行深度分析。"
                f"以 JSON 格式输出，不要添加任何额外说明文字。"
            )

            analysis_user_prompt = (
                f"请深度分析以下问题，并严格按照 JSON 格式输出：\n\n"
                f"主题：{self.topic}\n"
                f"阶段1调研摘要：\n{self._truncate_for_prompt(self.research_summary, MAX_RESEARCH_SUMMARY_PROMPT_CHARS)}\n\n"
                f"问题：{question_text}\n"
                f"级别：{level}\n\n"
                f"搜索结果参考：\n{self._truncate_for_prompt(search_result_text, MAX_SEARCH_RESULT_PROMPT_CHARS)}\n\n"
                f"输出要求：\n"
                f"1. analysis 字段控制在 200~500 字\n"
                f"2. key_points 最多 5 条\n"
                f"3. sources 最多 3 条\n"
                f"4. 只保留与当前问题最直接相关的信息\n\n"
                f"输出格式（必须严格遵守，确保是合法 JSON）：\n"
                f"{{\n"
                f'  "id": {q_id},\n'
                f'  "level": "{level}",\n'
                f'  "question": "{question_text}",\n'
                f'  "analysis": "深度分析内容（200~500字，结构清晰）",\n'
                f'  "key_points": ["要点1", "要点2", "要点3"],\n'
                f'  "sources": ["来源URL或摘要1", "来源URL或摘要2"],\n'
                f'  "difficulty": "初级/中级/高级"\n'
                f"}}"
            )

            record: dict[str, Any]

            try:
                self._log(f"阶段3-分析：问题 {q_id} 开始生成结构化答案", "INFO")
                analysis_response = self._call_api(
                    system_prompt=analysis_system_prompt,
                    user_prompt=analysis_user_prompt,
                    enable_json_mode=True,
                    max_tokens=MAX_TOKENS_ANALYSIS,
                )
                content = self._extract_message_content(analysis_response, default="{}")
                parsed = self._safe_parse_json(content)

                # 确保必要字段存在
                record = {
                    "phase": 3,
                    "type": "analysis",
                    "id": parsed.get("id", q_id),
                    "level": parsed.get("level", level),
                    "question": parsed.get("question", question_text),
                    "analysis": parsed.get("analysis", "分析内容为空"),
                    "key_points": parsed.get("key_points", []),
                    "sources": parsed.get("sources", []),
                    "difficulty": parsed.get("difficulty", level),
                    "timestamp": datetime.now().isoformat(),
                }

            except Exception as e:
                self._raise_if_auth_error(e)
                # JSON 解析或 API 调用失败时的降级处理
                print(f"[警告] 问题 {q_id} 分析解析失败，使用降级格式：{e}")
                record = {
                    "phase": 3,
                    "type": "analysis",
                    "id": q_id,
                    "level": level,
                    "question": question_text,
                    "analysis": f"分析生成失败（错误：{e}）。原始搜索结果：{search_result_text[:500]}",
                    "key_points": ["分析失败，请手动补充"],
                    "sources": [],
                    "difficulty": level,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                }

            # 写入结果（每 FLUSH_INTERVAL 条强制刷新）
            flush = idx % FLUSH_INTERVAL == 0
            self._write_jsonl(record, flush=flush)
            self._write_markdown_analysis(record)

            # 更新进度条
            pbar.update(1)

            # 每 COST_REPORT_INTERVAL 个问题打印一次累计报告
            if idx % COST_REPORT_INTERVAL == 0:
                self._print_progress_report(idx)

        pbar.close()
        print(f"\n[阶段 3] 全部 {len(questions)} 个问题分析完成！")

    # ==========================================================================
    # 主控流程
    # ==========================================================================

    def run(self) -> None:
        """
        执行完整的知识库构建流程
        """
        before_balance = self._fetch_balance()
        self._print_balance_snapshot("执行前余额", before_balance)

        print("\n" + "=" * 60)
        print("🚀 Kimi API 知识库构建脚本启动")
        print("=" * 60)
        print(f"主题：{self.topic}")
        print(f"受众：{self.audience}")
        print(f"输出：{os.path.abspath(self.output_path)}")
        print(f"Markdown目录：{os.path.abspath(self.markdown_output_dir)}")
        print(f"断点续传：从第 {self.resume_from} 个问题开始")
        print(f"最大问题数：{self.max_questions}")
        print(f"流式输出：{self.enable_stream}")
        print(f"详细日志：{self.verbose}")
        print("=" * 60)

        try:
            # 阶段 1：主题调研
            research_result = self.phase1_research()
            self.research_summary = str(research_result.get("summary", ""))

            # 阶段 2：生成问题清单
            # 如果 resume_from > 0，尝试从已有输出文件恢复问题清单（避免重复生成）
            if self.resume_from > 0 and os.path.exists(self.output_path):
                print("\n[断点续传] 尝试从已有输出文件中恢复问题清单...")
                restored = self._restore_questions_from_file()
                if restored:
                    self.all_questions = restored
                    print(f"[断点续传] 成功恢复 {len(restored)} 个问题")
                else:
                    print("[断点续传] 未能从文件中恢复问题清单，重新生成...")
                    self.all_questions = self.phase2_generate_questions(
                        research_result["summary"]
                    )
            else:
                self.all_questions = self.phase2_generate_questions(
                    research_result["summary"]
                )

            # 打印成本预估
            actual_max = min(len(self.all_questions), self.max_questions)
            self._print_cost_estimate(actual_max)

            # 阶段 3：逐个深度分析
            self.phase3_analyze_questions(self.all_questions)

            # 最终总结
            elapsed = time.time() - self.start_time
            print("\n" + "=" * 60)
            print("🎉 知识库构建全部完成！")
            print("=" * 60)
            print(f"📁 输出文件：{os.path.abspath(self.output_path)}")
            print(f"⏱️  总耗时：{elapsed:.1f} 秒")
            print(f"📡 API 总调用次数：{self.total_api_calls}")
            print(f"📝 累计输入 Token：{self.total_input_tokens:,}")
            print(f"📝 累计输出 Token：{self.total_output_tokens:,}")
            print("=" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n\n[中断] 用户手动中断执行，已保存的进度不会丢失。")
            print(f"[中断] 当前进度已写入：{os.path.abspath(self.output_path)}")
            print(
                f"[中断] 如需续传，请使用 --resume 参数指定已完成的最后一个问题序号。"
            )
            sys.exit(0)

        except Exception as e:
            print(f"\n[致命错误] 构建流程异常终止：{e}")
            print(f"[致命错误] 已保存的进度文件：{os.path.abspath(self.output_path)}")
            # 确保缓冲区数据落盘
            self.output_file.flush()
            os.fsync(self.output_file.fileno())
            sys.exit(1)

        finally:
            after_balance = self._fetch_balance()
            self._print_balance_snapshot("执行后余额", after_balance)
            self._print_balance_delta(before_balance, after_balance)

            # 确保文件正确关闭
            self.output_file.flush()
            os.fsync(self.output_file.fileno())
            self.output_file.close()
            print("[清理] 输出文件已安全关闭。")

    def _restore_questions_from_file(self) -> Optional[list[dict[str, Any]]]:
        """
        从已有的输出文件中恢复问题清单（用于断点续传）

        Returns:
            恢复成功返回问题列表，否则返回 None
        """
        questions: list[dict[str, Any]] = []
        try:
            with open(self.output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if (
                            record.get("phase") == 2
                            and record.get("type") == "question_list"
                        ):
                            level = record.get("level", "unknown")
                            for q in record.get("questions", []):
                                questions.append(
                                    {
                                        "id": len(questions) + 1,
                                        "level": level,
                                        "question": q,
                                    }
                                )
                    except json.JSONDecodeError:
                        continue
            return questions if questions else None
        except Exception:
            return None


# ==============================================================================
# 命令行参数解析与主入口
# ==============================================================================


def main() -> None:
    """
    命令行入口函数
    """
    parser = argparse.ArgumentParser(
        description="Kimi API 知识库构建脚本 —— 基于联网搜索自动生成结构化知识库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  export MOONSHOT_API_KEY="your-api-key"
  python knowledge_base_builder.py --topic "量子计算"
  python knowledge_base_builder.py --topic "React 18" --audience intermediate --output ./react.jsonl
  python knowledge_base_builder.py --topic "Docker" --resume 50 --max-questions 200
        """,
    )

    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="知识库主题（必填，例如：量子计算、React 18、Docker）",
    )
    parser.add_argument(
        "--audience",
        type=str,
        default=DEFAULT_AUDIENCE,
        choices=["beginner", "intermediate", "advanced"],
        help="目标受众级别（默认：beginner）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"输出文件路径（默认：{DEFAULT_OUTPUT}）",
    )
    parser.add_argument(
        "--markdown-output",
        type=str,
        default=None,
        help="Markdown 输出文件路径（默认：根据 --output 自动推导 .md 路径）",
    )
    parser.add_argument(
        "--resume",
        type=int,
        default=0,
        metavar="N",
        help="断点续传，从第 N 个问题开始（默认：0，即从头开始）",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=DEFAULT_MAX_QUESTIONS,
        help=f"最大问题数限制（默认：{DEFAULT_MAX_QUESTIONS}，即三级各100个）",
    )
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用流式输出（默认：启用）",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否输出详细调试日志（默认：启用）",
    )

    args = parser.parse_args()

    # 参数校验
    if args.max_questions <= 0:
        print("[错误] --max-questions 必须大于 0")
        sys.exit(1)
    if args.resume < 0:
        print("[错误] --resume 不能为负数")
        sys.exit(1)

    # 创建构建器并执行
    builder = KnowledgeBaseBuilder(
        topic=args.topic,
        audience=args.audience,
        output_path=args.output,
        markdown_output_path=args.markdown_output,
        resume_from=args.resume,
        max_questions=args.max_questions,
        enable_stream=args.stream,
        verbose=args.verbose,
    )
    builder.run()


if __name__ == "__main__":
    main()
