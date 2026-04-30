"""项目配置与通用校验模块。"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv  # type: ignore[import-not-found]

DEFAULT_MODEL_NAME = "kimi-k2.6"
DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_OUTPUT = "./knowledge_base.jsonl"
DEFAULT_AUDIENCE = "beginner"
DEFAULT_MAX_QUESTIONS = 300
QUESTIONS_PER_LEVEL = 100
FLUSH_INTERVAL = 10
COST_REPORT_INTERVAL = 20

SUPPORTED_AUDIENCES = {"beginner", "intermediate", "advanced"}

PHASE1_MAX_TOKENS = 10_000
PHASE2_MAX_TOKENS = 20_000
PHASE3_SEARCH_MAX_TOKENS = 20_000
PHASE3_ANALYSIS_MAX_TOKENS = 50_000

MAX_RESEARCH_SUMMARY_PROMPT_CHARS = 5_000
MAX_SEARCH_RESULT_PROMPT_CHARS = 10_000
MIN_RESEARCH_SUMMARY_BYTES = 200

MAX_WEB_SEARCH_TOOL_ROUNDS = 6

CONNECT_TIMEOUT_SECONDS = 10.0
READ_TIMEOUT_SECONDS = 60.0
WRITE_TIMEOUT_SECONDS = 30.0
POOL_TIMEOUT_SECONDS = 60.0


class SensitiveDataFilter(logging.Filter):
    """日志敏感信息过滤器。"""

    def __init__(self, secret: Optional[str] = None) -> None:
        super().__init__()
        self._secret = secret or ""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.getMessage())
        if self._secret:
            msg = msg.replace(self._secret, "****")
        msg = re.sub(r"(MOONSHOT_API_KEY\s*[=:]\s*)([^\s]+)", r"\1****", msg)
        record.msg = msg
        record.args = ()
        return True


class InPlaceProgressHandler(logging.StreamHandler):
    """支持同一行覆盖刷新的控制台日志处理器。"""

    def __init__(self) -> None:
        super().__init__(stream=sys.stderr)
        self._overwrite_active = False
        self._last_overwrite_length = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            overwrite = bool(getattr(record, "overwrite", False))

            if overwrite:
                padding = max(self._last_overwrite_length - len(message), 0)
                self.stream.write("\r" + message + (" " * padding))
                self.flush()
                self._overwrite_active = True
                self._last_overwrite_length = len(message)
                return

            if self._overwrite_active:
                self.stream.write("\n")
                self._overwrite_active = False
                self._last_overwrite_length = 0

            self.stream.write(message + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


@dataclass
class Settings:
    """运行时配置。"""

    topic: str
    audience: str = DEFAULT_AUDIENCE
    output_path: str = DEFAULT_OUTPUT
    markdown_output: Optional[str] = None
    resume: int = 0
    max_questions: int = DEFAULT_MAX_QUESTIONS
    stream: bool = True
    verbose: bool = True

    model_name: str = DEFAULT_MODEL_NAME
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""

    project_root: Path = Path.cwd()


def setup_logging(verbose: bool, secret: Optional[str] = None) -> logging.Logger:
    """初始化结构化日志。"""
    logger = logging.getLogger("knowledge_base_builder")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    handler = InPlaceProgressHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    handler.addFilter(SensitiveDataFilter(secret=secret))

    logger.addHandler(handler)
    logger.propagate = False
    return logger


def load_environment(project_root: Path) -> None:
    """加载 .env 配置（项目根 + 当前目录）。"""
    load_dotenv(project_root / ".env", override=True)
    if Path.cwd() != project_root:
        load_dotenv(Path.cwd() / ".env", override=True)


def sanitize_topic(topic: str) -> str:
    """校验并规范主题输入。"""
    clean = topic.strip()
    if not clean:
        raise ValueError("--topic 不能为空。")
    if len(clean) > 200:
        raise ValueError("--topic 过长，请控制在 200 字符内。")
    if re.search(r"[\x00-\x1f\x7f]", clean):
        raise ValueError("--topic 包含非法控制字符。")
    return clean


def validate_numeric_args(resume: int, max_questions: int) -> None:
    """校验数值参数。"""
    if resume < 0:
        raise ValueError("--resume 不能为负数。")
    if max_questions <= 0:
        raise ValueError("--max-questions 必须大于 0。")


def derive_markdown_output_dir(output_path: str) -> str:
    """根据 JSONL 路径推导 Markdown 输出目录。"""
    path = Path(output_path)
    if path.suffix:
        return str(path.with_suffix("")) + "_markdown"
    return str(path) + "_markdown"


def normalize_markdown_output_dir(
    markdown_output: Optional[str], output_path: str
) -> str:
    """将 markdown 输出参数统一为目录。"""
    if not markdown_output:
        return derive_markdown_output_dir(output_path)

    p = Path(markdown_output)
    if p.suffix.lower() == ".md":
        return str(p.with_suffix("")) + "_files"
    return str(p)


def secure_output_path(output_path: str, project_root: Path) -> Path:
    """输出文件路径安全检查，防止路径穿越。"""
    candidate = Path(output_path)
    if not candidate.is_absolute():
        candidate = project_root / candidate

    resolved = candidate.resolve()
    root = project_root.resolve()

    if root == resolved or root in resolved.parents:
        return resolved

    raise ValueError(
        f"输出路径不安全：{resolved}。请将输出路径限制在项目目录内：{root}。"
    )


def build_settings_from_args(args: object, project_root: Path) -> Settings:
    """由 CLI 参数构建 Settings。"""
    load_environment(project_root)

    topic = sanitize_topic(getattr(args, "topic"))
    audience = str(getattr(args, "audience"))
    if audience not in SUPPORTED_AUDIENCES:
        raise ValueError("--audience 必须是 beginner/intermediate/advanced。")

    resume = int(getattr(args, "resume"))
    max_questions = int(getattr(args, "max_questions"))
    validate_numeric_args(resume=resume, max_questions=max_questions)

    output_path = secure_output_path(str(getattr(args, "output")), project_root)
    markdown_output = getattr(args, "markdown_output")
    stream = bool(getattr(args, "stream"))
    verbose = bool(getattr(args, "verbose"))

    api_key = os.getenv("MOONSHOT_API_KEY", "").strip()
    base_url = (
        os.getenv("MOONSHOT_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    )

    return Settings(
        topic=topic,
        audience=audience,
        output_path=str(output_path),
        markdown_output=markdown_output,
        resume=resume,
        max_questions=max_questions,
        stream=stream,
        verbose=verbose,
        model_name=DEFAULT_MODEL_NAME,
        base_url=base_url,
        api_key=api_key,
        project_root=project_root,
    )
