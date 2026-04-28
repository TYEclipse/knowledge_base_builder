#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kimi API 知识库构建器主入口。"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from api_client import KimiApiClient
from answer_analyzer import AnswerAnalyzer
from config import (
    DEFAULT_AUDIENCE,
    DEFAULT_MAX_QUESTIONS,
    DEFAULT_OUTPUT,
    MIN_RESEARCH_SUMMARY_BYTES,
    PHASE1_MAX_TOKENS,
    build_settings_from_args,
    normalize_markdown_output_dir,
    setup_logging,
)
from models import JsonlRecordFactory, RuntimeStats
from question_generator import QuestionGenerator
from storage import AtomicJsonlWriter, MarkdownWriter


class ResearchQualityError(Exception):
    """阶段一调研质量不达标异常。"""


class KnowledgeBaseBuilder:
    """知识库构建主类。"""

    def __init__(self, settings: Any) -> None:
        """初始化构建器。"""
        if not settings.api_key:
            raise RuntimeError(
                "未检测到 MOONSHOT_API_KEY。请在项目根目录 .env 中配置 MOONSHOT_API_KEY=your-key。"
            )

        self.settings = settings
        self.stats = RuntimeStats()
        self.logger = setup_logging(verbose=settings.verbose, secret=settings.api_key)

        self.api_client = KimiApiClient(
            api_key=settings.api_key,
            base_url=settings.base_url,
            model_name=settings.model_name,
            enable_stream=settings.stream,
            logger=self.logger,
            stats=self.stats,
        )

        output_path = Path(settings.output_path)
        markdown_dir = Path(
            normalize_markdown_output_dir(settings.markdown_output, settings.output_path)
        )

        self.writer = AtomicJsonlWriter(path=output_path)
        self.markdown_writer = MarkdownWriter(
            topic=settings.topic,
            audience=settings.audience,
            output_path=output_path,
            markdown_dir=markdown_dir,
        )

        self.question_generator = QuestionGenerator(
            topic=settings.topic,
            api_client=self.api_client,
            logger=self.logger,
        )
        self.answer_analyzer = AnswerAnalyzer(
            topic=settings.topic,
            api_client=self.api_client,
            logger=self.logger,
            stats=self.stats,
        )

        self.research_summary = ""

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        """截断文本防止提示词过长。"""
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...(以下内容已截断)..."

    @staticmethod
    def _validate_research_summary(summary: str) -> None:
        """校验调研结果质量。"""
        size_bytes = len(summary.encode("utf-8"))
        if size_bytes < MIN_RESEARCH_SUMMARY_BYTES:
            raise ResearchQualityError(
                f"阶段1调研结果过短：{size_bytes} 字节，小于阈值 {MIN_RESEARCH_SUMMARY_BYTES} 字节。"
            )

    def phase1_research(self) -> Dict[str, Any]:
        """阶段1：联网调研。"""
        print("\n" + "=" * 60)
        print(f"🔍 阶段 1：主题调研 —— {self.settings.topic}")
        print("=" * 60)

        system_prompt = (
            f"你是 {self.settings.topic} 领域资深技术专家，请基于检索输出结构化调研摘要。"
        )
        user_prompt = (
            f"请搜索并调研主题 '{self.settings.topic}'：\n"
            "1. 核心定义与概念\n"
            "2. 主要技术分支\n"
            "3. 学习路径和里程碑\n"
            "4. 典型行业应用\n"
            "5. 与相关技术关系\n\n"
            "请用中文结构化输出，便于后续生成问题清单。"
        )

        try:
            response = self.api_client.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                enable_web_search=True,
                max_tokens=PHASE1_MAX_TOKENS,
            )
            summary = self.api_client.extract_message_content(response)
            self._validate_research_summary(summary)

            record = JsonlRecordFactory.phase1(topic=self.settings.topic, summary=summary)
            self.writer.append(record, flush=True)
            if self.settings.resume == 0:
                self.markdown_writer.write_research(summary)
            return record

        except Exception as exc:
            if self.api_client.is_auth_error(exc):
                raise RuntimeError(
                    "MOONSHOT_API_KEY 鉴权失败（401）。请检查 Key 是否正确且与 Base URL 匹配。"
                ) from exc

            fallback_summary = f"调研失败（错误：{exc}），将使用降级内容继续。"
            fallback = JsonlRecordFactory.phase1(
                topic=self.settings.topic,
                summary=fallback_summary,
                error=str(exc),
            )
            self.writer.append(fallback, flush=True)
            return fallback

    def phase2_questions(self, research_summary: str) -> List[Dict[str, Any]]:
        """阶段2：生成三级问题。"""
        print("\n" + "=" * 60)
        print("📝 阶段 2：生成三级问题清单")
        print("=" * 60)

        questions, phase2_records = self.question_generator.generate(research_summary)
        for record in phase2_records:
            self.writer.append(record)
            if self.settings.resume == 0:
                level = str(record.get("level", ""))
                level_cn_map = {
                    "beginner": "初学者",
                    "intermediate": "中级学习者",
                    "advanced": "高级学习者",
                }
                self.markdown_writer.write_question_list(
                    level_en=level,
                    level_cn=level_cn_map.get(level, level),
                    questions=list(record.get("questions", [])),
                )
        self.writer.flush()
        return questions

    def _restore_questions_for_resume(self) -> Optional[List[Dict[str, Any]]]:
        """断点续传时恢复阶段2问题。"""
        return self.writer.restore_phase2_questions()

    @staticmethod
    def _print_cost_estimate(num_questions: int) -> None:
        """打印 token 粗略预估。"""
        avg_input_research = 500
        avg_output_research = 800
        avg_input_generate = 400 * 3
        avg_output_generate = 1200 * 3
        avg_input_analyze = 800 * num_questions
        avg_output_analyze = 600 * num_questions

        total_input = avg_input_research + avg_input_generate + avg_input_analyze
        total_output = avg_output_research + avg_output_generate + avg_output_analyze

        print("\n" + "=" * 60)
        print("📊 Token 消耗预估（仅参考）")
        print("=" * 60)
        print(f"输入约：{total_input:,} | 输出约：{total_output:,}")
        print(f"总计约：{(total_input + total_output):,} tokens")
        print("=" * 60 + "\n")

    def run(self) -> None:
        """执行三阶段流程。"""
        start_time = time.time()

        print("\n" + "=" * 60)
        print("🚀 Kimi API 知识库构建器启动")
        print("=" * 60)
        print(f"主题：{self.settings.topic}")
        print(f"受众：{self.settings.audience}")
        print(f"输出：{self.settings.output_path}")
        print(f"断点续传：{self.settings.resume}")
        print(f"最大问题数：{self.settings.max_questions}")
        print("=" * 60)

        try:
            phase1 = self.phase1_research()
            self.research_summary = str(phase1.get("summary", ""))

            if self.settings.resume > 0:
                restored = self._restore_questions_for_resume()
                if restored:
                    questions = restored
                    self.logger.info("已从历史输出恢复 %d 个问题。", len(questions))
                else:
                    questions = self.phase2_questions(self.research_summary)
            else:
                questions = self.phase2_questions(self.research_summary)

            self._print_cost_estimate(num_questions=min(len(questions), self.settings.max_questions))

            print("\n" + "=" * 60)
            print("🧠 阶段 3：逐题深度分析")
            print("=" * 60)

            self.answer_analyzer.run(
                questions=questions,
                research_summary=self.research_summary,
                resume_from=self.settings.resume,
                max_questions=self.settings.max_questions,
                writer=self.writer,
                markdown_writer=self.markdown_writer,
                start_time=start_time,
            )

            self.writer.flush()
            elapsed = time.time() - start_time
            print("\n" + "=" * 60)
            print("🎉 构建完成")
            print("=" * 60)
            print(f"耗时：{elapsed:.1f}s")
            print(f"API 调用次数：{self.stats.total_api_calls}")
            print(f"输入 Token：{self.stats.total_input_tokens:,}")
            print(f"输出 Token：{self.stats.total_output_tokens:,}")
            print("=" * 60)

        finally:
            self.writer.flush()
            self.api_client.close()


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="Kimi API 知识库构建脚本 —— 调研、问题生成、逐题分析。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--topic", type=str, required=True, help="知识库主题")
    parser.add_argument(
        "--audience",
        type=str,
        default=DEFAULT_AUDIENCE,
        choices=["beginner", "intermediate", "advanced"],
        help="目标受众",
    )
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="JSONL 输出路径")
    parser.add_argument(
        "--markdown-output",
        type=str,
        default=None,
        help="Markdown 输出目录（默认随 output 自动推导）",
    )
    parser.add_argument("--resume", type=int, default=0, help="从第 N 个问题继续")
    parser.add_argument(
        "--max-questions",
        type=int,
        default=DEFAULT_MAX_QUESTIONS,
        help="最多处理问题数",
    )
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用流式输出",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用详细日志",
    )
    return parser


def main() -> None:
    """CLI 主入口。"""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        settings = build_settings_from_args(args=args, project_root=Path(__file__).resolve().parent)
        builder = KnowledgeBaseBuilder(settings=settings)
        builder.run()
    except Exception as exc:
        print(f"[错误] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
