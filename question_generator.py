"""阶段二：问题清单生成模块。"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Iterator, List, Tuple


from config import (
    MAX_RESEARCH_SUMMARY_PROMPT_CHARS,
    PHASE2_MAX_TOKENS,
    QUESTIONS_PER_LEVEL,
)
from models import PHASE2_SCHEMA, JsonlRecordFactory, QuestionItem


class QuestionGenerator:
    """问题清单生成器。"""

    def __init__(self, topic: str, api_client: Any, logger: Any) -> None:
        self.topic = topic
        self.api_client = api_client
        self.logger = logger
        self.reasoning_effort = "high"

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...(以下内容已截断)..."

    def _build_prompts(
        self, level_en: str, level_cn: str, research_summary: str
    ) -> Tuple[str, str]:
        system_prompt = (
            f"你是 {self.topic} 领域的专家。"
            f"请列出 {self.topic} 的 {level_cn} 100 个问题清单。"
            "只输出 JSON，不要回答问题，不要额外解释。"
        )
        user_prompt = (
            f"基于调研摘要，生成 {self.topic} 的 {level_cn}（{level_en}）100 个问题。\n\n"
            f"调研摘要：\n{self._truncate(research_summary, MAX_RESEARCH_SUMMARY_PROMPT_CHARS)}\n\n"
            "要求：\n"
            "1. 覆盖核心知识点\n"
            "2. 问题具体清晰\n"
            "3. 难度符合级别\n"
            "4. 仅输出 JSON\n\n"
            f'格式：{{"level":"{level_en}","topic":"{self.topic}","questions":["问题1","问题2"]}}'
        )
        return system_prompt, user_prompt

    def _generate_level_questions(
        self,
        *,
        level_en: str,
        level_cn: str,
        research_summary: str,
        level_idx: int,
        total_levels: int,
        start_qid: int,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """生成单个难度级别的问题列表。"""
        system_prompt, user_prompt = self._build_prompts(
            level_en, level_cn, research_summary
        )
        try:
            response = self.api_client.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                enable_json_mode=True,
                max_tokens=PHASE2_MAX_TOKENS,
                use_deepseek_thinking=True,
                reasoning_effort=self.reasoning_effort,
                progress_context={
                    "stage_name": "问题清单生成",
                    "stage_index": 2,
                    "stage_total": 3,
                    "substep_name": level_cn,
                    "substep_index": level_idx,
                    "substep_total": total_levels,
                    "request_group": "phase2_generate_questions",
                },
            )
            payload = self.api_client.safe_parse_json(
                self.api_client.extract_message_content(response, default="{}")
            )
            jsonschema = importlib.import_module("jsonschema")
            jsonschema.validate(instance=payload, schema=PHASE2_SCHEMA)

            questions = list(payload.get("questions", []))
            while len(questions) < QUESTIONS_PER_LEVEL:
                questions.append(
                    f"[{self.topic} {level_cn}] 补充问题 {len(questions) + 1}"
                )
            questions = questions[:QUESTIONS_PER_LEVEL]

            record = JsonlRecordFactory.phase2(
                level=payload.get("level", level_en),
                topic=payload.get("topic", self.topic),
                questions=questions,
            )
            level_value = str(payload.get("level", level_en))
        except Exception as exc:
            self.logger.warning("问题生成失败(level=%s): %s", level_en, exc)
            questions = [
                f"[{self.topic} {level_cn}] 默认问题 {i + 1}（生成失败降级）"
                for i in range(QUESTIONS_PER_LEVEL)
            ]
            record = JsonlRecordFactory.phase2(
                level=level_en, topic=self.topic, questions=questions
            )
            level_value = level_en

        items: List[Dict[str, Any]] = []
        for offset, q in enumerate(questions, start=1):
            item = QuestionItem(
                id=start_qid + offset,
                level=level_value,
                question=str(q),
            )
            items.append(item.model_dump())

        return items, record

    def generate_incrementally(
        self, research_summary: str
    ) -> Iterator[tuple[List[Dict[str, Any]], Dict[str, Any]]]:
        """按级别逐次生成问题，便于上层边生成边落盘。"""
        levels = [
            ("beginner", "初学者"),
            ("intermediate", "中级学习者"),
            ("advanced", "高级学习者"),
        ]
        next_qid = 1

        for level_idx, (level_en, level_cn) in enumerate(levels, start=1):
            items, record = self._generate_level_questions(
                level_en=level_en,
                level_cn=level_cn,
                research_summary=research_summary,
                level_idx=level_idx,
                total_levels=len(levels),
                start_qid=next_qid - 1,
            )
            next_qid += len(items)
            yield items, record

    def generate(
        self, research_summary: str
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """生成三级问题。"""
        all_questions: List[Dict[str, Any]] = []
        phase2_records: List[Dict[str, Any]] = []

        for items, record in self.generate_incrementally(research_summary):
            all_questions.extend(items)
            phase2_records.append(record)

        return all_questions, phase2_records
