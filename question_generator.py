"""阶段二：问题清单生成模块。"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Tuple


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

    def generate(
        self, research_summary: str
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """生成三级问题。"""
        levels = [
            ("beginner", "初学者"),
            ("intermediate", "中级学习者"),
            ("advanced", "高级学习者"),
        ]

        all_questions: List[Dict[str, Any]] = []
        phase2_records: List[Dict[str, Any]] = []
        qid = 0

        for level_idx, (level_en, level_cn) in enumerate(levels, start=1):
            system_prompt, user_prompt = self._build_prompts(
                level_en, level_cn, research_summary
            )
            try:
                response = self.api_client.call(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    enable_json_mode=True,
                    max_tokens=PHASE2_MAX_TOKENS,
                    progress_context={
                        "stage_name": "问题清单生成",
                        "stage_index": 2,
                        "stage_total": 3,
                        "substep_name": level_cn,
                        "substep_index": level_idx,
                        "substep_total": len(levels),
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

                phase2_records.append(
                    JsonlRecordFactory.phase2(
                        level=payload.get("level", level_en),
                        topic=payload.get("topic", self.topic),
                        questions=questions,
                    )
                )

                for q in questions:
                    qid += 1
                    item = QuestionItem(
                        id=qid,
                        level=str(payload.get("level", level_en)),
                        question=str(q),
                    )
                    all_questions.append(item.model_dump())
            except Exception as exc:
                self.logger.warning("问题生成失败(level=%s): %s", level_en, exc)
                fallback_questions = [
                    f"[{self.topic} {level_cn}] 默认问题 {i + 1}（生成失败降级）"
                    for i in range(QUESTIONS_PER_LEVEL)
                ]
                phase2_records.append(
                    JsonlRecordFactory.phase2(
                        level=level_en, topic=self.topic, questions=fallback_questions
                    )
                )
                for q in fallback_questions:
                    qid += 1
                    item = QuestionItem(id=qid, level=level_en, question=q)
                    all_questions.append(item.model_dump())

        return all_questions, phase2_records
