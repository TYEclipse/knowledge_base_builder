"""阶段三：逐题深度分析模块。"""

from __future__ import annotations

import importlib
from datetime import datetime
from typing import Any, Dict, List

from tqdm import tqdm

from config import (
    COST_REPORT_INTERVAL,
    FLUSH_INTERVAL,
    MAX_RESEARCH_SUMMARY_PROMPT_CHARS,
    MAX_SEARCH_RESULT_PROMPT_CHARS,
    PHASE3_ANALYSIS_MAX_TOKENS,
    PHASE3_SEARCH_MAX_TOKENS,
)
from models import PHASE3_SCHEMA, JsonlRecordFactory, Phase3Response


class AnswerAnalyzer:
    """问题分析器。"""

    def __init__(
        self,
        topic: str,
        api_client: Any,
        logger: Any,
        stats: Any,
        reasoning_effort: str = "max",
    ) -> None:
        self.topic = topic
        self.api_client = api_client
        self.logger = logger
        self.stats = stats
        self.reasoning_effort = reasoning_effort

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...(以下内容已截断)..."

    def _search_context(
        self, question: str, research_summary: str, item_index: int, item_total: int
    ) -> str:
        search_prompt = (
            f"请搜索网页，回答问题相关资料：\n"
            f"主题：{self.topic}\n"
            f"阶段1调研摘要：{self._truncate(research_summary, MAX_SEARCH_RESULT_PROMPT_CHARS)}\n"
            f"问题：{question}\n\n"
            "输出精简摘要：定义、关键结论、核心事实。"
        )
        response = self.api_client.call(
            system_prompt=f"你是 {self.topic} 领域专家，擅长基于检索做事实归纳。",
            user_prompt=search_prompt,
            enable_web_search=True,
            max_tokens=PHASE3_SEARCH_MAX_TOKENS,
            progress_context={
                "stage_name": "逐题深度分析",
                "stage_index": 3,
                "stage_total": 3,
                "substep_name": "搜索资料",
                "item_name": question[:30],
                "item_index": item_index,
                "item_total": item_total,
                "request_group": "phase3_search",
            },
        )
        return self.api_client.extract_message_content(response)

    def _analyze(
        self,
        q_id: int,
        level: str,
        question: str,
        research_summary: str,
        search_result: str,
        item_index: int,
        item_total: int,
    ) -> Dict[str, Any]:
        analysis_prompt = (
            "请深度分析并严格输出 JSON：\n\n"
            f"主题：{self.topic}\n"
            f"阶段1调研摘要：\n{self._truncate(research_summary, MAX_RESEARCH_SUMMARY_PROMPT_CHARS)}\n\n"
            f"问题：{question}\n"
            f"级别：{level}\n\n"
            f"搜索结果：\n{self._truncate(search_result, MAX_SEARCH_RESULT_PROMPT_CHARS)}\n\n"
            "输出格式："
            "{"
            f'"id":{q_id},'
            f'"level":"{level}",'
            f'"question":"{question}",'
            '"analysis":"...",'
            '"key_points":["..."],'
            '"sources":["..."],'
            '"difficulty":"初级/中级/高级"'
            "}"
        )
        response = self.api_client.call(
            system_prompt=f"你是 {self.topic} 资深专家，请输出结构化分析。",
            user_prompt=analysis_prompt,
            enable_json_mode=True,
            max_tokens=PHASE3_ANALYSIS_MAX_TOKENS,
            use_deepseek_thinking=True,
            reasoning_effort=self.reasoning_effort,
            progress_context={
                "stage_name": "逐题深度分析",
                "stage_index": 3,
                "stage_total": 3,
                "substep_name": "生成答案",
                "item_name": question[:30],
                "item_index": item_index,
                "item_total": item_total,
                "request_group": "phase3_analysis",
            },
        )
        payload = self.api_client.safe_parse_json(
            self.api_client.extract_message_content(response, default="{}")
        )
        jsonschema = importlib.import_module("jsonschema")
        jsonschema.validate(instance=payload, schema=PHASE3_SCHEMA)
        return payload

    @staticmethod
    def _print_progress_report(
        completed: int, max_questions: int, stats: Any, start_time: float
    ) -> None:
        import time

        elapsed = time.time() - start_time
        print("\n" + "-" * 60)
        print(f"⏱️  已运行时间：{elapsed:.1f} 秒")
        print(f"✅ 已完成问题：{completed} / {max_questions}")
        print(f"📡 API 调用次数：{stats.total_api_calls}")
        print(f"📝 累计输入 Token：{stats.total_input_tokens:,}")
        print(f"📝 累计输出 Token：{stats.total_output_tokens:,}")
        print("-" * 60 + "\n")

    def run(
        self,
        questions: List[Dict[str, Any]],
        research_summary: str,
        resume_from: int,
        max_questions: int,
        writer: Any,
        markdown_writer: Any,
        start_time: float,
    ) -> None:
        """执行阶段三分析。"""
        questions = questions[:max_questions]
        start_index = max(0, resume_from)
        questions = questions[start_index:]

        pbar = tqdm(total=len(questions), desc="深度分析进度", unit="题", ncols=80)

        total_items = start_index + len(questions)

        for idx, q_item in enumerate(questions, start=start_index + 1):
            q_id = int(q_item["id"])
            level = str(q_item["level"])
            question = str(q_item["question"])

            pbar.set_description(
                f"[{idx}/{start_index + len(questions)}] {question[:30]}..."
            )

            error_text = ""
            try:
                search_result = self._search_context(
                    question=question,
                    research_summary=research_summary,
                    item_index=idx,
                    item_total=total_items,
                )
            except Exception as exc:
                search_result = f"搜索阶段失败：{exc}"
                error_text = str(exc)

            try:
                payload = self._analyze(
                    q_id=q_id,
                    level=level,
                    question=question,
                    research_summary=research_summary,
                    search_result=search_result,
                    item_index=idx,
                    item_total=total_items,
                )
                model = Phase3Response(
                    id=int(payload.get("id", q_id)),
                    level=str(payload.get("level", level)),
                    question=str(payload.get("question", question)),
                    analysis=str(payload.get("analysis", "分析内容为空")),
                    key_points=list(payload.get("key_points", [])),
                    sources=list(payload.get("sources", [])),
                    difficulty=str(payload.get("difficulty", level)),
                )
                record = JsonlRecordFactory.phase3(model)
            except Exception as exc:
                error_text = str(exc)
                fallback = Phase3Response(
                    id=q_id,
                    level=level,
                    question=question,
                    analysis=f"分析生成失败（错误：{exc}）。搜索摘要：{search_result[:500]}",
                    key_points=["分析失败，请人工补充"],
                    sources=[],
                    difficulty=level,
                )
                record = JsonlRecordFactory.phase3(fallback, error=error_text)

            record["timestamp"] = datetime.now().isoformat()
            flush_now = idx % FLUSH_INTERVAL == 0
            writer.append(record, flush=flush_now)
            markdown_writer.write_analysis(record)

            pbar.update(1)
            if idx % COST_REPORT_INTERVAL == 0:
                self._print_progress_report(
                    completed=idx,
                    max_questions=max_questions,
                    stats=self.stats,
                    start_time=start_time,
                )

        pbar.close()
