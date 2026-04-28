from __future__ import annotations

import json
import time
from types import SimpleNamespace

import answer_analyzer as aa
from answer_analyzer import AnswerAnalyzer


class DummyApiClient:
    def __init__(self, fail_analyze: bool = False):
        self.fail_analyze = fail_analyze

    def call(self, **kwargs):
        if kwargs.get("enable_web_search"):
            return SimpleNamespace(kind="search")
        return SimpleNamespace(kind="analyze")

    def extract_message_content(self, response, default=""):
        if getattr(response, "kind", "") == "search":
            return "search-summary"
        if self.fail_analyze:
            return "{bad-json"
        payload = {
            "id": 1,
            "level": "beginner",
            "question": "Q1",
            "analysis": "A1",
            "key_points": ["k1"],
            "sources": ["s1"],
            "difficulty": "beginner",
        }
        return json.dumps(payload, ensure_ascii=False)

    def safe_parse_json(self, text: str):
        return json.loads(text)


class DummyApiClientSearchFail(DummyApiClient):
    def call(self, **kwargs):
        if kwargs.get("enable_web_search"):
            raise RuntimeError("search boom")
        return SimpleNamespace(kind="analyze")


class MemoryWriter:
    def __init__(self):
        self.records = []

    def append(self, record, flush=False):
        self.records.append((record, flush))


class MemoryMarkdownWriter:
    def __init__(self):
        self.records = []

    def write_analysis(self, record):
        self.records.append(record)


def test_answer_analyzer_run_success_path():
    stats = SimpleNamespace(total_api_calls=0, total_input_tokens=0, total_output_tokens=0)
    analyzer = AnswerAnalyzer(topic="AI", api_client=DummyApiClient(False), logger=SimpleNamespace(), stats=stats)

    writer = MemoryWriter()
    md_writer = MemoryMarkdownWriter()

    analyzer.run(
        questions=[{"id": 1, "level": "beginner", "question": "Q1"}],
        research_summary="summary",
        resume_from=0,
        max_questions=1,
        writer=writer,
        markdown_writer=md_writer,
        start_time=time.time(),
    )

    assert len(writer.records) == 1
    assert writer.records[0][0]["type"] == "analysis"
    assert len(md_writer.records) == 1


def test_answer_analyzer_run_fallback_on_parse_error():
    stats = SimpleNamespace(total_api_calls=0, total_input_tokens=0, total_output_tokens=0)
    analyzer = AnswerAnalyzer(topic="AI", api_client=DummyApiClient(True), logger=SimpleNamespace(), stats=stats)

    writer = MemoryWriter()
    md_writer = MemoryMarkdownWriter()

    analyzer.run(
        questions=[{"id": 1, "level": "beginner", "question": "Q1"}],
        research_summary="summary",
        resume_from=0,
        max_questions=1,
        writer=writer,
        markdown_writer=md_writer,
        start_time=time.time(),
    )

    record = writer.records[0][0]
    assert "error" in record
    assert "分析生成失败" in record["analysis"]


def test_truncate_and_progress_report_branch(capsys):
    stats = SimpleNamespace(total_api_calls=1, total_input_tokens=2, total_output_tokens=3)
    analyzer = AnswerAnalyzer(topic="AI", api_client=DummyApiClient(False), logger=SimpleNamespace(), stats=stats)

    assert analyzer._truncate("abc", 10) == "abc"
    assert "截断" in analyzer._truncate("x" * 50, 5)

    AnswerAnalyzer._print_progress_report(1, 2, stats, time.time() - 1)
    out = capsys.readouterr().out
    assert "已完成问题" in out


def test_answer_analyzer_search_exception_branch_and_cost_report(monkeypatch):
    stats = SimpleNamespace(total_api_calls=0, total_input_tokens=0, total_output_tokens=0)
    analyzer = AnswerAnalyzer(topic="AI", api_client=DummyApiClientSearchFail(True), logger=SimpleNamespace(), stats=stats)

    writer = MemoryWriter()
    md_writer = MemoryMarkdownWriter()
    calls = []

    monkeypatch.setattr(aa, "COST_REPORT_INTERVAL", 1)
    monkeypatch.setattr(AnswerAnalyzer, "_print_progress_report", staticmethod(lambda *args, **kwargs: calls.append(True)))

    analyzer.run(
        questions=[{"id": 1, "level": "beginner", "question": "Q1"}],
        research_summary="summary",
        resume_from=0,
        max_questions=1,
        writer=writer,
        markdown_writer=md_writer,
        start_time=time.time(),
    )

    assert len(writer.records) == 1
    assert "搜索阶段失败" in writer.records[0][0]["analysis"]
    assert calls
