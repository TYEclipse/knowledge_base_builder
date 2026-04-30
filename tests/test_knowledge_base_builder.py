from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import knowledge_base_builder as kbb


class DummyApiClient:
    def __init__(self, **kwargs):
        self.should_auth_error = False

    def call(self, **kwargs):
        return SimpleNamespace()

    def extract_message_content(self, response):
        return "x" * 300

    def is_auth_error(self, exc: Exception) -> bool:
        return self.should_auth_error

    def close(self):
        return None


class DummyWriter:
    def __init__(self, path):
        self.path = path
        self.records = []

    def append(self, record, flush=False):
        self.records.append(record)

    def flush(self):
        return None

    def restore_phase2_questions(self):
        return [{"id": 1, "level": "beginner", "question": "Q1"}]


class DummyMarkdownWriter:
    def __init__(self, **kwargs):
        self.calls = []

    def write_research(self, summary):
        self.calls.append(("research", summary))

    def write_question_list(self, level_en, level_cn, questions):
        self.calls.append(("q", level_en, level_cn, questions))


class DummyQuestionGenerator:
    def __init__(self, **kwargs):
        pass

    def generate(self, summary):
        return (
            [{"id": 1, "level": "beginner", "question": "Q1"}],
            [{"level": "beginner", "questions": ["Q1"]}],
        )


class DummyAnswerAnalyzer:
    def __init__(self, **kwargs):
        self.called = False

    def run(self, **kwargs):
        self.called = True


class DummyBuilderForMain:
    called = False

    def __init__(self, settings):
        self.settings = settings

    def run(self):
        DummyBuilderForMain.called = True


def make_settings(tmp_path: Path, api_key: str = "k"):
    return SimpleNamespace(
        topic="AI",
        audience="beginner",
        output_path=str(tmp_path / "kb.jsonl"),
        markdown_output=None,
        resume=0,
        max_questions=1,
        stream=False,
        verbose=False,
        model_name="kimi-k2.6",
        base_url="https://api.moonshot.cn/v1",
        api_key=api_key,
        deepseek_model_name="deepseek-v4-pro",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_api_key="ds-test",
        deepseek_reasoning_effort="high",
    )


def test_builder_init_requires_api_key(tmp_path: Path):
    with pytest.raises(RuntimeError):
        kbb.KnowledgeBaseBuilder(make_settings(tmp_path, api_key=""))


def test_builder_phase_methods_and_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(kbb, "KimiApiClient", DummyApiClient)
    monkeypatch.setattr(kbb, "AtomicJsonlWriter", DummyWriter)
    monkeypatch.setattr(kbb, "MarkdownWriter", DummyMarkdownWriter)
    monkeypatch.setattr(kbb, "QuestionGenerator", DummyQuestionGenerator)
    monkeypatch.setattr(kbb, "AnswerAnalyzer", DummyAnswerAnalyzer)

    builder = kbb.KnowledgeBaseBuilder(make_settings(tmp_path))

    phase1 = builder.phase1_research()
    assert phase1["phase"] == 1

    questions = builder.phase2_questions("summary")
    assert len(questions) == 1

    builder.run()


def test_validate_research_summary_short_raises():
    with pytest.raises(kbb.ResearchQualityError):
        kbb.KnowledgeBaseBuilder._validate_research_summary("short")


def test_phase1_auth_error_raises_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class AuthFailApi(DummyApiClient):
        def call(self, **kwargs):
            raise RuntimeError("auth")

        def is_auth_error(self, exc: Exception) -> bool:
            return True

    monkeypatch.setattr(kbb, "KimiApiClient", AuthFailApi)
    monkeypatch.setattr(kbb, "AtomicJsonlWriter", DummyWriter)
    monkeypatch.setattr(kbb, "MarkdownWriter", DummyMarkdownWriter)
    monkeypatch.setattr(kbb, "QuestionGenerator", DummyQuestionGenerator)
    monkeypatch.setattr(kbb, "AnswerAnalyzer", DummyAnswerAnalyzer)

    builder = kbb.KnowledgeBaseBuilder(make_settings(tmp_path))
    with pytest.raises(RuntimeError):
        builder.phase1_research()


def test_phase1_non_auth_fallback_and_resume_restore_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class NonAuthFailApi(DummyApiClient):
        def call(self, **kwargs):
            raise RuntimeError("network")

        def is_auth_error(self, exc: Exception) -> bool:
            return False

    monkeypatch.setattr(kbb, "KimiApiClient", NonAuthFailApi)
    monkeypatch.setattr(kbb, "AtomicJsonlWriter", DummyWriter)
    monkeypatch.setattr(kbb, "MarkdownWriter", DummyMarkdownWriter)
    monkeypatch.setattr(kbb, "QuestionGenerator", DummyQuestionGenerator)
    monkeypatch.setattr(kbb, "AnswerAnalyzer", DummyAnswerAnalyzer)

    settings = make_settings(tmp_path)
    settings.resume = 1
    builder = kbb.KnowledgeBaseBuilder(settings)

    rec = builder.phase1_research()
    assert "error" in rec

    monkeypatch.setattr(
        builder,
        "phase2_questions",
        cast(
            Any,
            lambda summary: (_ for _ in ()).throw(
                AssertionError("should not call phase2_questions when restored")
            ),
        ),
    )
    builder.run()


def test_build_arg_parser_and_main_failure_path(monkeypatch: pytest.MonkeyPatch):
    parser = kbb.build_arg_parser()
    args = parser.parse_args(["--topic", "AI"])
    assert args.topic == "AI"

    class DummyParser:
        def parse_args(self):
            return SimpleNamespace()

    monkeypatch.setattr(kbb, "build_arg_parser", lambda: DummyParser())
    monkeypatch.setattr(
        kbb,
        "build_settings_from_args",
        lambda args, project_root: (_ for _ in ()).throw(ValueError("bad")),
    )

    with pytest.raises(SystemExit):
        kbb.main()


def test_truncate_and_phase1_resume_skip_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(kbb, "KimiApiClient", DummyApiClient)
    monkeypatch.setattr(kbb, "AtomicJsonlWriter", DummyWriter)
    monkeypatch.setattr(kbb, "MarkdownWriter", DummyMarkdownWriter)
    monkeypatch.setattr(kbb, "QuestionGenerator", DummyQuestionGenerator)
    monkeypatch.setattr(kbb, "AnswerAnalyzer", DummyAnswerAnalyzer)

    settings = make_settings(tmp_path)
    settings.resume = 1
    b = kbb.KnowledgeBaseBuilder(settings)

    assert b._truncate("abc", 10) == "abc"
    assert "截断" in b._truncate("x" * 50, 5)

    rec = b.phase1_research()
    assert rec["phase"] == 1
    # resume>0 时不写 research markdown
    assert cast(Any, b.markdown_writer).calls == []


def test_phase2_resume_nonzero_skips_markdown_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(kbb, "KimiApiClient", DummyApiClient)
    monkeypatch.setattr(kbb, "AtomicJsonlWriter", DummyWriter)
    monkeypatch.setattr(kbb, "MarkdownWriter", DummyMarkdownWriter)
    monkeypatch.setattr(kbb, "QuestionGenerator", DummyQuestionGenerator)
    monkeypatch.setattr(kbb, "AnswerAnalyzer", DummyAnswerAnalyzer)

    settings = make_settings(tmp_path)
    settings.resume = 2
    b = kbb.KnowledgeBaseBuilder(settings)
    qs = b.phase2_questions("summary")
    assert len(qs) == 1
    assert cast(Any, b.markdown_writer).calls == []


def test_run_resume_restore_none_goes_to_phase2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(kbb, "KimiApiClient", DummyApiClient)
    monkeypatch.setattr(kbb, "AtomicJsonlWriter", DummyWriter)
    monkeypatch.setattr(kbb, "MarkdownWriter", DummyMarkdownWriter)
    monkeypatch.setattr(kbb, "QuestionGenerator", DummyQuestionGenerator)
    monkeypatch.setattr(kbb, "AnswerAnalyzer", DummyAnswerAnalyzer)

    settings = make_settings(tmp_path)
    settings.resume = 1
    b = kbb.KnowledgeBaseBuilder(settings)

    b.writer.restore_phase2_questions = lambda: None  # type: ignore[method-assign]
    marker = {"called": False}

    original_phase2 = b.phase2_questions

    def wrapped_phase2(summary):
        marker["called"] = True
        return original_phase2(summary)

    b.phase2_questions = wrapped_phase2  # type: ignore[method-assign]
    b.run()
    assert marker["called"] is True


def test_main_success_path(monkeypatch: pytest.MonkeyPatch):
    class DummyParser:
        def parse_args(self):
            return SimpleNamespace()

    monkeypatch.setattr(kbb, "build_arg_parser", lambda: DummyParser())
    monkeypatch.setattr(
        kbb, "build_settings_from_args", lambda args, project_root: SimpleNamespace()
    )
    monkeypatch.setattr(kbb, "KnowledgeBaseBuilder", DummyBuilderForMain)

    DummyBuilderForMain.called = False
    kbb.main()
    assert DummyBuilderForMain.called is True


def test_run_handles_keyboard_interrupt_gracefully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class KIAnswerAnalyzer(DummyAnswerAnalyzer):
        def run(self, **kwargs):
            raise KeyboardInterrupt()

    monkeypatch.setattr(kbb, "KimiApiClient", DummyApiClient)
    monkeypatch.setattr(kbb, "AtomicJsonlWriter", DummyWriter)
    monkeypatch.setattr(kbb, "MarkdownWriter", DummyMarkdownWriter)
    monkeypatch.setattr(kbb, "QuestionGenerator", DummyQuestionGenerator)
    monkeypatch.setattr(kbb, "AnswerAnalyzer", KIAnswerAnalyzer)

    b = kbb.KnowledgeBaseBuilder(make_settings(tmp_path))
    # 不应抛出 KeyboardInterrupt 到测试层
    b.run()


def test_phase2_incremental_generation_writes_markdown_per_level_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    events = []

    class TracingWriter(DummyWriter):
        def append(self, record, flush=False):
            super().append(record, flush=flush)
            events.append(("append", record.get("level")))

    class TracingMarkdownWriter(DummyMarkdownWriter):
        def write_question_list(self, level_en, level_cn, questions):
            super().write_question_list(level_en, level_cn, questions)
            events.append(("markdown", level_en))

    class IncrementalQuestionGenerator:
        def __init__(self, **kwargs):
            pass

        def generate_incrementally(self, summary):
            yield (
                [{"id": 1, "level": "beginner", "question": "Q1"}],
                {"level": "beginner", "questions": ["Q1"]},
            )
            assert events == [
                ("append", "beginner"),
                ("markdown", "beginner"),
            ]
            yield (
                [{"id": 2, "level": "intermediate", "question": "Q2"}],
                {"level": "intermediate", "questions": ["Q2"]},
            )

    monkeypatch.setattr(kbb, "KimiApiClient", DummyApiClient)
    monkeypatch.setattr(kbb, "AtomicJsonlWriter", TracingWriter)
    monkeypatch.setattr(kbb, "MarkdownWriter", TracingMarkdownWriter)
    monkeypatch.setattr(kbb, "QuestionGenerator", IncrementalQuestionGenerator)
    monkeypatch.setattr(kbb, "AnswerAnalyzer", DummyAnswerAnalyzer)

    b = kbb.KnowledgeBaseBuilder(make_settings(tmp_path))
    questions = b.phase2_questions("summary")

    assert [q["id"] for q in questions] == [1, 2]
    assert events == [
        ("append", "beginner"),
        ("markdown", "beginner"),
        ("append", "intermediate"),
        ("markdown", "intermediate"),
    ]


def test_add_boolean_optional_argument_fallback_without_boolean_optional_action(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delattr(kbb.argparse, "BooleanOptionalAction", raising=False)

    parser = argparse.ArgumentParser()
    kbb._add_boolean_optional_argument(
        parser,
        "--stream",
        default=True,
        help_text="是否启用流式输出",
    )

    assert parser.parse_args([]).stream is True
    assert parser.parse_args(["--no-stream"]).stream is False
    assert parser.parse_args(["--stream"]).stream is True


def test_phase2_incremental_generation_resume_nonzero_skips_markdown_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class IncrementalQuestionGenerator:
        def __init__(self, **kwargs):
            pass

        def generate_incrementally(self, summary):
            yield (
                [{"id": 1, "level": "beginner", "question": "Q1"}],
                {"level": "beginner", "questions": ["Q1"]},
            )

    monkeypatch.setattr(kbb, "KimiApiClient", DummyApiClient)
    monkeypatch.setattr(kbb, "AtomicJsonlWriter", DummyWriter)
    monkeypatch.setattr(kbb, "MarkdownWriter", DummyMarkdownWriter)
    monkeypatch.setattr(kbb, "QuestionGenerator", IncrementalQuestionGenerator)
    monkeypatch.setattr(kbb, "AnswerAnalyzer", DummyAnswerAnalyzer)

    settings = make_settings(tmp_path)
    settings.resume = 1
    b = kbb.KnowledgeBaseBuilder(settings)

    qs = b.phase2_questions("summary")
    assert len(qs) == 1
    assert cast(Any, b.markdown_writer).calls == []
