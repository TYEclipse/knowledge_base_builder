from __future__ import annotations

import json
from pathlib import Path

from storage import AtomicJsonlWriter, MarkdownWriter


def test_atomic_writer_append_flush_and_restore(tmp_path: Path):
    path = tmp_path / "kb.jsonl"
    writer = AtomicJsonlWriter(path=path)

    writer.append({"phase": 1, "type": "research", "summary": "ok"})
    writer.append(
        {
            "phase": 2,
            "type": "question_list",
            "level": "beginner",
            "questions": ["q1", "q2"],
        },
        flush=True,
    )

    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    restored = writer.restore_phase2_questions()
    assert restored is not None
    assert len(restored) == 2
    assert restored[0]["id"] == 1
    assert restored[1]["question"] == "q2"


def test_markdown_writer_outputs_files(tmp_path: Path):
    out = tmp_path / "a.jsonl"
    md_dir = tmp_path / "kb_markdown"
    writer = MarkdownWriter(topic="AI", audience="beginner", output_path=out, markdown_dir=md_dir)

    writer.write_research("这是摘要")
    writer.write_question_list("beginner", "初学者", ["什么是Agent？"]) 
    writer.write_analysis(
        {
            "id": 1,
            "question": "什么是Agent？",
            "level": "beginner",
            "difficulty": "beginner",
            "analysis": "分析内容",
            "key_points": ["要点1"],
            "sources": ["来源1"],
        }
    )

    assert (md_dir / "01_research_summary.md").exists()
    assert (md_dir / "02_beginner_questions.md").exists()
    answer_files = list((md_dir / "answers").glob("0001_*.md"))
    assert len(answer_files) == 1
    content = answer_files[0].read_text(encoding="utf-8")
    assert "### 分析" in content
    assert "要点1" in content


def test_atomic_writer_load_existing_and_empty_flush(tmp_path: Path):
    path = tmp_path / "old.jsonl"
    path.write_text('{"phase":2,"type":"question_list","level":"beginner","questions":["q"]}\n', encoding="utf-8")

    writer = AtomicJsonlWriter(path=path)
    assert len(writer.lines) == 1

    empty_path = tmp_path / "empty.jsonl"
    empty_writer = AtomicJsonlWriter(path=empty_path)
    empty_writer.flush()
    assert empty_path.exists()
    assert empty_path.read_text(encoding="utf-8") == ""


def test_restore_questions_handles_invalid_json_and_none(tmp_path: Path):
    path = tmp_path / "x.jsonl"
    path.write_text('not-json\n{"phase":1}\n', encoding="utf-8")
    writer = AtomicJsonlWriter(path=path)
    assert writer.restore_phase2_questions() is None


def test_markdown_write_analysis_with_empty_points_and_sources(tmp_path: Path):
    out = tmp_path / "a.jsonl"
    md_dir = tmp_path / "kb_markdown"
    writer = MarkdownWriter(topic="AI", audience="beginner", output_path=out, markdown_dir=md_dir)

    writer.write_analysis(
        {
            "id": 2,
            "question": "Q2",
            "level": "beginner",
            "difficulty": "beginner",
            "analysis": "分析内容",
            "key_points": [],
            "sources": [],
        }
    )

    answer_file = list((md_dir / "answers").glob("0002_*.md"))[0]
    content = answer_file.read_text(encoding="utf-8")
    assert "- 无" in content
