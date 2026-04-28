"""文件写入与恢复工具模块。"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AtomicJsonlWriter:
    """JSONL 原子写入器（先写临时文件再替换）。"""

    path: Path
    lines: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.lines = self.path.read_text(encoding="utf-8").splitlines()

    def append(self, record: Dict[str, Any], flush: bool = False) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        self.lines.append(line)
        if flush:
            self.flush()

    def flush(self) -> None:
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = "\n".join(self.lines)
        if payload:
            payload += "\n"

        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, self.path)

    def restore_phase2_questions(self) -> Optional[List[Dict[str, Any]]]:
        questions: List[Dict[str, Any]] = []
        for line in self.lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("phase") == 2 and record.get("type") == "question_list":
                level = str(record.get("level", "unknown"))
                for q in record.get("questions", []):
                    questions.append(
                        {
                            "id": len(questions) + 1,
                            "level": level,
                            "question": str(q),
                        }
                    )
        return questions or None


@dataclass
class MarkdownWriter:
    """Markdown 文件输出器。"""

    topic: str
    audience: str
    output_path: Path
    markdown_dir: Path

    def __post_init__(self) -> None:
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        self.answers_dir = self.markdown_dir / "answers"
        self.answers_dir.mkdir(parents=True, exist_ok=True)

        self.summary_path = self.markdown_dir / "01_research_summary.md"
        self.question_paths = {
            "beginner": self.markdown_dir / "02_beginner_questions.md",
            "intermediate": self.markdown_dir / "03_intermediate_questions.md",
            "advanced": self.markdown_dir / "04_advanced_questions.md",
        }

    @staticmethod
    def _sanitize_filename(text: str, max_length: int = 60) -> str:
        cleaned = re.sub(r'[<>:"/\\|?*\r\n\t]+', "_", text).strip()
        cleaned = re.sub(r"\s+", "_", cleaned)
        cleaned = re.sub(r"_+", "_", cleaned).strip("._")
        return cleaned[:max_length] if cleaned else "question"

    def _meta(self) -> str:
        return (
            f"# 知识库：{self.topic}\n\n"
            f"- 受众：`{self.audience}`\n"
            f"- JSONL 输出：`{self.output_path.resolve()}`\n\n"
        )

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

    def write_research(self, summary: str) -> None:
        content = f"{self._meta()}## 1. 主题调研摘要\n\n{summary}\n"
        self._write_text(self.summary_path, content)

    def write_question_list(self, level_en: str, level_cn: str, questions: List[str]) -> None:
        lines = [f"{self._meta()}## 2.{level_en} 问题清单（{level_cn}）\n\n"]
        for i, q in enumerate(questions, start=1):
            lines.append(f"{i}. {q}\n")
        self._write_text(self.question_paths.get(level_en, self.markdown_dir / f"questions_{level_en}.md"), "".join(lines))

    def write_analysis(self, record: Dict[str, Any]) -> None:
        q_id = int(record.get("id", 0))
        slug = self._sanitize_filename(str(record.get("question", "question")))
        path = self.answers_dir / f"{q_id:04d}_{slug}.md"

        key_points = record.get("key_points", [])
        sources = record.get("sources", [])

        lines = [
            f"{self._meta()}## 3.{q_id} {record.get('question', '')}\n\n",
            f"- 级别：`{record.get('level', '')}`\n",
            f"- 难度：`{record.get('difficulty', '')}`\n\n",
            "### 分析\n\n",
            f"{record.get('analysis', '')}\n\n",
            "### 要点\n\n",
        ]

        if isinstance(key_points, list) and key_points:
            for item in key_points:
                lines.append(f"- {item}\n")
        else:
            lines.append("- 无\n")

        lines.append("\n### 来源\n\n")
        if isinstance(sources, list) and sources:
            for item in sources:
                lines.append(f"- {item}\n")
        else:
            lines.append("- 无\n")

        self._write_text(path, "".join(lines))
