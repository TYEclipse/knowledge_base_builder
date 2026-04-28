"""数据模型与 JSON Schema 定义。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field


@dataclass
class RuntimeStats:
    """运行期统计指标。"""

    total_api_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


class QuestionItem(BaseModel):
    """单个问题项。"""

    id: int = Field(ge=1)
    level: str
    question: str


class Phase2Response(BaseModel):
    """阶段二问题清单响应。"""

    level: str
    topic: str
    questions: List[str]


class Phase3Response(BaseModel):
    """阶段三结构化分析响应。"""

    id: int = Field(ge=1)
    level: str
    question: str
    analysis: str
    key_points: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    difficulty: str


class JsonlRecordFactory:
    """统一生成 JSONL 记录。"""

    @staticmethod
    def phase1(topic: str, summary: str, error: str = "") -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "phase": 1,
            "type": "research",
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
        }
        if error:
            payload["error"] = error
        return payload

    @staticmethod
    def phase2(level: str, topic: str, questions: List[str]) -> Dict[str, Any]:
        return {
            "phase": 2,
            "type": "question_list",
            "level": level,
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "questions": questions,
        }

    @staticmethod
    def phase3(data: Phase3Response, error: str = "") -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "phase": 3,
            "type": "analysis",
            "id": data.id,
            "level": data.level,
            "question": data.question,
            "analysis": data.analysis,
            "key_points": data.key_points,
            "sources": data.sources,
            "difficulty": data.difficulty,
            "timestamp": datetime.now().isoformat(),
        }
        if error:
            payload["error"] = error
        return payload


PHASE2_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["level", "topic", "questions"],
    "properties": {
        "level": {"type": "string"},
        "topic": {"type": "string"},
        "questions": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
    },
    "additionalProperties": True,
}

PHASE3_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "id",
        "level",
        "question",
        "analysis",
        "key_points",
        "sources",
        "difficulty",
    ],
    "properties": {
        "id": {"type": "integer", "minimum": 1},
        "level": {"type": "string"},
        "question": {"type": "string"},
        "analysis": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "sources": {"type": "array", "items": {"type": "string"}},
        "difficulty": {"type": "string"},
    },
    "additionalProperties": True,
}
