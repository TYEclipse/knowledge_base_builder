from __future__ import annotations

import pytest
from pydantic import ValidationError

from models import JsonlRecordFactory, Phase3Response, QuestionItem


def test_question_item_validation():
    item = QuestionItem(id=1, level="beginner", question="Q")
    assert item.id == 1

    with pytest.raises(ValidationError):
        QuestionItem(id=0, level="beginner", question="Q")


def test_jsonl_record_factory_phase_records():
    p1 = JsonlRecordFactory.phase1(topic="AI", summary="S", error="E")
    assert p1["phase"] == 1
    assert p1["error"] == "E"

    p2 = JsonlRecordFactory.phase2(level="beginner", topic="AI", questions=["q1"])
    assert p2["phase"] == 2
    assert p2["questions"] == ["q1"]

    m = Phase3Response(
        id=1,
        level="beginner",
        question="Q",
        analysis="A",
        key_points=["k"],
        sources=["s"],
        difficulty="beginner",
    )
    p3 = JsonlRecordFactory.phase3(m, error="")
    assert p3["phase"] == 3
    assert p3["analysis"] == "A"
