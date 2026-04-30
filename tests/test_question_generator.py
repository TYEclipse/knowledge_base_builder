from __future__ import annotations

import json
from types import SimpleNamespace

from question_generator import QuestionGenerator


class DummyApiClientSuccess:
    def call(self, **kwargs):
        return SimpleNamespace(content="ok")

    def extract_message_content(self, response, default=""):
        payload = {
            "level": "beginner",
            "topic": "AI",
            "questions": ["q1", "q2"],
        }
        return json.dumps(payload, ensure_ascii=False)

    def safe_parse_json(self, text: str):
        return json.loads(text)


class DummyApiClientFail:
    def call(self, **kwargs):
        raise RuntimeError("boom")

    def extract_message_content(self, response, default=""):
        return "{}"

    def safe_parse_json(self, text: str):
        return {}


class DummyLogger:
    def __init__(self):
        self.messages = []

    def warning(self, msg, *args):
        self.messages.append(msg % args)


def test_generate_questions_success_padding():
    logger = DummyLogger()
    gen = QuestionGenerator(
        topic="AI", api_client=DummyApiClientSuccess(), logger=logger
    )

    all_questions, phase2_records = gen.generate("summary")

    assert len(phase2_records) == 3
    assert len(all_questions) == 300
    assert all_questions[0]["id"] == 1
    assert all_questions[-1]["id"] == 300


def test_generate_questions_fallback_on_exception():
    logger = DummyLogger()
    gen = QuestionGenerator(topic="AI", api_client=DummyApiClientFail(), logger=logger)

    all_questions, phase2_records = gen.generate("summary")

    assert len(phase2_records) == 3
    assert len(all_questions) == 300
    assert "生成失败" in all_questions[0]["question"]
    assert len(logger.messages) == 3


def test_generate_incrementally_emits_level_batches_with_continuous_ids():
    class SequencedApiClient:
        def __init__(self):
            self.calls = 0

        def call(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(content="ok")

        def extract_message_content(self, response, default=""):
            payloads = [
                {"level": "beginner", "topic": "AI", "questions": ["b1", "b2"]},
                {
                    "level": "intermediate",
                    "topic": "AI",
                    "questions": ["i1", "i2"],
                },
                {"level": "advanced", "topic": "AI", "questions": ["a1", "a2"]},
            ]
            return json.dumps(payloads[self.calls - 1], ensure_ascii=False)

        def safe_parse_json(self, text: str):
            return json.loads(text)

    logger = DummyLogger()
    gen = QuestionGenerator(topic="AI", api_client=SequencedApiClient(), logger=logger)

    batches = list(gen.generate_incrementally("summary"))

    assert len(batches) == 3
    assert batches[0][0][0]["id"] == 1
    assert batches[1][0][0]["id"] == 101
    assert batches[2][0][0]["id"] == 201
    assert batches[0][1]["level"] == "beginner"
    assert batches[1][1]["level"] == "intermediate"
    assert batches[2][1]["level"] == "advanced"
