from __future__ import annotations

import io
from types import SimpleNamespace
from pathlib import Path

import pytest

import config


def test_sanitize_topic_and_numeric_validation():
    assert config.sanitize_topic("  AI Agent  ") == "AI Agent"

    with pytest.raises(ValueError):
        config.sanitize_topic("   ")
    with pytest.raises(ValueError):
        config.sanitize_topic("a" * 201)
    with pytest.raises(ValueError):
        config.sanitize_topic("abc\x00")

    with pytest.raises(ValueError):
        config.validate_numeric_args(-1, 1)
    with pytest.raises(ValueError):
        config.validate_numeric_args(0, 0)


def test_path_helpers_and_secure_output(tmp_path: Path):
    assert config.derive_markdown_output_dir("out.jsonl").endswith("out_markdown")
    assert config.normalize_markdown_output_dir(None, "x.jsonl").endswith("x_markdown")
    assert config.normalize_markdown_output_dir("abc.md", "x.jsonl").endswith(
        "abc_files"
    )

    safe = config.secure_output_path("result.jsonl", tmp_path)
    assert str(safe).endswith("result.jsonl")

    outside = tmp_path.parent / "outside.jsonl"
    with pytest.raises(ValueError):
        config.secure_output_path(str(outside), tmp_path)


def test_build_settings_from_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MOONSHOT_API_KEY", "k-test")
    monkeypatch.setenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("DEEPSEEK_MODEL_NAME", "deepseek-v4-pro")
    monkeypatch.setenv("DEEPSEEK_REASONING_EFFORT", "medium")

    args = SimpleNamespace(
        topic="LLM",
        audience="beginner",
        output="out.jsonl",
        markdown_output=None,
        resume=0,
        max_questions=10,
        stream=True,
        verbose=False,
    )

    settings = config.build_settings_from_args(args, tmp_path)
    assert settings.topic == "LLM"
    assert settings.api_key == "k-test"
    assert settings.max_questions == 10
    assert settings.output_path.endswith("out.jsonl")
    assert settings.deepseek_api_key == "ds-test"
    assert settings.deepseek_base_url == "https://api.deepseek.com"
    assert settings.deepseek_model_name == "deepseek-v4-pro"
    assert settings.deepseek_reasoning_effort == "high"


def test_sensitive_data_filter_masks_secret_and_env_key():
    f = config.SensitiveDataFilter(secret="my-secret")
    rec = SimpleNamespace(
        msg="token=my-secret MOONSHOT_API_KEY=abcd",
        args=(),
        getMessage=lambda: "token=my-secret MOONSHOT_API_KEY=abcd",
    )

    assert f.filter(rec) is True
    assert "my-secret" not in rec.msg
    assert "MOONSHOT_API_KEY=****" in rec.msg


def test_build_settings_invalid_audience_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(
        topic="LLM",
        audience="expert",
        output="out.jsonl",
        markdown_output=None,
        resume=0,
        max_questions=10,
        stream=True,
        verbose=False,
    )

    with pytest.raises(ValueError):
        config.build_settings_from_args(args, tmp_path)


def test_load_environment_when_cwd_differs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    calls = []

    def fake_load_dotenv(path, override=True):
        calls.append((str(path), override))

    monkeypatch.setattr(config, "load_dotenv", fake_load_dotenv)
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)

    config.load_environment(tmp_path)
    assert len(calls) == 2


def test_path_helpers_remaining_branches():
    assert config.derive_markdown_output_dir("outfile").endswith("outfile_markdown")
    assert config.normalize_markdown_output_dir("some_dir", "x.jsonl") == "some_dir"


def test_inplace_progress_handler_overwrites_same_line_then_recovers_newline():
    handler = config.InPlaceProgressHandler()
    stream = io.StringIO()
    handler.stream = stream
    handler.setFormatter(config.logging.Formatter("%(message)s"))

    overwrite_record = config.logging.LogRecord(
        name="kb",
        level=config.logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="progress 1",
        args=(),
        exc_info=None,
    )
    overwrite_record.overwrite = True

    normal_record = config.logging.LogRecord(
        name="kb",
        level=config.logging.INFO,
        pathname=__file__,
        lineno=2,
        msg="done",
        args=(),
        exc_info=None,
    )

    handler.emit(overwrite_record)
    handler.emit(normal_record)

    output = stream.getvalue()
    assert "\rprogress 1" in output
    assert "\ndone\n" in output
