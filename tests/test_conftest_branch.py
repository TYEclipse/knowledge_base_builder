from __future__ import annotations

import sys
from pathlib import Path


def _exec_conftest_with_sys_path(sys_path: list[str]) -> list[str]:
    """在指定 sys.path 下执行 conftest 并返回执行后的 path。"""
    conftest_path = Path(__file__).resolve().parent / "conftest.py"
    code = conftest_path.read_text(encoding="utf-8")

    original = list(sys.path)
    try:
        sys.path[:] = list(sys_path)
        ns = {
            "__file__": str(conftest_path),
            "__name__": "__conftest_branch_test__",
        }
        exec(compile(code, str(conftest_path), "exec"), ns, ns)
        return list(sys.path)
    finally:
        sys.path[:] = original


def test_conftest_inserts_project_root_when_missing():
    root = str(Path(__file__).resolve().parents[1])
    after = _exec_conftest_with_sys_path(["dummy_path"])
    assert after[0] == root


def test_conftest_skips_insert_when_present():
    root = str(Path(__file__).resolve().parents[1])
    before = [root, "dummy_path"]
    after = _exec_conftest_with_sys_path(before)
    assert after == before
