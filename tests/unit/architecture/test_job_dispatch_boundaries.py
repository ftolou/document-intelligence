"""Architecture regression tests for background-job ownership."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src" / "receipt_intelligence"


def test_web_and_use_cases_do_not_create_threads() -> None:
    violations: list[str] = []
    for directory in (SRC / "web", SRC / "application" / "use_cases"):
        for path in directory.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "threading" or alias.name.startswith(
                            "concurrent.futures"
                        ):
                            violations.append(str(path.relative_to(ROOT)))
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if node.module == "threading" or node.module.startswith(
                        "concurrent.futures"
                    ):
                        violations.append(str(path.relative_to(ROOT)))
    assert violations == []


def test_only_job_adapter_owns_worker_primitives() -> None:
    violations: list[str] = []
    adapter_root = SRC / "adapters" / "jobs"
    for path in SRC.rglob("*.py"):
        if path.is_relative_to(adapter_root):
            continue
        text = path.read_text(encoding="utf-8-sig")
        if "threading.Thread(" in text or "ThreadPoolExecutor(" in text:
            violations.append(str(path.relative_to(ROOT)))
    assert violations == []
