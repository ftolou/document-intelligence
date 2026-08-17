#!/usr/bin/env python3
"""Enforce managed background-job execution and persistent lifecycle ownership."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "receipt_intelligence"
MANAGED_ADAPTER = SRC / "adapters" / "jobs"
FORBIDDEN_DIRECTORIES = (
    SRC / "web",
    SRC / "application" / "use_cases",
)


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def main() -> int:
    violations: list[str] = []
    for directory in FORBIDDEN_DIRECTORIES:
        for path in sorted(directory.rglob("*.py")):
            modules = _imports(path)
            for module in modules:
                if module == "threading" or module.startswith("concurrent.futures"):
                    violations.append(f"{path.relative_to(ROOT)} imports worker primitive {module}")
            text = path.read_text(encoding="utf-8-sig")
            if "threading.Thread(" in text or "ThreadPoolExecutor(" in text:
                violations.append(f"{path.relative_to(ROOT)} creates background workers directly")

    for path in sorted(SRC.rglob("*.py")):
        if path.is_relative_to(MANAGED_ADAPTER):
            continue
        text = path.read_text(encoding="utf-8-sig")
        if "threading.Thread(" in text:
            violations.append(
                f"{path.relative_to(ROOT)} creates background workers outside adapters/jobs"
            )

    use_case = (SRC / "application" / "use_cases" / "jobs.py").read_text(encoding="utf-8-sig")
    for marker in ("JobDispatcher", "JobDispatchRequest", "self._dispatcher.submit"):
        if marker not in use_case:
            violations.append(f"job use case is missing dispatcher boundary: {marker}")

    processor = (SRC / "services" / "job_processing.py").read_text(encoding="utf-8-sig")
    for marker in ('state="done"', 'state="error"', 'state="running"'):
        if marker in processor:
            violations.append(f"job processor owns persisted lifecycle transition {marker}")

    store = (SRC / "storage" / "job_store.py").read_text(encoding="utf-8-sig")
    for field in (
        '"attempt_count"',
        '"started_at"',
        '"finished_at"',
        'state="completed"',
        'state="failed"',
    ):
        if field not in store:
            violations.append(f"job store is missing lifecycle contract {field}")

    if violations:
        print("Background job boundary violations:")
        for violation in violations:
            print(f"- {violation}")
        return 1
    print("Background job boundary checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
