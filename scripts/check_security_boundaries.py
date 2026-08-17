#!/usr/bin/env python3
"""Fail CI when execution and transport boundaries regress."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
violations: list[str] = []

for path in (ROOT / "src").rglob("*.py"):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant):
                if keyword.value.value is True:
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: shell=True")

request_source = (ROOT / "src/receipt_intelligence/web/request_parsing.py").read_text(
    encoding="utf-8"
)
for field in ("ollama_url", "transcription_model"):
    if f'request.form.get("{field}")' in request_source:
        violations.append(f"Infrastructure field remains request-controlled: {field}")

if violations:
    print("Security boundary violations detected:")
    for violation in violations:
        print(f"- {violation}")
    raise SystemExit(1)

print("Security boundary checks passed.")
