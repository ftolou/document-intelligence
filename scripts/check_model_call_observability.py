#!/usr/bin/env python3
"""Enforce the model-call telemetry and pricing boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "receipt_intelligence"
violations: list[str] = []

for path in SRC.rglob("*.py"):
    text = path.read_text(encoding="utf-8")
    relative = path.relative_to(ROOT).as_posix()
    if "CREATE TABLE" in text and "model_calls" in text and "storage" not in relative:
        violations.append(f"{relative}: feature code owns model-call schema")
    if (
        path.name in {"ollama_gateway.py", "observed_gateway.py"}
        and "price_per_million" in text
    ):
        violations.append(f"{relative}: model adapter contains pricing policy")

for path in SRC.rglob("*.py"):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name) and function.id == "GenerationRequest":
            if not any(keyword.arg == "operation" for keyword in node.keywords):
                relative = path.relative_to(ROOT).as_posix()
                violations.append(
                    f"{relative}:{node.lineno}: GenerationRequest lacks operation"
                )

if violations:
    raise SystemExit(
        "Model-call observability violations:\n- " + "\n- ".join(violations)
    )
print("Model-call observability boundaries passed.")
