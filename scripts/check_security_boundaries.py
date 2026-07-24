#!/usr/bin/env python3
"""Fail CI when Phase 0 execution and transport boundaries regress."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"

violations: list[str] = []

for path in SRC_ROOT.rglob("*.py"):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if (
                keyword.arg == "shell"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
            ):
                violations.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}: shell=True is forbidden"
                )

request_source = (
    ROOT / "src" / "receipt_intelligence" / "web" / "request_parsing.py"
).read_text(encoding="utf-8")
for field in (
    "ollama_url",
    "vlm_service_url",
    "vlm_command",
    "ollama_control_mode",
    "ollama_unload_command",
    "ollama_start_command",
    "vlm_timeout_seconds",
    "vlm_gpu_orchestration",
    "ollama_unload_before_vlm",
    "ollama_reload_after_vlm",
    "ollama_control_timeout_seconds",
    "ollama_reload_prompt",
    "ollama_gpu_handoff_wait_seconds",
):
    fragment = f'request.form.get("{field}")'
    if fragment in request_source:
        violations.append(
            "src/receipt_intelligence/web/request_parsing.py: "
            f"infrastructure field remains request-controlled: {field}"
        )

vlm_service_source = (
    ROOT / "src" / "receipt_intelligence" / "services" / "vlm_service.py"
).read_text(encoding="utf-8")
for field in ("backend", "runner", "command", "timeout_seconds", "max_side_limit"):
    fragment = f'payload.get("{field}")'
    if fragment in vlm_service_source:
        violations.append(
            "src/receipt_intelligence/services/vlm_service.py: "
            f"execution-policy field remains request-controlled: {field}"
        )

compose_source = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
if '"7870:7870"' in compose_source:
    violations.append("docker-compose.yml: VLM service port 7870 must not be host-published")

if violations:
    print("Security boundary violations detected:")
    for violation in violations:
        print(f"- {violation}")
    raise SystemExit(1)

print("Phase 0 security boundary checks passed.")
