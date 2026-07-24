#!/usr/bin/env python3
"""Enforce the visual-model adapter and transport boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "receipt_intelligence"
violations: list[str] = []

obsolete_modules = (
    PACKAGE / "engines" / "vl_engine.py",
    PACKAGE / "services" / "vlm_service.py",
    PACKAGE / "adapters" / "vlm" / "legacy_vlm.py",
)
for path in obsolete_modules:
    if path.exists():
        violations.append(f"Obsolete VLM module still exists: {path.relative_to(ROOT)}")

required_modules = (
    PACKAGE / "adapters" / "vlm" / "paddle_python.py",
    PACKAGE / "adapters" / "vlm" / "paddle_cli.py",
    PACKAGE / "adapters" / "vlm" / "remote_client.py",
    PACKAGE / "adapters" / "vlm" / "trusted_command.py",
    PACKAGE / "application" / "vlm" / "engines.py",
    PACKAGE / "application" / "vlm" / "analysis_service.py",
    PACKAGE / "entrypoints" / "vlm_http" / "app.py",
)
for path in required_modules:
    if not path.exists():
        violations.append(f"Missing VLM boundary module: {path.relative_to(ROOT)}")

for path in PACKAGE.rglob("*.py"):
    source = path.read_text(encoding="utf-8")
    if "receipt_intelligence.engines.vl_engine" in source:
        violations.append(f"{path.relative_to(ROOT)} imports the obsolete VLM engine")
    if "receipt_intelligence.services.vlm_service" in source:
        violations.append(f"{path.relative_to(ROOT)} imports the obsolete VLM service")

entrypoint = PACKAGE / "entrypoints" / "vlm_http" / "app.py"
entrypoint_source = entrypoint.read_text(encoding="utf-8")
for forbidden in (
    "run_paddle_python(",
    "run_paddle_cli(",
    "run_trusted_command(",
    "PaddlePythonVlmEngine(",
    "PaddleCliVlmEngine(",
    "TrustedCommandVlmEngine(",
):
    if forbidden in entrypoint_source:
        violations.append(
            "VLM HTTP transport selects or invokes a concrete backend directly: "
            f"{forbidden}"
        )
if "analysis_service.execute(" not in entrypoint_source:
    violations.append("VLM HTTP transport must call VlmAnalysisService.execute()")

application_root = PACKAGE / "application"
for path in application_root.rglob("*.py"):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        module = None
        if isinstance(node, ast.ImportFrom):
            module = node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("receipt_intelligence.adapters"):
                    violations.append(
                        f"{path.relative_to(ROOT)} imports infrastructure adapter {alias.name}"
                    )
        if module and module.startswith("receipt_intelligence.adapters"):
            violations.append(
                f"{path.relative_to(ROOT)} imports infrastructure adapter {module}"
            )

adapters_init = (PACKAGE / "adapters" / "vlm" / "__init__.py").read_text(encoding="utf-8")
if "ConfiguredVlmEngine" in adapters_init or "legacy_vlm" in adapters_init:
    violations.append("VLM adapter package still exports the legacy selector")

if violations:
    print("VLM architecture violations detected:")
    for violation in violations:
        print(f"- {violation}")
    raise SystemExit(1)

print("VLM adapter and transport boundary checks passed.")
