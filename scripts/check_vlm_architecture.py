#!/usr/bin/env python3
"""Enforce the process, source, and dependency boundary around receipt-vlm."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_PACKAGE = ROOT / "src" / "receipt_intelligence"
SERVICE_ROOT = ROOT / "services" / "receipt-vlm"
SERVICE_PACKAGE = SERVICE_ROOT / "src" / "receipt_vlm_service"
violations: list[str] = []


legacy_paths = (
    ROOT / "vlm_service.py",
    APP_PACKAGE / "entrypoints" / "vlm_http",
    APP_PACKAGE / "vlm_composition.py",
    APP_PACKAGE / "adapters" / "vlm" / "paddle_cli.py",
    APP_PACKAGE / "adapters" / "vlm" / "paddle_python.py",
    APP_PACKAGE / "adapters" / "vlm" / "trusted_command.py",
)
for path in legacy_paths:
    if path.exists():
        violations.append(f"Obsolete in-process VLM path still exists: {path.relative_to(ROOT)}")

required_service_files = (
    SERVICE_ROOT / "pyproject.toml",
    SERVICE_PACKAGE / "app.py",
    SERVICE_PACKAGE / "cli.py",
    SERVICE_PACKAGE / "service.py",
    SERVICE_PACKAGE / "settings.py",
)
for path in required_service_files:
    if not path.exists():
        violations.append(f"Missing standalone VLM service file: {path.relative_to(ROOT)}")

for path in SERVICE_PACKAGE.rglob("*.py"):
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path), feature_version=(3, 10))
    except SyntaxError as exc:
        violations.append(
            f"Standalone VLM service is not Python 3.10 compatible: {path.relative_to(ROOT)}: {exc}"
        )
        continue
    for node in ast.walk(tree):
        imported: list[str] = []
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        for module in imported:
            if module == "receipt_intelligence" or module.startswith("receipt_intelligence."):
                violations.append(
                    f"Standalone VLM service imports main application code: "
                    f"{path.relative_to(ROOT)} -> {module}"
                )

client_composition = APP_PACKAGE / "vlm_client_composition.py"
if not client_composition.exists():
    violations.append("Missing application-side VLM client composition root")
else:
    source = client_composition.read_text(encoding="utf-8")
    for forbidden in (
        "PaddleCliVlmEngine",
        "PaddlePythonVlmEngine",
        "TrustedCommandVlmEngine",
        "build_vlm_service_engine",
        "receipt_vlm_service",
    ):
        if forbidden in source:
            violations.append(f"Application VLM client composition contains {forbidden}")
    if "RemoteVlmClient" not in source:
        violations.append("Application VLM composition must use the remote HTTP client")

main_composition = (APP_PACKAGE / "composition.py").read_text(encoding="utf-8")
if "vlm_client_composition" not in main_composition:
    violations.append("Main composition must import the remote-only VLM client composition")
if "vlm_composition" in main_composition:
    violations.append("Main composition still imports the obsolete shared VLM composition root")

adapter_init = (APP_PACKAGE / "adapters" / "vlm" / "__init__.py").read_text(encoding="utf-8")
for forbidden in ("paddle_cli", "paddle_python", "trusted_command"):
    if forbidden in adapter_init:
        violations.append(f"Application VLM adapter package eagerly exposes {forbidden}")

vlm_dockerfile = (ROOT / "docker" / "Dockerfile.vlm-app").read_text(encoding="utf-8")
if "COPY services/receipt-vlm/src /app/src" not in vlm_dockerfile:
    violations.append("VLM image must copy only services/receipt-vlm/src")
if "COPY src /app/src" in vlm_dockerfile:
    violations.append("VLM image must not copy the main application source tree")
if 'CMD ["python", "-m", "receipt_vlm_service.app"]' not in vlm_dockerfile:
    violations.append("VLM image must start the standalone receipt_vlm_service package")

dev_compose = (ROOT / "docker-compose.dev.yml").read_text(encoding="utf-8")
vlm_block = dev_compose.split("  receipt-vlm:", maxsplit=1)[-1]
if "- .:/app" in vlm_block:
    violations.append("Development VLM service must not mount the complete repository")
if "./services/receipt-vlm/src:/app/src:ro" not in vlm_block:
    violations.append("Development VLM service must mount only its standalone source package")
if "receipt_vlm_service.app" not in vlm_block:
    violations.append("Development VLM service must use the standalone entrypoint")

runtime_dockerfile = (ROOT / "docker" / "Dockerfile.vlm-runtime-cu126").read_text(encoding="utf-8")
for required in ("python3", "python3-pip", "python3-dev"):
    if required not in runtime_dockerfile:
        violations.append(f"Known-working VLM runtime dependency missing: {required}")
if "uv python install" in runtime_dockerfile:
    violations.append("VLM runtime must not install a second managed Python runtime")

requirements = (ROOT / "requirements" / "vlm-gpu-cu126.txt").read_text(encoding="utf-8")
if "paddlepaddle-gpu==3.2.1" not in requirements:
    violations.append("VLM requirements must retain paddlepaddle-gpu==3.2.1")
if "https://www.paddlepaddle.org.cn/packages/stable/cu126/" not in requirements:
    violations.append("VLM requirements must retain the known Paddle CUDA 12.6 index")

if violations:
    print("VLM architecture violations detected:")
    for violation in violations:
        print(f"- {violation}")
    raise SystemExit(1)

print("Standalone VLM service boundary checks passed.")
