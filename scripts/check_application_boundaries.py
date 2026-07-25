#!/usr/bin/env python3
"""Enforce the HTTP-to-application use-case boundary."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "receipt_intelligence"
ROUTES = SRC / "web" / "routes"
FORBIDDEN_ROUTE_PREFIXES = (
    "receipt_intelligence.services",
    "receipt_intelligence.storage",
    "receipt_intelligence.rag_sql",
    "receipt_intelligence.observability",
)
FORBIDDEN_ROUTE_MODULES = {"threading", "sqlite3", "subprocess"}
FORBIDDEN_USE_CASE_PREFIXES = (
    "flask",
    "receipt_intelligence.adapters",
    "receipt_intelligence.observability",
    "receipt_intelligence.services",
    "receipt_intelligence.storage",
    "receipt_intelligence.web",
)
TRANSPORT_PATH_MARKERS = (
    '"/api/artifact/',
    'f"/api/artifact/',
    '"/api/review/',
    'f"/api/review/',
    '"/api/status/',
    'f"/api/status/',
)


def imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def main() -> int:
    violations: list[str] = []
    for path in sorted(ROUTES.glob("*.py")):
        for module in imported_modules(path):
            if module in FORBIDDEN_ROUTE_MODULES or module.startswith(FORBIDDEN_ROUTE_PREFIXES):
                violations.append(f"{path.relative_to(ROOT)} imports {module}")

    for path in sorted((SRC / "application" / "use_cases").glob("*.py")):
        for module in imported_modules(path):
            if module.startswith(FORBIDDEN_USE_CASE_PREFIXES):
                violations.append(f"{path.relative_to(ROOT)} imports outward dependency {module}")

    for directory in (SRC / "application", SRC / "services"):
        for path in sorted(directory.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for marker in TRANSPORT_PATH_MARKERS:
                if marker in text:
                    violations.append(
                        f"{path.relative_to(ROOT)} contains transport path marker {marker}"
                    )

    dependency_source = (SRC / "web" / "dependencies.py").read_text(encoding="utf-8")
    app_services_block = dependency_source.split("class AppServices", 1)[-1].split(
        "def init_app_services", 1
    )[0]
    if "job_store:" in app_services_block or "receipt_db:" in app_services_block:
        violations.append("web AppServices exposes concrete persistence resources")

    if violations:
        print("Application boundary violations:")
        for violation in violations:
            print(f"- {violation}")
        return 1
    print("Application boundary checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
