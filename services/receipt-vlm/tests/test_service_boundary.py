from __future__ import annotations

import ast
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = SERVICE_ROOT / "src" / "receipt_vlm_service"


def test_vlm_service_does_not_import_main_application() -> None:
    violations: list[str] = []
    for path in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            for module in modules:
                if module == "receipt_intelligence" or module.startswith("receipt_intelligence."):
                    violations.append(f"{path.relative_to(SERVICE_ROOT)} imports {module}")

    assert not violations, "\n".join(violations)


def test_vlm_service_is_python_310_targeted() -> None:
    pyproject = (SERVICE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.10,<3.11"' in pyproject
    assert 'target-version = "py310"' in pyproject
