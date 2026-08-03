from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def _run_isolated_python(source: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(SRC_ROOT), current_pythonpath) if value
    )
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_isolated_import_succeeds(source: str) -> str:
    completed = _run_isolated_python(source)
    assert completed.returncode == 0, (
        f"isolated import failed with exit code {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    return completed.stdout


def test_receipt_database_import_is_cycle_free_in_clean_interpreter() -> None:
    stdout = _assert_isolated_import_succeeds(
        "from receipt_intelligence.storage.receipt_db import ReceiptDatabase; "
        "print(ReceiptDatabase.__name__)"
    )
    assert stdout.strip() == "ReceiptDatabase"


def test_extraction_package_does_not_eagerly_import_context() -> None:
    stdout = _assert_isolated_import_succeeds(
        "import sys; import receipt_intelligence.extraction as extraction; "
        "assert 'receipt_intelligence.extraction.context' not in sys.modules; "
        "assert extraction.ExtractionRequest.__name__ == 'ExtractionRequest'; "
        "assert 'receipt_intelligence.extraction.context' not in sys.modules; "
        "print('ok')"
    )
    assert stdout.strip() == "ok"


def test_observability_timing_does_not_eagerly_import_readiness() -> None:
    stdout = _assert_isolated_import_succeeds(
        "import sys; import receipt_intelligence.observability as observability; "
        "assert 'receipt_intelligence.observability.readiness' not in sys.modules; "
        "assert callable(observability.utc_now_iso); "
        "assert 'receipt_intelligence.observability.readiness' not in sys.modules; "
        "print('ok')"
    )
    assert stdout.strip() == "ok"


def test_domain_taxonomy_has_no_inward_application_dependencies() -> None:
    taxonomy_path = SRC_ROOT / "receipt_intelligence" / "domain" / "categorization_taxonomy.py"
    tree = ast.parse(taxonomy_path.read_text(encoding="utf-8"), filename=str(taxonomy_path))
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    forbidden_prefixes = (
        "receipt_intelligence.extraction",
        "receipt_intelligence.observability",
        "receipt_intelligence.runtime",
        "receipt_intelligence.services",
        "receipt_intelligence.storage",
        "receipt_intelligence.web",
    )
    violations = [module for module in imported_modules if module.startswith(forbidden_prefixes)]
    assert violations == []
