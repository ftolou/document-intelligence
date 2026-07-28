from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_vlm_client_composition_import_does_not_load_runtime_or_query_infrastructure() -> None:
    code = r"""
import importlib.abc
import sys

blocked = (
    "receipt_intelligence.adapters.llm",
    "receipt_intelligence.adapters.observability",
    "receipt_intelligence.adapters.storage",
    "receipt_intelligence.application.query_diagnostics",
    "receipt_intelligence.application.ports.llm",
    "receipt_intelligence.application.model_call_context",
    "receipt_intelligence.adapters.vlm.paddle_cli",
    "receipt_intelligence.adapters.vlm.paddle_python",
    "receipt_intelligence.adapters.vlm.trusted_command",
)


class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(blocked):
            raise AssertionError(f"forbidden VLM startup dependency: {fullname}")
        return None


sys.meta_path.insert(0, Blocker())
import receipt_intelligence.vlm_client_composition
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")

    subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
