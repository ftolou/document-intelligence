from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_core_rag_sql_imports_do_not_require_langgraph() -> None:
    root = Path(__file__).resolve().parents[3]
    code = r'''
import builtins
import importlib
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "langgraph" or name.startswith("langgraph."):
        raise AssertionError(f"unexpected optional import: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
for module_name in (
    "receipt_intelligence.rag_sql",
    "receipt_intelligence.rag_sql.models",
    "receipt_intelligence.rag_sql.engine",
    "receipt_intelligence.rag_sql.runtime",
    "receipt_intelligence.rag_sql.graph",
    "receipt_intelligence.rag_sql.orchestration",
):
    importlib.import_module(module_name)
assert not any(name == "langgraph" or name.startswith("langgraph.") for name in sys.modules)
'''
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env={"PYTHONPATH": str(root / "src")},
        check=True,
        capture_output=True,
        text=True,
    )
