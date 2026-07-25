#!/usr/bin/env python3
"""Enforce process-scoped RAG-SQL composition and optional graph isolation."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "receipt_intelligence" / "rag_sql"
LANGGRAPH_ADAPTER = PACKAGE / "orchestration" / "langgraph.py"
violations: list[str] = []


def imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


for path in sorted(PACKAGE.rglob("*.py")):
    for module in imported_modules(path):
        if module == "langgraph" or module.startswith("langgraph."):
            if path != LANGGRAPH_ADAPTER:
                violations.append(
                    f"{path.relative_to(ROOT)} imports optional graph runtime {module}"
                )

engine_source = (PACKAGE / "engine.py").read_text(encoding="utf-8")
if "receipt_intelligence.rag_sql.graph import" in engine_source:
    violations.append("rag_sql/engine.py must not import the compatibility graph facade")
if "orchestration.langgraph import" not in engine_source:
    violations.append("rag_sql/engine.py must select LangGraph only in the lazy factory")

runtime_path = PACKAGE / "runtime.py"
runtime_tree = ast.parse(runtime_path.read_text(encoding="utf-8"), filename=str(runtime_path))
runtime_classes = [
    node
    for node in runtime_tree.body
    if isinstance(node, ast.ClassDef) and node.name == "RagSqlRuntime"
]
execute_methods = (
    [
        node
        for node in runtime_classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "execute"
    ]
    if len(runtime_classes) == 1
    else []
)
if len(execute_methods) != 1:
    violations.append("RagSqlRuntime must define exactly one execute method")
else:
    execute_source = ast.unparse(execute_methods[0])
    forbidden_constructors = (
        "OllamaGateway(",
        "OllamaEmbeddingClient(",
        "ItemSemanticRetriever(",
        "RagSqlEngine(",
        "SQLiteSemanticSearchRepository(",
        "SQLiteAnalyticalQueryRepository(",
    )
    for token in forbidden_constructors:
        if token in execute_source:
            violations.append(f"RagSqlRuntime.execute composes per-request dependency {token[:-1]}")

runtime_source = runtime_path.read_text(encoding="utf-8")
for required in (
    "def close(self)",
    "def __enter__(self)",
    "def __exit__(self",
    "build_rag_sql_runtime_config_from_settings",
):
    if required not in runtime_source:
        violations.append(f"rag_sql/runtime.py is missing lifecycle contract {required!r}")

application_source = (PACKAGE / "application.py").read_text(encoding="utf-8")
if "receipt_intelligence.rag_sql.graph import" in application_source:
    violations.append("rag_sql/application.py must not import the graph adapter facade")
if "def close(self)" not in application_source:
    violations.append("ReceiptQueryService must release the process-scoped runtime")

init_source = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
if "from receipt_intelligence.rag_sql" in init_source:
    violations.append("rag_sql/__init__.py must retain lazy exports only")

web_dependencies = (ROOT / "src" / "receipt_intelligence" / "web" / "dependencies.py").read_text(
    encoding="utf-8"
)
if "query_executor=resolved_query_service" not in web_dependencies:
    violations.append("AppServices must retain the managed query executor")
if 'getattr(self.query_executor, "close", None)' not in web_dependencies:
    violations.append("AppServices.shutdown must close the managed query executor")

if violations:
    print("RAG-SQL composition violations detected:")
    for violation in violations:
        print(f"- {violation}")
    raise SystemExit(1)

print("RAG-SQL composition boundary checks passed.")
