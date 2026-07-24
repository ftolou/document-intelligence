#!/usr/bin/env python3
"""Fail CI when feature packages take ownership of concrete persistence."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "receipt_intelligence"
violations: list[str] = []


def reject_tokens(
    relative_root: str,
    tokens: tuple[str, ...],
    *,
    exclude: set[str] | None = None,
) -> None:
    excluded = exclude or set()
    for path in (SRC / relative_root).rglob("*.py"):
        relative = path.relative_to(SRC).as_posix()
        if relative in excluded:
            continue
        source = path.read_text(encoding="utf-8")
        for token in tokens:
            if token in source:
                violations.append(f"{relative} contains forbidden persistence dependency {token!r}")


reject_tokens(
    "rag",
    (
        "import sqlite3",
        "receipt_intelligence.storage",
        "receipt_intelligence.adapters.storage",
        "SQLiteConnectionFactory",
        "MigrationRunner",
        ".migrate()",
    ),
)
reject_tokens(
    "rag_sql",
    (
        "import sqlite3",
        "receipt_intelligence.storage",
        "SQLiteConnectionFactory",
        "MigrationRunner",
        ".migrate()",
    ),
)

runtime_source = (SRC / "rag_sql" / "runtime.py").read_text(encoding="utf-8")
if "MigrationRunner" in runtime_source or ".migrate()" in runtime_source:
    violations.append("rag_sql/runtime.py must not migrate the database during query execution")

required_files = (
    SRC / "rag" / "ports.py",
    SRC / "rag_sql" / "ports.py",
    SRC / "adapters" / "storage" / "sqlite" / "semantic_index.py",
    SRC / "adapters" / "storage" / "sqlite" / "semantic_search.py",
    SRC / "adapters" / "storage" / "sqlite" / "analytical_query.py",
    SRC / "storage" / "bootstrap.py",
)
for path in required_files:
    if not path.exists():
        violations.append(f"Missing persistence boundary module: {path.relative_to(ROOT)}")

indexer_source = (SRC / "rag" / "item_indexer.py").read_text(encoding="utf-8")
if "repository: SemanticIndexRepository" not in indexer_source:
    violations.append("ItemEmbeddingIndexer must require SemanticIndexRepository")

retriever_source = (SRC / "rag" / "item_retriever.py").read_text(encoding="utf-8")
if "repository: SemanticSearchRepository" not in retriever_source:
    violations.append("ItemSemanticRetriever must require SemanticSearchRepository")

executor_source = (SRC / "rag_sql" / "executor.py").read_text(encoding="utf-8")
if "repository: AnalyticalQueryRepository" not in executor_source:
    violations.append("ReadOnlySqlExecutor must require AnalyticalQueryRepository")

if violations:
    print("Persistence boundary violations detected:")
    for violation in violations:
        print(f"- {violation}")
    raise SystemExit(1)

print("Persistence ownership boundary checks passed.")
