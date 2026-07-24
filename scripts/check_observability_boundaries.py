#!/usr/bin/env python3
"""Enforce provider-neutral, one-way observability dependencies."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "receipt_intelligence"
OBSERVABILITY = SRC / "observability"
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


for path in sorted(OBSERVABILITY.rglob("*.py")):
    for module in imported_modules(path):
        if module.startswith(
            (
                "receipt_intelligence.extraction",
                "receipt_intelligence.rag",
                "receipt_intelligence.rag_sql",
                "receipt_intelligence.services",
                "receipt_intelligence.storage",
                "receipt_intelligence.web",
            )
        ):
            violations.append(
                f"{path.relative_to(ROOT)} imports feature/runtime dependency {module}"
            )

required_files = (
    SRC / "application" / "events.py",
    SRC / "application" / "ports" / "events.py",
    SRC / "adapters" / "observability" / "event_sinks.py",
)
for path in required_files:
    if not path.exists():
        violations.append(f"Missing event boundary module: {path.relative_to(ROOT)}")

workflow_source = (SRC / "extraction" / "workflow.py").read_text(encoding="utf-8")
if "ExtractionRunEvent(" not in workflow_source:
    violations.append("Extraction workflow must emit a typed ExtractionRunEvent")
if "dependencies.event_sink.publish(" not in workflow_source:
    violations.append("Extraction workflow must publish through the injected EventSink")
if "observability.extraction" in workflow_source:
    violations.append("Extraction workflow must not depend on an observability formatter")

query_application = (SRC / "rag_sql" / "application.py").read_text(encoding="utf-8")
if "EventSink" not in query_application:
    violations.append("ReceiptQueryService must depend on the EventSink port")
if "QueryTelemetrySink" in query_application:
    violations.append("ReceiptQueryService must not depend on QueryTelemetrySink")
if "query_execution_event_from_payload" not in query_application:
    violations.append("ReceiptQueryService must publish a typed query event")

web_dependencies = (SRC / "web" / "dependencies.py").read_text(encoding="utf-8")
if "JsonlEventSink" not in web_dependencies:
    violations.append("Web composition must select the concrete JSONL event adapter")
if "QueryTelemetrySink" in web_dependencies:
    violations.append("Web composition must not select the compatibility telemetry wrapper")

graph_support = (SRC / "rag_sql" / "graph_support.py").read_text(encoding="utf-8")
if 'return {"model_calls":' not in graph_support:
    violations.append("RAG-SQL stage diagnostics must use the neutral model_calls key")
if 'diagnostics["model_call_summary"]' not in graph_support:
    violations.append("RAG-SQL diagnostics must expose model_call_summary")

query_event_source = (SRC / "application" / "events.py").read_text(encoding="utf-8")
if '"ollama":' in query_event_source:
    violations.append("Neutral application events must not serialize an ollama field")

if violations:
    print("Observability boundary violations detected:")
    for violation in violations:
        print(f"- {violation}")
    raise SystemExit(1)

print("Observability event boundary checks passed.")
