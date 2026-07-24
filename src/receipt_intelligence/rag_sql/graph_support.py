"""Shared helpers for the LangGraph RAG-SQL orchestrator."""

from __future__ import annotations

import time
from typing import Protocol

from receipt_intelligence.application.ports.llm import (
    ModelCallMetrics,
    metrics_to_diagnostics,
)
from receipt_intelligence.rag.candidate_models import CandidateResolutionResult
from receipt_intelligence.rag.models import SemanticItemSearchResult
from receipt_intelligence.rag_sql.models import RagSqlResponse, ResolvedSemanticEntity


class RagSqlRetrievalError(RuntimeError):
    """Raised when semantic retrieval fails inside the RAG-SQL graph."""


class SemanticRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        limit: int,
        minimum_score: float | None = None,
        merchant: str | None = None,
        category: str | None = None,
    ) -> SemanticItemSearchResult: ...


def map_resolved_entity(
    entity_id: str,
    search_text: str,
    resolution: CandidateResolutionResult,
) -> ResolvedSemanticEntity:
    return ResolvedSemanticEntity(
        entity_id=entity_id,
        search_text=search_text,
        status=resolution.status,
        selected_item_ids=resolution.selected_item_ids,
        uncertain_item_ids=resolution.uncertain_item_ids,
        clarification_question=resolution.clarification_question,
    )


def append_stage(
    diagnostics: dict[str, object],
    name: str,
    status: str,
    duration_ms: float,
    details: dict[str, object] | None = None,
) -> None:
    stages = diagnostics.setdefault("stages", [])
    if not isinstance(stages, list):
        stages = []
        diagnostics["stages"] = stages
    stages.append(
        {
            "name": name,
            "status": status,
            "duration_ms": round(float(duration_ms), 3),
            **(details or {}),
        }
    )


def append_graph_trace(
    diagnostics: dict[str, object],
    *,
    node: str,
    route: str,
) -> None:
    trace = diagnostics.setdefault("graph_trace", [])
    if not isinstance(trace, list):
        trace = []
        diagnostics["graph_trace"] = trace
    trace.append({"node": node, "route": route})


def ollama_details(calls: object) -> dict[str, object]:
    if not isinstance(calls, (list, tuple)):
        return {}
    typed_calls = [call for call in calls if isinstance(call, ModelCallMetrics)]
    if not typed_calls:
        return {}
    return {"ollama_calls": metrics_to_diagnostics(typed_calls)}


def ollama_summary(diagnostics: dict[str, object]) -> dict[str, object]:
    stages = diagnostics.get("stages")
    if not isinstance(stages, list):
        return {"call_count": 0}

    calls: list[dict[str, object]] = []
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        stage_name = str(stage.get("name") or "")
        stage_calls = stage.get("ollama_calls")
        if not isinstance(stage_calls, list):
            continue
        for index, call in enumerate(stage_calls, start=1):
            if isinstance(call, dict):
                calls.append({"stage": stage_name, "attempt": index, **call})

    def total(field: str) -> float:
        return round(sum(float(call.get(field) or 0.0) for call in calls), 3)

    return {
        "call_count": len(calls),
        "total_request_duration_ms": total("request_duration_ms"),
        "total_provider_duration_ms": total("total_duration_ms"),
        "total_load_duration_ms": total("load_duration_ms"),
        "total_prompt_eval_duration_ms": total("prompt_eval_duration_ms"),
        "total_generation_duration_ms": total("eval_duration_ms"),
        "calls": calls,
    }


def terminal_response(
    question: str,
    *,
    status: str,
    answer: str,
    diagnostics: dict[str, object],
    started: float,
    clarification_question: str | None = None,
) -> RagSqlResponse:
    diagnostics["duration_ms"] = (time.perf_counter() - started) * 1000.0
    diagnostics["ollama_summary"] = ollama_summary(diagnostics)
    return RagSqlResponse(
        question=question,
        status=status,  # type: ignore[arg-type]
        answer=answer,
        clarification_question=clarification_question,
        diagnostics=diagnostics,
    )


def error_response(
    question: str,
    error_code: str,
    exc: Exception,
    diagnostics: dict[str, object],
    started: float,
) -> RagSqlResponse:
    exception_calls = getattr(exc, "ollama_calls", [])
    append_stage(
        diagnostics,
        error_code,
        "error",
        0.0,
        {
            "error": f"{type(exc).__name__}: {exc}",
            **ollama_details(exception_calls),
        },
    )
    diagnostics["duration_ms"] = (time.perf_counter() - started) * 1000.0
    diagnostics["ollama_summary"] = ollama_summary(diagnostics)
    return RagSqlResponse(
        question=question,
        status="error",
        answer="The RAG-SQL query could not be completed.",
        error_code=error_code,
        error=f"{type(exc).__name__}: {exc}",
        diagnostics=diagnostics,
    )


__all__ = [
    "RagSqlRetrievalError",
    "SemanticRetriever",
    "append_graph_trace",
    "append_stage",
    "error_response",
    "map_resolved_entity",
    "ollama_details",
    "ollama_summary",
    "terminal_response",
]
