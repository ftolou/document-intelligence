"""LangGraph adapter for the RAG-SQL orchestration contract.

This is the only production RAG-SQL module that imports ``langgraph``. Core
models, validators, planners, storage adapters, and the stable engine facade
remain importable when the optional graph runtime is unavailable.
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from receipt_intelligence.rag_sql.graph_nodes import RagSqlGraphNodes
from receipt_intelligence.rag_sql.graph_state import RagSqlGraphConfig, RagSqlGraphState
from receipt_intelligence.rag_sql.models import RagSqlResponse
from receipt_intelligence.rag_sql.orchestration.contracts import (
    RAG_SQL_GRAPH_VERSION,
    RagSqlComponents,
    RagSqlOrchestrator,
)


class LangGraphRagSqlOrchestrator(RagSqlOrchestrator):
    """Compile one immutable graph and reuse it for every query."""

    name = "langgraph"
    version = RAG_SQL_GRAPH_VERSION

    def __init__(
        self,
        components: RagSqlComponents,
        *,
        graph_config: RagSqlGraphConfig,
    ) -> None:
        self._graph_config = graph_config
        self._graph = compile_rag_sql_graph(components)

    def execute(self, question: str) -> RagSqlResponse:
        return run_compiled_rag_sql_graph(
            self._graph,
            question,
            graph_config=self._graph_config,
        )


class LangGraphRagSqlOrchestratorFactory:
    """Factory kept behind the provider-neutral orchestration boundary."""

    def build(
        self,
        components: RagSqlComponents,
        *,
        graph_config: RagSqlGraphConfig,
    ) -> RagSqlOrchestrator:
        return LangGraphRagSqlOrchestrator(
            components,
            graph_config=graph_config,
        )


def compile_rag_sql_graph(components: RagSqlComponents) -> Any:
    nodes = RagSqlGraphNodes(
        analyzer=components.analyzer,
        retriever=components.retriever,
        resolver=components.resolver,
        planner=components.planner,
        validator=components.validator,
        executor=components.executor,
        answer_formatter=components.answer_formatter,
        retrieval_limit=components.retrieval_limit,
        retrieval_minimum_score=components.retrieval_minimum_score,
        validation_repair_count=components.validation_repair_count,
    )

    graph = StateGraph(RagSqlGraphState)
    graph.add_node("analyze_question", nodes.analyze_question)
    graph.add_node("retrieve_entity", nodes.retrieve_entity)
    graph.add_node("generate_sql", nodes.generate_sql)
    graph.add_node("validate_sql", nodes.validate_sql)
    graph.add_node("repair_sql", nodes.repair_sql)
    graph.add_node("execute_sql", nodes.execute_sql)
    graph.add_node("extract_answer", nodes.extract_answer)
    graph.add_node("format_answer_with_llm", nodes.format_answer_with_llm)
    graph.add_node("validate_formatted_answer", nodes.validate_formatted_answer)
    graph.add_node("finalize_response", nodes.finalize_response)
    graph.add_node("terminal", nodes.terminal)
    graph.add_node("failure", nodes.failure)

    graph.add_edge(START, "analyze_question")
    graph.add_conditional_edges(
        "analyze_question",
        lambda state: state.get("route", "fail"),
        {
            "retrieve": "retrieve_entity",
            "plan": "generate_sql",
            "terminal": "terminal",
            "fail": "failure",
        },
    )
    graph.add_conditional_edges(
        "retrieve_entity",
        lambda state: state.get("route", "fail"),
        {
            "retrieve": "retrieve_entity",
            "plan": "generate_sql",
            "terminal": "terminal",
            "fail": "failure",
        },
    )
    graph.add_conditional_edges(
        "generate_sql",
        lambda state: state.get("route", "fail"),
        {
            "validate": "validate_sql",
            "terminal": "terminal",
            "fail": "failure",
        },
    )
    graph.add_conditional_edges(
        "validate_sql",
        lambda state: state.get("route", "fail"),
        {"execute": "execute_sql", "repair": "repair_sql", "fail": "failure"},
    )
    graph.add_conditional_edges(
        "repair_sql",
        lambda state: state.get("route", "fail"),
        {"validate": "validate_sql", "terminal": "terminal", "fail": "failure"},
    )
    graph.add_conditional_edges(
        "execute_sql",
        lambda state: state.get("route", "fail"),
        {"extract": "extract_answer", "fail": "failure"},
    )
    graph.add_conditional_edges(
        "extract_answer",
        lambda state: state.get("route", "fail"),
        {
            "llm_format": "format_answer_with_llm",
            "finalize": "finalize_response",
            "fail": "failure",
        },
    )
    graph.add_conditional_edges(
        "format_answer_with_llm",
        lambda state: state.get("route", "fail"),
        {
            "validate_answer": "validate_formatted_answer",
            "finalize": "finalize_response",
            "fail": "failure",
        },
    )
    graph.add_conditional_edges(
        "validate_formatted_answer",
        lambda state: state.get("route", "fail"),
        {"finalize": "finalize_response", "fail": "failure"},
    )
    graph.add_edge("finalize_response", END)
    graph.add_edge("terminal", END)
    graph.add_edge("failure", END)
    return graph.compile()


def run_compiled_rag_sql_graph(
    graph: Any,
    question: str,
    *,
    graph_config: RagSqlGraphConfig,
) -> RagSqlResponse:
    started = time.perf_counter()
    final_state = graph.invoke(
        {
            "question": question,
            "started_at_perf": started,
            "diagnostics": {
                "orchestrator": "langgraph",
                "graph_version": RAG_SQL_GRAPH_VERSION,
                "stages": [],
                "graph_trace": [],
            },
            "entity_index": 0,
            "resolved_entities": [],
            "retrieval_diagnostics": [],
            "validation_attempt": 1,
        },
        config={"recursion_limit": graph_config.recursion_limit},
    )
    response = final_state.get("final_response")
    if not isinstance(response, RagSqlResponse):
        raise RuntimeError("RAG-SQL graph completed without a final response.")
    return response


__all__ = [
    "LangGraphRagSqlOrchestrator",
    "LangGraphRagSqlOrchestratorFactory",
    "compile_rag_sql_graph",
    "run_compiled_rag_sql_graph",
]
