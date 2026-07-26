"""LangGraph nodes for the RAG-SQL execution pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass

from receipt_intelligence.rag.candidate_resolver import CandidateResolutionError
from receipt_intelligence.rag_sql.answer_formatter import (
    AnswerFormattingError,
    AnswerValidationResult,
    EvidenceBoundAnswerFormatter,
    render_validated_answer,
    validate_answer_formatter_result,
)
from receipt_intelligence.rag_sql.executor import ReadOnlySqlExecutor, SqlExecutionError
from receipt_intelligence.rag_sql.filter_resolution import (
    FilterResolutionError,
    QueryFilterResolver,
)
from receipt_intelligence.rag_sql.formatter import (
    DeterministicAnswerDecision,
    classify_rag_sql_outcome,
)
from receipt_intelligence.rag_sql.graph_state import RagSqlGraphState
from receipt_intelligence.rag_sql.graph_support import (
    append_graph_trace,
    append_stage,
    attach_model_call_summary,
    error_response,
    model_call_details,
    terminal_response,
)
from receipt_intelligence.rag_sql.models import RagSqlResponse
from receipt_intelligence.rag_sql.planner import (
    RagSqlPlanner,
    RagSqlPlanningError,
    build_protected_filter_parameters,
)
from receipt_intelligence.rag_sql.question_analyzer import (
    QuestionAnalysisError,
    RagSqlQuestionAnalyzer,
)
from receipt_intelligence.rag_sql.validator import RagSqlValidator, SqlValidationError


@dataclass(slots=True)
class RagSqlGraphNodes:
    analyzer: RagSqlQuestionAnalyzer
    filter_resolver: QueryFilterResolver
    planner: RagSqlPlanner
    validator: RagSqlValidator
    executor: ReadOnlySqlExecutor
    answer_formatter: EvidenceBoundAnswerFormatter | None
    validation_repair_count: int

    @staticmethod
    def _diagnostics(state: RagSqlGraphState) -> dict[str, object]:
        diagnostics = dict(state.get("diagnostics") or {})
        diagnostics["stages"] = list(diagnostics.get("stages") or [])
        diagnostics["graph_trace"] = list(diagnostics.get("graph_trace") or [])
        return diagnostics

    @staticmethod
    def _failure(
        state: RagSqlGraphState,
        *,
        node: str,
        error_code: str,
        exc: Exception,
        diagnostics: dict[str, object],
    ) -> RagSqlGraphState:
        append_graph_trace(diagnostics, node=node, route="fail")
        return {
            "diagnostics": diagnostics,
            "route": "fail",
            "error_code": error_code,
            "exception": exc,
        }

    def analyze_question(self, state: RagSqlGraphState) -> RagSqlGraphState:
        diagnostics = self._diagnostics(state)
        try:
            analysis = self.analyzer.analyze(state["question"])
        except QuestionAnalysisError as exc:
            return self._failure(
                state,
                node="analyze_question",
                error_code="question_analysis_failed",
                exc=exc,
                diagnostics=diagnostics,
            )
        except Exception as exc:
            return self._failure(
                state,
                node="analyze_question",
                error_code="question_analysis_failed",
                exc=exc,
                diagnostics=diagnostics,
            )

        resolution_filter_count = sum(
            value.field in {"product", "merchant", "category", "payment_method", "currency"}
            for value in analysis.filters
        )
        append_stage(
            diagnostics,
            "analyze_question",
            "done",
            analysis.duration_ms,
            {
                "status": analysis.status,
                "filter_count": len(analysis.filters),
                "resolution_filter_count": resolution_filter_count,
                "requires_product_resolution": analysis.requires_product_resolution,
                "entity_count": len(analysis.entities),
                "attempts": analysis.attempts,
                "model": analysis.model,
                **model_call_details(analysis.ollama_calls),
            },
        )
        diagnostics["analysis"] = analysis.model_dump(mode="json")

        if analysis.status == "needs_clarification":
            route = "terminal"
            result: RagSqlGraphState = {
                "analysis": analysis,
                "terminal_status": "needs_clarification",
                "terminal_answer": analysis.clarification_question or "Clarification is required.",
                "clarification_question": analysis.clarification_question,
            }
        elif analysis.status == "unsupported":
            route = "terminal"
            result = {
                "analysis": analysis,
                "terminal_status": "unsupported",
                "terminal_answer": analysis.reason
                or "The question is not supported by receipt analytics.",
            }
        else:
            route = "resolve" if analysis.filters else "plan"
            result = {
                "analysis": analysis,
                "filter_index": 0,
                "resolved_filters": [],
                "filter_diagnostics": [],
            }

        append_graph_trace(diagnostics, node="analyze_question", route=route)
        result.update({"diagnostics": diagnostics, "route": route})
        return result

    def resolve_filter(self, state: RagSqlGraphState) -> RagSqlGraphState:
        diagnostics = self._diagnostics(state)
        analysis = state["analysis"]
        filter_index = int(state.get("filter_index", 0))
        if filter_index >= len(analysis.filters):
            append_graph_trace(diagnostics, node="resolve_filter", route="plan")
            return {"diagnostics": diagnostics, "route": "plan"}

        query_filter = analysis.filters[filter_index]
        try:
            bundle = self.filter_resolver.resolve(
                query_filter,
                user_question=state["question"],
                language=analysis.language,
            )
        except CandidateResolutionError as exc:
            return self._failure(
                state,
                node="resolve_filter",
                error_code="candidate_resolution_failed",
                exc=exc,
                diagnostics=diagnostics,
            )
        except FilterResolutionError as exc:
            return self._failure(
                state,
                node="resolve_filter",
                error_code="filter_resolution_failed",
                exc=exc,
                diagnostics=diagnostics,
            )
        except Exception as exc:
            return self._failure(
                state,
                node="resolve_filter",
                error_code="filter_resolution_failed",
                exc=exc,
                diagnostics=diagnostics,
            )

        resolution = bundle.resolution
        filter_diagnostics = list(state.get("filter_diagnostics") or [])
        filter_event = {
            "filter_id": query_filter.filter_id,
            "field": query_filter.field,
            "operator": query_filter.operator,
            "status": resolution.status,
            "resolved_value_count": len(resolution.resolved_values),
            "candidate_value_count": len(resolution.candidate_values),
            **bundle.details,
            **model_call_details(bundle.model_calls),
        }
        filter_diagnostics.append(filter_event)
        append_stage(
            diagnostics,
            f"resolve_{query_filter.filter_id}",
            "done",
            bundle.duration_ms,
            filter_event,
        )

        resolved_filters = list(state.get("resolved_filters") or [])
        resolved_filters.append(resolution)
        diagnostics["filter_resolution"] = filter_diagnostics
        diagnostics["resolved_filters"] = [
            value.model_dump(mode="json") for value in resolved_filters
        ]

        result: RagSqlGraphState = {
            "diagnostics": diagnostics,
            "filter_diagnostics": filter_diagnostics,
            "resolved_filters": resolved_filters,
            "filter_index": filter_index + 1,
        }
        if resolution.status == "needs_clarification":
            route = "terminal"
            result.update(
                {
                    "terminal_status": "needs_clarification",
                    "terminal_answer": resolution.clarification_question
                    or "Clarification is required.",
                    "clarification_question": resolution.clarification_question,
                }
            )
        elif resolution.status == "not_found":
            route = "terminal"
            result.update(
                {
                    "terminal_status": "not_found",
                    "terminal_answer": _not_found_answer(
                        analysis.language,
                        query_filter.field,
                        query_filter.value,
                    ),
                }
            )
        elif filter_index + 1 < len(analysis.filters):
            route = "resolve"
        else:
            route = "plan"

        append_graph_trace(diagnostics, node="resolve_filter", route=route)
        result["route"] = route
        return result

    def generate_sql(self, state: RagSqlGraphState) -> RagSqlGraphState:
        diagnostics = self._diagnostics(state)
        resolved_filters = list(state.get("resolved_filters") or [])
        protected_parameters = build_protected_filter_parameters(resolved_filters)
        try:
            plan = self.planner.plan(
                state["question"],
                analysis=state["analysis"],
                resolved_entities=resolved_filters,
                protected_parameters=protected_parameters,
            )
        except RagSqlPlanningError as exc:
            return self._failure(
                state,
                node="generate_sql",
                error_code="sql_planning_failed",
                exc=exc,
                diagnostics=diagnostics,
            )
        except Exception as exc:
            return self._failure(
                state,
                node="generate_sql",
                error_code="sql_planning_failed",
                exc=exc,
                diagnostics=diagnostics,
            )

        append_stage(
            diagnostics,
            "generate_sql",
            "done",
            plan.duration_ms,
            {
                "status": plan.status,
                "attempts": plan.attempts,
                "model": plan.model,
                "result_shape": plan.result_shape,
                **model_call_details(plan.ollama_calls),
            },
        )
        diagnostics["sql_plan"] = plan.model_dump(mode="json")
        diagnostics["sql_plan_attempts"] = [plan.model_dump(mode="json")]

        result: RagSqlGraphState = {
            "diagnostics": diagnostics,
            "plan": plan,
            "protected_parameters": protected_parameters,
            "validation_attempt": 1,
            "first_validation_error": None,
        }
        if plan.status == "needs_clarification":
            route = "terminal"
            result.update(
                {
                    "terminal_status": "needs_clarification",
                    "terminal_answer": plan.clarification_question or "Clarification is required.",
                    "clarification_question": plan.clarification_question,
                }
            )
        elif plan.status == "unsupported":
            route = "terminal"
            result.update(
                {
                    "terminal_status": "unsupported",
                    "terminal_answer": plan.reason or "The requested analysis is unsupported.",
                }
            )
        else:
            route = "validate"

        append_graph_trace(diagnostics, node="generate_sql", route=route)
        result["route"] = route
        return result

    def validate_sql(self, state: RagSqlGraphState) -> RagSqlGraphState:
        diagnostics = self._diagnostics(state)
        attempt = int(state.get("validation_attempt", 1))
        started = time.perf_counter()
        try:
            validated_plan = self.validator.validate(
                state["plan"],
                protected_parameters=state.get("protected_parameters") or {},
                resolved_filters=list(state.get("resolved_filters") or []),
            )
        except SqlValidationError as exc:
            duration_ms = (time.perf_counter() - started) * 1000.0
            validation_error = f"{type(exc).__name__}: {exc}"
            first_validation_error = state.get("first_validation_error") or validation_error
            append_stage(
                diagnostics,
                f"validate_sql_attempt_{attempt}",
                "error",
                duration_ms,
                {"error": validation_error},
            )
            repairs_used = attempt - 1
            if repairs_used >= self.validation_repair_count:
                final_exc: Exception = exc
                if first_validation_error != validation_error:
                    final_exc = SqlValidationError(
                        "SQL validation still failed after "
                        f"{repairs_used} repair attempt(s). Initial error: "
                        f"{first_validation_error}. Last error: {validation_error}."
                    )
                return self._failure(
                    state,
                    node="validate_sql",
                    error_code="sql_validation_failed",
                    exc=final_exc,
                    diagnostics=diagnostics,
                )
            append_graph_trace(diagnostics, node="validate_sql", route="repair")
            return {
                "diagnostics": diagnostics,
                "route": "repair",
                "validation_error": validation_error,
                "first_validation_error": first_validation_error,
            }
        except Exception as exc:
            return self._failure(
                state,
                node="validate_sql",
                error_code="sql_validation_failed",
                exc=exc,
                diagnostics=diagnostics,
            )

        duration_ms = (time.perf_counter() - started) * 1000.0
        append_stage(
            diagnostics,
            f"validate_sql_attempt_{attempt}",
            "done",
            duration_ms,
            {
                "referenced_objects": validated_plan.referenced_objects,
                "referenced_functions": validated_plan.referenced_functions,
                "placeholder_count": len(validated_plan.placeholder_names),
            },
        )
        diagnostics["validated_sql"] = validated_plan.model_dump(mode="json")
        append_graph_trace(diagnostics, node="validate_sql", route="execute")
        return {
            "diagnostics": diagnostics,
            "route": "execute",
            "validated_plan": validated_plan,
        }

    def repair_sql(self, state: RagSqlGraphState) -> RagSqlGraphState:
        diagnostics = self._diagnostics(state)
        repairs_used = int(state.get("validation_attempt", 1)) - 1
        started = time.perf_counter()
        try:
            repaired_plan = self.planner.repair_after_validation_failure(
                state["question"],
                analysis=state["analysis"],
                resolved_entities=list(state.get("resolved_filters") or []),
                protected_parameters=state.get("protected_parameters") or {},
                previous_plan=state["plan"],
                validation_error=str(state.get("validation_error") or "SQL validation failed."),
            )
        except RagSqlPlanningError as exc:
            return self._failure(
                state,
                node="repair_sql",
                error_code="sql_planning_failed",
                exc=exc,
                diagnostics=diagnostics,
            )
        except Exception as exc:
            return self._failure(
                state,
                node="repair_sql",
                error_code="sql_planning_failed",
                exc=exc,
                diagnostics=diagnostics,
            )

        duration_ms = (time.perf_counter() - started) * 1000.0
        append_stage(
            diagnostics,
            f"repair_sql_attempt_{repairs_used + 1}",
            "done",
            duration_ms,
            {
                "status": repaired_plan.status,
                "attempts": repaired_plan.attempts,
                "model": repaired_plan.model,
                "result_shape": repaired_plan.result_shape,
                "validation_error": state.get("validation_error"),
                **model_call_details(repaired_plan.ollama_calls),
            },
        )
        sql_plan_attempts = list(diagnostics.get("sql_plan_attempts") or [])
        sql_plan_attempts.append(repaired_plan.model_dump(mode="json"))
        diagnostics["sql_plan_attempts"] = sql_plan_attempts
        diagnostics["sql_plan"] = repaired_plan.model_dump(mode="json")

        result: RagSqlGraphState = {
            "diagnostics": diagnostics,
            "plan": repaired_plan,
            "validation_attempt": int(state.get("validation_attempt", 1)) + 1,
        }
        if repaired_plan.status == "needs_clarification":
            route = "terminal"
            result.update(
                {
                    "terminal_status": "needs_clarification",
                    "terminal_answer": repaired_plan.clarification_question
                    or "Clarification is required.",
                    "clarification_question": repaired_plan.clarification_question,
                }
            )
        elif repaired_plan.status == "unsupported":
            route = "terminal"
            result.update(
                {
                    "terminal_status": "unsupported",
                    "terminal_answer": repaired_plan.reason
                    or "The requested analysis is unsupported.",
                }
            )
        else:
            route = "validate"

        append_graph_trace(diagnostics, node="repair_sql", route=route)
        result["route"] = route
        return result

    def execute_sql(self, state: RagSqlGraphState) -> RagSqlGraphState:
        diagnostics = self._diagnostics(state)
        validated_plan = state["validated_plan"]
        try:
            execution = self.executor.execute(validated_plan)
            if validated_plan.result_shape in {"scalar", "row"} and execution.row_count > 1:
                raise SqlExecutionError(
                    f"{validated_plan.result_shape} result returned {execution.row_count} rows."
                )
        except SqlExecutionError as exc:
            return self._failure(
                state,
                node="execute_sql",
                error_code="sql_execution_failed",
                exc=exc,
                diagnostics=diagnostics,
            )
        except Exception as exc:
            return self._failure(
                state,
                node="execute_sql",
                error_code="sql_execution_failed",
                exc=exc,
                diagnostics=diagnostics,
            )

        append_stage(
            diagnostics,
            "execute_sql",
            "done",
            execution.duration_ms,
            {"row_count": execution.row_count, "truncated": execution.truncated},
        )
        append_graph_trace(diagnostics, node="execute_sql", route="extract")
        return {
            "diagnostics": diagnostics,
            "route": "extract",
            "execution": execution,
        }

    @staticmethod
    def _insufficient_decision(
        state: RagSqlGraphState,
        *,
        reason: str,
    ) -> DeterministicAnswerDecision:
        german = state["analysis"].language == "de"
        return DeterministicAnswerDecision(
            status="no_evidence",
            response_status="insufficient_info",
            answer=(
                "Die geprüften Belegdaten enthalten dafür nicht genügend Produktinformationen."
                if german
                else "The reviewed receipt data does not contain enough product information for that answer."
            ),
            reason=reason,
            evidence_rows=tuple(state.get("deterministic_answer").evidence_rows)
            if state.get("deterministic_answer")
            else (),
            supporting_item_ids=tuple(state.get("deterministic_answer").supporting_item_ids)
            if state.get("deterministic_answer")
            else (),
        )

    def extract_answer(self, state: RagSqlGraphState) -> RagSqlGraphState:
        diagnostics = self._diagnostics(state)
        started = time.perf_counter()
        try:
            decision = classify_rag_sql_outcome(
                state["execution"],
                state["validated_plan"],
                language=state["analysis"].language,
                question=state["question"],
                requested_operation=state["analysis"].requested_operation,
                resolved_entities=list(state.get("resolved_filters") or []),
            )
        except Exception as exc:
            return self._failure(
                state,
                node="extract_answer",
                error_code="result_formatting_failed",
                exc=exc,
                diagnostics=diagnostics,
            )

        duration_ms = (time.perf_counter() - started) * 1000.0
        fallback_available = self.answer_formatter is not None
        route = (
            "llm_format" if decision.status == "ambiguous" and fallback_available else "finalize"
        )
        formatting = {
            "deterministic_status": decision.status,
            "deterministic_reason": decision.reason,
            "fallback_available": fallback_available,
            "fallback_used": route == "llm_format",
            "validation_status": "not_run",
            "supporting_item_ids": list(decision.supporting_item_ids),
        }
        diagnostics["answer_formatting"] = formatting
        append_stage(
            diagnostics,
            "extract_answer",
            "done",
            duration_ms,
            {
                "deterministic_status": decision.status,
                "reason": decision.reason,
                "fallback_used": route == "llm_format",
                "evidence_row_count": len(decision.evidence_rows),
            },
        )
        append_graph_trace(diagnostics, node="extract_answer", route=route)
        return {
            "diagnostics": diagnostics,
            "route": route,
            "deterministic_answer": decision,
        }

    def format_answer_with_llm(self, state: RagSqlGraphState) -> RagSqlGraphState:
        diagnostics = self._diagnostics(state)
        formatter = self.answer_formatter
        if formatter is None:
            decision = self._insufficient_decision(
                state,
                reason="hybrid_formatter_unavailable",
            )
            append_graph_trace(diagnostics, node="format_answer_with_llm", route="finalize")
            return {
                "diagnostics": diagnostics,
                "route": "finalize",
                "deterministic_answer": decision,
            }

        try:
            result = formatter.format(
                question=state["question"],
                requested_operation=str(state["analysis"].requested_operation or ""),
                language=state["analysis"].language,
                rows=state["deterministic_answer"].evidence_rows,
                answer_instruction=state["validated_plan"].answer_instruction,
            )
        except AnswerFormattingError as exc:
            formatting = dict(diagnostics.get("answer_formatting") or {})
            formatting.update(
                {
                    "fallback_used": True,
                    "fallback_status": "error",
                    "validation_status": "not_run",
                    "fallback_error": f"{type(exc).__name__}: {exc}",
                }
            )
            diagnostics["answer_formatting"] = formatting
            append_stage(
                diagnostics,
                "format_answer_with_llm",
                "error",
                0.0,
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    **model_call_details(exc.ollama_calls),
                },
            )
            append_graph_trace(diagnostics, node="format_answer_with_llm", route="finalize")
            return {
                "diagnostics": diagnostics,
                "route": "finalize",
                "deterministic_answer": self._insufficient_decision(
                    state,
                    reason="hybrid_formatter_failed",
                ),
            }
        except Exception as exc:
            append_stage(
                diagnostics,
                "format_answer_with_llm",
                "error",
                0.0,
                {"error": f"{type(exc).__name__}: {exc}"},
            )
            append_graph_trace(diagnostics, node="format_answer_with_llm", route="finalize")
            return {
                "diagnostics": diagnostics,
                "route": "finalize",
                "deterministic_answer": self._insufficient_decision(
                    state,
                    reason="hybrid_formatter_failed",
                ),
            }

        formatting = dict(diagnostics.get("answer_formatting") or {})
        formatting.update(
            {
                "fallback_used": True,
                "fallback_model": result.model,
                "fallback_status": result.status,
                "fallback_attempts": result.attempts,
            }
        )
        diagnostics["answer_formatting"] = formatting
        append_stage(
            diagnostics,
            "format_answer_with_llm",
            "done",
            result.duration_ms,
            {
                "status": result.status,
                "model": result.model,
                "attempts": result.attempts,
                "value_count": len(result.values),
                **model_call_details(result.ollama_calls),
            },
        )
        append_graph_trace(diagnostics, node="format_answer_with_llm", route="validate_answer")
        return {
            "diagnostics": diagnostics,
            "route": "validate_answer",
            "llm_answer_result": result,
        }

    def validate_formatted_answer(self, state: RagSqlGraphState) -> RagSqlGraphState:
        diagnostics = self._diagnostics(state)
        started = time.perf_counter()
        try:
            validation = validate_answer_formatter_result(
                state["llm_answer_result"],
                requested_operation=str(state["analysis"].requested_operation or ""),
                rows=state["deterministic_answer"].evidence_rows,
            )
        except Exception as exc:
            validation = AnswerValidationResult(
                status="invalid",
                reason=f"validator_exception:{type(exc).__name__}:{exc}",
            )

        duration_ms = (time.perf_counter() - started) * 1000.0
        formatting = dict(diagnostics.get("answer_formatting") or {})
        formatting.update(
            {
                "validation_status": validation.status,
                "validation_reason": validation.reason,
                "supporting_item_ids": validation.supporting_item_ids,
                "evidence_fields": validation.evidence_fields,
                "values": validation.values,
            }
        )
        diagnostics["answer_formatting"] = formatting
        append_stage(
            diagnostics,
            "validate_formatted_answer",
            "done" if validation.status == "valid" else "rejected",
            duration_ms,
            {
                "validation_status": validation.status,
                "reason": validation.reason,
                "supporting_item_ids": validation.supporting_item_ids,
            },
        )
        append_graph_trace(diagnostics, node="validate_formatted_answer", route="finalize")
        return {
            "diagnostics": diagnostics,
            "route": "finalize",
            "answer_validation": validation,
        }

    def finalize_response(self, state: RagSqlGraphState) -> RagSqlGraphState:
        diagnostics = self._diagnostics(state)
        decision = state["deterministic_answer"]
        validation = state.get("answer_validation")
        if validation is not None and validation.status == "valid":
            response_status = "completed"
            answer = render_validated_answer(
                validation,
                requested_operation=str(state["analysis"].requested_operation or ""),
                language=state["analysis"].language,
            )
        elif validation is not None:
            fallback = self._insufficient_decision(
                state,
                reason=f"hybrid_validation_{validation.status}",
            )
            response_status = fallback.response_status
            answer = fallback.answer
        else:
            response_status = decision.response_status
            answer = decision.answer

        append_graph_trace(diagnostics, node="finalize_response", route="terminal")
        diagnostics["duration_ms"] = (time.perf_counter() - state["started_at_perf"]) * 1000.0
        attach_model_call_summary(diagnostics)
        response = RagSqlResponse(
            question=state["question"],
            status=response_status,
            answer=answer,
            data=state["execution"],
            diagnostics=diagnostics,
        )
        return {"diagnostics": diagnostics, "final_response": response}

    def terminal(self, state: RagSqlGraphState) -> RagSqlGraphState:
        diagnostics = self._diagnostics(state)
        append_graph_trace(diagnostics, node="terminal", route="terminal")
        response = terminal_response(
            state["question"],
            status=state["terminal_status"],
            answer=state["terminal_answer"],
            clarification_question=state.get("clarification_question"),
            diagnostics=diagnostics,
            started=state["started_at_perf"],
        )
        return {"diagnostics": diagnostics, "final_response": response}

    def failure(self, state: RagSqlGraphState) -> RagSqlGraphState:
        diagnostics = self._diagnostics(state)
        append_graph_trace(diagnostics, node="failure", route="terminal")
        response = error_response(
            state["question"],
            state.get("error_code") or "rag_sql_graph_failed",
            state.get("exception") or RuntimeError("RAG-SQL graph failed."),
            diagnostics,
            state["started_at_perf"],
        )
        return {"diagnostics": diagnostics, "final_response": response}


def _not_found_answer(language: str, field: str, value: object) -> str:
    rendered = str(value)
    if language == "de":
        labels = {
            "product": "gekauften Produkte",
            "merchant": "Händler",
            "category": "Kategorien",
            "payment_method": "Zahlungsarten",
            "currency": "Währungen",
        }
        label = labels.get(field, "Werte")
        return f"Keine passenden {label} für „{rendered}“ gefunden."
    labels = {
        "product": "purchased products",
        "merchant": "merchants",
        "category": "categories",
        "payment_method": "payment methods",
        "currency": "currencies",
    }
    label = labels.get(field, "values")
    return f"No matching {label} for '{rendered}' were found."
