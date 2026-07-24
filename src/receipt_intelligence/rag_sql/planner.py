"""Structured LLM SQL planning for the isolated RAG-SQL strategy."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from receipt_intelligence.application.generation import (
    LegacyGenerateFunction,
    invoke_generation,
)
from receipt_intelligence.application.llm_json import parse_json_from_llm
from receipt_intelligence.application.ports.llm import (
    GenerationRequest,
    LlmGateway,
    ModelCallMetrics,
)
from receipt_intelligence.prompts import render_prompt_template
from receipt_intelligence.rag_sql.models import (
    QuestionAnalysisResult,
    RagSqlPlanPayload,
    RagSqlPlanResult,
    ResolvedSemanticEntity,
)
from receipt_intelligence.rag_sql.schema_catalog import (
    DEFAULT_SCHEMA_CATALOG,
    StaticSchemaCatalog,
)


class RagSqlPlanningError(RuntimeError):
    """Raised when the LLM cannot produce a valid structured SQL plan."""

    def __init__(
        self,
        message: str,
        *,
        ollama_calls: list[ModelCallMetrics] | None = None,
    ) -> None:
        super().__init__(message)
        self.ollama_calls = list(ollama_calls or [])


@dataclass(frozen=True)
class RagSqlPlannerConfig:
    enabled: bool = True
    ollama_url: str = "http://localhost:11434"
    model: str = "gemma4"
    num_ctx: int = 6144
    num_predict: int = 2048
    timeout_seconds: float = 120.0
    retry_count: int = 1
    format_json: bool = True
    keep_alive: str | None = None
    maximum_rows: int = 100

    def __post_init__(self) -> None:
        if not str(self.ollama_url or "").strip():
            raise ValueError("ollama_url must not be empty.")
        if not str(self.model or "").strip():
            raise ValueError("model must not be empty.")
        if self.num_ctx <= 0 or self.num_predict <= 0:
            raise ValueError("num_ctx and num_predict must be positive.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if self.retry_count < 0:
            raise ValueError("retry_count must not be negative.")
        if self.maximum_rows <= 0 or self.maximum_rows > 1000:
            raise ValueError("maximum_rows must be between 1 and 1000.")

class RagSqlPlanner:
    def __init__(
        self,
        config: RagSqlPlannerConfig,
        *,
        schema_catalog: StaticSchemaCatalog = DEFAULT_SCHEMA_CATALOG,
        llm_gateway: LlmGateway | None = None,
        generate: LegacyGenerateFunction | None = None,
    ) -> None:
        self.config = config
        self.schema_catalog = schema_catalog
        self.llm_gateway = llm_gateway
        self.generate = generate

    def plan(
        self,
        question: str,
        *,
        analysis: QuestionAnalysisResult,
        resolved_entities: Sequence[ResolvedSemanticEntity],
        protected_parameters: Mapping[str, int],
        previous_plan: RagSqlPlanResult | None = None,
        validation_error: str | None = None,
    ) -> RagSqlPlanResult:
        normalized_question = " ".join(str(question or "").split()).strip()
        if not normalized_question:
            raise ValueError("question must not be empty.")
        if not self.config.enabled:
            raise RagSqlPlanningError("RAG-SQL planning is disabled.")
        if (previous_plan is None) != (validation_error is None):
            raise ValueError(
                "previous_plan and validation_error must either both be provided or both be omitted."
            )

        resolved_payload = []
        for entity in resolved_entities:
            parameter_names = [
                name
                for name in protected_parameters
                if name.startswith(f"{entity.entity_id}_item_")
            ]
            resolved_payload.append(
                {
                    "entity_id": entity.entity_id,
                    "search_text": entity.search_text,
                    "status": entity.status,
                    "selected_item_ids": entity.selected_item_ids,
                    "protected_item_parameters": {
                        name: protected_parameters[name] for name in sorted(parameter_names)
                    },
                }
            )

        started = time.perf_counter()
        previous_error: str | None = None
        last_error: Exception | None = None
        attempts = max(1, self.config.retry_count + 1)
        ollama_calls: list[ModelCallMetrics] = []
        previous_raw_response: str | None = None
        validation_repair_block = ""
        if previous_plan is not None and validation_error is not None:
            previous_payload = previous_plan.model_dump(
                mode="json",
                include=set(RagSqlPlanPayload.model_fields),
            )
            validation_repair_block = (
                "A previous complete SQL plan passed JSON/schema validation but failed "
                "deterministic SQL validation.\n\n"
                "Deterministic validation error:\n"
                f"{validation_error}\n\n"
                "Previous complete SQL plan:\n"
                f"{json.dumps(previous_payload, ensure_ascii=False, indent=2)}\n\n"
                "Return a complete replacement JSON plan. Correct the exact validation "
                "failure without changing the analytical meaning. Preserve every protected "
                "item-ID parameter name and value exactly. For grouped_rows, preserve all "
                "possible groups by using a deterministic ORDER BY and LIMIT "
                f"{self.config.maximum_rows} unless the original question explicitly requests "
                "fewer groups. Never use LIMIT 1 merely to satisfy validation. Do not explain "
                "the repair outside the JSON object."
            )

        for attempt in range(1, attempts + 1):
            retry_block = ""
            if previous_error:
                previous_response_block = (
                    f"\nPrevious invalid JSON response:\n{previous_raw_response}\n"
                    if previous_raw_response
                    else ""
                )
                retry_block = (
                    "The previous response failed the strict plan contract.\n"
                    "Exact validation error:\n"
                    f"{previous_error}\n"
                    f"{previous_response_block}"
                    "Return a complete corrected JSON plan. Keep result_entity as one "
                    "lowercase snake_case identifier. display_columns must contain only "
                    "SELECT output names or aliases, never SQL expressions such as COUNT(*). "
                    "Preserve all protected parameters exactly."
                )
            prompt = render_prompt_template(
                "rag_sql_planner.txt",
                TODAY=datetime.now(ZoneInfo("Europe/Berlin")).date().isoformat(),
                MAX_ROWS=self.config.maximum_rows,
                SCHEMA_CONTEXT=self.schema_catalog.render_for_prompt(),
                ANALYSIS_JSON=json.dumps(
                    analysis.model_dump(mode="json"), ensure_ascii=False, indent=2
                ),
                RESOLVED_ENTITIES_JSON=json.dumps(resolved_payload, ensure_ascii=False, indent=2),
                QUESTION=normalized_question,
                VALIDATION_REPAIR_BLOCK=validation_repair_block,
                RETRY_BLOCK=retry_block,
            )
            try:
                generation = invoke_generation(
                    request=GenerationRequest(
                        model=self.config.model,
                        prompt=prompt,
                        operation="rag_sql_planning",
                        attempt=attempt,
                        num_ctx=self.config.num_ctx,
                        num_predict=self.config.num_predict,
                        temperature=0.0,
                        keep_alive=self.config.keep_alive,
                        timeout_seconds=self.config.timeout_seconds,
                        format_json=self.config.format_json,
                    ),
                    gateway=self.llm_gateway,
                    legacy_generate=self.generate,
                    legacy_base_url=self.config.ollama_url,
                )
                if generation.metrics is not None:
                    ollama_calls.append(generation.metrics)
                previous_raw_response = generation.text[:12000]
                payload = RagSqlPlanPayload.model_validate(parse_json_from_llm(generation))
                _validate_protected_parameters(payload, protected_parameters)
                return RagSqlPlanResult(
                    **payload.model_dump(mode="python"),
                    model=self.config.model,
                    attempts=attempt,
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                    ollama_calls=ollama_calls,
                )
            except Exception as exc:
                last_error = exc
                previous_error = f"{type(exc).__name__}: {exc}"

        duration_ms = (time.perf_counter() - started) * 1000.0
        raise RagSqlPlanningError(
            "RAG-SQL planning failed after "
            f"{attempts} attempt(s) in {duration_ms:.1f} ms: "
            f"{type(last_error).__name__ if last_error else 'UnknownError'}: {last_error}",
            ollama_calls=ollama_calls,
        ) from last_error

    def repair_after_validation_failure(
        self,
        question: str,
        *,
        analysis: QuestionAnalysisResult,
        resolved_entities: Sequence[ResolvedSemanticEntity],
        protected_parameters: Mapping[str, int],
        previous_plan: RagSqlPlanResult,
        validation_error: str,
    ) -> RagSqlPlanResult:
        """Regenerate a complete plan from deterministic validator feedback."""

        validation_error_text = str(validation_error or "").strip()
        if not validation_error_text:
            raise ValueError("validation_error must not be empty.")
        if previous_plan.status != "ready":
            raise ValueError("Only a ready SQL plan can be repaired.")
        return self.plan(
            question,
            analysis=analysis,
            resolved_entities=resolved_entities,
            protected_parameters=protected_parameters,
            previous_plan=previous_plan,
            validation_error=validation_error_text,
        )


def build_protected_item_parameters(
    entities: Sequence[ResolvedSemanticEntity],
) -> dict[str, int]:
    """Create deterministic app-owned bindings for resolved product IDs."""

    parameters: dict[str, int] = {}
    for entity in entities:
        if entity.status != "resolved":
            continue
        for index, item_id in enumerate(entity.selected_item_ids):
            parameters[f"{entity.entity_id}_item_{index}"] = int(item_id)
    return parameters


def _validate_protected_parameters(
    payload: RagSqlPlanPayload,
    protected_parameters: Mapping[str, int],
) -> None:
    if payload.status != "ready":
        return
    unexpected_protected = sorted(
        name
        for name in payload.parameters
        if name.startswith("e") and "_item_" in name and name not in protected_parameters
    )
    if unexpected_protected:
        raise ValueError(
            f"The SQL plan introduced unknown protected item parameters: {unexpected_protected}."
        )
    for name, expected_value in protected_parameters.items():
        if name not in payload.parameters:
            raise ValueError(f"The SQL plan omitted protected parameter {name!r}.")
        actual_value = payload.parameters[name]
        if isinstance(actual_value, bool) or actual_value != expected_value:
            raise ValueError(
                f"Protected parameter {name!r} changed from {expected_value!r} to {actual_value!r}."
            )


__all__ = [
    "RagSqlPlanner",
    "RagSqlPlannerConfig",
    "RagSqlPlanningError",
    "build_protected_item_parameters",
]
