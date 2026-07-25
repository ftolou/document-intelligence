"""LLM-based precision layer for hybrid receipt-item retrieval.

The retriever proposes a small high-recall candidate set. This resolver asks an
LLM for an evidence-based classification of every candidate and returns exact
application-owned item IDs.
There is intentionally no deterministic semantic fallback: invalid model output
raises ``CandidateResolutionError`` after the configured attempts.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

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
from receipt_intelligence.application.query_diagnostics import record_query_diagnostic
from receipt_intelligence.rag.candidate_models import (
    CandidateRecord,
    CandidateResolutionBundle,
    CandidateResolutionPayload,
    CandidateResolutionResult,
)
from receipt_intelligence.rag.candidate_prompt import build_candidate_resolution_prompt
from receipt_intelligence.rag.item_documents import is_indexable_description
from receipt_intelligence.rag.models import SemanticItemMatch, SemanticItemSearchResult


class CandidateResolutionError(RuntimeError):
    """Raised when the LLM cannot produce a valid candidate resolution."""

    def __init__(
        self,
        message: str,
        *,
        ollama_calls: list[ModelCallMetrics] | None = None,
    ) -> None:
        super().__init__(message)
        self.ollama_calls = list(ollama_calls or [])


@dataclass(frozen=True)
class CandidateResolverConfig:
    enabled: bool = True
    ollama_url: str = "http://localhost:11434"
    model: str = "gemma4"
    num_ctx: int = 4096
    num_predict: int = 1536
    timeout_seconds: float = 120.0
    retry_count: int = 1
    format_json: bool = True
    keep_alive: str | None = None
    maximum_candidates: int = 12

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
        if self.maximum_candidates <= 0 or self.maximum_candidates > 50:
            raise ValueError("maximum_candidates must be between 1 and 50.")


class CandidateResolver:
    """Resolve retrieved product identities into selected SQL item IDs."""

    def __init__(
        self,
        config: CandidateResolverConfig,
        *,
        llm_gateway: LlmGateway | None = None,
        generate: LegacyGenerateFunction | None = None,
    ) -> None:
        self.config = config
        self.llm_gateway = llm_gateway
        self.generate = generate

    def resolve(
        self,
        semantic_entity: str,
        candidates: Sequence[SemanticItemMatch],
        *,
        user_question: str | None = None,
    ) -> CandidateResolutionResult:
        """Classify a candidate set with structured LLM output.

        Empty or unusable candidate sets return ``not_found`` without an LLM
        call. Invalid or incomplete LLM output is retried only as configured;
        no heuristic selection or fallback candidate is introduced.
        """

        entity = " ".join(str(semantic_entity or "").split()).strip()
        if not entity:
            raise ValueError("semantic_entity must not be empty.")
        if not self.config.enabled:
            raise CandidateResolutionError("Candidate resolution is disabled.")

        records = build_candidate_records(
            candidates,
            maximum_candidates=self.config.maximum_candidates,
        )
        if not records:
            return CandidateResolutionResult(
                status="not_found",
                semantic_entity=entity,
                candidate_count=0,
                decisions=[],
                selected_candidate_ids=[],
                uncertain_candidate_ids=[],
                rejected_candidate_ids=[],
                selected_item_ids=[],
                uncertain_item_ids=[],
                rejected_item_ids=[],
                model=self.config.model,
                attempts=0,
                duration_ms=0.0,
                notes=["No indexable candidates were available."],
            )

        started = time.perf_counter()
        previous_error: str | None = None
        last_error: Exception | None = None
        attempts = max(1, self.config.retry_count + 1)
        ollama_calls: list[ModelCallMetrics] = []

        for attempt in range(1, attempts + 1):
            prompt = build_candidate_resolution_prompt(
                entity,
                records,
                user_question=user_question,
                previous_error=previous_error,
            )
            try:
                generation = invoke_generation(
                    request=GenerationRequest(
                        model=self.config.model,
                        prompt=prompt,
                        operation="rag_candidate_resolution",
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
                payload = CandidateResolutionPayload.model_validate(parse_json_from_llm(generation))
                _validate_payload_against_candidates(payload, records, entity)
                return _map_resolution(
                    payload,
                    records,
                    model=self.config.model,
                    attempts=attempt,
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                    ollama_calls=ollama_calls,
                )
            except Exception as exc:
                last_error = exc
                previous_error = f"{type(exc).__name__}: {exc}"
                record_query_diagnostic(
                    "rag.candidate_resolution.attempt_failed",
                    {
                        "attempt": attempt,
                        "maximum_attempts": attempts,
                        "semantic_entity": entity,
                        "error": previous_error,
                    },
                )

        duration_ms = (time.perf_counter() - started) * 1000.0
        raise CandidateResolutionError(
            "LLM candidate resolution failed after "
            f"{attempts} attempt(s) in {duration_ms:.1f} ms: "
            f"{type(last_error).__name__ if last_error else 'UnknownError'}: {last_error}",
            ollama_calls=ollama_calls,
        ) from last_error

    def resolve_search_result(
        self,
        search_result: SemanticItemSearchResult,
        *,
        semantic_entity: str | None = None,
        user_question: str | None = None,
    ) -> CandidateResolutionBundle:
        """Resolve one retriever result and retain compact retrieval evidence."""

        entity = semantic_entity or search_result.query
        records = build_candidate_records(
            search_result.matches,
            maximum_candidates=self.config.maximum_candidates,
        )
        resolution = self.resolve(
            entity,
            search_result.matches,
            user_question=user_question,
        )
        return CandidateResolutionBundle(
            query=search_result.query,
            retrieved_candidate_count=len(search_result.matches),
            candidates=records,
            resolution=resolution,
        )


def build_candidate_records(
    matches: Sequence[SemanticItemMatch],
    *,
    maximum_candidates: int = 12,
) -> list[CandidateRecord]:
    """Convert ranked matches into compact stable records for the LLM."""

    if maximum_candidates <= 0:
        raise ValueError("maximum_candidates must be positive.")

    records: list[CandidateRecord] = []
    for match in matches:
        if len(records) >= maximum_candidates:
            break
        if not is_indexable_description(match.description):
            continue
        item_ids = sorted(set(match.item_ids or [match.item_id]))
        records.append(
            CandidateRecord(
                candidate_id=f"c{len(records) + 1:03d}",
                item_ids=item_ids,
                description=match.description,
                normalized_description=match.normalized_description,
                occurrence_count=len(item_ids),
                category=match.category,
                semantic_description=match.semantic_description,
                merchant=match.merchant,
                lexical_rank=match.lexical_rank,
                vector_rank=match.vector_rank,
                similarity=match.similarity,
                fusion_score=match.fusion_score,
            )
        )
    return records


def _validate_payload_against_candidates(
    payload: CandidateResolutionPayload,
    candidates: Sequence[CandidateRecord],
    semantic_entity: str,
) -> None:
    expected_ids = {candidate.candidate_id for candidate in candidates}
    returned_ids = [decision.candidate_id for decision in payload.decisions]
    returned_set = set(returned_ids)

    if len(returned_ids) != len(returned_set):
        raise ValueError("The LLM returned duplicate candidate decisions.")
    if returned_set != expected_ids:
        missing = sorted(expected_ids - returned_set)
        unknown = sorted(returned_set - expected_ids)
        raise ValueError(
            "The LLM must classify every provided candidate exactly once. "
            f"missing={missing}, unknown={unknown}."
        )
    if payload.semantic_entity.casefold() != semantic_entity.casefold():
        raise ValueError(
            "The LLM changed the semantic_entity instead of resolving the provided one."
        )


def _map_resolution(
    payload: CandidateResolutionPayload,
    candidates: Sequence[CandidateRecord],
    *,
    model: str,
    attempts: int,
    duration_ms: float,
    ollama_calls: list[ModelCallMetrics],
) -> CandidateResolutionResult:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    selected = [
        decision.candidate_id for decision in payload.decisions if decision.decision == "selected"
    ]
    uncertain = [
        decision.candidate_id for decision in payload.decisions if decision.decision == "uncertain"
    ]
    rejected = [
        decision.candidate_id for decision in payload.decisions if decision.decision == "rejected"
    ]

    return CandidateResolutionResult(
        status=payload.status,
        semantic_entity=payload.semantic_entity,
        candidate_count=len(candidates),
        decisions=payload.decisions,
        selected_candidate_ids=selected,
        uncertain_candidate_ids=uncertain,
        rejected_candidate_ids=rejected,
        selected_item_ids=_flatten_item_ids(selected, by_id),
        uncertain_item_ids=_flatten_item_ids(uncertain, by_id),
        rejected_item_ids=_flatten_item_ids(rejected, by_id),
        clarification_question=payload.clarification_question,
        model=model,
        attempts=attempts,
        duration_ms=duration_ms,
        ollama_calls=ollama_calls,
        notes=payload.notes,
    )


def _flatten_item_ids(
    candidate_ids: Sequence[str],
    candidates: dict[str, CandidateRecord],
) -> list[int]:
    return sorted(
        {item_id for candidate_id in candidate_ids for item_id in candidates[candidate_id].item_ids}
    )


__all__ = [
    "CandidateResolutionError",
    "CandidateResolver",
    "CandidateResolverConfig",
    "build_candidate_records",
]
