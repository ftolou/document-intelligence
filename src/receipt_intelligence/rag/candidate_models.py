"""Typed contracts for LLM-based semantic candidate resolution.

Hybrid retrieval deliberately optimizes recall. These models define the strict
boundary where an LLM classifies a small candidate set before any IDs are used
by later SQL planning or execution stages.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from receipt_intelligence.application.ports.llm import ModelCallMetrics

CANDIDATE_RESOLUTION_SCHEMA_VERSION = "rag_candidate_resolution_v2"

EvidenceStrength = Literal[
    "explicit",
    "strong_contextual",
    "ambiguous",
    "unrelated",
]


class StrictModel(BaseModel):
    """Reject unexpected fields at the LLM integration boundary."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CandidateRecord(StrictModel):
    """Compact, application-owned candidate shown to the LLM.

    The model receives stable candidate IDs instead of database IDs in its
    output contract. Database IDs remain application-owned metadata and are
    mapped only after the response passes validation.
    """

    candidate_id: str = Field(pattern=r"^c\d{3}$")
    item_ids: list[int] = Field(min_length=1)
    description: str = Field(min_length=1, max_length=2000)
    normalized_description: str | None = Field(default=None, max_length=2000)
    occurrence_count: int = Field(default=1, ge=1)
    category: str | None = Field(default=None, max_length=500)
    semantic_description: str | None = Field(default=None, max_length=2000)
    merchant: str | None = Field(default=None, max_length=500)
    lexical_rank: int | None = Field(default=None, ge=1)
    vector_rank: int | None = Field(default=None, ge=1)
    similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    fusion_score: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_item_ids(self) -> Self:
        if any(item_id <= 0 for item_id in self.item_ids):
            raise ValueError("item_ids must contain positive integers only.")
        if len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("item_ids must not contain duplicates.")
        if self.occurrence_count != len(self.item_ids):
            raise ValueError("occurrence_count must equal len(item_ids).")
        return self


class CandidateDecision(StrictModel):
    """One evidence-based LLM classification for one candidate.

    ``evidence_strength`` is a categorical evidence label, not a probability.
    The decision/evidence mapping is intentionally strict so ambiguous product
    descriptions cannot be silently promoted to selected candidates.
    """

    candidate_id: str = Field(pattern=r"^c\d{3}$")
    decision: Literal["selected", "uncertain", "rejected"]
    evidence_strength: EvidenceStrength
    evidence: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_evidence_policy(self) -> Self:
        allowed_by_decision: dict[str, set[str]] = {
            "selected": {"explicit", "strong_contextual"},
            "uncertain": {"ambiguous"},
            "rejected": {"unrelated"},
        }
        if self.evidence_strength not in allowed_by_decision[self.decision]:
            raise ValueError(
                "decision and evidence_strength are inconsistent: "
                f"decision={self.decision!r}, evidence_strength={self.evidence_strength!r}."
            )
        return self


class CandidateResolutionPayload(StrictModel):
    """Raw structured response expected from the LLM."""

    schema_version: Literal["rag_candidate_resolution_v2"] = CANDIDATE_RESOLUTION_SCHEMA_VERSION
    status: Literal["resolved", "needs_clarification", "not_found"]
    semantic_entity: str = Field(min_length=1, max_length=1000)
    decisions: list[CandidateDecision] = Field(default_factory=list)
    clarification_question: str | None = Field(default=None, max_length=1000)
    notes: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_status_semantics(self) -> Self:
        selected = [decision for decision in self.decisions if decision.decision == "selected"]
        uncertain = [decision for decision in self.decisions if decision.decision == "uncertain"]

        if self.status == "resolved" and not selected:
            raise ValueError("resolved status requires at least one selected candidate.")
        if self.status == "resolved" and uncertain:
            raise ValueError(
                "resolved status cannot contain uncertain candidates; request clarification instead."
            )
        if self.status == "needs_clarification":
            if not self.clarification_question:
                raise ValueError("needs_clarification status requires clarification_question.")
            if not uncertain:
                raise ValueError(
                    "needs_clarification status requires at least one uncertain candidate."
                )
        if self.status == "not_found" and (selected or uncertain):
            raise ValueError("not_found status cannot contain selected or uncertain candidates.")
        if self.status != "needs_clarification" and self.clarification_question:
            raise ValueError(
                "clarification_question is allowed only for needs_clarification status."
            )
        return self


class CandidateResolutionResult(StrictModel):
    """Validated resolver result with application-mapped SQL item IDs."""

    schema_version: Literal["rag_candidate_resolution_v2"] = CANDIDATE_RESOLUTION_SCHEMA_VERSION
    status: Literal["resolved", "needs_clarification", "not_found"]
    semantic_entity: str = Field(min_length=1, max_length=1000)
    candidate_count: int = Field(ge=0)
    decisions: list[CandidateDecision] = Field(default_factory=list)
    selected_candidate_ids: list[str] = Field(default_factory=list)
    uncertain_candidate_ids: list[str] = Field(default_factory=list)
    rejected_candidate_ids: list[str] = Field(default_factory=list)
    selected_item_ids: list[int] = Field(default_factory=list)
    uncertain_item_ids: list[int] = Field(default_factory=list)
    rejected_item_ids: list[int] = Field(default_factory=list)
    clarification_question: str | None = Field(default=None, max_length=1000)
    model: str | None = Field(default=None, max_length=200)
    attempts: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0, ge=0.0)
    ollama_calls: list[ModelCallMetrics] = Field(default_factory=list, max_length=20)
    notes: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_partition(self) -> Self:
        candidate_sets = [
            set(self.selected_candidate_ids),
            set(self.uncertain_candidate_ids),
            set(self.rejected_candidate_ids),
        ]
        if any(
            candidate_sets[left] & candidate_sets[right]
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise ValueError("Candidate decision sets must not overlap.")
        if sum(len(values) for values in candidate_sets) != self.candidate_count:
            raise ValueError("Candidate decision sets must cover candidate_count exactly.")
        return self


class CandidateResolutionBundle(StrictModel):
    """Retrieval diagnostics plus the final semantic resolution."""

    query: str = Field(min_length=1, max_length=2000)
    retrieved_candidate_count: int = Field(ge=0)
    candidates: list[CandidateRecord] = Field(default_factory=list)
    resolution: CandidateResolutionResult
