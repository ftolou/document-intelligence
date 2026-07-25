from __future__ import annotations

import json

import pytest

from receipt_intelligence.application.ports.llm import GenerationRequest, GenerationResult
from receipt_intelligence.application.query_diagnostics import capture_query_diagnostics
from receipt_intelligence.rag.candidate_resolver import (
    CandidateResolutionError,
    CandidateResolver,
    CandidateResolverConfig,
    build_candidate_records,
)
from receipt_intelligence.rag.models import SemanticItemMatch


def _match(
    item_id: int,
    description: str,
    *,
    item_ids: list[int] | None = None,
    rank: int = 1,
    category: str = "clothing_shoes",
    semantic_description: str | None = None,
    merchant: str = "Store",
) -> SemanticItemMatch:
    ids = item_ids or [item_id]
    return SemanticItemMatch(
        rank=rank,
        item_id=item_id,
        item_ids=ids,
        occurrence_count=len(ids),
        receipt_id=1,
        description=description,
        normalized_description=description.casefold(),
        category=category,
        semantic_description=semantic_description,
        merchant=merchant,
        parser_item_type="item",
        line_total=10.0,
        unit_price=10.0,
        currency="EUR",
        similarity=0.5,
        vector_rank=rank,
        lexical_rank=rank,
        lexical_score=10.0,
        fusion_score=0.04,
    )


class FakeGenerate:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, **kwargs: object) -> str:
        self.calls.append(dict(kwargs))
        return self.responses.pop(0)


class FakeGateway:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        return GenerationResult(text=self.responses.pop(0))


def _config(**changes: object) -> CandidateResolverConfig:
    values = {
        "enabled": True,
        "ollama_url": "http://ollama.test",
        "model": "test-model",
        "retry_count": 0,
        "maximum_candidates": 12,
    }
    values.update(changes)
    return CandidateResolverConfig(**values)


def test_build_candidate_records_filters_placeholders_and_caps_candidates() -> None:
    records = build_candidate_records(
        [
            _match(1, "None", rank=1),
            _match(2, "HS-Halbschuhe", rank=2),
            _match(3, "KRAWATTE", rank=3),
        ],
        maximum_candidates=1,
    )

    assert [record.candidate_id for record in records] == ["c001"]
    assert records[0].description == "HS-Halbschuhe"


def test_candidate_records_preserve_reviewed_semantic_description() -> None:
    records = build_candidate_records(
        [
            _match(
                127,
                "VITTEL",
                category="Food & Groceries / beverages",
                semantic_description="Vittel is a brand of mineral water.",
            )
        ]
    )

    assert records[0].semantic_description == "Vittel is a brand of mineral water."


def test_resolver_maps_selected_identity_to_all_occurrence_ids() -> None:
    gateway = FakeGateway(
        [
            json.dumps(
                {
                    "schema_version": "rag_candidate_resolution_v2",
                    "status": "resolved",
                    "semantic_entity": "Schuhe",
                    "decisions": [
                        {
                            "candidate_id": "c001",
                            "decision": "selected",
                            "evidence_strength": "explicit",
                            "evidence": "'Halbschuhe' directly names a footwear subtype.",
                        },
                        {
                            "candidate_id": "c002",
                            "decision": "rejected",
                            "evidence_strength": "unrelated",
                            "evidence": "'KRAWATTE' denotes a necktie, not footwear.",
                        },
                    ],
                    "clarification_question": None,
                    "notes": [],
                }
            )
        ]
    )
    resolver = CandidateResolver(_config(), llm_gateway=gateway)

    result = resolver.resolve(
        "Schuhe",
        [
            _match(10, "HS-Halbschuhe", item_ids=[10, 11]),
            _match(20, "KRAWATTE", rank=2),
        ],
    )

    assert result.status == "resolved"
    assert result.selected_candidate_ids == ["c001"]
    assert result.selected_item_ids == [10, 11]
    assert result.decisions[0].evidence_strength == "explicit"
    assert result.rejected_item_ids == [20]
    assert result.attempts == 1
    assert len(gateway.requests) == 1


def test_resolver_classifies_ambiguous_description_as_uncertain() -> None:
    generator = FakeGenerate(
        [
            json.dumps(
                {
                    "schema_version": "rag_candidate_resolution_v2",
                    "status": "needs_clarification",
                    "semantic_entity": "Schuhe",
                    "decisions": [
                        {
                            "candidate_id": "c001",
                            "decision": "uncertain",
                            "evidence_strength": "ambiguous",
                            "evidence": (
                                "'SCHUHENGEL, GRAU paar' is shoe-related but does not "
                                "unambiguously identify footwear rather than an accessory."
                            ),
                        },
                        {
                            "candidate_id": "c002",
                            "decision": "selected",
                            "evidence_strength": "explicit",
                            "evidence": "'Halbschuhe' explicitly identifies footwear.",
                        },
                    ],
                    "clarification_question": (
                        "Should ambiguous shoe-related products or accessories also be included?"
                    ),
                    "notes": [],
                }
            )
        ]
    )

    result = CandidateResolver(_config(), generate=generator).resolve(
        "Schuhe",
        [
            _match(84, "SCHUHENGEL, GRAU paar", merchant="WERDICH - DAS SCHUHHAUS"),
            _match(126, "HS-Halbschuhe", rank=2, merchant="Modepark Röther"),
        ],
        user_question="Wie viel habe ich für Schuhe ausgegeben?",
    )

    assert result.status == "needs_clarification"
    assert result.selected_item_ids == [126]
    assert result.uncertain_item_ids == [84]
    assert result.decisions[0].evidence_strength == "ambiguous"
    assert result.clarification_question is not None


def test_resolver_retries_invalid_evidence_mapping_without_fallback() -> None:
    invalid = json.dumps(
        {
            "schema_version": "rag_candidate_resolution_v2",
            "status": "resolved",
            "semantic_entity": "Schuhe",
            "decisions": [
                {
                    "candidate_id": "c001",
                    "decision": "selected",
                    "evidence_strength": "ambiguous",
                    "evidence": "The description may denote an accessory.",
                }
            ],
            "clarification_question": None,
            "notes": [],
        }
    )
    generator = FakeGenerate([invalid, invalid])
    resolver = CandidateResolver(_config(retry_count=1), generate=generator)

    with pytest.raises(CandidateResolutionError, match="failed after 2 attempt"):
        resolver.resolve("Schuhe", [_match(1, "SCHUHENGEL, GRAU paar")])

    assert len(generator.calls) == 2
    assert "Previous response validation error" in str(generator.calls[1]["prompt"])


def test_resolver_records_each_validation_failure_for_opt_in_query_log() -> None:
    invalid = json.dumps(
        {
            "schema_version": "rag_candidate_resolution_v2",
            "status": "resolved",
            "semantic_entity": "Schuhe",
            "decisions": [
                {
                    "candidate_id": "c001",
                    "decision": "strong_contextual",
                    "evidence_strength": "strong_contextual",
                    "evidence": "Invalid decision vocabulary.",
                }
            ],
            "clarification_question": None,
            "notes": [],
        }
    )
    resolver = CandidateResolver(
        _config(retry_count=1),
        generate=FakeGenerate([invalid, invalid]),
    )

    with capture_query_diagnostics(enabled=True) as diagnostics:
        with pytest.raises(CandidateResolutionError):
            resolver.resolve("Schuhe", [_match(1, "HS-Halbschuhe")])

    failures = [
        record
        for record in diagnostics.snapshot()
        if record["event"] == "rag.candidate_resolution.attempt_failed"
    ]
    assert [record["attempt"] for record in failures] == [1, 2]
    assert "strong_contextual" in failures[0]["error"]


def test_resolver_retries_legacy_confidence_output_without_fallback() -> None:
    legacy = json.dumps(
        {
            "schema_version": "rag_candidate_resolution_v1",
            "status": "resolved",
            "semantic_entity": "Schuhe",
            "decisions": [
                {
                    "candidate_id": "c001",
                    "decision": "selected",
                    "confidence": 1.0,
                    "reason": "Legacy confidence output.",
                }
            ],
            "clarification_question": None,
            "notes": [],
        }
    )
    generator = FakeGenerate([legacy, legacy])
    resolver = CandidateResolver(_config(retry_count=1), generate=generator)

    with pytest.raises(CandidateResolutionError, match="failed after 2 attempt"):
        resolver.resolve("Schuhe", [_match(1, "HS-Halbschuhe")])

    assert len(generator.calls) == 2


def test_empty_or_placeholder_candidate_set_returns_not_found_without_llm() -> None:
    generator = FakeGenerate([])
    resolver = CandidateResolver(_config(), generate=generator)

    result = resolver.resolve("Schuhe", [_match(1, "Product Purchase")])

    assert result.status == "not_found"
    assert result.candidate_count == 0
    assert generator.calls == []
