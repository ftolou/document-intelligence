from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import interpretation_outcome_support as support
import pytest

from receipt_intelligence.application.ports.llm import (
    GenerationProviderUnavailableError,
    MalformedGenerationError,
)
from receipt_intelligence.application.ports.multimodal import (
    MultimodalGenerationRequest,
    MultimodalGenerationResult,
)
from receipt_intelligence.interpretation import (
    CandidateEntityReference,
    InterpretationExecutionLimitError,
    InterpretationExecutionLimits,
    InterpretationValidationStatus,
    run_document_interpretation,
)


class _SequenceGateway:
    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self._responses = iter(responses)
        self.requests: list[MultimodalGenerationRequest] = []

    def generate(self, request: MultimodalGenerationRequest) -> MultimodalGenerationResult:
        self.requests.append(request)
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return MultimodalGenerationResult(text=json.dumps(response))


def _window_response(
    start_page: int,
    end_page: int,
    *,
    page_handling: bool = True,
    confidence: float | None = None,
) -> dict[str, Any]:
    dimension: dict[str, Any] = {
        "dimension_key": "record_kind",
        "option_paths": [["supported_record"]],
        "evidence_refs": ["e-1"],
    }
    if confidence is not None:
        dimension["confidence"] = confidence
    return {
        "classification": {
            "status": "classified",
            "dimensions": [dimension],
            "evidence_refs": ["e-1"],
        },
        "document_map": {
            "nodes": [{"node_id": "section-1", "label": "Statement", "evidence_refs": ["e-1"]}]
        },
        "mentions": [
            {"mention_id": "mention-1", "observed_text": "Acme", "evidence_refs": ["e-1"]}
        ],
        "candidate_entities": [
            {
                "candidate_entity_id": "entity-1",
                "entity_type": "stated_party",
                "display_name": "Acme",
                "mention_refs": ["mention-1"],
                "evidence_refs": ["e-1"],
            }
        ],
        "candidate_facts": [
            {
                "fact_id": "fact-1",
                "subject": {"kind": "candidate_entity", "candidate_entity_id": "entity-1"},
                "predicate": "stated_amount",
                "object": {
                    "kind": "literal",
                    "literal_type": "amount",
                    "observed": "12.50",
                    "normalization_status": "normalized",
                    "normalized": "12.50",
                    "currency": "EUR",
                },
                "evidence_refs": ["e-1"],
            }
        ],
        "evidence": [
            {
                "evidence_id": "e-1",
                "source_id": "document-1",
                "page": {"page_number": start_page},
                "excerpt": "Acme 12.50",
                "excerpt_provenance": "model_observed",
            }
        ],
        "review_signals": [],
        "page_handling": (
            [
                {
                    "page_range": {"start_page": start_page, "end_page": end_page},
                    "state": "interpreted",
                }
            ]
            if page_handling
            else []
        ),
    }


def _run(
    tmp_path: Path,
    gateway: _SequenceGateway,
    *,
    pages: int,
    pages_per_call: int,
    max_calls: int,
):
    source_path, media_type = support.write_source(tmp_path, pages=pages)
    return run_document_interpretation(
        support.interpretation_request(media_type=media_type),
        source_path,
        gateway=gateway,
        model="generic-multimodal-model",
        source_limits=support.limits(max_pages=pages),
        execution_limits=InterpretationExecutionLimits(
            max_pages_per_call=pages_per_call,
            max_provider_calls=max_calls,
        ),
    )


def test_small_document_keeps_one_provider_call(tmp_path: Path) -> None:
    gateway = _SequenceGateway([_window_response(1, 2)])

    outcome = _run(tmp_path, gateway, pages=2, pages_per_call=2, max_calls=3)

    assert outcome.validation.status is InterpretationValidationStatus.VALID
    assert len(gateway.requests) == 1
    assert len(gateway.requests[0].image_paths) == 2


def test_oversized_document_uses_deterministic_windows_and_namespaces_local_ids(
    tmp_path: Path,
) -> None:
    gateway = _SequenceGateway(
        [_window_response(1, 2), _window_response(3, 4), _window_response(5, 5)]
    )

    outcome = _run(tmp_path, gateway, pages=5, pages_per_call=2, max_calls=3)
    result = outcome.interpretation

    assert [len(request.image_paths) for request in gateway.requests] == [2, 2, 1]
    assert "original source pages 1 through 2 of\n5" in gateway.requests[0].prompt
    assert "original source pages 3 through 4 of\n5" in gateway.requests[1].prompt
    assert [item.evidence_id for item in result.evidence] == ["w1-e1", "w2-e1", "w3-e1"]
    assert [item.page.page_number for item in result.evidence if item.page] == [1, 3, 5]
    assert {item.source_id for item in result.evidence} == {"document-1"}
    assert [item.candidate_entity_id for item in result.candidate_entities] == [
        "w1-c1",
        "w2-c1",
        "w3-c1",
    ]
    assert [item.display_name for item in result.candidate_entities] == ["Acme", "Acme", "Acme"]
    assert all(
        isinstance(fact.subject, CandidateEntityReference) for fact in result.candidate_facts
    )
    assert [
        (item.page_range.start_page, item.page_range.end_page) for item in result.page_handling
    ] == [
        (1, 2),
        (3, 4),
        (5, 5),
    ]
    assert outcome.validation.status is InterpretationValidationStatus.VALID


def test_missing_window_page_accounting_is_explicitly_review_required(tmp_path: Path) -> None:
    gateway = _SequenceGateway(
        [_window_response(1, 2), _window_response(3, 4, page_handling=False)]
    )

    outcome = _run(tmp_path, gateway, pages=4, pages_per_call=2, max_calls=2)

    assert outcome.validation.status is InterpretationValidationStatus.REVIEW_REQUIRED
    assert [issue.code for issue in outcome.validation.issues] == ["MISSING_PAGE_COVERAGE"]


def test_total_provider_work_limit_is_checked_before_generation(tmp_path: Path) -> None:
    gateway = _SequenceGateway([])

    with pytest.raises(InterpretationExecutionLimitError):
        _run(tmp_path, gateway, pages=5, pages_per_call=2, max_calls=2)

    assert gateway.requests == []


def test_partial_window_failure_propagates_without_a_completed_result(tmp_path: Path) -> None:
    failure = GenerationProviderUnavailableError("provider unavailable")
    gateway = _SequenceGateway([_window_response(1, 2), failure])

    with pytest.raises(GenerationProviderUnavailableError) as raised:
        _run(tmp_path, gateway, pages=3, pages_per_call=2, max_calls=2)

    assert raised.value is failure
    assert len(gateway.requests) == 2


def test_window_cannot_claim_evidence_from_an_unsupplied_page(tmp_path: Path) -> None:
    second = _window_response(3, 4)
    second["evidence"][0]["page"] = {"page_number": 1}
    gateway = _SequenceGateway([_window_response(1, 2), second])

    with pytest.raises(MalformedGenerationError, match="outside the supplied bounded window"):
        _run(tmp_path, gateway, pages=4, pages_per_call=2, max_calls=2)


def test_semantically_disagreeing_classifications_are_not_silently_merged(
    tmp_path: Path,
) -> None:
    gateway = _SequenceGateway([_window_response(1, 2), _window_response(3, 4, confidence=0.5)])

    with pytest.raises(MalformedGenerationError, match="classifications disagree"):
        _run(tmp_path, gateway, pages=4, pages_per_call=2, max_calls=2)


@pytest.mark.parametrize(
    ("field", "value"),
    [("max_pages_per_call", 0), ("max_provider_calls", 0)],
)
def test_execution_limits_must_be_positive(field: str, value: int) -> None:
    values = {"max_pages_per_call": 1, "max_provider_calls": 1, field: value}

    with pytest.raises(ValueError, match=field):
        InterpretationExecutionLimits(**values)
