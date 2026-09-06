from __future__ import annotations

import json
from pathlib import Path

import interpretation_outcome_support as support
import pytest

from receipt_intelligence.application.ports.llm import GenerationProviderUnavailableError
from receipt_intelligence.application.ports.multimodal import (
    MultimodalGenerationRequest,
    MultimodalGenerationResult,
)
from receipt_intelligence.interpretation import (
    DocumentInterpreter,
    InterpretationExecutionLimits,
    InterpretationValidationStatus,
    plan_interpretation_windows,
    run_document_interpretation,
)


class _SequenceGateway:
    def __init__(self, responses: list[dict[str, object] | Exception]) -> None:
        self._responses = responses
        self.requests: list[MultimodalGenerationRequest] = []
        self.page_counts: list[int] = []

    def generate(self, request: MultimodalGenerationRequest) -> MultimodalGenerationResult:
        self.requests.append(request)
        self.page_counts.append(len(request.image_paths))
        response = self._responses[len(self.requests) - 1]
        if isinstance(response, Exception):
            raise response
        return MultimodalGenerationResult(text=json.dumps(response))


def _response(
    start_page: int,
    end_page: int,
    *,
    evidence_page: int | None = None,
    unsupported: bool = False,
) -> dict[str, object]:
    evidence_page = start_page if evidence_page is None else evidence_page
    classification: dict[str, object]
    if unsupported:
        classification = {"status": "unsupported", "reason": "Not classified in this window."}
    else:
        classification = {
            "status": "classified",
            "dimensions": [
                {
                    "dimension_key": "record_kind",
                    "option_paths": [["supported_record"]],
                    "evidence_refs": ["e-1"],
                }
            ],
            "evidence_refs": ["e-1"],
        }
    return {
        "classification": classification,
        "document_map": {
            "nodes": [{"node_id": "node-1", "label": "Window section", "evidence_refs": ["e-1"]}]
        },
        "mentions": [
            {"mention_id": "mention-1", "observed_text": "Acme", "evidence_refs": ["e-1"]}
        ],
        "candidate_entities": [
            {
                "candidate_entity_id": "entity-1",
                "entity_type": "stated_party",
                "mention_refs": ["mention-1"],
                "evidence_refs": ["e-1"],
            }
        ],
        "candidate_facts": [
            {
                "fact_id": "fact-1",
                "subject": {"kind": "candidate_entity", "candidate_entity_id": "entity-1"},
                "predicate": support.FIELD_KEY,
                "object": {
                    "kind": "literal",
                    "literal_type": "text",
                    "observed": "same candidate value",
                },
                "evidence_refs": ["e-1"],
            }
        ],
        "evidence": [
            {
                "evidence_id": "e-1",
                "source_id": support.SOURCE_ID,
                "page": {"page_number": evidence_page},
            }
        ],
        "review_signals": [],
        "page_handling": [
            {
                "page_range": {"start_page": start_page, "end_page": end_page},
                "state": "interpreted",
            }
        ],
    }


def _interpreter(gateway: _SequenceGateway, *, max_calls: int = 2) -> DocumentInterpreter:
    return DocumentInterpreter(
        gateway=gateway,
        model="generic-multimodal-model",
        source_limits=support.limits(max_pages=4),
        execution_limits=InterpretationExecutionLimits(
            max_pages_per_window=2,
            max_windows=max_calls,
        ),
    )


def test_plans_deterministic_contiguous_bounded_windows() -> None:
    windows = plan_interpretation_windows(
        5,
        limits=InterpretationExecutionLimits(max_pages_per_window=2, max_windows=3),
    )

    assert [(window.start_page, window.end_page) for window in windows] == [
        (1, 2),
        (3, 4),
        (5, 5),
    ]


def test_aggregates_windows_with_original_pages_and_collision_safe_local_ids(
    tmp_path: Path,
) -> None:
    source_path, media_type = support.write_source(tmp_path, pages=3)
    gateway = _SequenceGateway([_response(1, 2), _response(3, 3)])

    outcome = run_document_interpretation(
        support.interpretation_request(media_type=media_type),
        source_path,
        gateway=gateway,
        model="generic-multimodal-model",
        source_limits=support.limits(max_pages=4),
        execution_limits=InterpretationExecutionLimits(max_pages_per_window=2, max_windows=2),
    )
    interpretation = outcome.interpretation

    assert gateway.page_counts == [2, 1]
    assert "original source pages 1-2" in gateway.requests[0].prompt
    assert "original source pages 3-3" in gateway.requests[1].prompt
    assert outcome.validation.status is InterpretationValidationStatus.VALID
    assert [item.page.page_number for item in interpretation.evidence if item.page] == [1, 3]
    assert len({item.evidence_id for item in interpretation.evidence}) == 2
    assert len({item.candidate_entity_id for item in interpretation.candidate_entities}) == 2
    assert len(interpretation.candidate_facts) == 2
    assert {
        item.subject.candidate_entity_id
        for item in interpretation.candidate_facts
        if item.subject.kind == "candidate_entity"
    } == {item.candidate_entity_id for item in interpretation.candidate_entities}


def test_conflicting_window_classifications_are_not_silently_selected(tmp_path: Path) -> None:
    source_path, media_type = support.write_source(tmp_path, pages=3)
    gateway = _SequenceGateway([_response(1, 2), _response(3, 3, unsupported=True)])

    outcome = _interpreter(gateway).interpret(
        support.interpretation_request(media_type=media_type),
        source_path,
    )

    assert outcome.validation.status is InterpretationValidationStatus.REVIEW_REQUIRED
    assert [issue.code for issue in outcome.validation.issues] == ["WINDOW_CLASSIFICATION_CONFLICT"]
    assert len(outcome.interpretation.candidate_entities) == 2


def test_window_scoped_evidence_violation_cannot_look_complete(tmp_path: Path) -> None:
    source_path, media_type = support.write_source(tmp_path, pages=3)
    gateway = _SequenceGateway([_response(1, 2), _response(3, 3, evidence_page=1)])

    outcome = _interpreter(gateway).interpret(
        support.interpretation_request(media_type=media_type),
        source_path,
    )

    assert outcome.validation.status is InterpretationValidationStatus.INVALID
    assert "WINDOW_NONEXISTENT_EVIDENCE_PAGE" in {issue.code for issue in outcome.validation.issues}


def test_provider_call_limit_is_checked_before_generation(tmp_path: Path) -> None:
    source_path, media_type = support.write_source(tmp_path, pages=3)
    gateway = _SequenceGateway([_response(1, 2)])

    with pytest.raises(ValueError, match="more windows"):
        _interpreter(gateway, max_calls=1).interpret(
            support.interpretation_request(media_type=media_type),
            source_path,
        )

    assert gateway.requests == []


def test_partial_provider_failure_propagates_without_an_aggregate_result(tmp_path: Path) -> None:
    source_path, media_type = support.write_source(tmp_path, pages=3)
    failure = GenerationProviderUnavailableError("provider unavailable")
    gateway = _SequenceGateway([_response(1, 2), failure])

    with pytest.raises(GenerationProviderUnavailableError) as raised:
        _interpreter(gateway).interpret(
            support.interpretation_request(media_type=media_type),
            source_path,
        )

    assert raised.value is failure
    assert len(gateway.requests) == 2
