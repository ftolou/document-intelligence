from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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
    DocumentInterpretation,
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
    reason: str | None = None,
) -> dict[str, object]:
    evidence_page = start_page if evidence_page is None else evidence_page
    classification: dict[str, object]
    if unsupported:
        classification = {
            "status": "unsupported",
            "reason": reason or "Not classified in this window.",
        }
    else:
        classification = {
            "status": "classified",
            "reason": reason,
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


def _interpreter(
    gateway: _SequenceGateway,
    *,
    max_calls: int = 2,
    max_pages_per_window: int = 2,
) -> DocumentInterpreter:
    return DocumentInterpreter(
        gateway=gateway,
        model="generic-multimodal-model",
        source_limits=support.limits(max_pages=4),
        execution_limits=InterpretationExecutionLimits(
            max_pages_per_window=max_pages_per_window,
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


@pytest.mark.parametrize("unsupported", [False, True])
def test_window_explanations_are_metadata_not_classification_conflicts(
    tmp_path: Path,
    unsupported: bool,
) -> None:
    source_path, media_type = support.write_source(tmp_path, pages=3)
    gateway = _SequenceGateway(
        [
            _response(1, 2, unsupported=unsupported, reason="First window explanation."),
            _response(3, 3, unsupported=unsupported, reason="Second window explanation."),
        ]
    )

    outcome = _interpreter(gateway).interpret(
        support.interpretation_request(media_type=media_type),
        source_path,
    )

    assert outcome.validation.status is InterpretationValidationStatus.VALID
    assert outcome.interpretation.classification.status.value == (
        "unsupported" if unsupported else "classified"
    )
    assert outcome.interpretation.classification.reason == (
        "Bounded windows agree that the document classification is unsupported."
        if unsupported
        else None
    )


def test_aggregate_collection_capacity_is_partitioned_across_window_schemas(
    tmp_path: Path,
) -> None:
    source_path, media_type = support.write_source(tmp_path, pages=3)
    gateway = _SequenceGateway([_response(1, 1), _response(2, 2), _response(3, 3)])

    _interpreter(gateway, max_calls=3, max_pages_per_window=1).interpret(
        support.interpretation_request(media_type=media_type),
        source_path,
    )

    for request, expected_capacity in zip(gateway.requests, (86, 85, 85), strict=True):
        schema = request.response_json_schema
        assert schema is not None
        for field_name in (
            "mentions",
            "candidate_entities",
            "candidate_facts",
            "evidence",
            "review_signals",
            "page_handling",
        ):
            assert schema["properties"][field_name]["maxItems"] == expected_capacity
        assert (
            schema["$defs"]["DocumentMap"]["properties"]["nodes"]["maxItems"] == expected_capacity
        )
        assert (
            schema["$defs"]["DocumentClassification"]["properties"]["evidence_refs"]["maxItems"]
            == expected_capacity
        )
        assert (
            schema["$defs"]["ClassificationDimensionResult"]["properties"]["evidence_refs"][
                "maxItems"
            ]
            == expected_capacity
        )


def test_unrepresentable_window_plan_is_rejected_before_generation(monkeypatch) -> None:
    gateway = _SequenceGateway([])
    page_count = 257
    monkeypatch.setattr(
        "receipt_intelligence.interpretation.workflow.normalize_document_source",
        lambda source_path, *, limits: SimpleNamespace(
            pages=(None,) * page_count,
            source_media_type="application/pdf",
        ),
    )
    interpreter = DocumentInterpreter(
        gateway=gateway,
        model="generic-multimodal-model",
        source_limits=support.limits(max_pages=1),
        execution_limits=InterpretationExecutionLimits(
            max_pages_per_window=1,
            max_windows=page_count,
        ),
    )

    with pytest.raises(ValueError, match="can be represented"):
        interpreter.interpret(
            support.interpretation_request(media_type="application/pdf"),
            "unused.pdf",
        )

    assert gateway.requests == []


def test_window_classification_evidence_cannot_exceed_aggregate_share(
    tmp_path: Path,
) -> None:
    source_path, media_type = support.write_source(tmp_path, pages=3)
    oversized = _response(1, 2)
    classification = oversized["classification"]
    assert isinstance(classification, dict)
    classification["evidence_refs"] = ["e-1"] * 129
    gateway = _SequenceGateway([oversized, _response(3, 3)])

    with pytest.raises(MalformedGenerationError, match="response schema"):
        _interpreter(gateway).interpret(
            support.interpretation_request(media_type=media_type),
            source_path,
        )

    assert len(gateway.requests) == 1


def test_maximum_aggregated_classification_evidence_round_trips(tmp_path: Path) -> None:
    source_path, media_type = support.write_source(tmp_path, pages=3)
    responses = [_response(1, 2), _response(3, 3)]
    for response in responses:
        classification = response["classification"]
        assert isinstance(classification, dict)
        classification["evidence_refs"] = ["e-1"] * 128
    gateway = _SequenceGateway(responses)

    outcome = _interpreter(gateway).interpret(
        support.interpretation_request(media_type=media_type),
        source_path,
    )

    assert len(outcome.interpretation.classification.evidence_refs) == 256
    assert (
        DocumentInterpretation.model_validate_json(outcome.interpretation.model_dump_json())
        == outcome.interpretation
    )


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
