from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from receipt_intelligence.application.ports.llm import (
    GenerationProviderUnavailableError,
    MalformedGenerationError,
)
from receipt_intelligence.application.ports.multimodal import (
    MultimodalGenerationRequest,
    MultimodalGenerationResult,
)
from receipt_intelligence.extraction import SourceNormalizationLimits
from receipt_intelligence.interpretation import (
    BoundedDocumentInterpreter,
    ClassificationDimension,
    ClassificationOption,
    DocumentInterpretationRequest,
    DocumentSource,
    InterpretationExecutionLimitError,
    InterpretationExecutionLimits,
    InterpretationField,
    InterpretationSpecification,
    InterpretationValidationStatus,
    PageHandlingState,
)


class _SequentialGateway:
    def __init__(self, *responses: dict[str, object] | Exception) -> None:
        self._responses = iter(responses)
        self.requests: list[MultimodalGenerationRequest] = []
        self.page_counts: list[int] = []

    def generate(self, request: MultimodalGenerationRequest) -> MultimodalGenerationResult:
        self.requests.append(request)
        self.page_counts.append(len(request.image_paths))
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return MultimodalGenerationResult(text=json.dumps(response))


def _limits(*, max_pages: int = 8) -> SourceNormalizationLimits:
    return SourceNormalizationLimits(
        max_source_bytes=1_000_000,
        max_pages=max_pages,
        max_page_width=100,
        max_page_height=100,
        max_page_pixels=10_000,
        max_total_pixels=80_000,
    )


def _request() -> DocumentInterpretationRequest:
    return DocumentInterpretationRequest(
        source=DocumentSource(source_id="document-1", media_type="application/pdf"),
        specification=InterpretationSpecification(
            specification_id="specification-1",
            description="Interpret the requested value.",
            classifications=(
                ClassificationDimension(
                    key="kind",
                    description="The record kind.",
                    options=(ClassificationOption(key="record", description="A record."),),
                ),
            ),
            fields=(InterpretationField(key="value", description="A stated value."),),
        ),
    )


def _write_pdf(path: Path, pages: int) -> Path:
    images = [Image.new("RGB", (12, 8), "white") for _ in range(pages)]
    try:
        images[0].save(
            path,
            format="PDF",
            save_all=True,
            append_images=images[1:],
            resolution=72,
        )
    finally:
        for image in images:
            image.close()
    return path


def _response(*, pages: int, observed: str = "same value") -> dict[str, object]:
    return {
        "classification": {
            "status": "classified",
            "dimensions": [
                {
                    "dimension_key": "kind",
                    "option_paths": [["record"]],
                    "evidence_refs": ["evidence-1"],
                }
            ],
            "evidence_refs": ["evidence-1"],
        },
        "document_map": {
            "nodes": [{"node_id": "node-1", "label": "Section", "evidence_refs": ["evidence-1"]}]
        },
        "mentions": [
            {
                "mention_id": "mention-1",
                "observed_text": observed,
                "evidence_refs": ["evidence-1"],
            }
        ],
        "candidate_entities": [
            {
                "candidate_entity_id": "entity-1",
                "entity_type": "document_party",
                "mention_refs": ["mention-1"],
                "evidence_refs": ["evidence-1"],
            }
        ],
        "candidate_facts": [
            {
                "fact_id": "fact-1",
                "subject": {"kind": "candidate_entity", "candidate_entity_id": "entity-1"},
                "predicate": "value",
                "object": {
                    "kind": "literal",
                    "literal_type": "text",
                    "observed": observed,
                },
                "evidence_refs": ["evidence-1"],
            }
        ],
        "evidence": [
            {
                "evidence_id": "evidence-1",
                "source_id": "document-1",
                "page": {"page_number": 1},
                "excerpt": observed,
                "excerpt_provenance": "model_observed",
            }
        ],
        "review_signals": [],
        "page_handling": [
            {
                "page_range": {"start_page": 1, "end_page": pages},
                "state": "interpreted",
            }
        ],
    }


def _interpreter(
    gateway: _SequentialGateway,
    *,
    pages_per_call: int = 2,
    calls: int = 4,
) -> BoundedDocumentInterpreter:
    return BoundedDocumentInterpreter(
        gateway=gateway,
        model="generic-multimodal-model",
        source_limits=_limits(),
        execution_limits=InterpretationExecutionLimits(
            max_pages_per_call=pages_per_call,
            max_provider_calls=calls,
        ),
    )


def test_windows_are_bounded_and_aggregate_with_original_page_references(tmp_path: Path) -> None:
    gateway = _SequentialGateway(_response(pages=2), _response(pages=1))

    outcome = _interpreter(gateway).interpret(_request(), _write_pdf(tmp_path / "source.pdf", 3))
    result = outcome.interpretation

    assert gateway.page_counts == [2, 1]
    assert "source-page window 1-2" in gateway.requests[0].prompt
    assert "source-page window 3-3" in gateway.requests[1].prompt
    assert [item.page.page_number for item in result.evidence if item.page] == [1, 3]
    assert [
        (item.page_range.start_page, item.page_range.end_page) for item in result.page_handling
    ] == [
        (1, 2),
        (3, 3),
    ]
    assert outcome.validation.status is InterpretationValidationStatus.VALID


def test_window_local_collisions_are_scoped_without_semantic_deduplication(
    tmp_path: Path,
) -> None:
    gateway = _SequentialGateway(_response(pages=1), _response(pages=1))

    result = (
        _interpreter(gateway, pages_per_call=1)
        .interpret(_request(), _write_pdf(tmp_path / "source.pdf", 2))
        .interpretation
    )

    assert len(result.candidate_entities) == 2
    assert len(result.candidate_facts) == 2
    assert len(result.mentions) == 2
    assert len({item.candidate_entity_id for item in result.candidate_entities}) == 2
    assert len({item.fact_id for item in result.candidate_facts}) == 2
    for fact in result.candidate_facts:
        assert getattr(fact.subject, "candidate_entity_id", None) in {
            item.candidate_entity_id for item in result.candidate_entities
        }
        assert fact.evidence_refs[0] in {item.evidence_id for item in result.evidence}


def test_small_document_keeps_single_pass_request_shape(tmp_path: Path) -> None:
    gateway = _SequentialGateway(_response(pages=2))

    outcome = _interpreter(gateway, pages_per_call=3).interpret(
        _request(), _write_pdf(tmp_path / "source.pdf", 2)
    )

    assert gateway.page_counts == [2]
    assert "source-page window" not in gateway.requests[0].prompt
    assert outcome.validation.status is InterpretationValidationStatus.VALID


def test_total_provider_work_limit_is_checked_before_generation(tmp_path: Path) -> None:
    gateway = _SequentialGateway(_response(pages=1), _response(pages=1))

    with pytest.raises(InterpretationExecutionLimitError):
        _interpreter(gateway, pages_per_call=1, calls=2).interpret(
            _request(), _write_pdf(tmp_path / "source.pdf", 3)
        )

    assert gateway.requests == []


def test_partial_provider_failure_aborts_the_complete_interpretation(tmp_path: Path) -> None:
    failure = GenerationProviderUnavailableError("provider unavailable")
    gateway = _SequentialGateway(_response(pages=2), failure)

    with pytest.raises(GenerationProviderUnavailableError) as raised:
        _interpreter(gateway).interpret(_request(), _write_pdf(tmp_path / "source.pdf", 3))

    assert raised.value is failure
    assert len(gateway.requests) == 2


def test_missing_window_page_is_explicitly_review_required(tmp_path: Path) -> None:
    incomplete = _response(pages=1)
    gateway = _SequentialGateway(incomplete, _response(pages=1))

    outcome = _interpreter(gateway).interpret(_request(), _write_pdf(tmp_path / "source.pdf", 3))

    assert any(
        item.page_range.start_page == 2
        and item.page_range.end_page == 2
        and item.state is PageHandlingState.UNPROCESSED_REVIEW_REQUIRED
        for item in outcome.interpretation.page_handling
    )
    assert outcome.validation.status is InterpretationValidationStatus.REVIEW_REQUIRED
    assert "UNPROCESSED_PAGE" in {issue.code for issue in outcome.validation.issues}


def test_incompatible_window_classifications_are_not_silently_merged(tmp_path: Path) -> None:
    second = _response(pages=1)
    second["classification"] = {
        "status": "unsupported",
        "reason": "This window is outside the supplied options.",
    }
    gateway = _SequentialGateway(_response(pages=1), second)

    outcome = _interpreter(gateway, pages_per_call=1).interpret(
        _request(), _write_pdf(tmp_path / "source.pdf", 2)
    )

    assert outcome.interpretation.classification.status.value == "unsupported"
    assert outcome.validation.status is InterpretationValidationStatus.REVIEW_REQUIRED
    assert "AMBIGUOUS_WINDOW_CLASSIFICATION" in {issue.code for issue in outcome.validation.issues}


def test_classification_metadata_differences_do_not_change_an_agreed_decision(
    tmp_path: Path,
) -> None:
    first = _response(pages=1)
    second = _response(pages=1)
    first_classification = first["classification"]
    second_classification = second["classification"]
    assert isinstance(first_classification, dict)
    assert isinstance(second_classification, dict)
    first_classification["reason"] = "The first window explanation."
    second_classification["reason"] = "The second window explanation."
    first_dimensions = first_classification["dimensions"]
    second_dimensions = second_classification["dimensions"]
    assert isinstance(first_dimensions, list)
    assert isinstance(second_dimensions, list)
    assert isinstance(first_dimensions[0], dict)
    assert isinstance(second_dimensions[0], dict)
    first_dimensions[0]["confidence"] = 0.6
    second_dimensions[0]["confidence"] = 0.8
    gateway = _SequentialGateway(first, second)

    outcome = _interpreter(gateway, pages_per_call=1).interpret(
        _request(), _write_pdf(tmp_path / "source.pdf", 2)
    )

    classification = outcome.interpretation.classification
    assert classification.status.value == "classified"
    assert classification.reason is None
    assert classification.dimensions[0].confidence is None
    assert len(classification.evidence_refs) == 2
    assert len(classification.dimensions[0].evidence_refs) == 2
    assert outcome.validation.status is InterpretationValidationStatus.VALID
    assert "AMBIGUOUS_WINDOW_CLASSIFICATION" not in {
        issue.code for issue in outcome.validation.issues
    }


def test_differing_unsupported_reasons_do_not_create_semantic_disagreement(
    tmp_path: Path,
) -> None:
    first = _response(pages=1)
    second = _response(pages=1)
    first["classification"] = {"status": "unsupported", "reason": "First explanation."}
    second["classification"] = {"status": "unsupported", "reason": "Second explanation."}
    gateway = _SequentialGateway(first, second)

    outcome = _interpreter(gateway, pages_per_call=1).interpret(
        _request(), _write_pdf(tmp_path / "source.pdf", 2)
    )

    classification = outcome.interpretation.classification
    assert classification.status.value == "unsupported"
    assert classification.reason == "All bounded windows reported an unsupported classification."
    assert outcome.validation.status is InterpretationValidationStatus.VALID


def test_windowed_review_signal_codes_are_preserved_while_references_are_scoped(
    tmp_path: Path,
) -> None:
    first = _response(pages=1)
    second = _response(pages=1)
    signal = {
        "code": "ambiguous_value",
        "message": "The value requires review.",
        "evidence_refs": ["evidence-1"],
        "fact_refs": ["fact-1"],
    }
    first["review_signals"] = [signal]
    second["review_signals"] = [signal]
    gateway = _SequentialGateway(first, second)

    result = (
        _interpreter(gateway, pages_per_call=1)
        .interpret(_request(), _write_pdf(tmp_path / "source.pdf", 2))
        .interpretation
    )

    assert [item.code for item in result.review_signals] == [
        "ambiguous_value",
        "ambiguous_value",
    ]
    assert len({item.evidence_refs[0] for item in result.review_signals}) == 2
    assert len({item.fact_refs[0] for item in result.review_signals}) == 2


def test_window_cannot_claim_evidence_from_an_unsupplied_page(tmp_path: Path) -> None:
    first = _response(pages=1)
    evidence = first["evidence"]
    assert isinstance(evidence, list)
    first_evidence = evidence[0]
    assert isinstance(first_evidence, dict)
    first_evidence["page"] = {"page_number": 2}
    gateway = _SequentialGateway(first, _response(pages=1))

    with pytest.raises(MalformedGenerationError, match="outside the supplied page window"):
        _interpreter(gateway, pages_per_call=1).interpret(
            _request(), _write_pdf(tmp_path / "source.pdf", 2)
        )

    assert len(gateway.requests) == 1
