from __future__ import annotations

import json
from pathlib import Path

import pytest

from receipt_intelligence.application.ports.llm import (
    GenerationProviderUnavailableError,
    MalformedGenerationError,
)
from receipt_intelligence.application.ports.multimodal import (
    MultimodalGenerationRequest,
    MultimodalGenerationResult,
)
from receipt_intelligence.extraction.source_normalization import (
    NormalizedDocumentSource,
    SourceNormalizationLimits,
    VisualPage,
)
from receipt_intelligence.interpretation import (
    BoundedDocumentInterpreter,
    CandidateEntityReference,
    ClassificationDimension,
    ClassificationOption,
    ClassificationStatus,
    DocumentInterpretationRequest,
    DocumentSource,
    InterpretationExecutionLimits,
    InterpretationField,
    InterpretationSpecification,
    InterpretationValidationStatus,
    PageHandlingState,
)


class _SequenceGateway:
    def __init__(self, responses: list[dict[str, object] | Exception]) -> None:
        self._responses = iter(responses)
        self.requests: list[MultimodalGenerationRequest] = []
        self.page_payloads: list[tuple[bytes, ...]] = []

    def generate(self, request: MultimodalGenerationRequest) -> MultimodalGenerationResult:
        self.requests.append(request)
        self.page_payloads.append(tuple(path.read_bytes() for path in request.image_paths))
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return MultimodalGenerationResult(text=json.dumps(response))


def _request() -> DocumentInterpretationRequest:
    return DocumentInterpretationRequest(
        source=DocumentSource(source_id="document-1", media_type="application/pdf"),
        specification=InterpretationSpecification(
            specification_id="specification-1",
            description="Extract only the requested concepts.",
            classifications=(
                ClassificationDimension(
                    key="record_kind",
                    description="Generic record kind.",
                    options=(
                        ClassificationOption(
                            key="supported_record",
                            description="A supported record.",
                        ),
                    ),
                ),
            ),
            fields=(InterpretationField(key="stated_name", description="A stated name."),),
        ),
    )


def _source(page_count: int) -> NormalizedDocumentSource:
    return NormalizedDocumentSource(
        source_path=Path("source.pdf"),
        source_media_type="application/pdf",
        pages=tuple(
            VisualPage(
                page_index=index,
                image_bytes=f"page-{index + 1}".encode(),
                media_type="image/png",
                width=1,
                height=1,
            )
            for index in range(page_count)
        ),
    )


def _source_limits(page_count: int) -> SourceNormalizationLimits:
    return SourceNormalizationLimits(
        max_source_bytes=1_000,
        max_pages=page_count,
        max_page_width=10,
        max_page_height=10,
        max_page_pixels=100,
        max_total_pixels=page_count * 100,
    )


def _response(*, evidence_page: int, handled_start: int, handled_end: int) -> dict[str, object]:
    return {
        "classification": {
            "status": "classified",
            "dimensions": [
                {
                    "dimension_key": "record_kind",
                    "option_paths": [["supported_record"]],
                    "confidence": 0.8,
                    "evidence_refs": ["evidence-local"],
                }
            ],
            "evidence_refs": ["evidence-local"],
        },
        "document_map": {
            "nodes": [
                {
                    "node_id": "node-local",
                    "label": "Record",
                    "evidence_refs": ["evidence-local"],
                }
            ]
        },
        "mentions": [
            {
                "mention_id": "mention-local",
                "observed_text": "Same party",
                "evidence_refs": ["evidence-local"],
            }
        ],
        "candidate_entities": [
            {
                "candidate_entity_id": "entity-local",
                "entity_type": "party",
                "display_name": "Same party",
                "mention_refs": ["mention-local"],
                "evidence_refs": ["evidence-local"],
            }
        ],
        "candidate_facts": [
            {
                "fact_id": "fact-local",
                "subject": {
                    "kind": "candidate_entity",
                    "candidate_entity_id": "entity-local",
                },
                "predicate": "stated_name",
                "object": {
                    "kind": "literal",
                    "literal_type": "text",
                    "observed": "Same party",
                },
                "evidence_refs": ["evidence-local"],
            }
        ],
        "evidence": [
            {
                "evidence_id": "evidence-local",
                "source_id": "document-1",
                "page": {"page_number": evidence_page},
            }
        ],
        "review_signals": [],
        "page_handling": [
            {
                "page_range": {"start_page": handled_start, "end_page": handled_end},
                "state": "interpreted",
            }
        ],
    }


def _interpreter(
    gateway: _SequenceGateway,
    *,
    page_count: int,
    pages_per_call: int = 2,
    max_model_calls: int = 3,
) -> BoundedDocumentInterpreter:
    return BoundedDocumentInterpreter(
        gateway=gateway,
        model="generic-model",
        source_limits=_source_limits(page_count),
        execution_limits=InterpretationExecutionLimits(
            reliable_single_pass_max_pages=pages_per_call,
            max_model_calls=max_model_calls,
        ),
    )


def test_windowed_interpretation_preserves_pages_and_namespaces_local_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "receipt_intelligence.interpretation.workflow.normalize_document_source",
        lambda source_path, *, limits: _source(3),
    )
    gateway = _SequenceGateway(
        [
            _response(evidence_page=1, handled_start=1, handled_end=2),
            _response(evidence_page=3, handled_start=3, handled_end=3),
        ]
    )

    outcome = _interpreter(gateway, page_count=3).interpret(_request(), "source.pdf")
    result = outcome.interpretation

    assert len(gateway.requests) == 2
    assert gateway.page_payloads == [(b"page-1", b"page-2"), (b"page-3",)]
    assert "original source page number(s): 1, 2" in gateway.requests[0].prompt
    assert "original source page number(s): 3" in gateway.requests[1].prompt
    assert [item.page.page_number for item in result.evidence if item.page is not None] == [1, 3]
    assert len({item.evidence_id for item in result.evidence}) == 2
    assert len({item.candidate_entity_id for item in result.candidate_entities}) == 2
    assert len({item.fact_id for item in result.candidate_facts}) == 2
    assert len(result.candidate_facts) == 2
    for fact in result.candidate_facts:
        assert isinstance(fact.subject, CandidateEntityReference)
        assert fact.subject.candidate_entity_id in {
            item.candidate_entity_id for item in result.candidate_entities
        }
        assert fact.evidence_refs[0] in {item.evidence_id for item in result.evidence}
    assert result.classification.status is ClassificationStatus.CLASSIFIED
    assert len(result.classification.evidence_refs) == 2
    assert outcome.validation.status is InterpretationValidationStatus.VALID


def test_missing_window_page_is_explicitly_review_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "receipt_intelligence.interpretation.workflow.normalize_document_source",
        lambda source_path, *, limits: _source(3),
    )
    gateway = _SequenceGateway(
        [
            _response(evidence_page=1, handled_start=1, handled_end=1),
            _response(evidence_page=3, handled_start=3, handled_end=3),
        ]
    )

    outcome = _interpreter(gateway, page_count=3).interpret(_request(), "source.pdf")

    assert any(
        item.page_range.start_page == 2
        and item.page_range.end_page == 2
        and item.state is PageHandlingState.UNPROCESSED_REVIEW_REQUIRED
        for item in outcome.interpretation.page_handling
    )
    assert outcome.validation.status is InterpretationValidationStatus.REVIEW_REQUIRED
    assert [issue.code for issue in outcome.validation.issues] == ["UNPROCESSED_PAGE"]


def test_work_limit_is_checked_before_model_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "receipt_intelligence.interpretation.workflow.normalize_document_source",
        lambda source_path, *, limits: _source(5),
    )
    gateway = _SequenceGateway([])

    with pytest.raises(ValueError, match="max_model_calls"):
        _interpreter(gateway, page_count=5, max_model_calls=2).interpret(_request(), "source.pdf")

    assert gateway.requests == []


def test_partial_window_failure_does_not_return_an_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "receipt_intelligence.interpretation.workflow.normalize_document_source",
        lambda source_path, *, limits: _source(3),
    )
    failure = GenerationProviderUnavailableError("provider unavailable")
    gateway = _SequenceGateway(
        [_response(evidence_page=1, handled_start=1, handled_end=2), failure]
    )

    with pytest.raises(GenerationProviderUnavailableError) as raised:
        _interpreter(gateway, page_count=3).interpret(_request(), "source.pdf")

    assert raised.value is failure
    assert len(gateway.requests) == 2


def test_window_cannot_report_another_windows_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "receipt_intelligence.interpretation.workflow.normalize_document_source",
        lambda source_path, *, limits: _source(3),
    )
    gateway = _SequenceGateway(
        [
            _response(evidence_page=1, handled_start=1, handled_end=2),
            _response(evidence_page=1, handled_start=3, handled_end=3),
        ]
    )

    with pytest.raises(MalformedGenerationError, match="assigned source window"):
        _interpreter(gateway, page_count=3).interpret(_request(), "source.pdf")


def test_incompatible_window_classification_requires_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "receipt_intelligence.interpretation.workflow.normalize_document_source",
        lambda source_path, *, limits: _source(3),
    )
    conflicting = _response(evidence_page=3, handled_start=3, handled_end=3)
    conflicting["classification"] = {
        "status": "unsupported",
        "reason": "This window does not support a classification.",
    }
    gateway = _SequenceGateway(
        [_response(evidence_page=1, handled_start=1, handled_end=2), conflicting]
    )

    outcome = _interpreter(gateway, page_count=3).interpret(_request(), "source.pdf")

    assert outcome.interpretation.classification.status is ClassificationStatus.UNSUPPORTED
    assert [signal.code for signal in outcome.interpretation.review_signals] == [
        "window_classification_conflict"
    ]
    assert outcome.validation.status is InterpretationValidationStatus.REVIEW_REQUIRED


def test_small_document_keeps_unmodified_one_pass_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "receipt_intelligence.interpretation.workflow.normalize_document_source",
        lambda source_path, *, limits: _source(2),
    )
    gateway = _SequenceGateway([_response(evidence_page=1, handled_start=1, handled_end=2)])

    outcome = _interpreter(gateway, page_count=2).interpret(_request(), "source.pdf")

    assert len(gateway.requests) == 1
    assert outcome.interpretation.evidence[0].evidence_id == "evidence-local"
    assert outcome.interpretation.candidate_entities[0].candidate_entity_id == "entity-local"
