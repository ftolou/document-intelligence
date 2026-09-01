from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from receipt_intelligence.application.ports.llm import (
    GenerationError,
    GenerationIncompleteError,
    GenerationProviderUnavailableError,
    GenerationRefusedError,
    MalformedGenerationError,
)
from receipt_intelligence.application.ports.multimodal import (
    MultimodalGenerationRequest,
    MultimodalGenerationResult,
)
from receipt_intelligence.extraction import SourceNormalizationLimits
from receipt_intelligence.interpretation import (
    ClassificationDimension,
    ClassificationOption,
    ClassificationStatus,
    DocumentInterpretationRequest,
    DocumentSource,
    InterpretationField,
    InterpretationSpecification,
    LiteralValue,
    NormalizationStatus,
    OnePassDocumentInterpreter,
    ReviewSeverity,
)
from receipt_intelligence.interpretation.contracts import MAX_COLLECTION_SIZE


class _RecordingGateway:
    def __init__(self, response: dict[str, object] | str) -> None:
        self._response = response
        self.requests: list[MultimodalGenerationRequest] = []
        self.page_payloads: list[tuple[bytes, ...]] = []

    def generate(self, request: MultimodalGenerationRequest) -> MultimodalGenerationResult:
        self.requests.append(request)
        self.page_payloads.append(tuple(path.read_bytes() for path in request.image_paths))
        text = self._response if isinstance(self._response, str) else json.dumps(self._response)
        return MultimodalGenerationResult(text=text)


class _FailingGateway:
    def __init__(self, error: GenerationError) -> None:
        self._error = error
        self.requests: list[MultimodalGenerationRequest] = []

    def generate(self, request: MultimodalGenerationRequest) -> MultimodalGenerationResult:
        self.requests.append(request)
        raise self._error


def _limits() -> SourceNormalizationLimits:
    return SourceNormalizationLimits(
        max_source_bytes=1_000_000,
        max_pages=4,
        max_page_width=100,
        max_page_height=100,
        max_page_pixels=10_000,
        max_total_pixels=20_000,
    )


def _request(
    *,
    field_key: str = "stated_right",
    field_description: str = "A right explicitly stated in the source.",
) -> DocumentInterpretationRequest:
    return DocumentInterpretationRequest(
        source=DocumentSource(
            source_id="document-1",
            media_type="image/png",
            name="source.png",
        ),
        specification=InterpretationSpecification(
            specification_id="caller-spec-v1",
            description="Interpret only the requested record concepts.",
            classifications=(
                ClassificationDimension(
                    key="record_kind",
                    description="The caller's record kind.",
                    options=(
                        ClassificationOption(
                            key="supported_record",
                            description="A supported generic record.",
                        ),
                    ),
                ),
            ),
            fields=(
                InterpretationField(
                    key=field_key,
                    description=field_description,
                ),
            ),
        ),
    )


def _response() -> dict[str, object]:
    return {
        "page_coverage": [{"page_number": 1, "status": "interpreted"}],
        "classification": {
            "status": "classified",
            "dimensions": [
                {
                    "dimension_key": "record_kind",
                    "option_paths": [["supported_record"]],
                    "evidence_refs": ["e-1"],
                }
            ],
            "evidence_refs": ["e-1"],
        },
        "document_map": {
            "nodes": [{"node_id": "section-1", "label": "Statement", "evidence_refs": ["e-1"]}]
        },
        "mentions": [
            {"mention_id": "mention-1", "observed_text": "12,?", "evidence_refs": ["e-1"]}
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
                "predicate": "stated_right",
                "object": {
                    "kind": "literal",
                    "literal_type": "amount",
                    "observed": "12,?",
                    "normalization_status": "failed",
                    "currency": "EUR",
                },
                "evidence_refs": ["e-1"],
            }
        ],
        "evidence": [
            {
                "evidence_id": "e-1",
                "source_id": "document-1",
                "page": {"page_number": 1},
                "excerpt": "12,?",
            }
        ],
        "review_signals": [
            {
                "code": "ambiguous_value",
                "message": "The observed amount is ambiguous.",
                "severity": "review_required",
                "evidence_refs": ["e-1"],
                "fact_refs": ["fact-1"],
            }
        ],
    }


def _write_source(path: Path) -> Path:
    Image.new("RGB", (8, 6), "white").save(path, format="PNG")
    return path


def _write_pdf(path: Path, *, page_count: int) -> Path:
    pages = [Image.new("RGB", (8, 6), "white") for _ in range(page_count)]
    try:
        pages[0].save(path, format="PDF", save_all=True, append_images=pages[1:], resolution=72)
    finally:
        for page in pages:
            page.close()
    return path


def _unsupported_response() -> dict[str, object]:
    return {
        "page_coverage": [{"page_number": 1, "status": "irrelevant"}],
        "classification": {"status": "unsupported", "reason": "Outside the supplied options."},
        "document_map": {"nodes": []},
        "mentions": [],
        "candidate_entities": [],
        "candidate_facts": [],
        "evidence": [],
        "review_signals": [],
    }


def _document_fact_response(
    *,
    predicate: str,
    literal_type: str,
    observed: str,
    normalization_status: str,
    normalized: str | None = None,
) -> dict[str, object]:
    response = _response()
    response["document_map"] = {"nodes": []}
    response["mentions"] = []
    response["candidate_entities"] = []
    response["review_signals"] = []
    literal: dict[str, object] = {
        "kind": "literal",
        "literal_type": literal_type,
        "observed": observed,
        "normalization_status": normalization_status,
    }
    if normalized is not None:
        literal["normalized"] = normalized
    response["candidate_facts"] = [
        {
            "fact_id": "fact-1",
            "subject": {"kind": "document", "source_id": "document-1"},
            "predicate": predicate,
            "object": literal,
            "evidence_refs": ["e-1"],
        }
    ]
    return response


def test_interprets_all_outputs_through_one_provider_neutral_call(tmp_path: Path) -> None:
    gateway = _RecordingGateway(_response())
    interpreter = OnePassDocumentInterpreter(
        gateway=gateway,
        model="generic-multimodal-model",
        source_limits=_limits(),
    )
    request = _request()

    result = interpreter.interpret(request, _write_source(tmp_path / "source.png"))

    assert len(gateway.requests) == 1
    generation_request = gateway.requests[0]
    assert generation_request.operation == "document_interpretation"
    assert generation_request.format_json is True
    assert generation_request.response_json_schema is not None
    assert len(generation_request.image_paths) == 1
    assert gateway.page_payloads[0][0].startswith(b"\x89PNG")
    assert request.specification.model_dump_json(indent=2) in generation_request.prompt
    assert "supported_record" in generation_request.prompt

    assert result.source is request.source
    assert result.specification is request.specification
    assert result.classification.status is ClassificationStatus.CLASSIFIED
    assert result.document_map.nodes[0].label == "Statement"
    assert result.mentions[0].observed_text == "12,?"
    assert result.candidate_entities[0].candidate_entity_id == "entity-1"
    assert result.evidence[0].source_id == "document-1"
    assert result.review_signals[0].severity is ReviewSeverity.REVIEW_REQUIRED
    assert result.requires_review is True
    assert result.review_signals[-1].code == "explicit_review_signal"
    fact_value = result.candidate_facts[0].object
    assert isinstance(fact_value, LiteralValue)
    assert fact_value.observed == "12,?"
    assert fact_value.normalized is None
    assert fact_value.normalization_status is NormalizationStatus.FAILED


def test_accepts_unselected_optional_classification_dimension(tmp_path: Path) -> None:
    request = _request()
    optional_dimension = ClassificationDimension(
        key="optional_record_detail",
        description="An optional caller-defined record detail.",
        options=(
            ClassificationOption(
                key="available_detail",
                description="A generic optional detail.",
            ),
        ),
        min_selections=0,
    )
    request = request.model_copy(
        update={
            "specification": request.specification.model_copy(
                update={
                    "classifications": (*request.specification.classifications, optional_dimension)
                }
            )
        }
    )
    response = _response()
    classification = response["classification"]
    assert isinstance(classification, dict)
    dimensions = classification["dimensions"]
    assert isinstance(dimensions, list)
    dimensions.append(
        {
            "dimension_key": "optional_record_detail",
            "option_paths": [],
            "evidence_refs": [],
        }
    )
    gateway = _RecordingGateway(response)
    interpreter = OnePassDocumentInterpreter(
        gateway=gateway,
        model="generic-multimodal-model",
        source_limits=_limits(),
    )

    result = interpreter.interpret(request, _write_source(tmp_path / "source.png"))

    assert len(gateway.requests) == 1
    assert result.classification.dimensions[0].evidence_refs == ("e-1",)
    assert result.classification.dimensions[1].option_paths == ()
    assert result.classification.dimensions[1].evidence_refs == ()


def test_rejects_output_outside_caller_classification_without_repair_call(
    tmp_path: Path,
) -> None:
    response = _response()
    classification = response["classification"]
    assert isinstance(classification, dict)
    dimensions = classification["dimensions"]
    assert isinstance(dimensions, list)
    dimensions[0]["option_paths"] = [["not_in_specification"]]
    gateway = _RecordingGateway(response)
    interpreter = OnePassDocumentInterpreter(
        gateway=gateway,
        model="generic-multimodal-model",
        source_limits=_limits(),
    )

    with pytest.raises(MalformedGenerationError, match="interpretation contract"):
        interpreter.interpret(_request(), _write_source(tmp_path / "source.png"))

    assert len(gateway.requests) == 1


def test_rejects_candidate_fact_not_requested_by_caller(tmp_path: Path) -> None:
    response = _response()
    candidate_facts = response["candidate_facts"]
    assert isinstance(candidate_facts, list)
    candidate_facts[0]["predicate"] = "unrequested_concept"
    gateway = _RecordingGateway(response)
    interpreter = OnePassDocumentInterpreter(
        gateway=gateway,
        model="generic-multimodal-model",
        source_limits=_limits(),
    )

    with pytest.raises(MalformedGenerationError, match="caller-supplied"):
        interpreter.interpret(_request(), _write_source(tmp_path / "source.png"))

    assert len(gateway.requests) == 1


def test_rejects_malformed_output_without_repair_call(tmp_path: Path) -> None:
    gateway = _RecordingGateway("not json")
    interpreter = OnePassDocumentInterpreter(
        gateway=gateway,
        model="generic-multimodal-model",
        source_limits=_limits(),
    )

    with pytest.raises(MalformedGenerationError):
        interpreter.interpret(_request(), _write_source(tmp_path / "source.png"))

    assert len(gateway.requests) == 1


@pytest.mark.parametrize(
    "missing_section",
    [
        "classification",
        "page_coverage",
        "document_map",
        "mentions",
        "candidate_entities",
        "candidate_facts",
        "evidence",
        "review_signals",
    ],
)
def test_rejects_response_with_missing_section_without_repair_call(
    tmp_path: Path,
    missing_section: str,
) -> None:
    response = _response()
    del response[missing_section]
    gateway = _RecordingGateway(response)
    interpreter = OnePassDocumentInterpreter(
        gateway=gateway,
        model="generic-multimodal-model",
        source_limits=_limits(),
    )

    with pytest.raises(MalformedGenerationError, match="response schema"):
        interpreter.interpret(_request(), _write_source(tmp_path / "source.png"))

    assert len(gateway.requests) == 1


@pytest.mark.parametrize(
    "error",
    [
        GenerationRefusedError("generation refused"),
        GenerationIncompleteError("generation incomplete"),
        GenerationProviderUnavailableError("provider unavailable"),
        GenerationError("provider failed"),
    ],
    ids=["refusal", "incomplete", "provider-unavailable", "provider-error"],
)
def test_propagates_provider_neutral_generation_failures(
    tmp_path: Path,
    error: GenerationError,
) -> None:
    gateway = _FailingGateway(error)
    interpreter = OnePassDocumentInterpreter(
        gateway=gateway,
        model="generic-multimodal-model",
        source_limits=_limits(),
    )

    with pytest.raises(type(error)) as raised:
        interpreter.interpret(_request(), _write_source(tmp_path / "source.png"))

    assert raised.value is error
    assert len(gateway.requests) == 1


def test_preserves_source_stated_expiry_without_deriving_expired_state(tmp_path: Path) -> None:
    gateway = _RecordingGateway(
        _document_fact_response(
            predicate="valid_until",
            literal_type="date",
            observed="2021-02-07",
            normalization_status="normalized",
            normalized="2021-02-07",
        )
    )
    interpreter = OnePassDocumentInterpreter(
        gateway=gateway,
        model="generic-multimodal-model",
        source_limits=_limits(),
    )
    request = _request(
        field_key="valid_until",
        field_description="An expiry date explicitly stated in the source.",
    )

    result = interpreter.interpret(request, _write_source(tmp_path / "source.png"))

    fact = result.candidate_facts[0]
    assert fact.predicate == "valid_until"
    assert isinstance(fact.object, LiteralValue)
    assert fact.object.observed == "2021-02-07"
    assert fact.object.normalized == "2021-02-07"
    assert "expired" not in result.model_dump_json()


def test_preserves_relative_deadline_without_external_date_inference(tmp_path: Path) -> None:
    gateway = _RecordingGateway(
        _document_fact_response(
            predicate="appeal_period",
            literal_type="text",
            observed="two weeks after notification",
            normalization_status="unsafe",
        )
    )
    interpreter = OnePassDocumentInterpreter(
        gateway=gateway,
        model="generic-multimodal-model",
        source_limits=_limits(),
    )
    request = _request(
        field_key="appeal_period",
        field_description="A relative deadline explicitly stated in the source.",
    )

    result = interpreter.interpret(request, _write_source(tmp_path / "source.png"))

    fact = result.candidate_facts[0]
    assert isinstance(fact.object, LiteralValue)
    assert fact.object.observed == "two weeks after notification"
    assert fact.object.normalized is None
    assert fact.object.normalization_status is NormalizationStatus.UNSAFE


def test_accepts_unsupported_output_without_extracted_assertions(tmp_path: Path) -> None:
    gateway = _RecordingGateway(_unsupported_response())
    interpreter = OnePassDocumentInterpreter(
        gateway=gateway,
        model="generic-multimodal-model",
        source_limits=_limits(),
    )

    result = interpreter.interpret(_request(), _write_source(tmp_path / "source.png"))

    assert result.classification.status is ClassificationStatus.UNSUPPORTED
    assert result.evidence == ()
    assert result.requires_review is False


@pytest.mark.parametrize(
    ("coverage_status", "model_signal_count", "validation_codes"),
    [
        ("irrelevant", 0, ("missing_page_coverage",)),
        ("irrelevant", MAX_COLLECTION_SIZE, ("missing_page_coverage",)),
        (
            "unreadable",
            MAX_COLLECTION_SIZE,
            ("missing_page_coverage", "unreadable_page"),
        ),
    ],
)
def test_exposes_validation_review_with_bounded_model_signal_capacity(
    tmp_path: Path,
    coverage_status: str,
    model_signal_count: int,
    validation_codes: tuple[str, ...],
) -> None:
    request = _request().model_copy(
        update={"source": DocumentSource(source_id="document-1", media_type="application/pdf")}
    )
    response = _unsupported_response()
    response["page_coverage"] = [{"page_number": 1, "status": coverage_status}]
    response["review_signals"] = [
        {
            "code": f"warning_{index}",
            "message": "A model-owned warning.",
            "severity": "warning",
        }
        for index in range(model_signal_count)
    ]
    interpreter = OnePassDocumentInterpreter(
        gateway=_RecordingGateway(response),
        model="generic-multimodal-model",
        source_limits=_limits(),
    )

    result = interpreter.interpret(request, _write_pdf(tmp_path / "source.pdf", page_count=2))

    assert result.requires_review is True
    assert len(result.review_signals) == min(
        model_signal_count + len(validation_codes), MAX_COLLECTION_SIZE
    )
    assert tuple(
        signal.code for signal in result.review_signals[-len(validation_codes) :]
    ) == validation_codes
    assert all(
        signal.severity is ReviewSeverity.REVIEW_REQUIRED
        for signal in result.review_signals[-len(validation_codes) :]
    )


def test_rejects_evidence_page_outside_normalized_source(tmp_path: Path) -> None:
    response = _response()
    evidence = response["evidence"]
    assert isinstance(evidence, list)
    evidence[0]["page"] = {"page_number": 2}
    gateway = _RecordingGateway(response)
    interpreter = OnePassDocumentInterpreter(
        gateway=gateway,
        model="generic-multimodal-model",
        source_limits=_limits(),
    )

    with pytest.raises(MalformedGenerationError, match="interpretation contract"):
        interpreter.interpret(_request(), _write_source(tmp_path / "source.png"))


@pytest.mark.parametrize("assertion_kind", ["classification", "document-map", "candidate-entity"])
def test_rejects_extracted_assertions_without_evidence(
    tmp_path: Path,
    assertion_kind: str,
) -> None:
    response = _unsupported_response()
    if assertion_kind == "classification":
        response["classification"] = {
            "status": "classified",
            "dimensions": [
                {
                    "dimension_key": "record_kind",
                    "option_paths": [["supported_record"]],
                    "evidence_refs": [],
                }
            ],
            "evidence_refs": [],
        }
    elif assertion_kind == "document-map":
        response["document_map"] = {
            "nodes": [{"node_id": "section-1", "label": "Statement", "evidence_refs": []}]
        }
    else:
        response["candidate_entities"] = [
            {
                "candidate_entity_id": "entity-1",
                "entity_type": "stated_party",
                "evidence_refs": [],
            }
        ]

    gateway = _RecordingGateway(response)
    interpreter = OnePassDocumentInterpreter(
        gateway=gateway,
        model="generic-multimodal-model",
        source_limits=_limits(),
    )

    with pytest.raises(MalformedGenerationError, match="interpretation contract"):
        interpreter.interpret(_request(), _write_source(tmp_path / "source.png"))


def test_rejects_declared_media_type_that_does_not_match_source(tmp_path: Path) -> None:
    request = _request().model_copy(
        update={"source": DocumentSource(source_id="document-1", media_type="application/pdf")}
    )
    gateway = _RecordingGateway(_response())
    interpreter = OnePassDocumentInterpreter(
        gateway=gateway,
        model="generic-multimodal-model",
        source_limits=_limits(),
    )

    with pytest.raises(ValueError, match="media_type"):
        interpreter.interpret(request, _write_source(tmp_path / "source.png"))

    assert gateway.requests == []
