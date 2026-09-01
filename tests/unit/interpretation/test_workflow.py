from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from receipt_intelligence.application.ports.llm import MalformedGenerationError
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


def _limits() -> SourceNormalizationLimits:
    return SourceNormalizationLimits(
        max_source_bytes=1_000_000,
        max_pages=4,
        max_page_width=100,
        max_page_height=100,
        max_page_pixels=10_000,
        max_total_pixels=20_000,
    )


def _request() -> DocumentInterpretationRequest:
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
                    key="stated_right",
                    description="A right explicitly stated in the source.",
                ),
            ),
        ),
    )


def _response() -> dict[str, object]:
    return {
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
            "nodes": [
                {"node_id": "section-1", "label": "Statement", "evidence_refs": ["e-1"]}
            ]
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
    fact_value = result.candidate_facts[0].object
    assert isinstance(fact_value, LiteralValue)
    assert fact_value.observed == "12,?"
    assert fact_value.normalized is None
    assert fact_value.normalization_status is NormalizationStatus.FAILED


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


def test_rejects_declared_media_type_that_does_not_match_source(tmp_path: Path) -> None:
    request = _request().model_copy(
        update={
            "source": DocumentSource(source_id="document-1", media_type="application/pdf")
        }
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
