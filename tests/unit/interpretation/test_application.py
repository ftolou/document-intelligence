from __future__ import annotations

from pathlib import Path

import interpretation_outcome_support as support
import pytest

from receipt_intelligence.application.ports.llm import GenerationProviderUnavailableError
from receipt_intelligence.interpretation import (
    DocumentInterpretationOutcome,
    InterpretationValidationStatus,
    run_document_interpretation,
)
from receipt_intelligence.pipeline.integrated_receipt_pipeline import run_receipt_extraction


def test_public_application_api_returns_typed_validated_outcome(tmp_path: Path) -> None:
    source_path, media_type = support.write_source(tmp_path)
    gateway = support.RecordingGateway(
        {
            "classification": {"status": "unsupported", "reason": "Outside the options."},
            "document_map": {"nodes": []},
            "mentions": [],
            "candidate_entities": [],
            "candidate_facts": [],
            "evidence": [],
            "review_signals": [],
            "page_handling": [
                {"page_range": {"start_page": 1, "end_page": 1}, "state": "irrelevant"}
            ],
        }
    )

    outcome = run_document_interpretation(
        support.interpretation_request(media_type=media_type),
        source_path,
        gateway=gateway,
        model="generic-multimodal-model",
        source_limits=support.limits(),
    )

    assert isinstance(outcome, DocumentInterpretationOutcome)
    assert outcome.validation.status is InterpretationValidationStatus.VALID
    assert len(gateway.requests) == 1


def test_public_application_api_propagates_provider_neutral_failure(tmp_path: Path) -> None:
    failure = GenerationProviderUnavailableError("provider unavailable")

    class FailingGateway:
        def generate(self, request: object) -> object:
            raise failure

    source_path, media_type = support.write_source(tmp_path)

    with pytest.raises(GenerationProviderUnavailableError) as raised:
        run_document_interpretation(
            support.interpretation_request(media_type=media_type),
            source_path,
            gateway=FailingGateway(),  # type: ignore[arg-type]
            model="generic-multimodal-model",
            source_limits=support.limits(),
        )

    assert raised.value is failure


def test_receipt_and_generic_application_apis_remain_independently_callable() -> None:
    assert callable(run_document_interpretation)
    assert callable(run_receipt_extraction)
