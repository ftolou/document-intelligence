from __future__ import annotations

import pytest
from pydantic import ValidationError

from receipt_intelligence.interpretation import (
    ClassificationStatus,
    DocumentClassification,
    DocumentInterpretation,
    DocumentSource,
    EvidenceReference,
    InterpretationField,
    InterpretationSpecification,
    InterpretationValidationFindingCode,
    InterpretationValidationStatus,
    ReviewSeverity,
    ReviewSignal,
    SourcePageObservation,
    SourcePageRange,
    SourcePageReference,
    SourcePageState,
    validate_document_interpretation,
)


def _interpretation(
    *,
    page_observations: tuple[SourcePageObservation, ...] = (),
    evidence: tuple[EvidenceReference, ...] = (),
    review_signals: tuple[ReviewSignal, ...] = (),
) -> DocumentInterpretation:
    return DocumentInterpretation(
        source=DocumentSource(source_id="source-1", media_type="image/png"),
        specification=InterpretationSpecification(
            specification_id="spec-1",
            description="Extract the requested value.",
            fields=(InterpretationField(key="value", description="A stated value."),),
        ),
        page_observations=page_observations,
        classification=DocumentClassification(
            status=ClassificationStatus.UNSUPPORTED,
            reason="No classifications were requested.",
        ),
        evidence=evidence,
        review_signals=review_signals,
    )


def _page(page_number: int, state: SourcePageState) -> SourcePageObservation:
    return SourcePageObservation(
        page=SourcePageReference(page_number=page_number),
        state=state,
    )


def test_validation_states_are_stable_machine_values() -> None:
    assert [status.value for status in InterpretationValidationStatus] == [
        "VALID",
        "REVIEW_REQUIRED",
        "INVALID",
    ]


@pytest.mark.parametrize(
    ("state", "expected_status"),
    [
        (SourcePageState.PROCESSED, InterpretationValidationStatus.VALID),
        (SourcePageState.BLANK, InterpretationValidationStatus.VALID),
        (SourcePageState.IRRELEVANT, InterpretationValidationStatus.VALID),
        (SourcePageState.UNREADABLE, InterpretationValidationStatus.REVIEW_REQUIRED),
        (SourcePageState.UNPROCESSED, InterpretationValidationStatus.REVIEW_REQUIRED),
    ],
)
def test_explicit_page_states_remain_distinct_and_drive_validation(
    state: SourcePageState,
    expected_status: InterpretationValidationStatus,
) -> None:
    interpretation = _interpretation(page_observations=(_page(1, state),))

    validation = validate_document_interpretation(interpretation, page_count=1)

    assert interpretation.page_observations[0].state is state
    assert validation.status is expected_status


def test_missing_page_coverage_requires_review_without_materializing_source_range() -> None:
    interpretation = _interpretation(
        page_observations=(
            _page(1, SourcePageState.PROCESSED),
            _page(3, SourcePageState.BLANK),
        )
    )

    validation = validate_document_interpretation(interpretation, page_count=3)

    assert validation.status is InterpretationValidationStatus.REVIEW_REQUIRED
    assert validation.findings[0].code is InterpretationValidationFindingCode.MISSING_PAGE_COVERAGE


def test_evidence_range_outside_source_is_invalid_without_range_materialization() -> None:
    interpretation = _interpretation(
        page_observations=(_page(1, SourcePageState.PROCESSED),),
        evidence=(
            EvidenceReference(
                evidence_id="e-1",
                source_id="source-1",
                page_range=SourcePageRange(start_page=1, end_page=10**100),
            ),
        ),
    )

    validation = validate_document_interpretation(interpretation, page_count=1)

    assert validation.status is InterpretationValidationStatus.INVALID
    assert validation.findings[0].code is (
        InterpretationValidationFindingCode.EVIDENCE_PAGE_OUT_OF_RANGE
    )


def test_invalid_evidence_page_range_is_rejected_structurally() -> None:
    with pytest.raises(ValidationError, match="must not precede"):
        SourcePageRange(start_page=2, end_page=1)


def test_model_review_signal_is_preserved_separately_from_validation_findings() -> None:
    signal = ReviewSignal(
        code="ambiguous_value",
        message="The model observed an ambiguous value.",
        severity=ReviewSeverity.REVIEW_REQUIRED,
    )
    interpretation = _interpretation(
        page_observations=(
            SourcePageObservation(
                page=SourcePageReference(page_number=1),
                state=SourcePageState.PROCESSED,
                model_observed_text="12,?",
            ),
        ),
        review_signals=(signal,),
    )
    before = interpretation.model_dump()

    validation = validate_document_interpretation(interpretation, page_count=1)

    assert validation.status is InterpretationValidationStatus.REVIEW_REQUIRED
    assert validation.findings[0].code is InterpretationValidationFindingCode.MODEL_REVIEW_REQUIRED
    assert interpretation.review_signals == (signal,)
    assert interpretation.page_observations[0].model_observed_text == "12,?"
    assert interpretation.model_dump() == before


def test_duplicate_or_nonexistent_page_observations_are_invalid() -> None:
    interpretation = _interpretation(
        page_observations=(
            _page(1, SourcePageState.PROCESSED),
            _page(1, SourcePageState.BLANK),
            _page(3, SourcePageState.IRRELEVANT),
        )
    )

    validation = validate_document_interpretation(interpretation, page_count=2)

    assert validation.status is InterpretationValidationStatus.INVALID
    assert {finding.code for finding in validation.findings} >= {
        InterpretationValidationFindingCode.DUPLICATE_PAGE_OBSERVATION,
        InterpretationValidationFindingCode.PAGE_OBSERVATION_OUT_OF_RANGE,
        InterpretationValidationFindingCode.MISSING_PAGE_COVERAGE,
    }
