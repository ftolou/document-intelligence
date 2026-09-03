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
    InterpretationValidationStatus,
    PageCoverage,
    PageInterpretationState,
    ReviewSeverity,
    ReviewSignal,
    SourcePageRange,
    SourcePageReference,
    validate_document_interpretation,
)


def _interpretation(
    *coverage: PageCoverage,
    evidence: tuple[EvidenceReference, ...] = (),
    review_signals: tuple[ReviewSignal, ...] = (),
) -> DocumentInterpretation:
    return DocumentInterpretation(
        source=DocumentSource(source_id="document-1", media_type="image/png"),
        specification=InterpretationSpecification(
            specification_id="spec-1",
            description="A generic requested value.",
            fields=(InterpretationField(key="value", description="A stated value."),),
        ),
        classification=DocumentClassification(
            status=ClassificationStatus.UNSUPPORTED,
            reason="No classification was requested.",
        ),
        page_coverage=coverage,
        evidence=evidence,
        review_signals=review_signals,
    )


def _coverage(
    start_page: int,
    end_page: int,
    state: PageInterpretationState = PageInterpretationState.INTERPRETED,
) -> PageCoverage:
    return PageCoverage(
        page_range=SourcePageRange(start_page=start_page, end_page=end_page),
        state=state,
    )


def _codes(interpretation: DocumentInterpretation, page_count: int) -> set[str]:
    result = validate_document_interpretation(
        interpretation,
        source_page_count=page_count,
    )
    return {issue.code for issue in result.issues}


def test_complete_multi_page_coverage_is_valid() -> None:
    result = validate_document_interpretation(
        _interpretation(_coverage(1, 3)),
        source_page_count=3,
    )

    assert result.status is InterpretationValidationStatus.VALID
    assert result.issues == ()


def test_page_coverage_at_contract_collection_boundary_is_valid() -> None:
    coverage = tuple(_coverage(page, page) for page in range(1, 257))

    result = validate_document_interpretation(
        _interpretation(*coverage),
        source_page_count=256,
    )

    assert result.status is InterpretationValidationStatus.VALID


@pytest.mark.parametrize(
    "state",
    [PageInterpretationState.BLANK, PageInterpretationState.IRRELEVANT],
)
def test_explicit_non_content_page_counts_as_covered(state: PageInterpretationState) -> None:
    result = validate_document_interpretation(
        _interpretation(_coverage(1, 1), _coverage(2, 2, state)),
        source_page_count=2,
    )

    assert result.status is InterpretationValidationStatus.VALID


@pytest.mark.parametrize(
    ("state", "code"),
    [
        (PageInterpretationState.UNREADABLE, "unreadable_page"),
        (PageInterpretationState.UNPROCESSED_REVIEW_REQUIRED, "unprocessed_page"),
    ],
)
def test_explicit_incomplete_page_state_requires_review(
    state: PageInterpretationState,
    code: str,
) -> None:
    result = validate_document_interpretation(
        _interpretation(_coverage(1, 1, state)),
        source_page_count=1,
    )

    assert result.status is InterpretationValidationStatus.REVIEW_REQUIRED
    assert result.issues[0].code == code


def test_missing_page_coverage_requires_review() -> None:
    result = validate_document_interpretation(
        _interpretation(_coverage(1, 1)),
        source_page_count=3,
    )

    assert result.status is InterpretationValidationStatus.REVIEW_REQUIRED
    assert result.issues[0].code == "missing_page_coverage"
    assert result.issues[0].page_range == SourcePageRange(start_page=2, end_page=3)


def test_duplicate_and_nonexistent_page_coverage_are_invalid() -> None:
    interpretation = _interpretation(
        _coverage(1, 2),
        _coverage(2, 2),
        _coverage(3, 3),
    )
    result = validate_document_interpretation(interpretation, source_page_count=2)

    assert result.status is InterpretationValidationStatus.INVALID
    assert {issue.code for issue in result.issues} == {
        "duplicate_page_coverage",
        "page_coverage_out_of_bounds",
    }


@pytest.mark.parametrize("end_page", [0, 1_000_000_000])
def test_untrusted_page_ranges_are_rejected_before_materialization(end_page: int) -> None:
    if end_page == 0:
        coverage = _coverage(2, 1)
        expected_code = "page_range_reversed"
    else:
        coverage = _coverage(1, end_page)
        expected_code = "page_coverage_out_of_bounds"

    result = validate_document_interpretation(
        _interpretation(coverage),
        source_page_count=1,
    )

    assert result.status is InterpretationValidationStatus.INVALID
    assert expected_code in {issue.code for issue in result.issues}
    assert len(result.issues) <= 2


@pytest.mark.parametrize(
    ("evidence", "expected_code"),
    [
        (
            EvidenceReference(
                evidence_id="e-1",
                source_id="document-1",
                page=SourcePageReference(page_number=2),
            ),
            "evidence_page_out_of_bounds",
        ),
        (
            EvidenceReference(
                evidence_id="e-1",
                source_id="document-1",
                page_range=SourcePageRange(start_page=2, end_page=1),
            ),
            "evidence_page_range_reversed",
        ),
        (
            EvidenceReference(
                evidence_id="e-1",
                source_id="document-1",
                page_range=SourcePageRange(start_page=1, end_page=1_000_000_000),
            ),
            "evidence_page_out_of_bounds",
        ),
    ],
)
def test_invalid_evidence_page_anchors_are_deterministic_findings(
    evidence: EvidenceReference,
    expected_code: str,
) -> None:
    assert expected_code in _codes(
        _interpretation(_coverage(1, 1), evidence=(evidence,)),
        1,
    )


@pytest.mark.parametrize(
    "state",
    [
        None,
        PageInterpretationState.BLANK,
        PageInterpretationState.IRRELEVANT,
        PageInterpretationState.UNREADABLE,
        PageInterpretationState.UNPROCESSED_REVIEW_REQUIRED,
    ],
)
def test_evidence_requires_interpreted_page_coverage(
    state: PageInterpretationState | None,
) -> None:
    evidence = EvidenceReference(
        evidence_id="e-1",
        source_id="document-1",
        page=SourcePageReference(page_number=1),
    )
    coverage = () if state is None else (_coverage(1, 1, state),)

    result = validate_document_interpretation(
        _interpretation(*coverage, evidence=(evidence,)),
        source_page_count=1,
    )

    assert result.status is InterpretationValidationStatus.INVALID
    assert "evidence_on_non_interpreted_page" in {issue.code for issue in result.issues}


def test_visual_excerpt_requires_explicit_model_observed_provenance() -> None:
    with pytest.raises(ValidationError, match="explicit text provenance"):
        EvidenceReference(
            evidence_id="e-1",
            source_id="document-1",
            page=SourcePageReference(page_number=1),
            excerpt="model-read text",
        )

    evidence = EvidenceReference(
        evidence_id="e-1",
        source_id="document-1",
        page=SourcePageReference(page_number=1),
        excerpt="model-read text",
        excerpt_provenance="model_observed",
    )
    assert evidence.excerpt == "model-read text"
    assert evidence.excerpt_provenance is not None
    assert evidence.excerpt_provenance.value == "model_observed"


def test_model_review_signals_are_preserved_and_influence_outcome_without_conversion() -> None:
    signals = tuple(
        ReviewSignal(
            code=f"model_signal_{index}",
            message=f"Model review signal {index}.",
            severity=ReviewSeverity.REVIEW_REQUIRED,
        )
        for index in range(256)
    )
    interpretation = _interpretation(_coverage(1, 1), review_signals=signals)

    result = validate_document_interpretation(interpretation, source_page_count=2)

    assert interpretation.review_signals == signals
    assert len(interpretation.review_signals) == 256
    assert result.status is InterpretationValidationStatus.REVIEW_REQUIRED
    assert [issue.code for issue in result.issues] == ["missing_page_coverage"]


def test_invalid_validation_and_model_review_metadata_remain_separate() -> None:
    signal = ReviewSignal(
        code="model_uncertainty",
        message="The model reported uncertainty.",
        severity=ReviewSeverity.REVIEW_REQUIRED,
    )
    interpretation = _interpretation(
        _coverage(1, 1),
        _coverage(1, 1),
        review_signals=(signal,),
    )
    serialized_signals = interpretation.review_signals

    result = validate_document_interpretation(interpretation, source_page_count=1)

    assert result.status is InterpretationValidationStatus.INVALID
    assert result.issues[0].code == "duplicate_page_coverage"
    assert interpretation.review_signals == serialized_signals == (signal,)


def test_source_page_count_must_come_from_a_nonempty_normalized_source() -> None:
    with pytest.raises(ValueError, match="positive"):
        validate_document_interpretation(_interpretation(), source_page_count=0)
