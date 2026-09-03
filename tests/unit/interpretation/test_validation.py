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
    PageCoverage,
    PageInterpretationState,
    ReviewSeverity,
    ReviewSignal,
    SourcePageRange,
    SourcePageReference,
    validate_document_interpretation,
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


def _interpretation(
    *coverage: PageCoverage,
    evidence: tuple[EvidenceReference, ...] = (),
    review_signals: tuple[ReviewSignal, ...] = (),
) -> DocumentInterpretation:
    return DocumentInterpretation(
        source=DocumentSource(source_id="document-1", media_type="image/png"),
        specification=InterpretationSpecification(
            specification_id="spec-1",
            description="Interpret a generic requested value.",
            fields=(InterpretationField(key="value", description="A stated value."),),
        ),
        page_coverage=coverage,
        classification=DocumentClassification(
            status=ClassificationStatus.UNSUPPORTED,
            reason="No classification was requested.",
        ),
        evidence=evidence,
        review_signals=review_signals,
    )


def _codes(
    interpretation: DocumentInterpretation,
    page_count: int,
) -> set[InterpretationValidationFindingCode]:
    result = validate_document_interpretation(
        interpretation,
        source_page_count=page_count,
    )
    return {finding.code for finding in result.findings}


def test_complete_page_coverage_is_valid() -> None:
    result = validate_document_interpretation(
        _interpretation(_coverage(1, 3)),
        source_page_count=3,
    )

    assert result.status is InterpretationValidationStatus.VALID
    assert result.findings == ()


@pytest.mark.parametrize(
    "state",
    [PageInterpretationState.BLANK, PageInterpretationState.IRRELEVANT],
)
def test_explicit_non_content_page_state_counts_as_covered(
    state: PageInterpretationState,
) -> None:
    result = validate_document_interpretation(
        _interpretation(_coverage(1, 1), _coverage(2, 2, state)),
        source_page_count=2,
    )

    assert result.status is InterpretationValidationStatus.VALID


@pytest.mark.parametrize(
    ("state", "code"),
    [
        (
            PageInterpretationState.UNREADABLE,
            InterpretationValidationFindingCode.UNREADABLE_PAGE,
        ),
        (
            PageInterpretationState.UNPROCESSED,
            InterpretationValidationFindingCode.UNPROCESSED_PAGE,
        ),
    ],
)
def test_explicit_incomplete_page_state_requires_review(
    state: PageInterpretationState,
    code: InterpretationValidationFindingCode,
) -> None:
    result = validate_document_interpretation(
        _interpretation(_coverage(1, 1, state)),
        source_page_count=1,
    )

    assert result.status is InterpretationValidationStatus.REVIEW_REQUIRED
    assert result.findings[0].code is code


def test_missing_page_coverage_requires_review() -> None:
    result = validate_document_interpretation(
        _interpretation(_coverage(1, 1)),
        source_page_count=3,
    )

    assert result.status is InterpretationValidationStatus.REVIEW_REQUIRED
    assert result.findings[0].code is (InterpretationValidationFindingCode.MISSING_PAGE_COVERAGE)


def test_duplicate_and_nonexistent_page_coverage_are_invalid() -> None:
    result = validate_document_interpretation(
        _interpretation(_coverage(1, 2), _coverage(2, 2), _coverage(3, 3)),
        source_page_count=2,
    )

    assert result.status is InterpretationValidationStatus.INVALID
    assert {finding.code for finding in result.findings} >= {
        InterpretationValidationFindingCode.DUPLICATE_PAGE_COVERAGE,
        InterpretationValidationFindingCode.PAGE_COVERAGE_OUT_OF_BOUNDS,
    }


@pytest.mark.parametrize(
    ("coverage", "expected_code"),
    [
        (
            _coverage(2, 1),
            InterpretationValidationFindingCode.PAGE_RANGE_REVERSED,
        ),
        (
            _coverage(1, 1_000_000_000),
            InterpretationValidationFindingCode.PAGE_COVERAGE_OUT_OF_BOUNDS,
        ),
    ],
)
def test_untrusted_coverage_ranges_are_rejected_before_materialization(
    coverage: PageCoverage,
    expected_code: InterpretationValidationFindingCode,
) -> None:
    result = validate_document_interpretation(
        _interpretation(coverage),
        source_page_count=1,
    )

    assert result.status is InterpretationValidationStatus.INVALID
    assert expected_code in {finding.code for finding in result.findings}


@pytest.mark.parametrize(
    ("evidence", "expected_code"),
    [
        (
            EvidenceReference(
                evidence_id="e-1",
                source_id="document-1",
                page=SourcePageReference(page_number=2),
            ),
            InterpretationValidationFindingCode.EVIDENCE_PAGE_OUT_OF_BOUNDS,
        ),
        (
            EvidenceReference(
                evidence_id="e-1",
                source_id="document-1",
                page_range=SourcePageRange(start_page=2, end_page=1),
            ),
            InterpretationValidationFindingCode.EVIDENCE_PAGE_RANGE_REVERSED,
        ),
        (
            EvidenceReference(
                evidence_id="e-1",
                source_id="document-1",
                page_range=SourcePageRange(start_page=1, end_page=1_000_000_000),
            ),
            InterpretationValidationFindingCode.EVIDENCE_PAGE_OUT_OF_BOUNDS,
        ),
    ],
)
def test_invalid_evidence_page_anchors_are_first_class_findings(
    evidence: EvidenceReference,
    expected_code: InterpretationValidationFindingCode,
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
        PageInterpretationState.UNPROCESSED,
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
    assert InterpretationValidationFindingCode.EVIDENCE_ON_NON_INTERPRETED_PAGE in {
        finding.code for finding in result.findings
    }


def test_visual_excerpt_requires_model_observed_provenance() -> None:
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
    assert evidence.excerpt_provenance.value == "model_observed"


def test_model_review_signals_are_preserved_without_becoming_findings() -> None:
    signals = tuple(
        ReviewSignal(
            code=f"model_signal_{index}",
            message=f"Model review signal {index}.",
            severity=ReviewSeverity.REVIEW_REQUIRED,
        )
        for index in range(256)
    )
    interpretation = _interpretation(_coverage(1, 1), review_signals=signals)

    result = validate_document_interpretation(interpretation, source_page_count=1)

    assert result.status is InterpretationValidationStatus.REVIEW_REQUIRED
    assert result.findings == ()
    assert interpretation.review_signals == signals
    assert len(interpretation.review_signals) == 256


def test_source_page_count_must_be_a_positive_integer() -> None:
    for invalid_count in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer"):
            validate_document_interpretation(
                _interpretation(),
                source_page_count=invalid_count,
            )
