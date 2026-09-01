from __future__ import annotations

import pytest
from pydantic import ValidationError

from receipt_intelligence.interpretation import (
    ClassificationStatus,
    DocumentClassification,
    DocumentInterpretation,
    DocumentMap,
    DocumentMapNode,
    DocumentSource,
    EvidenceReference,
    EvidenceTextOrigin,
    InterpretationField,
    InterpretationSpecification,
    InterpretationValidationIssueCode,
    InterpretationValidationStatus,
    PageCoverage,
    PageInterpretationStatus,
    ReviewSeverity,
    ReviewSignal,
    SourcePageReference,
    validate_document_interpretation,
)


def _interpretation(
    *,
    page_coverage: tuple[PageCoverage, ...],
    evidence: tuple[EvidenceReference, ...] = (),
    classification: DocumentClassification | None = None,
    document_map: DocumentMap | None = None,
    review_signals: tuple[ReviewSignal, ...] = (),
) -> DocumentInterpretation:
    return DocumentInterpretation(
        source=DocumentSource(source_id="document-1", media_type="application/pdf"),
        specification=InterpretationSpecification(
            specification_id="generic-v1",
            description="Extract a caller-defined field.",
            fields=(InterpretationField(key="reference", description="A stated reference."),),
        ),
        classification=classification
        or DocumentClassification(
            status=ClassificationStatus.UNSUPPORTED,
            reason="No classification was requested.",
        ),
        document_map=document_map or DocumentMap(),
        page_coverage=page_coverage,
        evidence=evidence,
        review_signals=review_signals,
    )


def _coverage(*statuses: PageInterpretationStatus) -> tuple[PageCoverage, ...]:
    return tuple(
        PageCoverage(page_number=page_number, status=status)
        for page_number, status in enumerate(statuses, start=1)
    )


def _codes(
    interpretation: DocumentInterpretation, page_count: int
) -> set[InterpretationValidationIssueCode]:
    result = validate_document_interpretation(
        interpretation,
        source_page_count=page_count,
    )
    return {issue.code for issue in result.issues}


def test_validates_complete_multi_page_coverage_and_inclusive_evidence_range() -> None:
    interpretation = _interpretation(
        page_coverage=_coverage(
            PageInterpretationStatus.INTERPRETED,
            PageInterpretationStatus.INTERPRETED,
            PageInterpretationStatus.BLANK,
        ),
        evidence=(
            EvidenceReference(
                evidence_id="e-1",
                source_id="document-1",
                page=SourcePageReference(page_number=1, end_page_number=2),
                excerpt="model-observed heading",
            ),
        ),
    )

    result = validate_document_interpretation(interpretation, source_page_count=3)

    assert result.status is InterpretationValidationStatus.VALID
    assert result.issues == ()
    assert interpretation.evidence[0].excerpt_origin is EvidenceTextOrigin.MODEL_OBSERVED


def test_rejects_invalid_page_range_structurally() -> None:
    with pytest.raises(ValidationError, match="cannot end before"):
        SourcePageReference(page_number=3, end_page_number=2)


@pytest.mark.parametrize(
    "reference",
    [
        SourcePageReference(page_number=3),
        SourcePageReference(page_number=1, end_page_number=3),
    ],
    ids=["single-page", "page-range"],
)
def test_nonexistent_evidence_page_is_invalid(reference: SourcePageReference) -> None:
    interpretation = _interpretation(
        page_coverage=_coverage(
            PageInterpretationStatus.INTERPRETED,
            PageInterpretationStatus.INTERPRETED,
        ),
        evidence=(
            EvidenceReference(
                evidence_id="e-1",
                source_id="document-1",
                page=reference,
            ),
        ),
    )

    result = validate_document_interpretation(interpretation, source_page_count=2)

    assert result.status is InterpretationValidationStatus.INVALID
    assert InterpretationValidationIssueCode.NONEXISTENT_PAGE in _codes(interpretation, 2)


def test_missing_page_coverage_is_review_required_not_silently_valid() -> None:
    interpretation = _interpretation(page_coverage=_coverage(PageInterpretationStatus.INTERPRETED))

    result = validate_document_interpretation(interpretation, source_page_count=3)

    assert result.status is InterpretationValidationStatus.REVIEW_REQUIRED
    assert result.issues[0].code is InterpretationValidationIssueCode.MISSING_PAGE_COVERAGE
    assert result.issues[0].page_numbers == (2, 3)


def test_explicit_blank_and_irrelevant_pages_count_as_complete_coverage() -> None:
    interpretation = _interpretation(
        page_coverage=_coverage(
            PageInterpretationStatus.INTERPRETED,
            PageInterpretationStatus.BLANK,
            PageInterpretationStatus.IRRELEVANT,
        )
    )

    result = validate_document_interpretation(interpretation, source_page_count=3)

    assert result.status is InterpretationValidationStatus.VALID


@pytest.mark.parametrize(
    ("status", "expected_code"),
    [
        (
            PageInterpretationStatus.UNREADABLE,
            InterpretationValidationIssueCode.UNREADABLE_PAGE,
        ),
        (
            PageInterpretationStatus.UNPROCESSED_REVIEW_REQUIRED,
            InterpretationValidationIssueCode.UNPROCESSED_PAGE,
        ),
    ],
)
def test_explicit_incomplete_page_state_has_stable_review_semantics(
    status: PageInterpretationStatus,
    expected_code: InterpretationValidationIssueCode,
) -> None:
    interpretation = _interpretation(page_coverage=_coverage(status))

    result = validate_document_interpretation(interpretation, source_page_count=1)

    assert result.status is InterpretationValidationStatus.REVIEW_REQUIRED
    assert result.issues[0].code is expected_code
    assert interpretation.requires_review is True


def test_duplicate_or_nonexistent_page_coverage_is_invalid() -> None:
    interpretation = _interpretation(
        page_coverage=(
            PageCoverage(page_number=1, status=PageInterpretationStatus.INTERPRETED),
            PageCoverage(page_number=1, status=PageInterpretationStatus.BLANK),
            PageCoverage(page_number=2, status=PageInterpretationStatus.INTERPRETED),
        )
    )

    result = validate_document_interpretation(interpretation, source_page_count=1)

    assert result.status is InterpretationValidationStatus.INVALID
    assert {issue.code for issue in result.issues} == {
        InterpretationValidationIssueCode.DUPLICATE_PAGE_COVERAGE,
        InterpretationValidationIssueCode.NONEXISTENT_PAGE,
    }


def test_visual_evidence_requires_page_anchor_and_interpreted_page() -> None:
    locator_only = _interpretation(
        page_coverage=_coverage(PageInterpretationStatus.INTERPRETED),
        evidence=(
            EvidenceReference(
                evidence_id="e-1",
                source_id="document-1",
                locator="upper-left",
                excerpt="model-observed text",
            ),
        ),
    )
    evidence_on_blank = _interpretation(
        page_coverage=_coverage(PageInterpretationStatus.BLANK),
        evidence=(
            EvidenceReference(
                evidence_id="e-1",
                source_id="document-1",
                page=SourcePageReference(page_number=1),
            ),
        ),
    )

    assert (
        validate_document_interpretation(locator_only, source_page_count=1).status
        is InterpretationValidationStatus.INVALID
    )
    assert InterpretationValidationIssueCode.MISSING_PAGE_ANCHOR in _codes(locator_only, 1)
    assert (
        validate_document_interpretation(evidence_on_blank, source_page_count=1).status
        is InterpretationValidationStatus.INVALID
    )
    assert InterpretationValidationIssueCode.EVIDENCE_ON_NON_INTERPRETED_PAGE in _codes(
        evidence_on_blank, 1
    )


def test_required_evidence_and_explicit_review_signal_have_stable_outcomes() -> None:
    missing_evidence = _interpretation(
        page_coverage=_coverage(PageInterpretationStatus.INTERPRETED),
        document_map=DocumentMap(
            nodes=(DocumentMapNode(node_id="section-1", label="A source section."),)
        ),
    )
    review_signal = _interpretation(
        page_coverage=_coverage(PageInterpretationStatus.BLANK),
        review_signals=(
            ReviewSignal(
                code="ambiguous_value",
                message="An observed value is ambiguous.",
                severity=ReviewSeverity.REVIEW_REQUIRED,
            ),
        ),
    )

    assert (
        validate_document_interpretation(missing_evidence, source_page_count=1).status
        is InterpretationValidationStatus.INVALID
    )
    assert InterpretationValidationIssueCode.MISSING_REQUIRED_EVIDENCE in _codes(
        missing_evidence, 1
    )
    assert (
        validate_document_interpretation(review_signal, source_page_count=1).status
        is InterpretationValidationStatus.REVIEW_REQUIRED
    )
    assert InterpretationValidationIssueCode.EXPLICIT_REVIEW_SIGNAL in _codes(review_signal, 1)
