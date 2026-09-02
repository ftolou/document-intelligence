from __future__ import annotations

import pytest

from receipt_intelligence.interpretation import (
    MAX_COLLECTION_SIZE,
    CandidateFact,
    ClassificationDimension,
    ClassificationDimensionResult,
    ClassificationOption,
    ClassificationStatus,
    DocumentClassification,
    DocumentInterpretation,
    DocumentReference,
    DocumentSource,
    EvidenceReference,
    EvidenceTextProvenance,
    InterpretationField,
    InterpretationSpecification,
    LiteralType,
    LiteralValue,
    NormalizationStatus,
    PageInterpretation,
    PageInterpretationState,
    ReviewSeverity,
    ReviewSignal,
    SourcePageRange,
    ValidationIssueCode,
    ValidationStatus,
    validate_document_interpretation,
)


def _page(
    start: int,
    end: int,
    state: PageInterpretationState = PageInterpretationState.INTERPRETED,
) -> PageInterpretation:
    return PageInterpretation(
        page_range=SourcePageRange(start_page_number=start, end_page_number=end),
        state=state,
    )


def _interpretation(
    *pages: PageInterpretation,
    evidence: tuple[EvidenceReference, ...] = (),
    review_signals: tuple[ReviewSignal, ...] = (),
) -> DocumentInterpretation:
    return DocumentInterpretation(
        source=DocumentSource(source_id="source-1", media_type="image/png"),
        specification=InterpretationSpecification(
            specification_id="spec-1",
            description="A bounded generic specification.",
            fields=(InterpretationField(key="value", description="An observed value."),),
        ),
        classification=DocumentClassification(
            status=ClassificationStatus.UNSUPPORTED,
            reason="No classification was requested.",
        ),
        page_interpretations=pages,
        evidence=evidence,
        review_signals=review_signals,
    )


def _codes(interpretation: DocumentInterpretation, page_count: int) -> set[ValidationIssueCode]:
    result = validate_document_interpretation(
        interpretation,
        source_page_count=page_count,
    )
    return {issue.code for issue in result.issues}


def test_complete_multi_page_coverage_is_valid() -> None:
    result = validate_document_interpretation(
        _interpretation(_page(1, 2)),
        source_page_count=2,
    )

    assert result.status is ValidationStatus.VALID
    assert result.issues == ()


@pytest.mark.parametrize(
    "state",
    [PageInterpretationState.BLANK, PageInterpretationState.IRRELEVANT],
)
def test_explicit_noncontent_page_counts_as_covered(state: PageInterpretationState) -> None:
    result = validate_document_interpretation(
        _interpretation(_page(1, 1, state)),
        source_page_count=1,
    )

    assert result.status is ValidationStatus.VALID


@pytest.mark.parametrize(
    ("state", "code"),
    [
        (PageInterpretationState.UNREADABLE, ValidationIssueCode.UNREADABLE_PAGE),
        (
            PageInterpretationState.UNPROCESSED_REVIEW_REQUIRED,
            ValidationIssueCode.UNPROCESSED_PAGE,
        ),
    ],
)
def test_explicit_unhandled_page_requires_review(
    state: PageInterpretationState,
    code: ValidationIssueCode,
) -> None:
    result = validate_document_interpretation(
        _interpretation(_page(1, 1, state)),
        source_page_count=1,
    )

    assert result.status is ValidationStatus.REVIEW_REQUIRED
    assert {issue.code for issue in result.issues} == {code}


def test_missing_page_coverage_requires_review() -> None:
    result = validate_document_interpretation(
        _interpretation(_page(1, 1)),
        source_page_count=2,
    )

    assert result.status is ValidationStatus.REVIEW_REQUIRED
    assert {issue.code for issue in result.issues} == {ValidationIssueCode.MISSING_PAGE_COVERAGE}


@pytest.mark.parametrize(
    ("pages", "page_count", "code"),
    [
        ((_page(1, 1), _page(1, 2)), 2, ValidationIssueCode.DUPLICATE_PAGE_COVERAGE),
        ((_page(1, 3),), 2, ValidationIssueCode.PAGE_COVERAGE_OUT_OF_BOUNDS),
        ((_page(2, 1),), 2, ValidationIssueCode.PAGE_RANGE_REVERSED),
        ((_page(1, 1_000_000_000),), 1, ValidationIssueCode.PAGE_COVERAGE_OUT_OF_BOUNDS),
    ],
)
def test_invalid_page_coverage_is_bounded_and_machine_testable(
    pages: tuple[PageInterpretation, ...],
    page_count: int,
    code: ValidationIssueCode,
) -> None:
    result = validate_document_interpretation(
        _interpretation(*pages),
        source_page_count=page_count,
    )

    assert result.status is ValidationStatus.INVALID
    assert code in {issue.code for issue in result.issues}
    assert len(result.issues) <= 2


@pytest.mark.parametrize(
    ("page_range", "code"),
    [
        (
            SourcePageRange(start_page_number=2, end_page_number=2),
            ValidationIssueCode.EVIDENCE_PAGE_OUT_OF_BOUNDS,
        ),
        (
            SourcePageRange(start_page_number=2, end_page_number=1),
            ValidationIssueCode.EVIDENCE_PAGE_RANGE_REVERSED,
        ),
        (
            SourcePageRange(start_page_number=1, end_page_number=1_000_000_000),
            ValidationIssueCode.EVIDENCE_PAGE_OUT_OF_BOUNDS,
        ),
    ],
)
def test_invalid_evidence_page_range_is_invalid_without_expansion(
    page_range: SourcePageRange,
    code: ValidationIssueCode,
) -> None:
    evidence = EvidenceReference(
        evidence_id="e-1",
        source_id="source-1",
        page_range=page_range,
    )
    result = validate_document_interpretation(
        _interpretation(_page(1, 1), evidence=(evidence,)),
        source_page_count=1,
    )

    assert result.status is ValidationStatus.INVALID
    assert {issue.code for issue in result.issues} == {code}


def test_evidence_cannot_target_non_interpreted_page() -> None:
    evidence = EvidenceReference(
        evidence_id="e-1",
        source_id="source-1",
        page_range=SourcePageRange(start_page_number=1, end_page_number=1),
    )

    assert ValidationIssueCode.EVIDENCE_ON_NON_INTERPRETED_PAGE in _codes(
        _interpretation(_page(1, 1, PageInterpretationState.BLANK), evidence=(evidence,)),
        1,
    )


def test_missing_required_classification_evidence_is_invalid() -> None:
    specification = InterpretationSpecification(
        specification_id="spec-1",
        description="Classify a generic document.",
        classifications=(
            ClassificationDimension(
                key="kind",
                description="Generic kind.",
                options=(ClassificationOption(key="known", description="Known kind."),),
            ),
        ),
    )
    interpretation = DocumentInterpretation(
        source=DocumentSource(source_id="source-1", media_type="image/png"),
        specification=specification,
        classification=DocumentClassification(
            status=ClassificationStatus.CLASSIFIED,
            dimensions=(
                ClassificationDimensionResult(
                    dimension_key="kind",
                    option_paths=(("known",),),
                ),
            ),
        ),
        page_interpretations=(_page(1, 1),),
    )

    result = validate_document_interpretation(interpretation, source_page_count=1)

    assert result.status is ValidationStatus.INVALID
    assert {issue.code for issue in result.issues} == {
        ValidationIssueCode.CLASSIFICATION_EVIDENCE_MISSING,
        ValidationIssueCode.CLASSIFICATION_SELECTION_EVIDENCE_MISSING,
    }


def test_visual_excerpt_remains_explicitly_model_observed() -> None:
    evidence = EvidenceReference(
        evidence_id="e-1",
        source_id="source-1",
        page_range=SourcePageRange(start_page_number=1, end_page_number=1),
        excerpt="visually reported text",
        excerpt_provenance=EvidenceTextProvenance.MODEL_OBSERVED,
    )
    interpretation = _interpretation(_page(1, 1), evidence=(evidence,))

    result = validate_document_interpretation(interpretation, source_page_count=1)

    assert result.status is ValidationStatus.VALID
    assert interpretation.evidence[0].excerpt == "visually reported text"
    assert interpretation.evidence[0].excerpt_provenance is EvidenceTextProvenance.MODEL_OBSERVED


def test_validation_never_repairs_malformed_observed_value() -> None:
    evidence = EvidenceReference(
        evidence_id="e-1",
        source_id="source-1",
        page_range=SourcePageRange(start_page_number=1, end_page_number=1),
    )
    base = _interpretation(_page(1, 1), evidence=(evidence,))
    malformed = LiteralValue(
        literal_type=LiteralType.DATE,
        observed="27.05.20024",
        normalization_status=NormalizationStatus.FAILED,
    )
    interpretation = base.model_copy(
        update={
            "candidate_facts": (
                CandidateFact(
                    fact_id="fact-1",
                    subject=DocumentReference(source_id="source-1"),
                    predicate="value",
                    object=malformed,
                    evidence_refs=("e-1",),
                ),
            )
        }
    )

    result = validate_document_interpretation(interpretation, source_page_count=1)

    assert result.status is ValidationStatus.VALID
    value = interpretation.candidate_facts[0].object
    assert isinstance(value, LiteralValue)
    assert value.observed == "27.05.20024"
    assert value.normalized is None


def test_model_review_signals_and_validation_issues_remain_separate_and_lossless() -> None:
    signals = tuple(
        ReviewSignal(
            code=f"model-signal-{index}",
            message=f"Model signal {index}.",
            severity=ReviewSeverity.REVIEW_REQUIRED if index == 0 else ReviewSeverity.WARNING,
        )
        for index in range(MAX_COLLECTION_SIZE)
    )
    interpretation = _interpretation(review_signals=signals)

    result = validate_document_interpretation(interpretation, source_page_count=1)

    assert result.status is ValidationStatus.REVIEW_REQUIRED
    assert {issue.code for issue in result.issues} == {ValidationIssueCode.MISSING_PAGE_COVERAGE}
    assert interpretation.review_signals == signals
    assert len(interpretation.review_signals) == MAX_COLLECTION_SIZE
    assert all(issue.code != signal.code for issue in result.issues for signal in signals)


def test_model_review_signal_alone_makes_authoritative_outcome_require_review() -> None:
    signal = ReviewSignal(
        code="model-review",
        message="The model explicitly requests review.",
        severity=ReviewSeverity.REVIEW_REQUIRED,
    )

    result = validate_document_interpretation(
        _interpretation(_page(1, 1), review_signals=(signal,)),
        source_page_count=1,
    )

    assert result.status is ValidationStatus.REVIEW_REQUIRED
    assert result.issues == ()
