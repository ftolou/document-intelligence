from __future__ import annotations

import pytest

from receipt_intelligence.interpretation import (
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
    InterpretationValidationStatus,
    LiteralType,
    LiteralValue,
    NormalizationStatus,
    PageHandlingState,
    ReviewSeverity,
    ReviewSignal,
    SourcePageHandling,
    SourcePageRange,
    SourcePageReference,
    validate_document_interpretation,
)


def _interpretation(
    *page_handling: SourcePageHandling,
    evidence: tuple[EvidenceReference, ...] = (),
    review_signals: tuple[ReviewSignal, ...] = (),
    candidate_facts: tuple[CandidateFact, ...] = (),
) -> DocumentInterpretation:
    return DocumentInterpretation(
        source=DocumentSource(source_id="document-1", media_type="image/png"),
        specification=InterpretationSpecification(
            specification_id="specification-1",
            description="Interpret the requested value.",
            fields=(InterpretationField(key="value", description="An observed value."),),
        ),
        classification=DocumentClassification(
            status=ClassificationStatus.UNSUPPORTED,
            reason="No classification was requested.",
        ),
        evidence=evidence,
        candidate_facts=candidate_facts,
        review_signals=review_signals,
        page_handling=page_handling,
    )


def _handling(
    start_page: int,
    end_page: int,
    state: PageHandlingState = PageHandlingState.INTERPRETED,
) -> SourcePageHandling:
    return SourcePageHandling(
        page_range=SourcePageRange(start_page=start_page, end_page=end_page),
        state=state,
    )


def _validate(
    interpretation: DocumentInterpretation,
    *,
    source_page_count: int,
):
    return validate_document_interpretation(
        interpretation,
        source_page_count=source_page_count,
    )


def test_complete_multi_page_coverage_is_valid_without_expanding_ranges() -> None:
    interpretation = _interpretation(_handling(1, 3), _handling(4, 5))

    validation = _validate(interpretation, source_page_count=5)

    assert validation.status is InterpretationValidationStatus.VALID
    assert validation.issues == ()


@pytest.mark.parametrize(
    "state",
    [PageHandlingState.BLANK, PageHandlingState.IRRELEVANT],
)
def test_explicit_non_content_page_counts_as_covered(state: PageHandlingState) -> None:
    validation = _validate(_interpretation(_handling(1, 1, state)), source_page_count=1)

    assert validation.status is InterpretationValidationStatus.VALID
    assert validation.issues == ()


@pytest.mark.parametrize(
    ("state", "issue_code"),
    [
        (PageHandlingState.UNREADABLE, "UNREADABLE_PAGE"),
        (PageHandlingState.UNPROCESSED_REVIEW_REQUIRED, "UNPROCESSED_PAGE"),
    ],
)
def test_explicit_incomplete_page_state_requires_review(
    state: PageHandlingState,
    issue_code: str,
) -> None:
    validation = _validate(_interpretation(_handling(1, 1, state)), source_page_count=1)

    assert validation.status is InterpretationValidationStatus.REVIEW_REQUIRED
    assert [issue.code for issue in validation.issues] == [issue_code]


def test_missing_page_coverage_requires_review() -> None:
    validation = _validate(_interpretation(_handling(1, 1)), source_page_count=2)

    assert validation.status is InterpretationValidationStatus.REVIEW_REQUIRED
    assert [issue.code for issue in validation.issues] == ["MISSING_PAGE_COVERAGE"]


@pytest.mark.parametrize(
    ("page_handling", "issue_code"),
    [
        ((_handling(1, 2), _handling(2, 3)), "DUPLICATE_PAGE_COVERAGE"),
        ((_handling(1, 4),), "NONEXISTENT_PAGE_COVERAGE"),
        ((_handling(2, 1),), "INVALID_PAGE_RANGE"),
    ],
)
def test_invalid_page_accounting_is_rejected(
    page_handling: tuple[SourcePageHandling, ...],
    issue_code: str,
) -> None:
    validation = _validate(_interpretation(*page_handling), source_page_count=3)

    assert validation.status is InterpretationValidationStatus.INVALID
    assert issue_code in {issue.code for issue in validation.issues}


def test_huge_page_range_is_rejected_with_bounded_diagnostics() -> None:
    interpretation = _interpretation(_handling(1, 1_000_000_000))

    validation = _validate(interpretation, source_page_count=1)

    assert validation.status is InterpretationValidationStatus.INVALID
    assert len(validation.issues) == 2
    assert len(validation.model_dump_json()) < 1000
    assert "1000000000" not in validation.model_dump_json()


@pytest.mark.parametrize(
    ("page", "page_range", "issue_code"),
    [
        (SourcePageReference(page_number=3), None, "NONEXISTENT_EVIDENCE_PAGE"),
        (None, SourcePageRange(start_page=3, end_page=2), "INVALID_EVIDENCE_PAGE_RANGE"),
        (None, SourcePageRange(start_page=1, end_page=3), "NONEXISTENT_EVIDENCE_PAGE"),
    ],
)
def test_invalid_evidence_page_anchor_is_rejected(
    page: SourcePageReference | None,
    page_range: SourcePageRange | None,
    issue_code: str,
) -> None:
    evidence = EvidenceReference(
        evidence_id="evidence-1",
        source_id="document-1",
        page=page,
        page_range=page_range,
    )
    validation = _validate(
        _interpretation(_handling(1, 2), evidence=(evidence,)),
        source_page_count=2,
    )

    assert validation.status is InterpretationValidationStatus.INVALID
    assert issue_code in {issue.code for issue in validation.issues}


def test_evidence_cannot_be_attached_to_non_interpreted_page() -> None:
    evidence = EvidenceReference(
        evidence_id="evidence-1",
        source_id="document-1",
        page=SourcePageReference(page_number=1),
    )

    validation = _validate(
        _interpretation(
            _handling(1, 1, PageHandlingState.BLANK),
            evidence=(evidence,),
        ),
        source_page_count=1,
    )

    assert validation.status is InterpretationValidationStatus.INVALID
    assert [issue.code for issue in validation.issues] == ["EVIDENCE_ON_NON_INTERPRETED_PAGE"]


def test_visual_excerpt_retains_model_observed_provenance() -> None:
    evidence = EvidenceReference(
        evidence_id="evidence-1",
        source_id="document-1",
        page=SourcePageReference(page_number=1),
        excerpt="visually observed text",
        excerpt_provenance=EvidenceTextProvenance.MODEL_OBSERVED,
    )
    interpretation = _interpretation(_handling(1, 1), evidence=(evidence,))

    validation = _validate(interpretation, source_page_count=1)

    assert validation.status is InterpretationValidationStatus.VALID
    assert interpretation.evidence[0].excerpt == "visually observed text"
    assert interpretation.evidence[0].excerpt_provenance is EvidenceTextProvenance.MODEL_OBSERVED


def test_visual_excerpt_without_model_observed_provenance_is_invalid() -> None:
    evidence = EvidenceReference(
        evidence_id="evidence-1",
        source_id="document-1",
        page=SourcePageReference(page_number=1),
        excerpt="unverified visual text",
    )

    validation = _validate(
        _interpretation(_handling(1, 1), evidence=(evidence,)),
        source_page_count=1,
    )

    assert validation.status is InterpretationValidationStatus.INVALID
    assert [issue.code for issue in validation.issues] == ["MISSING_MODEL_OBSERVED_PROVENANCE"]


def test_required_classification_evidence_is_deterministically_invalid() -> None:
    specification = InterpretationSpecification(
        specification_id="specification-1",
        description="Classify this document.",
        classifications=(
            ClassificationDimension(
                key="kind",
                description="Document kind.",
                options=(ClassificationOption(key="record", description="A record."),),
            ),
        ),
    )
    interpretation = DocumentInterpretation(
        source=DocumentSource(source_id="document-1", media_type="image/png"),
        specification=specification,
        classification=DocumentClassification(
            status=ClassificationStatus.CLASSIFIED,
            dimensions=(
                ClassificationDimensionResult(
                    dimension_key="kind",
                    option_paths=(("record",),),
                ),
            ),
        ),
        page_handling=(_handling(1, 1),),
    )

    validation = _validate(interpretation, source_page_count=1)

    assert validation.status is InterpretationValidationStatus.INVALID
    assert {issue.code for issue in validation.issues} == {"MISSING_REQUIRED_EVIDENCE"}


def test_validation_never_repairs_malformed_observed_value() -> None:
    evidence = EvidenceReference(
        evidence_id="evidence-1",
        source_id="document-1",
        page=SourcePageReference(page_number=1),
    )
    fact = CandidateFact(
        fact_id="fact-1",
        subject=DocumentReference(source_id="document-1"),
        predicate="value",
        object=LiteralValue(
            literal_type=LiteralType.DATE,
            observed="27.05.20024",
            normalization_status=NormalizationStatus.FAILED,
        ),
        evidence_refs=("evidence-1",),
    )
    interpretation = _interpretation(
        _handling(1, 1),
        evidence=(evidence,),
        candidate_facts=(fact,),
    )

    validation = _validate(interpretation, source_page_count=1)

    assert validation.status is InterpretationValidationStatus.VALID
    value = interpretation.candidate_facts[0].object
    assert isinstance(value, LiteralValue)
    assert value.observed == "27.05.20024"
    assert value.normalized is None
    assert value.normalization_status is NormalizationStatus.FAILED


def test_model_review_signals_remain_lossless_and_separate_from_validation_issues() -> None:
    signals = tuple(
        ReviewSignal(
            code=f"model-signal-{index}",
            message=f"Model review signal {index}",
            severity=(ReviewSeverity.REVIEW_REQUIRED if index % 2 else ReviewSeverity.WARNING),
        )
        for index in range(256)
    )
    interpretation = _interpretation(
        _handling(1, 1),
        review_signals=signals,
    )

    validation = _validate(interpretation, source_page_count=2)

    assert validation.status is InterpretationValidationStatus.REVIEW_REQUIRED
    assert [issue.code for issue in validation.issues] == ["MISSING_PAGE_COVERAGE"]
    assert interpretation.review_signals == signals


def test_invalid_validation_preserves_existing_model_review_signal() -> None:
    signal = ReviewSignal(
        code="model-review",
        message="The model requested review.",
        severity=ReviewSeverity.REVIEW_REQUIRED,
    )
    interpretation = _interpretation(
        _handling(1, 2),
        review_signals=(signal,),
    )

    validation = _validate(interpretation, source_page_count=1)

    assert validation.status is InterpretationValidationStatus.INVALID
    assert interpretation.review_signals == (signal,)
    assert all(issue.code != signal.code for issue in validation.issues)
