"""Deterministic structural and source-anchor validation for interpretations."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from receipt_intelligence.interpretation.contracts import (
    MAX_COLLECTION_SIZE,
    ClassificationStatus,
    ContractModel,
    DocumentInterpretation,
    PageInterpretationStatus,
    ReviewSeverity,
)


class InterpretationValidationStatus(StrEnum):
    """Stable caller-facing outcome of deterministic validation."""

    VALID = "valid"
    REVIEW_REQUIRED = "review_required"
    INVALID = "invalid"


class InterpretationValidationIssueCode(StrEnum):
    """Stable, provider- and application-independent validation issue codes."""

    DUPLICATE_PAGE_COVERAGE = "duplicate_page_coverage"
    NONEXISTENT_PAGE = "nonexistent_page"
    MISSING_PAGE_ANCHOR = "missing_page_anchor"
    MISSING_PAGE_COVERAGE = "missing_page_coverage"
    EVIDENCE_ON_NON_INTERPRETED_PAGE = "evidence_on_non_interpreted_page"
    MISSING_REQUIRED_EVIDENCE = "missing_required_evidence"
    UNREADABLE_PAGE = "unreadable_page"
    UNPROCESSED_PAGE = "unprocessed_page"
    EXPLICIT_REVIEW_SIGNAL = "explicit_review_signal"


class InterpretationValidationIssue(ContractModel):
    """One deterministic issue, optionally scoped to source pages."""

    code: InterpretationValidationIssueCode
    message: str = Field(min_length=1, max_length=4000)
    page_numbers: tuple[int, ...] = Field(default=(), max_length=MAX_COLLECTION_SIZE)


class InterpretationValidationResult(ContractModel):
    """Machine-testable validation result without application policy."""

    status: InterpretationValidationStatus
    issues: tuple[InterpretationValidationIssue, ...] = Field(
        default=(), max_length=MAX_COLLECTION_SIZE
    )

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.status is InterpretationValidationStatus.VALID and self.issues:
            raise ValueError("A valid interpretation cannot contain validation issues.")
        if self.status is not InterpretationValidationStatus.VALID and not self.issues:
            raise ValueError("A non-valid interpretation requires validation issues.")
        return self


def validate_document_interpretation(
    interpretation: DocumentInterpretation,
    *,
    source_page_count: int,
) -> InterpretationValidationResult:
    """Validate one structured interpretation against its normalized page set.

    Contract construction validates closed shapes and document-local references.
    This function validates facts that require the caller-owned normalized source:
    page existence, complete page accounting, and evidence anchoring.
    """

    if source_page_count < 1:
        raise ValueError("source_page_count must be at least 1.")

    invalid: list[InterpretationValidationIssue] = []
    review: list[InterpretationValidationIssue] = []
    source_pages = set(range(1, source_page_count + 1))

    coverage_numbers = [coverage.page_number for coverage in interpretation.page_coverage]
    duplicate_coverage = sorted(
        page_number
        for page_number in set(coverage_numbers)
        if coverage_numbers.count(page_number) > 1
    )
    if duplicate_coverage:
        invalid.append(
            _issue(
                InterpretationValidationIssueCode.DUPLICATE_PAGE_COVERAGE,
                "Source pages must have exactly one page-coverage state.",
                duplicate_coverage,
            )
        )

    nonexistent_coverage = sorted(set(coverage_numbers) - source_pages)
    if nonexistent_coverage:
        invalid.append(
            _issue(
                InterpretationValidationIssueCode.NONEXISTENT_PAGE,
                "Page coverage references pages outside the normalized source.",
                nonexistent_coverage,
            )
        )

    missing_coverage = sorted(source_pages - set(coverage_numbers))
    if missing_coverage:
        review.append(
            _issue(
                InterpretationValidationIssueCode.MISSING_PAGE_COVERAGE,
                "Normalized source pages are not accounted for by the interpretation.",
                missing_coverage,
            )
        )

    coverage_by_page = {
        coverage.page_number: coverage.status
        for coverage in interpretation.page_coverage
        if coverage.page_number in source_pages
    }
    unreadable = sorted(
        page_number
        for page_number, status in coverage_by_page.items()
        if status is PageInterpretationStatus.UNREADABLE
    )
    if unreadable:
        review.append(
            _issue(
                InterpretationValidationIssueCode.UNREADABLE_PAGE,
                "Unreadable source pages prevent complete interpretation.",
                unreadable,
            )
        )
    unprocessed = sorted(
        page_number
        for page_number, status in coverage_by_page.items()
        if status is PageInterpretationStatus.UNPROCESSED_REVIEW_REQUIRED
    )
    if unprocessed:
        review.append(
            _issue(
                InterpretationValidationIssueCode.UNPROCESSED_PAGE,
                "Source pages were explicitly left unprocessed for review.",
                unprocessed,
            )
        )

    evidence_pages: set[int] = set()
    missing_page_anchor = False
    nonexistent_evidence_pages: set[int] = set()
    for evidence in interpretation.evidence:
        if evidence.page is None:
            missing_page_anchor = True
            continue
        pages = set(evidence.page.page_numbers())
        nonexistent_evidence_pages.update(pages - source_pages)
        evidence_pages.update(pages & source_pages)

    if missing_page_anchor:
        invalid.append(
            _issue(
                InterpretationValidationIssueCode.MISSING_PAGE_ANCHOR,
                "Evidence for a normalized page source requires a page anchor.",
            )
        )
    if nonexistent_evidence_pages:
        invalid.append(
            _issue(
                InterpretationValidationIssueCode.NONEXISTENT_PAGE,
                "Evidence references pages outside the normalized source.",
                sorted(nonexistent_evidence_pages),
            )
        )

    non_interpreted_evidence_pages = sorted(
        page_number
        for page_number in evidence_pages
        if coverage_by_page.get(page_number) is not PageInterpretationStatus.INTERPRETED
    )
    if non_interpreted_evidence_pages:
        invalid.append(
            _issue(
                InterpretationValidationIssueCode.EVIDENCE_ON_NON_INTERPRETED_PAGE,
                "Evidence can reference only pages explicitly marked as interpreted.",
                non_interpreted_evidence_pages,
            )
        )

    if _has_missing_required_evidence(interpretation):
        invalid.append(
            _issue(
                InterpretationValidationIssueCode.MISSING_REQUIRED_EVIDENCE,
                "Extracted assertions require document-local evidence.",
            )
        )

    if any(
        signal.severity is ReviewSeverity.REVIEW_REQUIRED
        for signal in interpretation.review_signals
    ):
        review.append(
            _issue(
                InterpretationValidationIssueCode.EXPLICIT_REVIEW_SIGNAL,
                "The interpretation contains an explicit review-required signal.",
            )
        )

    if invalid:
        return InterpretationValidationResult(
            status=InterpretationValidationStatus.INVALID,
            issues=tuple((*invalid, *review)),
        )
    if review:
        return InterpretationValidationResult(
            status=InterpretationValidationStatus.REVIEW_REQUIRED,
            issues=tuple(review),
        )
    return InterpretationValidationResult(status=InterpretationValidationStatus.VALID)


def _has_missing_required_evidence(interpretation: DocumentInterpretation) -> bool:
    if interpretation.classification.status is ClassificationStatus.CLASSIFIED:
        if not interpretation.classification.evidence_refs:
            return True
        if any(
            dimension.option_paths and not dimension.evidence_refs
            for dimension in interpretation.classification.dimensions
        ):
            return True

    pending_nodes = list(interpretation.document_map.nodes)
    while pending_nodes:
        node = pending_nodes.pop()
        if not node.evidence_refs:
            return True
        pending_nodes.extend(node.children)

    return any(not entity.evidence_refs for entity in interpretation.candidate_entities)


def _issue(
    code: InterpretationValidationIssueCode,
    message: str,
    page_numbers: list[int] | tuple[int, ...] = (),
) -> InterpretationValidationIssue:
    return InterpretationValidationIssue(
        code=code,
        message=message,
        page_numbers=tuple(page_numbers),
    )


__all__ = [
    "InterpretationValidationIssue",
    "InterpretationValidationIssueCode",
    "InterpretationValidationResult",
    "InterpretationValidationStatus",
    "validate_document_interpretation",
]
