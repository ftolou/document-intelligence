"""Deterministic source and reference validation for document interpretations."""

from __future__ import annotations

from collections.abc import Iterable

from receipt_intelligence.interpretation.contracts import (
    ClassificationStatus,
    DocumentInterpretation,
    DocumentInterpretationValidation,
    PageInterpretationState,
    ValidationIssue,
    ValidationIssueCode,
    ValidationIssueSeverity,
    ValidationStatus,
)


class _Issues:
    """Collect at most one bounded diagnostic for each stable rule."""

    def __init__(self) -> None:
        self._items: list[ValidationIssue] = []
        self._codes: set[ValidationIssueCode] = set()

    def add(
        self,
        code: ValidationIssueCode,
        message: str,
        severity: ValidationIssueSeverity,
    ) -> None:
        if code in self._codes:
            return
        self._codes.add(code)
        self._items.append(ValidationIssue(code=code, message=message, severity=severity))

    def result(self, *, model_requires_review: bool) -> DocumentInterpretationValidation:
        items = tuple(self._items)
        if any(item.severity is ValidationIssueSeverity.INVALID for item in items):
            status = ValidationStatus.INVALID
        elif items or model_requires_review:
            status = ValidationStatus.REVIEW_REQUIRED
        else:
            status = ValidationStatus.VALID
        return DocumentInterpretationValidation(status=status, issues=items)


def validate_document_interpretation(
    interpretation: DocumentInterpretation,
    *,
    source_page_count: int,
) -> DocumentInterpretationValidation:
    """Validate one typed interpretation against its trusted normalized source.

    Work is bounded by ``source_page_count`` and the already bounded contract
    collections. Untrusted page-range endpoints are checked before iteration.
    The interpretation and its model-produced review signals are never changed.
    """

    if (
        not isinstance(source_page_count, int)
        or isinstance(source_page_count, bool)
        or source_page_count < 1
    ):
        raise ValueError("source_page_count must be a positive integer.")

    issues = _Issues()
    page_states: list[PageInterpretationState | None] = [None] * (source_page_count + 1)

    for page_interpretation in interpretation.page_interpretations:
        page_range = page_interpretation.page_range
        start = page_range.start_page_number
        end = page_range.end_page_number
        if end < start:
            issues.add(
                ValidationIssueCode.PAGE_RANGE_REVERSED,
                "A page coverage range has reversed endpoints.",
                ValidationIssueSeverity.INVALID,
            )
            continue
        if start > source_page_count or end > source_page_count:
            issues.add(
                ValidationIssueCode.PAGE_COVERAGE_OUT_OF_BOUNDS,
                "Page coverage references pages outside the normalized source.",
                ValidationIssueSeverity.INVALID,
            )
            continue

        # Endpoints are trusted-source-bounded before this loop.
        for page_number in range(start, end + 1):
            if page_states[page_number] is not None:
                issues.add(
                    ValidationIssueCode.DUPLICATE_PAGE_COVERAGE,
                    "A normalized source page is covered more than once.",
                    ValidationIssueSeverity.INVALID,
                )
                continue
            page_states[page_number] = page_interpretation.state

    if any(state is None for state in page_states[1:]):
        issues.add(
            ValidationIssueCode.MISSING_PAGE_COVERAGE,
            "At least one normalized source page has no declared handling state.",
            ValidationIssueSeverity.REVIEW_REQUIRED,
        )
    if PageInterpretationState.UNREADABLE in page_states:
        issues.add(
            ValidationIssueCode.UNREADABLE_PAGE,
            "At least one normalized source page is declared unreadable.",
            ValidationIssueSeverity.REVIEW_REQUIRED,
        )
    if PageInterpretationState.UNPROCESSED_REVIEW_REQUIRED in page_states:
        issues.add(
            ValidationIssueCode.UNPROCESSED_PAGE,
            "At least one normalized source page is explicitly unprocessed.",
            ValidationIssueSeverity.REVIEW_REQUIRED,
        )

    for evidence in interpretation.evidence:
        if evidence.page is not None and evidence.page_range is not None:
            issues.add(
                ValidationIssueCode.EVIDENCE_MULTIPLE_PAGE_ANCHORS,
                "Evidence must use either a page or a page range, not both.",
                ValidationIssueSeverity.INVALID,
            )
            continue

        evidence_pages: Iterable[int]
        if evidence.page is not None:
            page_number = evidence.page.page_number
            if page_number > source_page_count:
                issues.add(
                    ValidationIssueCode.EVIDENCE_PAGE_OUT_OF_BOUNDS,
                    "Evidence references a page outside the normalized source.",
                    ValidationIssueSeverity.INVALID,
                )
                continue
            evidence_pages = (page_number,)
        elif evidence.page_range is not None:
            start = evidence.page_range.start_page_number
            end = evidence.page_range.end_page_number
            if end < start:
                issues.add(
                    ValidationIssueCode.EVIDENCE_PAGE_RANGE_REVERSED,
                    "An evidence page range has reversed endpoints.",
                    ValidationIssueSeverity.INVALID,
                )
                continue
            if start > source_page_count or end > source_page_count:
                issues.add(
                    ValidationIssueCode.EVIDENCE_PAGE_OUT_OF_BOUNDS,
                    "Evidence references pages outside the normalized source.",
                    ValidationIssueSeverity.INVALID,
                )
                continue
            # Endpoints are trusted-source-bounded before this loop.
            evidence_pages = range(start, end + 1)
        else:
            issues.add(
                ValidationIssueCode.EVIDENCE_PAGE_MISSING,
                "Visual evidence requires a page anchor in the normalized source.",
                ValidationIssueSeverity.INVALID,
            )
            continue

        if any(
            page_states[page_number] is not PageInterpretationState.INTERPRETED
            for page_number in evidence_pages
        ):
            issues.add(
                ValidationIssueCode.EVIDENCE_ON_NON_INTERPRETED_PAGE,
                "Evidence is attached to a page not declared interpreted.",
                ValidationIssueSeverity.INVALID,
            )

    _validate_required_evidence(interpretation, issues)
    return issues.result(model_requires_review=interpretation.requires_review)


def _validate_required_evidence(
    interpretation: DocumentInterpretation,
    issues: _Issues,
) -> None:
    if interpretation.classification.status is ClassificationStatus.CLASSIFIED:
        if not interpretation.classification.evidence_refs:
            issues.add(
                ValidationIssueCode.CLASSIFICATION_EVIDENCE_MISSING,
                "A classified result requires evidence.",
                ValidationIssueSeverity.INVALID,
            )
        if any(
            dimension.option_paths and not dimension.evidence_refs
            for dimension in interpretation.classification.dimensions
        ):
            issues.add(
                ValidationIssueCode.CLASSIFICATION_SELECTION_EVIDENCE_MISSING,
                "Each classification selection requires evidence.",
                ValidationIssueSeverity.INVALID,
            )

    pending_nodes = list(interpretation.document_map.nodes)
    while pending_nodes:
        node = pending_nodes.pop()
        if not node.evidence_refs:
            issues.add(
                ValidationIssueCode.DOCUMENT_MAP_EVIDENCE_MISSING,
                "Each document map node requires evidence.",
                ValidationIssueSeverity.INVALID,
            )
        pending_nodes.extend(node.children)

    if any(not entity.evidence_refs for entity in interpretation.candidate_entities):
        issues.add(
            ValidationIssueCode.CANDIDATE_ENTITY_EVIDENCE_MISSING,
            "Each candidate entity requires evidence.",
            ValidationIssueSeverity.INVALID,
        )


__all__ = ["validate_document_interpretation"]
