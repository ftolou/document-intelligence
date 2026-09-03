"""Deterministic source-integrity validation for document interpretations."""

from __future__ import annotations

from receipt_intelligence.interpretation.contracts import (
    ClassificationStatus,
    DocumentInterpretation,
    DocumentInterpretationValidation,
    EvidenceTextProvenance,
    InterpretationValidationStatus,
    PageHandlingState,
    ReviewSeverity,
    SourcePageHandling,
    ValidationIssue,
)


def validate_document_interpretation(
    interpretation: DocumentInterpretation,
    *,
    source_page_count: int,
) -> DocumentInterpretationValidation:
    """Validate integrity that depends on the complete normalized visual source.

    Model-controlled ranges are compared with trusted source bounds before any
    range-dependent work. The implementation never expands a page range.
    """

    if source_page_count < 1:
        raise ValueError("source_page_count must be positive.")

    issues: list[ValidationIssue] = []
    valid_handling = _validate_page_handling(
        interpretation.page_handling,
        source_page_count=source_page_count,
        issues=issues,
    )
    _validate_evidence_pages(
        interpretation,
        source_page_count=source_page_count,
        valid_handling=valid_handling,
        issues=issues,
    )
    _validate_required_evidence(interpretation, issues=issues)

    if any(issue.status is InterpretationValidationStatus.INVALID for issue in issues):
        status = InterpretationValidationStatus.INVALID
    elif issues or any(
        signal.severity is ReviewSeverity.REVIEW_REQUIRED
        for signal in interpretation.review_signals
    ):
        status = InterpretationValidationStatus.REVIEW_REQUIRED
    else:
        status = InterpretationValidationStatus.VALID

    return DocumentInterpretationValidation(status=status, issues=tuple(issues))


def _validate_page_handling(
    page_handling: tuple[SourcePageHandling, ...],
    *,
    source_page_count: int,
    issues: list[ValidationIssue],
) -> list[SourcePageHandling]:
    valid: list[SourcePageHandling] = []
    for handling in page_handling:
        page_range = handling.page_range
        if page_range.start_page > page_range.end_page:
            issues.append(
                _invalid(
                    "INVALID_PAGE_RANGE",
                    "A source page range has reversed endpoints.",
                )
            )
            continue
        if page_range.end_page > source_page_count:
            issues.append(
                _invalid(
                    "NONEXISTENT_PAGE_COVERAGE",
                    "Source page handling references a page outside the normalized source.",
                )
            )
            continue
        valid.append(handling)
        if handling.state is PageHandlingState.UNREADABLE:
            issues.append(
                _review_required(
                    "UNREADABLE_PAGE",
                    "A source page was explicitly reported as unreadable.",
                )
            )
        elif handling.state is PageHandlingState.UNPROCESSED_REVIEW_REQUIRED:
            issues.append(
                _review_required(
                    "UNPROCESSED_PAGE",
                    "A source page was explicitly left unprocessed for review.",
                )
            )

    cursor = 1
    for handling in sorted(
        valid,
        key=lambda item: (item.page_range.start_page, item.page_range.end_page),
    ):
        start = handling.page_range.start_page
        end = handling.page_range.end_page
        if start > cursor:
            issues.append(
                _review_required(
                    "MISSING_PAGE_COVERAGE",
                    "One or more normalized source pages were not accounted for.",
                )
            )
        elif start < cursor:
            issues.append(
                _invalid(
                    "DUPLICATE_PAGE_COVERAGE",
                    "A normalized source page was accounted for more than once.",
                )
            )
        cursor = max(cursor, end + 1)

    if cursor <= source_page_count:
        issues.append(
            _review_required(
                "MISSING_PAGE_COVERAGE",
                "One or more normalized source pages were not accounted for.",
            )
        )
    return valid


def _validate_evidence_pages(
    interpretation: DocumentInterpretation,
    *,
    source_page_count: int,
    valid_handling: list[SourcePageHandling],
    issues: list[ValidationIssue],
) -> None:
    for evidence in interpretation.evidence:
        if (
            evidence.excerpt is not None
            and evidence.excerpt_provenance is not EvidenceTextProvenance.MODEL_OBSERVED
        ):
            issues.append(
                _invalid(
                    "MISSING_MODEL_OBSERVED_PROVENANCE",
                    "Visual evidence excerpt text must be identified as model-observed.",
                )
            )
        if evidence.page is not None:
            start = end = evidence.page.page_number
        elif evidence.page_range is not None:
            start = evidence.page_range.start_page
            end = evidence.page_range.end_page
        else:
            issues.append(
                _invalid(
                    "MISSING_EVIDENCE_PAGE",
                    "Visual evidence requires a normalized source page anchor.",
                )
            )
            continue

        if start > end:
            issues.append(
                _invalid(
                    "INVALID_EVIDENCE_PAGE_RANGE",
                    "An evidence page range has reversed endpoints.",
                )
            )
            continue
        if end > source_page_count:
            issues.append(
                _invalid(
                    "NONEXISTENT_EVIDENCE_PAGE",
                    "Evidence references a page outside the normalized source.",
                )
            )
            continue

        if any(
            handling.state is not PageHandlingState.INTERPRETED
            and handling.page_range.start_page <= end
            and handling.page_range.end_page >= start
            for handling in valid_handling
        ):
            issues.append(
                _invalid(
                    "EVIDENCE_ON_NON_INTERPRETED_PAGE",
                    "Evidence is attached to a page not reported as interpreted.",
                )
            )


def _validate_required_evidence(
    interpretation: DocumentInterpretation,
    *,
    issues: list[ValidationIssue],
) -> None:
    if (
        interpretation.classification.status is ClassificationStatus.CLASSIFIED
        and not interpretation.classification.evidence_refs
    ):
        issues.append(
            _invalid(
                "MISSING_REQUIRED_EVIDENCE",
                "A classified interpretation requires evidence.",
            )
        )

    for dimension in interpretation.classification.dimensions:
        if dimension.option_paths and not dimension.evidence_refs:
            issues.append(
                _invalid(
                    "MISSING_REQUIRED_EVIDENCE",
                    "Each classification selection requires evidence.",
                )
            )

    pending_nodes = list(interpretation.document_map.nodes)
    while pending_nodes:
        node = pending_nodes.pop()
        if not node.evidence_refs:
            issues.append(
                _invalid(
                    "MISSING_REQUIRED_EVIDENCE",
                    "Each document map node requires evidence.",
                )
            )
        pending_nodes.extend(node.children)

    for entity in interpretation.candidate_entities:
        if not entity.evidence_refs:
            issues.append(
                _invalid(
                    "MISSING_REQUIRED_EVIDENCE",
                    "Each candidate entity requires evidence.",
                )
            )


def _invalid(code: str, message: str) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        message=message,
        status=InterpretationValidationStatus.INVALID,
    )


def _review_required(code: str, message: str) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        message=message,
        status=InterpretationValidationStatus.REVIEW_REQUIRED,
    )


__all__ = ["validate_document_interpretation"]
