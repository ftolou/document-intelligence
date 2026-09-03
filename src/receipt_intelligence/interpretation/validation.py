"""Deterministic, source-aware validation for document interpretations."""

from __future__ import annotations

from receipt_intelligence.interpretation.contracts import (
    MAX_COLLECTION_SIZE,
    ClassificationStatus,
    DocumentInterpretation,
    InterpretationValidationIssue,
    InterpretationValidationResult,
    InterpretationValidationStatus,
    PageInterpretationState,
    ReviewSeverity,
    SourcePageRange,
)


def validate_document_interpretation(
    interpretation: DocumentInterpretation,
    *,
    source_page_count: int,
) -> InterpretationValidationResult:
    """Validate one typed interpretation against its trusted normalized page count.

    Contract construction remains responsible for the closed graph and literal
    invariants. This function adds only checks that require the trusted source
    boundary or the complete interpretation.
    """

    if source_page_count < 1:
        raise ValueError("source_page_count must be positive.")

    issues: list[InterpretationValidationIssue] = []
    page_states: list[PageInterpretationState | None] = [None] * source_page_count

    for coverage in interpretation.page_coverage:
        page_range = coverage.page_range
        if page_range.start_page > page_range.end_page:
            _add_issue(
                issues,
                code="page_range_reversed",
                message="Page coverage range start exceeds its end.",
                severity=InterpretationValidationStatus.INVALID,
                page_range=page_range,
            )
            continue
        if page_range.end_page > source_page_count:
            _add_issue(
                issues,
                code="page_coverage_out_of_bounds",
                message="Page coverage references a page outside the normalized source.",
                severity=InterpretationValidationStatus.INVALID,
                page_range=page_range,
            )
            continue

        duplicate = False
        # Range endpoints are checked against trusted bounds before enumeration.
        for page_number in range(page_range.start_page, page_range.end_page + 1):
            page_index = page_number - 1
            if page_states[page_index] is not None:
                duplicate = True
            else:
                page_states[page_index] = coverage.state
        if duplicate:
            _add_issue(
                issues,
                code="duplicate_page_coverage",
                message="A normalized source page is accounted for more than once.",
                severity=InterpretationValidationStatus.INVALID,
                page_range=page_range,
            )

        if coverage.state is PageInterpretationState.UNREADABLE:
            _add_issue(
                issues,
                code="unreadable_page",
                message="A source page was explicitly declared unreadable.",
                severity=InterpretationValidationStatus.REVIEW_REQUIRED,
                page_range=page_range,
            )
        elif coverage.state is PageInterpretationState.UNPROCESSED_REVIEW_REQUIRED:
            _add_issue(
                issues,
                code="unprocessed_page",
                message="A source page was explicitly left unprocessed.",
                severity=InterpretationValidationStatus.REVIEW_REQUIRED,
                page_range=page_range,
            )

    for missing_range in _missing_page_ranges(page_states):
        _add_issue(
            issues,
            code="missing_page_coverage",
            message="A normalized source page is not accounted for.",
            severity=InterpretationValidationStatus.REVIEW_REQUIRED,
            page_range=missing_range,
        )

    for evidence in interpretation.evidence:
        if evidence.page is not None:
            evidence_range = SourcePageRange(
                start_page=evidence.page.page_number,
                end_page=evidence.page.page_number,
            )
        elif evidence.page_range is not None:
            evidence_range = evidence.page_range
        else:
            _add_issue(
                issues,
                code="evidence_page_anchor_missing",
                message="Visual evidence requires a normalized source page anchor.",
                severity=InterpretationValidationStatus.INVALID,
            )
            continue

        if evidence_range.start_page > evidence_range.end_page:
            _add_issue(
                issues,
                code="evidence_page_range_reversed",
                message="Evidence page range start exceeds its end.",
                severity=InterpretationValidationStatus.INVALID,
                page_range=evidence_range,
            )
            continue
        if evidence_range.end_page > source_page_count:
            _add_issue(
                issues,
                code="evidence_page_out_of_bounds",
                message="Evidence references a page outside the normalized source.",
                severity=InterpretationValidationStatus.INVALID,
                page_range=evidence_range,
            )
            continue

        # As above, enumerate only after both endpoints are source-bounded.
        anchored_states = (
            page_states[page_number - 1]
            for page_number in range(evidence_range.start_page, evidence_range.end_page + 1)
        )
        if any(state is not PageInterpretationState.INTERPRETED for state in anchored_states):
            _add_issue(
                issues,
                code="evidence_on_non_interpreted_page",
                message="Evidence is attached to a page not declared interpreted.",
                severity=InterpretationValidationStatus.INVALID,
                page_range=evidence_range,
            )

    _validate_required_evidence(interpretation, issues)
    _validate_requested_predicates(interpretation, issues)

    if any(issue.severity is InterpretationValidationStatus.INVALID for issue in issues):
        status = InterpretationValidationStatus.INVALID
    elif issues or any(
        signal.severity is ReviewSeverity.REVIEW_REQUIRED
        for signal in interpretation.review_signals
    ):
        status = InterpretationValidationStatus.REVIEW_REQUIRED
    else:
        status = InterpretationValidationStatus.VALID
    return InterpretationValidationResult(status=status, issues=tuple(issues))


def _validate_required_evidence(
    interpretation: DocumentInterpretation,
    issues: list[InterpretationValidationIssue],
) -> None:
    if interpretation.classification.status is ClassificationStatus.CLASSIFIED:
        if not interpretation.classification.evidence_refs:
            _add_issue(
                issues,
                code="classification_evidence_missing",
                message="A classified result requires evidence.",
                severity=InterpretationValidationStatus.INVALID,
            )
        if any(
            dimension.option_paths and not dimension.evidence_refs
            for dimension in interpretation.classification.dimensions
        ):
            _add_issue(
                issues,
                code="classification_selection_evidence_missing",
                message="Each classification selection requires evidence.",
                severity=InterpretationValidationStatus.INVALID,
            )

    pending_nodes = list(interpretation.document_map.nodes)
    while pending_nodes:
        node = pending_nodes.pop()
        if not node.evidence_refs:
            _add_issue(
                issues,
                code="document_map_evidence_missing",
                message="Each document map node requires evidence.",
                severity=InterpretationValidationStatus.INVALID,
            )
        pending_nodes.extend(node.children)

    if any(not entity.evidence_refs for entity in interpretation.candidate_entities):
        _add_issue(
            issues,
            code="candidate_entity_evidence_missing",
            message="Each candidate entity requires evidence.",
            severity=InterpretationValidationStatus.INVALID,
        )


def _validate_requested_predicates(
    interpretation: DocumentInterpretation,
    issues: list[InterpretationValidationIssue],
) -> None:
    allowed_predicates: set[str] = set()
    pending_fields = list(interpretation.specification.fields)
    while pending_fields:
        field = pending_fields.pop()
        allowed_predicates.add(field.key)
        pending_fields.extend(field.children)
    if any(fact.predicate not in allowed_predicates for fact in interpretation.candidate_facts):
        _add_issue(
            issues,
            code="predicate_not_requested",
            message="Candidate facts contain concepts absent from the specification.",
            severity=InterpretationValidationStatus.INVALID,
        )


def _missing_page_ranges(
    page_states: list[PageInterpretationState | None],
) -> list[SourcePageRange]:
    """Collapse missing trusted pages into bounded diagnostic ranges."""

    missing: list[SourcePageRange] = []
    start_page: int | None = None
    for page_number, state in enumerate(page_states, start=1):
        if state is None and start_page is None:
            start_page = page_number
        elif state is not None and start_page is not None:
            missing.append(SourcePageRange(start_page=start_page, end_page=page_number - 1))
            start_page = None
    if start_page is not None:
        missing.append(SourcePageRange(start_page=start_page, end_page=len(page_states)))
    return missing


def _add_issue(
    issues: list[InterpretationValidationIssue],
    *,
    code: str,
    message: str,
    severity: InterpretationValidationStatus,
    page_range: SourcePageRange | None = None,
) -> None:
    if len(issues) >= MAX_COLLECTION_SIZE:
        if severity is InterpretationValidationStatus.INVALID and not any(
            issue.severity is InterpretationValidationStatus.INVALID for issue in issues
        ):
            issues[-1] = InterpretationValidationIssue(
                code=code,
                message=message,
                severity=severity,
                page_range=page_range,
            )
        return
    issues.append(
        InterpretationValidationIssue(
            code=code,
            message=message,
            severity=severity,
            page_range=page_range,
        )
    )


__all__ = ["validate_document_interpretation"]
