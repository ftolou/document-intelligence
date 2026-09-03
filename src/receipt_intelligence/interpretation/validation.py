"""Deterministic source and reference validation for document interpretations."""

from __future__ import annotations

from collections.abc import Iterable

from receipt_intelligence.interpretation.contracts import (
    ClassificationStatus,
    DocumentInterpretation,
    DocumentInterpretationValidation,
    InterpretationValidationFinding,
    InterpretationValidationFindingCode,
    InterpretationValidationStatus,
    PageInterpretationState,
)

_FindingDefinition = tuple[InterpretationValidationStatus, str]

_FINDINGS: dict[InterpretationValidationFindingCode, _FindingDefinition] = {
    InterpretationValidationFindingCode.PAGE_RANGE_REVERSED: (
        InterpretationValidationStatus.INVALID,
        "A page coverage range has reversed endpoints.",
    ),
    InterpretationValidationFindingCode.PAGE_COVERAGE_OUT_OF_BOUNDS: (
        InterpretationValidationStatus.INVALID,
        "Page coverage references pages outside the normalized source.",
    ),
    InterpretationValidationFindingCode.DUPLICATE_PAGE_COVERAGE: (
        InterpretationValidationStatus.INVALID,
        "A normalized source page is covered more than once.",
    ),
    InterpretationValidationFindingCode.MISSING_PAGE_COVERAGE: (
        InterpretationValidationStatus.REVIEW_REQUIRED,
        "At least one normalized source page has no declared handling state.",
    ),
    InterpretationValidationFindingCode.UNREADABLE_PAGE: (
        InterpretationValidationStatus.REVIEW_REQUIRED,
        "At least one normalized source page is declared unreadable.",
    ),
    InterpretationValidationFindingCode.UNPROCESSED_PAGE: (
        InterpretationValidationStatus.REVIEW_REQUIRED,
        "At least one normalized source page is explicitly unprocessed.",
    ),
    InterpretationValidationFindingCode.EVIDENCE_MULTIPLE_PAGE_ANCHORS: (
        InterpretationValidationStatus.INVALID,
        "Evidence must use either a page or a page range, not both.",
    ),
    InterpretationValidationFindingCode.EVIDENCE_PAGE_MISSING: (
        InterpretationValidationStatus.INVALID,
        "Visual evidence requires a page anchor in the normalized source.",
    ),
    InterpretationValidationFindingCode.EVIDENCE_PAGE_RANGE_REVERSED: (
        InterpretationValidationStatus.INVALID,
        "An evidence page range has reversed endpoints.",
    ),
    InterpretationValidationFindingCode.EVIDENCE_PAGE_OUT_OF_BOUNDS: (
        InterpretationValidationStatus.INVALID,
        "Evidence references pages outside the normalized source.",
    ),
    InterpretationValidationFindingCode.EVIDENCE_ON_NON_INTERPRETED_PAGE: (
        InterpretationValidationStatus.INVALID,
        "Evidence is attached to a page not declared interpreted.",
    ),
    InterpretationValidationFindingCode.CLASSIFICATION_EVIDENCE_MISSING: (
        InterpretationValidationStatus.INVALID,
        "A classified result requires evidence.",
    ),
    InterpretationValidationFindingCode.CLASSIFICATION_SELECTION_EVIDENCE_MISSING: (
        InterpretationValidationStatus.INVALID,
        "Each classification selection requires evidence.",
    ),
    InterpretationValidationFindingCode.DOCUMENT_MAP_EVIDENCE_MISSING: (
        InterpretationValidationStatus.INVALID,
        "Each document map node requires evidence.",
    ),
    InterpretationValidationFindingCode.CANDIDATE_ENTITY_EVIDENCE_MISSING: (
        InterpretationValidationStatus.INVALID,
        "Each candidate entity requires evidence.",
    ),
    InterpretationValidationFindingCode.UNREQUESTED_PREDICATE: (
        InterpretationValidationStatus.INVALID,
        "Candidate facts contain concepts absent from the interpretation specification.",
    ),
}


def validate_document_interpretation(
    interpretation: DocumentInterpretation,
    *,
    source_page_count: int,
) -> DocumentInterpretationValidation:
    """Validate a typed result against the trusted normalized source without changing it."""

    if (
        not isinstance(source_page_count, int)
        or isinstance(source_page_count, bool)
        or source_page_count < 1
    ):
        raise ValueError("source_page_count must be a positive integer.")

    codes: set[InterpretationValidationFindingCode] = set()
    page_states: list[PageInterpretationState | None] = [None] * (source_page_count + 1)

    for coverage in interpretation.page_coverage:
        start = coverage.page_range.start_page
        end = coverage.page_range.end_page
        if end < start:
            codes.add(InterpretationValidationFindingCode.PAGE_RANGE_REVERSED)
            continue
        if start > source_page_count or end > source_page_count:
            codes.add(InterpretationValidationFindingCode.PAGE_COVERAGE_OUT_OF_BOUNDS)
            continue

        # Model-controlled endpoints are enumerated only after trusted-source bounds checks.
        for page_number in range(start, end + 1):
            if page_states[page_number] is not None:
                codes.add(InterpretationValidationFindingCode.DUPLICATE_PAGE_COVERAGE)
            else:
                page_states[page_number] = coverage.state

    if any(state is None for state in page_states[1:]):
        codes.add(InterpretationValidationFindingCode.MISSING_PAGE_COVERAGE)
    if PageInterpretationState.UNREADABLE in page_states:
        codes.add(InterpretationValidationFindingCode.UNREADABLE_PAGE)
    if PageInterpretationState.UNPROCESSED in page_states:
        codes.add(InterpretationValidationFindingCode.UNPROCESSED_PAGE)

    for evidence in interpretation.evidence:
        evidence_pages: Iterable[int]
        if evidence.page is not None and evidence.page_range is not None:
            codes.add(InterpretationValidationFindingCode.EVIDENCE_MULTIPLE_PAGE_ANCHORS)
            continue
        if evidence.page is not None:
            page_number = evidence.page.page_number
            if page_number > source_page_count:
                codes.add(InterpretationValidationFindingCode.EVIDENCE_PAGE_OUT_OF_BOUNDS)
                continue
            evidence_pages = (page_number,)
        elif evidence.page_range is not None:
            start = evidence.page_range.start_page
            end = evidence.page_range.end_page
            if end < start:
                codes.add(InterpretationValidationFindingCode.EVIDENCE_PAGE_RANGE_REVERSED)
                continue
            if start > source_page_count or end > source_page_count:
                codes.add(InterpretationValidationFindingCode.EVIDENCE_PAGE_OUT_OF_BOUNDS)
                continue
            # As above, endpoints are source-bounded before range construction.
            evidence_pages = range(start, end + 1)
        else:
            codes.add(InterpretationValidationFindingCode.EVIDENCE_PAGE_MISSING)
            continue

        if any(
            page_states[page_number] is not PageInterpretationState.INTERPRETED
            for page_number in evidence_pages
        ):
            codes.add(InterpretationValidationFindingCode.EVIDENCE_ON_NON_INTERPRETED_PAGE)

    _validate_required_evidence(interpretation, codes)

    allowed_predicates = _field_keys(interpretation)
    if any(fact.predicate not in allowed_predicates for fact in interpretation.candidate_facts):
        codes.add(InterpretationValidationFindingCode.UNREQUESTED_PREDICATE)

    findings = tuple(
        InterpretationValidationFinding(
            code=code,
            status=_FINDINGS[code][0],
            message=_FINDINGS[code][1],
        )
        for code in InterpretationValidationFindingCode
        if code in codes
    )
    if any(finding.status is InterpretationValidationStatus.INVALID for finding in findings):
        status = InterpretationValidationStatus.INVALID
    elif findings or interpretation.requires_review:
        status = InterpretationValidationStatus.REVIEW_REQUIRED
    else:
        status = InterpretationValidationStatus.VALID
    return DocumentInterpretationValidation(status=status, findings=findings)


def _validate_required_evidence(
    interpretation: DocumentInterpretation,
    codes: set[InterpretationValidationFindingCode],
) -> None:
    if interpretation.classification.status is ClassificationStatus.CLASSIFIED:
        if not interpretation.classification.evidence_refs:
            codes.add(InterpretationValidationFindingCode.CLASSIFICATION_EVIDENCE_MISSING)
        if any(
            dimension.option_paths and not dimension.evidence_refs
            for dimension in interpretation.classification.dimensions
        ):
            codes.add(InterpretationValidationFindingCode.CLASSIFICATION_SELECTION_EVIDENCE_MISSING)

    pending_nodes = list(interpretation.document_map.nodes)
    while pending_nodes:
        node = pending_nodes.pop()
        if not node.evidence_refs:
            codes.add(InterpretationValidationFindingCode.DOCUMENT_MAP_EVIDENCE_MISSING)
        pending_nodes.extend(node.children)

    if any(not entity.evidence_refs for entity in interpretation.candidate_entities):
        codes.add(InterpretationValidationFindingCode.CANDIDATE_ENTITY_EVIDENCE_MISSING)


def _field_keys(interpretation: DocumentInterpretation) -> set[str]:
    keys: set[str] = set()
    pending = list(interpretation.specification.fields)
    while pending:
        field = pending.pop()
        keys.add(field.key)
        pending.extend(field.children)
    return keys


__all__ = ["validate_document_interpretation"]
