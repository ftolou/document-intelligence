"""Deterministic integrity validation for typed document interpretations."""

from __future__ import annotations

from receipt_intelligence.interpretation.contracts import (
    ClassificationStatus,
    DocumentInterpretation,
    DocumentInterpretationValidation,
    InterpretationValidationFinding,
    InterpretationValidationFindingCode,
    InterpretationValidationStatus,
    ReviewSeverity,
    SourcePageState,
)

_FindingDefinition = tuple[InterpretationValidationStatus, str]

_FINDINGS: dict[InterpretationValidationFindingCode, _FindingDefinition] = {
    InterpretationValidationFindingCode.EVIDENCE_PAGE_OUT_OF_RANGE: (
        InterpretationValidationStatus.INVALID,
        "Evidence references a page outside the normalized source.",
    ),
    InterpretationValidationFindingCode.PAGE_OBSERVATION_OUT_OF_RANGE: (
        InterpretationValidationStatus.INVALID,
        "A page observation references a page outside the normalized source.",
    ),
    InterpretationValidationFindingCode.DUPLICATE_PAGE_OBSERVATION: (
        InterpretationValidationStatus.INVALID,
        "A source page has more than one page observation.",
    ),
    InterpretationValidationFindingCode.MISSING_PAGE_COVERAGE: (
        InterpretationValidationStatus.REVIEW_REQUIRED,
        "One or more source pages have no page observation.",
    ),
    InterpretationValidationFindingCode.UNREADABLE_PAGE: (
        InterpretationValidationStatus.REVIEW_REQUIRED,
        "One or more source pages were reported as unreadable.",
    ),
    InterpretationValidationFindingCode.UNPROCESSED_PAGE: (
        InterpretationValidationStatus.REVIEW_REQUIRED,
        "One or more source pages were reported as unprocessed.",
    ),
    InterpretationValidationFindingCode.MISSING_ASSERTION_EVIDENCE: (
        InterpretationValidationStatus.INVALID,
        "One or more extracted assertions lack evidence.",
    ),
    InterpretationValidationFindingCode.UNREQUESTED_PREDICATE: (
        InterpretationValidationStatus.INVALID,
        "Candidate facts contain concepts absent from the interpretation specification.",
    ),
    InterpretationValidationFindingCode.MODEL_REVIEW_REQUIRED: (
        InterpretationValidationStatus.REVIEW_REQUIRED,
        "The model supplied one or more review-required signals.",
    ),
}


def validate_document_interpretation(
    interpretation: DocumentInterpretation,
    *,
    page_count: int,
) -> DocumentInterpretationValidation:
    """Validate runtime source integrity without changing the model-produced value."""

    if page_count < 1:
        raise ValueError("page_count must be positive.")

    codes: set[InterpretationValidationFindingCode] = set()

    for evidence in interpretation.evidence:
        if evidence.page is not None and evidence.page.page_number > page_count:
            codes.add(InterpretationValidationFindingCode.EVIDENCE_PAGE_OUT_OF_RANGE)
        if evidence.page_range is not None and evidence.page_range.end_page > page_count:
            # Comparing endpoints keeps model-controlled ranges constant-space.
            codes.add(InterpretationValidationFindingCode.EVIDENCE_PAGE_OUT_OF_RANGE)

    observed_pages: set[int] = set()
    for observation in interpretation.page_observations:
        page_number = observation.page.page_number
        if page_number > page_count:
            codes.add(InterpretationValidationFindingCode.PAGE_OBSERVATION_OUT_OF_RANGE)
        elif page_number in observed_pages:
            codes.add(InterpretationValidationFindingCode.DUPLICATE_PAGE_OBSERVATION)
        else:
            observed_pages.add(page_number)

        if observation.state is SourcePageState.UNREADABLE:
            codes.add(InterpretationValidationFindingCode.UNREADABLE_PAGE)
        elif observation.state is SourcePageState.UNPROCESSED:
            codes.add(InterpretationValidationFindingCode.UNPROCESSED_PAGE)

    if len(observed_pages) < page_count:
        codes.add(InterpretationValidationFindingCode.MISSING_PAGE_COVERAGE)

    if _has_ungrounded_assertion(interpretation):
        codes.add(InterpretationValidationFindingCode.MISSING_ASSERTION_EVIDENCE)

    allowed_predicates = _field_keys(interpretation)
    if any(fact.predicate not in allowed_predicates for fact in interpretation.candidate_facts):
        codes.add(InterpretationValidationFindingCode.UNREQUESTED_PREDICATE)

    if any(
        signal.severity is ReviewSeverity.REVIEW_REQUIRED
        for signal in interpretation.review_signals
    ):
        codes.add(InterpretationValidationFindingCode.MODEL_REVIEW_REQUIRED)

    findings = tuple(
        InterpretationValidationFinding(
            code=code, status=_FINDINGS[code][0], message=_FINDINGS[code][1]
        )
        for code in InterpretationValidationFindingCode
        if code in codes
    )
    if any(finding.status is InterpretationValidationStatus.INVALID for finding in findings):
        status = InterpretationValidationStatus.INVALID
    elif findings:
        status = InterpretationValidationStatus.REVIEW_REQUIRED
    else:
        status = InterpretationValidationStatus.VALID
    return DocumentInterpretationValidation(status=status, findings=findings)


def _has_ungrounded_assertion(interpretation: DocumentInterpretation) -> bool:
    classification = interpretation.classification
    if classification.status is ClassificationStatus.CLASSIFIED:
        if not classification.evidence_refs:
            return True
        if any(
            dimension.option_paths and not dimension.evidence_refs
            for dimension in classification.dimensions
        ):
            return True

    pending_nodes = list(interpretation.document_map.nodes)
    while pending_nodes:
        node = pending_nodes.pop()
        if not node.evidence_refs:
            return True
        pending_nodes.extend(node.children)

    return any(not entity.evidence_refs for entity in interpretation.candidate_entities)


def _field_keys(interpretation: DocumentInterpretation) -> set[str]:
    keys: set[str] = set()
    pending = list(interpretation.specification.fields)
    while pending:
        field = pending.pop()
        keys.add(field.key)
        pending.extend(field.children)
    return keys


__all__ = ["validate_document_interpretation"]
