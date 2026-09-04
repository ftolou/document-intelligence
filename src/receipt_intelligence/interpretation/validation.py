"""Deterministic source-integrity validation for document interpretations."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from receipt_intelligence.interpretation.contracts import (
    MAX_COLLECTION_SIZE,
    ClassificationStatus,
    ContractModel,
    DocumentInterpretation,
    NonBlankText,
    PageInterpretationState,
    ReviewSeverity,
)


class InterpretationValidationStatus(StrEnum):
    """Stable acceptance state produced only by deterministic validation."""

    VALID = "VALID"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    INVALID = "INVALID"


class InterpretationValidationIssue(ContractModel):
    """One deterministic finding, kept separate from model review signals."""

    code: str = Field(min_length=1, max_length=200, pattern=r"\S")
    message: NonBlankText
    status: InterpretationValidationStatus


class InterpretationValidationResult(ContractModel):
    """The authoritative deterministic acceptance/review result."""

    status: InterpretationValidationStatus
    issues: tuple[InterpretationValidationIssue, ...] = Field(
        default=(), max_length=MAX_COLLECTION_SIZE
    )


class DocumentInterpretationOutcome(ContractModel):
    """Completed one-pass interpretation with its deterministic trust decision."""

    interpretation: DocumentInterpretation
    validation: InterpretationValidationResult


def validate_document_interpretation(
    interpretation: DocumentInterpretation,
    *,
    page_count: int,
) -> InterpretationValidationResult:
    """Validate source coverage and grounding without changing model output."""

    issues: list[InterpretationValidationIssue] = []
    invalid = False
    review_required = False

    def add_issue(
        code: str,
        message: str,
        status: InterpretationValidationStatus,
    ) -> None:
        nonlocal invalid, review_required
        invalid = invalid or status is InterpretationValidationStatus.INVALID
        review_required = (
            review_required or status is InterpretationValidationStatus.REVIEW_REQUIRED
        )
        if len(issues) < MAX_COLLECTION_SIZE:
            issues.append(InterpretationValidationIssue(code=code, message=message, status=status))

    page_states: dict[int, PageInterpretationState] = {}
    for entry in interpretation.page_accounting:
        if entry.page_number > page_count:
            add_issue(
                "page_accounting_out_of_bounds",
                f"Page accounting references page {entry.page_number} outside the source.",
                InterpretationValidationStatus.INVALID,
            )
            continue
        if entry.page_number in page_states:
            add_issue(
                "duplicate_page_accounting",
                f"Page {entry.page_number} has more than one accounting entry.",
                InterpretationValidationStatus.INVALID,
            )
            continue
        page_states[entry.page_number] = entry.state

    missing_page_count = page_count - len(page_states)
    if missing_page_count > 0:
        add_issue(
            "missing_page_coverage",
            f"{missing_page_count} source page(s) have no accounting entry.",
            InterpretationValidationStatus.REVIEW_REQUIRED,
        )

    review_page_count = sum(
        state
        in {
            PageInterpretationState.UNREADABLE,
            PageInterpretationState.UNPROCESSED_REVIEW_REQUIRED,
        }
        for state in page_states.values()
    )
    if review_page_count:
        add_issue(
            "page_review_required",
            f"{review_page_count} source page(s) were unreadable or unprocessed.",
            InterpretationValidationStatus.REVIEW_REQUIRED,
        )

    for evidence in interpretation.evidence:
        if evidence.page is not None:
            page_number = evidence.page.page_number
            if page_number > page_count:
                add_issue(
                    "evidence_page_out_of_bounds",
                    f"Evidence {evidence.evidence_id!r} references page {page_number} outside the source.",
                    InterpretationValidationStatus.INVALID,
                )
            elif page_states.get(page_number) is not PageInterpretationState.INTERPRETED:
                add_issue(
                    "evidence_page_not_interpreted",
                    f"Evidence {evidence.evidence_id!r} references a page not declared interpreted.",
                    InterpretationValidationStatus.INVALID,
                )

        if evidence.page_range is not None:
            start = evidence.page_range.start_page
            end = evidence.page_range.end_page
            if end < start:
                add_issue(
                    "evidence_page_range_reversed",
                    f"Evidence {evidence.evidence_id!r} has a reversed page range.",
                    InterpretationValidationStatus.INVALID,
                )
            elif end > page_count:
                add_issue(
                    "evidence_page_range_out_of_bounds",
                    f"Evidence {evidence.evidence_id!r} has a page range outside the source.",
                    InterpretationValidationStatus.INVALID,
                )
            else:
                expected_pages = end - start + 1
                interpreted_pages = sum(
                    start <= number <= end and state is PageInterpretationState.INTERPRETED
                    for number, state in page_states.items()
                )
                if interpreted_pages != expected_pages:
                    add_issue(
                        "evidence_page_range_not_interpreted",
                        f"Evidence {evidence.evidence_id!r} spans a page not declared interpreted.",
                        InterpretationValidationStatus.INVALID,
                    )

    if interpretation.classification.status is ClassificationStatus.CLASSIFIED:
        if not interpretation.classification.evidence_refs:
            add_issue(
                "classification_missing_evidence",
                "A classified result requires evidence.",
                InterpretationValidationStatus.INVALID,
            )
        for dimension in interpretation.classification.dimensions:
            if dimension.option_paths and not dimension.evidence_refs:
                add_issue(
                    "classification_selection_missing_evidence",
                    f"Classification dimension {dimension.dimension_key!r} requires evidence.",
                    InterpretationValidationStatus.INVALID,
                )

    pending_nodes = list(interpretation.document_map.nodes)
    while pending_nodes:
        node = pending_nodes.pop()
        if not node.evidence_refs:
            add_issue(
                "document_map_node_missing_evidence",
                f"Document map node {node.node_id!r} requires evidence.",
                InterpretationValidationStatus.INVALID,
            )
        pending_nodes.extend(node.children)

    for entity in interpretation.candidate_entities:
        if not entity.evidence_refs:
            add_issue(
                "candidate_entity_missing_evidence",
                f"Candidate entity {entity.candidate_entity_id!r} requires evidence.",
                InterpretationValidationStatus.INVALID,
            )

    review_required = review_required or any(
        signal.severity is ReviewSeverity.REVIEW_REQUIRED
        for signal in interpretation.review_signals
    )

    if invalid:
        status = InterpretationValidationStatus.INVALID
    elif review_required:
        status = InterpretationValidationStatus.REVIEW_REQUIRED
    else:
        status = InterpretationValidationStatus.VALID
    return InterpretationValidationResult(status=status, issues=tuple(issues))


__all__ = [
    "DocumentInterpretationOutcome",
    "InterpretationValidationIssue",
    "InterpretationValidationResult",
    "InterpretationValidationStatus",
    "validate_document_interpretation",
]
