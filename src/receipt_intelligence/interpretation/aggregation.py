"""Mechanical aggregation of bounded document interpretation windows."""

from __future__ import annotations

from itertools import count

from receipt_intelligence.interpretation.contracts import (
    CandidateEntityReference,
    ClassificationDimensionResult,
    ClassificationStatus,
    DocumentClassification,
    DocumentInterpretation,
    DocumentMap,
    DocumentMapNode,
    DocumentReference,
    InterpretationValidationStatus,
    ValidationIssue,
)


def aggregate_window_interpretations(
    interpretations: tuple[DocumentInterpretation, ...],
) -> tuple[DocumentInterpretation, tuple[ValidationIssue, ...]]:
    """Combine window graphs without semantic deduplication or entity resolution."""

    if not interpretations:
        raise ValueError("At least one window interpretation is required.")

    source = interpretations[0].source
    specification = interpretations[0].specification
    if any(
        item.source != source or item.specification != specification for item in interpretations[1:]
    ):
        raise ValueError("Window interpretations must share one source and specification.")

    namespaced = tuple(
        _namespace_window(item, window_number=index)
        for index, item in enumerate(interpretations, start=1)
    )
    classification, issues = _aggregate_classification(namespaced)

    return (
        DocumentInterpretation(
            source=source,
            specification=specification,
            classification=classification,
            document_map=DocumentMap(
                nodes=tuple(node for item in namespaced for node in item.document_map.nodes)
            ),
            mentions=tuple(value for item in namespaced for value in item.mentions),
            candidate_entities=tuple(
                value for item in namespaced for value in item.candidate_entities
            ),
            candidate_facts=tuple(value for item in namespaced for value in item.candidate_facts),
            evidence=tuple(value for item in namespaced for value in item.evidence),
            review_signals=tuple(value for item in namespaced for value in item.review_signals),
            page_handling=tuple(value for item in namespaced for value in item.page_handling),
        ),
        issues,
    )


def _namespace_window(
    interpretation: DocumentInterpretation,
    *,
    window_number: int,
) -> DocumentInterpretation:
    """Replace model-local IDs with deterministic globally unique IDs."""

    evidence_ids = {
        item.evidence_id: _identifier(window_number, "evidence", index)
        for index, item in enumerate(interpretation.evidence, start=1)
    }
    mention_ids = {
        item.mention_id: _identifier(window_number, "mention", index)
        for index, item in enumerate(interpretation.mentions, start=1)
    }
    entity_ids = {
        item.candidate_entity_id: _identifier(window_number, "entity", index)
        for index, item in enumerate(interpretation.candidate_entities, start=1)
    }
    fact_ids = {
        item.fact_id: _identifier(window_number, "fact", index)
        for index, item in enumerate(interpretation.candidate_facts, start=1)
    }

    def evidence_refs(values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(evidence_ids[value] for value in values)

    node_numbers = count(1)

    def map_node(node: DocumentMapNode) -> DocumentMapNode:
        node_number = next(node_numbers)
        return node.model_copy(
            update={
                "node_id": _identifier(window_number, "node", node_number),
                "evidence_refs": evidence_refs(node.evidence_refs),
                "children": tuple(map_node(child) for child in node.children),
            }
        )

    classification = interpretation.classification.model_copy(
        update={
            "evidence_refs": evidence_refs(interpretation.classification.evidence_refs),
            "dimensions": tuple(
                dimension.model_copy(
                    update={"evidence_refs": evidence_refs(dimension.evidence_refs)}
                )
                for dimension in interpretation.classification.dimensions
            ),
        }
    )
    mentions = tuple(
        mention.model_copy(
            update={
                "mention_id": mention_ids[mention.mention_id],
                "evidence_refs": evidence_refs(mention.evidence_refs),
            }
        )
        for mention in interpretation.mentions
    )
    entities = tuple(
        entity.model_copy(
            update={
                "candidate_entity_id": entity_ids[entity.candidate_entity_id],
                "mention_refs": tuple(mention_ids[value] for value in entity.mention_refs),
                "evidence_refs": evidence_refs(entity.evidence_refs),
            }
        )
        for entity in interpretation.candidate_entities
    )

    facts = []
    for fact in interpretation.candidate_facts:
        subject = fact.subject
        if not isinstance(subject, DocumentReference):
            subject = subject.model_copy(
                update={"candidate_entity_id": entity_ids[subject.candidate_entity_id]}
            )
        fact_object = fact.object
        if isinstance(fact_object, CandidateEntityReference):
            fact_object = fact_object.model_copy(
                update={"candidate_entity_id": entity_ids[fact_object.candidate_entity_id]}
            )
        facts.append(
            fact.model_copy(
                update={
                    "fact_id": fact_ids[fact.fact_id],
                    "subject": subject,
                    "object": fact_object,
                    "evidence_refs": evidence_refs(fact.evidence_refs),
                }
            )
        )

    return DocumentInterpretation(
        source=interpretation.source,
        specification=interpretation.specification,
        classification=classification,
        document_map=DocumentMap(
            nodes=tuple(map_node(node) for node in interpretation.document_map.nodes)
        ),
        mentions=mentions,
        candidate_entities=entities,
        candidate_facts=tuple(facts),
        evidence=tuple(
            item.model_copy(update={"evidence_id": evidence_ids[item.evidence_id]})
            for item in interpretation.evidence
        ),
        review_signals=tuple(
            signal.model_copy(
                update={
                    "evidence_refs": evidence_refs(signal.evidence_refs),
                    "fact_refs": tuple(fact_ids[value] for value in signal.fact_refs),
                }
            )
            for signal in interpretation.review_signals
        ),
        page_handling=interpretation.page_handling,
    )


def _aggregate_classification(
    interpretations: tuple[DocumentInterpretation, ...],
) -> tuple[DocumentClassification, tuple[ValidationIssue, ...]]:
    classifications = tuple(item.classification for item in interpretations)
    if len({_classification_signature(item) for item in classifications}) != 1:
        return (
            DocumentClassification(
                status=ClassificationStatus.UNSUPPORTED,
                reason="Bounded window classifications conflict; human review is required.",
            ),
            (
                ValidationIssue(
                    code="WINDOW_CLASSIFICATION_CONFLICT",
                    message="Bounded windows reported incompatible document classifications.",
                    status=InterpretationValidationStatus.REVIEW_REQUIRED,
                ),
            ),
        )

    first = classifications[0]
    reasons = {item.reason for item in classifications}
    common_reason = next(iter(reasons)) if len(reasons) == 1 else None
    if first.status is ClassificationStatus.UNSUPPORTED:
        return (
            DocumentClassification.model_validate(
                {
                    **first.model_dump(),
                    "reason": common_reason
                    or "Bounded windows agree that the document classification is unsupported.",
                    "evidence_refs": tuple(
                        value for item in classifications for value in item.evidence_refs
                    ),
                }
            ),
            (),
        )

    dimensions_by_window = tuple(
        {dimension.dimension_key: dimension for dimension in item.dimensions}
        for item in classifications
    )
    dimensions = []
    for first_dimension in first.dimensions:
        matching = tuple(values[first_dimension.dimension_key] for values in dimensions_by_window)
        confidences = {item.confidence for item in matching}
        dimensions.append(
            ClassificationDimensionResult.model_validate(
                {
                    **first_dimension.model_dump(),
                    "confidence": next(iter(confidences)) if len(confidences) == 1 else None,
                    "evidence_refs": tuple(
                        value for item in matching for value in item.evidence_refs
                    ),
                }
            )
        )
    return (
        DocumentClassification.model_validate(
            {
                **first.model_dump(),
                "reason": common_reason,
                "dimensions": tuple(dimensions),
                "evidence_refs": tuple(
                    value for item in classifications for value in item.evidence_refs
                ),
            }
        ),
        (),
    )


def _classification_signature(classification: DocumentClassification) -> tuple[object, ...]:
    dimensions = tuple(
        sorted(
            (dimension.dimension_key, tuple(sorted(dimension.option_paths)))
            for dimension in classification.dimensions
        )
    )
    return classification.status, dimensions


def _identifier(window_number: int, kind: str, item_number: int) -> str:
    return f"window-{window_number}-{kind}-{item_number}"


__all__ = ["aggregate_window_interpretations"]
