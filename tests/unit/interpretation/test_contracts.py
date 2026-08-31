from __future__ import annotations

import pytest
from pydantic import ValidationError

from receipt_intelligence.interpretation import (
    CandidateEntity,
    CandidateEntityReference,
    CandidateFact,
    ClassificationOption,
    ClassificationStatus,
    DocumentClassification,
    DocumentInterpretation,
    DocumentInterpretationRequest,
    DocumentMap,
    DocumentMapNode,
    DocumentReference,
    DocumentSource,
    EvidenceReference,
    InterpretationField,
    InterpretationSpecification,
    LiteralValue,
    Mention,
    ReviewSeverity,
    ReviewSignal,
)


def test_specification_supports_flat_and_hierarchical_caller_fields() -> None:
    specification = InterpretationSpecification(
        specification_id="caller-spec-v1",
        description="Interpret the supplied document.",
        classifications=(
            ClassificationOption(key="supported", description="A supported document."),
        ),
        fields=(
            InterpretationField(key="reference", description="A top-level reference."),
            InterpretationField(
                key="party",
                description="A party described by the document.",
                children=(
                    InterpretationField(key="name", description="The observed party name."),
                ),
            ),
        ),
    )

    assert specification.fields[0].children == ()
    assert specification.fields[1].children[0].key == "name"

    request = DocumentInterpretationRequest(
        source=DocumentSource(source_id="document-1", media_type="text/plain"),
        specification=specification,
    )
    assert request.specification.specification_id == "caller-spec-v1"


def test_specification_rejects_unbounded_depth() -> None:
    field = InterpretationField(key="level-9", description="Deepest field.")
    for depth in range(8, 0, -1):
        field = InterpretationField(
            key=f"level-{depth}",
            description="Nested field.",
            children=(field,),
        )

    with pytest.raises(ValidationError, match="at most 8 levels"):
        InterpretationSpecification(
            specification_id="too-deep",
            description="An invalid specification.",
            fields=(field,),
        )


def test_interpretation_represents_atomic_evidence_backed_candidate_fact() -> None:
    interpretation = DocumentInterpretation(
        source=DocumentSource(
            source_id="document-1",
            media_type="text/plain",
            name="example.txt",
        ),
        classification=DocumentClassification(
            status=ClassificationStatus.CLASSIFIED,
            label="supported",
            evidence_refs=("e-1",),
        ),
        document_map=DocumentMap(
            nodes=(
                DocumentMapNode(node_id="section-1", label="Header", evidence_refs=("e-1",)),
            )
        ),
        evidence=(
            EvidenceReference(
                evidence_id="e-1",
                source_id="document-1",
                locator="characters 0-15",
                excerpt=" Amount: 12,? ",
            ),
        ),
        mentions=(
            Mention(
                mention_id="m-1",
                observed_text=" Amount: 12,? ",
                evidence_refs=("e-1",),
            ),
        ),
        candidate_entities=(
            CandidateEntity(
                candidate_entity_id="entity-1",
                entity_type="document_party",
                mention_refs=("m-1",),
                evidence_refs=("e-1",),
            ),
        ),
        candidate_facts=(
            CandidateFact(
                fact_id="fact-1",
                subject=CandidateEntityReference(candidate_entity_id="entity-1"),
                predicate="observed_amount",
                object=LiteralValue(observed="12,?"),
                evidence_refs=("e-1",),
            ),
        ),
        review_signals=(
            ReviewSignal(
                code="ambiguous_value",
                message="The observed amount is malformed.",
                severity=ReviewSeverity.REVIEW_REQUIRED,
                evidence_refs=("e-1",),
            ),
        ),
    )

    fact = interpretation.candidate_facts[0]
    assert fact.subject.candidate_entity_id == "entity-1"
    assert fact.predicate == "observed_amount"
    assert fact.object.observed == "12,?"
    assert fact.object.normalized is None
    assert interpretation.evidence[0].excerpt == " Amount: 12,? "
    assert interpretation.requires_review is True


def test_interpretation_rejects_dangling_evidence_and_entity_references() -> None:
    source = DocumentSource(source_id="document-1", media_type="text/plain")
    classification = DocumentClassification(
        status=ClassificationStatus.CLASSIFIED,
        label="supported",
    )

    with pytest.raises(ValidationError, match="Unknown evidence"):
        DocumentInterpretation(
            source=source,
            classification=classification,
            candidate_facts=(
                CandidateFact(
                    fact_id="fact-1",
                    subject=DocumentReference(source_id="document-1"),
                    predicate="reference",
                    object=LiteralValue(observed="A-1"),
                    evidence_refs=("missing",),
                ),
            ),
        )

    with pytest.raises(ValidationError, match="unknown candidate entity"):
        DocumentInterpretation(
            source=source,
            classification=classification,
            evidence=(
                EvidenceReference(
                    evidence_id="e-1",
                    source_id="document-1",
                    locator="line 1",
                ),
            ),
            candidate_facts=(
                CandidateFact(
                    fact_id="fact-1",
                    subject=CandidateEntityReference(candidate_entity_id="missing"),
                    predicate="reference",
                    object=LiteralValue(observed="A-1"),
                    evidence_refs=("e-1",),
                ),
            ),
        )


def test_unsupported_classification_is_an_explicit_safe_fallback() -> None:
    fallback = DocumentClassification(
        status=ClassificationStatus.UNSUPPORTED,
        reason="No caller-supplied classification applies.",
    )

    assert fallback.label is None

    with pytest.raises(ValidationError, match="cannot have a classification label"):
        DocumentClassification(
            status=ClassificationStatus.UNSUPPORTED,
            label="other",
            reason="No classification applies.",
        )
