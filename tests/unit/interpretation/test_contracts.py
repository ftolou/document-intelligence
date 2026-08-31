from __future__ import annotations

import pytest
from pydantic import ValidationError

from receipt_intelligence.interpretation import (
    CandidateEntity,
    CandidateEntityReference,
    CandidateFact,
    ClassificationDimension,
    ClassificationDimensionResult,
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
    LiteralType,
    LiteralValue,
    Mention,
    NormalizationStatus,
    ReviewSeverity,
    ReviewSignal,
    SourcePageReference,
)


def _specification() -> InterpretationSpecification:
    return InterpretationSpecification(
        specification_id="caller-spec-v1",
        description="Interpret the supplied document.",
        classifications=(
            ClassificationDimension(
                key="document_kind",
                description="The caller's document classification.",
                options=(
                    ClassificationOption(
                        key="record",
                        description="A record.",
                        children=(
                            ClassificationOption(
                                key="supported", description="A supported record."
                            ),
                            ClassificationOption(key="other", description="Another record."),
                        ),
                    ),
                ),
            ),
        ),
        fields=(InterpretationField(key="reference", description="A reference."),),
    )


def _classification() -> DocumentClassification:
    return DocumentClassification(
        status=ClassificationStatus.CLASSIFIED,
        dimensions=(
            ClassificationDimensionResult(
                dimension_key="document_kind",
                option_paths=(("record", "supported"),),
                confidence=0.9,
            ),
        ),
    )


def test_specification_supports_flat_and_hierarchical_caller_fields() -> None:
    specification = InterpretationSpecification(
        specification_id="caller-spec-v1",
        description="Interpret the supplied document.",
        classifications=(
            ClassificationDimension(
                key="document_kind",
                description="The document kind.",
                options=(
                    ClassificationOption(key="supported", description="A supported document."),
                ),
            ),
        ),
        fields=(
            InterpretationField(key="reference", description="A top-level reference."),
            InterpretationField(
                key="party",
                description="A party described by the document.",
                children=(InterpretationField(key="name", description="The observed party name."),),
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
        specification=_specification(),
        classification=_classification(),
        document_map=DocumentMap(
            nodes=(DocumentMapNode(node_id="section-1", label="Header", evidence_refs=("e-1",)),)
        ),
        evidence=(
            EvidenceReference(
                evidence_id="e-1",
                source_id="document-1",
                locator="characters 0-15",
                excerpt=" Amount: 12,? ",
            ),
            EvidenceReference(
                evidence_id="e-2",
                source_id="document-1",
                locator="characters 8-12",
                excerpt="12,?",
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
                object=LiteralValue(
                    literal_type=LiteralType.AMOUNT,
                    observed="12,?",
                    normalization_status=NormalizationStatus.FAILED,
                    currency="EUR",
                ),
                evidence_refs=("e-1", "e-2"),
            ),
        ),
        review_signals=(
            ReviewSignal(
                code="ambiguous_value",
                message="The observed amount is malformed.",
                severity=ReviewSeverity.REVIEW_REQUIRED,
                evidence_refs=("e-1",),
                fact_refs=("fact-1",),
            ),
        ),
    )

    fact = interpretation.candidate_facts[0]
    assert isinstance(fact.subject, CandidateEntityReference)
    assert isinstance(fact.object, LiteralValue)
    assert fact.subject.candidate_entity_id == "entity-1"
    assert fact.predicate == "observed_amount"
    assert fact.object.observed == "12,?"
    assert fact.object.normalized is None
    assert fact.object.normalization_status is NormalizationStatus.FAILED
    assert fact.object.currency == "EUR"
    assert fact.evidence_refs == ("e-1", "e-2")
    assert interpretation.classification.dimensions[0].option_paths == (("record", "supported"),)
    assert interpretation.evidence[0].excerpt == " Amount: 12,? "
    assert interpretation.requires_review is True


def test_interpretation_rejects_dangling_evidence_and_entity_references() -> None:
    source = DocumentSource(source_id="document-1", media_type="text/plain")
    specification = _specification()
    classification = _classification()

    with pytest.raises(ValidationError, match="Unknown evidence"):
        DocumentInterpretation(
            source=source,
            specification=specification,
            classification=classification,
            candidate_facts=(
                CandidateFact(
                    fact_id="fact-1",
                    subject=DocumentReference(source_id="document-1"),
                    predicate="reference",
                    object=LiteralValue(literal_type=LiteralType.IDENTIFIER, observed="A-1"),
                    evidence_refs=("missing",),
                ),
            ),
        )

    with pytest.raises(ValidationError, match="unknown candidate entity"):
        DocumentInterpretation(
            source=source,
            specification=specification,
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
                    object=LiteralValue(literal_type=LiteralType.IDENTIFIER, observed="A-1"),
                    evidence_refs=("e-1",),
                ),
            ),
        )

    with pytest.raises(ValidationError, match="Unknown mention references"):
        DocumentInterpretation(
            source=source,
            specification=specification,
            classification=classification,
            candidate_entities=(
                CandidateEntity(
                    candidate_entity_id="entity-1",
                    entity_type="document_party",
                    mention_refs=("missing-mention",),
                ),
            ),
        )


def test_candidate_fact_rejects_multiple_objects() -> None:
    literal = {
        "kind": "literal",
        "literal_type": "identifier",
        "observed": "A-1",
    }

    with pytest.raises(ValidationError):
        CandidateFact.model_validate(
            {
                "fact_id": "fact-1",
                "subject": {"kind": "document", "source_id": "document-1"},
                "predicate": "reference",
                "object": [literal, literal],
                "evidence_refs": ["e-1"],
            }
        )


def test_unsupported_classification_is_an_explicit_safe_fallback() -> None:
    fallback = DocumentClassification(
        status=ClassificationStatus.UNSUPPORTED,
        reason="No caller-supplied classification applies.",
    )

    assert fallback.dimensions == ()

    with pytest.raises(ValidationError, match="cannot have classification selections"):
        DocumentClassification(
            status=ClassificationStatus.UNSUPPORTED,
            dimensions=(
                ClassificationDimensionResult(
                    dimension_key="document_kind",
                    option_paths=(("other",),),
                ),
            ),
            reason="No classification applies.",
        )


def test_classification_is_bounded_by_caller_dimensions_options_and_cardinality() -> None:
    source = DocumentSource(source_id="document-1", media_type="text/plain")
    specification = _specification()

    with pytest.raises(ValidationError, match="requires classification selections"):
        DocumentClassification(status=ClassificationStatus.CLASSIFIED)

    with pytest.raises(ValidationError, match="caller-supplied dimensions"):
        DocumentInterpretation(
            source=source,
            specification=specification,
            classification=DocumentClassification(
                status=ClassificationStatus.CLASSIFIED,
                dimensions=(
                    ClassificationDimensionResult(
                        dimension_key="not_declared",
                        option_paths=(("record", "supported"),),
                    ),
                ),
            ),
        )

    with pytest.raises(ValidationError, match="Unknown classification option path"):
        DocumentInterpretation(
            source=source,
            specification=specification,
            classification=DocumentClassification(
                status=ClassificationStatus.CLASSIFIED,
                dimensions=(
                    ClassificationDimensionResult(
                        dimension_key="document_kind",
                        option_paths=(("record", "not_allowed"),),
                    ),
                ),
            ),
        )

    with pytest.raises(ValidationError, match="selection cardinality"):
        DocumentInterpretation(
            source=source,
            specification=specification,
            classification=DocumentClassification(
                status=ClassificationStatus.CLASSIFIED,
                dimensions=(
                    ClassificationDimensionResult(
                        dimension_key="document_kind",
                        option_paths=(),
                    ),
                ),
            ),
        )

    with pytest.raises(ValidationError, match="less than or equal to 1"):
        ClassificationDimensionResult(
            dimension_key="document_kind",
            option_paths=(("record", "supported"),),
            confidence=1.01,
        )

    with pytest.raises(ValidationError, match="at most 8 items"):
        ClassificationDimensionResult(
            dimension_key="document_kind",
            option_paths=(("1", "2", "3", "4", "5", "6", "7", "8", "9"),),
        )


def test_evidence_supports_paginated_and_non_paginated_source_locations() -> None:
    text_evidence = EvidenceReference(
        evidence_id="text-evidence",
        source_id="text-document",
        locator="characters 10-24",
    )
    audio_evidence = EvidenceReference(
        evidence_id="audio-evidence",
        source_id="audio-document",
        locator="00:01:12.500-00:01:16.000",
    )
    page_evidence = EvidenceReference(
        evidence_id="page-evidence",
        source_id="paged-document",
        page=SourcePageReference(page_number=2),
    )

    assert text_evidence.page is None
    assert text_evidence.locator == "characters 10-24"
    assert audio_evidence.page is None
    assert audio_evidence.locator == "00:01:12.500-00:01:16.000"
    assert page_evidence.page == SourcePageReference(page_number=2)


def test_evidence_requires_a_valid_source_location() -> None:
    with pytest.raises(ValidationError, match="requires a locator or page"):
        EvidenceReference(evidence_id="e-1", source_id="document-1")

    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        SourcePageReference(page_number=0)


@pytest.mark.parametrize(
    ("literal_type", "normalized", "currency", "unit"),
    [
        (LiteralType.TEXT, "normalized text", None, None),
        (LiteralType.IDENTIFIER, "A-1", None, None),
        (LiteralType.DATE, "2026-08-31", None, None),
        (LiteralType.TIME, "14:30:00", None, None),
        (LiteralType.DATETIME, "2026-08-31T14:30:00Z", None, None),
        (LiteralType.AMOUNT, "12.50", "EUR", None),
        (LiteralType.MEASUREMENT, 42, None, "kg"),
        (LiteralType.NUMBER, 7, None, None),
        (LiteralType.BOOLEAN, False, None, None),
    ],
)
def test_literal_normalization_preserves_structural_type_and_metadata(
    literal_type: LiteralType,
    normalized: str | int | float | bool,
    currency: str | None,
    unit: str | None,
) -> None:
    value = LiteralValue(
        literal_type=literal_type,
        observed=" source content ",
        normalization_status=NormalizationStatus.NORMALIZED,
        normalized=normalized,
        currency=currency,
        unit=unit,
    )

    assert value.observed == " source content "
    assert value.literal_type is literal_type
    assert value.normalized == normalized
    assert LiteralValue.model_validate_json(value.model_dump_json()) == value


@pytest.mark.parametrize(
    "status",
    [
        NormalizationStatus.NOT_ATTEMPTED,
        NormalizationStatus.FAILED,
        NormalizationStatus.UNSAFE,
    ],
)
def test_literal_non_normalized_states_preserve_malformed_observation(
    status: NormalizationStatus,
) -> None:
    value = LiteralValue(
        literal_type=LiteralType.DATE,
        observed="2026-99-?",
        normalization_status=status,
    )

    assert value.normalized is None
    assert value.observed == "2026-99-?"


def test_literal_rejects_inconsistent_normalization_and_metadata() -> None:
    with pytest.raises(ValidationError, match="require a normalized value"):
        LiteralValue(
            literal_type=LiteralType.NUMBER,
            observed="seven",
            normalization_status=NormalizationStatus.NORMALIZED,
        )

    with pytest.raises(ValidationError, match="does not match the literal type"):
        LiteralValue(
            literal_type=LiteralType.BOOLEAN,
            observed="yes",
            normalization_status=NormalizationStatus.NORMALIZED,
            normalized="true",
        )

    with pytest.raises(ValidationError, match="Currency is valid only"):
        LiteralValue(literal_type=LiteralType.TEXT, observed="EUR", currency="EUR")

    with pytest.raises(ValidationError, match="Unit is valid only"):
        LiteralValue(literal_type=LiteralType.NUMBER, observed="12", unit="kg")


@pytest.mark.parametrize(
    ("literal_type", "normalized"),
    [
        (LiteralType.DATE, "not-a-date"),
        (LiteralType.DATE, "2026-02-30"),
        (LiteralType.TIME, "25:00:00"),
        (LiteralType.TIME, "14:30"),
        (LiteralType.DATETIME, "2026-08-31 14:30:00"),
        (LiteralType.DATETIME, "2026-02-30T14:30:00Z"),
        (LiteralType.AMOUNT, 12.5),
        (LiteralType.AMOUNT, "12,50"),
        (LiteralType.AMOUNT, "NaN"),
        (LiteralType.AMOUNT, "Infinity"),
        (LiteralType.NUMBER, float("nan")),
        (LiteralType.NUMBER, float("inf")),
        (LiteralType.MEASUREMENT, float("-inf")),
    ],
)
def test_literal_rejects_malformed_or_non_finite_normalized_values(
    literal_type: LiteralType, normalized: str | float
) -> None:
    with pytest.raises(ValidationError):
        LiteralValue(
            literal_type=literal_type,
            observed="source content",
            normalization_status=NormalizationStatus.NORMALIZED,
            normalized=normalized,
        )


def test_review_signal_rejects_dangling_candidate_fact_reference() -> None:
    with pytest.raises(ValidationError, match="Unknown candidate fact references"):
        DocumentInterpretation(
            source=DocumentSource(source_id="document-1", media_type="text/plain"),
            specification=_specification(),
            classification=_classification(),
            review_signals=(
                ReviewSignal(
                    code="normalization_failed",
                    message="The candidate fact could not be normalized.",
                    fact_refs=("missing-fact",),
                ),
            ),
        )
