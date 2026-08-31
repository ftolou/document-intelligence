"""Generic value contracts for evidence-backed document interpretation.

These models describe inputs and outputs only. They deliberately contain no
model invocation, document processing, persistence, entity resolution, or
domain-specific taxonomy.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_SPECIFICATION_DEPTH = 8
MAX_SPECIFICATION_NODES = 256
MAX_COLLECTION_SIZE = 256

NonBlankText = Annotated[str, Field(min_length=1, max_length=4000, pattern=r"\S")]
Identifier = Annotated[str, Field(min_length=1, max_length=200, pattern=r"\S")]
JsonScalar: TypeAlias = str | int | float | bool | None


class ContractModel(BaseModel):
    """Immutable, closed value object used at the interpretation boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DocumentSource(ContractModel):
    """Opaque identity and media metadata for one caller-owned document."""

    source_id: Identifier
    media_type: Identifier
    name: NonBlankText | None = None


class ClassificationOption(ContractModel):
    """One caller-defined classification option, not a global registry entry."""

    key: Identifier
    description: NonBlankText


class InterpretationField(ContractModel):
    """One requested concept in a flat or hierarchical interpretation spec."""

    key: Identifier
    description: NonBlankText
    children: tuple[InterpretationField, ...] = Field(
        default=(), max_length=MAX_COLLECTION_SIZE
    )

    @model_validator(mode="after")
    def validate_unique_children(self) -> Self:
        keys = [child.key for child in self.children]
        if len(keys) != len(set(keys)):
            raise ValueError("InterpretationField child keys must be unique among siblings.")
        return self


class InterpretationSpecification(ContractModel):
    """Bounded, caller-supplied guidance for interpreting one document.

    Field nodes may be flat or nested. The fixed limits keep arbitrary caller
    input bounded without introducing a document-type or predicate registry.
    """

    specification_id: Identifier
    description: NonBlankText
    classifications: tuple[ClassificationOption, ...] = Field(
        default=(), max_length=MAX_COLLECTION_SIZE
    )
    fields: tuple[InterpretationField, ...] = Field(
        default=(), max_length=MAX_COLLECTION_SIZE
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if not self.classifications and not self.fields:
            raise ValueError("InterpretationSpecification must request classifications or fields.")

        classification_keys = [option.key for option in self.classifications]
        if len(classification_keys) != len(set(classification_keys)):
            raise ValueError("Classification option keys must be unique.")

        root_keys = [field.key for field in self.fields]
        if len(root_keys) != len(set(root_keys)):
            raise ValueError("InterpretationField root keys must be unique.")

        node_count = 0
        stack = [(field, 1) for field in self.fields]
        while stack:
            node, depth = stack.pop()
            node_count += 1
            if node_count > MAX_SPECIFICATION_NODES:
                raise ValueError(
                    f"InterpretationSpecification supports at most {MAX_SPECIFICATION_NODES} fields."
                )
            if depth > MAX_SPECIFICATION_DEPTH:
                raise ValueError(
                    f"InterpretationSpecification supports at most {MAX_SPECIFICATION_DEPTH} levels."
                )
            stack.extend((child, depth + 1) for child in node.children)
        return self


class DocumentInterpretationRequest(ContractModel):
    """A document source paired with its caller-supplied interpretation spec."""

    source: DocumentSource
    specification: InterpretationSpecification


class EvidenceReference(ContractModel):
    """A stable reference to source evidence; ``excerpt`` preserves source text."""

    evidence_id: Identifier
    source_id: Identifier
    locator: NonBlankText
    excerpt: str | None = Field(default=None, max_length=10000)


class ClassificationStatus(StrEnum):
    CLASSIFIED = "classified"
    UNSUPPORTED = "unsupported"


class DocumentClassification(ContractModel):
    """Classification outcome with an explicit safe unsupported fallback."""

    status: ClassificationStatus
    label: Identifier | None = None
    reason: NonBlankText | None = None
    evidence_refs: tuple[Identifier, ...] = Field(default=(), max_length=MAX_COLLECTION_SIZE)

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.status is ClassificationStatus.CLASSIFIED and self.label is None:
            raise ValueError("A classified document requires a label.")
        if self.status is ClassificationStatus.UNSUPPORTED:
            if self.label is not None:
                raise ValueError("An unsupported document cannot have a classification label.")
            if self.reason is None:
                raise ValueError("An unsupported document requires a reason.")
        return self


class DocumentMapNode(ContractModel):
    """One logical region in a hierarchical map of a document."""

    node_id: Identifier
    label: NonBlankText
    evidence_refs: tuple[Identifier, ...] = Field(default=(), max_length=MAX_COLLECTION_SIZE)
    children: tuple[DocumentMapNode, ...] = Field(default=(), max_length=MAX_COLLECTION_SIZE)


class DocumentMap(ContractModel):
    """Logical document structure, independent of a file format or OCR engine."""

    nodes: tuple[DocumentMapNode, ...] = Field(default=(), max_length=MAX_COLLECTION_SIZE)


class Mention(ContractModel):
    """Observed source text that may refer to an entity or value."""

    mention_id: Identifier
    observed_text: str = Field(min_length=1, max_length=10000)
    evidence_refs: tuple[Identifier, ...] = Field(min_length=1, max_length=MAX_COLLECTION_SIZE)


class CandidateEntity(ContractModel):
    """A document-scoped candidate, explicitly not a persistent resolved entity."""

    candidate_entity_id: Identifier
    entity_type: Identifier
    display_name: NonBlankText | None = None
    mention_refs: tuple[Identifier, ...] = Field(default=(), max_length=MAX_COLLECTION_SIZE)
    evidence_refs: tuple[Identifier, ...] = Field(default=(), max_length=MAX_COLLECTION_SIZE)


class DocumentReference(ContractModel):
    """Reference to the interpreted document when it is a fact subject."""

    kind: Literal["document"] = "document"
    source_id: Identifier


class CandidateEntityReference(ContractModel):
    """Reference to a document-scoped candidate entity."""

    kind: Literal["candidate_entity"] = "candidate_entity"
    candidate_entity_id: Identifier


class LiteralValue(ContractModel):
    """Literal preserving observed content separately from optional normalization.

    ``observed`` is never silently stripped or corrected. Ambiguous or malformed
    source content remains valid even when no normalized value can be supplied.
    """

    kind: Literal["literal"] = "literal"
    observed: str = Field(min_length=1, max_length=10000)
    normalized: JsonScalar = None


FactSubject: TypeAlias = Annotated[
    DocumentReference | CandidateEntityReference,
    Field(discriminator="kind"),
]
FactObject: TypeAlias = Annotated[
    LiteralValue | CandidateEntityReference,
    Field(discriminator="kind"),
]


class CandidateFact(ContractModel):
    """One atomic candidate assertion, never an authoritative persisted fact."""

    fact_id: Identifier
    subject: FactSubject
    predicate: Identifier
    object: FactObject
    evidence_refs: tuple[Identifier, ...] = Field(min_length=1, max_length=MAX_COLLECTION_SIZE)


class ReviewSeverity(StrEnum):
    WARNING = "warning"
    REVIEW_REQUIRED = "review_required"


class ReviewSignal(ContractModel):
    """Generic warning or explicit Human Review signal."""

    code: Identifier
    message: NonBlankText
    severity: ReviewSeverity = ReviewSeverity.WARNING
    evidence_refs: tuple[Identifier, ...] = Field(default=(), max_length=MAX_COLLECTION_SIZE)


class DocumentInterpretation(ContractModel):
    """Evidence-backed, document-scoped interpretation result.

    Cross-reference validation ensures that candidates and facts cannot claim
    evidence, mentions, entities, or documents absent from this result.
    """

    source: DocumentSource
    classification: DocumentClassification
    document_map: DocumentMap = Field(default_factory=DocumentMap)
    mentions: tuple[Mention, ...] = Field(default=(), max_length=MAX_COLLECTION_SIZE)
    candidate_entities: tuple[CandidateEntity, ...] = Field(
        default=(), max_length=MAX_COLLECTION_SIZE
    )
    candidate_facts: tuple[CandidateFact, ...] = Field(
        default=(), max_length=MAX_COLLECTION_SIZE
    )
    evidence: tuple[EvidenceReference, ...] = Field(default=(), max_length=MAX_COLLECTION_SIZE)
    review_signals: tuple[ReviewSignal, ...] = Field(
        default=(), max_length=MAX_COLLECTION_SIZE
    )

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        evidence_ids = _unique_ids("evidence", (item.evidence_id for item in self.evidence))
        mention_ids = _unique_ids("mention", (item.mention_id for item in self.mentions))
        entity_ids = _unique_ids(
            "candidate entity", (item.candidate_entity_id for item in self.candidate_entities)
        )
        _unique_ids("candidate fact", (item.fact_id for item in self.candidate_facts))

        invalid_sources = {
            item.source_id for item in self.evidence if item.source_id != self.source.source_id
        }
        if invalid_sources:
            raise ValueError("Evidence references must target the interpretation source.")

        map_nodes = _flatten_map(self.document_map.nodes)
        _unique_ids("document map node", (node.node_id for node in map_nodes))

        evidence_owners: list[
            DocumentClassification
            | DocumentMapNode
            | Mention
            | CandidateEntity
            | CandidateFact
            | ReviewSignal
        ] = [self.classification, *map_nodes, *self.mentions]
        evidence_owners.extend(self.candidate_entities)
        evidence_owners.extend(self.candidate_facts)
        evidence_owners.extend(self.review_signals)
        for owner in evidence_owners:
            missing = set(owner.evidence_refs) - evidence_ids
            if missing:
                raise ValueError(f"Unknown evidence references: {sorted(missing)!r}.")

        for entity in self.candidate_entities:
            missing = set(entity.mention_refs) - mention_ids
            if missing:
                raise ValueError(f"Unknown mention references: {sorted(missing)!r}.")

        for fact in self.candidate_facts:
            if isinstance(fact.subject, DocumentReference):
                if fact.subject.source_id != self.source.source_id:
                    raise ValueError("Candidate fact document subjects must target the source.")
            elif fact.subject.candidate_entity_id not in entity_ids:
                raise ValueError("Candidate fact subject references an unknown candidate entity.")
            if (
                isinstance(fact.object, CandidateEntityReference)
                and fact.object.candidate_entity_id not in entity_ids
            ):
                raise ValueError("Candidate fact object references an unknown candidate entity.")
        return self

    @property
    def requires_review(self) -> bool:
        return any(
            signal.severity is ReviewSeverity.REVIEW_REQUIRED for signal in self.review_signals
        )


def _unique_ids(kind: str, values: Iterable[str]) -> set[str]:
    identifiers = list(values)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{kind.capitalize()} IDs must be unique.")
    return set(identifiers)


def _flatten_map(nodes: tuple[DocumentMapNode, ...]) -> list[DocumentMapNode]:
    flattened: list[DocumentMapNode] = []
    stack = list(nodes)
    while stack:
        node = stack.pop()
        flattened.append(node)
        stack.extend(node.children)
    return flattened


__all__ = [
    "CandidateEntity",
    "CandidateEntityReference",
    "CandidateFact",
    "ClassificationOption",
    "ClassificationStatus",
    "DocumentClassification",
    "DocumentInterpretation",
    "DocumentInterpretationRequest",
    "DocumentMap",
    "DocumentMapNode",
    "DocumentReference",
    "DocumentSource",
    "EvidenceReference",
    "FactObject",
    "FactSubject",
    "InterpretationField",
    "InterpretationSpecification",
    "JsonScalar",
    "LiteralValue",
    "MAX_SPECIFICATION_DEPTH",
    "MAX_SPECIFICATION_NODES",
    "Mention",
    "ReviewSeverity",
    "ReviewSignal",
]
