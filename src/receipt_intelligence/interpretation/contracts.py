"""Generic value contracts for evidence-backed document interpretation.

These models describe inputs and outputs only. They deliberately contain no
model invocation, document processing, persistence, entity resolution, or
domain-specific taxonomy.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import date, datetime, time
from enum import StrEnum
from math import isfinite
from re import fullmatch
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_SPECIFICATION_DEPTH = 8
MAX_SPECIFICATION_NODES = 256
MAX_COLLECTION_SIZE = 256

NonBlankText = Annotated[str, Field(min_length=1, max_length=4000, pattern=r"\S")]
Identifier = Annotated[str, Field(min_length=1, max_length=200, pattern=r"\S")]
JsonScalar: TypeAlias = str | int | float | bool | None
ClassificationOptionPath: TypeAlias = Annotated[
    tuple[Identifier, ...], Field(min_length=1, max_length=MAX_SPECIFICATION_DEPTH)
]

_DATE_PATTERN = r"[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
_TIME_PATTERN = (
    r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]"
    r"(?:\.[0-9]{1,6})?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])?"
)
_DATETIME_PATTERN = rf"{_DATE_PATTERN}T{_TIME_PATTERN}"
_DECIMAL_PATTERN = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?"


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
    children: tuple[ClassificationOption, ...] = Field(default=(), max_length=MAX_COLLECTION_SIZE)

    @model_validator(mode="after")
    def validate_unique_children(self) -> Self:
        keys = [child.key for child in self.children]
        if len(keys) != len(set(keys)):
            raise ValueError("Classification option keys must be unique among siblings.")
        return self


class ClassificationDimension(ContractModel):
    """A caller-defined classification dimension and its allowed choices."""

    key: Identifier
    description: NonBlankText
    options: tuple[ClassificationOption, ...] = Field(min_length=1, max_length=MAX_COLLECTION_SIZE)
    min_selections: int = Field(default=1, ge=0, le=MAX_COLLECTION_SIZE)
    max_selections: int = Field(default=1, ge=1, le=MAX_COLLECTION_SIZE)

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.min_selections > self.max_selections:
            raise ValueError("Classification minimum selections cannot exceed the maximum.")

        root_keys = [option.key for option in self.options]
        if len(root_keys) != len(set(root_keys)):
            raise ValueError("Classification option keys must be unique among siblings.")

        node_count = 0
        stack = [(option, 1) for option in self.options]
        while stack:
            option, depth = stack.pop()
            node_count += 1
            if node_count > MAX_SPECIFICATION_NODES:
                raise ValueError(
                    f"ClassificationDimension supports at most {MAX_SPECIFICATION_NODES} options."
                )
            if depth > MAX_SPECIFICATION_DEPTH:
                raise ValueError(
                    f"ClassificationDimension supports at most {MAX_SPECIFICATION_DEPTH} levels."
                )
            stack.extend((child, depth + 1) for child in option.children)

        selectable_count = sum(
            1 for option in _flatten_options(self.options) if not option.children
        )
        if self.max_selections > selectable_count:
            raise ValueError("Classification maximum selections exceeds the selectable options.")
        return self


class InterpretationField(ContractModel):
    """One requested concept in a flat or hierarchical interpretation spec."""

    key: Identifier
    description: NonBlankText
    children: tuple[InterpretationField, ...] = Field(default=(), max_length=MAX_COLLECTION_SIZE)

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
    classifications: tuple[ClassificationDimension, ...] = Field(
        default=(), max_length=MAX_COLLECTION_SIZE
    )
    fields: tuple[InterpretationField, ...] = Field(default=(), max_length=MAX_COLLECTION_SIZE)

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if not self.classifications and not self.fields:
            raise ValueError("InterpretationSpecification must request classifications or fields.")

        classification_keys = [dimension.key for dimension in self.classifications]
        if len(classification_keys) != len(set(classification_keys)):
            raise ValueError("Classification dimension keys must be unique.")

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


class SourcePageReference(ContractModel):
    """A validated one-based source page plus an optional finer-grained locator."""

    page_number: int = Field(ge=1)
    locator: NonBlankText | None = None


class EvidenceReference(ContractModel):
    """A stable reference to source evidence; ``excerpt`` preserves source text."""

    evidence_id: Identifier
    source_id: Identifier
    page: SourcePageReference
    excerpt: str | None = Field(default=None, max_length=10000)


class ClassificationStatus(StrEnum):
    CLASSIFIED = "classified"
    UNSUPPORTED = "unsupported"


class ClassificationDimensionResult(ContractModel):
    """Selections for one caller-defined classification dimension."""

    dimension_key: Identifier
    option_paths: tuple[ClassificationOptionPath, ...] = Field(
        default=(), max_length=MAX_COLLECTION_SIZE
    )
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence_refs: tuple[Identifier, ...] = Field(default=(), max_length=MAX_COLLECTION_SIZE)

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        if len(self.option_paths) != len(set(self.option_paths)):
            raise ValueError("Classification option paths must be unique.")
        return self


class DocumentClassification(ContractModel):
    """Classification outcome with an explicit safe unsupported fallback."""

    status: ClassificationStatus
    dimensions: tuple[ClassificationDimensionResult, ...] = Field(
        default=(), max_length=MAX_COLLECTION_SIZE
    )
    reason: NonBlankText | None = None
    evidence_refs: tuple[Identifier, ...] = Field(default=(), max_length=MAX_COLLECTION_SIZE)

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.status is ClassificationStatus.CLASSIFIED and not self.dimensions:
            raise ValueError("A classified document requires classification selections.")
        if self.status is ClassificationStatus.UNSUPPORTED:
            if self.dimensions:
                raise ValueError("An unsupported document cannot have classification selections.")
            if self.reason is None:
                raise ValueError("An unsupported document requires a reason.")
        dimension_keys = [dimension.dimension_key for dimension in self.dimensions]
        if len(dimension_keys) != len(set(dimension_keys)):
            raise ValueError("Classification dimension results must be unique.")
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


class LiteralType(StrEnum):
    """Provider-neutral structural type of an observed literal."""

    TEXT = "text"
    IDENTIFIER = "identifier"
    DATE = "date"
    TIME = "time"
    DATETIME = "datetime"
    AMOUNT = "amount"
    MEASUREMENT = "measurement"
    NUMBER = "number"
    BOOLEAN = "boolean"


class NormalizationStatus(StrEnum):
    """Outcome of an optional normalization attempt."""

    NOT_ATTEMPTED = "not_attempted"
    NORMALIZED = "normalized"
    FAILED = "failed"
    UNSAFE = "unsafe"


class LiteralValue(ContractModel):
    """Literal preserving observed content separately from optional normalization.

    ``observed`` is never silently stripped or corrected. Ambiguous or malformed
    source content remains valid even when no normalized value can be supplied.
    """

    kind: Literal["literal"] = "literal"
    literal_type: LiteralType
    observed: str = Field(min_length=1, max_length=10000)
    normalization_status: NormalizationStatus = NormalizationStatus.NOT_ATTEMPTED
    normalized: JsonScalar = None
    currency: Identifier | None = None
    unit: NonBlankText | None = None

    @model_validator(mode="after")
    def validate_normalization(self) -> Self:
        if self.normalization_status is NormalizationStatus.NORMALIZED:
            if self.normalized is None:
                raise ValueError("Normalized literals require a normalized value.")
        elif self.normalized is not None:
            raise ValueError("A normalized value requires normalization status 'normalized'.")

        if self.currency is not None and self.literal_type is not LiteralType.AMOUNT:
            raise ValueError("Currency is valid only for amount literals.")
        if self.unit is not None and self.literal_type is not LiteralType.MEASUREMENT:
            raise ValueError("Unit is valid only for measurement literals.")

        normalized = self.normalized
        if normalized is None:
            return self
        if self.literal_type is LiteralType.BOOLEAN:
            valid_type = isinstance(normalized, bool)
        elif self.literal_type is LiteralType.AMOUNT:
            valid_type = isinstance(normalized, str) and fullmatch(
                _DECIMAL_PATTERN, normalized
            ) is not None
        elif self.literal_type in {LiteralType.MEASUREMENT, LiteralType.NUMBER}:
            valid_type = isinstance(normalized, (int, float)) and not isinstance(normalized, bool)
            if valid_type and isinstance(normalized, float) and not isfinite(normalized):
                raise ValueError("Normalized numeric values must be finite.")
        elif self.literal_type is LiteralType.DATE:
            valid_type = isinstance(normalized, str) and _matches_temporal_format(
                normalized, _DATE_PATTERN, date.fromisoformat
            )
        elif self.literal_type is LiteralType.TIME:
            valid_type = isinstance(normalized, str) and _matches_temporal_format(
                normalized, _TIME_PATTERN, time.fromisoformat
            )
        elif self.literal_type is LiteralType.DATETIME:
            valid_type = isinstance(normalized, str) and _matches_temporal_format(
                normalized, _DATETIME_PATTERN, datetime.fromisoformat
            )
        else:
            valid_type = isinstance(normalized, str)
        if not valid_type:
            raise ValueError("Normalized value does not match the literal type.")
        return self


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
    fact_refs: tuple[Identifier, ...] = Field(default=(), max_length=MAX_COLLECTION_SIZE)


class DocumentInterpretation(ContractModel):
    """Evidence-backed, document-scoped interpretation result.

    Cross-reference validation ensures that candidates and facts cannot claim
    evidence, mentions, entities, or documents absent from this result.
    """

    source: DocumentSource
    specification: InterpretationSpecification
    classification: DocumentClassification
    document_map: DocumentMap = Field(default_factory=DocumentMap)
    mentions: tuple[Mention, ...] = Field(default=(), max_length=MAX_COLLECTION_SIZE)
    candidate_entities: tuple[CandidateEntity, ...] = Field(
        default=(), max_length=MAX_COLLECTION_SIZE
    )
    candidate_facts: tuple[CandidateFact, ...] = Field(default=(), max_length=MAX_COLLECTION_SIZE)
    evidence: tuple[EvidenceReference, ...] = Field(default=(), max_length=MAX_COLLECTION_SIZE)
    review_signals: tuple[ReviewSignal, ...] = Field(default=(), max_length=MAX_COLLECTION_SIZE)

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        evidence_ids = _unique_ids("evidence", (item.evidence_id for item in self.evidence))
        mention_ids = _unique_ids("mention", (item.mention_id for item in self.mentions))
        entity_ids = _unique_ids(
            "candidate entity", (item.candidate_entity_id for item in self.candidate_entities)
        )
        fact_ids = _unique_ids("candidate fact", (item.fact_id for item in self.candidate_facts))

        if self.classification.status is ClassificationStatus.CLASSIFIED:
            expected_dimensions = {
                dimension.key: dimension for dimension in self.specification.classifications
            }
            actual_dimensions = {
                dimension.dimension_key: dimension for dimension in self.classification.dimensions
            }
            if actual_dimensions.keys() != expected_dimensions.keys():
                raise ValueError(
                    "Classification results must match the caller-supplied dimensions."
                )
            for key, result in actual_dimensions.items():
                definition = expected_dimensions[key]
                selection_count = len(result.option_paths)
                if not definition.min_selections <= selection_count <= definition.max_selections:
                    raise ValueError(
                        f"Classification dimension {key!r} violates its selection cardinality."
                    )
                for path in result.option_paths:
                    if not _option_path_exists(definition.options, path):
                        raise ValueError(
                            f"Unknown classification option path for dimension {key!r}: {path!r}."
                        )

        invalid_sources = {
            item.source_id for item in self.evidence if item.source_id != self.source.source_id
        }
        if invalid_sources:
            raise ValueError("Evidence references must target the interpretation source.")

        map_nodes = _flatten_map(self.document_map.nodes)
        _unique_ids("document map node", (node.node_id for node in map_nodes))

        evidence_owners: list[
            DocumentClassification
            | ClassificationDimensionResult
            | DocumentMapNode
            | Mention
            | CandidateEntity
            | CandidateFact
            | ReviewSignal
        ] = [
            self.classification,
            *self.classification.dimensions,
            *map_nodes,
            *self.mentions,
        ]
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

        for signal in self.review_signals:
            missing = set(signal.fact_refs) - fact_ids
            if missing:
                raise ValueError(f"Unknown candidate fact references: {sorted(missing)!r}.")

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


def _matches_temporal_format(
    value: str, pattern: str, parser: Callable[[str], object]
) -> bool:
    if fullmatch(pattern, value) is None:
        return False
    try:
        parser(value)
    except ValueError:
        return False
    return True


def _flatten_map(nodes: tuple[DocumentMapNode, ...]) -> list[DocumentMapNode]:
    flattened: list[DocumentMapNode] = []
    stack = list(nodes)
    while stack:
        node = stack.pop()
        flattened.append(node)
        stack.extend(node.children)
    return flattened


def _flatten_options(
    options: tuple[ClassificationOption, ...],
) -> list[ClassificationOption]:
    flattened: list[ClassificationOption] = []
    stack = list(options)
    while stack:
        option = stack.pop()
        flattened.append(option)
        stack.extend(option.children)
    return flattened


def _option_path_exists(options: tuple[ClassificationOption, ...], path: tuple[str, ...]) -> bool:
    remaining = options
    for key in path:
        option = next((item for item in remaining if item.key == key), None)
        if option is None:
            return False
        remaining = option.children
    return not remaining


__all__ = [
    "CandidateEntity",
    "CandidateEntityReference",
    "CandidateFact",
    "ClassificationDimension",
    "ClassificationDimensionResult",
    "ClassificationOption",
    "ClassificationOptionPath",
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
    "LiteralType",
    "MAX_SPECIFICATION_DEPTH",
    "MAX_SPECIFICATION_NODES",
    "Mention",
    "NormalizationStatus",
    "ReviewSeverity",
    "ReviewSignal",
    "SourcePageReference",
]
