"""Test-only builders and readers for the deterministic validation boundary.

Everything the tests assume about the *new* wire shape introduced by the
deterministic validation boundary is defined here and nowhere else, so that a
different (but equivalent) Core naming decision can be adopted by editing this
module only.

The module deliberately contains no production abstraction: it builds model
payloads, writes source files, invokes the existing public one-pass workflow and
reads the completed-workflow result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
import pytest

from receipt_intelligence.application.ports.multimodal import (
    MultimodalGenerationRequest,
    MultimodalGenerationResult,
)
from receipt_intelligence.extraction import SourceNormalizationLimits
from receipt_intelligence.interpretation import (
    ClassificationDimension,
    ClassificationOption,
    DocumentInterpretation,
    DocumentInterpretationRequest,
    DocumentSource,
    InterpretationField,
    InterpretationSpecification,
    OnePassDocumentInterpreter,
)

# --- pinned naming seam -------------------------------------------------------
# WP1 page accounting and WP2/WP6 page ranges are new model-produced information
# and therefore need a concrete payload shape. Only these constants encode it.
PAGE_ACCOUNTING_FIELD = "page_accounting"
PAGE_NUMBER_KEY = "page_number"
PAGE_STATE_KEY = "state"
PAGE_RANGE_KEY = "page_range"
RANGE_START_KEY = "start_page"
RANGE_END_KEY = "end_page"

INTERPRETED = "interpreted"
BLANK = "blank"
IRRELEVANT = "irrelevant"
UNREADABLE = "unreadable"
UNPROCESSED = "unprocessed_review_required"

PAGE_STATES = (INTERPRETED, BLANK, IRRELEVANT, UNREADABLE, UNPROCESSED)

VALID = "VALID"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
INVALID = "INVALID"

SOURCE_ID = "document-1"
FIELD_KEY = "stated_amount"


class RangeMaterialized(BaseException):
    """Raised by the bounded-work sentinel; not catchable by ``except Exception``."""


class RecordingGateway:
    """Returns one canned model response and records the requests it received."""

    def __init__(self, response: dict[str, Any] | str) -> None:
        self._response = response
        self.requests: list[MultimodalGenerationRequest] = []

    def generate(self, request: MultimodalGenerationRequest) -> MultimodalGenerationResult:
        self.requests.append(request)
        text = self._response if isinstance(self._response, str) else json.dumps(self._response)
        return MultimodalGenerationResult(text=text)


def limits(*, max_pages: int = 4) -> SourceNormalizationLimits:
    return SourceNormalizationLimits(
        max_source_bytes=1_000_000,
        max_pages=max_pages,
        max_page_width=100,
        max_page_height=100,
        max_page_pixels=10_000,
        max_total_pixels=40_000,
    )


def write_source(directory: Path, *, pages: int = 1) -> tuple[Path, str]:
    """Write a bounded source with ``pages`` normalized pages."""

    if pages == 1:
        path = directory / "source.png"
        Image.new("RGB", (8, 6), "white").save(path, format="PNG")
        return path, "image/png"

    path = directory / "source.pdf"
    images = [Image.new("RGB", (12, 8), "white") for _ in range(pages)]
    try:
        images[0].save(
            path,
            format="PDF",
            save_all=True,
            append_images=images[1:],
            resolution=72,
        )
    finally:
        for image in images:
            image.close()
    return path, "application/pdf"


def interpretation_request(
    *,
    media_type: str = "image/png",
    field_key: str = FIELD_KEY,
) -> DocumentInterpretationRequest:
    return DocumentInterpretationRequest(
        source=DocumentSource(source_id=SOURCE_ID, media_type=media_type),
        specification=InterpretationSpecification(
            specification_id="caller-spec-v1",
            description="Interpret only the requested record concepts.",
            classifications=(
                ClassificationDimension(
                    key="record_kind",
                    description="The caller's record kind.",
                    options=(
                        ClassificationOption(
                            key="supported_record",
                            description="A supported generic record.",
                        ),
                    ),
                ),
            ),
            fields=(
                InterpretationField(
                    key=field_key,
                    description="A value explicitly stated in the source.",
                ),
            ),
        ),
    )


def page_entry(page_number: int, state: str = INTERPRETED) -> dict[str, Any]:
    return {PAGE_NUMBER_KEY: page_number, PAGE_STATE_KEY: state}


def page_entries(states: tuple[str, ...]) -> list[dict[str, Any]]:
    return [page_entry(index + 1, state) for index, state in enumerate(states)]


def evidence_entry(
    *,
    evidence_id: str = "e-1",
    page_number: int | None = 1,
    page_range: tuple[int, int] | None = None,
    excerpt: str | None = "12.50",
) -> dict[str, Any]:
    entry: dict[str, Any] = {"evidence_id": evidence_id, "source_id": SOURCE_ID}
    if page_number is not None:
        entry["page"] = {"page_number": page_number}
    if page_range is not None:
        start, end = page_range
        # A locator keeps the reference structurally located regardless of how the
        # range itself is modelled.
        entry["locator"] = f"pages {start}-{end}"
        entry[PAGE_RANGE_KEY] = {RANGE_START_KEY: start, RANGE_END_KEY: end}
    if excerpt is not None:
        entry["excerpt"] = excerpt
    return entry


def review_signal(
    *,
    code: str,
    message: str,
    severity: str = "warning",
    fact_refs: tuple[str, ...] = ("fact-1",),
) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "severity": severity,
        "evidence_refs": ["e-1"],
        "fact_refs": list(fact_refs),
    }


def generated_response(
    *,
    page_states: tuple[str, ...] = (INTERPRETED,),
    pages: list[dict[str, Any]] | None = None,
    evidence: list[dict[str, Any]] | None = None,
    review_signals: list[dict[str, Any]] | None = None,
    literal: dict[str, Any] | None = None,
    classification_evidence_refs: list[str] | None = None,
    field_key: str = FIELD_KEY,
) -> dict[str, Any]:
    """Build one structurally valid model response for the one-pass workflow."""

    classification_refs = (
        ["e-1"] if classification_evidence_refs is None else (classification_evidence_refs)
    )
    fact_object = literal or {
        "kind": "literal",
        "literal_type": "amount",
        "observed": "12.50",
        "normalization_status": "normalized",
        "normalized": "12.50",
        "currency": "EUR",
    }
    return {
        "classification": {
            "status": "classified",
            "dimensions": [
                {
                    "dimension_key": "record_kind",
                    "option_paths": [["supported_record"]],
                    "evidence_refs": ["e-1"],
                }
            ],
            "evidence_refs": list(classification_refs),
        },
        "document_map": {
            "nodes": [{"node_id": "section-1", "label": "Statement", "evidence_refs": ["e-1"]}]
        },
        "mentions": [
            {"mention_id": "mention-1", "observed_text": "12.50", "evidence_refs": ["e-1"]}
        ],
        "candidate_entities": [
            {
                "candidate_entity_id": "entity-1",
                "entity_type": "stated_party",
                "mention_refs": ["mention-1"],
                "evidence_refs": ["e-1"],
            }
        ],
        "candidate_facts": [
            {
                "fact_id": "fact-1",
                "subject": {"kind": "candidate_entity", "candidate_entity_id": "entity-1"},
                "predicate": field_key,
                "object": fact_object,
                "evidence_refs": ["e-1"],
            }
        ],
        "evidence": evidence if evidence is not None else [evidence_entry()],
        "review_signals": review_signals if review_signals is not None else [],
        PAGE_ACCOUNTING_FIELD: pages if pages is not None else page_entries(page_states),
    }


def interpret(
    response: dict[str, Any] | str,
    *,
    tmp_path: Path,
    request: DocumentInterpretationRequest | None = None,
    pages: int = 1,
) -> Any:
    """Run the public one-pass workflow and return its completed-workflow result.

    A deterministic validation failure must be reported through the returned
    outcome, so any raised exception is a behavioural failure of the contract
    rather than a test infrastructure problem.
    """

    source_path, media_type = write_source(tmp_path, pages=pages)
    request = request or interpretation_request(media_type=media_type)
    interpreter = OnePassDocumentInterpreter(
        gateway=RecordingGateway(response),
        model="generic-multimodal-model",
        source_limits=limits(max_pages=max(pages, 1)),
    )
    try:
        return interpreter.interpret(request, source_path)
    except Exception as error:  # noqa: BLE001 - deliberate behavioural assertion
        pytest.fail(
            "The one-pass workflow raised "
            f"{type(error).__name__}({error}) instead of returning one completed "
            "interpretation outcome carrying a deterministic validation result."
        )


def outcome_interpretation(result: Any) -> DocumentInterpretation:
    interpretation = getattr(result, "interpretation", None)
    assert isinstance(interpretation, DocumentInterpretation), (
        "The completed-workflow result does not expose the typed interpretation "
        f"(expected outcome.interpretation, got {type(result).__name__})."
    )
    return interpretation


def outcome_validation(result: Any) -> Any:
    validation = getattr(result, "validation", None)
    assert validation is not None, (
        "The completed-workflow result does not expose the deterministic validation "
        f"result (expected outcome.validation, got {type(result).__name__})."
    )
    return validation


def validation_status(result: Any) -> str:
    """Return the validation state as a stable upper-case name."""

    status = getattr(outcome_validation(result), "status", None)
    assert status is not None, "The deterministic validation result has no status."
    name = getattr(status, "name", None) or str(status)
    return str(name).upper()


def validation_issues(result: Any) -> tuple[Any, ...]:
    issues = getattr(outcome_validation(result), "issues", None)
    assert issues is not None, (
        "The deterministic validation result does not expose its findings "
        "(expected outcome.validation.issues)."
    )
    return tuple(issues)


def issues_text(result: Any) -> str:
    """Serialize validator findings for bounded, provenance-oriented assertions."""

    return json.dumps([_plain(issue) for issue in validation_issues(result)], default=str)


def page_states(interpretation: DocumentInterpretation) -> dict[int, str]:
    """Return the declared handling state per source page."""

    accounting = getattr(interpretation, PAGE_ACCOUNTING_FIELD, None)
    assert accounting is not None, (
        "The interpretation does not represent explicit page accounting "
        f"(expected DocumentInterpretation.{PAGE_ACCOUNTING_FIELD})."
    )
    states: dict[int, str] = {}
    for entry in accounting:
        number = getattr(entry, PAGE_NUMBER_KEY, None)
        state = getattr(entry, PAGE_STATE_KEY, None)
        assert number is not None and state is not None, (
            "Each page accounting entry must expose its page number and handling state."
        )
        states[int(number)] = str(getattr(state, "value", state))
    return states


def signal_identity(signals: Any) -> list[tuple[str, str, str, tuple[str, ...], tuple[str, ...]]]:
    """Element-for-element identity of model review signals."""

    return [
        (
            signal.code,
            signal.message,
            str(signal.severity),
            tuple(signal.evidence_refs),
            tuple(signal.fact_refs),
        )
        for signal in signals
    ]


def _plain(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return value
