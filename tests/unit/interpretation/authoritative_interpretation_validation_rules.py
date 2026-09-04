"""Full deterministic validation rules for the interpretation trust boundary.

These cases complete the issue's required coverage around page accounting,
reference/graph integrity, provenance preservation and bounded diagnostics. They
are authoritative validation rather than the initial behaviour-contract gate.
"""

from __future__ import annotations

from pathlib import Path

import interpretation_outcome_support as support
import pytest
from receipt_intelligence.interpretation.contracts import MAX_COLLECTION_SIZE


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (support.BLANK, support.VALID),
        (support.IRRELEVANT, support.VALID),
        (support.UNREADABLE, support.REVIEW_REQUIRED),
        (support.UNPROCESSED, support.REVIEW_REQUIRED),
    ],
)
def test_explicit_page_states_decide_acceptance(
    tmp_path: Path,
    state: str,
    expected: str,
) -> None:
    response = support.generated_response(page_states=(support.INTERPRETED, state))

    result = support.interpret(response, tmp_path=tmp_path, pages=2)

    assert support.validation_status(result) == expected


def test_every_declared_page_state_remains_distinguishable(tmp_path: Path) -> None:
    response = support.generated_response(page_states=support.PAGE_STATES)

    result = support.interpret(response, tmp_path=tmp_path, pages=len(support.PAGE_STATES))

    states = support.page_states(support.outcome_interpretation(result))
    assert [states[number] for number in sorted(states)] == list(support.PAGE_STATES)
    assert support.validation_status(result) == support.REVIEW_REQUIRED


def test_evidence_page_outside_the_source_is_invalid_not_a_generation_failure(
    tmp_path: Path,
) -> None:
    response = support.generated_response(evidence=[support.evidence_entry(page_number=2)])

    result = support.interpret(response, tmp_path=tmp_path, pages=1)

    assert support.validation_status(result) == support.INVALID


def test_evidence_attached_to_a_non_interpreted_page_is_invalid(tmp_path: Path) -> None:
    response = support.generated_response(
        page_states=(support.INTERPRETED, support.BLANK),
        evidence=[support.evidence_entry(page_number=2)],
    )

    result = support.interpret(response, tmp_path=tmp_path, pages=2)

    assert support.validation_status(result) == support.INVALID


@pytest.mark.parametrize(
    ("page_range", "expected"),
    [
        pytest.param((1, 2), support.VALID, id="range-within-source"),
        pytest.param((1, 3), support.INVALID, id="range-one-past-source"),
        pytest.param((2, 1), support.INVALID, id="reversed-range"),
    ],
)
def test_page_ranges_must_be_ordered_and_source_bounded(
    tmp_path: Path,
    page_range: tuple[int, int],
    expected: str,
) -> None:
    response = support.generated_response(
        page_states=(support.INTERPRETED, support.INTERPRETED),
        evidence=[support.evidence_entry(page_number=None, page_range=page_range)],
    )

    result = support.interpret(response, tmp_path=tmp_path, pages=2)

    assert support.validation_status(result) == expected


def test_missing_required_evidence_is_invalid(tmp_path: Path) -> None:
    response = support.generated_response(classification_evidence_refs=[])

    result = support.interpret(response, tmp_path=tmp_path, pages=1)

    assert support.validation_status(result) == support.INVALID


def test_observed_excerpt_stays_model_observed_metadata(tmp_path: Path) -> None:
    response = support.generated_response(
        evidence=[support.evidence_entry(excerpt="Betrag 12,50 EUR")]
    )

    result = support.interpret(response, tmp_path=tmp_path, pages=1)

    interpretation = support.outcome_interpretation(result)
    assert interpretation.evidence[0].excerpt == "Betrag 12,50 EUR"
    assert support.validation_status(result) == support.VALID
    assert "verified" not in interpretation.model_dump_json()
    assert "verified" not in support.issues_text(result)


def test_model_review_required_signal_yields_review_state_without_a_duplicate_finding(
    tmp_path: Path,
) -> None:
    signal = support.review_signal(
        code="manual_check",
        message="A human should confirm the stated amount.",
        severity="review_required",
    )
    response = support.generated_response(review_signals=[signal])

    result = support.interpret(response, tmp_path=tmp_path, pages=1)

    interpretation = support.outcome_interpretation(result)
    assert support.validation_status(result) == support.REVIEW_REQUIRED
    assert support.signal_identity(interpretation.review_signals) == [
        (
            "manual_check",
            "A human should confirm the stated amount.",
            "review_required",
            ("e-1",),
            ("fact-1",),
        )
    ]
    assert "manual_check" not in support.issues_text(result)


def test_maximum_model_review_signal_collection_is_preserved(tmp_path: Path) -> None:
    signals = [
        support.review_signal(code=f"signal-{index}", message=f"Observation {index}.")
        for index in range(MAX_COLLECTION_SIZE)
    ]
    response = support.generated_response(review_signals=signals)

    result = support.interpret(response, tmp_path=tmp_path, pages=1)

    interpretation = support.outcome_interpretation(result)
    assert len(interpretation.review_signals) == MAX_COLLECTION_SIZE
    assert support.signal_identity(interpretation.review_signals) == [
        (f"signal-{index}", f"Observation {index}.", "warning", ("e-1",), ("fact-1",))
        for index in range(MAX_COLLECTION_SIZE)
    ]
    assert support.validation_status(result) != support.INVALID


def test_maximum_model_signals_with_missing_coverage_still_reviews_without_truncation(
    tmp_path: Path,
) -> None:
    signals = [
        support.review_signal(code=f"signal-{index}", message=f"Observation {index}.")
        for index in range(MAX_COLLECTION_SIZE)
    ]
    response = support.generated_response(
        pages=support.page_entries((support.INTERPRETED,)),
        review_signals=signals,
    )

    result = support.interpret(response, tmp_path=tmp_path, pages=2)

    interpretation = support.outcome_interpretation(result)
    assert support.validation_status(result) == support.REVIEW_REQUIRED
    assert len(interpretation.review_signals) == MAX_COLLECTION_SIZE


def test_diagnostics_stay_bounded_for_bulk_malformed_page_accounting(tmp_path: Path) -> None:
    response = support.generated_response(
        pages=[support.page_entry(1) for _ in range(MAX_COLLECTION_SIZE)]
    )

    result = support.interpret(response, tmp_path=tmp_path, pages=1)

    assert support.validation_status(result) == support.INVALID
    assert len(support.validation_issues(result)) <= MAX_COLLECTION_SIZE
    assert len(support.issues_text(result)) <= 32_000
