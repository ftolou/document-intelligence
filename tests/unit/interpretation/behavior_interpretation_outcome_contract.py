"""Behaviour contract for the deterministic interpretation validation boundary.

Smallest representative set proving the required externally observable behaviour:
one first-class workflow outcome, stable deterministic validation states, lossless
provenance separation, and bounded work on model-controlled references.
"""

from __future__ import annotations

import builtins
from pathlib import Path

import interpretation_outcome_support as support
import pytest

from receipt_intelligence.interpretation import ReviewSignal

pytestmark = pytest.mark.behavior_contract


def test_complete_page_coverage_produces_one_valid_outcome(tmp_path: Path) -> None:
    response = support.generated_response(
        page_states=(support.INTERPRETED, support.INTERPRETED, support.INTERPRETED)
    )

    result = support.interpret(response, tmp_path=tmp_path, pages=3)

    interpretation = support.outcome_interpretation(result)
    assert interpretation.source.source_id == support.SOURCE_ID
    assert support.validation_status(result) == support.VALID
    assert support.validation_issues(result) == ()
    assert getattr(interpretation, "validation", None) is None
    assert getattr(interpretation, "interpretation", None) is None


def test_missing_page_coverage_is_review_required_without_model_signals(tmp_path: Path) -> None:
    response = support.generated_response(
        pages=support.page_entries((support.INTERPRETED, support.INTERPRETED))
    )

    result = support.interpret(response, tmp_path=tmp_path, pages=3)

    interpretation = support.outcome_interpretation(result)
    assert support.validation_status(result) == support.REVIEW_REQUIRED
    assert support.validation_issues(result) != ()
    assert interpretation.review_signals == ()
    assert interpretation.requires_review is False


@pytest.mark.parametrize(
    "pages",
    [
        pytest.param(
            [support.page_entry(1), support.page_entry(1)],
            id="duplicate-page-coverage",
        ),
        pytest.param(
            [support.page_entry(1), support.page_entry(2)],
            id="nonexistent-page-coverage",
        ),
    ],
)
def test_page_coverage_defects_are_deterministically_invalid(
    tmp_path: Path,
    pages: list[dict[str, object]],
) -> None:
    response = support.generated_response(pages=pages)

    result = support.interpret(response, tmp_path=tmp_path, pages=1)

    assert support.validation_status(result) == support.INVALID
    assert support.validation_issues(result) != ()


def test_model_signals_and_validator_findings_remain_separate_and_lossless(
    tmp_path: Path,
) -> None:
    signals = [
        support.review_signal(code="ambiguous_value", message="The observed amount is ambiguous."),
        support.review_signal(
            code="manual_check",
            message="A human should confirm the stated amount.",
            severity="review_required",
        ),
    ]
    response = support.generated_response(
        pages=[support.page_entry(1), support.page_entry(2)],
        review_signals=signals,
    )

    result = support.interpret(response, tmp_path=tmp_path, pages=1)

    interpretation = support.outcome_interpretation(result)
    issues = support.validation_issues(result)
    assert support.validation_status(result) == support.INVALID
    assert issues != ()
    assert support.signal_identity(interpretation.review_signals) == [
        ("ambiguous_value", "The observed amount is ambiguous.", "warning", ("e-1",), ("fact-1",)),
        (
            "manual_check",
            "A human should confirm the stated amount.",
            "review_required",
            ("e-1",),
            ("fact-1",),
        ),
    ]
    assert not any(isinstance(issue, ReviewSignal) for issue in issues)


def test_oversized_page_range_is_rejected_without_materializing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = support.generated_response(
        evidence=[support.evidence_entry(page_number=None, page_range=(1, 1_000_000_000))]
    )
    real_range = builtins.range
    bound = 10_000

    def guarded_range(*args: int) -> object:
        span = args[1] - args[0] if len(args) >= 2 else (args[0] if args else 0)
        if isinstance(span, int) and span > bound:
            raise support.RangeMaterialized(
                f"Validation enumerated an untrusted range of {span} pages; "
                "model-controlled ranges must be compared against source bounds first."
            )
        return real_range(*args)

    monkeypatch.setattr(builtins, "range", guarded_range)

    result = support.interpret(response, tmp_path=tmp_path, pages=1)

    assert support.validation_status(result) == support.INVALID
    assert len(support.issues_text(result)) <= 8_000


def test_validation_never_repairs_a_malformed_observed_value(tmp_path: Path) -> None:
    response = support.generated_response(
        literal={
            "kind": "literal",
            "literal_type": "date",
            "observed": "27.05.20024",
            "normalization_status": "failed",
        },
        field_key="stated_date",
    )
    request = support.interpretation_request(field_key="stated_date")

    result = support.interpret(response, tmp_path=tmp_path, request=request, pages=1)

    fact_object = support.outcome_interpretation(result).candidate_facts[0].object
    assert fact_object.observed == "27.05.20024"
    assert fact_object.normalized is None
    assert support.validation_status(result) in {support.VALID, support.REVIEW_REQUIRED}
