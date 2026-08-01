from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from correction.coordinator import CorrectionCallbacks, run_correction_coordinator
from correction.profile import CorrectionProfile, StrategyConfig


SOURCE = """BEGIN_RECEIPT
R0001 :: APPLE 1,00
R0002 :: SUMME 2,00
R0003 :: MwSt % Netto Steuer Brutto
R0004 :: 19% 10,00 1,90 11,90
END_RECEIPT"""


def _validation(
    receipt: dict[str, Any],
    *,
    include_unrouted_failure: bool = False,
) -> dict[str, Any]:
    item_sum = sum(float(item["final_price"]) for item in receipt["items"])
    total = float(receipt["totals"]["final_purchase_total"]["final_purchase_total"])
    item_ok = abs(item_sum - total) < 0.001
    rate = float(receipt["tax"]["vat_lines"][0]["rate_percent"])
    vat_ok = abs(rate - 19.0) < 0.001
    checks: list[dict[str, Any]] = [
        {
            "code": "BASE_CONTRACT",
            "status": "passed",
            "severity": "info",
            "message": "base remains valid",
        },
        {
            "code": "ITEM_SUM_RECONCILIATION",
            "status": "passed" if item_ok else "failed",
            "severity": "error",
            "message": "item sum",
            "values": {"item_sum": item_sum, "final_purchase_total": total},
        },
        {
            "code": "VAT_LINE_RATE_ARITHMETIC",
            "status": "passed" if vat_ok else "failed",
            "severity": "review",
            "message": "vat rate",
        },
    ]
    if include_unrouted_failure:
        checks.insert(
            1,
            {
                "code": "ITEM_CONTRACT",
                "status": "failed",
                "severity": "error",
                "message": "no specialist registered",
            },
        )
    failed = [check for check in checks if check["status"] == "failed"]
    return {
        "status": "review_required" if failed else "valid",
        "policy": {"changes_model_values": False, "correction_applied": False},
        "summary": {
            "passed": len(checks) - len(failed),
            "failed": len(failed),
            "skipped": 0,
            "observed": 0,
            "error_count": sum(
                1 for check in failed if check.get("severity") == "error"
            ),
            "review_count": sum(
                1 for check in failed if check.get("severity") == "review"
            ),
        },
        "checks": checks,
    }


def _profile(routes: dict[str, tuple[str, ...]]) -> CorrectionProfile:
    strategies = {
        "item_sum_source_blocks_v3": StrategyConfig(
            strategy_id="item_sum_source_blocks_v3",
            kind="source_evidence",
            prompt_id="item.prompt",
            prompt_version="1",
            max_attempts=1,
            max_patches=8,
        ),
        "vat_source_evidence_v9": StrategyConfig(
            strategy_id="vat_source_evidence_v9",
            kind="source_evidence",
            prompt_id="vat.prompt",
            prompt_version="1",
            max_attempts=1,
            max_patches=4,
        ),
    }
    return CorrectionProfile(
        profile_version="test",
        automatic_patching=True,
        retain_accepted_partial_corrections=True,
        max_rounds=6,
        routes=routes,
        strategies=strategies,
        source_path=Path("test-profile.json"),
    )


def _receipt() -> dict[str, Any]:
    return {
        "items": [
            {
                "name": "APPLE",
                "final_price": 1.0,
                "quantity": None,
                "unit": None,
                "discount_amount": None,
                "original_price": None,
            }
        ],
        "totals": {
            "final_purchase_total": {
                "final_purchase_total": 2.0,
                "currency": "EUR",
            }
        },
        "receipt_metadata": {"currency": "EUR"},
        "tax": {
            "vat_amount": {"vat_amount": 1.9, "currency": "EUR"},
            "vat_lines": [
                {
                    "source_rows": ["R0004"],
                    "rate_percent": 7.0,
                    "net_amount": 10.0,
                    "vat_amount": 1.9,
                }
            ],
        },
    }


class CorrectionCoordinatorTests(unittest.TestCase):
    def test_exhausted_target_does_not_block_later_specialist(self) -> None:
        receipt = _receipt()
        artifacts: dict[str, Any] = {}

        def invoke_source(strategy: StrategyConfig, *args: Any) -> dict[str, Any]:
            if strategy.strategy_id == "item_sum_source_blocks_v3":
                # Valid evidence that produces no mutation, so this target is exhausted.
                return {
                    "answer": {
                        "item_blocks": [
                            {
                                "source_rows": ["R0001"],
                                "name": "APPLE",
                                "line_amount": "1,00",
                                "unit_price": None,
                            }
                        ],
                        "unresolved_candidate_rows": [],
                    },
                    "request": {"prompt": {"id": "item.prompt"}},
                    "metrics": {},
                }
            self.assertEqual("vat_source_evidence_v9", strategy.strategy_id)
            return {
                "answer": {
                    "vat_evidence_blocks": [
                        {
                            "context_rows": ["R0003"],
                            "source_row": "R0004 :: 19% 10,00 1,90 11,90",
                            "row_label": None,
                            "fields": [
                                {"role": "rate_percent", "value": "19%"},
                                {"role": "net_amount", "value": "10,00"},
                                {"role": "vat_amount", "value": "1,90"},
                                {"role": "gross_amount", "value": "11,90"},
                            ],
                        }
                    ],
                    "unresolved_candidate_rows": [],
                },
                "request": {"prompt": {"id": "vat.prompt"}},
                "metrics": {},
            }

        callbacks = CorrectionCallbacks(
            invoke_source_evidence=invoke_source,
            validate_receipt=lambda candidate, _: _validation(candidate),
            effective_item_pipeline=lambda previous, candidate: previous,
            write_artifact=lambda name, value: artifacts.__setitem__(
                name, copy.deepcopy(value)
            ),
        )
        corrected, validation, _, report = run_correction_coordinator(
            profile=_profile(
                {
                    "ITEM_SUM_RECONCILIATION": ("item_sum_source_blocks_v3",),
                    "VAT_LINE_RATE_ARITHMETIC": ("vat_source_evidence_v9",),
                }
            ),
            callbacks=callbacks,
            transcription=SOURCE,
            receipt=receipt,
            initial_validation=_validation(receipt),
            item_pipeline_result=None,
            enabled=True,
        )

        self.assertEqual(1.0, corrected["items"][0]["final_price"])
        self.assertEqual(19.0, corrected["tax"]["vat_lines"][0]["rate_percent"])
        self.assertEqual("review_required", validation["status"])
        self.assertEqual(
            ["ITEM_SUM_RECONCILIATION"], report["exhausted_target_codes"]
        )
        self.assertEqual(
            "accepted_partial_open_failures", report["status"]
        )
        self.assertEqual(1, report["normalization_summary"]["normalized_attempt_count"])
        self.assertEqual(
            "normalized",
            report["accepted_corrections"][0]["normalization"]["status"],
        )

    def test_unrouted_failure_remains_open_and_does_not_block_other_target(self) -> None:
        receipt = _receipt()
        # Remove the item-sum failure so ITEM_CONTRACT is the only error-severity target.
        receipt["totals"]["final_purchase_total"]["final_purchase_total"] = 1.0

        def invoke_source(strategy: StrategyConfig, *args: Any) -> dict[str, Any]:
            self.assertEqual("vat_source_evidence_v9", strategy.strategy_id)
            return {
                "answer": {
                    "vat_evidence_blocks": [
                        {
                            "context_rows": ["R0003"],
                            "source_row": "R0004",
                            "row_label": None,
                            "fields": [
                                {"role": "rate_percent", "value": "19%"},
                                {"role": "net_amount", "value": "10,00"},
                                {"role": "vat_amount", "value": "1,90"},
                                {"role": "gross_amount", "value": "11,90"},
                            ],
                        }
                    ],
                    "unresolved_candidate_rows": [],
                },
                "request": {"prompt": {"id": "vat.prompt"}},
                "metrics": {},
            }

        callbacks = CorrectionCallbacks(
            invoke_source_evidence=invoke_source,
            validate_receipt=lambda candidate, _: _validation(
                candidate, include_unrouted_failure=True
            ),
            effective_item_pipeline=lambda previous, candidate: previous,
            write_artifact=lambda name, value: None,
        )
        corrected, validation, _, report = run_correction_coordinator(
            profile=_profile(
                {"VAT_LINE_RATE_ARITHMETIC": ("vat_source_evidence_v9",)}
            ),
            callbacks=callbacks,
            transcription=SOURCE,
            receipt=receipt,
            initial_validation=_validation(
                receipt, include_unrouted_failure=True
            ),
            item_pipeline_result=None,
            enabled=True,
        )

        self.assertEqual(19.0, corrected["tax"]["vat_lines"][0]["rate_percent"])
        self.assertEqual("review_required", validation["status"])
        self.assertEqual(["ITEM_CONTRACT"], report["open_no_strategy_codes"])
        self.assertEqual(
            ["ITEM_CONTRACT"], report["remaining_failed_codes"]
        )
        self.assertEqual(
            ["open_no_strategy", "accepted"],
            [entry["status"] for entry in report["target_outcomes"]],
        )

    def test_clean_validation_skips_correction(self) -> None:
        receipt = _receipt()
        receipt["totals"]["final_purchase_total"]["final_purchase_total"] = 1.0
        receipt["tax"]["vat_lines"][0]["rate_percent"] = 19.0
        callbacks = CorrectionCallbacks(
            invoke_source_evidence=lambda *args: self.fail("must not invoke model"),
            validate_receipt=lambda candidate, _: _validation(candidate),
            effective_item_pipeline=lambda previous, candidate: previous,
            write_artifact=lambda name, value: None,
        )
        corrected, validation, _, report = run_correction_coordinator(
            profile=_profile({}),
            callbacks=callbacks,
            transcription=SOURCE,
            receipt=receipt,
            initial_validation=_validation(receipt),
            item_pipeline_result=None,
            enabled=True,
        )
        self.assertEqual(receipt, corrected)
        self.assertEqual("valid", validation["status"])
        self.assertEqual("skipped_validation_clean", report["status"])


if __name__ == "__main__":
    unittest.main()
