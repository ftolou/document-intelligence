from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from correction.strategies.final_total import (
    build_final_total_patch,
    validate_final_total_evidence,
)
from correction.strategies.item_sum import (
    build_item_sum_patch,
    validate_item_sum_evidence,
)
from correction.strategies.vat import build_vat_patch, validate_vat_evidence

SOURCE = """BEGIN_RECEIPT
R0001 :: APPLE 1,20
R0002 :: BREAD
R0003 :: 2,30
R0004 :: MwSt % Netto Steuer Brutto
R0005 :: 19% 10,00 1,90 11,90
R0006 :: MwSt gesamt 1,90
R0007 :: SUMME 11,90
END_RECEIPT"""


class CorrectionStrategyTests(unittest.TestCase):
    def test_item_sum_evidence_builds_bounded_patch(self) -> None:
        answer = {
            "item_blocks": [
                {
                    "source_rows": ["R0001"],
                    "name": "APPLE",
                    "line_amount": "1,20",
                    "unit_price": None,
                },
                {
                    "source_rows": ["R0002", "R0003"],
                    "name": "BREAD",
                    "line_amount": "2,30",
                    "unit_price": None,
                },
            ],
            "unresolved_candidate_rows": [],
        }
        validation = validate_item_sum_evidence(answer, SOURCE)
        self.assertEqual("valid", validation["status"])
        receipt = {
            "items": [
                {
                    "name": "APPLE",
                    "final_price": 1.0,
                    "quantity": None,
                    "unit": None,
                    "discount_amount": None,
                    "original_price": None,
                }
            ]
        }
        patch, diagnostics = build_item_sum_patch(answer, receipt)
        self.assertEqual("patch_built", diagnostics["status"])
        self.assertEqual(2, len(patch["patches"]))
        self.assertEqual("/items/0/final_price", patch["patches"][0]["path"])
        self.assertEqual("/items", patch["patches"][1]["path"])

    def test_item_evidence_rejects_nonliteral_value(self) -> None:
        answer = {
            "item_blocks": [
                {
                    "source_rows": ["R0001"],
                    "name": "ORANGE",
                    "line_amount": "1,20",
                    "unit_price": None,
                }
            ],
            "unresolved_candidate_rows": [],
        }
        validation = validate_item_sum_evidence(answer, SOURCE)
        self.assertEqual("invalid", validation["status"])
        self.assertTrue(
            any(
                error["code"] == "VALUE_NOT_LITERAL_IN_SOURCE_ROWS"
                for error in validation["errors"]
            )
        )

    def test_vat_evidence_rebuilds_vat_fields(self) -> None:
        answer = {
            "vat_evidence_blocks": [
                {
                    "context_rows": ["R0004"],
                    "source_row": "R0005",
                    "row_label": None,
                    "fields": [
                        {"role": "rate_percent", "value": "19%"},
                        {"role": "net_amount", "value": "10,00"},
                        {"role": "vat_amount", "value": "1,90"},
                        {"role": "gross_amount", "value": "11,90"},
                    ],
                },
                {
                    "context_rows": [],
                    "source_row": "R0006",
                    "row_label": "MwSt gesamt",
                    "fields": [{"role": "vat_amount", "value": "1,90"}],
                },
            ],
            "unresolved_candidate_rows": [],
        }
        validation = validate_vat_evidence(answer, SOURCE)
        self.assertEqual("valid", validation["status"])
        receipt = {
            "receipt_metadata": {"currency": "EUR"},
            "tax": {
                "vat_amount": {"vat_amount": 2.0, "currency": "EUR"},
                "vat_lines": [],
            },
        }
        patch, diagnostics = build_vat_patch(answer, receipt)
        self.assertEqual("patch_built", diagnostics["status"])
        paths = {entry["path"] for entry in patch["patches"]}
        self.assertEqual({"/tax/vat_lines", "/tax/vat_amount/vat_amount"}, paths)

    def test_final_total_evidence_builds_one_patch(self) -> None:
        answer = {
            "status": "resolved",
            "label_row": "R0007",
            "source_row": "R0007",
            "label_text": "SUMME",
            "value_text": "11,90",
        }
        validation = validate_final_total_evidence(answer, SOURCE)
        self.assertEqual("valid", validation["status"])
        receipt = {
            "receipt_metadata": {"currency": "EUR"},
            "totals": {
                "final_purchase_total": {
                    "final_purchase_total": 12.9,
                    "currency": "EUR",
                }
            },
        }
        patch, diagnostics = build_final_total_patch(answer, receipt)
        self.assertEqual("patch_built", diagnostics["status"])
        self.assertEqual(1, len(patch["patches"]))
        self.assertEqual(
            "/totals/final_purchase_total/final_purchase_total",
            patch["patches"][0]["path"],
        )
        self.assertEqual(11.9, patch["patches"][0]["value"])


if __name__ == "__main__":
    unittest.main()
