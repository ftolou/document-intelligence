from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from correction.normalization import normalize_source_evidence
from correction.strategies.item_sum import validate_item_sum_evidence
from correction.strategies.vat import validate_vat_evidence


SOURCE = """BEGIN_RECEIPT
R0001 :: APPLE 1,00
R0002 :: Artikel 12345
R0003 :: MwSt % Netto Steuer Brutto
R0004 :: 19% 10,00 1,90 11,90
END_RECEIPT"""


class EvidenceNormalizationTests(unittest.TestCase):
    def test_item_singleton_unresolved_group_and_known_extra_fields_are_normalized(self) -> None:
        answer = {
            "item_blocks": [
                {
                    "source_rows": ["R0001"],
                    "name": "APPLE",
                    "line_amount": "1,00",
                    "unit_price": None,
                }
            ],
            "unresolved_candidate_rows": {
                "source_rows": ["R0002"],
                "name": "Artikel 12345",
                "line_amount": None,
                "unit_price": None,
            },
        }
        normalized, report = normalize_source_evidence(
            "item_sum_source_blocks_v3", answer, SOURCE
        )
        self.assertEqual("normalized", report["status"])
        self.assertEqual(
            [{"source_rows": ["R0002"]}],
            normalized["unresolved_candidate_rows"],
        )
        self.assertEqual(
            "valid", validate_item_sum_evidence(normalized, SOURCE)["status"]
        )
        self.assertEqual(answer["unresolved_candidate_rows"]["name"], "Artikel 12345")

    def test_vat_full_source_row_reference_is_reduced_to_existing_row_id(self) -> None:
        answer = {
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
        }
        normalized, report = normalize_source_evidence(
            "vat_source_evidence_v9", answer, SOURCE
        )
        self.assertEqual("normalized", report["status"])
        self.assertEqual(
            "R0004", normalized["vat_evidence_blocks"][0]["source_row"]
        )
        self.assertEqual("valid", validate_vat_evidence(normalized, SOURCE)["status"])

    def test_unknown_row_reference_is_not_rewritten(self) -> None:
        answer = {
            "status": "resolved",
            "label_row": "R9999 :: SUMME",
            "source_row": "R9999 :: SUMME 1,00",
            "label_text": "SUMME",
            "value_text": "1,00",
        }
        normalized, report = normalize_source_evidence(
            "final_total_source_evidence_v2_4", answer, SOURCE
        )
        self.assertEqual("unchanged", report["status"])
        self.assertEqual(answer, normalized)


if __name__ == "__main__":
    unittest.main()
