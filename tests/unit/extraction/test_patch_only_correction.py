"""Regression tests for compact patch-only correction behavior."""

from __future__ import annotations

import unittest

import receipt_intelligence.extraction.repair.patch_correction as patch_module
from receipt_intelligence.extraction.repair.patch_correction import (
    _compact_previous_receipt,
    _normalize_patch_obj,
    apply_correction_patches,
    build_patch_correction_prompt,
    run_patch_correction_pass,
)


class PatchOnlyCorrectionTests(unittest.TestCase):
    def test_prompt_compaction_patch_application_and_retry(self) -> None:
        receipt = {
            "currency": "EUR",
            "merchant": {"name": "REWE", "source_line_ids": ["line_1"]},
            "totals": {"grand_total": 10.0, "tax_total": 9.6},
            "items": [
                {
                    "description": "MILCH 3,8%",
                    "line_total": 2.58,
                    "source_line_ids": ["region_line_4"],
                    "category_group": "Food",
                }
            ],
            "pipeline": {"large": "metadata"},
        }
        report = {
            "import_decision": "needs_review",
            "balanced": True,
            "calculated_item_total": 10.0,
            "stated_total": 10.0,
            "difference": 0.0,
            "issues": [
                {
                    "code": "TAX_TOTAL_CONFLICTS_WITH_TAX_TABLE_EVIDENCE",
                    "severity": "medium",
                }
            ],
        }

        prompt = build_patch_correction_prompt(receipt, report, {})
        compact = _compact_previous_receipt(receipt)
        self.assertNotIn('"source_line_ids":', prompt)
        self.assertNotIn("category_group", prompt)
        self.assertNotIn("pipeline", compact)

        patch = _normalize_patch_obj(
            {
                "status": "ok",
                "patches": [
                    {
                        "op": "replace_field",
                        "path": "/totals/tax_total",
                        "value": None,
                        "reason": "Ambiguous tax table",
                    }
                ],
                "confidence": 0.9,
            }
        )
        corrected, actions = apply_correction_patches(receipt, patch)
        self.assertIsNone(corrected["totals"]["tax_total"])
        self.assertEqual(len(actions), 1)

        outputs = iter(
            [
                '{"schema_version":"v14_18_correction_patch_1","status":"ok","patches":[',
                '{"schema_version":"v14_18_correction_patch_1","status":"no_patch","patches":[],"warnings":[],"confidence":0.8}',
            ]
        )
        original_generate = patch_module.ollama_generate
        try:
            patch_module.ollama_generate = lambda **_: next(outputs)
            retried = run_patch_correction_pass(
                previous_receipt=receipt,
                validation_report=report,
                visual_evidence={},
                ollama_url="http://unused",
                model="unused",
            )
        finally:
            patch_module.ollama_generate = original_generate

        self.assertEqual(retried["status"], "no_patch")
        self.assertTrue(retried["retry_used"])
        self.assertEqual(retried["attempt_count"], 2)
