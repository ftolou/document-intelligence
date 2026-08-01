from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from correction.profile import load_correction_profile
from prompt_registry import PromptRegistry


class CorrectionProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = load_correction_profile(
            ROOT / "correction" / "config" / "production.json"
        )
        self.registry = PromptRegistry(ROOT / "prompts")

    def test_item_sum_route_contains_only_existing_specialists(self) -> None:
        self.assertEqual(
            (
                "item_sum_source_blocks_v3",
                "final_total_source_evidence_v2_4",
            ),
            tuple(
                strategy.strategy_id
                for strategy in self.profile.strategy_chain(
                    "ITEM_SUM_RECONCILIATION"
                )
            ),
        )

    def test_unknown_validation_code_has_no_fallback_strategy(self) -> None:
        self.assertEqual((), self.profile.strategy_chain("ITEM_CONTRACT"))

    def test_vat_and_final_total_are_enabled(self) -> None:
        vat_chain = {
            strategy.strategy_id
            for strategy in self.profile.strategy_chain(
                "VAT_LINES_GROSS_RECONCILIATION"
            )
        }
        self.assertIn("vat_source_evidence_v9", vat_chain)
        self.assertIn("final_total_source_evidence_v2_4", vat_chain)

    def test_only_source_evidence_strategies_are_configured(self) -> None:
        self.assertTrue(self.profile.strategies)
        self.assertEqual(
            {"source_evidence"},
            {strategy.kind for strategy in self.profile.strategies.values()},
        )
        self.assertNotIn("generic_targeted_patch", self.profile.strategies)

    def test_generic_strategy_kind_is_rejected_by_profile_loader(self) -> None:
        payload = {
            "profile_version": "test",
            "automatic_patching": True,
            "retain_accepted_partial_corrections": True,
            "max_rounds": 1,
            "routes": {"X": ["generic"]},
            "strategies": {
                "generic": {
                    "kind": "generic_patch",
                    "prompt_id": "gemma.correction.generic",
                    "prompt_version": "1.0.0",
                    "max_attempts": 1,
                    "max_patches": 1,
                }
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "profile.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Only source_evidence"):
                load_correction_profile(path)

    def test_source_evidence_prompts_render_with_profile_variables(self) -> None:
        unresolved_tokens = (
            "$source_evidence",
            "$output_schema",
            "${SOURCE_EVIDENCE}",
            "${OUTPUT_SCHEMA}",
            "{{SOURCE_EVIDENCE}}",
            "{{OUTPUT_SCHEMA}}",
        )
        for strategy in self.profile.strategies.values():
            rendered = self.registry.render(
                strategy.prompt_id,
                strategy.prompt_version,
                source_evidence=(
                    "BEGIN_RECEIPT\nR0001 :: TEST 1,00\nEND_RECEIPT"
                ),
                output_schema='{"type":"object"}',
            )
            self.assertFalse(
                any(token in rendered for token in unresolved_tokens),
                strategy.strategy_id,
            )

    def test_every_strategy_prompt_binding_exists(self) -> None:
        for strategy in self.profile.strategies.values():
            self.registry.load(strategy.prompt_id, strategy.prompt_version)
            self.registry.load_schema(
                strategy.prompt_id, strategy.prompt_version
            )


if __name__ == "__main__":
    unittest.main()
