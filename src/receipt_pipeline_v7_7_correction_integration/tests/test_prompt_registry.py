from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prompt_registry import PromptRegistry


class PromptRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = PromptRegistry(ROOT / "prompts")
        cls.manifest = json.loads((ROOT / "prompts" / "manifest.json").read_text(encoding="utf-8"))

    def test_all_manifest_entries_load_and_hash_verify(self) -> None:
        entries = self.manifest["prompts"]
        self.assertGreaterEqual(len(entries), 26)
        for entry in entries:
            text = self.registry.load(entry["id"], entry["version"])
            self.assertTrue(text.strip(), entry["id"])
            if entry.get("schema_path"):
                schema = self.registry.load_schema(entry["id"], entry["version"])
                self.assertIsInstance(schema, dict)
                self.assertEqual("object", schema.get("type"))

    def test_correction_evidence_prompts_have_registered_schemas(self) -> None:
        for prompt_id in (
            "gemma.correction.item_sum_source_blocks",
            "gemma.correction.vat_source_evidence",
            "gemma.correction.final_total_source_evidence",
        ):
            record = self.registry.record(prompt_id, "1.0.0")
            self.assertIsNotNone(record.schema_path)
            self.assertIsNotNone(record.schema_sha256)
            self.assertTrue(self.registry.load_schema(prompt_id, "1.0.0"))

    def test_task_envelope_renders(self) -> None:
        rendered = self.registry.render(
            "gemma.template.task_envelope",
            "1.0.0",
            question="QUESTION",
            schema_json='{"type":"object"}',
            evidence="EVIDENCE",
        )
        self.assertIn("QUESTION", rendered)
        self.assertIn("EVIDENCE", rendered)
        self.assertIn("Required JSON schema", rendered)

    def test_every_gemma_task_call_pins_a_prompt_id(self) -> None:
        script_path = ROOT / (
            "experiment_batch_paddle_snapped_crops_qwen35_gemma_items_scalars_"
            "v7_7_correction_coordinator.py"
        )
        tree = ast.parse(script_path.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "invoke_gemma_task"
        ]
        self.assertGreaterEqual(len(calls), 2)
        for call in calls:
            keyword_names = {keyword.arg for keyword in call.keywords}
            self.assertIn("prompt_id", keyword_names)

    def test_json_repair_prompt_renders_without_strategy_schema_changes(self) -> None:
        rendered = self.registry.render(
            "gemma.correction.json_repair",
            "1.0.0",
            invalid_json='{ "broken": true',
        )
        self.assertIn('{ "broken": true', rendered)
        self.assertIn("serialization structure only", rendered)
        record = self.registry.record("gemma.correction.json_repair", "1.0.0")
        self.assertIsNone(record.schema_path)

    def test_unknown_version_fails(self) -> None:
        with self.assertRaises(KeyError):
            self.registry.load("gemma.items.direct", "9.9.9")


if __name__ == "__main__":
    unittest.main()
