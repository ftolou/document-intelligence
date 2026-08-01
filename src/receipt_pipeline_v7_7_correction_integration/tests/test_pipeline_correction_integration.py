from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from correction.coordinator import CorrectionCallbacks, run_correction_coordinator
from correction.profile import load_correction_profile


class PipelineCorrectionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = ROOT / (
            "experiment_batch_paddle_snapped_crops_qwen35_gemma_items_"
            "scalars_v7_7_correction_coordinator.py"
        )
        spec = importlib.util.spec_from_file_location("receipt_pipeline_v77_test", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls.pipeline = module

    def test_item_specialist_integrates_with_production_validator(self) -> None:
        pipeline = self.pipeline
        receipt = {
            "merchant": {"name": None, "address": None},
            "receipt_metadata": {
                "date": None,
                "time": None,
                "receipt_number": None,
                "transaction_status": "not_clear",
                "currency": "EUR",
            },
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
                },
                "pre_discount_total": None,
                "net_amount": None,
            },
            "discount": {"discount_total": None},
            "payment": {
                "payment_method": None,
                "payment_received": None,
                "change_returned": None,
            },
            "tax": {"vat_amount": None, "vat_lines": []},
        }
        item_pipeline = {
            "status": "completed",
            "items": receipt["items"],
            "validation": pipeline.validate_direct_items({"items": receipt["items"]}),
        }
        initial = pipeline.validate_receipt_deterministically(
            receipt=receipt,
            item_pipeline_result=item_pipeline,
            item_pipeline_enabled=True,
            selected_scalar_tasks=[],
        )
        self.assertEqual(
            ["ITEM_SUM_RECONCILIATION"],
            [check["code"] for check in initial["checks"] if check["status"] == "failed"],
        )

        def invoke_source(*args: Any) -> dict[str, Any]:
            return {
                "answer": {
                    "item_blocks": [
                        {
                            "source_rows": ["R0001"],
                            "name": "APPLE",
                            "line_amount": "2,00",
                            "unit_price": None,
                        }
                    ],
                    "unresolved_candidate_rows": [],
                },
                "request": {"prompt": {"id": "gemma.correction.item_sum_source_blocks"}},
                "metrics": {},
            }

        callbacks = CorrectionCallbacks(
            invoke_source_evidence=invoke_source,
            validate_receipt=lambda candidate, candidate_items: (
                pipeline.validate_receipt_deterministically(
                    receipt=candidate,
                    item_pipeline_result=candidate_items,
                    item_pipeline_enabled=True,
                    selected_scalar_tasks=[],
                )
            ),
            effective_item_pipeline=lambda previous, candidate: (
                pipeline._effective_item_pipeline_result(previous, candidate, enabled=True)
            ),
            write_artifact=lambda name, value: None,
        )
        corrected, final_validation, corrected_items, report = run_correction_coordinator(
            profile=load_correction_profile(ROOT / "correction" / "config" / "production.json"),
            callbacks=callbacks,
            transcription=("BEGIN_RECEIPT\nR0001 :: APPLE 2,00\nR0002 :: SUMME 2,00\nEND_RECEIPT"),
            receipt=receipt,
            initial_validation=initial,
            item_pipeline_result=item_pipeline,
            enabled=True,
        )

        self.assertEqual(2.0, corrected["items"][0]["final_price"])
        self.assertEqual("valid", final_validation["status"])
        self.assertEqual(2.0, corrected_items["items"][0]["final_price"])
        self.assertEqual("accepted_all_targets_resolved", report["status"])
        self.assertEqual(
            "item_sum_source_blocks_v3",
            report["accepted_corrections"][0]["strategy_id"],
        )

    def test_missing_final_price_contract_and_completeness_share_one_item_route(
        self,
    ) -> None:
        pipeline = self.pipeline
        receipt = {
            "merchant": {"name": None, "address": None},
            "receipt_metadata": {
                "date": None,
                "time": None,
                "receipt_number": None,
                "transaction_status": "not_clear",
                "currency": "EUR",
            },
            "items": [
                {
                    "name": "APPLE",
                    "final_price": 1.0,
                    "quantity": None,
                    "unit": None,
                    "discount_amount": None,
                    "original_price": None,
                },
                {
                    "name": "BANANA",
                    "final_price": None,
                    "quantity": None,
                    "unit": None,
                    "discount_amount": None,
                    "original_price": None,
                },
            ],
            "totals": {
                "final_purchase_total": {
                    "final_purchase_total": 3.0,
                    "currency": "EUR",
                },
                "pre_discount_total": None,
                "net_amount": None,
            },
            "discount": {"discount_total": None},
            "payment": {
                "payment_method": None,
                "payment_received": None,
                "change_returned": None,
            },
            "tax": {"vat_amount": None, "vat_lines": []},
        }
        item_pipeline = {
            "status": "completed",
            "items": receipt["items"],
            "validation": pipeline.validate_direct_items({"items": receipt["items"]}),
        }
        initial = pipeline.validate_receipt_deterministically(
            receipt=receipt,
            item_pipeline_result=item_pipeline,
            item_pipeline_enabled=True,
            selected_scalar_tasks=[],
        )
        self.assertEqual(
            ["ITEM_CONTRACT", "ITEM_PRICES_COMPLETE"],
            [check["code"] for check in initial["checks"] if check["status"] == "failed"],
        )

        calls: list[str] = []

        def invoke_source(strategy: Any, *args: Any) -> dict[str, Any]:
            calls.append(strategy.strategy_id)
            self.assertEqual("item_sum_source_blocks_v3", strategy.strategy_id)
            return {
                "answer": {
                    "item_blocks": [
                        {
                            "source_rows": ["R0001"],
                            "name": "APPLE",
                            "line_amount": "1,00",
                            "unit_price": None,
                        },
                        {
                            "source_rows": ["R0002"],
                            "name": "BANANA",
                            "line_amount": "2,00",
                            "unit_price": None,
                        },
                    ],
                    "unresolved_candidate_rows": [],
                },
                "request": {"prompt": {"id": "gemma.correction.item_sum_source_blocks"}},
                "metrics": {},
            }

        callbacks = CorrectionCallbacks(
            invoke_source_evidence=invoke_source,
            validate_receipt=lambda candidate, candidate_items: (
                pipeline.validate_receipt_deterministically(
                    receipt=candidate,
                    item_pipeline_result=candidate_items,
                    item_pipeline_enabled=True,
                    selected_scalar_tasks=[],
                )
            ),
            effective_item_pipeline=lambda previous, candidate: (
                pipeline._effective_item_pipeline_result(previous, candidate, enabled=True)
            ),
            write_artifact=lambda name, value: None,
        )
        corrected, final_validation, corrected_items, report = run_correction_coordinator(
            profile=load_correction_profile(ROOT / "correction" / "config" / "production.json"),
            callbacks=callbacks,
            transcription=(
                "BEGIN_RECEIPT\n"
                "R0001 :: APPLE 1,00\n"
                "R0002 :: BANANA 2,00\n"
                "R0003 :: SUMME 3,00\n"
                "END_RECEIPT"
            ),
            receipt=receipt,
            initial_validation=initial,
            item_pipeline_result=item_pipeline,
            enabled=True,
        )

        self.assertEqual(["item_sum_source_blocks_v3"], calls)
        self.assertEqual(2.0, corrected["items"][1]["final_price"])
        self.assertEqual(2.0, corrected_items["items"][1]["final_price"])
        self.assertEqual("valid", final_validation["status"])
        self.assertEqual(
            ["ITEM_CONTRACT", "ITEM_PRICES_COMPLETE"],
            report["accepted_corrections"][0]["target_codes"],
        )
        self.assertEqual(
            ["ITEM_CONTRACT", "ITEM_PRICES_COMPLETE"],
            report["corrected_target_codes"],
        )

    def test_vat_specialist_integrates_with_production_validator(self) -> None:
        pipeline = self.pipeline
        receipt = {
            "merchant": {"name": None, "address": None},
            "receipt_metadata": {
                "date": None,
                "time": None,
                "receipt_number": None,
                "transaction_status": "not_clear",
                "currency": "EUR",
            },
            "items": [],
            "totals": {
                "final_purchase_total": {
                    "final_purchase_total": 11.9,
                    "currency": "EUR",
                },
                "pre_discount_total": None,
                "net_amount": {"net_amount": 10.0, "currency": "EUR"},
            },
            "discount": {"discount_total": None},
            "payment": {
                "payment_method": None,
                "payment_received": None,
                "change_returned": None,
            },
            "tax": {
                "vat_amount": {"vat_amount": 1.9, "currency": "EUR"},
                "vat_lines": [
                    {
                        "source_rows": ["R0002"],
                        "rate_percent": 7.0,
                        "net_amount": 10.0,
                        "vat_amount": 1.9,
                    }
                ],
            },
        }
        initial = pipeline.validate_receipt_deterministically(
            receipt=receipt,
            item_pipeline_result=None,
            item_pipeline_enabled=False,
            selected_scalar_tasks=[],
        )
        self.assertEqual(
            ["VAT_LINE_RATE_ARITHMETIC"],
            [check["code"] for check in initial["checks"] if check["status"] == "failed"],
        )

        def invoke_source(strategy: Any, *args: Any) -> dict[str, Any]:
            self.assertEqual("vat_source_evidence_v9", strategy.strategy_id)
            return {
                "answer": {
                    "vat_evidence_blocks": [
                        {
                            "context_rows": ["R0001"],
                            "source_row": "R0002",
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
                "request": {"prompt": {"id": "gemma.correction.vat_source_evidence"}},
                "metrics": {},
            }

        callbacks = CorrectionCallbacks(
            invoke_source_evidence=invoke_source,
            validate_receipt=lambda candidate, _: pipeline.validate_receipt_deterministically(
                receipt=candidate,
                item_pipeline_result=None,
                item_pipeline_enabled=False,
                selected_scalar_tasks=[],
            ),
            effective_item_pipeline=lambda previous, candidate: previous,
            write_artifact=lambda name, value: None,
        )
        corrected, final_validation, _, report = run_correction_coordinator(
            profile=load_correction_profile(ROOT / "correction" / "config" / "production.json"),
            callbacks=callbacks,
            transcription=(
                "BEGIN_RECEIPT\n"
                "R0001 :: MwSt % Netto Steuer Brutto\n"
                "R0002 :: 19% 10,00 1,90 11,90\n"
                "END_RECEIPT"
            ),
            receipt=receipt,
            initial_validation=initial,
            item_pipeline_result=None,
            enabled=True,
        )
        self.assertEqual(19.0, corrected["tax"]["vat_lines"][0]["rate_percent"])
        self.assertEqual("valid", final_validation["status"])
        self.assertEqual(
            "vat_source_evidence_v9",
            report["accepted_corrections"][0]["strategy_id"],
        )

    def test_final_total_specialist_integrates_with_production_validator(self) -> None:
        pipeline = self.pipeline
        receipt = {
            "merchant": {"name": None, "address": None},
            "receipt_metadata": {
                "date": None,
                "time": None,
                "receipt_number": None,
                "transaction_status": "not_clear",
                "currency": "EUR",
            },
            "items": [],
            "totals": {
                "final_purchase_total": {
                    "final_purchase_total": 13.0,
                    "currency": "EUR",
                },
                "pre_discount_total": None,
                "net_amount": None,
            },
            "discount": {"discount_total": None},
            "payment": {
                "payment_method": None,
                "payment_received": {
                    "payment_received": 20.0,
                    "currency": "EUR",
                },
                "change_returned": {
                    "change_returned": 8.0,
                    "currency": "EUR",
                },
            },
            "tax": {"vat_amount": None, "vat_lines": []},
        }
        initial = pipeline.validate_receipt_deterministically(
            receipt=receipt,
            item_pipeline_result=None,
            item_pipeline_enabled=False,
            selected_scalar_tasks=[],
        )
        self.assertEqual(
            ["PAYMENT_CHANGE_RECONCILIATION"],
            [check["code"] for check in initial["checks"] if check["status"] == "failed"],
        )

        def invoke_source(strategy: Any, *args: Any) -> dict[str, Any]:
            self.assertEqual("final_total_source_evidence_v2_4", strategy.strategy_id)
            return {
                "answer": {
                    "status": "resolved",
                    "label_row": "R0001",
                    "source_row": "R0001",
                    "label_text": "SUMME",
                    "value_text": "12,00",
                },
                "request": {"prompt": {"id": "gemma.correction.final_total_source_evidence"}},
                "metrics": {},
            }

        callbacks = CorrectionCallbacks(
            invoke_source_evidence=invoke_source,
            validate_receipt=lambda candidate, _: pipeline.validate_receipt_deterministically(
                receipt=candidate,
                item_pipeline_result=None,
                item_pipeline_enabled=False,
                selected_scalar_tasks=[],
            ),
            effective_item_pipeline=lambda previous, candidate: previous,
            write_artifact=lambda name, value: None,
        )
        corrected, final_validation, _, report = run_correction_coordinator(
            profile=load_correction_profile(ROOT / "correction" / "config" / "production.json"),
            callbacks=callbacks,
            transcription=(
                "BEGIN_RECEIPT\n"
                "R0001 :: SUMME 12,00\n"
                "R0002 :: GEGEBEN 20,00\n"
                "R0003 :: RUECKGELD 8,00\n"
                "END_RECEIPT"
            ),
            receipt=receipt,
            initial_validation=initial,
            item_pipeline_result=None,
            enabled=True,
        )
        self.assertEqual(
            12.0,
            corrected["totals"]["final_purchase_total"]["final_purchase_total"],
        )
        self.assertEqual("valid", final_validation["status"])
        self.assertEqual(
            "final_total_source_evidence_v2_4",
            report["accepted_corrections"][0]["strategy_id"],
        )

    def test_invalid_specialist_json_is_repaired_with_existing_schema(self) -> None:
        pipeline = self.pipeline
        strategy = pipeline.CORRECTION_PROFILE.strategies["final_total_source_evidence_v2_4"]
        invalid_content = (
            '{"status":"resolved","label_row":"R0001",'
            '"source_row":"R0001","label_text":"SUMME",'
            '"value_text":"24,40"'
        )
        repaired_content = (
            '{"status":"resolved","label_row":"R0001",'
            '"source_row":"R0001","label_text":"SUMME",'
            '"value_text":"24,40"}'
        )
        calls: list[dict[str, Any]] = []

        def fake_post_json(url: str, payload: dict[str, Any], timeout: float):
            calls.append(payload)
            if len(calls) == 1:
                return {
                    "model": "gemma4",
                    "message": {
                        "content": invalid_content,
                        "thinking": "source reasoning",
                    },
                    "done_reason": "stop",
                }
            return {
                "model": "gemma4",
                "message": {"content": repaired_content},
                "done_reason": "stop",
            }

        args = SimpleNamespace(
            gemma_model="gemma4",
            gemma_keep_alive="5m",
            temperature=0.0,
            gemma_seed=42,
            gemma_num_ctx=8192,
            gemma_timeout=180.0,
            ollama_url="http://localhost:11434",
        )
        with patch.object(pipeline, "post_json", side_effect=fake_post_json):
            result = pipeline.invoke_gemma_correction_source_evidence_task(
                args,
                strategy=strategy,
                transcription="BEGIN_RECEIPT\nR0001 :: SUMME 24,40\nEND_RECEIPT",
                num_predict=4096,
            )

        self.assertEqual("completed", result["status"])
        self.assertEqual("24,40", result["answer"]["value_text"])
        self.assertTrue(result["json_repair"]["triggered"])
        self.assertEqual("completed", result["json_repair"]["status"])
        self.assertEqual(2, len(calls))
        self.assertTrue(calls[0]["think"])
        self.assertNotIn("format", calls[0])
        self.assertFalse(calls[1]["think"])
        self.assertEqual(
            pipeline.PROMPT_REGISTRY.load_schema(strategy.prompt_id, strategy.prompt_version),
            calls[1]["format"],
        )
        self.assertEqual(0.0, calls[1]["options"]["temperature"])

    def test_invalid_json_with_non_stop_done_reason_is_not_repaired(self) -> None:
        pipeline = self.pipeline
        strategy = pipeline.CORRECTION_PROFILE.strategies["final_total_source_evidence_v2_4"]
        calls: list[dict[str, Any]] = []

        def fake_post_json(url: str, payload: dict[str, Any], timeout: float):
            calls.append(payload)
            return {
                "model": "gemma4",
                "message": {"content": '{"status":"resolved"'},
                "done_reason": "length",
            }

        args = SimpleNamespace(
            gemma_model="gemma4",
            gemma_keep_alive="5m",
            temperature=0.0,
            gemma_seed=42,
            gemma_num_ctx=8192,
            gemma_timeout=180.0,
            ollama_url="http://localhost:11434",
        )
        with patch.object(pipeline, "post_json", side_effect=fake_post_json):
            result = pipeline.invoke_gemma_correction_source_evidence_task(
                args,
                strategy=strategy,
                transcription="BEGIN_RECEIPT\nR0001 :: SUMME 24,40\nEND_RECEIPT",
                num_predict=4096,
            )

        self.assertEqual(1, len(calls))
        self.assertEqual("invalid_json", result["status"])
        self.assertFalse(result["json_repair"]["triggered"])
        self.assertEqual(
            "skipped_non_stop_done_reason",
            result["json_repair"]["status"],
        )

    def test_valid_specialist_json_does_not_trigger_repair(self) -> None:
        pipeline = self.pipeline
        strategy = pipeline.CORRECTION_PROFILE.strategies["final_total_source_evidence_v2_4"]
        valid_content = (
            '{"status":"resolved","label_row":"R0001",'
            '"source_row":"R0001","label_text":"SUMME",'
            '"value_text":"24,40"}'
        )
        calls: list[dict[str, Any]] = []

        def fake_post_json(url: str, payload: dict[str, Any], timeout: float):
            calls.append(payload)
            return {
                "model": "gemma4",
                "message": {"content": valid_content, "thinking": "reasoning"},
                "done_reason": "stop",
            }

        args = SimpleNamespace(
            gemma_model="gemma4",
            gemma_keep_alive="5m",
            temperature=0.0,
            gemma_seed=42,
            gemma_num_ctx=8192,
            gemma_timeout=180.0,
            ollama_url="http://localhost:11434",
        )
        with patch.object(pipeline, "post_json", side_effect=fake_post_json):
            result = pipeline.invoke_gemma_correction_source_evidence_task(
                args,
                strategy=strategy,
                transcription="BEGIN_RECEIPT\nR0001 :: SUMME 24,40\nEND_RECEIPT",
                num_predict=4096,
            )

        self.assertEqual(1, len(calls))
        self.assertFalse(result["json_repair"]["triggered"])
        self.assertEqual("not_needed", result["json_repair"]["status"])


if __name__ == "__main__":
    unittest.main()
