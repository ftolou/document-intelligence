from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from receipt_intelligence.composition import build_extraction_dependencies
from receipt_intelligence.extraction import (
    ExtractionConfig,
    ExtractionContext,
    ReceiptExtractionWorkflow,
    build_default_extraction_workflow,
)
from receipt_intelligence.pipeline.integrated_receipt_pipeline import (
    run_integrated_receipt_pipeline,
)


class _RecordingStage:
    def __init__(self, name: str, calls: list[str], *, fail: bool = False) -> None:
        self.name = name
        self.calls = calls
        self.fail = fail

    def run(self, context: ExtractionContext) -> ExtractionContext:
        self.calls.append(self.name)
        if self.fail:
            raise RuntimeError("stage failed")
        return context


class StagedExtractionWorkflowTests(unittest.TestCase):
    def test_default_workflow_has_explicit_ordered_stages(self) -> None:
        workflow = build_default_extraction_workflow()
        self.assertEqual(
            [stage.name for stage in workflow.stages],
            [
                "prepare",
                "visual_evidence",
                "spatial_overview",
                "main_parsing",
                "repair_and_correction",
                "finalize",
            ],
        )

    def test_workflow_records_stage_order_and_stops_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = ExtractionConfig(
                ocr_json_path=Path(temporary_directory) / "ocr.json",
                result_dir=Path(temporary_directory),
                run_id="test",
                ollama_url="http://ollama",
                model="model",
            )
            context = ExtractionContext(
                config=config,
                dependencies=build_extraction_dependencies(config),
            )
            calls: list[str] = []
            workflow = ReceiptExtractionWorkflow(
                [
                    _RecordingStage("first", calls),
                    _RecordingStage("second", calls, fail=True),
                    _RecordingStage("third", calls),
                ]
            )

            with self.assertRaisesRegex(RuntimeError, "stage failed"):
                workflow.run(context)

            self.assertEqual(calls, ["first", "second"])
            self.assertEqual(context.stage_trace[0]["status"], "done")
            self.assertEqual(context.stage_trace[1]["status"], "error")

    def test_spatial_strategy_builds_geometry_before_main_parser_without_overview_llm(self) -> None:
        report = {
            "import_decision": "needs_review",
            "balanced": False,
            "difference": 1.0,
            "issues": ["test"],
            "failure_diagnosis": None,
        }
        receipt = {"schema_version": "test", "items": [], "warnings": []}
        ocr_context = {
            "image_width": 100,
            "image_height": 200,
            "words": [],
            "lines": [
                {
                    "line_id": "line_000",
                    "text": "TEST 1,00",
                    "bbox": {"x": 0.1, "y": 0.2, "w": 0.8, "h": 0.05},
                    "source_word_ids": [],
                }
            ],
            "layout_context": {},
            "layout_rows": [],
        }
        llm_result = {
            "receipt": receipt,
            "ocr_context": ocr_context,
            "prompt": "spatial prompt",
            "raw_output": "{}",
            "error": None,
            "attempts": [],
            "visual_evidence_used": False,
            "spatial_overview_used": True,
            "spatial_geometry_used": True,
            "response_schema_enforced": True,
            "duration_seconds": 0.01,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            ocr_path = root / "ocr.json"
            ocr_path.write_text(json.dumps({"words": []}), encoding="utf-8")
            with (
                patch(
                    "receipt_intelligence.extraction.stages.prepare.build_ocr_context",
                    return_value=ocr_context,
                ),
                patch(
                    "receipt_intelligence.extraction.stages.parse.run_llm_main_parser",
                    return_value=llm_result,
                ) as parser_call,
                patch(
                    "receipt_intelligence.extraction.stages.parse.build_compact_evidence",
                    return_value={},
                ),
                patch(
                    "receipt_intelligence.extraction.stages.parse.build_grouped_evidence",
                    return_value={},
                ),
                patch(
                    "receipt_intelligence.extraction.stages.parse.apply_consistency_postprocess",
                    return_value=(receipt, []),
                ),
                patch(
                    "receipt_intelligence.extraction.stages.parse.validate_receipt",
                    return_value=report,
                ),
                patch(
                    "receipt_intelligence.extraction.stages.finalize.sanitize_model_warnings",
                    return_value=(receipt, []),
                ),
            ):
                result = run_integrated_receipt_pipeline(
                    ocr_json_path=ocr_path,
                    result_dir=root,
                    run_id="spatial-1",
                    ollama_url="http://ollama",
                    model="gemma",
                    vlm_enabled=False,
                    correction_enabled=False,
                    categorization_enabled=False,
                )

            self.assertIsNotNone(parser_call.call_args.kwargs["spatial_document_map"])
            self.assertTrue(result["pipeline_meta"]["spatial_overview"]["enabled"])
            self.assertEqual(
                result["pipeline_meta"]["spatial_overview"]["mode"],
                "deterministic_geometry",
            )
            self.assertTrue((root / "spatial-1_spatial_document_map.json").exists())
            self.assertTrue((root / "spatial-1_spatial_overview.json").exists())

    def test_compatibility_entry_point_runs_the_staged_workflow(self) -> None:
        report = {
            "import_decision": "import",
            "balanced": True,
            "difference": 0.0,
            "issues": [],
            "failure_diagnosis": None,
        }
        receipt = {"schema_version": "test", "items": [], "warnings": []}
        ocr_context = {"layout_context": {}, "layout_rows": []}
        llm_result = {
            "receipt": receipt,
            "ocr_context": ocr_context,
            "prompt": "prompt",
            "raw_output": "{}",
            "error": None,
            "attempts": 1,
            "visual_evidence_used": False,
            "duration_seconds": 0.01,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            ocr_path = root / "ocr.json"
            ocr_path.write_text(json.dumps({"pages": []}), encoding="utf-8")

            with (
                patch(
                    "receipt_intelligence.extraction.stages.prepare.build_ocr_context",
                    return_value=ocr_context,
                ),
                patch(
                    "receipt_intelligence.extraction.stages.parse.run_llm_main_parser",
                    return_value=llm_result,
                ),
                patch(
                    "receipt_intelligence.extraction.stages.parse.build_compact_evidence",
                    return_value={},
                ),
                patch(
                    "receipt_intelligence.extraction.stages.parse.build_grouped_evidence",
                    return_value={},
                ),
                patch(
                    "receipt_intelligence.extraction.stages.parse.apply_consistency_postprocess",
                    return_value=(receipt, []),
                ),
                patch(
                    "receipt_intelligence.extraction.stages.parse.validate_receipt",
                    return_value=report,
                ),
                patch(
                    "receipt_intelligence.extraction.stages.finalize.sanitize_model_warnings",
                    return_value=(receipt, []),
                ),
            ):
                result = run_integrated_receipt_pipeline(
                    ocr_json_path=ocr_path,
                    result_dir=root,
                    run_id="receipt-1",
                    ollama_url="http://ollama",
                    model="gemma",
                    vlm_enabled=False,
                    correction_enabled=False,
                    categorization_enabled=False,
                )

            self.assertEqual(result["report"], report)
            self.assertEqual(
                result["pipeline_meta"]["workflow"]["stages"],
                [
                    "prepare",
                    "visual_evidence",
                    "spatial_overview",
                    "main_parsing",
                    "repair_and_correction",
                    "finalize",
                ],
            )
            self.assertTrue((root / "receipt-1_receipt_final.json").exists())
            self.assertTrue((root / "receipt-1_extraction_stage_trace.json").exists())
            metrics_path = root / "receipt-1_extraction_metrics.json"
            self.assertTrue(metrics_path.exists())
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["status"], "completed")
            self.assertEqual(metrics["completed_stage_count"], 6)
            self.assertTrue(all("duration_ms" in row for row in metrics["stages"]))
            self.assertTrue((root / "latest_receipt_final.json").exists())
            self.assertIn("latest_receipt_final", result["paths"])
            self.assertIn("latest_v14_validation_report", result["paths"])
            self.assertIn("latest_extraction_metrics", result["paths"])


if __name__ == "__main__":
    unittest.main()
