from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from receipt_intelligence.composition import build_extraction_dependencies
from receipt_intelligence.extraction import ExtractionConfig, ExtractionContext
from receipt_intelligence.extraction.state import (
    ExtractionPhase,
    ParsingArtifacts,
    PreparedArtifacts,
    StageContractError,
    VisualArtifacts,
)
from receipt_intelligence.extraction.workflow import ReceiptExtractionWorkflow


class _PhaseStage:
    name = "phase_stage"
    input_phase = ExtractionPhase.PREPARED
    output_phase = ExtractionPhase.VISUAL_READY

    def run(self, context: ExtractionContext) -> ExtractionContext:
        context.visual = VisualArtifacts()
        return context


class ExtractionStateContractTests(unittest.TestCase):
    def _context(self, root: Path) -> ExtractionContext:
        config = ExtractionConfig(
            ocr_json_path=root / "ocr.json",
            result_dir=root,
            run_id="state-contract",
            ollama_url="http://ollama",
            model="model",
        )
        return ExtractionContext(
            config=config,
            dependencies=build_extraction_dependencies(config),
        )

    def test_context_starts_without_illegal_partial_stage_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self._context(Path(temporary_directory))

            self.assertEqual(context.phase, ExtractionPhase.CREATED)
            self.assertIsNone(context.prepared)
            self.assertIsNone(context.visual)
            self.assertIsNone(context.parsed)
            self.assertIsNone(context.repair)
            self.assertIsNone(context.finalized)
            with self.assertRaisesRegex(StageContractError, "prepared artifacts"):
                _ = context.paths

    def test_workflow_rejects_a_stage_started_from_the_wrong_phase(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self._context(Path(temporary_directory))
            workflow = ReceiptExtractionWorkflow([_PhaseStage()])

            with self.assertRaisesRegex(StageContractError, "requires phase 'prepared'"):
                workflow.run(context)

            self.assertEqual(context.phase, ExtractionPhase.CREATED)

    def test_phase_advance_occurs_only_after_stage_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self._context(Path(temporary_directory))
            context.prepared = PreparedArtifacts(paths={})
            context.phase = ExtractionPhase.PREPARED

            result = ReceiptExtractionWorkflow([_PhaseStage()]).run(context)

            self.assertIs(result, context)
            self.assertEqual(context.phase, ExtractionPhase.VISUAL_READY)
            self.assertIsNotNone(context.visual)
            self.assertEqual(context.stage_trace[0]["input_phase"], "prepared")
            self.assertEqual(context.stage_trace[0]["output_phase"], "visual_ready")

    def test_selected_candidate_is_owned_by_repair_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = self._context(Path(temporary_directory))
            context.prepared = PreparedArtifacts(paths={})
            context.visual = VisualArtifacts()
            context.parsed = ParsingArtifacts(
                receipt={"id": "initial"},
                current_report={"import_decision": "review"},
            )

            repair = context.begin_repair_stage()
            context.receipt = {"id": "selected"}
            context.report = {"import_decision": "import"}
            context.correction_used = True

            self.assertEqual(repair.selected_receipt, {"id": "selected"})
            self.assertEqual(repair.selected_report, {"import_decision": "import"})
            self.assertTrue(repair.correction_used)
            self.assertEqual(context.parsed.receipt, {"id": "initial"})


if __name__ == "__main__":
    unittest.main()
