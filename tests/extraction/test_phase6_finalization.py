from __future__ import annotations

import json

from receipt_intelligence.extraction.contracts.presentation import (
    CategorizationResult,
    CategorizationStatus,
    FinalizationRequest,
)
from receipt_intelligence.extraction.contracts.validation import ValidationReport
from receipt_intelligence.extraction.presentation.artifacts import (
    CompatibilityFilesystemArtifactStore,
)
from receipt_intelligence.extraction.presentation.finalization import (
    CompatibilityFinalizationService,
)


def test_finalization_writes_compatible_artifacts_and_aliases(tmp_path) -> None:
    validation = ValidationReport.from_legacy(
        {"status": "valid", "checks": [], "import_decision": "valid"}
    )
    categorized = CategorizationResult(
        status=CategorizationStatus.OK,
        receipt={
            "items": [
                {"description": "Milk", "line_total": 1.29, "category_key": "groceries_dairy_eggs"}
            ],
            "totals": {"grand_total": 1.29},
            "categorization": {"item_count": 1, "categorized_count": 1},
        },
        prompt="prompt",
        raw_output="raw",
        model="gemma4",
    )
    service = CompatibilityFinalizationService(
        artifact_store=CompatibilityFilesystemArtifactStore(tmp_path),
        app_version="test-version",
    )
    result = service.finalize(
        FinalizationRequest(
            run_id="r1",
            receipt=categorized.receipt,
            validation=validation,
            categorization=categorized,
            stage_trace=({"stage": "next_finalize", "status": "running"},),
        )
    )
    expected = {
        "r1_receipt_final.json",
        "r1_receipt_final_reconciled.json",
        "r1_receipt_final_categorized.json",
        "r1_v14_validation_report.json",
        "r1_reconciliation_report.json",
        "r1_pipeline_meta.json",
        "r1_extraction_stage_trace.json",
        "r1_v14_14_categorization_prompt.txt",
        "r1_v14_14_categorization_raw.txt",
        "r1_v14_14_categorization_result.json",
        "latest_receipt_final.json",
        "latest_receipt_final_reconciled.json",
        "latest_receipt_final_categorized.json",
        "latest_v14_validation_report.json",
        "latest_pipeline_meta.json",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    categorization_artifact = json.loads(
        (tmp_path / "r1_v14_14_categorization_result.json").read_text()
    )
    assert "receipt" not in categorization_artifact
    assert categorization_artifact["schema_version"] == "v14_14_item_categories_1"
    persisted = json.loads((tmp_path / "r1_receipt_final.json").read_text())
    assert persisted["totals"]["grand_total"] == 1.29
    assert persisted["validation"]["import_decision"] == "valid"
    assert persisted["pipeline"]["no_generic_correction_fallback"] is True
    assert result.paths["receipt_final"].name == "r1_receipt_final.json"


def test_finalization_metadata_records_safety_invariants(tmp_path) -> None:
    validation = ValidationReport.from_legacy(
        {
            "status": "review_required",
            "checks": [
                {
                    "code": "ITEM_SUM_RECONCILIATION",
                    "status": "failed",
                    "severity": "review",
                    "message": "open",
                }
            ],
        }
    )
    categorization = CategorizationResult(
        status=CategorizationStatus.DISABLED,
        receipt={"items": [], "totals": {"grand_total": 10.0}},
    )
    service = CompatibilityFinalizationService(
        artifact_store=CompatibilityFilesystemArtifactStore(tmp_path),
        app_version="test-version",
    )
    result = service.finalize(
        FinalizationRequest(
            run_id="r2",
            receipt=categorization.receipt,
            validation=validation,
            categorization=categorization,
        )
    )
    safety = result.pipeline_metadata["safety"]
    assert safety["read_only_validation"] is True
    assert safety["no_generic_correction_fallback"] is True
    assert result.pipeline_metadata["validation"]["failed_codes"] == ["ITEM_SUM_RECONCILIATION"]
