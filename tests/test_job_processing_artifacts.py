from __future__ import annotations

import json
from pathlib import Path

from receipt_intelligence.services.job_processing import JobProcessingService


class _Store:
    def __init__(self) -> None:
        self.registered: list[str] = []
        self.jobs: dict[str, dict] = {}

    def get(self, job_id: str) -> dict | None:
        return self.jobs.get(job_id)

    def register_artifact(
        self, job_id: str, key: str, path: Path, category: str | None = None
    ) -> None:
        del job_id, path, category
        self.registered.append(key)


def _write(path: Path, content: str = "{}") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_key_artifacts_omit_missing_legacy_files_for_next_pipeline(tmp_path: Path) -> None:
    service = JobProcessingService.__new__(JobProcessingService)
    service.store = _Store()

    image = _write(tmp_path / "receipt.jpg", "image")
    ocr = _write(tmp_path / "ocr.json", json.dumps({"lines": []}))
    final = _write(tmp_path / "receipt_final_reconciled.json")
    validation = _write(tmp_path / "validation.json")
    pipeline_meta = _write(tmp_path / "pipeline_meta.json")
    stage_trace = _write(tmp_path / "stage_trace.json", "[]")
    metrics = _write(tmp_path / "metrics.json")

    artifacts = service._build_key_artifacts(
        "job-1",
        image_path=image,
        ocr_json_path=ocr,
        paths={
            "receipt_final_reconciled": str(final),
            "validation_report": str(validation),
            "pipeline_meta": str(pipeline_meta),
            "stage_trace": str(stage_trace),
            "extraction_metrics": str(metrics),
            "llm_main_prompt": str(tmp_path / "missing_prompt.txt"),
            "llm_main_raw": str(tmp_path / "missing_raw.txt"),
            "ocr_context": str(tmp_path / "missing_ocr_context.json"),
        },
    )

    assert "final_receipt" in artifacts
    assert "final_receipt_reconciled" in artifacts
    assert "validation_report" in artifacts
    assert "pipeline_meta" in artifacts
    assert "stage_trace" in artifacts
    assert "extraction_metrics" in artifacts
    assert "llm_prompt" not in artifacts
    assert "llm_raw" not in artifacts
    assert "ocr_context" not in artifacts


def test_batch_summary_adapts_next_validation_report() -> None:
    service = JobProcessingService.__new__(JobProcessingService)
    service.store = _Store()
    service.store.jobs["job-1"] = {
        "state": "completed",
        "result": {
            "report": {
                "status": "review_required",
                "metrics": {
                    "item_sum": 24.63,
                    "final_purchase_total": 60.5,
                },
                "checks": [
                    {
                        "code": "ITEM_PRICES_COMPLETE",
                        "status": "failed",
                        "severity": "review",
                        "message": "One item has no price.",
                    }
                ],
            },
            "artifacts": {},
        },
    }

    item = service._batch_item_from_job("job-1", Path("ikea.jpg"))
    assert item["decision"] == "needs_review"
    assert item["balanced"] is False
    assert item["difference"] == -35.87
    assert item["issue_count"] == 1
