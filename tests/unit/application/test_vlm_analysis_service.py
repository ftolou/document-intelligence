from __future__ import annotations

import json
from pathlib import Path

from receipt_intelligence.application.vlm import VlmAnalysisService


class FakeEngine:
    def __init__(self) -> None:
        self.image_path: Path | None = None
        self.result_dir: Path | None = None

    def analyze(self, request):
        self.image_path = request.image_path
        self.result_dir = request.result_dir
        return {"status": "ok", "backend": "fake", "raw_result": {"value": 1}}


def test_analysis_service_invokes_engine_and_enriches_result(tmp_path: Path) -> None:
    image = tmp_path / "receipt.jpg"
    image.write_bytes(b"not-an-image")
    engine = FakeEngine()
    service = VlmAnalysisService(
        engine=engine,
        results_dir=tmp_path / "results",
        service_version="test-version",
        timeout_seconds=45.0,
        max_side_limit=0,
        runner_name="cli",
        engine_name="transformers",
        device_name="gpu:0",
    )

    result = service.execute(image_path=image, run_id="run-7")

    assert result["status"] == "ok"
    assert result["service_version"] == "test-version"
    assert result["run_id"] == "run-7"
    assert engine.image_path == image
    assert engine.result_dir == tmp_path / "results" / "run-7"
    output = tmp_path / "results" / "run-7" / "run-7_vlm_service_raw.json"
    assert json.loads(output.read_text(encoding="utf-8"))["raw_result"] == {"value": 1}
