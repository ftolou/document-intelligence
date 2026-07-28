from __future__ import annotations

import json
from pathlib import Path

import pytest

from receipt_intelligence.application.ports.vlm import VlmRequest
from receipt_intelligence.application.vlm import RequiredVlmEngine


class RecordingEngine:
    def __init__(self, result: dict[str, object]) -> None:
        self.result = result
        self.calls: list[VlmRequest] = []

    def analyze(self, request: VlmRequest) -> dict[str, object]:
        self.calls.append(request)
        return dict(self.result)


def request(tmp_path: Path) -> VlmRequest:
    image = tmp_path / "receipt.jpg"
    image.write_bytes(b"image")
    return VlmRequest(
        image_path=image,
        result_dir=tmp_path,
        run_id="run-1",
        timeout_seconds=12.0,
    )


def test_required_engine_rejects_missing_source_image(tmp_path: Path) -> None:
    delegate = RecordingEngine({"status": "ok"})
    engine = RequiredVlmEngine(delegate, backend_name="http_service")
    missing = VlmRequest(
        image_path=tmp_path / "missing.jpg",
        result_dir=tmp_path,
        run_id="run-1",
    )

    with pytest.raises(FileNotFoundError, match="source receipt image"):
        engine.analyze(missing)

    assert delegate.calls == []


def test_required_engine_delegates_and_persists_result(tmp_path: Path) -> None:
    delegate = RecordingEngine({"status": "ok", "raw_result": {"tables": 1}})
    engine = RequiredVlmEngine(delegate, backend_name="http_service")

    result = engine.analyze(request(tmp_path))

    assert result["status"] == "ok"
    assert result["image_path"].endswith("receipt.jpg")
    assert len(delegate.calls) == 1
    output = tmp_path / "run-1_v14_7_vlm_raw_output.json"
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["raw_result"] == {"tables": 1}


def test_composition_selects_remote_client_for_application(tmp_path: Path) -> None:
    from receipt_intelligence.adapters.vlm import RemoteVlmClient
    from receipt_intelligence.extraction.config import ExtractionConfig
    from receipt_intelligence.vlm_client_composition import build_client_vlm_engine

    config = ExtractionConfig(
        ocr_json_path=tmp_path / "ocr.json",
        result_dir=tmp_path,
        run_id="run-remote",
        ollama_url="http://ollama:11434",
        model="test-model",
        vlm_backend="http_service",
        vlm_service_url="http://receipt-vlm:7870",
    )

    engine = build_client_vlm_engine(config)

    assert isinstance(engine, RequiredVlmEngine)
    assert isinstance(engine.delegate, RemoteVlmClient)


def test_historical_local_backend_name_routes_to_remote_service(tmp_path: Path) -> None:
    from receipt_intelligence.adapters.vlm import RemoteVlmClient
    from receipt_intelligence.extraction.config import ExtractionConfig
    from receipt_intelligence.vlm_client_composition import build_client_vlm_engine

    config = ExtractionConfig(
        ocr_json_path=tmp_path / "ocr.json",
        result_dir=tmp_path,
        run_id="run-legacy-local",
        ollama_url="http://ollama:11434",
        model="test-model",
        vlm_backend="paddleocr_vl",
        vlm_service_url="http://receipt-vlm:7870",
    )

    engine = build_client_vlm_engine(config)

    assert isinstance(engine, RequiredVlmEngine)
    assert isinstance(engine.delegate, RemoteVlmClient)
    assert engine.backend_name == "http_service"
