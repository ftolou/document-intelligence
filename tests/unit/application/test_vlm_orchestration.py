from __future__ import annotations

import json
from pathlib import Path

from receipt_intelligence.application.ports.vlm import VlmRequest
from receipt_intelligence.application.vlm import FallbackVlmEngine, OptionalVlmEngine


class RecordingEngine:
    def __init__(self, result: dict[str, object]) -> None:
        self.result = result
        self.calls: list[VlmRequest] = []

    def analyze(self, request: VlmRequest) -> dict[str, object]:
        self.calls.append(request)
        return dict(self.result)


def request(tmp_path: Path, *, enabled: bool = True) -> VlmRequest:
    image = tmp_path / "receipt.jpg"
    image.write_bytes(b"image")
    return VlmRequest(
        image_path=image,
        result_dir=tmp_path,
        run_id="run-1",
        enabled=enabled,
        timeout_seconds=12.0,
    )


def test_fallback_is_not_called_when_primary_succeeds(tmp_path: Path) -> None:
    primary = RecordingEngine({"status": "ok", "backend": "python"})
    fallback = RecordingEngine({"status": "ok", "backend": "cli"})

    result = FallbackVlmEngine(primary, fallback).analyze(request(tmp_path))

    assert result["backend"] == "python"
    assert len(primary.calls) == 1
    assert fallback.calls == []


def test_fallback_result_records_primary_failure(tmp_path: Path) -> None:
    primary = RecordingEngine(
        {"status": "error", "backend": "python", "error": "initialization failed"}
    )
    fallback = RecordingEngine({"status": "ok", "backend": "cli"})

    result = FallbackVlmEngine(primary, fallback).analyze(request(tmp_path))

    assert result["backend"] == "cli"
    assert result["primary_backend"] == "python"
    assert result["primary_error"] == "initialization failed"
    assert len(fallback.calls) == 1


def test_optional_engine_persists_disabled_result_without_calling_adapter(tmp_path: Path) -> None:
    delegate = RecordingEngine({"status": "ok"})
    engine = OptionalVlmEngine(delegate, backend_name="http_service")

    result = engine.analyze(request(tmp_path, enabled=False))

    assert result["status"] == "disabled"
    assert delegate.calls == []
    output = tmp_path / "run-1_v14_7_vlm_raw_output.json"
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "disabled"


def test_optional_engine_delegates_and_persists_result(tmp_path: Path) -> None:
    delegate = RecordingEngine({"status": "ok", "raw_result": {"tables": 1}})
    engine = OptionalVlmEngine(delegate, backend_name="http_service")

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
        vlm_enabled=True,
        vlm_backend="http_service",
        vlm_service_url="http://receipt-vlm:7870",
    )

    engine = build_client_vlm_engine(config)

    assert isinstance(engine, OptionalVlmEngine)
    assert isinstance(engine.delegate, RemoteVlmClient)


def test_legacy_local_backend_name_still_routes_to_remote_service(tmp_path: Path) -> None:
    from receipt_intelligence.adapters.vlm import RemoteVlmClient
    from receipt_intelligence.extraction.config import ExtractionConfig
    from receipt_intelligence.vlm_client_composition import build_client_vlm_engine

    config = ExtractionConfig(
        ocr_json_path=tmp_path / "ocr.json",
        result_dir=tmp_path,
        run_id="run-legacy-local",
        ollama_url="http://ollama:11434",
        model="test-model",
        vlm_enabled=True,
        vlm_backend="paddleocr_vl",
        vlm_service_url="http://receipt-vlm:7870",
    )

    engine = build_client_vlm_engine(config)

    assert isinstance(engine, OptionalVlmEngine)
    assert isinstance(engine.delegate, RemoteVlmClient)
    assert engine.backend_name == "http_service"
