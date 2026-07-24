from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("flask")
from flask import Flask  # noqa: E402

from receipt_intelligence.application.vlm import VlmAnalysisService  # noqa: E402
from receipt_intelligence.entrypoints.vlm_http import app as vlm_http  # noqa: E402
from receipt_intelligence.entrypoints.vlm_http.settings import VlmHttpSettings  # noqa: E402
from receipt_intelligence.web import request_parsing  # noqa: E402


def test_upload_options_ignore_external_urls_and_commands(monkeypatch) -> None:
    monkeypatch.setattr(request_parsing.settings, "OLLAMA_URL", "http://trusted-ollama:11434")
    monkeypatch.setattr(request_parsing.settings, "VLM_SERVICE_URL", "http://trusted-vlm:7870")
    monkeypatch.setattr(request_parsing.settings, "VLM_BACKEND", "http_service")
    monkeypatch.setattr(request_parsing.settings, "VLM_COMMAND", "trusted-vlm-command")
    monkeypatch.setattr(request_parsing.settings, "OLLAMA_CONTROL_MODE", "command")
    monkeypatch.setattr(request_parsing.settings, "OLLAMA_UNLOAD_COMMAND", "trusted-unload")
    monkeypatch.setattr(request_parsing.settings, "OLLAMA_START_COMMAND", "trusted-start")
    monkeypatch.setattr(request_parsing.settings, "VLM_TIMEOUT_SECONDS", 900.0)
    monkeypatch.setattr(request_parsing.settings, "VLM_GPU_ORCHESTRATION", "none")
    monkeypatch.setattr(request_parsing.settings, "OLLAMA_UNLOAD_BEFORE_VLM", False)
    monkeypatch.setattr(request_parsing.settings, "OLLAMA_RELOAD_AFTER_VLM", False)
    monkeypatch.setattr(request_parsing.settings, "OLLAMA_CONTROL_TIMEOUT_SECONDS", 120.0)
    monkeypatch.setattr(request_parsing.settings, "OLLAMA_RELOAD_PROMPT", "trusted-warmup")
    monkeypatch.setattr(request_parsing.settings, "OLLAMA_GPU_HANDOFF_WAIT_SECONDS", 3.0)

    app = Flask(__name__)
    with app.test_request_context(
        "/api/upload",
        method="POST",
        data={
            "ollama_url": "http://attacker/internal",
            "vlm_service_url": "http://attacker/vlm",
            "vlm_backend": "local",
            "vlm_command": "touch /tmp/pwned",
            "ollama_control_mode": "api",
            "ollama_unload_command": "touch /tmp/unload-pwned",
            "ollama_start_command": "touch /tmp/start-pwned",
            "vlm_timeout_seconds": "999999",
            "vlm_gpu_orchestration": "sequential",
            "ollama_unload_before_vlm": "1",
            "ollama_reload_after_vlm": "1",
            "ollama_control_timeout_seconds": "999999",
            "ollama_reload_prompt": "untrusted-warmup",
            "ollama_gpu_handoff_wait_seconds": "999999",
        },
    ):
        options = request_parsing.build_options_from_request()

    assert options["ollama_url"] == "http://trusted-ollama:11434"
    assert options["vlm_service_url"] == "http://trusted-vlm:7870"
    assert options["vlm_backend"] == "http_service"
    assert options["vlm_command"] == "trusted-vlm-command"
    assert options["ollama_control_mode"] == "command"
    assert options["ollama_unload_command"] == "trusted-unload"
    assert options["ollama_start_command"] == "trusted-start"
    assert options["vlm_timeout_seconds"] == 900.0
    assert options["gpu_orchestration"] == "none"
    assert options["unload_llm_before_vlm"] is False
    assert options["reload_llm_after_vlm"] is False
    assert options["ollama_control_timeout_seconds"] == 120.0
    assert options["ollama_reload_prompt"] == "trusted-warmup"
    assert options["ollama_gpu_handoff_wait_seconds"] == 3.0


def test_vlm_service_rejects_path_outside_allowed_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    image = outside / "receipt.jpg"
    image.write_bytes(b"image")
    path, error = vlm_http.resolve_allowed_image_path(
        image,
        (allowed.resolve(),),
    )

    assert path is None
    assert error == "Image path is outside the configured VLM input roots."


def test_vlm_service_uses_only_server_execution_policy(tmp_path: Path) -> None:
    allowed = tmp_path / "var"
    results = allowed / "vlm-results"
    uploads = allowed / "uploads"
    allowed.mkdir()
    uploads.mkdir()
    image = allowed / "receipt.jpg"
    image.write_bytes(b"not-a-real-image")

    class FakeEngine:
        def analyze(self, request):
            return {
                "status": "ok",
                "backend": "paddleocr_vl_cli",
                "observed_image": str(request.image_path),
                "observed_timeout": request.timeout_seconds,
            }

    settings = VlmHttpSettings(
        app_version="test",
        upload_dir=uploads,
        results_dir=results,
        allowed_input_roots=(allowed.resolve(),),
        backend="paddleocr_vl",
        command="",
        timeout_seconds=900.0,
        max_upload_mb=25,
        max_side_limit=0,
        runner="cli",
        engine="transformers",
        device="gpu:0",
        allow_cpu_fallback=False,
        host="127.0.0.1",
        port=7870,
    )
    service = VlmAnalysisService(
        engine=FakeEngine(),
        results_dir=results,
        service_version="test",
        timeout_seconds=settings.timeout_seconds,
        max_side_limit=settings.max_side_limit,
        runner_name=settings.runner,
        engine_name=settings.engine,
        device_name=settings.device,
    )
    app = vlm_http.create_app(settings=settings, analysis_service=service)
    client = app.test_client()
    response = client.post(
        "/api/vlm/analyze",
        json={
            "image_path": str(image),
            "run_id": "../../unsafe-run",
            "backend": "unsupported",
            "runner": "python",
            "command": "touch /tmp/pwned",
            "timeout_seconds": 999999,
            "max_side_limit": 999999,
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["run_id"] == "unsafe-run"
    assert payload["runner"] == "cli"
    assert payload["observed_timeout"] == settings.timeout_seconds
    assert (results / "unsafe-run").is_dir()
