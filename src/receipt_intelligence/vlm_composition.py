"""Legacy in-process VLM composition retained for tests and migration only.

Production containers do not copy or execute this module. The standalone
`receipt_vlm_service` package owns the live VLM HTTP/CLI process.
"""

from __future__ import annotations

from receipt_intelligence.adapters.vlm.paddle_cli import PaddleCliVlmEngine
from receipt_intelligence.adapters.vlm.paddle_python import PaddlePythonVlmEngine
from receipt_intelligence.adapters.vlm.trusted_command import TrustedCommandVlmEngine
from receipt_intelligence.application.ports.vlm import VlmEngine
from receipt_intelligence.application.vlm.engines import FallbackVlmEngine, UnsupportedVlmEngine
from receipt_intelligence.vlm_client_composition import build_client_vlm_engine

_LOCAL_BACKENDS = {"paddleocr_vl", "paddleocr-vl", "paddleocrvl", "local"}
_CLI_RUNNERS = {"cli", "doc_parser", "paddleocr_cli"}
_PYTHON_RUNNERS = {"python", "python_api"}


def _local_vlm_engine(runner_name: str) -> VlmEngine:
    runner = (runner_name or "auto").strip().lower()
    if runner in _CLI_RUNNERS:
        return PaddleCliVlmEngine()
    if runner in _PYTHON_RUNNERS:
        return PaddlePythonVlmEngine()
    return FallbackVlmEngine(PaddlePythonVlmEngine(), PaddleCliVlmEngine())


def build_vlm_service_engine(
    *,
    backend_name: str,
    runner_name: str,
    trusted_command: str = "",
) -> VlmEngine:
    """Build the deprecated in-process service engine for compatibility tests."""
    if trusted_command.strip():
        return TrustedCommandVlmEngine(trusted_command)
    backend = (backend_name or "paddleocr_vl").strip().lower()
    if backend in _LOCAL_BACKENDS:
        return _local_vlm_engine(runner_name)
    return UnsupportedVlmEngine(backend)


__all__ = ["build_client_vlm_engine", "build_vlm_service_engine"]
