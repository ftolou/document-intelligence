"""Visual-model infrastructure adapters."""

from receipt_intelligence.adapters.vlm.paddle_cli import PaddleCliVlmEngine
from receipt_intelligence.adapters.vlm.paddle_python import PaddlePythonVlmEngine
from receipt_intelligence.adapters.vlm.remote_client import RemoteVlmClient
from receipt_intelligence.adapters.vlm.trusted_command import TrustedCommandVlmEngine

__all__ = [
    "PaddleCliVlmEngine",
    "PaddlePythonVlmEngine",
    "RemoteVlmClient",
    "TrustedCommandVlmEngine",
]
