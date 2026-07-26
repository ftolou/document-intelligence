"""Application-side VLM adapters.

The main application imports only the remote HTTP client. Local PaddleOCR-VL
engines remain legacy modules and are never imported by package initialization.
"""

from receipt_intelligence.adapters.vlm.remote_client import RemoteVlmClient, call_remote_vlm

__all__ = ["RemoteVlmClient", "call_remote_vlm"]
