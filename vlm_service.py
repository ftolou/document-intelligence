#!/usr/bin/env python3
"""Compatibility entrypoint for the standalone VLM HTTP transport."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from receipt_intelligence.entrypoints.vlm_http.app import app as app, main  # noqa: E402

if __name__ == "__main__":
    main()
