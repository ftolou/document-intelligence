#!/usr/bin/env python3
"""Compatibility entrypoint for the src-layout package."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from receipt_intelligence.app import app, main

if __name__ == "__main__":
    main()
