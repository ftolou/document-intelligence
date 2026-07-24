"""Filesystem-safe filename normalization without framework dependencies."""

from __future__ import annotations

import re
from pathlib import Path

_SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(filename: str, *, fallback: str = "") -> str:
    leaf = Path(str(filename).replace("\\", "/")).name.strip().replace(" ", "_")
    safe = _SAFE_FILENAME_PATTERN.sub("_", leaf).strip("._")
    if not safe:
        return fallback
    suffix = Path(leaf).suffix.lower()
    if suffix and not safe.lower().endswith(suffix):
        safe = f"{safe}{suffix}"
    return safe[:240]


__all__ = ["safe_filename"]
