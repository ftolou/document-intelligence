"""Provider- and storage-neutral text normalization helpers."""

from __future__ import annotations

import re
from typing import Any


def normalize_text(value: Any) -> str:
    """Normalize text for deterministic matching and identity keys."""

    text = str(value or "").lower()
    replacements = {
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "ß": "ss",
        "&": " and ",
        "+": " plus ",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


__all__ = ["normalize_text"]
