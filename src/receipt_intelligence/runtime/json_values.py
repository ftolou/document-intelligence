"""JSON normalization helpers for model and transport boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def jsonable(value: Any) -> Any:
    """Convert provider-specific result objects into JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    for attribute in ("json", "to_dict", "dict", "model_dump"):
        if not hasattr(value, attribute):
            continue
        try:
            candidate = getattr(value, attribute)
            candidate = candidate() if callable(candidate) else candidate
            if isinstance(candidate, str):
                try:
                    return json.loads(candidate)
                except Exception:
                    return candidate
            return jsonable(candidate)
        except Exception:
            continue
    if hasattr(value, "res"):
        try:
            return jsonable(value.res)
        except Exception:
            pass
    return str(value)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


__all__ = ["jsonable", "save_json"]
