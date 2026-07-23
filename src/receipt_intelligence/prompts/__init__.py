"""Prompt template loader for receipt intelligence LLM stages.

Prompts are intentionally stored as separate text files so pipeline behaviour can
be reviewed and iterated without editing Python logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_PROMPT_DIR = Path(__file__).resolve().parent


def load_prompt_template(name: str) -> str:
    path = _PROMPT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def render_prompt_template(name: str, **values: Any) -> str:
    text = load_prompt_template(name)
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", "" if value is None else str(value))
    return text
