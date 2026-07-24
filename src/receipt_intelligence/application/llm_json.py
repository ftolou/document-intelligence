"""Provider-neutral helpers for strict JSON model responses."""

from __future__ import annotations

import json
import re
from typing import Any

from receipt_intelligence.application.ports.llm import GenerationResult


class LLMJsonParseError(ValueError):
    """Raised when a model answered but did not return usable JSON."""

    def __init__(
        self,
        message: str,
        *,
        raw_len: int = 0,
        raw_head: str = "",
        raw_tail: str = "",
    ) -> None:
        super().__init__(message)
        self.raw_len = raw_len
        self.raw_head = raw_head
        self.raw_tail = raw_tail


def _strip_code_fences(text: str) -> str:
    text = (text or "").strip().lstrip("\ufeff")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()
    return text


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escape:
                escape = False
            elif character == "\\":
                escape = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_json_from_llm(raw: str | GenerationResult) -> dict[str, Any]:
    """Parse one JSON object from a model result without provider coupling."""

    raw_text = raw.text if isinstance(raw, GenerationResult) else raw
    text = _strip_code_fences(raw_text)
    raw_head = text[:240].replace("\n", " ")
    raw_tail = text[-240:].replace("\n", " ")
    if not text:
        raise LLMJsonParseError("LLM returned empty text instead of JSON", raw_len=0)

    candidates: list[str] = [text]
    balanced = _extract_first_json_object(text)
    if balanced:
        candidates.append(balanced)

    last_exc: Exception | None = None
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if not isinstance(obj, dict):
                raise LLMJsonParseError(
                    "LLM JSON root must be an object",
                    raw_len=len(text),
                    raw_head=raw_head,
                    raw_tail=raw_tail,
                )
            return obj
        except json.JSONDecodeError as exc:
            last_exc = exc

    if isinstance(last_exc, json.JSONDecodeError):
        message = (
            f"Invalid/incomplete JSON from LLM at line {last_exc.lineno}, "
            f"column {last_exc.colno}: {last_exc.msg}. "
            f"raw_len={len(text)}, head={raw_head!r}, tail={raw_tail!r}"
        )
        raise LLMJsonParseError(
            message,
            raw_len=len(text),
            raw_head=raw_head,
            raw_tail=raw_tail,
        ) from last_exc
    raise LLMJsonParseError(
        f"Invalid LLM JSON. raw_len={len(text)}, head={raw_head!r}, tail={raw_tail!r}",
        raw_len=len(text),
        raw_head=raw_head,
        raw_tail=raw_tail,
    )


__all__ = ["LLMJsonParseError", "parse_json_from_llm"]
