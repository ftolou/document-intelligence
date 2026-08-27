"""Provider-neutral helpers for strict JSON model responses."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from jsonschema import SchemaError, ValidationError
from jsonschema.validators import validator_for

from receipt_intelligence.application.ports.llm import MalformedGenerationError


class GenerationText(Protocol):
    @property
    def text(self) -> str: ...


class LLMJsonParseError(MalformedGenerationError):
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


def parse_json_from_llm(
    raw: str | GenerationText,
    *,
    response_json_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse and optionally schema-validate one provider-neutral model result."""

    raw_text = raw if isinstance(raw, str) else raw.text
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
            if response_json_schema is not None:
                _validate_json_schema(obj, response_json_schema, text=text)
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


def _validate_json_schema(
    value: dict[str, Any],
    schema: dict[str, Any],
    *,
    text: str,
) -> None:
    if not isinstance(schema, dict) or not schema:
        raise ValueError("response_json_schema must be a non-empty object.")
    try:
        validator_class = validator_for(schema)
        validator_class.check_schema(schema)
        validator_class(schema).validate(value)
    except SchemaError as exc:
        raise ValueError(f"Invalid response_json_schema: {exc.message}") from exc
    except ValidationError as exc:
        path = ".".join(str(part) for part in exc.absolute_path) or "<root>"
        raise LLMJsonParseError(
            f"LLM JSON does not match the response schema at {path}: {exc.message}",
            raw_len=len(text),
            raw_head=text[:240].replace("\n", " "),
            raw_tail=text[-240:].replace("\n", " "),
        ) from exc


__all__ = ["LLMJsonParseError", "parse_json_from_llm"]
