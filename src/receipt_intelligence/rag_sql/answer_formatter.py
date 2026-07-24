"""Bounded LLM formatter for ambiguous reviewed RAG-SQL evidence.

The model may normalize values only from approved SQL result fields. A
separate deterministic validator rejects unsupported values, unknown item IDs,
and merchant-like brand guesses before any user-facing answer is rendered.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from receipt_intelligence.application.generation import (
    LegacyGenerateFunction,
    invoke_generation,
)
from receipt_intelligence.application.llm_json import parse_json_from_llm
from receipt_intelligence.application.ports.llm import (
    GenerationRequest,
    LlmGateway,
    ModelCallMetrics,
)
from receipt_intelligence.prompts import render_prompt_template
from receipt_intelligence.rag_sql.formatter import (
    APPROVED_EVIDENCE_FIELDS,
    format_descriptive_values,
)
from receipt_intelligence.rag_sql.models import StrictModel

ANSWER_FORMAT_SCHEMA_VERSION = "rag_sql_answer_format_v1"
_ALLOWED_EVIDENCE_FIELDS = frozenset(APPROVED_EVIDENCE_FIELDS - {"item_id"})
_PRODUCT_IDENTITY_FIELDS = frozenset(
    {"description", "normalized_name", "semantic_description", "reviewed_brand"}
)
_OPERATION_FIELDS: dict[str, frozenset[str]] = {
    "identify_brand": frozenset(
        {
            "description",
            "normalized_name",
            "semantic_description",
            "category_reason",
            "reviewed_brand",
        }
    ),
    "identify_product_type": frozenset(
        {
            "description",
            "normalized_name",
            "semantic_description",
            "category",
            "category_reason",
        }
    ),
    "describe_product": frozenset(
        {
            "description",
            "normalized_name",
            "semantic_description",
            "category",
            "category_reason",
        }
    ),
}


class AnswerFormatterPayload(StrictModel):
    schema_version: Literal["rag_sql_answer_format_v1"] = ANSWER_FORMAT_SCHEMA_VERSION
    status: Literal["resolved", "insufficient_info", "ambiguous"]
    values: list[str] = Field(default_factory=list, max_length=20)
    supporting_item_ids: list[int] = Field(default_factory=list, max_length=100)
    evidence_fields: list[str] = Field(default_factory=list, max_length=10)
    reason: str = Field(min_length=1, max_length=1000)

    @field_validator("values")
    @classmethod
    def validate_values(cls, values: list[str]) -> list[str]:
        normalized = [" ".join(value.split()).strip() for value in values]
        if any(not value or len(value) > 200 for value in normalized):
            raise ValueError("values must contain non-empty strings of at most 200 characters.")
        if len({value.casefold() for value in normalized}) != len(normalized):
            raise ValueError("values must not contain duplicates.")
        return normalized

    @field_validator("supporting_item_ids")
    @classmethod
    def validate_item_ids(cls, values: list[int]) -> list[int]:
        if any(value <= 0 for value in values):
            raise ValueError("supporting_item_ids must be positive integers.")
        if len(set(values)) != len(values):
            raise ValueError("supporting_item_ids must not contain duplicates.")
        return values

    @field_validator("evidence_fields")
    @classmethod
    def validate_evidence_fields(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("evidence_fields must not contain duplicates.")
        unknown = sorted(set(values) - _ALLOWED_EVIDENCE_FIELDS)
        if unknown:
            raise ValueError(f"Unsupported evidence fields: {unknown}.")
        return values

    @model_validator(mode="after")
    def validate_status_contract(self) -> Self:
        if self.status == "resolved":
            if not self.values or not self.supporting_item_ids or not self.evidence_fields:
                raise ValueError(
                    "resolved status requires values, supporting_item_ids, and evidence_fields."
                )
        elif self.values:
            raise ValueError("Only resolved status may return values.")
        return self


class AnswerFormatterResult(AnswerFormatterPayload):
    model: str | None = Field(default=None, max_length=200)
    attempts: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0, ge=0.0)
    ollama_calls: list[ModelCallMetrics] = Field(default_factory=list, max_length=20)


class AnswerValidationResult(StrictModel):
    status: Literal["valid", "invalid", "insufficient_info"]
    values: list[str] = Field(default_factory=list)
    supporting_item_ids: list[int] = Field(default_factory=list)
    evidence_fields: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1, max_length=2000)


class AnswerFormattingError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        ollama_calls: list[ModelCallMetrics] | None = None,
    ) -> None:
        super().__init__(message)
        self.ollama_calls = list(ollama_calls or [])


@dataclass(frozen=True)
class AnswerFormatterConfig:
    enabled: bool = True
    ollama_url: str = "http://localhost:11434"
    model: str = "gemma4"
    num_ctx: int = 6144
    num_predict: int = 768
    timeout_seconds: float = 120.0
    retry_count: int = 1
    format_json: bool = True
    keep_alive: str | None = None
    maximum_rows: int = 100

    def __post_init__(self) -> None:
        if not str(self.ollama_url or "").strip():
            raise ValueError("ollama_url must not be empty.")
        if not str(self.model or "").strip():
            raise ValueError("model must not be empty.")
        if self.num_ctx <= 0 or self.num_predict <= 0:
            raise ValueError("num_ctx and num_predict must be positive.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if self.retry_count < 0 or self.retry_count > 3:
            raise ValueError("retry_count must be between 0 and 3.")
        if self.maximum_rows <= 0 or self.maximum_rows > 100:
            raise ValueError("maximum_rows must be between 1 and 100.")


class EvidenceBoundAnswerFormatter:
    def __init__(
        self,
        config: AnswerFormatterConfig,
        *,
        llm_gateway: LlmGateway | None = None,
        generate: LegacyGenerateFunction | None = None,
    ) -> None:
        self.config = config
        self.llm_gateway = llm_gateway
        self.generate = generate

    def format(
        self,
        *,
        question: str,
        requested_operation: str,
        language: str,
        rows: Sequence[Mapping[str, Any]],
        answer_instruction: str,
    ) -> AnswerFormatterResult:
        if not self.config.enabled:
            raise AnswerFormattingError("Hybrid answer formatting is disabled.")
        operation = str(requested_operation or "").casefold()
        if operation not in _OPERATION_FIELDS:
            raise AnswerFormattingError(f"Unsupported answer-format operation: {operation!r}.")
        evidence_rows = _sanitize_rows(rows, maximum_rows=self.config.maximum_rows)
        if not evidence_rows:
            raise AnswerFormattingError("No approved reviewed evidence rows were available.")

        started = time.perf_counter()
        previous_error: str | None = None
        last_error: Exception | None = None
        attempts = max(1, self.config.retry_count + 1)
        ollama_calls: list[ModelCallMetrics] = []

        for attempt in range(1, attempts + 1):
            retry_block = ""
            if previous_error:
                retry_block = (
                    "Previous response validation error:\n"
                    f"{previous_error}\n"
                    "Correct only the JSON. Do not add facts or evidence."
                )
            prompt = render_prompt_template(
                "rag_sql_answer_formatter.txt",
                QUESTION=" ".join(str(question or "").split()).strip(),
                REQUESTED_OPERATION=operation,
                LANGUAGE="de" if language == "de" else "en",
                ANSWER_INSTRUCTION=" ".join(str(answer_instruction or "").split()).strip(),
                EVIDENCE_ROWS_JSON=json.dumps(
                    evidence_rows,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                RETRY_BLOCK=retry_block,
            )
            try:
                generation = invoke_generation(
                    request=GenerationRequest(
                        model=self.config.model,
                        prompt=prompt,
                        operation="rag_sql_answer_formatting",
                        attempt=attempt,
                        num_ctx=self.config.num_ctx,
                        num_predict=self.config.num_predict,
                        temperature=0.0,
                        keep_alive=self.config.keep_alive,
                        timeout_seconds=self.config.timeout_seconds,
                        format_json=self.config.format_json,
                    ),
                    gateway=self.llm_gateway,
                    legacy_generate=self.generate,
                    legacy_base_url=self.config.ollama_url,
                )
                if generation.metrics is not None:
                    ollama_calls.append(generation.metrics)
                payload = AnswerFormatterPayload.model_validate(parse_json_from_llm(generation))
                return AnswerFormatterResult(
                    **payload.model_dump(mode="python"),
                    model=self.config.model,
                    attempts=attempt,
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                    ollama_calls=ollama_calls,
                )
            except Exception as exc:
                last_error = exc
                previous_error = f"{type(exc).__name__}: {exc}"

        duration_ms = (time.perf_counter() - started) * 1000.0
        raise AnswerFormattingError(
            "Evidence-bound answer formatting failed after "
            f"{attempts} attempt(s) in {duration_ms:.1f} ms: "
            f"{type(last_error).__name__ if last_error else 'UnknownError'}: {last_error}",
            ollama_calls=ollama_calls,
        ) from last_error


def validate_answer_formatter_result(
    result: AnswerFormatterResult,
    *,
    requested_operation: str,
    rows: Sequence[Mapping[str, Any]],
) -> AnswerValidationResult:
    """Validate that every model value is visibly supported by returned rows."""

    operation = str(requested_operation or "").casefold()
    allowed_fields = _OPERATION_FIELDS.get(operation)
    if allowed_fields is None:
        return AnswerValidationResult(
            status="invalid",
            reason=f"unsupported_operation:{operation or 'missing'}",
        )
    if result.status != "resolved":
        return AnswerValidationResult(
            status="insufficient_info",
            reason=f"formatter_status:{result.status}",
        )

    evidence_rows = _sanitize_rows(rows, maximum_rows=100)
    rows_by_id = {
        int(row["item_id"]): row
        for row in evidence_rows
        if isinstance(row.get("item_id"), int) and int(row["item_id"]) > 0
    }
    unknown_ids = sorted(set(result.supporting_item_ids) - set(rows_by_id))
    if unknown_ids:
        return AnswerValidationResult(
            status="invalid",
            reason=f"unknown_supporting_item_ids:{unknown_ids}",
        )

    invalid_fields = sorted(set(result.evidence_fields) - allowed_fields)
    if invalid_fields:
        return AnswerValidationResult(
            status="invalid",
            reason=f"invalid_evidence_fields:{invalid_fields}",
        )

    supporting_rows = [rows_by_id[item_id] for item_id in result.supporting_item_ids]
    for value in result.values:
        if not _value_supported(
            value,
            supporting_rows,
            evidence_fields=result.evidence_fields,
        ):
            return AnswerValidationResult(
                status="invalid",
                reason=f"unsupported_value:{value}",
            )
        if operation == "identify_brand" and not _value_supported(
            value,
            supporting_rows,
            evidence_fields=sorted(_PRODUCT_IDENTITY_FIELDS),
        ):
            return AnswerValidationResult(
                status="invalid",
                reason=f"brand_not_present_in_product_identity:{value}",
            )
        if operation == "identify_brand" and not _brand_has_positive_evidence(
            value,
            supporting_rows,
        ):
            return AnswerValidationResult(
                status="invalid",
                reason=f"brand_not_explicitly_supported:{value}",
            )
        if operation == "identify_brand" and _brand_has_explicit_non_brand_role(
            value,
            supporting_rows,
        ):
            return AnswerValidationResult(
                status="invalid",
                reason=f"brand_assigned_non_brand_role:{value}",
            )

    return AnswerValidationResult(
        status="valid",
        values=result.values,
        supporting_item_ids=result.supporting_item_ids,
        evidence_fields=result.evidence_fields,
        reason="all_values_supported_by_reviewed_rows",
    )


def render_validated_answer(
    validation: AnswerValidationResult,
    *,
    requested_operation: str,
    language: str,
) -> str:
    if validation.status != "valid":
        raise ValueError("Only a valid answer-format result may be rendered.")
    return format_descriptive_values(
        validation.values,
        operation=str(requested_operation or "").casefold(),
        german=language == "de",
    )


def _sanitize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    maximum_rows: int,
) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for row in rows[:maximum_rows]:
        compact: dict[str, Any] = {}
        item_id = row.get("item_id")
        if isinstance(item_id, int) and item_id > 0:
            compact["item_id"] = item_id
        for field in sorted(_ALLOWED_EVIDENCE_FIELDS):
            value = row.get(field)
            text = " ".join(str(value or "").split()).strip()
            if text:
                compact[field] = text[:2000]
        if compact:
            sanitized.append(compact)
    return sanitized


def _brand_has_positive_evidence(
    value: str,
    rows: Sequence[Mapping[str, Any]],
) -> bool:
    value_tokens = _tokens(value)
    if not value_tokens:
        return False
    for row in rows:
        reviewed_brand = str(row.get("reviewed_brand") or "")
        if _contains_token_sequence(_tokens(reviewed_brand), value_tokens):
            return True
        for field in ("semantic_description", "category_reason"):
            text = str(row.get(field) or "")
            tokens = _tokens(text)
            if _contains_token_sequence(tokens, value_tokens) and re.search(
                r"\b(?:brand|marke|branding)\b",
                text,
                re.IGNORECASE,
            ):
                return True
    return False


def _brand_has_explicit_non_brand_role(
    value: str,
    rows: Sequence[Mapping[str, Any]],
) -> bool:
    escaped = re.escape(" ".join(value.split()).strip())
    if not escaped:
        return True
    patterns = (
        rf"\b{escaped}\b\s+(?:is|as|identifies?|denotes?)\s+(?:the\s+)?"
        rf"(?:compatible system|compatibility system|merchant|seller|retailer)\b",
        rf"\b(?:compatible system|compatibility system|merchant|seller|retailer)\s*"
        rf"(?:is|:|=)\s*{escaped}\b",
        rf"\b(?:compatible with|for use with|sold by)\s+{escaped}\b",
    )
    for row in rows:
        for field in ("semantic_description", "category_reason"):
            text = " ".join(str(row.get(field) or "").split())
            if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
                return True
    return False


def _value_supported(
    value: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    evidence_fields: Sequence[str],
) -> bool:
    value_tokens = _tokens(value)
    if not value_tokens:
        return False
    for row in rows:
        for field in evidence_fields:
            if field not in _ALLOWED_EVIDENCE_FIELDS:
                continue
            field_tokens = _tokens(str(row.get(field) or ""))
            if _contains_token_sequence(field_tokens, value_tokens):
                return True
    return False


def _tokens(value: str) -> list[str]:
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", value.casefold())


def _contains_token_sequence(container: Sequence[str], candidate: Sequence[str]) -> bool:
    if not container or not candidate or len(candidate) > len(container):
        return False
    width = len(candidate)
    return any(
        list(container[index : index + width]) == list(candidate)
        for index in range(0, len(container) - width + 1)
    )


__all__ = [
    "ANSWER_FORMAT_SCHEMA_VERSION",
    "AnswerFormatterConfig",
    "AnswerFormatterPayload",
    "AnswerFormatterResult",
    "AnswerFormattingError",
    "AnswerValidationResult",
    "EvidenceBoundAnswerFormatter",
    "render_validated_answer",
    "validate_answer_formatter_result",
]
