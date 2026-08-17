"""Typed adapter around the existing post-reconciliation LLM categorizer."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from receipt_intelligence.application.ports.llm import LlmGateway
from receipt_intelligence.extraction.contracts.presentation import (
    CategorizationRequest,
    CategorizationResult,
    CategorizationStatus,
)
from receipt_intelligence.extraction.services.categorization import ReceiptCategorizationService

Categorizer = Callable[..., dict[str, Any]]


class ReceiptCategorizationAdapter(ReceiptCategorizationService):
    """Preserve the current categorizer while exposing a typed application boundary.

    Categorization runs only after deterministic validation/correction. The wrapped
    implementation is responsible only for category metadata and must never change
    receipt arithmetic, source references, payments, taxes, or validation status.
    """

    def __init__(
        self,
        *,
        llm_gateway: LlmGateway,
        ollama_url: str,
        model: str,
        num_ctx: int = 16384,
        num_predict: int = 4096,
        keep_alive: str | None = None,
        timeout_seconds: float = 180.0,
        format_json: bool = True,
        categorizer: Categorizer | None = None,
    ) -> None:
        self._llm_gateway = llm_gateway
        self._ollama_url = str(ollama_url or "").strip()
        self._model = str(model or "").strip()
        self._num_ctx = num_ctx
        self._num_predict = num_predict
        self._keep_alive = keep_alive
        self._timeout_seconds = timeout_seconds
        self._format_json = format_json
        self._categorizer = categorizer
        if not self._ollama_url or not self._model:
            raise ValueError("Categorization service requires ollama_url and model.")

    def categorize(self, request: CategorizationRequest) -> CategorizationResult:
        if not request.enabled:
            return CategorizationResult(
                status=CategorizationStatus.DISABLED,
                receipt=copy.deepcopy(request.receipt),
                warnings=("Item categorization disabled.",),
                model=self._model,
            )
        categorizer = self._categorizer or _load_existing_categorizer()
        categorization_input = _legacy_categorization_input(request.receipt)
        raw = categorizer(
            categorization_input,
            ollama_url=self._ollama_url,
            model=self._model,
            num_ctx=self._num_ctx,
            num_predict=self._num_predict,
            keep_alive=self._keep_alive,
            timeout=self._timeout_seconds,
            format_json=self._format_json,
            llm_gateway=self._llm_gateway,
        )
        status = _status(raw.get("status"))
        candidate = raw.get("receipt") if isinstance(raw.get("receipt"), dict) else request.receipt
        receipt = _category_overlay(request.receipt, candidate)
        return CategorizationResult(
            status=status,
            receipt=receipt,
            categories=tuple(
                value for value in (raw.get("categories") or []) if isinstance(value, dict)
            ),
            merchant_classification=(
                raw.get("merchant_classification")
                if isinstance(raw.get("merchant_classification"), dict)
                else {}
            ),
            warnings=tuple(str(value) for value in (raw.get("warnings") or [])),
            prompt=str(raw.get("prompt") or ""),
            raw_output=str(raw.get("raw_output") or ""),
            duration_seconds=_optional_float(raw.get("duration_seconds")),
            error=str(raw.get("error")) if raw.get("error") else None,
            model=self._model,
        )


def _legacy_categorization_input(receipt: dict[str, Any]) -> dict[str, Any]:
    """Adapt the canonical item contract to the categorizer on an isolated copy."""

    adapted = copy.deepcopy(receipt)
    items = adapted.get("items") if isinstance(adapted.get("items"), list) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name.strip():
            if not str(item.get("description") or "").strip():
                item["description"] = name
            if not str(item.get("product_description") or "").strip():
                item["product_description"] = name
        if item.get("line_total") is None and item.get("final_price") is not None:
            item["line_total"] = item["final_price"]
    return adapted


_ITEM_CATEGORY_FIELDS = frozenset(
    {
        "category_key",
        "category_group",
        "category_path",
        "category_taxonomy_version",
        "category_confidence",
        "category_confidence_raw",
        "category_confidence_calibrated",
        "category_review_required",
        "category_review_reasons",
        "category_text_certainty",
        "category_evidence_terms",
        "category_unsupported_evidence_terms",
        "category_source",
        "category_reason",
    }
)
_MERCHANT_CATEGORY_FIELDS = frozenset(
    {
        "category_key",
        "category_confidence",
        "category_reason",
        "category_source",
        "category_taxonomy_version",
    }
)


def _category_overlay(original: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Copy only category metadata onto the immutable corrected receipt shape."""

    result = copy.deepcopy(original)
    original_items = result.get("items") if isinstance(result.get("items"), list) else []
    candidate_items = candidate.get("items") if isinstance(candidate.get("items"), list) else []
    for index, item in enumerate(original_items):
        if not isinstance(item, dict) or index >= len(candidate_items):
            continue
        candidate_item = candidate_items[index]
        if not isinstance(candidate_item, dict):
            continue
        for key in _ITEM_CATEGORY_FIELDS:
            if key in candidate_item:
                item[key] = copy.deepcopy(candidate_item[key])

    original_merchant = result.get("merchant") if isinstance(result.get("merchant"), dict) else {}
    candidate_merchant = (
        candidate.get("merchant") if isinstance(candidate.get("merchant"), dict) else {}
    )
    merchant = dict(original_merchant)
    for key in _MERCHANT_CATEGORY_FIELDS:
        if key in candidate_merchant:
            merchant[key] = copy.deepcopy(candidate_merchant[key])
    if merchant or "merchant" in result:
        result["merchant"] = merchant

    if isinstance(candidate.get("categorization"), dict):
        result["categorization"] = copy.deepcopy(candidate["categorization"])
    return result


def _load_existing_categorizer() -> Categorizer:
    from receipt_intelligence.extraction.categorization.items import (
        categorize_receipt_items_llm,
    )

    return categorize_receipt_items_llm


def _status(value: Any) -> CategorizationStatus:
    text = str(value or "error").strip().lower()
    try:
        return CategorizationStatus(text)
    except ValueError:
        return CategorizationStatus.ERROR


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["ReceiptCategorizationAdapter"]
