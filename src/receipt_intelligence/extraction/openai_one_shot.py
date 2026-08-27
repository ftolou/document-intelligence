"""One-shot OpenAI receipt extraction backend.

This backend intentionally performs exactly one multimodal OpenAI Responses API
request per receipt. It bypasses the local Paddle/Qwen/Gemma extraction stages,
then rejoins the application at deterministic validation, category calibration,
final publication, review, and persistence.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from receipt_intelligence.app_version import get_app_version
from receipt_intelligence.application.llm_json import parse_json_from_llm
from receipt_intelligence.application.ports.artifacts import ArtifactKind
from receipt_intelligence.application.ports.multimodal import (
    MultimodalGateway,
    MultimodalGenerationRequest,
)
from receipt_intelligence.domain.categorization_taxonomy import (
    ITEM_TAXONOMY_VERSION,
    MERCHANT_TAXONOMY_VERSION,
    SPECIFIC_FASHION_ITEM_KEYS,
)
from receipt_intelligence.extraction.artifacts import save_json
from receipt_intelligence.extraction.categorization.items import (
    CATEGORY_TAXONOMY,
    MERCHANT_TAXONOMY,
    TAXONOMY_BY_KEY,
    VALID_MERCHANT_CATEGORY_KEYS,
    VALID_TEXT_CERTAINTY,
    calibrate_category_assignment,
    merge_categories_into_receipt,
)
from receipt_intelligence.extraction.config import ExtractionRequest
from receipt_intelligence.extraction.contracts.presentation import (
    CategorizationResult,
    CategorizationStatus,
)
from receipt_intelligence.extraction.contracts.validation import ValidationRequest
from receipt_intelligence.extraction.presentation.artifacts import (
    CompatibilityFilesystemArtifactStore,
)
from receipt_intelligence.extraction.structured.item_contract import validate_direct_items
from receipt_intelligence.extraction.validation.engine import DeterministicValidationEngine

BACKEND_NAME = "openai_one_shot"


def _taxonomy_prompt_text() -> str:
    lines = ["key|group|path|meaning"]
    for row in CATEGORY_TAXONOMY:
        lines.append(
            f"{row['key']}|{row['group']}|{row.get('path') or row['key']}|{row['description']}"
        )
    return "\n".join(lines)


def _merchant_taxonomy_prompt_text() -> str:
    lines = ["key|meaning"]
    lines.extend(f"{row['key']}|{row['description']}" for row in MERCHANT_TAXONOMY)
    return "\n".join(lines)


VALID_CATEGORY_KEYS = tuple(sorted(TAXONOMY_BY_KEY))
VALID_MERCHANT_KEYS = tuple(sorted(VALID_MERCHANT_CATEGORY_KEYS))
VALID_CERTAINTY = tuple(sorted(VALID_TEXT_CERTAINTY))

RECEIPT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "merchant": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"type": ["string", "null"]},
                "category_key": {"type": "string", "enum": list(VALID_MERCHANT_KEYS)},
                "category_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "category_reason": {"type": "string"},
                "address": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "full": {"type": ["string", "null"]},
                        "street": {"type": ["string", "null"]},
                        "postal_code": {"type": ["string", "null"]},
                        "city": {"type": ["string", "null"]},
                        "country": {"type": ["string", "null"]},
                    },
                    "required": ["full", "street", "postal_code", "city", "country"],
                },
            },
            "required": [
                "name",
                "category_key",
                "category_confidence",
                "category_reason",
                "address",
            ],
        },
        "receipt_metadata": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "date": {"type": ["string", "null"]},
                "time": {"type": ["string", "null"]},
                "currency": {"type": ["string", "null"]},
                "printed_item_count": {"type": ["integer", "null"], "minimum": 0},
                "document_identifiers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "label": {"type": "string"},
                            "normalized_type": {
                                "type": "string",
                                "enum": [
                                    "receipt_number",
                                    "payment_receipt_number",
                                    "trace_number",
                                    "transaction_number",
                                    "terminal_id",
                                    "register_number",
                                    "cashier_number",
                                    "other",
                                ],
                            },
                            "value": {"type": "string"},
                            "source_text": {"type": "string"},
                        },
                        "required": ["label", "normalized_type", "value", "source_text"],
                    },
                },
            },
            "required": ["date", "time", "currency", "printed_item_count", "document_identifiers"],
        },
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "quantity": {"type": ["number", "null"]},
                    "unit": {"type": ["string", "null"]},
                    "unit_price": {"type": ["number", "null"]},
                    "final_price": {"type": ["number", "null"]},
                    "discount_amount": {"type": ["number", "null"]},
                    "original_price": {"type": ["number", "null"]},
                    "vat_rate": {"type": ["number", "null"]},
                    "source_text": {"type": ["string", "null"]},
                    "category_key": {"type": "string", "enum": list(VALID_CATEGORY_KEYS)},
                    "category_confidence_raw": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "category_text_certainty": {"type": "string", "enum": list(VALID_CERTAINTY)},
                    "category_evidence_terms": {
                        "type": "array",
                        "maxItems": 8,
                        "items": {"type": "string"},
                    },
                    "category_reason": {"type": "string"},
                },
                "required": [
                    "name",
                    "quantity",
                    "unit",
                    "unit_price",
                    "final_price",
                    "discount_amount",
                    "original_price",
                    "vat_rate",
                    "source_text",
                    "category_key",
                    "category_confidence_raw",
                    "category_text_certainty",
                    "category_evidence_terms",
                    "category_reason",
                ],
            },
        },
        "totals": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "final_purchase_total": {"type": ["number", "null"]},
                "pre_discount_total": {"type": ["number", "null"]},
                "discount_total": {"type": ["number", "null"]},
                "net_amount": {"type": ["number", "null"]},
                "vat_amount": {"type": ["number", "null"]},
            },
            "required": [
                "final_purchase_total",
                "pre_discount_total",
                "discount_total",
                "net_amount",
                "vat_amount",
            ],
        },
        "payment": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "payment_method": {"type": ["string", "null"]},
                "payment_received": {"type": ["number", "null"]},
                "change_returned": {"type": ["number", "null"]},
            },
            "required": ["payment_method", "payment_received", "change_returned"],
        },
        "transaction_status": {"type": ["string", "null"]},
        "vat_lines": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "tax_class": {"type": ["string", "null"]},
                    "rate": {"type": ["number", "null"]},
                    "net": {"type": ["number", "null"]},
                    "tax": {"type": ["number", "null"]},
                    "gross": {"type": ["number", "null"]},
                    "source_text": {"type": ["string", "null"]},
                },
                "required": ["tax_class", "rate", "net", "tax", "gross", "source_text"],
            },
        },
        "vat_summary": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "net": {"type": ["number", "null"]},
                "tax": {"type": ["number", "null"]},
                "gross": {"type": ["number", "null"]},
                "source_text": {"type": ["string", "null"]},
            },
            "required": ["net", "tax", "gross", "source_text"],
        },
    },
    "required": [
        "merchant",
        "receipt_metadata",
        "items",
        "totals",
        "payment",
        "transaction_status",
        "vat_lines",
        "vat_summary",
    ],
}

SYSTEM_INSTRUCTIONS = f"""\
You are a high-precision receipt extraction and item-categorization engine.

Analyze the receipt IMAGE directly and produce one structured result. Extraction and
categorization happen in the SAME response, but they have different evidence rules.

SOURCE AUTHORITY FOR EXTRACTION
- The image is the only source of document facts.
- Extract only values that are visible in the receipt.
- If a value cannot be read reliably, return null.
- Do not invent missing values from retail conventions or geographic knowledge.
- Do not repair an unreadable value merely because arithmetic suggests what it should be.
- Arithmetic may help distinguish plausible visual interpretations, but it is not source evidence.
- Do not set a missing monetary field to 0 merely because zero would be plausible.

MERCHANT / METADATA EXTRACTION
- country must be null unless the country itself is explicitly visible on the receipt.
- Extract document identifiers as printed label/value pairs into document_identifiers.
- Preserve the printed label in label and the shortest useful source evidence in source_text.
- normalized_type is only a normalization of the visible label:
  * receipt_number: labels such as Bon-Nr., receipt no., receipt number
  * payment_receipt_number: labels such as Beleg-Nr. when they identify the payment receipt
  * trace_number: labels such as Trace-Nr.
  * transaction_number: explicit transaction-number labels
  * terminal_id: explicit terminal identifiers
  * register_number: explicit till/register identifiers
  * cashier_number: explicit cashier/operator identifiers
  * other: any other visible document identifier
- If several identifiers are printed, extract all relevant identifiers. Do not silently choose
  one identifier as the canonical receipt number.
- printed_item_count is only an explicitly printed total number of purchased articles/items.
  If no such aggregate count is printed, return null. Do not derive it from extracted lines.

ITEM EXTRACTION
- Extract purchased line items only.
- Do not classify subtotal, total, VAT/tax summary, cash/card tender, amount paid, change,
  loyalty balance, receipt metadata, or informational lines as purchased items.
- final_price is the amount that the item contributes to the purchase after item-level discounts
  when the receipt makes that relationship clear.
- original_price is the pre-discount line/item amount only when explicitly supported.
- discount_amount is positive when it represents a reduction.
- quantity and unit_price must be null unless visible or directly supported by the item lines.
- Preserve abbreviated product names rather than expanding them from general knowledge.
- Include deposits, bags, or other charged purchasable lines when they contribute to the total.
- When printed tax-class markers such as A/B are visibly mapped to rates in the tax table,
  vat_rate may use that visible mapping.

TOTALS / PAYMENT
- final_purchase_total is the amount owed for the purchase, not cash received or card tender.
- payment_received is the tendered/charged amount when explicitly shown.
- change_returned is change explicitly printed as returned to the customer.
- For a card transaction with no printed change value, change_returned must be null, not 0.
- net_amount and vat_amount should come from printed tax information when available.
- pre_discount_total and discount_total are only explicitly printed aggregate values; do not
  derive them from item arithmetic.

VAT
- vat_lines contains ONLY independent rate-specific VAT/tax rows.
- Each populated vat_lines value must be supported by that same printed rate-specific row.
- Never place an aggregate/total VAT row such as Gesamtbetrag, Total, Summe, or equivalent
  into vat_lines.
- Preserve a printed tax-class marker (for example A or B) in tax_class when visible.
- Do not create VAT rows that are not printed or clearly represented on the receipt.
- vat_summary is ONLY the printed aggregate VAT/tax total row when one exists.
- If no aggregate VAT/tax total row is printed, return null for all vat_summary values.
- Do not infer vat_summary by summing vat_lines.

CATEGORIZATION BOUNDARY
- Categorization is semantic metadata. It MUST NOT change item names, quantities, prices,
  discounts, VAT, totals, item order, or any other extracted document fact.
- First classify the merchant. Use merchant type only as context for item categorization,
  never as proof that every item belongs to the merchant's primary vertical.
- Choose category_key only from the item taxonomy below.
- category_confidence_raw is confidence before deterministic calibration, from 0.0 to 1.0.
- category_text_certainty must be one of explicit, contextual, incomplete_or_unfamiliar, ambiguous.
- category_evidence_terms may contain at most 8 short exact visible terms. Never put inferred
  expansions or synonyms in evidence_terms.
- For incomplete, unfamiliar, code-like, or visibly truncated item text, use a broad category or
  unknown, set text certainty appropriately, and keep confidence <= 0.60.
- Confidence > 0.80 requires explicit item evidence or strong merchant-and-peer context.
- Pfand/LEERGUT positive charges are deposit_pfand. Negative returns are deposit_refund.
- A purchased product is not discount_coupon merely because a promotion note is nearby.
- A fashion merchant alone does not justify a specific fashion subtype. Use fashion_unknown when
  merchant/peer context supports fashion but item text does not explicitly support the subtype.

MERCHANT TAXONOMY
{_merchant_taxonomy_prompt_text()}

ITEM TAXONOMY
{_taxonomy_prompt_text()}

NORMALIZATION
- Monetary values are JSON numbers using a dot as decimal separator.
- Currency should be an ISO-style code such as EUR when clearly identifiable.
- Normalize an unambiguous date to YYYY-MM-DD and time to HH:MM or HH:MM:SS.
- source_text should contain the shortest useful visible receipt text supporting the value.

Return only the schema-constrained structured result.
"""

USER_PROMPT = """\
Extract and categorize the complete receipt in one pass from this image.

Pay special attention to:
1. correct item boundaries,
2. quantities and unit prices,
3. item-level discounts,
4. the distinction between total and payment/tender values,
5. rate-specific VAT rows versus an aggregate VAT summary row,
6. multiline product descriptions,
7. all visible document identifiers without forcing one ambiguous canonical receipt number,
8. an explicitly printed aggregate article/item count when present,
9. a taxonomy-constrained category for every purchased item, with concise reason and evidence terms,
10. keeping semantic categorization strictly separate from extracted financial/document facts.

Do not omit an item simply because its layout is unusual.
Do not invent rows or values merely to make arithmetic reconcile.
"""


def _emit(config: ExtractionRequest, stage: str, status: str, message: str, **details: Any) -> None:
    callback = config.progress_callback
    if callback is None:
        return
    callback(
        {
            "stage": stage,
            "status": status,
            "message": message,
            "details": details,
        }
    )


def _usage_to_dict(response: dict[str, Any]) -> dict[str, Any] | None:
    usage = response.get("usage")
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return dict(usage)
    return {"value": str(usage)}


def _call_openai_once(
    config: ExtractionRequest,
    *,
    gateway: MultimodalGateway,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    started = time.perf_counter()
    result = gateway.generate(
        MultimodalGenerationRequest(
            model=config.openai_model,
            system_prompt=SYSTEM_INSTRUCTIONS,
            prompt=USER_PROMPT,
            image_paths=(config.source_image_path,),
            operation="receipt_one_shot_extraction_and_categorization",
            num_predict=config.openai_max_output_tokens,
            temperature=None,
            timeout_seconds=config.openai_timeout_seconds,
            format_json=True,
            response_json_schema=RECEIPT_SCHEMA,
        )
    )
    elapsed_seconds = time.perf_counter() - started
    payload = parse_json_from_llm(result, response_json_schema=RECEIPT_SCHEMA)
    response = dict(result.raw_response or {"output_text": result.text})
    return payload, response, elapsed_seconds


def _money_payload(field: str, value: Any, currency: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    payload: dict[str, Any] = {field: value}
    if currency:
        payload["currency"] = currency
    return payload


def _canonical_receipt_from_model(model_receipt: dict[str, Any]) -> dict[str, Any]:
    merchant = (
        model_receipt.get("merchant") if isinstance(model_receipt.get("merchant"), dict) else {}
    )
    metadata = (
        model_receipt.get("receipt_metadata")
        if isinstance(model_receipt.get("receipt_metadata"), dict)
        else {}
    )
    totals = model_receipt.get("totals") if isinstance(model_receipt.get("totals"), dict) else {}
    payment = model_receipt.get("payment") if isinstance(model_receipt.get("payment"), dict) else {}
    currency = metadata.get("currency") if isinstance(metadata.get("currency"), str) else None
    identifiers = metadata.get("document_identifiers")
    identifiers = identifiers if isinstance(identifiers, list) else []

    receipt_number = None
    for normalized_type in ("receipt_number", "payment_receipt_number", "transaction_number"):
        match = next(
            (
                value
                for value in identifiers
                if isinstance(value, dict)
                and value.get("normalized_type") == normalized_type
                and str(value.get("value") or "").strip()
            ),
            None,
        )
        if match is not None:
            receipt_number = str(match["value"])
            break

    canonical_items: list[dict[str, Any]] = []
    raw_items = model_receipt.get("items") if isinstance(model_receipt.get("items"), list) else []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        canonical_items.append(
            {
                "name": name,
                "description": name,
                "product_description": name,
                "quantity": raw.get("quantity"),
                "unit": raw.get("unit"),
                "unit_price": raw.get("unit_price"),
                "final_price": raw.get("final_price"),
                "line_total": raw.get("final_price"),
                "discount_amount": raw.get("discount_amount"),
                "original_price": raw.get("original_price"),
                "vat_rate": raw.get("vat_rate"),
                "tax_rate": raw.get("vat_rate"),
                "source_text": raw.get("source_text"),
                "raw_description": raw.get("source_text") or name,
            }
        )

    vat_lines: list[dict[str, Any]] = []
    raw_vat_lines = model_receipt.get("vat_lines")
    if isinstance(raw_vat_lines, list):
        for line in raw_vat_lines:
            if not isinstance(line, dict):
                continue
            vat_lines.append(
                {
                    "tax_class": line.get("tax_class"),
                    "rate_percent": line.get("rate"),
                    "net_amount": line.get("net"),
                    "vat_amount": line.get("tax"),
                    "gross_amount": line.get("gross"),
                    "source_text": line.get("source_text"),
                }
            )

    return {
        "merchant": {
            "name": merchant.get("name"),
            "address": merchant.get("address") if isinstance(merchant.get("address"), dict) else {},
        },
        "receipt_metadata": {
            "date": metadata.get("date"),
            "time": metadata.get("time"),
            "receipt_number": receipt_number,
            "currency": currency,
            "printed_item_count": metadata.get("printed_item_count"),
            "document_identifiers": [
                dict(value) for value in identifiers if isinstance(value, dict)
            ],
        },
        "items": canonical_items,
        "totals": {
            "final_purchase_total": _money_payload(
                "final_purchase_total", totals.get("final_purchase_total"), currency
            ),
            "pre_discount_total": _money_payload(
                "pre_discount_total", totals.get("pre_discount_total"), currency
            ),
            "net_amount": _money_payload("net_amount", totals.get("net_amount"), currency),
        },
        "discount": {
            "discount_total": _money_payload(
                "discount_total", totals.get("discount_total"), currency
            )
        },
        "payment": {
            "payment_method": payment.get("payment_method"),
            "payment_received": _money_payload(
                "payment_received", payment.get("payment_received"), currency
            ),
            "change_returned": _money_payload(
                "change_returned", payment.get("change_returned"), currency
            ),
        },
        "transaction_status": model_receipt.get("transaction_status"),
        "tax": {
            "vat_amount": _money_payload("vat_amount", totals.get("vat_amount"), currency),
            "vat_lines": vat_lines,
            "vat_summary": (
                dict(model_receipt["vat_summary"])
                if isinstance(model_receipt.get("vat_summary"), dict)
                else None
            ),
        },
    }


def _normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(1.0, confidence)), 3)


def _category_result(
    model_receipt: dict[str, Any],
    canonical_receipt: dict[str, Any],
    *,
    model: str,
    enabled: bool,
    elapsed_seconds: float,
) -> CategorizationResult:
    if not enabled:
        return CategorizationResult(
            status=CategorizationStatus.DISABLED,
            receipt=canonical_receipt,
            duration_seconds=elapsed_seconds,
            model=model,
        )

    raw_merchant = (
        model_receipt.get("merchant") if isinstance(model_receipt.get("merchant"), dict) else {}
    )
    merchant_key = str(raw_merchant.get("category_key") or "unknown").strip()
    if merchant_key not in VALID_MERCHANT_CATEGORY_KEYS:
        merchant_key = "unknown"
    merchant_classification = {
        "category_key": merchant_key,
        "confidence": _normalize_confidence(raw_merchant.get("category_confidence")),
        "reason": str(raw_merchant.get("category_reason") or "").strip()[:300],
        "source": BACKEND_NAME,
        "taxonomy_version": MERCHANT_TAXONOMY_VERSION,
    }

    raw_items = model_receipt.get("items") if isinstance(model_receipt.get("items"), list) else []
    canonical_items = (
        canonical_receipt.get("items") if isinstance(canonical_receipt.get("items"), list) else []
    )
    merchant_context = " ".join(
        str(value or "")
        for value in (
            raw_merchant.get("name"),
            (raw_merchant.get("address") or {}).get("full")
            if isinstance(raw_merchant.get("address"), dict)
            else None,
        )
    )
    categories: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict) or index >= len(canonical_items):
            continue
        key = str(raw.get("category_key") or "unknown").strip()
        if key not in TAXONOMY_BY_KEY:
            warnings.append(
                f"Unknown OpenAI category_key {key!r} for item_index={index}; using unknown."
            )
            key = "unknown"
        certainty = str(raw.get("category_text_certainty") or "contextual").strip().lower()
        if key in SPECIFIC_FASHION_ITEM_KEYS and certainty != "explicit":
            warnings.append(
                f"Downgraded contextual fashion subtype {key!r} to fashion_unknown "
                f"for item_index={index}."
            )
            key = "fashion_unknown"
        calibration = calibrate_category_assignment(
            item=canonical_items[index] if isinstance(canonical_items[index], dict) else {},
            category_key=key,
            confidence=_normalize_confidence(raw.get("category_confidence_raw")),
            reason=raw.get("category_reason"),
            source=BACKEND_NAME,
            text_certainty=raw.get("category_text_certainty"),
            evidence_terms=(
                raw.get("category_evidence_terms")
                if isinstance(raw.get("category_evidence_terms"), list)
                else []
            ),
            context_text=merchant_context,
        )
        taxonomy = TAXONOMY_BY_KEY[key]
        categories.append(
            {
                "item_index": index,
                "category_key": key,
                "category_group": taxonomy["group"],
                "category_path": str(taxonomy.get("path") or key),
                "category_taxonomy_version": ITEM_TAXONOMY_VERSION,
                **calibration,
                "category_reason": str(raw.get("category_reason") or "").strip()[:300],
                "category_source": BACKEND_NAME,
            }
        )

    merged = merge_categories_into_receipt(
        canonical_receipt,
        categories,
        status="completed",
        merchant_classification=merchant_classification,
        warnings=warnings,
        duration_seconds=elapsed_seconds,
        model=model,
    )
    return CategorizationResult(
        status=CategorizationStatus.OK_WITH_WARNINGS if warnings else CategorizationStatus.OK,
        receipt=merged,
        categories=tuple(categories),
        merchant_classification=merchant_classification,
        warnings=tuple(warnings),
        prompt=f"{SYSTEM_INSTRUCTIONS}\n\n{USER_PROMPT}",
        raw_output=json.dumps(model_receipt, ensure_ascii=False, indent=2),
        duration_seconds=elapsed_seconds,
        model=model,
    )


def _stage(stage: str, status: str, started: float, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "stage": stage,
        "status": status,
        "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    payload.update(extra)
    return payload


def _finalize_openai(
    request: ExtractionRequest,
    *,
    canonical_receipt: dict[str, Any],
    validation: Any,
    categorization: CategorizationResult,
    stage_trace: list[dict[str, Any]],
    api_elapsed: float,
    response: Any,
) -> dict[str, Any]:
    store = CompatibilityFilesystemArtifactStore(request.result_dir)
    store.prepare_run(run_id=request.run_id, overwrite=True)

    final_receipt = dict(categorization.receipt or canonical_receipt)
    final_receipt["validation"] = validation.to_dict()
    pipeline_meta = {
        "schema_version": "receipt_pipeline_meta_1",
        "app_version": get_app_version(),
        "architecture": (
            "OpenAI one-shot multimodal structured extraction + categorization -> "
            "read-only deterministic validation -> deterministic category calibration -> "
            "final publication"
        ),
        "backend": BACKEND_NAME,
        "provider": "openai",
        "model": request.openai_model,
        "reasoning_effort": request.openai_reasoning_effort,
        "image_detail": request.openai_image_detail,
        "api_calls": 1,
        "workflow": {
            "name": "OpenAIOneShotReceiptWorkflow",
            "staged_execution": False,
            "stage_count": len(stage_trace),
            "stages": [str(value.get("stage") or "") for value in stage_trace],
            "stage_trace": [dict(value) for value in stage_trace],
        },
        "safety": {
            "read_only_validation": True,
            "no_deterministic_semantic_parser": True,
            "no_generic_correction_fallback": True,
            "category_calibration_changes_document_facts": False,
        },
        "validation": {
            "status": validation.status,
            "failed_codes": sorted(validation.failed_codes),
            "failure_count": len(validation.failed_codes),
            "money_tolerance": 0.0,
            "vat_rate_tolerance": 0.02,
        },
        "categorization": {
            "status": categorization.status.value,
            "model": request.openai_model,
            "same_model_call_as_extraction": True,
            "warning_count": len(categorization.warnings),
        },
        "openai": {
            "elapsed_seconds": api_elapsed,
            "usage": _usage_to_dict(response),
            "response_id": getattr(response, "id", None),
        },
    }
    final_receipt["pipeline"] = {
        "architecture": pipeline_meta["architecture"],
        "app_version": get_app_version(),
        "workflow": "OpenAIOneShotReceiptWorkflow",
        "staged_execution": False,
        "read_only_validation": True,
        "no_generic_correction_fallback": True,
        "categorization_status": categorization.status.value,
        "backend": BACKEND_NAME,
        "provider": "openai",
        "model": request.openai_model,
        "reasoning_effort": request.openai_reasoning_effort,
        "image_detail": request.openai_image_detail,
        "api_calls": 1,
    }

    references = []
    if categorization.status is not CategorizationStatus.DISABLED:
        references.extend(
            [
                store.write_text(
                    run_id=request.run_id,
                    kind=ArtifactKind.CATEGORIZATION_PROMPT,
                    text=categorization.prompt,
                ),
                store.write_text(
                    run_id=request.run_id,
                    kind=ArtifactKind.CATEGORIZATION_RAW,
                    text=categorization.raw_output,
                ),
                store.write_json(
                    run_id=request.run_id,
                    kind=ArtifactKind.CATEGORIZATION_RESULT,
                    payload=categorization.to_dict(include_model_io=False),
                ),
            ]
        )
    references.extend(
        [
            store.write_json(
                run_id=request.run_id,
                kind=ArtifactKind.FINAL_VALIDATION,
                payload=validation.to_dict(),
            ),
            store.write_json(
                run_id=request.run_id,
                kind=ArtifactKind.RECONCILIATION_REPORT,
                payload=validation.to_dict(),
            ),
            store.write_json(
                run_id=request.run_id,
                kind=ArtifactKind.FINAL_RECEIPT,
                payload=final_receipt,
            ),
            store.write_json(
                run_id=request.run_id,
                kind=ArtifactKind.FINAL_RECEIPT_RECONCILED,
                payload=final_receipt,
            ),
            store.write_json(
                run_id=request.run_id,
                kind=ArtifactKind.FINAL_RECEIPT_CATEGORIZED,
                payload=final_receipt,
            ),
            store.write_json(
                run_id=request.run_id,
                kind=ArtifactKind.PIPELINE_METADATA,
                payload=pipeline_meta,
            ),
            store.write_json(
                run_id=request.run_id,
                kind=ArtifactKind.STAGE_TRACE,
                payload=stage_trace,
            ),
        ]
    )
    aliases = store.publish_aliases(
        run_id=request.run_id,
        kinds=tuple(reference.kind for reference in references),
    )
    paths: dict[str, Path] = {}
    for reference in tuple(references) + aliases:
        key = store.path_key(reference.kind)
        if reference.path.name.startswith("latest_"):
            key = f"latest_{key}"
        paths[key] = reference.path

    return {
        "receipt": final_receipt,
        "report": validation.to_dict(),
        "paths": paths,
        "logs": [],
        "pipeline_meta": pipeline_meta,
        "observability": {"stage_trace": stage_trace, "metrics_path": None},
    }


def run_openai_one_shot_extraction(
    request: ExtractionRequest,
    *,
    gateway: MultimodalGateway,
) -> dict[str, Any]:
    """Run the cloud one-shot backend and return the canonical application result."""

    request.result_dir.mkdir(parents=True, exist_ok=True)
    stage_trace: list[dict[str, Any]] = []

    started = time.perf_counter()
    _emit(request, "prepare", "running", "Preparing OpenAI one-shot extraction.")
    if not request.source_image_path.is_file():
        raise FileNotFoundError(f"Receipt image does not exist: {request.source_image_path}")
    stage_trace.append(_stage("prepare", "done", started))
    _emit(request, "prepare", "done", "Receipt image ready for OpenAI.")

    started = time.perf_counter()
    _emit(
        request,
        "openai_one_shot",
        "running",
        "Extracting receipt and categories in one OpenAI multimodal request.",
        model=request.openai_model,
        reasoning_effort=request.openai_reasoning_effort,
        image_detail=request.openai_image_detail,
    )
    model_receipt, response, api_elapsed = _call_openai_once(request, gateway=gateway)
    stage_trace.append(
        _stage(
            "openai_one_shot",
            "done",
            started,
            model=request.openai_model,
            api_calls=1,
        )
    )
    _emit(
        request,
        "openai_one_shot",
        "done",
        "OpenAI one-shot extraction completed.",
        model=request.openai_model,
        duration_seconds=round(api_elapsed, 3),
    )

    raw_path = request.result_dir / f"{request.run_id}_openai_model_raw.json"
    response_path = request.result_dir / f"{request.run_id}_openai_api_response.json"
    run_metadata_path = request.result_dir / f"{request.run_id}_openai_run_metadata.json"
    save_json(raw_path, model_receipt)
    save_json(response_path, response)
    save_json(
        run_metadata_path,
        {
            "backend": BACKEND_NAME,
            "provider": "openai",
            "model": request.openai_model,
            "reasoning_effort": request.openai_reasoning_effort,
            "image_detail": request.openai_image_detail,
            "api_calls": 1,
            "elapsed_seconds": api_elapsed,
            "usage": _usage_to_dict(response),
        },
    )

    canonical_receipt = _canonical_receipt_from_model(model_receipt)

    started = time.perf_counter()
    _emit(request, "validation", "running", "Running deterministic receipt validation.")
    item_contract = validate_direct_items({"items": canonical_receipt.get("items") or []})
    validation = DeterministicValidationEngine().validate(
        ValidationRequest(
            receipt=canonical_receipt,
            item_contract=item_contract,
            item_pipeline_enabled=True,
            selected_scalar_tasks=(BACKEND_NAME,),
            # Direct printed-money identities are exact to cents. The VAT-rate
            # derivation retains the app's rounding-aware tolerance.
            money_tolerance=0.0,
            vat_rate_tolerance=0.02,
        )
    )
    stage_trace.append(_stage("validation", "done", started, validation_status=validation.status))
    _emit(
        request,
        "validation",
        "done",
        "Deterministic validation completed.",
        validation_status=validation.status,
        failed_codes=sorted(validation.failed_codes),
    )

    started = time.perf_counter()
    _emit(
        request,
        "categorization",
        "running" if request.categorization_enabled else "skipped",
        "Calibrating categories from the same OpenAI response without changing receipt facts.",
    )
    categorization = _category_result(
        model_receipt,
        canonical_receipt,
        model=request.openai_model,
        enabled=request.categorization_enabled,
        elapsed_seconds=api_elapsed,
    )
    stage_trace.append(
        _stage("categorization", "done", started, categorization_status=categorization.status.value)
    )
    _emit(
        request,
        "categorization",
        "done",
        "Deterministic category calibration completed.",
        categorization_status=categorization.status.value,
    )

    started = time.perf_counter()
    _emit(request, "finalize", "running", "Publishing canonical OpenAI receipt artifacts.")
    # Finalization is deterministic; include it in the persisted trace before writing artifacts.
    final_trace = [*stage_trace, {"stage": "finalize", "status": "done"}]
    result = _finalize_openai(
        request,
        canonical_receipt=canonical_receipt,
        validation=validation,
        categorization=categorization,
        stage_trace=final_trace,
        api_elapsed=api_elapsed,
        response=response,
    )
    stage_trace.append(_stage("finalize", "done", started))
    _emit(request, "finalize", "done", "OpenAI receipt artifacts published.")

    result_paths = dict(result.get("paths") or {})
    result_paths.update(
        {
            "openai_model_raw": raw_path,
            "openai_api_response": response_path,
            "openai_run_metadata": run_metadata_path,
        }
    )
    result["paths"] = result_paths
    result["observability"] = {"stage_trace": stage_trace, "metrics_path": None}
    return result


__all__ = ["BACKEND_NAME", "run_openai_one_shot_extraction"]
