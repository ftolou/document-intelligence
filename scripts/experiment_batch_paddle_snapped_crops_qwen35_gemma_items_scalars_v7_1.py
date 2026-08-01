#!/usr/bin/env python3
"""
Experiment: PaddleOCR geometry -> aspect-ratio-adaptive safe crops -> parallel
Qwen3.5 transcription -> direct Gemma item extraction plus scalar specialists.

Primary path
------------
1. Detect text boxes with PaddleOCR and cluster them into approximate receipt rows.
2. Choose the target crop count from detected-row density and the image H/W aspect
   ratio, capped by --crops.
3. Snap nominal internal boundaries to Paddle-proposed, pixel-verified whitespace.
4. When safe multi-crop boundaries cannot be found, transcribe the whole image with
   Qwen instead of stopping the receipt.
5. Accept the first nonempty Qwen response for each crop without post-transcription
   protocol, line-count, duplication, or semantic validation.
6. If any planned crop call fails, discard partial crop output and retry Qwen once on
   the whole image path (subject to the configured transport retries).
7. Concatenate successful crop transcriptions in geometric order, assign global
   R0001... row IDs, and pass the result directly to Gemma.
8. Run direct item extraction and parallel scalar specialists, including discount,
   then derive accepted/review_required semantic status.

Python performs detector-box clustering, pixel-level boundary selection, crop planning,
ordered concatenation, artifact storage, and JSON contract validation for Gemma output.
It does not validate, repair, align, or semantically judge Qwen transcription text.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import inspect
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageOps

SCHEMA_VERSION = "paddle_snapped_crops_qwen35_gemma_items_scalars_batch.v7.1"

DEFAULT_BATCH_INPUT = Path("/app/var/batch_input")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}


QWEN_TRANSCRIPTION_PROMPT_TEMPLATE = """
Transcribe every visible physical receipt row in this image.

PaddleOCR estimated approximately {estimated_count} row band(s) in this image region.
That number is diagnostic context only. Do not force your output to match it.

Rules:

1. Preserve visible text, numbers, punctuation, decimal separators, currency symbols,
   article numbers, identifiers, prices, quantities, and truncated text.
2. Output one physical receipt row per output line in strict top-to-bottom order.
3. Do not omit a visible row.
4. Do not combine text from different vertical positions.
5. Do not split one physical row into multiple output lines.
6. Keep horizontally aligned cells from the same physical row on the same line.
7. Do not add row numbers, JSON, Markdown, bullets, labels, code fences, or explanations.
8. Do not calculate, interpret, normalize, translate, correct, or complete text.
9. Use [unclear] only for unreadable text.
10. Do not output blank lines.
11. Return only the transcription lines.
""".strip()


GEMMA_SYSTEM_PROMPT = """
You are a receipt interpreter answering one narrowly defined semantic question.

Rules:
- Answer only the requested field or structure.
- Use only the supplied receipt evidence.
- Do not silently repair OCR text.
- Do not invent missing values.
- Do not choose a value merely because it makes the receipt balance.
- Keep item price, receipt total, payment, change, discount, net amount, and VAT
  semantically distinct.
- Return null when the requested value is not supported.
- Return only JSON matching the supplied schema.
""".strip()


ROW_ROLES = (
    "purchased_item",
    "item_component",
    "item_detail",
    "item_price_continuation",
    "article_id",
    "item_discount",
    "receipt_discount",
    "subtotal_or_total",
    "payment",
    "change",
    "vat",
    "merchant_or_header",
    "receipt_metadata",
    "footer_or_advertisement",
    "other",
)

ITEM_RELATED_ROLES = {
    "purchased_item",
    "item_component",
    "item_detail",
    "item_price_continuation",
    "article_id",
    "item_discount",
}


def nullable(kind: str) -> dict[str, Any]:
    return {"type": [kind, "null"]}


def named_text_schema(field_name: str) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            field_name: nullable("string"),
        },
        "required": [field_name],
    }


def named_money_schema(field_name: str) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            field_name: nullable("number"),
            "currency": nullable("string"),
        },
        "required": [field_name, "currency"],
    }


MERCHANT_ADDRESS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "street": nullable("string"),
        "postal_code": nullable("string"),
        "city": nullable("string"),
        "country": nullable("string"),
    },
    "required": ["street", "postal_code", "city", "country"],
}


TRANSACTION_STATUS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "transaction_status": {
            "type": "string",
            "enum": [
                "completed",
                "cancelled",
                "refunded",
                "not_clear",
            ],
        },
    },
    "required": ["transaction_status"],
}


VAT_LINES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "vat_lines": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "source_rows": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "rate_percent": nullable("number"),
                    "net_amount": nullable("number"),
                    "vat_amount": nullable("number"),
                },
                "required": [
                    "source_rows",
                    "rate_percent",
                    "net_amount",
                    "vat_amount",
                ],
            },
        },
    },
    "required": ["vat_lines"],
}


ROW_ROLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "row_id": {"type": "string"},
        "row_role": {
            "type": "string",
            "enum": list(ROW_ROLES),
        },
    },
    "required": ["row_id", "row_role"],
}


ITEM_BLOCKS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "item_blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "block_id": {"type": "string"},
                    "source_rows": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["block_id", "source_rows"],
            },
        },
    },
    "required": ["item_blocks"],
}


ITEM_NAME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "item_name": nullable("string"),
    },
    "required": ["item_name"],
}


ITEM_PRICE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "item_price": nullable("number"),
    },
    "required": ["item_price"],
}


DIRECT_ITEMS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "final_price": nullable("number"),
                    "quantity": nullable("number"),
                    "unit": nullable("string"),
                    "discount_amount": nullable("number"),
                    "original_price": nullable("number"),
                },
                "required": [
                    "name",
                    "final_price",
                    "quantity",
                    "unit",
                    "discount_amount",
                    "original_price",
                ],
            },
        },
    },
    "required": ["items"],
}


DIRECT_ITEMS_QUESTION = """
Extract every top-level separately purchased or separately charged item from the
complete receipt transcription.

For each item return:
- name
- final_price
- quantity
- unit
- discount_amount
- original_price


Rules:

1. A named product or service row identifies a candidate purchased item.
2. A monetary amount on a named product row is only a price candidate. 
Do not assign final_price until all contiguous rows belonging to that item have been examined.
3. Letters such as A, B, E, F, O, or V following an amount may be VAT category
   signs and are not part of the price or currency.
4. When the named row has no clear final amount and an immediately following
   quantity, weight, or calculation row belongs to it, use the final amount from
   that continuation row as final_price.
5. A quantity, weight, or "N x unit-price" row immediately following a named
   product and before the next named product MUST be attached to that preceding
   product.
6. Do not create a separate item from a quantity, weight, unit-price, article-ID,
   discount, or other continuation row.
7. For a row shaped as "N x unit-price final" or "N * unit-price final":
   - quantity = N;
   - final_price = final.
8. For "weight unit x price/unit final", set quantity to the printed weight,
   unit to the printed unit, and final_price to final.
9. When quantity multiplied by unit price equals the item's final line price
   within normal currency rounding, this confirms the relationship.
10. Never use the unit price as final_price when a separate final amount exists.
11. Included menu or bundle components without a separate charge are not
    separate items.
12. Separately charged deposits, bags, and services are items.
13. Do not create items from totals, VAT, payment, change, headers, metadata,
    receipt-wide discounts, or footer rows.
14. When no explicit quantity is printed, return null. Never assume quantity 1.
15. Return null for a missing unit. Never return an empty string.
16. Preserve the printed OCR product text. Do not silently correct or translate
    product names.
17. Return numeric JSON values using a decimal point.
18. Return only JSON matching the supplied schema.
19. An item block starts with a named product or service and continues through related quantity, 
identifier, variant, size, unit-price, price-adjustment, discount and explanatory rows. 
Intervening non-product detail rows do not end the block. 
The block ends at the next named purchased item or at an unambiguous receipt-level total, 
payment, tax or footer section.
20. When an item block contains multiple amounts, classify all of them before producing the item. 
Use signs, percentages, row order, semantic labels and arithmetic relationships to distinguish original price, 
discount or surcharge, unit price and final charged price. 
A later supported effective amount may supersede an earlier price candidate.
""".strip()


SCALAR_TASK_ORDER = (
    "merchant_name",
    "merchant_address",
    "receipt_date",
    "receipt_time",
    "receipt_number",
    "currency",
    "final_purchase_total",
    "pre_discount_total",
    "discount_total",
    "payment_method",
    "payment_received",
    "change_returned",
    "transaction_status",
    "net_amount",
    "vat_amount",
    "vat_lines",
)


DEFAULT_BATCH_SCALAR_TASKS = (
    "merchant_name",
    "merchant_address",
    "currency",
    "final_purchase_total",
    "discount_total",
    "vat_amount",
    "vat_lines",
)


SCALAR_TASKS: dict[str, dict[str, Any]] = {
    "merchant_name": {
        "schema": named_text_schema("merchant_name"),
        "num_predict": 96,
        "question": """
What is the name of the business that sold the purchased items?

Do not return a shopping centre, slogan, customer name, payment provider, or
legal footer company when a clear store brand is printed.

Return only merchant_name.
""".strip(),
    },
    "merchant_address": {
        "schema": MERCHANT_ADDRESS_SCHEMA,
        "num_predict": 192,
        "question": """
What is the postal address of the selling business?

Return only street, postal_code, city, and country.
Do not return a customer address or payment-provider address.
Use null for fields that are not printed.
""".strip(),
    },
    "receipt_date": {
        "schema": named_text_schema("receipt_date"),
        "num_predict": 64,
        "question": """
What is the printed receipt transaction date?

Do not infer or normalize a missing date.
Return only receipt_date.
""".strip(),
    },
    "receipt_time": {
        "schema": named_text_schema("receipt_time"),
        "num_predict": 64,
        "question": """
What is the printed receipt transaction time?

Do not infer a missing time.
Return only receipt_time.
""".strip(),
    },
    "receipt_number": {
        "schema": named_text_schema("receipt_number"),
        "num_predict": 96,
        "question": """
What is the receipt, transaction, order, or document number?

Do not return a customer-card number, tax ID, telephone number, article number,
or register number unless it is explicitly the receipt or order number.

Return only receipt_number.
""".strip(),
    },
    "currency": {
        "schema": named_text_schema("currency"),
        "num_predict": 48,
        "question": """
What currency is used for the purchase amounts?

Return a short currency code such as EUR when clearly supported.
Return only currency.
""".strip(),
    },
    "final_purchase_total": {
        "schema": named_money_schema("final_purchase_total"),
        "num_predict": 128,
        "question": """
What was the final gross amount charged for the purchased items after all
discounts and including VAT?

Do not return:
- gross item value before discounts;
- an intermediate total after only some discounts;
- net amount before VAT;
- VAT amount;
- amount tendered or payment received;
- or change.

A higher amount may be cash tendered or payment received.
A lower amount labelled Rückgeld, change, Wechselgeld, or similar is change.
Do not assume the largest amount is the answer.
Prefer an explicitly labelled final purchase total.

Return only final_purchase_total and currency.
""".strip(),
    },
    "pre_discount_total": {
        "schema": named_money_schema("pre_discount_total"),
        "num_predict": 128,
        "question": """
What explicit gross purchase total was printed before all discounts?

This amount includes VAT and precedes all price discounts.
Do not return the final payable amount, net amount, VAT amount, payment
received, or change.
Return null when no explicit pre-discount gross total is printed.

Return only pre_discount_total and currency.
""".strip(),
    },
    "discount_total": {
        "schema": named_money_schema("discount_total"),
        "num_predict": 128,
        "question": """
What explicit total price discount was applied to this purchase?

Prefer a printed aggregate such as Rabatt Gesamt.
Do not return a subtotal, final payable amount, payment voucher, payment
received, change, net amount, or VAT.
Do not calculate an unprinted discount total.

Return the positive discount magnitude.
Return only discount_total and currency.
""".strip(),
    },
    "payment_method": {
        "schema": named_text_schema("payment_method"),
        "num_predict": 64,
        "question": """
What payment method was used?

Examples include cash, credit card, debit card, EC card, voucher, or mixed.
Return null when it is not printed.
Return only payment_method.
""".strip(),
    },
    "payment_received": {
        "schema": named_money_schema("payment_received"),
        "num_predict": 128,
        "question": """
How much money did the customer provide or tender as payment?

This may be higher than the purchase total when cash was provided and change
was returned. For card payment it is often equal to the purchase total.

Do not return change, discount, net amount, or VAT.
Return only payment_received and currency.
""".strip(),
    },
    "change_returned": {
        "schema": named_money_schema("change_returned"),
        "num_predict": 128,
        "question": """
How much change was returned to the customer?

Look for Rückgeld, Rueckgeld, Wechselgeld, change, rendu, or an equivalent
label. Return the positive amount returned even when the receipt prints a
bookkeeping minus sign.

Do not return payment received, purchase total, discount, net amount, or VAT.
Return only change_returned and currency.
""".strip(),
    },
    "transaction_status": {
        "schema": TRANSACTION_STATUS_SCHEMA,
        "num_predict": 48,
        "question": """
Was the transaction completed, cancelled, or refunded?

Use explicit text such as storniert, cancelled, void, annulé, refund, retour,
or equivalent. Do not infer cancellation only from unusual arithmetic.

Return only transaction_status.
""".strip(),
    },
    "net_amount": {
        "schema": named_money_schema("net_amount"),
        "num_predict": 128,
        "question": """
What is the final net amount before VAT after all discounts?

Do not return gross purchase total, pre-discount total, VAT amount, payment
received, or change.
Prefer a value explicitly labelled Netto when the VAT-table relationship
supports that role.

Return only net_amount and currency.
""".strip(),
    },
    "vat_amount": {
        "schema": named_money_schema("vat_amount"),
        "num_predict": 128,
        "question": """
What is the total VAT amount included in the final purchase total?

Do not return net amount, gross purchase total, tax rate, discount, payment, or
change. When several VAT rows are printed, return the total VAT amount only when
it is explicitly printed or clearly represented as the VAT total; otherwise
return null.

Return only vat_amount and currency.
""".strip(),
    },
    "vat_lines": {
        "schema": VAT_LINES_SCHEMA,
        "num_predict": 768,
        "question": """
Extract only the printed VAT rows.

For each VAT row return:
- source_rows: row IDs supporting the VAT row;
- rate_percent;
- net_amount;
- vat_amount.

Do not include the gross receipt total as VAT.
Do not reverse net and VAT columns.
Use null when the column relationship is unclear.
""".strip(),
    },
}


@dataclass(frozen=True)
class DetectionBox:
    index: int
    polygon: tuple[tuple[float, float], ...]
    score: float | None
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max) / 2.0


@dataclass(frozen=True)
class DetectedLine:
    index: int
    box_indices: tuple[int, ...]
    polygons: tuple[tuple[tuple[float, float], ...], ...]
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max) / 2.0


@dataclass(frozen=True)
class LineGroupSpec:
    group_id: str
    line_indices: tuple[int, ...]
    top: int
    bottom: int
    left: int
    right: int
    image: Image.Image


@dataclass(frozen=True)
class GroupCallResult:
    spec: LineGroupSpec
    lines: tuple[str, ...]
    response: dict[str, Any]
    metrics: dict[str, Any]
    attempt: int


@dataclass(frozen=True)
class VerifiedCutBoundary:
    cut_index: int
    y: int
    geometric_gap_pixels: float
    ink_density: float
    strip_top: int
    strip_bottom: int
    roi_left: int
    roi_right: int


_PADDLE_DETECTOR_CACHE: dict[tuple[str, str | None, str, str], tuple[str, Any]] = {}


def preprocess_crop(
    crop: Image.Image,
    *,
    scale: float,
    contrast: float,
    sharpen: bool,
) -> Image.Image:
    processed = ImageOps.exif_transpose(crop).convert("L")
    # processed = ImageOps.autocontrast(processed)
    # processed = ImageEnhance.Contrast(processed).enhance(contrast)

    if sharpen:
        processed = processed.filter(ImageFilter.SHARPEN)

    if scale != 1.0:
        width, height = processed.size
        processed = processed.resize(
            (
                max(1, round(width * scale)),
                max(1, round(height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )

    return processed.convert("RGB")


def image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def clean_plain_lines(text: str) -> list[str]:
    lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in {"```", "```text", "```txt", "BEGIN_RECEIPT", "END_RECEIPT"}:
            continue

        line = re.sub(r"^(?:[-*]\s+)", "", line).strip()
        line = re.sub(r"^R\d{4}\s*::\s*", "", line).strip()
        if line:
            lines.append(line)

    if not lines:
        raise RuntimeError("The Qwen transcription contained no usable lines")

    return lines


def serialize_receipt(lines: Sequence[str]) -> str:
    output = ["BEGIN_RECEIPT"]
    output.extend(f"R{index:04d} :: {line}" for index, line in enumerate(lines, start=1))
    output.append("END_RECEIPT")
    return "\n".join(output)


def _to_plain_python(value: Any) -> Any:
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_point(value: Any) -> bool:
    value = _to_plain_python(value)
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and _is_number(value[0])
        and _is_number(value[1])
    )


def _coerce_polygon(value: Any) -> tuple[tuple[float, float], ...] | None:
    value = _to_plain_python(value)

    if (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(_is_number(item) for item in value)
    ):
        x_min, y_min, x_max, y_max = (float(item) for item in value)
        return (
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
        )

    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    if not all(_is_point(point) for point in value):
        return None

    return tuple(
        (float(_to_plain_python(point)[0]), float(_to_plain_python(point)[1])) for point in value
    )


def _object_to_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value

    for attribute_name in ("json", "to_dict", "dict"):
        attribute = getattr(value, attribute_name, None)
        try:
            candidate = attribute() if callable(attribute) else attribute
        except Exception:
            continue

        if isinstance(candidate, str):
            try:
                candidate = json.loads(candidate)
            except Exception:
                continue
        if isinstance(candidate, dict):
            return candidate

    raw_dict = getattr(value, "__dict__", None)
    return raw_dict if isinstance(raw_dict, dict) else None


def _find_named_value(value: Any, keys: set[str]) -> Any:
    value = _to_plain_python(value)
    mapping = _object_to_mapping(value)
    if mapping is not None:
        for key, candidate in mapping.items():
            if str(key).casefold() in keys:
                return candidate
        for candidate in mapping.values():
            found = _find_named_value(candidate, keys)
            if found is not None:
                return found
        return None

    if isinstance(value, (list, tuple)):
        for candidate in value:
            found = _find_named_value(candidate, keys)
            if found is not None:
                return found
    return None


def _collect_polygons(value: Any) -> list[tuple[tuple[float, float], ...]]:
    value = _to_plain_python(value)
    polygon = _coerce_polygon(value)
    if polygon is not None:
        return [polygon]

    mapping = _object_to_mapping(value)
    if mapping is not None:
        collected: list[tuple[tuple[float, float], ...]] = []
        for candidate in mapping.values():
            collected.extend(_collect_polygons(candidate))
        return collected

    if isinstance(value, (list, tuple)):
        collected = []
        for candidate in value:
            collected.extend(_collect_polygons(candidate))
        return collected

    return []


def _deduplicate_polygons(
    polygons: Sequence[tuple[tuple[float, float], ...]],
) -> list[tuple[tuple[float, float], ...]]:
    unique: list[tuple[tuple[float, float], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()

    for polygon in polygons:
        key = tuple((round(point[0]), round(point[1])) for point in polygon)
        if key in seen:
            continue
        seen.add(key)
        unique.append(polygon)

    return unique


def _extract_polygons_and_scores(
    raw_result: Any,
) -> tuple[
    list[tuple[tuple[float, float], ...]],
    list[float | None],
]:
    polygon_value = _find_named_value(
        raw_result,
        {
            "dt_polys",
            "det_polys",
            "rec_polys",
            "polys",
            "boxes",
            "text_boxes",
        },
    )
    polygons = _collect_polygons(polygon_value if polygon_value is not None else raw_result)
    polygons = _deduplicate_polygons(polygons)

    score_value = _find_named_value(
        raw_result,
        {
            "dt_scores",
            "det_scores",
            "scores",
            "text_scores",
        },
    )
    raw_scores = _to_plain_python(score_value)
    scores: list[float | None] = []

    if isinstance(raw_scores, (list, tuple)):
        for score in raw_scores:
            score = _to_plain_python(score)
            scores.append(float(score) if _is_number(score) else None)

    if len(scores) < len(polygons):
        scores.extend([None] * (len(polygons) - len(scores)))

    return polygons, scores[: len(polygons)]


def _filtered_kwargs(callable_object: Any, values: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_object)
    except (TypeError, ValueError):
        return {key: value for key, value in values.items() if value is not None}

    has_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if has_var_kwargs:
        return {key: value for key, value in values.items() if value is not None}

    return {
        key: value
        for key, value in values.items()
        if key in signature.parameters and value is not None
    }


def _initialize_paddle_detector(
    args: argparse.Namespace,
) -> tuple[str, Any]:
    backend = str(args.paddle_backend)
    model_name = str(args.paddle_det_model) if args.paddle_det_model else None
    cache_key = (
        backend,
        model_name,
        str(args.paddle_device),
        str(args.paddle_lang),
    )
    cached = _PADDLE_DETECTOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    errors: list[str] = []

    if backend in {"auto", "text_detection"}:
        try:
            from paddleocr import TextDetection

            kwargs = _filtered_kwargs(
                TextDetection,
                {
                    "model_name": model_name,
                    "device": args.paddle_device,
                },
            )
            engine = TextDetection(**kwargs)
            result = ("text_detection", engine)
            _PADDLE_DETECTOR_CACHE[cache_key] = result
            return result
        except Exception as exc:
            errors.append(f"TextDetection initialization failed: {type(exc).__name__}: {exc}")
            if backend == "text_detection":
                raise RuntimeError("; ".join(errors)) from exc

    if backend in {"auto", "paddleocr"}:
        try:
            from paddleocr import PaddleOCR

            constructor_variants = [
                {
                    "lang": args.paddle_lang,
                    "device": args.paddle_device,
                    "use_doc_orientation_classify": False,
                    "use_doc_unwarping": False,
                    "use_textline_orientation": False,
                    "show_log": False,
                },
                {
                    "lang": args.paddle_lang,
                    "use_angle_cls": False,
                    "use_gpu": str(args.paddle_device).casefold().startswith("gpu"),
                    "show_log": False,
                },
                {
                    "lang": args.paddle_lang,
                },
                {},
            ]

            last_error: Exception | None = None
            for candidate_kwargs in constructor_variants:
                try:
                    kwargs = _filtered_kwargs(PaddleOCR, candidate_kwargs)
                    engine = PaddleOCR(**kwargs)
                    result = ("paddleocr", engine)
                    _PADDLE_DETECTOR_CACHE[cache_key] = result
                    return result
                except Exception as exc:
                    last_error = exc

            if last_error is not None:
                raise last_error
        except Exception as exc:
            errors.append(f"PaddleOCR initialization failed: {type(exc).__name__}: {exc}")

    raise RuntimeError("Could not initialize a PaddleOCR detector. " + "; ".join(errors))


def _predict_paddle(
    backend: str,
    engine: Any,
    image_path: Path,
    image: Image.Image,
) -> Any:
    errors: list[str] = []

    if backend == "text_detection":
        predict = getattr(engine, "predict", None)
        if not callable(predict):
            raise RuntimeError("TextDetection engine has no predict() method")

        variants = [
            {"input": str(image_path), "batch_size": 1},
            {"input": str(image_path)},
            {"input": image},
        ]
        for kwargs in variants:
            try:
                result = predict(**_filtered_kwargs(predict, kwargs))
                return list(result) if not isinstance(result, list) else result
            except Exception as exc:
                errors.append(f"TextDetection.predict failed: {type(exc).__name__}: {exc}")
        raise RuntimeError("; ".join(errors))

    legacy_ocr = getattr(engine, "ocr", None)
    if callable(legacy_ocr):
        try:
            import numpy as np

            return legacy_ocr(
                np.asarray(image),
                det=True,
                rec=False,
                cls=False,
            )
        except Exception as exc:
            errors.append(f"PaddleOCR.ocr(det=True, rec=False) failed: {type(exc).__name__}: {exc}")

    predict = getattr(engine, "predict", None)
    if callable(predict):
        variants = [
            {"input": str(image_path)},
            {"input": image},
        ]
        for kwargs in variants:
            try:
                result = predict(**_filtered_kwargs(predict, kwargs))
                return list(result) if not isinstance(result, list) else result
            except Exception as exc:
                errors.append(f"PaddleOCR.predict failed: {type(exc).__name__}: {exc}")

    raise RuntimeError("PaddleOCR detector call failed. " + "; ".join(errors))


def detect_paddle_boxes(
    args: argparse.Namespace,
    image_path: Path,
    image: Image.Image,
) -> tuple[list[DetectionBox], dict[str, Any]]:
    backend, engine = _initialize_paddle_detector(args)

    started = time.perf_counter()
    raw_result = _predict_paddle(
        backend,
        engine,
        image_path,
        image,
    )
    wall_seconds = time.perf_counter() - started

    polygons, scores = _extract_polygons_and_scores(raw_result)
    if not polygons:
        raise RuntimeError("PaddleOCR returned no parseable detection polygons")

    image_width, image_height = image.size
    boxes: list[DetectionBox] = []
    filtered_low_score = 0
    filtered_small = 0

    for source_index, (polygon, score) in enumerate(
        zip(polygons, scores, strict=False),
        start=1,
    ):
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        x_min = max(0.0, min(xs))
        y_min = max(0.0, min(ys))
        x_max = min(float(image_width), max(xs))
        y_max = min(float(image_height), max(ys))

        width = x_max - x_min
        height = y_max - y_min

        if score is not None and score < args.paddle_min_score:
            filtered_low_score += 1
            continue
        if width < args.paddle_min_box_width or height < args.paddle_min_box_height:
            filtered_small += 1
            continue

        boxes.append(
            DetectionBox(
                index=source_index,
                polygon=polygon,
                score=score,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
            )
        )

    if not boxes:
        raise RuntimeError("All PaddleOCR detections were filtered out")

    metadata = {
        "backend": backend,
        "model_name": args.paddle_det_model,
        "device": args.paddle_device,
        "language": args.paddle_lang,
        "wall_duration_seconds": round(wall_seconds, 3),
        "raw_polygon_count": len(polygons),
        "accepted_box_count": len(boxes),
        "filtered_low_score_count": filtered_low_score,
        "filtered_small_count": filtered_small,
    }
    return boxes, metadata


def _vertical_overlap_ratio(
    first_y_min: float,
    first_y_max: float,
    second_y_min: float,
    second_y_max: float,
) -> float:
    overlap = max(
        0.0,
        min(first_y_max, second_y_max) - max(first_y_min, second_y_min),
    )
    denominator = max(
        1.0,
        min(
            first_y_max - first_y_min,
            second_y_max - second_y_min,
        ),
    )
    return overlap / denominator


def cluster_boxes_into_lines(
    boxes: Sequence[DetectionBox],
    *,
    overlap_threshold: float,
    center_factor: float,
    min_line_width: float,
    min_line_height: float,
) -> list[DetectedLine]:
    working: list[dict[str, Any]] = []

    for box in sorted(
        boxes,
        key=lambda value: (value.center_y, value.x_min),
    ):
        best_index: int | None = None
        best_score = -1.0

        for index in range(max(0, len(working) - 4), len(working)):
            line = working[index]
            line_height = max(1.0, line["y_max"] - line["y_min"])
            overlap_ratio = _vertical_overlap_ratio(
                box.y_min,
                box.y_max,
                line["y_min"],
                line["y_max"],
            )
            center_delta = abs(box.center_y - ((line["y_min"] + line["y_max"]) / 2.0))
            center_limit = center_factor * min(
                max(1.0, box.height),
                line_height,
            )

            compatible = overlap_ratio >= overlap_threshold or center_delta <= center_limit
            if not compatible:
                continue

            score = overlap_ratio - (center_delta / max(1.0, center_limit)) * 0.05
            if score > best_score:
                best_index = index
                best_score = score

        if best_index is None:
            working.append(
                {
                    "boxes": [box],
                    "x_min": box.x_min,
                    "y_min": box.y_min,
                    "x_max": box.x_max,
                    "y_max": box.y_max,
                }
            )
            continue

        line = working[best_index]
        line["boxes"].append(box)
        line["x_min"] = min(line["x_min"], box.x_min)
        line["y_min"] = min(line["y_min"], box.y_min)
        line["x_max"] = max(line["x_max"], box.x_max)
        line["y_max"] = max(line["y_max"], box.y_max)

    detected_lines: list[DetectedLine] = []
    for line in sorted(
        working,
        key=lambda value: (
            (value["y_min"] + value["y_max"]) / 2.0,
            value["x_min"],
        ),
    ):
        member_boxes = sorted(
            line["boxes"],
            key=lambda value: value.x_min,
        )
        width = line["x_max"] - line["x_min"]
        height = line["y_max"] - line["y_min"]
        if width < min_line_width or height < min_line_height:
            continue

        detected_lines.append(
            DetectedLine(
                index=len(detected_lines),
                box_indices=tuple(member.index for member in member_boxes),
                polygons=tuple(member.polygon for member in member_boxes),
                x_min=float(line["x_min"]),
                y_min=float(line["y_min"]),
                x_max=float(line["x_max"]),
                y_max=float(line["y_max"]),
            )
        )

    return detected_lines


def build_line_groups(
    image: Image.Image,
    lines: Sequence[DetectedLine],
    *,
    lines_per_group: int,
    overlap_lines: int,
    horizontal_padding: int,
) -> list[LineGroupSpec]:
    if not lines:
        return []

    if lines_per_group < 1:
        raise ValueError("lines_per_group must be at least 1")
    if overlap_lines < 0 or overlap_lines >= lines_per_group:
        raise ValueError("line group overlap must be >= 0 and smaller than lines_per_group")

    image_width, image_height = image.size
    overall_left = max(
        0,
        math.floor(min(line.x_min for line in lines) - horizontal_padding),
    )
    overall_right = min(
        image_width,
        math.ceil(max(line.x_max for line in lines) + horizontal_padding),
    )
    if overall_right <= overall_left:
        overall_left = 0
        overall_right = image_width

    stride = lines_per_group - overlap_lines
    specs: list[LineGroupSpec] = []
    start = 0

    while start < len(lines):
        end = min(len(lines), start + lines_per_group)
        selected = lines[start:end]

        if start == 0:
            top = 0
        else:
            top = math.floor((lines[start - 1].y_max + lines[start].y_min) / 2.0)

        if end == len(lines):
            bottom = image_height
        else:
            bottom = math.ceil((lines[end - 1].y_max + lines[end].y_min) / 2.0)

        top = max(0, min(image_height - 1, top))
        bottom = max(top + 1, min(image_height, bottom))

        specs.append(
            LineGroupSpec(
                group_id=f"G{len(specs) + 1:03d}",
                line_indices=tuple(line.index for line in selected),
                top=top,
                bottom=bottom,
                left=overall_left,
                right=overall_right,
                image=image.crop(
                    (
                        overall_left,
                        top,
                        overall_right,
                        bottom,
                    )
                ),
            )
        )

        if end == len(lines):
            break
        start += stride

    return specs


def build_subgroup(
    image: Image.Image,
    lines: Sequence[DetectedLine],
    line_indices: Sequence[int],
    *,
    group_id: str,
    horizontal_padding: int,
) -> LineGroupSpec:
    if not line_indices:
        raise ValueError("Cannot build an empty subgroup")

    image_width, image_height = image.size
    first_index = min(line_indices)
    last_index = max(line_indices)

    if first_index == 0:
        top = 0
    else:
        top = math.floor((lines[first_index - 1].y_max + lines[first_index].y_min) / 2.0)

    if last_index == len(lines) - 1:
        bottom = image_height
    else:
        bottom = math.ceil((lines[last_index].y_max + lines[last_index + 1].y_min) / 2.0)

    left = max(
        0,
        math.floor(min(line.x_min for line in lines) - horizontal_padding),
    )
    right = min(
        image_width,
        math.ceil(max(line.x_max for line in lines) + horizontal_padding),
    )
    if right <= left:
        left = 0
        right = image_width

    top = max(0, min(image_height - 1, top))
    bottom = max(top + 1, min(image_height, bottom))

    return LineGroupSpec(
        group_id=group_id,
        line_indices=tuple(line_indices),
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        image=image.crop((left, top, right, bottom)),
    )


def safe_gap_pixels(
    lines: Sequence[DetectedLine],
    cut_index: int,
) -> float:
    """Return geometric whitespace between detected rows around cut_index."""
    if cut_index <= 0 or cut_index >= len(lines):
        return float("inf")
    return float(lines[cut_index].y_min - lines[cut_index - 1].y_max)


def _horizontal_text_roi(
    image: Image.Image,
    lines: Sequence[DetectedLine],
    horizontal_padding: int,
    full_width: bool,
) -> tuple[int, int]:
    image_width, _ = image.size
    if full_width or not lines:
        return 0, image_width

    left = max(
        0,
        math.floor(min(line.x_min for line in lines) - horizontal_padding),
    )
    right = min(
        image_width,
        math.ceil(max(line.x_max for line in lines) + horizontal_padding),
    )
    if right <= left:
        return 0, image_width
    return left, right


def _strip_ink_density(
    grayscale: Image.Image,
    *,
    y: int,
    left: int,
    right: int,
    half_height: int,
    dark_threshold: int,
) -> tuple[float, int, int]:
    image_width, image_height = grayscale.size
    left = max(0, min(image_width - 1, left))
    right = max(left + 1, min(image_width, right))
    strip_top = max(0, y - max(0, half_height))
    strip_bottom = min(image_height, y + max(0, half_height) + 1)
    strip = grayscale.crop((left, strip_top, right, strip_bottom))
    histogram = strip.histogram()
    threshold = max(0, min(255, int(dark_threshold)))
    dark_pixels = sum(histogram[: threshold + 1])
    total_pixels = max(1, strip.width * strip.height)
    return dark_pixels / total_pixels, strip_top, strip_bottom


def find_verified_cut_boundary(
    image: Image.Image,
    grayscale: Image.Image,
    lines: Sequence[DetectedLine],
    cut_index: int,
    *,
    roi_left: int,
    roi_right: int,
    min_safe_gap: float,
    search_margin: int,
    strip_half_height: int,
    dark_threshold: int,
    max_ink_density: float,
) -> VerifiedCutBoundary | None:
    """
    Verify one between-row boundary using original-image pixels.

    Paddle proposes the gap. The accepted y-coordinate is the lowest-ink horizontal
    strip inside that gap. A missed Paddle row therefore tends to invalidate the cut.
    """
    image_height = image.height
    if cut_index <= 0:
        return VerifiedCutBoundary(
            cut_index=0,
            y=0,
            geometric_gap_pixels=float("inf"),
            ink_density=0.0,
            strip_top=0,
            strip_bottom=0,
            roi_left=roi_left,
            roi_right=roi_right,
        )
    if cut_index >= len(lines):
        return VerifiedCutBoundary(
            cut_index=len(lines),
            y=image_height,
            geometric_gap_pixels=float("inf"),
            ink_density=0.0,
            strip_top=image_height,
            strip_bottom=image_height,
            roi_left=roi_left,
            roi_right=roi_right,
        )

    previous = lines[cut_index - 1]
    following = lines[cut_index]
    geometric_gap = float(following.y_min - previous.y_max)
    if geometric_gap < min_safe_gap:
        return None

    margin = max(0, int(search_margin))
    start_y = max(1, math.ceil(previous.y_max) + margin)
    end_y = min(image_height - 1, math.floor(following.y_min) - margin)
    if end_y < start_y:
        return None

    midpoint = (previous.y_max + following.y_min) / 2.0
    candidates: list[tuple[float, float, int, int, int]] = []
    for y in range(start_y, end_y + 1):
        density, strip_top, strip_bottom = _strip_ink_density(
            grayscale,
            y=y,
            left=roi_left,
            right=roi_right,
            half_height=strip_half_height,
            dark_threshold=dark_threshold,
        )
        candidates.append((density, abs(float(y) - midpoint), y, strip_top, strip_bottom))

    if not candidates:
        return None
    density, _, y, strip_top, strip_bottom = min(candidates)
    if density > max_ink_density:
        return None

    return VerifiedCutBoundary(
        cut_index=cut_index,
        y=y,
        geometric_gap_pixels=geometric_gap,
        ink_density=density,
        strip_top=strip_top,
        strip_bottom=strip_bottom,
        roi_left=roi_left,
        roi_right=roi_right,
    )


def precompute_verified_boundaries(
    image: Image.Image,
    lines: Sequence[DetectedLine],
    *,
    full_width: bool,
    horizontal_padding: int,
    min_safe_gap: float,
    search_margin: int,
    strip_half_height: int,
    dark_threshold: int,
    max_ink_density: float,
) -> dict[int, VerifiedCutBoundary]:
    grayscale = ImageOps.grayscale(image)
    roi_left, roi_right = _horizontal_text_roi(
        image,
        lines,
        horizontal_padding,
        full_width,
    )
    boundaries: dict[int, VerifiedCutBoundary] = {}
    for cut_index in range(0, len(lines) + 1):
        boundary = find_verified_cut_boundary(
            image,
            grayscale,
            lines,
            cut_index,
            roi_left=roi_left,
            roi_right=roi_right,
            min_safe_gap=min_safe_gap,
            search_margin=search_margin,
            strip_half_height=strip_half_height,
            dark_threshold=dark_threshold,
            max_ink_density=max_ink_density,
        )
        if boundary is not None:
            boundaries[cut_index] = boundary
    return boundaries


def choose_verified_cut_index(
    boundaries: dict[int, VerifiedCutBoundary],
    *,
    start_index: int,
    end_index: int,
    preferred_index: int,
    minimum_group_lines: int,
) -> int | None:
    minimum = start_index + max(1, minimum_group_lines)
    candidates = [
        boundary for cut_index, boundary in boundaries.items() if minimum <= cut_index < end_index
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda boundary: (
            abs(boundary.cut_index - preferred_index),
            boundary.ink_density,
            -boundary.geometric_gap_pixels,
            boundary.cut_index,
        ),
    ).cut_index


def _build_group_spec_from_boundaries(
    image: Image.Image,
    lines: Sequence[DetectedLine],
    line_indices: Sequence[int],
    *,
    group_id: str,
    top: int,
    bottom: int,
    full_width: bool,
    horizontal_padding: int,
) -> LineGroupSpec:
    if not line_indices:
        raise ValueError("Cannot build an empty safe-cut group")
    ordered = tuple(sorted(line_indices))
    if tuple(range(ordered[0], ordered[-1] + 1)) != ordered:
        raise ValueError("Safe-cut groups must contain contiguous detected rows")

    image_width, image_height = image.size
    top = max(0, min(image_height - 1, int(top)))
    bottom = max(top + 1, min(image_height, int(bottom)))
    left, right = _horizontal_text_roi(
        image,
        lines,
        horizontal_padding,
        full_width,
    )
    return LineGroupSpec(
        group_id=group_id,
        line_indices=ordered,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        image=image.crop((left, top, right, bottom)),
    )


def determine_effective_crop_count(
    *,
    requested_crops: int,
    detected_line_count: int,
    image_width: int,
    image_height: int,
    target_rows_per_crop: int,
    single_crop_max_rows: int,
    single_crop_max_aspect_ratio: float,
) -> tuple[int, dict[str, Any]]:
    """Determine crop count from row density and scale-independent H/W ratio."""
    if requested_crops < 1:
        raise ValueError("requested_crops must be at least 1")
    if target_rows_per_crop < 1:
        raise ValueError("target_rows_per_crop must be at least 1")
    if single_crop_max_rows < 1 or single_crop_max_aspect_ratio <= 0:
        raise ValueError("single-crop thresholds must be positive")
    if image_width < 1 or image_height < 1:
        raise ValueError("image dimensions must be positive")

    aspect_ratio = float(image_height) / float(image_width)
    small_by_rows = detected_line_count <= single_crop_max_rows
    small_by_aspect_ratio = aspect_ratio <= single_crop_max_aspect_ratio

    if requested_crops == 1 or (small_by_rows and small_by_aspect_ratio):
        effective = 1
        reason = "requested_single_crop" if requested_crops == 1 else "small_receipt"
    else:
        row_based = max(
            1,
            math.ceil(detected_line_count / target_rows_per_crop),
        )
        aspect_ratio_based = max(
            1,
            math.ceil(aspect_ratio / single_crop_max_aspect_ratio),
        )
        effective = min(
            requested_crops,
            max(2, row_based, aspect_ratio_based),
        )
        reason = "adaptive_up_to_requested_crop_count"

    return effective, {
        "requested_crops": requested_crops,
        "initial_effective_crops": effective,
        "reason": reason,
        "detected_line_count_estimate": detected_line_count,
        "image_width": image_width,
        "image_height": image_height,
        "image_aspect_ratio_h_over_w": round(aspect_ratio, 6),
        "target_rows_per_crop": target_rows_per_crop,
        "single_crop_max_rows": single_crop_max_rows,
        "single_crop_max_aspect_ratio": single_crop_max_aspect_ratio,
        "small_by_rows": small_by_rows,
        "small_by_aspect_ratio": small_by_aspect_ratio,
    }


def _select_boundary_near_nominal_y(
    boundaries: dict[int, VerifiedCutBoundary],
    *,
    nominal_y: float,
    previous_cut_index: int,
    detected_line_count: int,
    remaining_crops_after_cut: int,
    minimum_lines_per_crop: int,
    normal_radius: float,
    maximum_radius: float,
) -> tuple[VerifiedCutBoundary | None, str]:
    minimum_cut_index = previous_cut_index + minimum_lines_per_crop
    maximum_cut_index = detected_line_count - remaining_crops_after_cut * minimum_lines_per_crop
    if minimum_cut_index > maximum_cut_index:
        return None, "insufficient_detected_lines_for_remaining_crops"

    candidates = [
        boundary
        for cut_index, boundary in boundaries.items()
        if (
            minimum_cut_index <= cut_index <= maximum_cut_index
            and 0 < cut_index < detected_line_count
        )
    ]
    if not candidates:
        return None, "no_verified_boundary_in_allowed_line_range"

    normal = [
        boundary for boundary in candidates if abs(float(boundary.y) - nominal_y) <= normal_radius
    ]
    if normal:
        pool = normal
        search_stage = "normal_search_radius"
    else:
        expanded = [
            boundary
            for boundary in candidates
            if abs(float(boundary.y) - nominal_y) <= maximum_radius
        ]
        if not expanded:
            return None, "no_verified_boundary_within_max_search_radius"
        pool = expanded
        search_stage = "expanded_search_radius"

    selected = min(
        pool,
        key=lambda boundary: (
            abs(float(boundary.y) - nominal_y),
            boundary.ink_density,
            -boundary.geometric_gap_pixels,
            boundary.cut_index,
        ),
    )
    return selected, search_stage


def _try_build_snapped_crop_plan(
    image: Image.Image,
    lines: Sequence[DetectedLine],
    boundaries: dict[int, VerifiedCutBoundary],
    *,
    crop_count: int,
    minimum_lines_per_crop: int,
    safe_cut_search_ratio: float,
    max_safe_cut_search_ratio: float,
    full_width: bool,
    horizontal_padding: int,
) -> tuple[
    list[LineGroupSpec] | None,
    list[dict[str, Any]],
    str | None,
]:
    if crop_count < 1:
        return None, [], "crop_count_below_one"
    if 0 not in boundaries or len(lines) not in boundaries:
        return None, [], "missing_image_edge_boundaries"

    if crop_count == 1:
        spec = _build_group_spec_from_boundaries(
            image,
            lines,
            tuple(range(len(lines))),
            group_id="GFULL",
            top=0,
            bottom=image.height,
            full_width=full_width,
            horizontal_padding=horizontal_padding,
        )
        return (
            [spec],
            [
                {
                    "boundary_number": 0,
                    "selection": "single_full_image_crop",
                    "top": 0,
                    "bottom": image.height,
                }
            ],
            None,
        )

    if len(lines) < crop_count * minimum_lines_per_crop:
        return (
            None,
            [],
            "insufficient_detected_lines_for_minimum_lines_per_crop",
        )

    nominal_crop_height = float(image.height) / float(crop_count)
    normal_radius = max(
        1.0,
        nominal_crop_height * safe_cut_search_ratio,
    )
    maximum_radius = max(
        normal_radius,
        nominal_crop_height * max_safe_cut_search_ratio,
    )

    selected_boundaries = [boundaries[0]]
    decisions: list[dict[str, Any]] = []

    previous_cut_index = 0
    for boundary_number in range(1, crop_count):
        nominal_y = float(image.height) * boundary_number / crop_count
        remaining_crops_after_cut = crop_count - boundary_number
        selected, stage = _select_boundary_near_nominal_y(
            boundaries,
            nominal_y=nominal_y,
            previous_cut_index=previous_cut_index,
            detected_line_count=len(lines),
            remaining_crops_after_cut=remaining_crops_after_cut,
            minimum_lines_per_crop=minimum_lines_per_crop,
            normal_radius=normal_radius,
            maximum_radius=maximum_radius,
        )
        if selected is None:
            return (
                None,
                decisions,
                f"boundary_{boundary_number}_{stage}",
            )

        if selected.y <= selected_boundaries[-1].y:
            return (
                None,
                decisions,
                f"boundary_{boundary_number}_not_monotonic",
            )

        decisions.append(
            {
                "boundary_number": boundary_number,
                "nominal_y": round(nominal_y, 3),
                "selected_y": selected.y,
                "distance_from_nominal_pixels": round(
                    abs(float(selected.y) - nominal_y),
                    3,
                ),
                "search_stage": stage,
                "cut_index": selected.cut_index,
                "ink_density": selected.ink_density,
                "geometric_gap_pixels": selected.geometric_gap_pixels,
                "strip_top": selected.strip_top,
                "strip_bottom": selected.strip_bottom,
            }
        )
        selected_boundaries.append(selected)
        previous_cut_index = selected.cut_index

    selected_boundaries.append(boundaries[len(lines)])

    specs: list[LineGroupSpec] = []
    for crop_index in range(crop_count):
        top_boundary = selected_boundaries[crop_index]
        bottom_boundary = selected_boundaries[crop_index + 1]
        start_index = top_boundary.cut_index
        end_index = bottom_boundary.cut_index
        if end_index <= start_index:
            return None, decisions, f"crop_{crop_index + 1}_empty_line_range"

        spec = _build_group_spec_from_boundaries(
            image,
            lines,
            tuple(range(start_index, end_index)),
            group_id=f"G{crop_index + 1:03d}",
            top=top_boundary.y,
            bottom=bottom_boundary.y,
            full_width=full_width,
            horizontal_padding=horizontal_padding,
        )
        specs.append(spec)

    return specs, decisions, None


def build_nominal_snapped_safe_cut_groups(
    image: Image.Image,
    lines: Sequence[DetectedLine],
    *,
    requested_crops: int,
    target_rows_per_crop: int,
    single_crop_max_rows: int,
    single_crop_max_aspect_ratio: float,
    minimum_lines_per_crop: int,
    safe_cut_search_ratio: float,
    max_safe_cut_search_ratio: float,
    full_width: bool,
    horizontal_padding: int,
    min_safe_gap: float,
    search_margin: int,
    strip_half_height: int,
    dark_threshold: int,
    max_ink_density: float,
) -> tuple[
    list[LineGroupSpec],
    dict[int, VerifiedCutBoundary],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """
    Build up to --crops non-overlapping crops.

    Equal-height nominal boundaries are snapped to the nearest Paddle-proposed,
    pixel-verified whitespace gap. If a safe plan cannot preserve the current crop
    count, the planner reduces the count until a complete plan is available.
    """
    if not lines:
        return (
            [],
            {},
            [],
            {
                "requested_crops": requested_crops,
                "effective_crops": 0,
                "status": "no_detected_lines",
            },
        )
    if not 0.0 <= safe_cut_search_ratio <= max_safe_cut_search_ratio <= 1.0:
        raise ValueError("safe-cut search ratios must satisfy 0 <= normal <= maximum <= 1")

    boundaries = precompute_verified_boundaries(
        image,
        lines,
        full_width=full_width,
        horizontal_padding=horizontal_padding,
        min_safe_gap=min_safe_gap,
        search_margin=search_margin,
        strip_half_height=strip_half_height,
        dark_threshold=dark_threshold,
        max_ink_density=max_ink_density,
    )
    if 0 not in boundaries or len(lines) not in boundaries:
        raise RuntimeError("Image-edge safe boundaries could not be established")

    desired_count, count_metadata = determine_effective_crop_count(
        requested_crops=requested_crops,
        detected_line_count=len(lines),
        image_width=image.width,
        image_height=image.height,
        target_rows_per_crop=target_rows_per_crop,
        single_crop_max_rows=single_crop_max_rows,
        single_crop_max_aspect_ratio=single_crop_max_aspect_ratio,
    )

    attempts: list[dict[str, Any]] = []
    for crop_count in range(desired_count, 0, -1):
        specs, decisions, failure = _try_build_snapped_crop_plan(
            image,
            lines,
            boundaries,
            crop_count=crop_count,
            minimum_lines_per_crop=minimum_lines_per_crop,
            safe_cut_search_ratio=safe_cut_search_ratio,
            max_safe_cut_search_ratio=max_safe_cut_search_ratio,
            full_width=full_width,
            horizontal_padding=horizontal_padding,
        )
        attempts.append(
            {
                "crop_count": crop_count,
                "status": "planned" if specs is not None else "rejected",
                "failure": failure,
            }
        )
        if specs is not None:
            plan = {
                **count_metadata,
                "effective_crops": crop_count,
                "crop_count_reduced_for_safety": crop_count < desired_count,
                "status": "planned",
                "safe_cut_search_ratio": safe_cut_search_ratio,
                "max_safe_cut_search_ratio": max_safe_cut_search_ratio,
                "minimum_lines_per_crop": minimum_lines_per_crop,
                "verified_boundary_count": len(boundaries),
                "attempts": attempts,
            }
            return specs, boundaries, decisions, plan

    raise RuntimeError("Could not create even a single complete Qwen crop plan")


def _transcribe_qwen_group_with_retries(
    args: argparse.Namespace,
    spec: LineGroupSpec,
    work_dir: Path,
) -> tuple[
    list[GroupCallResult],
    list[GroupCallResult],
    list[dict[str, Any]],
]:
    """Accept the first nonempty Qwen result; retry only transport/call failures."""
    call_results: list[GroupCallResult] = []
    diagnostics: list[dict[str, Any]] = []

    for attempt in range(1, args.qwen_group_retries + 2):
        try:
            result = _invoke_qwen_group_once(
                args,
                spec,
                work_dir,
                attempt=attempt,
            )
        except Exception as exc:
            diagnostics.append(
                {
                    "group_id": spec.group_id,
                    "attempt": attempt,
                    "status": "qwen_call_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "bbox": [spec.left, spec.top, spec.right, spec.bottom],
                }
            )
            continue

        call_results.append(result)
        diagnostics.append(
            {
                "group_id": spec.group_id,
                "attempt": attempt,
                "status": "accepted_without_post_transcription_validation",
                "paddle_detected_line_estimate": len(spec.line_indices),
                "qwen_returned_line_count": len(result.lines),
                "transcription_text_source": result.metrics.get("transcription_text_source"),
                "bbox": [spec.left, spec.top, spec.right, spec.bottom],
            }
        )
        return [result], call_results, diagnostics

    return [], call_results, diagnostics


def save_detection_overlay(
    path: Path,
    image: Image.Image,
    lines: Sequence[DetectedLine],
) -> None:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    for line in lines:
        draw.rectangle(
            (
                round(line.x_min),
                round(line.y_min),
                round(line.x_max),
                round(line.y_max),
            ),
            width=2,
        )
        draw.text(
            (
                round(line.x_min),
                max(0, round(line.y_min) - 12),
            ),
            f"L{line.index + 1:04d}",
        )

    overlay.save(path, format="PNG")


def detection_report(
    boxes: Sequence[DetectionBox],
    lines: Sequence[DetectedLine],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "metadata": metadata,
        "box_count": len(boxes),
        "line_count": len(lines),
        "boxes": [
            {
                "box_index": box.index,
                "score": box.score,
                "polygon": [[round(x, 3), round(y, 3)] for x, y in box.polygon],
                "bbox": [
                    round(box.x_min, 3),
                    round(box.y_min, 3),
                    round(box.x_max, 3),
                    round(box.y_max, 3),
                ],
            }
            for box in boxes
        ],
        "lines": [
            {
                "line_id": f"L{line.index + 1:04d}",
                "line_index": line.index,
                "box_indices": list(line.box_indices),
                "bbox": [
                    round(line.x_min, 3),
                    round(line.y_min, 3),
                    round(line.x_max, 3),
                    round(line.y_max, 3),
                ],
                "center_y": round(line.center_y, 3),
            }
            for line in lines
        ],
    }


def aggregate_group_metrics(
    results: Sequence[GroupCallResult],
    detector_metadata: dict[str, Any],
) -> dict[str, Any]:
    def numeric_sum(field: str) -> int | float | None:
        values = [
            result.metrics.get(field)
            for result in results
            if isinstance(result.metrics.get(field), (int, float))
        ]
        return sum(values) if values else None

    return {
        "detector_wall_duration_seconds": detector_metadata.get("wall_duration_seconds"),
        "qwen_call_count": len(results),
        "qwen_wall_duration_seconds_sum": round(
            sum(float(result.metrics.get("wall_duration_seconds") or 0.0) for result in results),
            3,
        ),
        "prompt_eval_count": numeric_sum("prompt_eval_count"),
        "eval_count": numeric_sum("eval_count"),
        "total_duration_ms": numeric_sum("total_duration_ms"),
        "load_duration_ms": numeric_sum("load_duration_ms"),
        "prompt_eval_duration_ms": numeric_sum("prompt_eval_duration_ms"),
        "eval_duration_ms": numeric_sum("eval_duration_ms"),
        "done_reason": [result.metrics.get("done_reason") for result in results],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_BATCH_INPUT,
        help=("Receipt image or folder containing receipt images (default: /app/var/batch_input)."),
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Continue with the next receipt after a per-receipt failure "
            "(default: enabled; use --no-continue-on-error to stop)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/app/var/experiments/paddle_safe_cuts_qwen_gemma_batch"),
    )
    parser.add_argument("--run-name")
    parser.add_argument(
        "--archive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Create a folder-preserving ZIP archive after the run finishes. "
            "Images and crop PNGs are excluded by default."
        ),
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        help=("Optional explicit output path for the archive ZIP. Default: <run-output-dir>.zip."),
    )

    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--vlm-model", default="qwen3.5:4b")
    parser.add_argument("--gemma-model", default="gemma4")

    parser.add_argument("--vlm-num-ctx", type=int, default=8192)
    parser.add_argument("--vlm-num-predict", type=int, default=4096)
    parser.add_argument("--vlm-timeout", type=float, default=300.0)
    parser.add_argument("--vlm-keep-alive", default="5m")
    parser.add_argument("--vlm-temperature", type=float, default=0.0)
    parser.add_argument("--vlm-seed", type=int, default=42)

    parser.add_argument(
        "--paddle-backend",
        choices=("auto", "text_detection", "paddleocr"),
        default="auto",
        help=(
            "Paddle detection API to use. 'auto' prefers paddleocr.TextDetection "
            "and falls back to PaddleOCR detection-only APIs."
        ),
    )
    parser.add_argument(
        "--paddle-det-model",
        help=(
            "Optional Paddle text-detection model name. Leave unset to use the "
            "installed PaddleOCR default."
        ),
    )
    parser.add_argument("--paddle-device", default="cpu")
    parser.add_argument("--paddle-lang", default="en")
    parser.add_argument("--paddle-min-score", type=float, default=0.20)
    parser.add_argument("--paddle-min-box-width", type=float, default=6.0)
    parser.add_argument("--paddle-min-box-height", type=float, default=4.0)
    parser.add_argument("--paddle-min-line-width", type=float, default=8.0)
    parser.add_argument("--paddle-min-line-height", type=float, default=4.0)
    parser.add_argument(
        "--paddle-line-overlap-threshold",
        type=float,
        default=0.40,
        help="Minimum vertical overlap ratio for boxes to share a physical row.",
    )
    parser.add_argument(
        "--paddle-line-center-factor",
        type=float,
        default=0.60,
        help=(
            "Alternative same-row test: allowed centre distance as a fraction "
            "of the smaller box height."
        ),
    )
    parser.add_argument("--paddle-min-lines", type=int, default=3)
    parser.add_argument("--paddle-max-lines", type=int, default=300)

    parser.add_argument(
        "--crops",
        "--max-crops",
        dest="crops",
        type=int,
        default=4,
        help=(
            "Maximum requested number of Qwen crops per receipt. Small receipts "
            "use one crop; unsafe multi-crop plans automatically reduce the count."
        ),
    )
    parser.add_argument(
        "--target-rows-per-crop",
        "--lines-per-group",
        dest="target_rows_per_crop",
        type=int,
        default=18,
        help=(
            "Approximate detected-row target used only to choose the effective "
            "crop count. --lines-per-group remains as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--single-crop-max-rows",
        type=int,
        default=25,
        help=(
            "Use one full-image Qwen call when detected rows do not exceed this "
            "value and H/W also satisfies --single-crop-max-aspect-ratio."
        ),
    )
    parser.add_argument(
        "--single-crop-max-aspect-ratio",
        type=float,
        default=2.0,
        help=(
            "Maximum image H/W ratio for automatic one-crop handling of a small "
            "receipt. This is scale-independent."
        ),
    )
    parser.add_argument(
        "--safe-cut-search-ratio",
        type=float,
        default=0.20,
        help=(
            "Initial search radius around each nominal boundary, expressed as a "
            "fraction of nominal crop height."
        ),
    )
    parser.add_argument(
        "--max-safe-cut-search-ratio",
        type=float,
        default=0.35,
        help=(
            "Maximum expanded search radius around a nominal boundary. When no "
            "safe cut exists inside it, the effective crop count is reduced."
        ),
    )
    parser.add_argument(
        "--line-group-overlap",
        type=int,
        default=0,
        help=(
            "Compatibility option. Non-overlapping crop concatenation requires 0 "
            "because Qwen crops are non-overlapping and concatenated in order."
        ),
    )
    parser.add_argument(
        "--line-group-horizontal-padding",
        type=int,
        default=20,
    )
    parser.add_argument("--crop-scale", type=float, default=1.5)
    parser.add_argument("--crop-contrast", type=float, default=1.15)
    parser.add_argument(
        "--crop-sharpen",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--qwen-group-parallelism",
        "--qwen-crop-parallelism",
        dest="qwen_group_parallelism",
        type=int,
        default=1,
        help=(
            "Maximum concurrent Qwen line-group calls per receipt. "
            "--qwen-crop-parallelism remains as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--qwen-group-retries",
        type=int,
        default=1,
        help="Retries after Qwen transport/call failure; successful text is accepted directly.",
    )
    parser.add_argument(
        "--full-width-crops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the full image width for each safe crop. Disable only when large "
            "side margins materially reduce Qwen readability."
        ),
    )
    parser.add_argument(
        "--safe-cut-padding",
        type=int,
        default=0,
        help=(
            "Optional pixels added outward from each midpoint cut. Default 0 keeps "
            "adjacent crops strictly non-overlapping."
        ),
    )
    parser.add_argument(
        "--min-safe-gap",
        type=float,
        default=1.0,
        help=(
            "Minimum detected whitespace in pixels required for an internal crop "
            "boundary. If no boundary exists, Qwen receives the whole image."
        ),
    )
    parser.add_argument(
        "--min-lines-per-crop",
        "--min-lines-per-group",
        dest="min_lines_per_crop",
        type=int,
        default=3,
        help=(
            "Minimum Paddle-detected row estimate retained in every planned crop. "
            "--min-lines-per-group remains as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--cut-search-margin",
        type=int,
        default=1,
        help="Pixels excluded next to detected row boxes when searching for a cut.",
    )
    parser.add_argument(
        "--cut-strip-half-height",
        type=int,
        default=3,
        help="Half-height of the horizontal strip used for pixel ink validation.",
    )
    parser.add_argument(
        "--cut-ink-threshold",
        type=int,
        default=190,
        help="Grayscale value at or below which a pixel counts as ink.",
    )
    parser.add_argument(
        "--max-cut-ink-density",
        type=float,
        default=0.01,
        help=("Maximum dark-pixel fraction allowed in a verified cut strip (default: 0.01 = 1%%)."),
    )
    parser.add_argument("--gemma-num-ctx", type=int, default=16384)
    parser.add_argument("--gemma-timeout", type=float, default=300.0)
    parser.add_argument("--gemma-keep-alive", default="10m")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gemma-seed", type=int, default=42)
    parser.add_argument("--item-num-predict", type=int, default=4096)
    parser.add_argument(
        "--item-think",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture Gemma thinking separately for the direct item call.",
    )
    parser.add_argument(
        "--gemma-parallelism",
        type=int,
        default=2,
        help=(
            "Maximum concurrent Gemma calls within each stage. "
            "Ollama must also allow parallel requests."
        ),
    )

    parser.add_argument(
        "--scalar-tasks",
        nargs="+",
        choices=SCALAR_TASK_ORDER,
        default=list(DEFAULT_BATCH_SCALAR_TASKS),
        help=(
            "Scalar specialists to run. Default batch tasks are merchant_name, "
            "merchant_address, currency, final_purchase_total, discount_total, "
            "vat_amount, and vat_lines."
        ),
    )
    parser.add_argument(
        "--skip-scalars",
        action="store_true",
        help="Skip all scalar specialists and run transcription/items only.",
    )
    parser.add_argument(
        "--skip-item-pipeline",
        action="store_true",
        help="Run scalar specialists only.",
    )
    parser.add_argument(
        "--row-context-radius",
        type=int,
        default=2,
        help="Number of neighbouring rows shown around each classified row.",
    )
    parser.add_argument(
        "--keep-vlm-loaded",
        action="store_true",
        help="Do not explicitly unload Qwen before the Gemma phase.",
    )

    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def timestamp_name() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def unique_ordered(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned[:100] or "receipt"


def strip_code_fences(value: str) -> str:
    text = value.strip()
    match = re.fullmatch(
        r"```(?:text|txt|markdown|json)?\s*(.*?)\s*```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else text


def discover_images(input_path: Path, recursive: bool) -> list[Path]:
    path = input_path.expanduser().resolve()

    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix}")
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    iterator = path.rglob("*") if recursive else path.glob("*")
    images = [
        item for item in iterator if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=lambda item: item.as_posix().casefold())


def receipt_key(image_path: Path, input_root: Path) -> str:
    try:
        relative = image_path.resolve().relative_to(input_root.resolve()).as_posix()
    except ValueError:
        relative = image_path.name

    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:8]
    return f"{sanitize_name(image_path.stem)}_{digest}"


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code} at {url}: {details or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot connect to Ollama at {url}: {exc}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"Ollama request timed out after {timeout:.1f} seconds") from exc

    try:
        result = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama returned invalid API JSON: {body[:1000]}") from exc

    if not isinstance(result, dict):
        raise RuntimeError("Ollama returned a non-object response")
    if result.get("error"):
        raise RuntimeError(str(result["error"]))

    return result


def ns_to_ms(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return round(float(value) / 1_000_000.0, 3)


def response_metrics(
    response: dict[str, Any],
    wall_seconds: float,
) -> dict[str, Any]:
    return {
        "wall_duration_seconds": round(wall_seconds, 3),
        "total_duration_ms": ns_to_ms(response.get("total_duration")),
        "load_duration_ms": ns_to_ms(response.get("load_duration")),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "prompt_eval_duration_ms": ns_to_ms(response.get("prompt_eval_duration")),
        "eval_count": response.get("eval_count"),
        "eval_duration_ms": ns_to_ms(response.get("eval_duration")),
        "done_reason": response.get("done_reason"),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_transcription_rows(
    transcription: str,
) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    warnings: list[str] = []
    seen: set[str] = set()

    for raw_line in transcription.splitlines():
        line = raw_line.strip()

        if not line or line in {"BEGIN_RECEIPT", "END_RECEIPT"}:
            continue

        match = re.match(r"^(R\d{4})\s*::\s*(.*)$", line)
        if not match:
            warnings.append(f"Unparsed line: {line}")
            continue

        row_id, text = match.groups()

        if row_id in seen:
            warnings.append(f"Duplicate row ID: {row_id}")
        seen.add(row_id)

        rows.append(
            {
                "row_id": row_id,
                "text": text.strip(),
            }
        )

    if not rows:
        raise RuntimeError("Qwen transcription contains no parseable R#### :: rows")

    return rows, warnings


# def _qwen_group_prompt(estimated_count: int) -> str:
#     return QWEN_TRANSCRIPTION_PROMPT_TEMPLATE.format(
#         estimated_count=estimated_count
#     )


def _qwen_group_prompt(estimated_count: int) -> str:
    return QWEN_TRANSCRIPTION_PROMPT_TEMPLATE


def _write_group_artifacts(
    work_dir: Path,
    result: GroupCallResult,
) -> None:
    prefix = work_dir / f"group_{result.spec.group_id}_attempt_{result.attempt:02d}"
    write_json(
        prefix.with_name(prefix.name + "_response.json"),
        result.response,
    )
    prefix.with_name(prefix.name + "_lines.txt").write_text(
        "\n".join(result.lines) + "\n",
        encoding="utf-8",
    )


def _extract_qwen_transcription_text(
    response: dict[str, Any],
    *,
    group_id: str,
) -> tuple[str, str]:
    """Return Qwen's nonempty visual answer without judging its content.

    Ollama normally returns the final answer in ``message.content``. Some
    Qwen3.5 vision builds instead place the complete image answer in
    ``message.thinking`` while leaving ``message.content`` empty, even when
    ``think`` is disabled. Treat that as a transport-field variation rather
    than a transcription failure.
    """
    message = response.get("message")
    if not isinstance(message, dict):
        raise RuntimeError(f"Qwen group {group_id} response has no message object")

    for field_name in ("content", "thinking"):
        value = message.get(field_name)
        if isinstance(value, str) and value.strip():
            return value, f"message.{field_name}"

    legacy_response = response.get("response")
    if isinstance(legacy_response, str) and legacy_response.strip():
        return legacy_response, "response"

    raise RuntimeError(
        f"Qwen group {group_id} returned no nonempty text in "
        "message.content, message.thinking, or response"
    )


def _invoke_qwen_group_once(
    args: argparse.Namespace,
    spec: LineGroupSpec,
    work_dir: Path,
    *,
    attempt: int,
) -> GroupCallResult:
    expected_count = len(spec.line_indices)
    processed = preprocess_crop(
        spec.image,
        scale=args.crop_scale,
        contrast=args.crop_contrast,
        sharpen=args.crop_sharpen,
    )

    image_path = work_dir / f"group_{spec.group_id}.png"
    if not image_path.exists() or args.overwrite:
        processed.save(image_path, format="PNG")

    prompt = _qwen_group_prompt(expected_count)
    payload = {
        "model": args.vlm_model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_to_base64(processed)],
            }
        ],
        "stream": False,
        "think": False,
        "keep_alive": args.vlm_keep_alive,
        "options": {
            "temperature": args.vlm_temperature,
            "seed": args.vlm_seed,
            "top_k": 1,
            "top_p": 1.0,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "repeat_last_n": 0,
            "num_ctx": args.vlm_num_ctx,
            "num_predict": (
                args.vlm_num_predict
                if spec.group_id.startswith("GFULL")
                else min(
                    args.vlm_num_predict,
                    max(256, expected_count * 160),
                )
            ),
        },
    }

    started = time.perf_counter()
    response = post_json(
        f"{args.ollama_url.rstrip('/')}/api/chat",
        payload,
        args.vlm_timeout,
    )
    wall_seconds = time.perf_counter() - started

    raw_response_path = work_dir / f"group_{spec.group_id}_attempt_{attempt:02d}_raw_response.json"
    write_json(raw_response_path, response)

    transcription_text, transcription_text_source = _extract_qwen_transcription_text(
        response,
        group_id=spec.group_id,
    )
    lines = tuple(clean_plain_lines(strip_code_fences(transcription_text)))
    metrics = response_metrics(response, wall_seconds)
    metrics["transcription_text_source"] = transcription_text_source
    result = GroupCallResult(
        spec=spec,
        lines=lines,
        response=response,
        metrics=metrics,
        attempt=attempt,
    )
    _write_group_artifacts(work_dir, result)
    return result


def _full_image_group_spec(
    image: Image.Image,
    detected_lines: Sequence[DetectedLine],
    *,
    group_id: str,
) -> LineGroupSpec:
    return LineGroupSpec(
        group_id=group_id,
        line_indices=tuple(range(len(detected_lines))),
        top=0,
        bottom=image.height,
        left=0,
        right=image.width,
        image=image.copy(),
    )


def invoke_paddle_qwen_transcription(
    args: argparse.Namespace,
    image_path: Path,
    receipt_dir: Path,
) -> dict[str, Any]:
    phase_started = time.perf_counter()
    work_dir = receipt_dir / "10_paddle_snapped_safe_crops"
    work_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    print("    PaddleOCR: detecting text boxes for crop proposals", flush=True)
    boxes: list[DetectionBox] = []
    detected_lines: list[DetectedLine] = []
    detector_error: dict[str, Any] | None = None
    try:
        boxes, detector_metadata = detect_paddle_boxes(args, image_path, image)
        detected_lines = cluster_boxes_into_lines(
            boxes,
            overlap_threshold=args.paddle_line_overlap_threshold,
            center_factor=args.paddle_line_center_factor,
            min_line_width=args.paddle_min_line_width,
            min_line_height=args.paddle_min_line_height,
        )
    except Exception as exc:
        detector_error = {
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        detector_metadata = {
            "status": "error",
            "backend": None,
            **detector_error,
        }

    detector_report = detection_report(boxes, detected_lines, detector_metadata)
    if detector_error is not None:
        detector_report["error"] = detector_error
    write_json(receipt_dir / "10_paddle_line_boxes.json", detector_report)
    save_detection_overlay(
        receipt_dir / "10_paddle_line_overlay.png",
        image,
        detected_lines,
    )

    planning_fallback_reason: str | None = None
    if detector_error is not None:
        planning_fallback_reason = "paddle_detection_failed"
    elif len(detected_lines) < args.paddle_min_lines:
        planning_fallback_reason = "too_few_detected_lines_for_safe_cropping"
    elif len(detected_lines) > args.paddle_max_lines:
        planning_fallback_reason = "too_many_detected_lines_for_safe_cropping"

    verified_boundaries: dict[int, VerifiedCutBoundary] = {}
    boundary_decisions: list[dict[str, Any]] = []
    if planning_fallback_reason is None:
        try:
            specs, verified_boundaries, boundary_decisions, crop_plan = (
                build_nominal_snapped_safe_cut_groups(
                    image,
                    detected_lines,
                    requested_crops=args.crops,
                    target_rows_per_crop=args.target_rows_per_crop,
                    single_crop_max_rows=args.single_crop_max_rows,
                    single_crop_max_aspect_ratio=(args.single_crop_max_aspect_ratio),
                    minimum_lines_per_crop=args.min_lines_per_crop,
                    safe_cut_search_ratio=args.safe_cut_search_ratio,
                    max_safe_cut_search_ratio=args.max_safe_cut_search_ratio,
                    full_width=args.full_width_crops,
                    horizontal_padding=args.line_group_horizontal_padding,
                    min_safe_gap=args.min_safe_gap,
                    search_margin=args.cut_search_margin,
                    strip_half_height=args.cut_strip_half_height,
                    dark_threshold=args.cut_ink_threshold,
                    max_ink_density=args.max_cut_ink_density,
                )
            )
            if not specs:
                planning_fallback_reason = "no_safe_crop_plan"
        except Exception as exc:
            planning_fallback_reason = f"safe_crop_planning_failed:{type(exc).__name__}:{exc}"

    if planning_fallback_reason is not None:
        specs = [
            _full_image_group_spec(
                image,
                detected_lines,
                group_id="GFULL",
            )
        ]
        aspect_ratio = float(image.height) / float(max(1, image.width))
        crop_plan = {
            "requested_crops": args.crops,
            "initial_effective_crops": 1,
            "effective_crops": 1,
            "status": "fallback_full_image",
            "fallback_reason": planning_fallback_reason,
            "image_width": image.width,
            "image_height": image.height,
            "image_aspect_ratio_h_over_w": round(aspect_ratio, 6),
            "single_crop_max_aspect_ratio": (args.single_crop_max_aspect_ratio),
        }

    group_plan = {
        "method": "aspect_ratio_adaptive_paddle_snapped_or_full_image_fallback",
        "paddle_text_used": False,
        "matching_used": False,
        "fuzzy_merge_used": False,
        "post_transcription_validation_used": False,
        "non_overlapping": True,
        "requested_crop_count_is_maximum": True,
        "detected_line_count_is_diagnostic_only": True,
        "crop_plan": crop_plan,
        "full_width_crops": args.full_width_crops,
        "verified_boundary_count": len(verified_boundaries),
        "boundary_decisions": boundary_decisions,
        "groups": [
            {
                "group_id": spec.group_id,
                "line_indices": list(spec.line_indices),
                "paddle_detected_line_estimate": len(spec.line_indices),
                "bbox": [spec.left, spec.top, spec.right, spec.bottom],
            }
            for spec in specs
        ],
    }
    write_json(receipt_dir / "10_qwen_safe_cut_groups.json", group_plan)
    write_json(receipt_dir / "10_qwen_line_groups.json", group_plan)

    plan_label = (
        "whole-image fallback"
        if crop_plan.get("status") == "fallback_full_image"
        else "non-overlapping crop"
    )
    print(
        f"    Qwen: {len(specs)} {plan_label}{'s' if len(specs) != 1 else ''}, "
        f"parallelism={args.qwen_group_parallelism}",
        flush=True,
    )

    def run_spec(spec: LineGroupSpec):
        return _transcribe_qwen_group_with_retries(args, spec, work_dir)

    max_workers = min(len(specs), max(1, args.qwen_group_parallelism))
    completed_results: list[
        tuple[
            list[GroupCallResult],
            list[GroupCallResult],
            list[dict[str, Any]],
        ]
    ] = []
    if max_workers == 1:
        completed_results = [run_spec(spec) for spec in specs]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_spec = {executor.submit(run_spec, spec): spec for spec in specs}
            for future in concurrent.futures.as_completed(future_to_spec):
                spec = future_to_spec[future]
                completed = future.result()
                completed_results.append(completed)
                accepted, _, _ = completed
                print(
                    f"      crop {spec.group_id} {'accepted' if accepted else 'returned no text'}",
                    flush=True,
                )

    accepted_chunks: list[GroupCallResult] = []
    all_call_results: list[GroupCallResult] = []
    group_diagnostics: list[dict[str, Any]] = []
    all_planned_calls_succeeded = True
    for accepted, call_results, diagnostics in completed_results:
        if not accepted:
            all_planned_calls_succeeded = False
        accepted_chunks.extend(accepted)
        all_call_results.extend(call_results)
        group_diagnostics.extend(diagnostics)

    runtime_full_image_fallback = False
    if not all_planned_calls_succeeded and not specs[0].group_id.startswith("GFULL"):
        runtime_full_image_fallback = True
        group_diagnostics.append(
            {
                "status": "discarded_partial_crop_transcription",
                "reason": "one_or_more_crop_calls_failed",
            }
        )
        fallback_spec = _full_image_group_spec(
            image,
            detected_lines,
            group_id="GFULL_RUNTIME",
        )
        fallback_accepted, fallback_calls, fallback_diagnostics = (
            _transcribe_qwen_group_with_retries(
                args,
                fallback_spec,
                work_dir,
            )
        )
        all_call_results.extend(fallback_calls)
        group_diagnostics.extend(fallback_diagnostics)
        accepted_chunks = fallback_accepted

    if not accepted_chunks:
        raise RuntimeError(
            "Qwen produced no nonempty transcription, including whole-image fallback"
        )

    accepted_chunks.sort(key=lambda result: (result.spec.top, result.spec.bottom))
    accepted_report = [
        {
            "group_id": chunk.spec.group_id,
            "paddle_detected_line_estimate": len(chunk.spec.line_indices),
            "qwen_returned_line_count": len(chunk.lines),
            "transcription_text_source": chunk.metrics.get("transcription_text_source"),
            "bbox": [
                chunk.spec.left,
                chunk.spec.top,
                chunk.spec.right,
                chunk.spec.bottom,
            ],
            "lines": list(chunk.lines),
            "attempt": chunk.attempt,
            "accepted_without_post_transcription_validation": True,
        }
        for chunk in accepted_chunks
    ]

    report = {
        "method": "ordered_qwen_transcription_without_post_validation",
        "matching_used": False,
        "paddle_text_used": False,
        "fuzzy_merge_used": False,
        "individual_line_alignment_used": False,
        "line_count_is_acceptance_contract": False,
        "post_transcription_validation_used": False,
        "detector_backend": detector_metadata.get("backend"),
        "detector_error": detector_error,
        "detected_box_count": len(boxes),
        "detected_line_count_estimate": len(detected_lines),
        "initial_group_count": len(specs),
        "accepted_chunk_count": len(accepted_chunks),
        "qwen_call_count": len(all_call_results),
        "planning_full_image_fallback": (crop_plan.get("status") == "fallback_full_image"),
        "runtime_full_image_fallback": runtime_full_image_fallback,
        "accepted_chunks": accepted_report,
        "group_diagnostics": group_diagnostics,
        "boundary_decisions": boundary_decisions,
        "crop_plan": crop_plan,
    }
    write_json(receipt_dir / "10_qwen_safe_cut_report.json", report)
    write_json(receipt_dir / "10_qwen_alignment_report.json", report)

    concatenated_lines: list[str] = []
    for chunk in accepted_chunks:
        concatenated_lines.extend(chunk.lines)
    if not concatenated_lines:
        raise RuntimeError("Concatenated Qwen transcription is empty")

    transcription = serialize_receipt(concatenated_lines)
    rows, protocol_warnings = parse_transcription_rows(transcription)
    used_full_image = any(chunk.spec.group_id.startswith("GFULL") for chunk in accepted_chunks)
    merge_report = {
        "method": (
            "single_whole_image_qwen_transcription"
            if used_full_image
            else "ordered_non_overlapping_crop_concatenation"
        ),
        "matching_used": False,
        "fuzzy_merge_used": False,
        "post_transcription_validation_used": False,
        "note": (
            "Qwen text was accepted without line-count, duplication, protocol, or "
            "semantic validation and passed directly to Gemma."
        ),
    }
    metrics = aggregate_group_metrics(all_call_results, detector_metadata)
    metrics["wall_duration_seconds"] = round(
        time.perf_counter() - phase_started,
        3,
    )

    return {
        "status": "completed",
        "transcription_status": "completed",
        "model": args.vlm_model,
        "image": str(image_path),
        "transcription": transcription,
        "row_count": len(rows),
        "protocol_warnings": protocol_warnings,
        "detector": detector_report,
        "alignment_report": report,
        "safe_cut_report": report,
        "merge_report": merge_report,
        "request": {
            "prompt_template": QWEN_TRANSCRIPTION_PROMPT_TEMPLATE,
            "think": False,
            "temperature": args.vlm_temperature,
            "seed": args.vlm_seed,
            "num_ctx": args.vlm_num_ctx,
            "num_predict": args.vlm_num_predict,
            "requested_crops": args.crops,
            "effective_crops": crop_plan.get("effective_crops"),
            "target_rows_per_crop": args.target_rows_per_crop,
            "single_crop_max_rows": args.single_crop_max_rows,
            "single_crop_max_aspect_ratio": (args.single_crop_max_aspect_ratio),
            "minimum_lines_per_crop": args.min_lines_per_crop,
            "safe_cut_search_ratio": args.safe_cut_search_ratio,
            "max_safe_cut_search_ratio": args.max_safe_cut_search_ratio,
            "line_group_overlap": 0,
            "full_width_crops": args.full_width_crops,
            "min_safe_gap": args.min_safe_gap,
            "cut_search_margin": args.cut_search_margin,
            "cut_strip_half_height": args.cut_strip_half_height,
            "cut_ink_threshold": args.cut_ink_threshold,
            "max_cut_ink_density": args.max_cut_ink_density,
            "qwen_group_parallelism": max_workers,
            "qwen_group_retries": args.qwen_group_retries,
            "post_transcription_validation_used": False,
        },
        "metrics": metrics,
    }


def invoke_qwen_transcription_adaptive(
    args: argparse.Namespace,
    image_path: Path,
    receipt_dir: Path,
) -> dict[str, Any]:
    # Compatibility orchestration name. The implementation uses Paddle only to
    # choose between-line cut positions, then concatenates accepted Qwen crops.
    return invoke_paddle_qwen_transcription(
        args,
        image_path,
        receipt_dir,
    )


def invoke_gemma_task(
    args: argparse.Namespace,
    *,
    task_name: str,
    question: str,
    schema: dict[str, Any],
    evidence: str,
    num_predict: int,
    think: bool = False,
) -> dict[str, Any]:
    prompt = (
        f"{question}\n\n"
        "Required JSON schema:\n"
        f"{json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}"
        "\n\n"
        "----- BEGIN RECEIPT EVIDENCE -----\n"
        f"{evidence}\n"
        "----- END RECEIPT EVIDENCE -----"
    )

    payload = {
        "model": args.gemma_model,
        "messages": [
            {
                "role": "system",
                "content": GEMMA_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "stream": False,
        "think": think,
        "format": schema,
        "keep_alive": args.gemma_keep_alive,
        "options": {
            "temperature": args.temperature,
            "seed": args.gemma_seed,
            "num_ctx": args.gemma_num_ctx,
            "num_predict": num_predict,
        },
    }

    started = time.perf_counter()
    response = post_json(
        f"{args.ollama_url.rstrip('/')}/api/chat",
        payload,
        args.gemma_timeout,
    )
    wall_seconds = time.perf_counter() - started

    message = response.get("message")
    if not isinstance(message, dict):
        raise RuntimeError(f"Gemma task {task_name!r} response has no message object")

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Gemma task {task_name!r} returned empty content")

    try:
        answer = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Gemma task {task_name!r} returned invalid JSON: {content[:1000]}"
        ) from exc

    return {
        "status": "completed",
        "task": task_name,
        "model": str(response.get("model") or args.gemma_model),
        "answer": answer,
        "raw_model_content": content,
        "thinking": message.get("thinking"),
        "request": {
            "question": question,
            "schema": schema,
            "think": think,
            "temperature": args.temperature,
            "seed": args.gemma_seed,
            "num_ctx": args.gemma_num_ctx,
            "num_predict": num_predict,
        },
        "metrics": response_metrics(response, wall_seconds),
        "raw_api_response": response,
    }


def unload_model(
    args: argparse.Namespace,
    model: str,
    timeout: float = 30.0,
) -> None:
    post_json(
        f"{args.ollama_url.rstrip('/')}/api/generate",
        {
            "model": model,
            "prompt": "",
            "stream": False,
            "keep_alive": 0,
        },
        timeout,
    )


def format_rows(rows: Sequence[dict[str, str]]) -> str:
    return "\n".join(f"{row['row_id']} :: {row['text']}" for row in rows)


def completed_answer(result: Any) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    if result.get("status") != "completed":
        return None

    answer = result.get("answer")
    return answer if isinstance(answer, dict) else None


def run_scalar_specialists(
    *,
    args: argparse.Namespace,
    image_path: Path,
    receipt_dir: Path,
    transcription: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    selected_tasks = [] if args.skip_scalars else unique_ordered(args.scalar_tasks)
    results: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    def run_one(
        task_index: int,
        task_name: str,
    ) -> tuple[str, dict[str, Any], dict[str, Any] | None, str]:
        task = SCALAR_TASKS[task_name]
        result_path = receipt_dir / f"{20 + task_index:02d}_gemma_scalar_{task_name}.json"
        error_path = receipt_dir / f"{20 + task_index:02d}_gemma_scalar_{task_name}_error.json"

        try:
            if result_path.exists() and not args.overwrite:
                return (
                    task_name,
                    read_json(result_path),
                    None,
                    "cached",
                )

            result = invoke_gemma_task(
                args,
                task_name=f"scalar_{task_name}",
                question=task["question"],
                schema=task["schema"],
                evidence=transcription,
                num_predict=task["num_predict"],
            )
            write_json(result_path, result)

            if error_path.exists():
                error_path.unlink()

            return task_name, result, None, "completed"
        except Exception as exc:
            failure = {
                "image": str(image_path),
                "stage": f"scalar_{task_name}",
                "task": task_name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            write_json(error_path, failure)
            return (
                task_name,
                {
                    "status": "error",
                    **failure,
                },
                failure,
                "error",
            )

    for task_name in selected_tasks:
        print(
            f"    queued scalar: {task_name}",
            flush=True,
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.gemma_parallelism)
    ) as executor:
        future_to_task = {
            executor.submit(
                run_one,
                index,
                task_name,
            ): task_name
            for index, task_name in enumerate(
                selected_tasks,
                start=1,
            )
        }

        for future in concurrent.futures.as_completed(future_to_task):
            task_name = future_to_task[future]

            try:
                (
                    completed_task,
                    result,
                    failure,
                    outcome,
                ) = future.result()
            except Exception as exc:
                failure = {
                    "image": str(image_path),
                    "stage": f"scalar_{task_name}",
                    "task": task_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                completed_task = task_name
                result = {
                    "status": "error",
                    **failure,
                }
                outcome = "error"

            results[completed_task] = result

            if failure is not None:
                failures.append(failure)
                print(
                    f"    done scalar: {completed_task} | "
                    f"ERROR {failure['error_type']}: "
                    f"{failure['error']}",
                    file=sys.stderr,
                )
            elif outcome == "cached":
                print(f"    done scalar: {completed_task} | cached")
            else:
                metrics = result.get("metrics", {})
                print(
                    f"    done scalar: {completed_task} | "
                    f"tokens={metrics.get('prompt_eval_count')}/"
                    f"{metrics.get('eval_count')} "
                    f"wall={metrics.get('wall_duration_seconds')}s"
                )

    return results, failures


def validate_direct_items(answer: Any) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "item_count": 0,
        "items_with_price": 0,
        "items_with_quantity": 0,
        "items_with_unit": 0,
        "extracted_price_sum": None,
    }

    if not isinstance(answer, dict):
        errors.append(
            {
                "code": "ANSWER_NOT_OBJECT",
                "message": "The direct item answer is not a JSON object.",
            }
        )
        return {
            "status": "invalid",
            "errors": errors,
            "warnings": warnings,
            "metrics": metrics,
        }

    items = answer.get("items")
    if not isinstance(items, list):
        errors.append(
            {
                "code": "ITEMS_NOT_ARRAY",
                "message": "The items field is not an array.",
            }
        )
        return {
            "status": "invalid",
            "errors": errors,
            "warnings": warnings,
            "metrics": metrics,
        }

    price_sum = 0.0
    seen: set[tuple[str, float | None, float | None, str | None]] = set()

    for index, item in enumerate(items):
        location = f"items[{index}]"
        if not isinstance(item, dict):
            errors.append(
                {
                    "code": "ITEM_NOT_OBJECT",
                    "location": location,
                    "message": "Item is not an object.",
                }
            )
            continue

        name = item.get("name")
        final_price = item.get("final_price")
        quantity = item.get("quantity")
        unit = item.get("unit")

        if not isinstance(name, str) or not name.strip():
            errors.append(
                {
                    "code": "INVALID_ITEM_NAME",
                    "location": f"{location}.name",
                    "message": "Item name must be a non-empty string.",
                }
            )
            normalized_name = ""
        else:
            normalized_name = " ".join(name.casefold().split())

        if final_price is not None and (
            isinstance(final_price, bool) or not isinstance(final_price, (int, float))
        ):
            errors.append(
                {
                    "code": "INVALID_FINAL_PRICE_TYPE",
                    "location": f"{location}.final_price",
                    "message": "final_price must be a number or null.",
                }
            )
        elif isinstance(final_price, (int, float)):
            if final_price < 0:
                errors.append(
                    {
                        "code": "NEGATIVE_FINAL_PRICE",
                        "location": f"{location}.final_price",
                        "message": "final_price cannot be negative.",
                    }
                )
            else:
                metrics["items_with_price"] += 1
                price_sum += float(final_price)
        else:
            warnings.append(
                {
                    "code": "MISSING_FINAL_PRICE",
                    "location": f"{location}.final_price",
                    "message": "The model returned no final price for this item.",
                }
            )

        if quantity is not None and (
            isinstance(quantity, bool) or not isinstance(quantity, (int, float))
        ):
            errors.append(
                {
                    "code": "INVALID_QUANTITY_TYPE",
                    "location": f"{location}.quantity",
                    "message": "quantity must be a number or null.",
                }
            )
        elif isinstance(quantity, (int, float)):
            if quantity < 0:
                errors.append(
                    {
                        "code": "NEGATIVE_QUANTITY",
                        "location": f"{location}.quantity",
                        "message": "quantity cannot be negative.",
                    }
                )
            else:
                metrics["items_with_quantity"] += 1

        if unit is not None and (not isinstance(unit, str) or not unit.strip()):
            errors.append(
                {
                    "code": "INVALID_UNIT",
                    "location": f"{location}.unit",
                    "message": "unit must be a non-empty string or null.",
                }
            )
        elif isinstance(unit, str):
            metrics["items_with_unit"] += 1

        if quantity is None and unit is not None:
            warnings.append(
                {
                    "code": "UNIT_WITHOUT_QUANTITY",
                    "location": location,
                    "message": "A unit was returned while quantity is null.",
                }
            )

        key = (
            normalized_name,
            float(final_price) if isinstance(final_price, (int, float)) else None,
            float(quantity) if isinstance(quantity, (int, float)) else None,
            unit.casefold().strip() if isinstance(unit, str) else None,
        )
        if normalized_name and key in seen:
            warnings.append(
                {
                    "code": "EXACT_DUPLICATE_ITEM",
                    "location": location,
                    "message": "An identical item object appears more than once.",
                }
            )
        seen.add(key)

    metrics["item_count"] = len(items)
    metrics["extracted_price_sum"] = round(price_sum, 2)

    status = "invalid" if errors else ("valid_with_warnings" if warnings else "valid")
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "metrics": metrics,
    }


def run_item_pipeline(
    *,
    args: argparse.Namespace,
    image_path: Path,
    receipt_dir: Path,
    transcription: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result_path = receipt_dir / "60_gemma_direct_items.json"
    validation_path = receipt_dir / "61_gemma_direct_items_validation.json"
    combined_path = receipt_dir / "63_item_pipeline_combined.json"
    error_path = receipt_dir / "60_gemma_direct_items_error.json"
    failures: list[dict[str, Any]] = []

    try:
        prompt_path = receipt_dir / "59_gemma_direct_items_prompt.txt"
        prompt_path.write_text(
            DIRECT_ITEMS_QUESTION
            + "\n\nRequired JSON schema:\n"
            + json.dumps(DIRECT_ITEMS_SCHEMA, ensure_ascii=False, indent=2)
            + "\n\n----- BEGIN RECEIPT EVIDENCE -----\n"
            + transcription
            + "\n----- END RECEIPT EVIDENCE -----\n",
            encoding="utf-8",
        )

        if result_path.exists() and not args.overwrite:
            direct_result = read_json(result_path)
        else:
            direct_result = invoke_gemma_task(
                args,
                task_name="direct_receipt_items",
                question=DIRECT_ITEMS_QUESTION,
                schema=DIRECT_ITEMS_SCHEMA,
                evidence=transcription,
                num_predict=args.item_num_predict,
                think=False,
            )
            write_json(result_path, direct_result)
            if error_path.exists():
                error_path.unlink()

        thinking = direct_result.get("thinking")
        if isinstance(thinking, str) and thinking.strip():
            (receipt_dir / "60_gemma_direct_items_thinking.txt").write_text(
                thinking.strip() + "\n",
                encoding="utf-8",
            )

        answer = completed_answer(direct_result)
        validation = validate_direct_items(answer)
        write_json(validation_path, validation)

        if answer is None:
            raise RuntimeError("Direct Gemma item result has no completed answer")

        if validation.get("status") == "invalid":
            failures.append(
                {
                    "image": str(image_path),
                    "stage": "direct_item_validation",
                    "error_type": "ValidationError",
                    "error": "Direct Gemma item output failed contract validation",
                    "validation": validation,
                }
            )

        result = {
            "status": ("completed" if not failures else "completed_with_errors"),
            "strategy": "complete_receipt_direct_item_extraction",
            "items": answer.get("items"),
            "direct_result": direct_result,
            "validation": validation,
            "failure_count": len(failures),
        }
        write_json(combined_path, result)
        return result, failures

    except Exception as exc:
        failure = {
            "image": str(image_path),
            "stage": "direct_item_extraction",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        failures.append(failure)
        write_json(error_path, failure)

        result = {
            "status": "error",
            "strategy": "complete_receipt_direct_item_extraction",
            "items": None,
            "validation": None,
            "failure_count": len(failures),
        }
        write_json(combined_path, result)
        return result, failures


def scalar_answer(
    scalar_results: dict[str, dict[str, Any]],
    task_name: str,
) -> dict[str, Any] | None:
    return completed_answer(scalar_results.get(task_name))


def scalar_field(
    scalar_results: dict[str, dict[str, Any]],
    task_name: str,
    field_name: str,
) -> Any:
    answer = scalar_answer(
        scalar_results,
        task_name,
    )
    return answer.get(field_name) if answer else None


def derive_semantic_status(
    *,
    qwen_result: dict[str, Any],
    scalar_results: dict[str, dict[str, Any]],
    missing_scalar_tasks: list[str],
    item_pipeline_result: dict[str, Any] | None,
    item_pipeline_enabled: bool,
) -> dict[str, Any]:
    reasons: list[dict[str, Any]] = []

    transcription_status = qwen_result.get("transcription_status")
    transcription_text = qwen_result.get("transcription")
    if (
        qwen_result.get("status") != "completed"
        or transcription_status not in {"completed", "safe"}
        or not isinstance(transcription_text, str)
        or not transcription_text.strip()
    ):
        reasons.append(
            {
                "code": "TRANSCRIPTION_MISSING",
                "message": "Qwen did not provide a nonempty completed transcription.",
                "details": {
                    "status": qwen_result.get("status"),
                    "transcription_status": transcription_status,
                },
            }
        )

    if missing_scalar_tasks:
        reasons.append(
            {
                "code": "SCALAR_TASK_FAILURE",
                "message": "One or more scalar specialist tasks failed or are missing.",
                "tasks": missing_scalar_tasks,
            }
        )

    if item_pipeline_enabled:
        if not isinstance(item_pipeline_result, dict):
            reasons.append(
                {
                    "code": "ITEM_PIPELINE_MISSING",
                    "message": "Item pipeline did not produce a result object.",
                }
            )
        else:
            pipeline_status = item_pipeline_result.get("status")
            if pipeline_status != "completed":
                reasons.append(
                    {
                        "code": "ITEM_PIPELINE_NOT_CLEAN",
                        "message": "Item pipeline did not complete cleanly.",
                        "status": pipeline_status,
                    }
                )

            validation = item_pipeline_result.get("validation")
            validation = validation if isinstance(validation, dict) else {}
            validation_status = validation.get("status")
            if validation_status != "valid":
                reasons.append(
                    {
                        "code": "ITEM_VALIDATION_REVIEW",
                        "message": "Direct item extraction produced warnings or errors.",
                        "validation_status": validation_status,
                        "warning_count": len(validation.get("warnings") or []),
                        "error_count": len(validation.get("errors") or []),
                    }
                )

            metrics = validation.get("metrics")
            metrics = metrics if isinstance(metrics, dict) else {}
            if metrics.get("item_count") == 0:
                reasons.append(
                    {
                        "code": "ZERO_ITEMS",
                        "message": "No purchased items were extracted.",
                    }
                )

            items = item_pipeline_result.get("items")
            items = items if isinstance(items, list) else []
            numeric_prices = [
                item.get("final_price")
                for item in items
                if isinstance(item, dict)
                and isinstance(item.get("final_price"), (int, float))
                and not isinstance(item.get("final_price"), bool)
            ]
            final_total_answer = scalar_answer(
                scalar_results,
                "final_purchase_total",
            )
            final_total = (
                final_total_answer.get("final_purchase_total")
                if isinstance(final_total_answer, dict)
                else None
            )
            if (
                items
                and len(numeric_prices) == len(items)
                and isinstance(final_total, (int, float))
                and not isinstance(final_total, bool)
            ):
                item_sum = round(sum(float(value) for value in numeric_prices), 2)
                difference = round(item_sum - float(final_total), 2)
                if abs(difference) > 0.02:
                    reasons.append(
                        {
                            "code": "ITEM_SUM_FINAL_TOTAL_MISMATCH",
                            "message": (
                                "The sum of model-extracted item final prices differs "
                                "from the model-extracted final purchase total. Receipt-"
                                "level discounts may explain this, so review is required "
                                "rather than deterministic correction."
                            ),
                            "item_sum": item_sum,
                            "final_purchase_total": final_total,
                            "difference": difference,
                        }
                    )

    return {
        "status": "accepted" if not reasons else "review_required",
        "reasons": reasons,
    }


def assemble_final_receipt(
    *,
    args: argparse.Namespace,
    image_path: Path,
    transcription: str,
    qwen_result: dict[str, Any],
    scalar_results: dict[str, dict[str, Any]],
    item_pipeline_result: dict[str, Any] | None,
) -> dict[str, Any]:
    selected_scalar_tasks = [] if args.skip_scalars else unique_ordered(args.scalar_tasks)

    missing_scalar_tasks = [
        task_name
        for task_name in selected_scalar_tasks
        if scalar_answer(
            scalar_results,
            task_name,
        )
        is None
    ]

    vat_lines_answer = scalar_answer(
        scalar_results,
        "vat_lines",
    )

    items = (
        item_pipeline_result.get("items")
        if isinstance(
            item_pipeline_result,
            dict,
        )
        else None
    )

    receipt = {
        "merchant": {
            "name": scalar_field(
                scalar_results,
                "merchant_name",
                "merchant_name",
            ),
            "address": scalar_answer(
                scalar_results,
                "merchant_address",
            ),
        },
        "receipt_metadata": {
            "date": scalar_field(
                scalar_results,
                "receipt_date",
                "receipt_date",
            ),
            "time": scalar_field(
                scalar_results,
                "receipt_time",
                "receipt_time",
            ),
            "receipt_number": scalar_field(
                scalar_results,
                "receipt_number",
                "receipt_number",
            ),
            "currency": scalar_field(
                scalar_results,
                "currency",
                "currency",
            ),
        },
        "items": items,
        "totals": {
            "final_purchase_total": scalar_answer(
                scalar_results,
                "final_purchase_total",
            ),
            "pre_discount_total": scalar_answer(
                scalar_results,
                "pre_discount_total",
            ),
            "net_amount": scalar_answer(
                scalar_results,
                "net_amount",
            ),
        },
        "discount": {
            "discount_total": scalar_answer(
                scalar_results,
                "discount_total",
            ),
        },
        "payment": {
            "payment_method": scalar_field(
                scalar_results,
                "payment_method",
                "payment_method",
            ),
            "payment_received": scalar_answer(
                scalar_results,
                "payment_received",
            ),
            "change_returned": scalar_answer(
                scalar_results,
                "change_returned",
            ),
        },
        "transaction_status": scalar_field(
            scalar_results,
            "transaction_status",
            "transaction_status",
        ),
        "tax": {
            "vat_amount": scalar_answer(
                scalar_results,
                "vat_amount",
            ),
            "vat_lines": (vat_lines_answer.get("vat_lines") if vat_lines_answer else None),
        },
    }

    item_pipeline_complete = args.skip_item_pipeline or (
        isinstance(
            item_pipeline_result,
            dict,
        )
        and item_pipeline_result.get("status")
        in {
            "completed",
            "completed_with_errors",
        }
    )

    semantic_status = derive_semantic_status(
        qwen_result=qwen_result,
        scalar_results=scalar_results,
        missing_scalar_tasks=missing_scalar_tasks,
        item_pipeline_result=item_pipeline_result,
        item_pipeline_enabled=not args.skip_item_pipeline,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "source": {
            "image": str(image_path),
            "image_name": image_path.name,
        },
        "models": {
            "text_detection": (
                qwen_result.get("detector", {}).get("metadata", {}).get("backend")
                if isinstance(qwen_result.get("detector"), dict)
                else None
            ),
            "visual_transcription": args.vlm_model,
            "semantic_specialists": args.gemma_model,
        },
        "experiment": {
            "visual_source": ("paddle_geometry_when_available_plus_qwen_transcription"),
            "transcription_strategy": "aspect_ratio_adaptive_crops_or_whole_image_without_qwen_validation_with_thinking_field_compatibility",
            "scalar_strategy": "semantically_named_micro_specialists",
            "item_strategy": "complete_receipt_direct_item_extraction",
            "communication_strategy": "structured_state_handoff",
            "assembly_strategy": "direct_copy_only",
            "deterministic_semantic_correction": False,
            "arithmetic_reconciliation": False,
            "cross_specialist_conflict_resolution": False,
        },
        "transcription": {
            "text": transcription,
            "characters": len(transcription),
            "qwen_metrics": qwen_result.get("metrics"),
            "protocol_warnings": qwen_result.get("protocol_warnings"),
            "transcription_status": qwen_result.get("transcription_status"),
            "detector": qwen_result.get("detector"),
            "alignment_report": qwen_result.get("alignment_report"),
        },
        "receipt": receipt,
        "semantic_status": semantic_status,
        "execution_status": "completed",
        "scalar_results": {
            task_name: (
                result.get("answer")
                if isinstance(result, dict) and result.get("status") == "completed"
                else result
            )
            for task_name, result in scalar_results.items()
        },
        "scalar_metrics": {
            task_name: (result.get("metrics") if isinstance(result, dict) else None)
            for task_name, result in scalar_results.items()
        },
        "item_pipeline": item_pipeline_result,
        "assembly": {
            "complete": (not missing_scalar_tasks and item_pipeline_complete),
            "missing_or_failed_scalar_tasks": (missing_scalar_tasks),
            "item_pipeline_enabled": (not args.skip_item_pipeline),
            "note": (
                "Model outputs were handed to later model stages and copied "
                "without deterministic semantic correction."
            ),
        },
    }


def run_qwen_phase(
    *,
    args: argparse.Namespace,
    entries: list[tuple[Path, Path]],
    input_root: Path,
) -> tuple[
    list[tuple[Path, Path, str, dict[str, Any]]],
    list[dict[str, Any]],
]:
    completed: list[tuple[Path, Path, str, dict[str, Any]]] = []
    failures: list[dict[str, Any]] = []

    print("\nPhase 1/2: aspect-ratio crops or whole-image fallback + Qwen3.5")

    for index, (image_path, receipt_dir) in enumerate(
        entries,
        start=1,
    ):
        receipt_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[{index}/{len(entries)}] {image_path.name}",
            flush=True,
        )

        source_file = receipt_dir / "00_source.json"
        qwen_file = receipt_dir / "10_qwen_transcription.json"
        transcription_file = receipt_dir / "11_transcription.txt"

        write_json(
            source_file,
            {
                "image": str(image_path),
                "relative_path": str(image_path.resolve().relative_to(input_root.resolve())),
                "size_bytes": image_path.stat().st_size,
                "modified_at_utc": datetime.fromtimestamp(
                    image_path.stat().st_mtime,
                    tz=UTC,
                ).isoformat(),
            },
        )

        try:
            if qwen_file.exists() and transcription_file.exists() and not args.overwrite:
                qwen_result = read_json(qwen_file)
                transcription = transcription_file.read_text(encoding="utf-8").strip()

                if not transcription:
                    raise RuntimeError("Cached Qwen transcription is empty")

                if qwen_result.get("status") != "completed":
                    raise RuntimeError(
                        "Cached Qwen result is not completed; rerun with --overwrite."
                    )
                print("    cached")
            else:
                qwen_result = invoke_qwen_transcription_adaptive(
                    args,
                    image_path,
                    receipt_dir,
                )
                transcription = qwen_result["transcription"]

                write_json(qwen_file, qwen_result)
                transcription_file.write_text(
                    transcription + "\n",
                    encoding="utf-8",
                )

                metrics = qwen_result.get("metrics", {})
                print(
                    "    OK "
                    f"rows={qwen_result.get('row_count')} "
                    f"chars={len(transcription)} "
                    f"tokens={metrics.get('prompt_eval_count')}/"
                    f"{metrics.get('eval_count')} "
                    f"wall={metrics.get('wall_duration_seconds')}s"
                )

            completed.append(
                (
                    image_path,
                    receipt_dir,
                    transcription,
                    qwen_result,
                )
            )
        except Exception as exc:
            failure = {
                "image": str(image_path),
                "stage": "qwen_transcription",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)

            write_json(
                receipt_dir / "10_qwen_transcription_error.json",
                failure,
            )
            print(
                f"    ERROR {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

            if not args.continue_on_error:
                break

    return completed, failures


def summarize_final(
    image_path: Path,
    receipt_dir: Path,
    final_receipt: dict[str, Any],
) -> dict[str, Any]:
    receipt = final_receipt.get("receipt")
    receipt = receipt if isinstance(receipt, dict) else {}

    merchant = receipt.get("merchant")
    merchant = merchant if isinstance(merchant, dict) else {}

    items = receipt.get("items")
    item_count = len(items) if isinstance(items, list) else None

    totals = receipt.get("totals")
    totals = totals if isinstance(totals, dict) else {}

    final_total = totals.get("final_purchase_total")
    final_total = final_total if isinstance(final_total, dict) else {}

    metadata = receipt.get("receipt_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}

    item_pipeline = final_receipt.get("item_pipeline")
    item_pipeline = item_pipeline if isinstance(item_pipeline, dict) else {}

    semantic = final_receipt.get("semantic_status")
    semantic = semantic if isinstance(semantic, dict) else {}

    transcription = final_receipt.get("transcription")
    transcription = transcription if isinstance(transcription, dict) else {}

    return {
        "image": str(image_path),
        "receipt_dir": str(receipt_dir),
        "final_json": str(receipt_dir / "90_receipt_combined_final.json"),
        "assembly_complete": (
            final_receipt.get("assembly", {}).get("complete")
            if isinstance(
                final_receipt.get("assembly"),
                dict,
            )
            else False
        ),
        "merchant_name": merchant.get("name"),
        "item_count": item_count,
        "item_pipeline_status": item_pipeline.get("status"),
        "item_pipeline_failure_count": item_pipeline.get("failure_count"),
        "final_purchase_total": final_total.get("final_purchase_total"),
        "currency": (final_total.get("currency") or metadata.get("currency")),
        "transcription_status": transcription.get("transcription_status"),
        "semantic_status": semantic.get("status"),
        "semantic_reason_count": len(semantic.get("reasons") or []),
        "transaction_status": receipt.get("transaction_status"),
    }


def should_exclude_from_archive(path: Path) -> bool:
    name = path.name.casefold()
    suffix = path.suffix.casefold()
    if suffix in IMAGE_EXTENSIONS:
        return True
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}:
        return True
    if name.endswith(".zip"):
        return True
    return False


def archive_run_outputs(
    output_dir: Path,
    archive_path: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    archive_path = archive_path.resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "created_at_utc": utc_now(),
        "output_dir": str(output_dir),
        "archive_path": str(archive_path),
        "excluded_file_types": sorted(IMAGE_EXTENSIONS),
        "excluded_images": True,
        "files_added": 0,
        "files_excluded": 0,
        "bytes_added": 0,
    }

    if archive_path.exists():
        archive_path.unlink()

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*"), key=lambda item: item.as_posix()):
            if not path.is_file():
                continue
            if path.resolve() == archive_path:
                continue
            if should_exclude_from_archive(path):
                manifest["files_excluded"] += 1
                continue

            arcname = Path(output_dir.name) / path.relative_to(output_dir)
            archive.write(path, arcname.as_posix())
            manifest["files_added"] += 1
            manifest["bytes_added"] += path.stat().st_size

    manifest["archive_size_bytes"] = archive_path.stat().st_size
    write_json(output_dir / "archive_manifest.json", manifest)

    # Rebuild once so the manifest itself is inside the archive as well.
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*"), key=lambda item: item.as_posix()):
            if not path.is_file():
                continue
            if path.resolve() == archive_path:
                continue
            if should_exclude_from_archive(path):
                continue
            arcname = Path(output_dir.name) / path.relative_to(output_dir)
            archive.write(path, arcname.as_posix())

    manifest["archive_size_bytes"] = archive_path.stat().st_size
    write_json(output_dir / "archive_manifest.json", manifest)
    return manifest


def main() -> int:
    args = parse_args()

    if args.gemma_parallelism < 1:
        print(
            "Input error: --gemma-parallelism must be at least 1",
            file=sys.stderr,
        )
        return 2

    if args.row_context_radius < 0:
        print(
            "Input error: --row-context-radius cannot be negative",
            file=sys.stderr,
        )
        return 2

    if args.crop_scale <= 0 or args.crop_contrast <= 0:
        print(
            "Input error: crop scale and contrast must be positive",
            file=sys.stderr,
        )
        return 2
    if not 0.0 <= args.paddle_min_score <= 1.0:
        print("Input error: --paddle-min-score must be in [0, 1]", file=sys.stderr)
        return 2
    if not 0.0 <= args.paddle_line_overlap_threshold <= 1.0:
        print(
            "Input error: --paddle-line-overlap-threshold must be in [0, 1]",
            file=sys.stderr,
        )
        return 2
    if args.paddle_line_center_factor < 0:
        print(
            "Input error: --paddle-line-center-factor cannot be negative",
            file=sys.stderr,
        )
        return 2
    if args.paddle_min_lines < 1 or args.paddle_max_lines < args.paddle_min_lines:
        print(
            "Input error: invalid Paddle line-count limits",
            file=sys.stderr,
        )
        return 2
    if args.crops < 1:
        print("Input error: --crops must be at least 1", file=sys.stderr)
        return 2
    if args.target_rows_per_crop < 1:
        print(
            "Input error: --target-rows-per-crop must be at least 1",
            file=sys.stderr,
        )
        return 2
    if args.single_crop_max_rows < 1 or args.single_crop_max_aspect_ratio <= 0:
        print(
            "Input error: single-crop thresholds must be positive",
            file=sys.stderr,
        )
        return 2
    if not (0.0 <= args.safe_cut_search_ratio <= args.max_safe_cut_search_ratio <= 1.0):
        print(
            "Input error: safe-cut search ratios must satisfy 0 <= normal <= maximum <= 1",
            file=sys.stderr,
        )
        return 2
    if args.line_group_overlap != 0:
        print(
            "Input error: --line-group-overlap must be 0; the crop "
            "implementation; crops are concatenated without matching.",
            file=sys.stderr,
        )
        return 2
    if args.safe_cut_padding < 0:
        print(
            "Input error: --safe-cut-padding cannot be negative",
            file=sys.stderr,
        )
        return 2
    if args.min_safe_gap < 0:
        print(
            "Input error: --min-safe-gap cannot be negative",
            file=sys.stderr,
        )
        return 2
    if args.safe_cut_padding != 0:
        print(
            "Input error: --safe-cut-padding must be 0; adjacent crops "
            "are deliberately non-overlapping.",
            file=sys.stderr,
        )
        return 2
    if args.min_lines_per_crop < 1:
        print(
            "Input error: --min-lines-per-crop must be at least 1",
            file=sys.stderr,
        )
        return 2
    if args.cut_search_margin < 0 or args.cut_strip_half_height < 0:
        print("Input error: cut margins cannot be negative", file=sys.stderr)
        return 2
    if not 0 <= args.cut_ink_threshold <= 255:
        print("Input error: --cut-ink-threshold must be in [0, 255]", file=sys.stderr)
        return 2
    if not 0.0 <= args.max_cut_ink_density <= 1.0:
        print("Input error: --max-cut-ink-density must be in [0, 1]", file=sys.stderr)
        return 2
    if args.qwen_group_parallelism < 1:
        print(
            "Input error: --qwen-group-parallelism must be at least 1",
            file=sys.stderr,
        )
        return 2
    if args.qwen_group_retries < 0:
        print(
            "Input error: Qwen retries cannot be negative",
            file=sys.stderr,
        )
        return 2
    if args.item_num_predict < 1:
        print(
            "Input error: --item-num-predict must be at least 1",
            file=sys.stderr,
        )
        return 2

    try:
        images = discover_images(
            args.input,
            args.recursive,
        )
    except Exception as exc:
        print(
            f"Input error: {exc}",
            file=sys.stderr,
        )
        return 2

    if not images:
        print(
            f"Input error: no supported images found in {args.input}",
            file=sys.stderr,
        )
        return 2

    input_path = args.input.expanduser().resolve()
    input_root = input_path if input_path.is_dir() else input_path.parent

    run_name = args.run_name or timestamp_name()
    output_dir = args.output_dir.expanduser().resolve() / sanitize_name(run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = [
        (
            image_path,
            output_dir
            / receipt_key(
                image_path,
                input_root,
            ),
        )
        for image_path in images
    ]

    selected_scalar_tasks = [] if args.skip_scalars else unique_ordered(args.scalar_tasks)

    run_config = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "input": str(input_path),
        "output_dir": str(output_dir),
        "recursive": args.recursive,
        "ollama_url": args.ollama_url,
        "vlm_model": args.vlm_model,
        "gemma_model": args.gemma_model,
        "vlm_temperature": args.vlm_temperature,
        "vlm_seed": args.vlm_seed,
        "gemma_temperature": args.temperature,
        "gemma_seed": args.gemma_seed,
        "gemma_parallelism": args.gemma_parallelism,
        "row_context_radius": args.row_context_radius,
        "transcription_strategy": ("aspect_ratio_adaptive_safe_crops_with_whole_image_fallback"),
        "paddle_backend": args.paddle_backend,
        "paddle_det_model": args.paddle_det_model,
        "paddle_device": args.paddle_device,
        "paddle_lang": args.paddle_lang,
        "paddle_min_score": args.paddle_min_score,
        "paddle_line_overlap_threshold": (args.paddle_line_overlap_threshold),
        "paddle_line_center_factor": args.paddle_line_center_factor,
        "paddle_min_lines": args.paddle_min_lines,
        "paddle_max_lines": args.paddle_max_lines,
        "requested_crops": args.crops,
        "target_rows_per_crop": args.target_rows_per_crop,
        "single_crop_max_rows": args.single_crop_max_rows,
        "single_crop_max_aspect_ratio": args.single_crop_max_aspect_ratio,
        "safe_cut_search_ratio": args.safe_cut_search_ratio,
        "max_safe_cut_search_ratio": args.max_safe_cut_search_ratio,
        "min_lines_per_crop": args.min_lines_per_crop,
        "line_group_overlap": 0,
        "full_width_crops": args.full_width_crops,
        "safe_cut_padding": args.safe_cut_padding,
        "min_safe_gap": args.min_safe_gap,
        "cut_search_margin": args.cut_search_margin,
        "cut_strip_half_height": args.cut_strip_half_height,
        "cut_ink_threshold": args.cut_ink_threshold,
        "max_cut_ink_density": args.max_cut_ink_density,
        "line_group_horizontal_padding": (args.line_group_horizontal_padding),
        "crop_scale": args.crop_scale,
        "crop_contrast": args.crop_contrast,
        "crop_sharpen": args.crop_sharpen,
        "qwen_group_parallelism": args.qwen_group_parallelism,
        "qwen_group_retries": args.qwen_group_retries,
        "archive_enabled": args.archive,
        "archive_path": str(args.archive_path) if args.archive_path else None,
        "item_think": args.item_think,
        "item_num_predict": args.item_num_predict,
        "scalar_tasks": selected_scalar_tasks,
        "default_batch_input": str(DEFAULT_BATCH_INPUT),
        "default_batch_scalar_tasks": list(DEFAULT_BATCH_SCALAR_TASKS),
        "scalar_specialists_enabled": not args.skip_scalars,
        "item_pipeline_enabled": (not args.skip_item_pipeline),
        "item_pipeline_stages": [
            "complete_receipt_direct_item_extraction",
            "contract_validation",
        ],
        "deterministic_semantic_correction": False,
        "arithmetic_reconciliation": False,
        "cross_specialist_conflict_resolution": False,
        "assembly_strategy": "direct_copy_only",
    }
    write_json(
        output_dir / "run_config.json",
        run_config,
    )

    print(f"Images:               {len(entries)}")
    print(f"Qwen model:           {args.vlm_model}")
    print(f"Qwen decoding:        temperature={args.vlm_temperature}, seed={args.vlm_seed}")
    print(
        f"Paddle detector:      backend={args.paddle_backend}, "
        f"device={args.paddle_device}, lang={args.paddle_lang}"
    )
    print(
        f"Crop planning:        up to {args.crops} crop(s), "
        f"target_rows={args.target_rows_per_crop}, "
        f"single_crop<={args.single_crop_max_rows} rows and "
        f"H/W<={args.single_crop_max_aspect_ratio:.2f}"
    )
    print(
        f"Boundary snapping:    search={args.safe_cut_search_ratio:.2f}, "
        f"max={args.max_safe_cut_search_ratio:.2f}, "
        "non-overlapping, no text matching"
    )
    print(
        f"Crop width:           {'full image' if args.full_width_crops else 'detected text bounds'}"
    )
    print(f"Qwen group parallelism:{args.qwen_group_parallelism}")
    print(f"Gemma model:          {args.gemma_model}")
    print(f"Gemma decoding:       temperature={args.temperature}, seed={args.gemma_seed}")
    print(f"Gemma parallelism:    {args.gemma_parallelism}")
    print(f"Scalar specialists:   {'disabled' if args.skip_scalars else 'enabled'}")
    print(f"Item pipeline:        {'disabled' if args.skip_item_pipeline else 'enabled'}")
    print(f"Input:                {input_path}")
    print(f"Output:               {output_dir}")

    transcription_entries, failures = run_qwen_phase(
        args=args,
        entries=entries,
        input_root=input_root,
    )

    if transcription_entries and not args.keep_vlm_loaded and args.vlm_model != args.gemma_model:
        print(f"\nUnloading visual model: {args.vlm_model}")

        try:
            unload_model(
                args,
                args.vlm_model,
            )
        except Exception as exc:
            print(
                f"Warning: could not unload {args.vlm_model}: {exc}",
                file=sys.stderr,
            )

    print("\nPhase 2/2: Gemma scalar specialists and direct complete-receipt item extraction")

    receipt_summaries: list[dict[str, Any]] = []

    for index, (
        image_path,
        receipt_dir,
        transcription,
        qwen_result,
    ) in enumerate(
        transcription_entries,
        start=1,
    ):
        print(
            f"[{index}/{len(transcription_entries)}] {image_path.name}",
            flush=True,
        )

        scalar_results, scalar_failures = run_scalar_specialists(
            args=args,
            image_path=image_path,
            receipt_dir=receipt_dir,
            transcription=transcription,
        )
        failures.extend(scalar_failures)

        item_pipeline_result: dict[str, Any] | None = None
        item_failures: list[dict[str, Any]] = []

        if not args.skip_item_pipeline:
            item_pipeline_result, item_failures = run_item_pipeline(
                args=args,
                image_path=image_path,
                receipt_dir=receipt_dir,
                transcription=transcription,
            )
            failures.extend(item_failures)

        final_receipt = assemble_final_receipt(
            args=args,
            image_path=image_path,
            transcription=transcription,
            qwen_result=qwen_result,
            scalar_results=scalar_results,
            item_pipeline_result=item_pipeline_result,
        )
        write_json(
            receipt_dir / "90_receipt_combined_final.json",
            final_receipt,
        )

        receipt_summaries.append(
            summarize_final(
                image_path,
                receipt_dir,
                final_receipt,
            )
        )

        receipt_failures = scalar_failures + item_failures

        if receipt_failures and not args.continue_on_error:
            break

    archive_path = (
        args.archive_path.expanduser().resolve()
        if args.archive_path
        else output_dir.with_suffix(".zip")
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "completed_at_utc": utc_now(),
        "configuration": run_config,
        "summary": {
            "image_count": len(entries),
            "qwen_transcription_completed": len(transcription_entries),
            "final_receipts_created": len(receipt_summaries),
            "assembly_complete_count": sum(
                1 for result in receipt_summaries if result.get("assembly_complete")
            ),
            "semantic_accepted_count": sum(
                1 for result in receipt_summaries if result.get("semantic_status") == "accepted"
            ),
            "semantic_review_required_count": sum(
                1
                for result in receipt_summaries
                if result.get("semantic_status") == "review_required"
            ),
            "qwen_transcription_failed_or_unsafe_count": sum(
                1 for failure in failures if failure.get("stage") == "qwen_transcription"
            ),
            "failure_count": len(failures),
            "archive_enabled": bool(args.archive),
            "archive_path": str(archive_path) if args.archive else None,
        },
        "results": receipt_summaries,
        "failures": failures,
    }
    write_json(
        output_dir / "summary.json",
        summary,
    )

    archive_manifest = None
    if args.archive:
        archive_manifest = archive_run_outputs(output_dir, archive_path)
        summary["archive"] = archive_manifest
        write_json(
            output_dir / "summary.json",
            summary,
        )
        # Recreate the archive once more so the updated summary is included.
        # The on-disk archive_manifest.json may contain the final archive size,
        # while summary.json contains the prior manifest object. This avoids
        # rewriting summary after the final archive pass.
        archive_run_outputs(output_dir, archive_path)

    print("\nExperiment completed.")
    print(
        json.dumps(
            summary["summary"],
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Summary: {output_dir / 'summary.json'}")
    if args.archive:
        print(f"Archive: {archive_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
