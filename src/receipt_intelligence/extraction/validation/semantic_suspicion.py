"""Non-mutating semantic plausibility checks for extracted receipt items.

The checks in this module never reinterpret OCR rows and never modify receipt
fields. They only identify generic internal contradictions that justify an
evidence-backed semantic LLM review.
"""

from __future__ import annotations

import copy
import re
from typing import Any

SCHEMA_VERSION = "semantic_suspicion_1"


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalized_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9äöüß]+", " ", str(value or "").casefold()).strip()


def _description_conflicts(item: dict[str, Any]) -> bool:
    description = _normalized_text(item.get("description"))
    product_description = _normalized_text(item.get("product_description"))
    if not description or not product_description or description == product_description:
        return False
    if description in product_description or product_description in description:
        return False
    left = set(description.split())
    right = set(product_description.split())
    return not left.intersection(right)


def _semantic_issue(
    code: str,
    severity: str,
    message: str,
    *,
    item_index: int | None = None,
    **details: Any,
) -> dict[str, Any]:
    payload = {
        "code": code,
        "severity": severity,
        "message": message,
        "details": details,
        "source": "semantic_suspicion",
    }
    if item_index is not None:
        payload["item_index"] = item_index
        payload["details"] = {"item_index": item_index, **details}
    return payload


def evaluate_semantic_suspicion(
    receipt: dict[str, Any],
    validation_report: dict[str, Any] | None = None,
    *,
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """Return generic semantic warning signals without changing the receipt.

    A review is triggered by at least one high-severity signal or two independent
    medium-severity signals. Low-severity observations remain diagnostic only.
    """

    issues: list[dict[str, Any]] = []
    items = [item for item in (receipt.get("items") or []) if isinstance(item, dict)]

    for index, item in enumerate(items):
        description = str(item.get("description") or item.get("product_description") or "").strip()
        quantity = _number(item.get("quantity"))
        unit_price = _number(item.get("unit_price"))
        line_total = _number(item.get("line_total"))

        if line_total is not None and not description:
            issues.append(
                _semantic_issue(
                    "PRICED_ITEM_WITHOUT_DESCRIPTION",
                    "high",
                    "A priced item has no usable description.",
                    item_index=index,
                    line_total=line_total,
                )
            )

        if quantity is not None and quantity > 1:
            if unit_price is not None and line_total is not None:
                expected = quantity * unit_price
                if abs(expected - line_total) > max(tolerance, 0.05):
                    issues.append(
                        _semantic_issue(
                            "QUANTITY_AMOUNT_CONFLICT",
                            "high",
                            "Quantity multiplied by unit price does not match line total.",
                            item_index=index,
                            quantity=quantity,
                            unit_price=unit_price,
                            line_total=line_total,
                            expected_line_total=round(expected, 2),
                        )
                    )
            elif quantity >= 10 and unit_price is None:
                issues.append(
                    _semantic_issue(
                        "LARGE_QUANTITY_WITHOUT_UNIT_PRICE",
                        "high",
                        "A large quantity has no unit-price or multiplication support.",
                        item_index=index,
                        quantity=quantity,
                        line_total=line_total,
                    )
                )

        if line_total is not None and abs(line_total) <= tolerance:
            issues.append(
                _semantic_issue(
                    "ZERO_VALUE_ITEM",
                    "low",
                    "A zero-value row may be a free item, modifier, or document metadata.",
                    item_index=index,
                    description=description,
                )
            )

        if _description_conflicts(item):
            issues.append(
                _semantic_issue(
                    "DESCRIPTION_FIELD_CONFLICT",
                    "medium",
                    "Canonical and product description fields do not describe the same text.",
                    item_index=index,
                    description=item.get("description"),
                    product_description=item.get("product_description"),
                    raw_description=item.get("raw_description"),
                )
            )

    validation_codes = {
        str(issue.get("code"))
        for issue in ((validation_report or {}).get("issues") or [])
        if isinstance(issue, dict)
    }
    semantic_validation_codes = {
        "ITEM_SUM_MISMATCH",
        "NO_ITEMS",
        "ITEMS_WITHOUT_LINE_TOTAL",
        "DISCOUNT_LIKELY_ALREADY_APPLIED",
    }
    if validation_codes.intersection(semantic_validation_codes):
        issues.append(
            _semantic_issue(
                "VALIDATION_REQUIRES_ITEM_REINTERPRETATION",
                "high",
                "Deterministic validation found an item-structure contradiction.",
                validation_codes=sorted(validation_codes.intersection(semantic_validation_codes)),
            )
        )

    high_count = sum(1 for item in issues if item.get("severity") == "high")
    medium_count = sum(1 for item in issues if item.get("severity") == "medium")
    low_count = sum(1 for item in issues if item.get("severity") == "low")
    triggered = high_count > 0 or medium_count >= 2
    score = high_count * 3 + medium_count + low_count * 0.1

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "suspicious" if triggered else ("observed" if issues else "clean"),
        "triggered": triggered,
        "score": round(score, 2),
        "issue_count": len(issues),
        "high_count": high_count,
        "medium_count": medium_count,
        "low_count": low_count,
        "issues": issues,
        "recommended_action": "semantic_llm_review" if triggered else "continue",
    }


def attach_semantic_suspicion(
    validation_report: dict[str, Any],
    suspicion: dict[str, Any],
) -> dict[str, Any]:
    """Attach semantic diagnostics and prevent unsafe automatic import.

    Arithmetic balance remains unchanged. A suspicious result is downgraded to
    ``needs_review`` until an evidence-backed correction removes the signals.
    """

    report = copy.deepcopy(validation_report)
    if not suspicion.get("issues") and not suspicion.get("triggered"):
        return report
    report["semantic_suspicion"] = copy.deepcopy(suspicion)
    existing_codes = {
        str(item.get("code")) for item in (report.get("issues") or []) if isinstance(item, dict)
    }
    semantic_issues = [
        copy.deepcopy(item)
        for item in (suspicion.get("issues") or [])
        if isinstance(item, dict)
        and item.get("severity") in {"medium", "high"}
        and str(item.get("code")) not in existing_codes
    ]
    report.setdefault("issues", []).extend(semantic_issues)
    if suspicion.get("triggered") and report.get("import_decision") == "import":
        report["import_decision"] = "needs_review"
    return report


__all__ = [
    "SCHEMA_VERSION",
    "attach_semantic_suspicion",
    "evaluate_semantic_suspicion",
]
