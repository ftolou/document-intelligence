"""Small deterministic helpers shared by extraction stages."""

from __future__ import annotations

from typing import Any


def report_score(report: dict[str, Any]) -> tuple[int, int, int, float]:
    """Return a deterministic score used only to select between candidates."""

    decision_rank = {
        "import": 3,
        "needs_review": 2,
        "reject": 1,
        "llm_failed": 0,
    }.get(str(report.get("import_decision")), 0)
    issues = [item for item in report.get("issues") or [] if isinstance(item, dict)]
    high = sum(1 for item in issues if item.get("severity") in {"critical", "high"})
    medium = sum(1 for item in issues if item.get("severity") == "medium")
    difference = report.get("difference")
    try:
        absolute_difference = abs(float(difference)) if difference is not None else 9999.0
    except (TypeError, ValueError):
        absolute_difference = 9999.0
    return (decision_rank, -high, -medium, -absolute_difference)


def should_run_visual_layer(report: dict[str, Any]) -> bool:
    if str(report.get("import_decision")) in {"import", "llm_failed"}:
        return False
    codes = {str(item.get("code")) for item in report.get("issues") or [] if isinstance(item, dict)}
    interesting = {
        "ITEM_SUM_MISMATCH",
        "TAX_TOTAL_CONFLICTS_WITH_TAX_TABLE_EVIDENCE",
        "TAX_TABLE_OVER_SPLIT",
        "CHANGE_WITHOUT_PAYMENT_AMOUNT",
        "PAYMENT_TOTAL_MISMATCH",
        "UNRESOLVED_AMOUNT_LINES",
        "NO_PAYMENT",
        "NO_ITEMS",
        "MISSING_TOTAL",
    }
    return bool(codes.intersection(interesting))


def merge_visual_evidence(
    primary: dict[str, Any] | None,
    extra: dict[str, Any] | None,
    *,
    backend_suffix: str = "",
) -> dict[str, Any] | None:
    if not primary:
        return extra
    if not extra:
        return primary
    merged = dict(primary)
    for key in (
        "lines",
        "amount_lines",
        "payment_change_lines",
        "tax_like_lines",
        "item_price_like_lines",
        "structured_tables",
        "quantity_hint_rows",
        "item_candidate_rows",
        "total_payment_rows",
        "tax_context_rows",
    ):
        merged[key] = (merged.get(key) or []) + (extra.get(key) or [])
    if backend_suffix:
        merged["backend"] = f"{merged.get('backend') or 'visual'}+{backend_suffix}"
    if extra.get("semantic_guidance") and not merged.get("semantic_guidance"):
        merged["semantic_guidance"] = extra.get("semantic_guidance")
    structured_rows = [
        row
        for table in merged.get("structured_tables") or []
        if isinstance(table, dict)
        for row in table.get("rows") or []
    ]
    merged["summary"] = {
        "line_count": len(merged.get("lines") or []),
        "amount_line_count": len(merged.get("amount_lines") or []),
        "structured_table_count": len(merged.get("structured_tables") or []),
        "structured_table_row_count": len(structured_rows),
        "quantity_hint_row_count": len(merged.get("quantity_hint_rows") or []),
        "item_candidate_row_count": len(merged.get("item_candidate_rows") or []),
        "has_payment_like": bool(
            merged.get("payment_change_lines") or merged.get("total_payment_rows")
        ),
        "has_change_like": bool(merged.get("payment_change_lines")),
        "has_tax_like": bool(merged.get("tax_like_lines") or merged.get("tax_context_rows")),
    }
    return merged
