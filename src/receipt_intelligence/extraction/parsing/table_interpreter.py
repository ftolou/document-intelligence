#!/usr/bin/env python3
"""Dedicated LLM table-interpretation stage for receipt item tables.

This module is intentionally not a deterministic semantic parser. It prepares
VLM/OCR table evidence and asks the LLM to infer the implicit table schema:
which columns/cells are description, quantity, unit/original price, discount,
final line total, tax marker, total, payment, or change.

Why this exists:
    The main receipt parser previously received structured VLM evidence, but it
    had to solve too many tasks at once. Clear table rows could still be
    flattened into noisy descriptions such as "SHAMPOO Coupon SORTIMENT". This
    stage creates a separate intermediate artifact that the main parser can use
    as high-priority evidence before assembling the final receipt JSON.

The deterministic code in this module only:
    - selects/truncates evidence for the prompt
    - validates/coerces the JSON wrapper
    - records failures
It does not invent item rows or hard-code merchant/receipt-specific semantics.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from receipt_intelligence.application.llm_json import parse_json_from_llm
from receipt_intelligence.application.ports.llm import (
    GenerationRequest,
    LlmGateway,
    coerce_generation_result,
)
from receipt_intelligence.extraction.parsing.llm_parser import ollama_generate
from receipt_intelligence.prompts import render_prompt_template

TABLE_INTERPRETATION_SCHEMA_VERSION = "v14_17_table_interpretation_compact_1"


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_list(value: Any, limit: int) -> list[Any]:
    if not isinstance(value, list):
        return []
    return value[: max(0, limit)]


def _compact_structured_tables(
    visual_evidence: dict[str, Any], *, max_tables: int = 3, max_rows: int = 55
) -> list[dict[str, Any]]:
    """Keep table evidence compact enough to avoid JSON truncation.

    The LLM receives raw cells, row ids, row order and only short role hints.
    Long hint reasons and neighbour rows are intentionally omitted from the
    table-interpreter prompt. They are still available in the saved visual
    evidence artifact if deeper debugging is needed.
    """
    out: list[dict[str, Any]] = []
    for table in _safe_list(visual_evidence.get("structured_tables"), max_tables):
        if not isinstance(table, dict):
            continue
        rows: list[dict[str, Any]] = []
        for row in _safe_list(table.get("rows"), max_rows):
            if not isinstance(row, dict):
                continue
            cells = [str(c)[:180] for c in (row.get("cells") or [])]
            role_hints = [str(h) for h in (row.get("role_hints") or [])[:5]]
            rows.append(
                {
                    "id": row.get("id"),
                    "i": row.get("row_index"),
                    "src_i": row.get("source_row_index"),
                    "cells": cells,
                    "amounts": row.get("amounts") or [],
                    "hints": role_hints,
                }
            )
        ctx = table.get("table_context") if isinstance(table.get("table_context"), dict) else {}
        out.append(
            {
                "id": table.get("id"),
                "row_count": table.get("row_count"),
                "ctx": {
                    "header_index": ctx.get("header_index"),
                    "header_cells": ctx.get("header_cells") or [],
                    "item_table": ctx.get("item_table"),
                    "qty_column_table": ctx.get("qty_column_table"),
                    "amount_cols": ctx.get("amount_col_indices") or [],
                    "desc_cols": ctx.get("description_col_indices") or [],
                    "qty_cols": ctx.get("qty_col_indices") or [],
                    "tax_cols": ctx.get("tax_col_indices") or [],
                    "first_total_row_index": ctx.get("first_total_row_index"),
                    "first_tax_header_index": ctx.get("first_tax_header_index"),
                },
                "rows": rows,
            }
        )
    return out


def _compact_auxiliary_evidence(visual_evidence: dict[str, Any]) -> dict[str, Any]:
    """Small evidence bundle for settlement/quantity/tax reasoning."""

    def slim_rows(rows: Any, limit: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not isinstance(rows, list):
            return out
        for r in rows[:limit]:
            if not isinstance(r, dict):
                continue
            out.append(
                {
                    "id": r.get("id") or r.get("row_id") or r.get("candidate_id"),
                    "text": r.get("text")
                    or r.get("quantity_row_text")
                    or r.get("product_row_text"),
                    "amount": r.get("amount")
                    or r.get("computed_line_total")
                    or r.get("settlement_total"),
                    "raw": r.get("raw") or r.get("raw_label"),
                    "type": r.get("type") or r.get("pattern"),
                }
            )
        return out

    tax = (
        visual_evidence.get("tax_table_candidates")
        if isinstance(visual_evidence.get("tax_table_candidates"), dict)
        else {}
    )
    return {
        "summary": visual_evidence.get("summary")
        if isinstance(visual_evidence.get("summary"), dict)
        else {},
        "payment_or_change_lines": slim_rows(visual_evidence.get("payment_change_lines"), 12),
        "payment_reconciliation": _safe_list(
            visual_evidence.get("total_payment_reconciliation_candidates"), 6
        ),
        "quantity_note_links": _safe_list(visual_evidence.get("quantity_note_link_candidates"), 10),
        "final_price_groups": _safe_list(visual_evidence.get("final_price_adjustment_groups"), 8),
        "tax_table_summary": {
            "tax_total_candidate": tax.get("tax_total_candidate")
            if isinstance(tax, dict)
            else None,
            "rows": (tax.get("rows") or [])[:8] if isinstance(tax, dict) else [],
        },
    }


def table_interpretation_schema_for_prompt() -> dict[str, Any]:
    """Compact intermediate schema.

    Keep this intentionally small. Previous verbose schemas caused valid semantic
    reasoning to be truncated before JSON completion.
    """
    return {
        "schema_version": TABLE_INTERPRETATION_SCHEMA_VERSION,
        "status": "ok|partial|failed",
        "tables": [
            {
                "source_table_id": "vlm_table_00",
                "table_type": "headerless_item_table|item_table_with_headers|tax_table|payment_table|mixed|unknown",
                "confidence": 0.0,
                "column_roles": {
                    "0": "quantity|description|unit_or_original_price|discount_amount|line_total|tax_code|total|payment|change|unknown"
                },
                "rows": [
                    {
                        "source_row_id": "vlm_table_00_row_001",
                        "row_type": "item|quantity_note|discount|deposit|refund|subtotal|total|payment|change|tax|header|note|unknown",
                        "raw_cells": [],
                        "raw_description": None,
                        "description": None,
                        "product_description": None,
                        "line_note": None,
                        "promotion_note": None,
                        "quantity": None,
                        "unit": None,
                        "unit_price": None,
                        "original_price": None,
                        "discount_amount": None,
                        "line_total": None,
                        "tax_code": None,
                        "linked_to_row_id": None,
                        "confidence": 0.0,
                    }
                ],
                "sum_line_total": None,
                "printed_total_candidate": None,
                "difference": None,
            }
        ],
        "settlement": {
            "amount_due": None,
            "tenders": [
                {
                    "type": "cash|card|voucher|coupon|gift_card|store_credit|unknown",
                    "amount": None,
                    "sign_meaning": "paid_by_customer|discount_against_due|returned_to_customer|unknown",
                    "source_row_id": None,
                    "raw_label": None,
                }
            ],
            "change_due_to_customer": None,
            "printed_change_raw": None,
            "settlement_balanced": None,
            "settlement_equation": None,
            "confidence": 0.0,
        },
        "warnings": [],
        "overall_confidence": 0.0,
    }


def build_table_interpretation_prompt(visual_evidence: dict[str, Any]) -> str:
    tables = _compact_structured_tables(visual_evidence)
    aux = _compact_auxiliary_evidence(visual_evidence)
    schema = table_interpretation_schema_for_prompt()
    return render_prompt_template(
        "table_interpreter_compact.txt",
        SCHEMA_VERSION=TABLE_INTERPRETATION_SCHEMA_VERSION,
        SCHEMA_JSON=json.dumps(schema, ensure_ascii=False, indent=2),
        STRUCTURED_TABLES_JSON=json.dumps(tables, ensure_ascii=False, indent=2),
        AUXILIARY_EVIDENCE_JSON=json.dumps(aux, ensure_ascii=False, indent=2),
    )


def _normalize_status(value: Any) -> str:
    text = str(value or "partial").strip().lower()
    return text if text in {"ok", "partial", "failed", "skipped"} else "partial"


def normalize_table_interpretation(obj: dict[str, Any]) -> dict[str, Any]:
    """Coerce the LLM wrapper without creating semantic content."""
    if not isinstance(obj, dict):
        return failed_table_interpretation("LLM output root was not an object")
    out = dict(obj)
    out["schema_version"] = TABLE_INTERPRETATION_SCHEMA_VERSION
    out["status"] = _normalize_status(out.get("status"))
    out["tables"] = [t for t in (out.get("tables") or []) if isinstance(t, dict)]
    settlement = out.get("settlement") if isinstance(out.get("settlement"), dict) else {}
    out["settlement"] = settlement
    out["warnings"] = [str(w) for w in (out.get("warnings") or []) if str(w).strip()]
    try:
        out["overall_confidence"] = max(0.0, min(1.0, float(out.get("overall_confidence", 0.0))))
    except Exception:
        out["overall_confidence"] = 0.0
    return out


def failed_table_interpretation(error: str) -> dict[str, Any]:
    return {
        "schema_version": TABLE_INTERPRETATION_SCHEMA_VERSION,
        "status": "failed",
        "tables": [],
        "settlement": {
            "amount_due": None,
            "tenders": [],
            "payments": [],
            "change_due_to_customer": None,
            "change": None,
            "settlement_balanced": None,
            "confidence": 0.0,
        },
        "warnings": [f"Table interpretation failed: {error}"],
        "overall_confidence": 0.0,
    }


def skipped_table_interpretation(reason: str) -> dict[str, Any]:
    return {
        "schema_version": TABLE_INTERPRETATION_SCHEMA_VERSION,
        "status": "skipped",
        "tables": [],
        "settlement": {
            "amount_due": None,
            "tenders": [],
            "payments": [],
            "change_due_to_customer": None,
            "change": None,
            "settlement_balanced": None,
            "confidence": 0.0,
        },
        "warnings": [reason],
        "overall_confidence": 0.0,
    }


def has_structured_table_evidence(visual_evidence: dict[str, Any] | None) -> bool:
    return bool(isinstance(visual_evidence, dict) and visual_evidence.get("structured_tables"))


def run_table_interpreter(
    *,
    visual_evidence: dict[str, Any] | None,
    ollama_url: str,
    model: str,
    num_ctx: int = 16384,
    num_predict: int = 4096,
    keep_alive: str | None = None,
    timeout: float = 180.0,
    format_json: bool = True,
    llm_gateway: LlmGateway | None = None,
) -> dict[str, Any]:
    """Run the dedicated table-interpretation LLM step.

    Failure is non-fatal for the pipeline; callers should continue with the
    original visual evidence if this returns status=failed/skipped.
    """
    started = time.perf_counter()
    if not has_structured_table_evidence(visual_evidence):
        result = skipped_table_interpretation(
            "No structured VLM table evidence was available for table interpretation."
        )
        result["prompt"] = ""
        result["raw_output"] = ""
        result["duration_seconds"] = round(time.perf_counter() - started, 2)
        return result

    prompt = build_table_interpretation_prompt(visual_evidence or {})
    raw_text = ""
    try:
        generation = (
            llm_gateway.generate(
                GenerationRequest(
                    model=model,
                    prompt=prompt,
                    num_ctx=num_ctx,
                    num_predict=num_predict,
                    temperature=0.0,
                    keep_alive=keep_alive,
                    timeout_seconds=timeout,
                    format_json=format_json,
                )
            )
            if llm_gateway is not None
            else coerce_generation_result(
                ollama_generate(
                    ollama_url=ollama_url,
                    model=model,
                    prompt=prompt,
                    num_ctx=num_ctx,
                    num_predict=num_predict,
                    temperature=0.0,
                    keep_alive=keep_alive,
                    timeout=timeout,
                    format_json=format_json,
                )
            )
        )
        raw_text = generation.text
        parsed = parse_json_from_llm(generation)
        result = normalize_table_interpretation(parsed)
        if result.get("schema_version") != TABLE_INTERPRETATION_SCHEMA_VERSION:
            result["schema_version"] = TABLE_INTERPRETATION_SCHEMA_VERSION
        if result.get("status") == "failed" and result.get("tables"):
            result["status"] = "partial"
        result["duration_seconds"] = round(time.perf_counter() - started, 2)
        result["model"] = model
        result["ollama_url"] = ollama_url
        result["prompt"] = prompt
        result["raw_output"] = raw_text
        return result
    except Exception as exc:
        result = failed_table_interpretation(f"{type(exc).__name__}: {exc}")
        result["duration_seconds"] = round(time.perf_counter() - started, 2)
        result["model"] = model
        result["ollama_url"] = ollama_url
        result["prompt"] = prompt
        result["raw_output"] = raw_text
        return result


def attach_table_interpretation_to_visual_evidence(
    visual_evidence: dict[str, Any] | None,
    table_interpretation: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Attach a compact interpretation artifact to visual evidence for main LLM prompt."""
    if not isinstance(visual_evidence, dict):
        return visual_evidence
    if not isinstance(table_interpretation, dict):
        return visual_evidence

    # Avoid embedding the prompt/raw output back into the main parser prompt.
    compact = {k: v for k, v in table_interpretation.items() if k not in {"prompt", "raw_output"}}
    enriched = dict(visual_evidence)
    enriched["table_interpretation"] = compact
    summary = dict(enriched.get("summary") or {})
    summary["table_interpretation_status"] = compact.get("status")
    summary["table_interpretation_table_count"] = len(compact.get("tables") or [])
    summary["table_interpretation_confidence"] = compact.get("overall_confidence")
    enriched["summary"] = summary
    guidance = list(enriched.get("semantic_guidance") or [])
    guidance.append(
        "A dedicated table_interpretation artifact is available. Use it as high-priority evidence for item rows, discount columns, line totals, settlement rows, and headerless table semantics; raw VLM cells remain traceability evidence."
    )
    enriched["semantic_guidance"] = guidance
    return enriched


def table_interpretation_to_prompt_text(table_interpretation: dict[str, Any] | None) -> str:
    if not isinstance(table_interpretation, dict):
        return ""
    compact = {k: v for k, v in table_interpretation.items() if k not in {"prompt", "raw_output"}}
    return json.dumps(compact, ensure_ascii=False, indent=2)
