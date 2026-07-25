#!/usr/bin/env python3
"""Patch-style validation correction.

This module asks the LLM for compact, evidence-supported JSON Patch-like
operations, including an optional item-list semantic rewrite. Only a narrow
whitelist is applied. If patches do not improve deterministic validation, the
caller keeps the previous receipt.
"""

from __future__ import annotations

import copy
import json
import re
import time
from typing import Any

from receipt_intelligence.application.llm_json import parse_json_from_llm
from receipt_intelligence.application.ports.llm import (
    GenerationRequest,
    LlmGateway,
    coerce_generation_result,
)
from receipt_intelligence.extraction.parsing.llm_parser import ollama_generate
from receipt_intelligence.prompts import render_prompt_template

SCHEMA_VERSION = "v14_18_correction_patch_1"


def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "import_decision": report.get("import_decision"),
        "balanced": report.get("balanced"),
        "calculated_item_total": report.get("calculated_item_total"),
        "stated_total": report.get("stated_total"),
        "difference": report.get("difference"),
        "payment_sum": report.get("payment_sum"),
        "issues": [
            {
                "code": i.get("code"),
                "severity": i.get("severity"),
                "message": i.get("message"),
                "details": i.get("details"),
            }
            for i in (report.get("issues") or [])
            if isinstance(i, dict)
        ][:14],
    }


def _strip_trace_fields(value: Any, *, depth: int = 0) -> Any:
    """Remove bulky trace fields that are unnecessary for patch planning."""
    if depth > 5:
        return None
    if isinstance(value, dict):
        excluded = {
            "source_line_ids",
            "source_cell_ids",
            "bbox",
            "box",
            "box_2d",
            "polygon",
            "points",
            "raw_output",
            "prompt",
            "embedding",
        }
        out: dict[str, Any] = {}
        for key, item in value.items():
            if str(key) in excluded:
                continue
            compact = _strip_trace_fields(item, depth=depth + 1)
            if compact is not None:
                out[str(key)] = compact
        return out
    if isinstance(value, list):
        return [_strip_trace_fields(item, depth=depth + 1) for item in value[:50]]
    if isinstance(value, str):
        return value[:500]
    return value


def _compact_visual_summary(visual_evidence: dict[str, Any] | None) -> dict[str, Any]:
    ve = visual_evidence if isinstance(visual_evidence, dict) else {}
    arbitration = (
        ve.get("table_arbitration") if isinstance(ve.get("table_arbitration"), dict) else {}
    )
    return {
        "summary": ve.get("summary") or {},
        "table_arbitration": {
            "summary": arbitration.get("summary") or {},
            "warnings": (arbitration.get("warnings") or [])[:5],
            "ocr_layout_item_candidates": _strip_trace_fields(
                (arbitration.get("ocr_layout_item_candidates") or [])[:30]
            ),
            "quantity_note_candidates": _strip_trace_fields(
                (arbitration.get("quantity_note_candidates") or [])[:12]
            ),
            "product_percent_not_tax_rows": _strip_trace_fields(
                (arbitration.get("product_percent_not_tax_rows") or [])[:12]
            ),
        },
        "payment_change_lines": _strip_trace_fields((ve.get("payment_change_lines") or [])[:10]),
        "total_payment_reconciliation_candidates": _strip_trace_fields(
            (ve.get("total_payment_reconciliation_candidates") or [])[:6]
        ),
    }


def _compact_previous_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields needed for correction planning.

    Large source-line arrays, category metadata, pipeline metadata and validation
    payloads are deliberately excluded. The patch planner must match items by
    description and operate on a small, stable representation rather than
    reproducing the complete receipt schema.
    """
    r = receipt if isinstance(receipt, dict) else {}
    merchant = r.get("merchant") if isinstance(r.get("merchant"), dict) else {}
    raw_totals = r.get("totals") if isinstance(r.get("totals"), dict) else {}
    totals = {
        k: raw_totals.get(k)
        for k in ("subtotal", "discount_total", "tax_total", "grand_total", "paid_total", "change")
        if raw_totals.get(k) is not None
    }

    items: list[dict[str, Any]] = []
    for index, item in enumerate(r.get("items") or []):
        if not isinstance(item, dict):
            continue
        compact = {
            "index": index,
            "description": item.get("description") or item.get("product_description"),
            "raw_description": item.get("raw_description"),
            "line_note": item.get("line_note"),
            "quantity": item.get("quantity"),
            "unit": item.get("unit"),
            "unit_price": item.get("unit_price"),
            "original_price": item.get("original_price"),
            "discount_amount": item.get("discount_amount"),
            "line_total": item.get("line_total"),
            "tax_code": item.get("tax_code"),
            "parser_item_type": item.get("parser_item_type") or item.get("category"),
        }
        items.append({k: v for k, v in compact.items() if v is not None})

    payments = []
    for payment in r.get("payments") or []:
        if not isinstance(payment, dict):
            continue
        compact = {
            "method": payment.get("method") or payment.get("type"),
            "amount": payment.get("amount"),
            "label": payment.get("label") or payment.get("description"),
        }
        payments.append({k: v for k, v in compact.items() if v is not None})

    taxes = []
    for tax in r.get("taxes") or []:
        if not isinstance(tax, dict):
            continue
        compact = {
            "rate": tax.get("rate") or tax.get("tax_rate"),
            "net": tax.get("net") or tax.get("net_amount"),
            "tax": tax.get("tax") or tax.get("tax_amount"),
            "gross": tax.get("gross") or tax.get("gross_amount"),
            "tax_code": tax.get("tax_code") or tax.get("code"),
        }
        taxes.append({k: v for k, v in compact.items() if v is not None})

    return {
        "currency": r.get("currency"),
        "merchant": {
            k: v
            for k, v in {
                "name": merchant.get("name"),
                "address": merchant.get("address"),
                "tax_id": merchant.get("tax_id"),
            }.items()
            if v is not None
        },
        "date": r.get("date"),
        "time": r.get("time"),
        "totals": totals,
        "payments": payments,
        "taxes": taxes,
        "items": items,
    }


def _compact_ocr_text(ocr_context: dict[str, Any] | None, *, max_chars: int = 14000) -> str:
    context = ocr_context if isinstance(ocr_context, dict) else {}
    rendered: list[str] = []
    for index, line in enumerate(context.get("lines") or []):
        if not isinstance(line, dict):
            continue
        line_id = str(line.get("line_id") or line.get("id") or f"line_{index:03d}")
        text = re.sub(r"\s+", " ", str(line.get("text") or "")).strip()
        if not text:
            continue
        rendered.append(f"[{line_id}] {text}")
        if sum(len(value) + 1 for value in rendered) >= max_chars:
            break
    return "\n".join(rendered)[:max_chars]


def _compact_spatial_evidence(spatial_evidence: str | None, *, max_chars: int = 18000) -> str:
    return str(spatial_evidence or "").strip()[:max_chars]


def build_patch_correction_prompt(
    previous_receipt: dict[str, Any],
    validation_report: dict[str, Any],
    visual_evidence: dict[str, Any] | None,
    *,
    ocr_context: dict[str, Any] | None = None,
    spatial_evidence: str | None = None,
    semantic_suspicion: dict[str, Any] | None = None,
) -> str:
    return render_prompt_template(
        "correction_patch.txt",
        REPORT_JSON=json.dumps(
            _compact_report(validation_report), ensure_ascii=False, separators=(",", ":")
        ),
        SEMANTIC_SUSPICION_JSON=json.dumps(
            semantic_suspicion or {}, ensure_ascii=False, separators=(",", ":")
        ),
        OCR_TEXT=_compact_ocr_text(ocr_context),
        SPATIAL_EVIDENCE=_compact_spatial_evidence(spatial_evidence),
        VISUAL_SUMMARY_JSON=json.dumps(
            _compact_visual_summary(visual_evidence), ensure_ascii=False, separators=(",", ":")
        ),
        PREVIOUS_JSON=json.dumps(
            _compact_previous_receipt(previous_receipt), ensure_ascii=False, separators=(",", ":")
        ),
    )


def _normalize_patch_obj(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "patches": [],
            "warnings": ["root not object"],
            "confidence": 0.0,
        }

    allowed_ops = {
        "replace_field",
        "replace_payments",
        "remove_items",
        "update_item",
        "add_item",
        "replace_items",
    }
    patches: list[dict[str, Any]] = []
    warnings: list[str] = [str(w)[:240] for w in (obj.get("warnings") or []) if str(w).strip()][:12]
    for patch in obj.get("patches") or []:
        if not isinstance(patch, dict):
            continue
        op = str(patch.get("op") or "").strip()
        if op not in allowed_ops:
            warnings.append(f"ignored unsupported operation: {op or '<empty>'}")
            continue
        clean = dict(patch)
        if clean.get("reason") is not None:
            clean["reason"] = str(clean.get("reason"))[:180]
        if op == "replace_items":
            items = [item for item in (clean.get("items") or []) if isinstance(item, dict)]
            clean["items"] = items[:80]
        patches.append(clean)
        if len(patches) >= 8:
            break

    try:
        conf = max(0.0, min(1.0, float(obj.get("confidence") or 0.0)))
    except Exception:
        conf = 0.0
    status = str(obj.get("status") or ("ok" if patches else "no_patch")).lower()
    if status not in {"ok", "no_patch", "failed"}:
        status = "ok" if patches else "no_patch"
    if status == "ok" and not patches:
        status = "no_patch"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "patches": patches,
        "warnings": warnings[:20],
        "confidence": conf,
    }


def run_patch_correction_pass(
    *,
    previous_receipt: dict[str, Any],
    validation_report: dict[str, Any],
    visual_evidence: dict[str, Any] | None,
    ocr_context: dict[str, Any] | None = None,
    spatial_evidence: str | None = None,
    semantic_suspicion: dict[str, Any] | None = None,
    ollama_url: str,
    model: str,
    num_ctx: int = 16384,
    num_predict: int = 2048,
    keep_alive: str | None = None,
    timeout: float = 180.0,
    format_json: bool = True,
    llm_gateway: LlmGateway | None = None,
) -> dict[str, Any]:
    """Generate a compact patch plan and retry once only on malformed JSON.

    The retry remains patch-only. It never asks the model to reproduce the full
    receipt, which removes the large-output truncation path that previously
    produced unterminated JSON strings.
    """
    started = time.perf_counter()
    prompt = build_patch_correction_prompt(
        previous_receipt,
        validation_report,
        visual_evidence,
        ocr_context=ocr_context,
        spatial_evidence=spatial_evidence,
        semantic_suspicion=semantic_suspicion,
    )
    raw_outputs: list[str] = []
    errors: list[str] = []

    for attempt in range(1, 3):
        attempt_prompt = prompt
        if attempt == 2:
            attempt_prompt += (
                "\n\nRETRY AFTER INVALID JSON: Return one JSON object under 2000 characters. "
                "Use at most 5 patches and omit unnecessary fields. A replace_items operation "
                "may include only the corrected item list when semantic structure is wrong."
            )
        raw = ""
        try:
            generation = (
                llm_gateway.generate(
                    GenerationRequest(
                        model=model,
                        prompt=attempt_prompt,
                        operation="receipt_patch_repair",
                        attempt=attempt,
                        num_ctx=max(8192, min(num_ctx, 18432)),
                        num_predict=max(1024, min(num_predict, 2048)),
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
                        prompt=attempt_prompt,
                        num_ctx=max(8192, min(num_ctx, 18432)),
                        num_predict=max(1024, min(num_predict, 2048)),
                        temperature=0.0,
                        keep_alive=keep_alive,
                        timeout=timeout,
                        format_json=format_json,
                    )
                )
            )
            raw = generation.text
            raw_outputs.append(raw)
            obj = _normalize_patch_obj(parse_json_from_llm(generation))
            obj.update(
                {
                    "prompt": prompt,
                    "raw_output": raw
                    if attempt == 1
                    else "\n\n--- RETRY OUTPUT ---\n".join(raw_outputs),
                    "duration_seconds": round(time.perf_counter() - started, 2),
                    "prompt_chars": len(prompt),
                    "raw_chars": sum(len(x) for x in raw_outputs),
                    "attempt_count": attempt,
                    "retry_used": attempt > 1,
                    "errors": errors,
                    "mode": "semantic_patch",
                }
            )
            return obj
        except Exception as exc:
            errors.append(f"attempt_{attempt}: {type(exc).__name__}: {exc}")
            if raw and (not raw_outputs or raw_outputs[-1] != raw):
                raw_outputs.append(raw)

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "patches": [],
        "warnings": errors[:20],
        "confidence": 0.0,
        "prompt": prompt,
        "raw_output": "\n\n--- RETRY OUTPUT ---\n".join(raw_outputs),
        "duration_seconds": round(time.perf_counter() - started, 2),
        "prompt_chars": len(prompt),
        "raw_chars": sum(len(x) for x in raw_outputs),
        "attempt_count": 2,
        "retry_used": True,
        "errors": errors,
        "mode": "semantic_patch",
    }


def _set_path(root: dict[str, Any], path: str, value: Any) -> bool:
    if not path.startswith("/"):
        return False
    parts = [p for p in path.strip("/").split("/") if p]
    cur: Any = root
    for part in parts[:-1]:
        if isinstance(cur, dict):
            cur = cur.setdefault(part, {})
        else:
            return False
    if isinstance(cur, dict) and parts:
        cur[parts[-1]] = value
        return True
    return False


def _match_item(item: dict[str, Any], match: dict[str, Any]) -> bool:
    desc = str(item.get("description") or item.get("product_description") or "")
    exact = match.get("description")
    if exact is not None and desc.strip().upper() == str(exact).strip().upper():
        return True
    rx = match.get("description_regex")
    if rx:
        try:
            return re.search(str(rx), desc, re.I) is not None
        except re.error:
            return False
    return False


def apply_correction_patches(
    receipt: dict[str, Any], patch_result: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    r = copy.deepcopy(receipt) if isinstance(receipt, dict) else {}
    applied: list[dict[str, Any]] = []
    for patch in patch_result.get("patches") or []:
        if not isinstance(patch, dict):
            continue
        op = str(patch.get("op") or "").strip()
        if op == "replace_field":
            path = str(patch.get("path") or "")
            if path.startswith("/totals/") or path in {"/date", "/time"}:
                if _set_path(r, path, patch.get("value")):
                    applied.append({"op": op, "path": path, "reason": patch.get("reason")})
        elif op == "replace_payments":
            payments = [p for p in (patch.get("payments") or []) if isinstance(p, dict)]
            if payments:
                r["payments"] = payments
                totals = r.setdefault("totals", {}) if isinstance(r.get("totals"), dict) else {}
                r["totals"] = totals
                if patch.get("paid_total") is not None:
                    totals["paid_total"] = patch.get("paid_total")
                applied.append(
                    {"op": op, "payment_count": len(payments), "reason": patch.get("reason")}
                )
        elif op == "remove_items":
            match = patch.get("match") if isinstance(patch.get("match"), dict) else {}
            items = [it for it in (r.get("items") or []) if isinstance(it, dict)]
            keep = [it for it in items if not _match_item(it, match)]
            if len(keep) < len(items):
                r["items"] = keep
                applied.append(
                    {
                        "op": op,
                        "removed_count": len(items) - len(keep),
                        "reason": patch.get("reason"),
                    }
                )
        elif op == "update_item":
            match = patch.get("match") if isinstance(patch.get("match"), dict) else {}
            fields = patch.get("fields") if isinstance(patch.get("fields"), dict) else {}
            changed = 0
            for it in r.get("items") or []:
                if isinstance(it, dict) and _match_item(it, match):
                    for k, v in fields.items():
                        if k in {
                            "description",
                            "product_description",
                            "raw_description",
                            "line_note",
                            "promotion_note",
                            "quantity",
                            "unit",
                            "unit_price",
                            "original_price",
                            "discount_amount",
                            "line_total",
                            "tax_code",
                            "category",
                            "parser_item_type",
                        }:
                            it[k] = v
                    changed += 1
            if changed:
                applied.append({"op": op, "updated_count": changed, "reason": patch.get("reason")})
        elif op == "add_item":
            item = patch.get("item") if isinstance(patch.get("item"), dict) else None
            if item and item.get("description") and item.get("line_total") is not None:
                r.setdefault("items", [])
                if isinstance(r["items"], list):
                    r["items"].append(item)
                    applied.append(
                        {
                            "op": op,
                            "description": item.get("description"),
                            "line_total": item.get("line_total"),
                            "reason": patch.get("reason"),
                        }
                    )
        elif op == "replace_items":
            items = [it for it in (patch.get("items") or []) if isinstance(it, dict)]
            allowed_item_fields = {
                "description",
                "product_description",
                "raw_description",
                "line_note",
                "promotion_note",
                "quantity",
                "unit",
                "unit_price",
                "original_price",
                "discount_amount",
                "line_total",
                "tax_code",
                "category",
                "parser_item_type",
                "source_line_ids",
            }
            cleaned_items: list[dict[str, Any]] = []
            for item in items:
                clean_item = {
                    key: value for key, value in item.items() if key in allowed_item_fields
                }
                description = clean_item.get("description") or clean_item.get("product_description")
                if description and clean_item.get("line_total") is not None:
                    cleaned_items.append(clean_item)
            if cleaned_items:
                old_count = len([it for it in (r.get("items") or []) if isinstance(it, dict)])
                r["items"] = cleaned_items
                applied.append(
                    {
                        "op": op,
                        "old_item_count": old_count,
                        "new_item_count": len(cleaned_items),
                        "reason": patch.get("reason"),
                    }
                )
    return r, applied
