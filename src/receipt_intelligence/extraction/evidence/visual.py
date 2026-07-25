#!/usr/bin/env python3
"""Build compact visual evidence from VLM output and region-first crop re-OCR evidence.

This module still does not decide receipt semantics. It preserves PaddleOCR-VL
layout/table structure so the LLM correction pass can classify rows itself.
The deterministic hints are deliberately weak and generic.
"""

from __future__ import annotations

import json
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

AMOUNT_RE = re.compile(
    r"(?<!\d)([-+−]?\s*\d{1,5}(?:[.\s]\d{3})*(?:[,\.]\s*\d{2})|[-+−]?\s*\d{1,5}\s+\d{2})(?:\s*[-−])?(?!\d)"
)
DATE_RE = re.compile(r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")
PAYMENT_WORD_RE = re.compile(
    r"\b(BAR|CASH|GEGEBEN|PAID|ZAHLUNG|KARTE|CARD|EC|GIROCARD|LASTSCHRIFT|VISA|MASTERCARD)\b", re.I
)
CHANGE_WORD_RE = re.compile(
    r"\b(RÜCKGELD|RUECKGELD|ROCKGELD|ZURÜCK|ZURUECK|CHANGE|WECHSELGELD)\b", re.I
)
TOTAL_WORD_RE = re.compile(
    r"\b(SUMME|ZWISCHENSUMME|SUBTOTAL|BONSUMME|GESAMT|TOTAL|ZU\s+(?:ZAHLEN|BEZAHLEN)|AMOUNT\s+DUE|ENDS?SUMME)\b",
    re.I,
)
TAX_CONTEXT_WORD_RE = re.compile(r"\b(MWST|UST|VAT|TAX|STEUER|NETTO|BRUTTO|GROSS|NET)\b", re.I)
PERCENT_RE = re.compile(r"\b\d{1,2}(?:[,\.]\d+)?\s*%")
ITEM_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]{3,}")

# Weak quantity/unit-price note signals. These are intentionally broad and
# language-agnostic-ish examples, not receipt-specific rules.
QTY_UNIT_WORD_RE = re.compile(
    r"\b(STK|STÜCK|STUECK|PCS?|PIECE|QTY|ANZ|ANZAHL|KG|G|GRAMM|L|ML|M|PACK|PK|BUND|FL|BTL|X)\b",
    re.I,
)
QTY_MULTIPLIER_RE = re.compile(
    r"(?:^|\b)(?:\d+[,.]?\d*\s*)?(?:STK|STÜCK|STUECK|PCS?|QTY|KG|G|L|ML)?\s*(?:x|×|@|à|a)\s*[-+]?\d+[,.]\d{1,3}\b",
    re.I,
)
QTY_START_RE = re.compile(
    r"^\s*\d+[,.]?\d*\s*(?:STK|STÜCK|STUECK|PCS?|KG|G|L|ML|PACK|PK)?\s*(?:x|×|@|à|a)?\s*[-+]?\d*(?:[,\.]\d+)?\s*$",
    re.I,
)
NEGATIVE_AMOUNT_RE = re.compile(r"[-−]\s*\d|\d+[,\.]\d{2}\s*[-−]")

FINAL_PRICE_RE = re.compile(
    r"\b(IHR\s+PREIS|DEIN\s+PREIS|AKTIONSPREIS|ENDPREIS|SALE\s*PRICE|FINAL\s*PRICE|YOUR\s*PRICE|NOW\s*PRICE|REDUZIERT|REDUCED|SONDERPREIS|ANGEBOTSPREIS)\b",
    re.I,
)
DISCOUNT_WORD_RE = re.compile(
    r"\b(RABATT|DISCOUNT|COUPON|AKTION|NACHLASS|GUTSCHRIFT|VOUCHER|BONUS|PROMO|PROMOTION|REDUZIERUNG)\b",
    re.I,
)
NET_WORD_RE = re.compile(r"\b(NETTO|NET)\b", re.I)
GROSS_WORD_RE = re.compile(r"\b(BRUTTO|GROSS)\b", re.I)
TAX_AMOUNT_WORD_RE = re.compile(r"\b(MWST|UST|VAT|TAX|STEUER)\b", re.I)


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.cell_parts: list[str] = []
        self.current_row: list[str] = []
        self.tables: list[list[list[str]]] = []
        self.current_table: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "table":
            self.in_table = True
            self.current_table = []
        elif tag == "tr" and self.in_table:
            self.in_row = True
            self.current_row = []
        elif tag in {"td", "th"} and self.in_row:
            self.in_cell = True
            self.cell_parts = []
        elif tag == "br" and self.in_cell:
            self.cell_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self.in_cell:
            text = _clean_cell_text("".join(self.cell_parts))
            self.current_row.append(text)
            self.in_cell = False
            self.cell_parts = []
        elif tag == "tr" and self.in_row:
            if any(c.strip() for c in self.current_row):
                self.current_table.append(self.current_row)
            self.in_row = False
            self.current_row = []
        elif tag == "table" and self.in_table:
            if self.current_table:
                self.tables.append(self.current_table)
            self.in_table = False
            self.current_table = []

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.cell_parts.append(data)


def _clean_cell_text(text: str) -> str:
    text = unescape(text or "")
    text = re.sub(r"\$\s*\^\{\{?\*\}?\}\s*\$", "*", text)
    text = re.sub(r"\s+", " ", text.replace("\\n", " ").replace("\n", " ")).strip()
    return text


def _parse_html_tables(text: str) -> list[list[list[str]]]:
    if not text or "<table" not in text.lower():
        return []
    parser = _TableHTMLParser()
    try:
        parser.feed(text)
    except Exception:
        return []
    return parser.tables


def _parse_markdown_table(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        cells = [_clean_cell_text(c) for c in line.strip("|").split("|")]
        if cells and not all(re.fullmatch(r"[-: ]*", c or "") for c in cells):
            rows.append(cells)
    return rows


def _parse_amount_token(raw: str) -> float | None:
    s = raw.strip().replace("−", "-")
    negative = (
        s.startswith("-") or s.endswith("-") or re.search(r"[,\.]\s*\d{2}\s*[-−]", s) is not None
    )
    s = re.sub(r"[^0-9,\.\s]", "", s).replace(" ", "")
    if not s:
        return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        v = round(float(s), 2)
    except Exception:
        return None
    return -abs(v) if negative else v


def _amounts(text: str) -> list[dict[str, Any]]:
    if DATE_RE.search(text) and "," not in text:
        return []
    out = []
    for m in AMOUNT_RE.finditer(text or ""):
        val = _parse_amount_token(m.group(0))
        if val is not None:
            out.append({"raw": m.group(0).strip(), "value": val})
    return out


def _walk(obj: Any, path: str = "") -> list[tuple[str, str]]:
    """Collect short text snippets from arbitrary VLM JSON."""
    rows: list[tuple[str, str]] = []
    if obj is None:
        return rows
    if isinstance(obj, str):
        text = obj.strip()
        if text:
            rows.append((path, text))
        return rows
    if isinstance(obj, (int, float, bool)):
        return rows
    if isinstance(obj, list):
        for i, x in enumerate(obj[:500]):
            rows.extend(_walk(x, f"{path}[{i}]"))
        return rows
    if isinstance(obj, dict):
        for key in (
            "text",
            "content",
            "markdown",
            "html",
            "table",
            "cell",
            "label",
            "text_content",
            "block_content",
        ):
            if key in obj and isinstance(obj[key], str) and obj[key].strip():
                rows.append((f"{path}.{key}" if path else key, obj[key].strip()))
        for k, v in list(obj.items())[:500]:
            if k in {"image", "img", "base64", "embedding"}:
                continue
            rows.extend(_walk(v, f"{path}.{k}" if path else str(k)))
        return rows
    return rows


def _alpha_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÄÖÜäöüß]{3,}", text or "")


def _norm_key(text: str) -> str:
    return re.sub(
        r"\s+", " ", re.sub(r"[^0-9A-Za-zÄÖÜäöüß%+\-−,. ]+", " ", text or "").strip().lower()
    )


def _cell_words(cell: str) -> list[str]:
    return re.findall(r"[A-Za-zÄÖÜäöüß]{2,}", cell or "")


def _is_mostly_quantity_expression(cells: list[str]) -> bool:
    """Weak, generic quantity/unit-price-note detector.

    This deliberately avoids receipt-specific strings. It only detects rows that
    are mostly quantity/multiplier/unit-price notation and have no meaningful
    product description. Header context can override this later.
    """
    row_text = " ".join(cells).strip()
    if not row_text:
        return False
    if (
        TOTAL_WORD_RE.search(row_text)
        or PAYMENT_WORD_RE.search(row_text)
        or CHANGE_WORD_RE.search(row_text)
        or TAX_CONTEXT_WORD_RE.search(row_text)
    ):
        return False
    words = [w for w in _cell_words(row_text) if not QTY_UNIT_WORD_RE.fullmatch(w)]
    # Product description normally has at least one non-unit word. Very short words
    # like x, kg, pcs, stk are treated as unit/multiplier signals, not products.
    meaningful_word_count = len([w for w in words if len(w) >= 3])
    has_qty_syntax = bool(QTY_MULTIPLIER_RE.search(row_text) or QTY_START_RE.search(row_text))
    has_unit_word_with_digits = bool(
        QTY_UNIT_WORD_RE.search(row_text) and re.search(r"\d", row_text)
    )
    mostly_numeric_or_units = meaningful_word_count == 0 and bool(re.search(r"\d", row_text))
    return (has_qty_syntax or has_unit_word_with_digits) and mostly_numeric_or_units


HEADER_KEYWORDS = {
    "description": re.compile(
        r"\b(BESCHREIBUNG|ARTIKEL|ARTICLE|ITEM|TEXT|WARE|PRODUKT|PRODUCT|BEZEICHNUNG|NAME)\b", re.I
    ),
    "qty": re.compile(r"\b(MENGE|QTY|QUANTITY|ANZ|ANZAHL|STK|PCS|COUNT)\b", re.I),
    "sum": re.compile(r"\b(SUMME|TOTAL|BETRAG|AMOUNT|PREIS|PRICE|EUR|BRUTTO)\b", re.I),
    "tax": re.compile(r"\b(MWST|UST|VAT|TAX|STEUER|NETTO|BRUTTO|GROSS|NET|RATE|SATZ)\b", re.I),
    "payment": PAYMENT_WORD_RE,
}


def _looks_like_header(cells: list[str]) -> bool:
    row_text = " | ".join(cells)
    hits = 0
    for rx in HEADER_KEYWORDS.values():
        if rx.search(row_text):
            hits += 1
    # Header rows usually contain words and no monetary amount, but tax headers
    # may contain MwSt% and still be a header.
    return hits >= 2 or (hits >= 1 and not _amounts(row_text) and len(_cell_words(row_text)) >= 2)


def _has_total_keyword(text: str) -> bool:
    if TOTAL_WORD_RE.search(text or ""):
        return True
    compact = re.sub(r"[^A-ZÄÖÜ]", "", (text or "").upper())
    return any(
        k in compact
        for k in ("SUMME", "ZWISCHENSUMME", "GESAMT", "TOTAL", "ZUBEZAHLEN", "ZUZAHLEN", "ENDSUMME")
    )


def _infer_table_context(rows: list[list[str]]) -> dict[str, Any]:
    """Infer generic table context from row/header terms; no receipt-specific rules."""
    header_index = None
    header_cells: list[str] = []
    for i, cells in enumerate(rows[:8]):
        if _looks_like_header(cells):
            header_index = i
            header_cells = cells
            break

    tax_header_indices: list[int] = []
    total_row_indices: list[int] = []
    payment_row_indices: list[int] = []
    for i, cells in enumerate(rows):
        row_text = " | ".join(cells)
        if _has_total_keyword(row_text):
            total_row_indices.append(i)
        if PAYMENT_WORD_RE.search(row_text) or CHANGE_WORD_RE.search(row_text):
            payment_row_indices.append(i)
        tax_hits = len(
            re.findall(
                r"\b(MWST|UST|VAT|TAX|STEUER|NETTO|BRUTTO|GROSS|NET)\b", row_text, flags=re.I
            )
        )
        # A tax header typically combines rate/tax words with net/gross/brutto columns.
        if tax_hits >= 2 or (
            re.search(r"\b(MWST|UST|VAT|TAX|STEUER)\b", row_text, re.I)
            and re.search(r"\b(NETTO|BRUTTO|GROSS|NET|RATE|SATZ)\b", row_text, re.I)
        ):
            tax_header_indices.append(i)

    item_table = bool(
        header_cells
        and any(HEADER_KEYWORDS["description"].search(c) for c in header_cells)
        and any(
            (HEADER_KEYWORDS["sum"].search(c) or re.fullmatch(r"(?i)EUR|€", c.strip()))
            for c in header_cells
        )
    )
    qty_column_table = bool(
        header_cells and any(HEADER_KEYWORDS["qty"].search(c) for c in header_cells)
    )
    amount_col_indices: list[int] = []
    description_col_indices: list[int] = []
    qty_col_indices: list[int] = []
    tax_col_indices: list[int] = []
    if header_cells:
        for ci, cell in enumerate(header_cells):
            if HEADER_KEYWORDS["description"].search(cell):
                description_col_indices.append(ci)
            if HEADER_KEYWORDS["qty"].search(cell):
                qty_col_indices.append(ci)
            if HEADER_KEYWORDS["tax"].search(cell):
                tax_col_indices.append(ci)
            if HEADER_KEYWORDS["sum"].search(cell) or re.fullmatch(r"(?i)EUR|€", cell.strip()):
                amount_col_indices.append(ci)

    return {
        "header_index": header_index,
        "header_cells": header_cells,
        "tax_header_indices": tax_header_indices,
        "first_tax_header_index": min(tax_header_indices) if tax_header_indices else None,
        "total_row_indices": total_row_indices,
        "first_total_row_index": min(total_row_indices) if total_row_indices else None,
        "payment_row_indices": payment_row_indices,
        "explicit_tax_table": bool(tax_header_indices),
        "item_table": item_table,
        "qty_column_table": qty_column_table,
        "amount_col_indices": amount_col_indices,
        "description_col_indices": description_col_indices,
        "qty_col_indices": qty_col_indices,
        "tax_col_indices": tax_col_indices,
    }


def _row_has_product_description(cells: list[str], context: dict[str, Any]) -> bool:
    desc_indices = context.get("description_col_indices") or []
    if desc_indices:
        for ci in desc_indices:
            if ci < len(cells):
                words = [w for w in _cell_words(cells[ci]) if not QTY_UNIT_WORD_RE.fullmatch(w)]
                if any(len(w) >= 3 for w in words):
                    return True
    # Fallback: first non-empty cell often carries description in receipt tables.
    if cells:
        words = [w for w in _cell_words(cells[0]) if not QTY_UNIT_WORD_RE.fullmatch(w)]
        if any(len(w) >= 3 for w in words):
            return True
    row_words = [w for w in _cell_words(" ".join(cells)) if not QTY_UNIT_WORD_RE.fullmatch(w)]
    return len([w for w in row_words if len(w) >= 3]) >= 1


def _row_role_hints(
    cells: list[str],
    prev_cells: list[str] | None = None,
    next_cells: list[str] | None = None,
    context: dict[str, Any] | None = None,
    row_index: int | None = None,
) -> tuple[list[str], list[str]]:
    context = context or {}
    row_text = " | ".join(c for c in cells if c is not None).strip()
    hints: list[str] = []
    reasons: list[str] = []
    is_header = row_index is not None and context.get("header_index") == row_index
    if is_header:
        hints.append("table_header")
        reasons.append(
            "row appears to be a table header; use it to classify following rows, do not output as item"
        )
        if context.get("explicit_tax_table"):
            hints.append("tax_table_header")
        return hints, reasons

    amts = _amounts(row_text)
    has_desc = _row_has_product_description(cells, context)
    qty_like = _is_mostly_quantity_expression(cells)

    if _has_total_keyword(row_text):
        hints.append("possible_total_or_subtotal")
        reasons.append("contains total/sum keyword")
    if PAYMENT_WORD_RE.search(row_text):
        hints.append("possible_payment")
        reasons.append("contains payment keyword")
    if CHANGE_WORD_RE.search(row_text):
        hints.append("possible_change")
        reasons.append("contains change keyword")

    # Tax context is row-local or starts at an explicit tax header row. A mixed receipt table
    # can contain item rows first and a tax table later; do not mark pre-tax item rows as tax.
    first_tax = context.get("first_tax_header_index")
    in_tax_section = first_tax is not None and row_index is not None and row_index >= first_tax
    if TAX_CONTEXT_WORD_RE.search(row_text) or in_tax_section:
        if TAX_CONTEXT_WORD_RE.search(row_text) or (in_tax_section and amts):
            hints.append("possible_tax_context")
            reasons.append(
                "row or later table section contains explicit tax/net/gross/VAT/MwSt context"
            )
    elif PERCENT_RE.search(row_text) and ITEM_WORD_RE.search(row_text):
        hints.append("contains_percent_in_product_text_not_tax_by_itself")
        reasons.append(
            "percentage appears with product-like text but no explicit tax-table keyword"
        )

    # In an item table with description+sum columns, a quantity column is normal. It should not by
    # itself turn a product row into a quantity-note row.
    if (
        qty_like
        and not has_desc
        and not set(hints).intersection(
            {
                "possible_total_or_subtotal",
                "possible_payment",
                "possible_change",
                "possible_tax_context",
            }
        )
    ):
        hints.append("possible_quantity_or_unit_price_note")
        reasons.append(
            "row mainly resembles quantity/multiplier/unit-price notation and has weak product description"
        )
        if prev_cells or next_cells:
            hints.append("requires_neighbor_classification")
            reasons.append(
                "quantity/unit-price notes often explain nearby item amounts; classify with adjacent rows before outputting"
            )

    item_excluded = set(hints).intersection(
        {
            "possible_total_or_subtotal",
            "possible_payment",
            "possible_change",
            "possible_tax_context",
            "possible_quantity_or_unit_price_note",
            "table_header",
        }
    )
    if amts and has_desc and not item_excluded:
        hints.append("possible_item_charge_or_credit")
        reasons.append(
            "contains product/charge-like description plus amount and no total/payment/tax/quantity-note context"
        )
    elif amts and context.get("item_table") and has_desc and not item_excluded:
        hints.append("possible_item_charge_or_credit")
        reasons.append("row is in a description+sum style item table and contains an amount")

    if NEGATIVE_AMOUNT_RE.search(row_text) and not set(hints).intersection(
        {"possible_tax_context", "possible_payment", "possible_change"}
    ):
        hints.append("contains_negative_amount")
        reasons.append(
            "contains a negative amount; may be discount, return, deposit refund, or correction line"
        )
    return hints, reasons


def _extract_structured_tables_from_obj(obj: Any, path: str = "") -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    if obj is None:
        return tables
    if isinstance(obj, list):
        for i, x in enumerate(obj[:500]):
            tables.extend(_extract_structured_tables_from_obj(x, f"{path}[{i}]"))
        return tables
    if isinstance(obj, dict):
        label = str(obj.get("block_label") or obj.get("label") or "").lower()
        content = (
            obj.get("block_content")
            or obj.get("content")
            or obj.get("html")
            or obj.get("markdown")
            or obj.get("text")
        )
        if isinstance(content, str):
            parsed_tables = _parse_html_tables(content)
            if not parsed_tables and label == "table":
                md_rows = _parse_markdown_table(content)
                if md_rows:
                    parsed_tables = [md_rows]
            for t in parsed_tables:
                tables.append(
                    {
                        "source_path": path,
                        "source_label": label or None,
                        "block_bbox": obj.get("block_bbox")
                        or obj.get("bbox")
                        or obj.get("coordinate"),
                        "rows_raw": t,
                    }
                )
        for k, v in list(obj.items())[:500]:
            if k in {"image", "img", "base64", "embedding"}:
                continue
            tables.extend(_extract_structured_tables_from_obj(v, f"{path}.{k}" if path else str(k)))
        return tables
    if isinstance(obj, str):
        parsed = _parse_html_tables(obj)
        if parsed:
            for t in parsed:
                tables.append(
                    {"source_path": path, "source_label": None, "block_bbox": None, "rows_raw": t}
                )
        else:
            md_rows = _parse_markdown_table(obj)
            if md_rows:
                tables.append(
                    {
                        "source_path": path,
                        "source_label": "markdown_table",
                        "block_bbox": None,
                        "rows_raw": md_rows,
                    }
                )
    return tables


def _build_structured_tables(
    raw: Any, max_tables: int = 12, max_rows_per_table: int = 90
) -> list[dict[str, Any]]:
    raw_tables = _extract_structured_tables_from_obj(raw)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tbl in raw_tables[:max_tables]:
        rows_raw = tbl.get("rows_raw") or []
        context = _infer_table_context(rows_raw)
        normalized_rows: list[dict[str, Any]] = []
        for ri, cells0 in enumerate(rows_raw[:max_rows_per_table]):
            cells = [_clean_cell_text(str(c)) for c in cells0]
            while cells and not cells[-1].strip():
                cells.pop()
            if not any(c.strip() for c in cells):
                continue
            key = "|".join(cells).lower()
            if key in seen:
                continue
            seen.add(key)
            prev_cells = rows_raw[ri - 1] if ri > 0 else None
            next_cells = rows_raw[ri + 1] if ri + 1 < len(rows_raw) else None
            hints, reasons = _row_role_hints(cells, prev_cells, next_cells, context, ri)
            row_text = " | ".join(cells)
            normalized_rows.append(
                {
                    "id": f"vlm_table_{len(out):02d}_row_{len(normalized_rows):03d}",
                    "row_index": len(normalized_rows),
                    "source_row_index": ri,
                    "cells": cells,
                    "row_text": row_text,
                    "amounts": _amounts(row_text),
                    "role_hints": hints,
                    "hint_reasons": reasons,
                    "prev_row_text": " | ".join(prev_cells) if prev_cells else None,
                    "next_row_text": " | ".join(next_cells) if next_cells else None,
                }
            )
        if normalized_rows:
            out.append(
                {
                    "id": f"vlm_table_{len(out):02d}",
                    "source_path": tbl.get("source_path"),
                    "source_label": tbl.get("source_label"),
                    "block_bbox": tbl.get("block_bbox"),
                    "row_count": len(normalized_rows),
                    "table_context": context,
                    "rows": normalized_rows,
                }
            )
    return out


def _lineize(snippets: list[tuple[str, str]], max_lines: int = 240) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for path, text in snippets:
        for raw_line in re.split(r"\r?\n|<br\s*/?>|</tr>|</p>", text):
            line = re.sub(r"<\s*/td\s*>", " | ", raw_line, flags=re.I)
            line = re.sub(r"<[^>]+>", " ", line)
            line = _clean_cell_text(line).strip(" |\t")
            if not line or len(line) < 2:
                continue
            if len(line) > 220:
                line = line[:220]
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            amts = _amounts(line)
            tags: list[str] = []
            if amts:
                tags.append("has_amount")
            if _has_total_keyword(line):
                tags.append("total_like")
            if PAYMENT_WORD_RE.search(line):
                tags.append("payment_like")
            if CHANGE_WORD_RE.search(line):
                tags.append("change_like")
            if TAX_CONTEXT_WORD_RE.search(line):
                tags.append("tax_like")
            elif PERCENT_RE.search(line) and ITEM_WORD_RE.search(line):
                tags.append("percent_in_product_text_not_tax_by_itself")
            qty_like = _is_mostly_quantity_expression([line])
            if qty_like:
                tags.append("possible_quantity_or_unit_price_note")
            if (
                ITEM_WORD_RE.search(line)
                and amts
                and not (
                    _has_total_keyword(line)
                    or PAYMENT_WORD_RE.search(line)
                    or CHANGE_WORD_RE.search(line)
                    or TAX_CONTEXT_WORD_RE.search(line)
                    or qty_like
                )
            ):
                tags.append("item_price_like")
            out.append(
                {
                    "id": f"vlm_line_{len(out):03d}",
                    "text": line,
                    "amounts": amts,
                    "tags": tags,
                    "path": path,
                }
            )
            if len(out) >= max_lines:
                return out
    return out


def _table_row_keys(structured_tables: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for table in structured_tables:
        for row in table.get("rows") or []:
            txt = row.get("row_text") or " | ".join(row.get("cells") or [])
            if txt:
                keys.add(_norm_key(txt))
                # Also add compact no-pipe version for flattened HTML lines.
                keys.add(_norm_key(txt.replace("|", " ")))
    return {k for k in keys if k}


def _filter_flat_lines_when_structured(
    lines: list[dict[str, Any]], structured_tables: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    """When structured tables exist, avoid giving the LLM duplicated flattened table rows.

    Keep only genuinely non-table total/payment/change lines. Do not repeat item/tax rows as
    generic flattened amount lines because that confused the correction pass.
    """
    if not structured_tables:
        return lines, 0
    table_keys = _table_row_keys(structured_tables)
    kept: list[dict[str, Any]] = []
    suppressed = 0
    for line in lines:
        tags = set(line.get("tags") or [])
        key = _norm_key(line.get("text") or "")
        from_table = any(
            key and (key == tk or key in tk or tk in key) for tk in list(table_keys)[:500]
        )
        if from_table or tags.intersection(
            {"item_price_like", "tax_like", "possible_quantity_or_unit_price_note"}
        ):
            # Structured rows/cells are the authoritative representation.
            suppressed += 1
            continue
        if tags.intersection({"total_like", "payment_like", "change_like"}):
            kept.append(line)
    return kept, suppressed


def _parse_qty_unit_total_from_text(text: str) -> tuple[float, float, float] | None:
    m = re.search(
        r"\b(\d+(?:[,.]\d+)?)\s*(?:STK|STÜCK|STUECK|PCS?|KG|G|L|ML)?\s*(?:x|×|\*|@|à|a)\s*([-+]?\d{1,5}(?:[,.]\d{1,3}))\b",
        text or "",
        re.I,
    )
    if not m:
        return None
    try:
        qty = float(m.group(1).replace(",", "."))
        unit = float(m.group(2).replace(",", "."))
        return qty, unit, round(qty * unit, 2)
    except Exception:
        return None


def _amount_values(row: dict[str, Any]) -> list[float]:
    vals: list[float] = []
    for a in row.get("amounts") or []:
        try:
            vals.append(round(float(a.get("value")), 2))
        except Exception:
            pass
    return vals


def _is_itemish_row(row: dict[str, Any]) -> bool:
    hints = set(row.get("role_hints") or [])
    tags = set(row.get("tags") or [])
    if hints.intersection(
        {
            "possible_total_or_subtotal",
            "possible_payment",
            "possible_change",
            "possible_tax_context",
            "possible_quantity_or_unit_price_note",
            "table_header",
        }
    ):
        return False
    if tags.intersection(
        {
            "total_like",
            "payment_like",
            "change_like",
            "tax_like",
            "possible_quantity_or_unit_price_note",
        }
    ):
        return False
    text = row.get("row_text") or row.get("text") or " ".join(row.get("cells") or [])
    return bool(_amount_values(row) and ITEM_WORD_RE.search(text or ""))


def _row_text_any(row: dict[str, Any]) -> str:
    return str(row.get("row_text") or row.get("text") or " | ".join(row.get("cells") or []) or "")


def _build_quantity_note_link_candidates(
    table_rows: list[dict[str, Any]], flat_lines: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Evidence only: quantity/unit rows that likely explain nearby product totals."""
    rows = list(table_rows) + list(flat_lines)
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        text = _row_text_any(row)
        parsed = _parse_qty_unit_total_from_text(text)
        if not parsed:
            continue
        qty, unit, computed = parsed
        # Treat row as explanatory if it has weak product text or already has quantity-note hint.
        words = [w for w in _cell_words(text) if not QTY_UNIT_WORD_RE.fullmatch(w)]
        weak_product_text = len([w for w in words if len(w) >= 3]) == 0
        if not weak_product_text and "possible_quantity_or_unit_price_note" not in (
            row.get("role_hints") or row.get("tags") or []
        ):
            continue
        matches = []
        for j in range(max(0, i - 4), min(len(rows), i + 5)):
            if j == i:
                continue
            other = rows[j]
            if not _is_itemish_row(other):
                continue
            if any(abs(v - computed) <= 0.03 for v in _amount_values(other)):
                matches.append(
                    {
                        "row_id": other.get("id"),
                        "text": _row_text_any(other),
                        "amount_match": computed,
                    }
                )
        if matches:
            out.append(
                {
                    "candidate_id": f"visual_qty_note_{len(out):03d}",
                    "pattern": "quantity_unit_price_note_explains_adjacent_item_total",
                    "quantity_row_id": row.get("id"),
                    "quantity_row_text": text,
                    "quantity": qty,
                    "unit_price_candidate": unit,
                    "computed_line_total": computed,
                    "matching_item_rows": matches[:4],
                    "generic_rule": "Do not output the quantity/unit-price row as a separate item when a nearby product row already has the computed line total.",
                }
            )
    return out[:60]


def _build_final_price_adjustment_groups(
    table_rows: list[dict[str, Any]], flat_lines: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Evidence only: original price + final/customer price + discount clusters."""
    rows = list(table_rows) + list(flat_lines)
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        text = _row_text_any(row)
        if not _is_itemish_row(row):
            continue
        vals = _amount_values(row)
        if not vals:
            continue
        original = vals[-1]
        nearby = rows[i + 1 : i + 7]
        final_row = None
        discount_row = None
        for cand in nearby:
            ctext = _row_text_any(cand)
            if FINAL_PRICE_RE.search(ctext) and _amount_values(cand):
                final_row = cand
            if DISCOUNT_WORD_RE.search(ctext) and _amount_values(cand):
                discount_row = cand
        if not final_row:
            continue
        final_val = _amount_values(final_row)[-1]
        discount_val = None
        if discount_row and _amount_values(discount_row):
            # A discount may be encoded positive or negative in OCR/VLM; semantically it reduces original.
            discount_val = -abs(_amount_values(discount_row)[-1])
        relationship_ok = (
            discount_val is None or abs(round(original + discount_val, 2) - final_val) <= 0.05
        )
        out.append(
            {
                "candidate_id": f"visual_final_price_{len(out):03d}",
                "pattern": "original_price_final_price_adjustment_group",
                "product_row_id": row.get("id"),
                "product_row_text": text,
                "original_or_reference_price": original,
                "final_price_row_id": final_row.get("id"),
                "final_price_row_text": _row_text_any(final_row),
                "final_sale_price_candidate": final_val,
                "discount_row_id": discount_row.get("id") if discount_row else None,
                "discount_row_text": _row_text_any(discount_row) if discount_row else None,
                "discount_candidate": discount_val,
                "relationship_ok": relationship_ok,
                "generic_rule": "Output one item at the final/customer price. Do not output original price, final price and discount as three separate contributing items.",
            }
        )
    return out[:40]


def _rate_from_text(text: str) -> float | None:
    m = PERCENT_RE.search(text or "")
    if not m:
        m = re.search(r"\b(\d{1,2}(?:[,.]\d+)?)\b", text or "")
    if not m:
        return None
    raw = m.group(0).replace("%", "").strip().replace(",", ".")
    try:
        val = float(raw)
    except Exception:
        return None
    if 0 < val < 1:
        val *= 100.0
    if 1 <= val <= 30:
        return round(val, 2)
    return None


def _build_tax_table_candidates(table_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Evidence only: map tax rows to rate/net/tax/gross where headers make it possible."""
    candidates: list[dict[str, Any]] = []
    for row in table_rows:
        hints = set(row.get("role_hints") or [])
        if "possible_tax_context" not in hints:
            continue
        cells = row.get("cells") or []
        row_text = _row_text_any(row)
        if not _amount_values(row) and not PERCENT_RE.search(row_text):
            continue
        # Skip pure tax headers.
        if "table_header" in hints or "tax_table_header" in hints:
            continue
        ctx = row.get("table_context") or {}
        header = ctx.get("header_cells") or []
        mapped: dict[str, float | None] = {
            "rate": _rate_from_text(row_text),
            "net": None,
            "tax": None,
            "gross": None,
        }
        for ci, cell in enumerate(cells):
            vals = _amounts(str(cell))
            if not vals:
                continue
            v = vals[-1].get("value")
            h = str(header[ci]) if ci < len(header) else ""
            if NET_WORD_RE.search(h):
                mapped["net"] = v
            elif GROSS_WORD_RE.search(h) or re.search(r"\bBRUTTO\b", h, re.I):
                mapped["gross"] = v
            elif TAX_AMOUNT_WORD_RE.search(h):
                # If the header is MwSt/Tax and not a rate-only header, this is the tax amount column.
                mapped["tax"] = v
        # Fallback for common tax rows with rate + 3 amounts and no usable header: do not guess too hard, but include amounts.
        candidates.append(
            {
                "row_id": row.get("id"),
                "row_text": row_text,
                "rate": mapped["rate"],
                "net": mapped["net"],
                "tax": mapped["tax"],
                "gross": mapped["gross"],
                "amounts": row.get("amounts") or [],
                "generic_rule": "Use the tax/MwSt/VAT amount column for tax_total, not the rate, net or gross column.",
            }
        )
    tax_sum = (
        round(sum(float(c["tax"]) for c in candidates if c.get("tax") is not None), 2)
        if any(c.get("tax") is not None for c in candidates)
        else None
    )
    return {"rows": candidates[:80], "tax_total_candidate": tax_sum}


def _is_totalish_row(row: dict[str, Any]) -> bool:
    hints = set(row.get("role_hints") or [])
    tags = set(row.get("tags") or [])
    txt = _row_text_any(row)
    return "possible_total_or_subtotal" in hints or "total_like" in tags or _has_total_keyword(txt)


def _is_paymentish_row(row: dict[str, Any]) -> bool:
    hints = set(row.get("role_hints") or [])
    tags = set(row.get("tags") or [])
    txt = _row_text_any(row)
    return (
        "possible_payment" in hints or "payment_like" in tags or bool(PAYMENT_WORD_RE.search(txt))
    )


def _is_changeish_row(row: dict[str, Any]) -> bool:
    hints = set(row.get("role_hints") or [])
    tags = set(row.get("tags") or [])
    txt = _row_text_any(row)
    return "possible_change" in hints or "change_like" in tags or bool(CHANGE_WORD_RE.search(txt))


def _netto_penalty(row: dict[str, Any]) -> bool:
    txt = _row_text_any(row)
    return bool(NET_WORD_RE.search(txt))


def _build_total_payment_reconciliation_candidates(
    table_rows: list[dict[str, Any]],
    flat_lines: list[dict[str, Any]],
    final_price_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evidence only: printed total candidates supported by payment minus change.

    This gives the LLM a stronger total priority signal without deterministically
    constructing the final receipt JSON.
    """
    rows = list(table_rows) + list(flat_lines)
    totals = []
    payments = []
    changes = []
    for row in rows:
        vals = _amount_values(row)
        if not vals:
            continue
        entry = {
            "row_id": row.get("id"),
            "text": _row_text_any(row),
            "amount": vals[-1],
            "netto_or_net_hint": _netto_penalty(row),
        }
        if _is_totalish_row(row):
            totals.append(entry)
        if _is_paymentish_row(row):
            payments.append(entry)
        if _is_changeish_row(row):
            changes.append(entry)
    out = []
    final_vals = []
    for g in final_price_groups or []:
        try:
            if g.get("final_sale_price_candidate") is not None:
                final_vals.append(round(float(g.get("final_sale_price_candidate")), 2))
        except Exception:
            pass
    for p in payments[:12]:
        for c in changes[:12]:
            settle = round(float(p["amount"]) - abs(float(c["amount"])), 2)
            matching_totals = []
            for t in totals[:30]:
                if abs(float(t["amount"]) - settle) <= 0.05:
                    matching_totals.append(t)
            final_match = any(abs(v - settle) <= 0.05 for v in final_vals)
            if matching_totals or final_match:
                out.append(
                    {
                        "candidate_id": f"visual_payment_recon_{len(out):03d}",
                        "payment_row": p,
                        "change_row": c,
                        "settlement_total": settle,
                        "matching_printed_total_rows": matching_totals[:6],
                        "matches_final_price_adjustment_group": final_match,
                        "generic_rule": "When payment - abs(change) matches a printed total/final-price candidate, prefer that value as grand_total over NETTO/net/tax rows.",
                    }
                )
    return out[:20]


def build_visual_evidence(
    vlm_result: dict[str, Any], validation_report: dict[str, Any], max_chars: int = 12000
) -> dict[str, Any]:
    raw = vlm_result.get("raw_result") if isinstance(vlm_result, dict) else None
    snippets = _walk(raw)
    raw_lines = _lineize(snippets)
    structured_tables = _build_structured_tables(raw)
    table_rows = [r for t in structured_tables for r in (t.get("rows") or [])]
    lines, suppressed_flattened_line_count = _filter_flat_lines_when_structured(
        raw_lines, structured_tables
    )
    issue_codes = [
        str(i.get("code")) for i in validation_report.get("issues") or [] if isinstance(i, dict)
    ]
    amount_lines = [line for line in lines if line.get("amounts")]
    quantity_hint_rows = [
        r
        for r in table_rows
        if "possible_quantity_or_unit_price_note" in (r.get("role_hints") or [])
    ]
    item_candidate_rows = [
        r for r in table_rows if "possible_item_charge_or_credit" in (r.get("role_hints") or [])
    ]
    total_payment_rows = [
        r
        for r in table_rows
        if set(r.get("role_hints") or []).intersection(
            {"possible_total_or_subtotal", "possible_payment", "possible_change"}
        )
    ]
    tax_context_rows = [
        r for r in table_rows if "possible_tax_context" in (r.get("role_hints") or [])
    ]
    quantity_note_link_candidates = _build_quantity_note_link_candidates(table_rows, lines)
    final_price_adjustment_groups = _build_final_price_adjustment_groups(table_rows, lines)
    tax_table_candidates = _build_tax_table_candidates(table_rows)
    total_payment_reconciliation_candidates = _build_total_payment_reconciliation_candidates(
        table_rows, lines, final_price_adjustment_groups
    )
    evidence = {
        "schema_version": "v14_11_visual_semantic_evidence_1",
        "status": vlm_result.get("status") if isinstance(vlm_result, dict) else "unknown",
        "backend": vlm_result.get("backend") if isinstance(vlm_result, dict) else None,
        "runner": vlm_result.get("runner") if isinstance(vlm_result, dict) else None,
        "engine": vlm_result.get("engine") if isinstance(vlm_result, dict) else None,
        "device": vlm_result.get("device") if isinstance(vlm_result, dict) else None,
        "triggered_by_issue_codes": issue_codes,
        "semantic_guidance": [
            "Structured VLM tables are primary evidence. Do not prefer duplicated flattened lines over table cells.",
            "Role hints are weak diagnostics, not forced decisions. The correction LLM must classify each row itself using header/cell/neighbour context.",
            "A row in a description+total table can be an item even if it has a quantity column. Quantity columns do not make item rows into quantity-note rows.",
            "A quantity/unit-price note is generally a row that mainly shows quantity, multiplier, unit, or unit price and has weak product description; it may explain a nearby item but is not automatically an item.",
            "Rows in explicit tax/net/gross/VAT/MwSt tables are tax evidence, not item lines. Tax total is the tax amount column, not the rate or gross/net columns.",
            "Total-like rows outrank payment-like rows when choosing grand_total; payment/change rows explain settlement, not the receipt total.",
            "When payment minus absolute change equals a printed Bonsumme/Summe/final-price candidate, that settlement total is stronger grand_total evidence than any NETTO/net/tax row.",
            "Only output a row as an item when it represents one charge or credit that contributes once to the printed receipt total.",
            "Final-price adjustment groups mean original/list price and final/customer price are alternatives in one product group; do not sum both.",
            "Quantity-note link candidates identify rows like quantity × unit price that explain a nearby line total and normally should not contribute separately.",
        ],
        "summary": {
            "raw_line_count": len(raw_lines),
            "line_count_after_structured_suppression": len(lines),
            "suppressed_flattened_line_count": suppressed_flattened_line_count,
            "amount_line_count": len(amount_lines),
            "structured_table_count": len(structured_tables),
            "structured_table_row_count": len(table_rows),
            "quantity_hint_row_count": len(quantity_hint_rows),
            "item_candidate_row_count": len(item_candidate_rows),
            "tax_context_row_count": len(tax_context_rows),
            "quantity_note_link_candidate_count": len(quantity_note_link_candidates),
            "final_price_adjustment_group_count": len(final_price_adjustment_groups),
            "tax_table_candidate_count": len(tax_table_candidates.get("rows") or []),
            "total_payment_reconciliation_candidate_count": len(
                total_payment_reconciliation_candidates
            ),
            "has_payment_like": any("payment_like" in (line.get("tags") or []) for line in lines)
            or any("possible_payment" in (r.get("role_hints") or []) for r in table_rows),
            "has_change_like": any("change_like" in (line.get("tags") or []) for line in lines)
            or any("possible_change" in (r.get("role_hints") or []) for r in table_rows),
            "has_tax_like": any("tax_like" in (line.get("tags") or []) for line in lines)
            or any("possible_tax_context" in (r.get("role_hints") or []) for r in table_rows),
        },
        "structured_tables": structured_tables[:8],
        "quantity_hint_rows": quantity_hint_rows[:80],
        "item_candidate_rows": item_candidate_rows[:120],
        "total_payment_rows": total_payment_rows[:100],
        "tax_context_rows": tax_context_rows[:100],
        "quantity_note_link_candidates": quantity_note_link_candidates[:60],
        "final_price_adjustment_groups": final_price_adjustment_groups[:40],
        "tax_table_candidates": tax_table_candidates,
        "total_payment_reconciliation_candidates": total_payment_reconciliation_candidates[:30],
        "lines": lines[:80],
        "amount_lines": amount_lines[:60],
        "payment_change_lines": [
            line
            for line in lines
            if set(line.get("tags") or []).intersection({"payment_like", "change_like"})
        ][:60],
        "tax_like_lines": [line for line in lines if "tax_like" in (line.get("tags") or [])][:40],
        "item_price_like_lines": []
        if structured_tables
        else [line for line in lines if "item_price_like" in (line.get("tags") or [])][:100],
        "engine_error": vlm_result.get("error") if isinstance(vlm_result, dict) else None,
    }
    text = visual_evidence_to_prompt_text(evidence)
    if len(text) > max_chars:
        evidence["lines"] = evidence["lines"][:30]
        evidence["amount_lines"] = evidence["amount_lines"][:30]
        evidence["structured_tables"] = evidence["structured_tables"][:4]
        for table in evidence["structured_tables"]:
            table["rows"] = (table.get("rows") or [])[:70]
            table["row_count_in_prompt"] = len(table["rows"])
    return evidence


def visual_evidence_to_prompt_text(evidence: dict[str, Any]) -> str:
    parts: list[str] = []
    parts.append(
        "VISUAL/VLM EVIDENCE (VLM-first when enabled; correction evidence when validation-triggered):"
    )
    parts.append(
        json.dumps(
            {
                k: evidence.get(k)
                for k in (
                    "schema_version",
                    "status",
                    "backend",
                    "runner",
                    "engine",
                    "device",
                    "triggered_by_issue_codes",
                    "summary",
                    "engine_error",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if evidence.get("semantic_guidance"):
        parts.append("\nGeneral VLM/region evidence guidance for the LLM:")
        for g in evidence.get("semantic_guidance") or []:
            parts.append(f"- {g}")

    # V14.13.3: region crop OCR is the highest-priority item evidence when it
    # reconciles to a printed total. VLM locates regions; OCR reads them.
    preferred_blocks = evidence.get("preferred_item_blocks") or []
    best_block = evidence.get("best_preferred_item_block") or (
        preferred_blocks[0] if preferred_blocks else None
    )
    if best_block:
        parts.append("\nPREFERRED ITEM BLOCK FROM VLM-REGION CROP RE-OCR (PRIMARY ITEM EVIDENCE):")
        pt = (
            best_block.get("printed_total")
            if isinstance(best_block.get("printed_total"), dict)
            else None
        )
        parts.append(
            json.dumps(
                {
                    "region_id": best_block.get("region_id"),
                    "crop_path": best_block.get("crop_path"),
                    "candidate_sum": best_block.get("candidate_sum"),
                    "printed_total": pt,
                    "balanced_to_printed_total": best_block.get("balanced_to_printed_total"),
                    "confidence": best_block.get("confidence"),
                    "method": best_block.get("method"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        rows = best_block.get("rows") or []
        if rows:
            parts.append(
                "Preferred item rows. If this block is balanced_to_printed_total=true, do not omit these rows unless explicit evidence proves a row is not a purchased item:"
            )
            for row in rows[:80]:
                parts.append(
                    f"- {row.get('row_id')}: {row.get('description_candidate')} => {row.get('amount')} raw={row.get('amount_raw')} sources={row.get('source_line_ids')}"
                )
        qty_links = best_block.get("quantity_note_links") or []
        if qty_links:
            parts.append(
                "Linked quantity/unit-price notes from the same crop; these are supporting evidence and usually not standalone items:"
            )
            for q in qty_links[:50]:
                parts.append(
                    f"- {q.get('quantity_row_id')}: {q.get('quantity_text')} -> {q.get('linked_item_description')} contributes={q.get('contributes_hint')}"
                )
        region_final_groups = best_block.get("final_price_adjustment_groups") or []
        if region_final_groups:
            parts.append(
                "Region final-price groups already applied in the preferred block; output one final customer price and do not add barcode/reference rows as separate items:"
            )
            for g in region_final_groups[:20]:
                parts.append(
                    f"- {g.get('candidate_id')}: product=[{g.get('product_description_candidate')}] original={g.get('original_or_reference_price')} final={g.get('final_sale_price_candidate')} final_row={g.get('final_price_row_id')}"
                )

    # Compact semantic groups are first because they reduce token pressure and tell the LLM what matters.
    qty_links = evidence.get("quantity_note_link_candidates") or []
    if qty_links:
        parts.append("\nQuantity-note link candidates (usually explanatory, not standalone items):")
        for c in qty_links[:30]:
            parts.append(
                f"- {c.get('candidate_id')}: qty_row={c.get('quantity_row_id')} text=[{c.get('quantity_row_text')}] qty={c.get('quantity')} unit={c.get('unit_price_candidate')} computed_total={c.get('computed_line_total')} matches={json.dumps(c.get('matching_item_rows') or [], ensure_ascii=False)}"
            )

    final_groups = evidence.get("final_price_adjustment_groups") or []
    if final_groups:
        parts.append(
            "\nFinal-price adjustment groups (choose final/customer price once; if final price is used, do not subtract the same printed discount again):"
        )
        for c in final_groups[:25]:
            parts.append(
                f"- {c.get('candidate_id')}: product=[{c.get('product_row_text')}] original={c.get('original_or_reference_price')} final=[{c.get('final_price_row_text')}] final_price={c.get('final_sale_price_candidate')} discount=[{c.get('discount_row_text')}] discount={c.get('discount_candidate')} relationship_ok={c.get('relationship_ok')}"
            )

    tax_candidates = evidence.get("tax_table_candidates") or {}
    tax_rows = tax_candidates.get("rows") if isinstance(tax_candidates, dict) else []
    if tax_rows:
        parts.append(
            "\nTax table candidates (tax_total is tax/MwSt/VAT amount column, not rate/net/gross):"
        )
        parts.append(
            json.dumps(
                {
                    "tax_total_candidate": tax_candidates.get("tax_total_candidate"),
                    "rows": tax_rows[:30],
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    payment_recon = evidence.get("total_payment_reconciliation_candidates") or []
    if payment_recon:
        parts.append(
            "\nTotal/payment/change reconciliation candidates (prefer settlement_total as grand_total when supported by printed total/final price):"
        )
        parts.append(json.dumps(payment_recon[:20], ensure_ascii=False, indent=2))


    table_arbitration = (
        evidence.get("table_arbitration")
        if isinstance(evidence.get("table_arbitration"), dict)
        else None
    )
    if table_arbitration:
        parts.append("\nTABLE EVIDENCE ARBITRATION (OCR/VLM CROSS-CHECK):")
        parts.append(
            "Use this when VLM table rows conflict with OCR layout rows. If warnings mention row-shift or alignment risk, prefer OCR layout item/price pairing for the affected area unless VLM arithmetic is clearly stronger."
        )
        compact_arbitration = {
            "summary": table_arbitration.get("summary"),
            "warnings": (table_arbitration.get("warnings") or [])[:8],
            "ocr_layout_item_candidates": (
                table_arbitration.get("ocr_layout_item_candidates") or []
            )[:70],
            "quantity_note_candidates": (table_arbitration.get("quantity_note_candidates") or [])[
                :35
            ],
            "product_percent_not_tax_rows": (
                table_arbitration.get("product_percent_not_tax_rows") or []
            )[:25],
            "guidance": table_arbitration.get("guidance") or [],
        }
        parts.append(json.dumps(compact_arbitration, ensure_ascii=False, indent=2))

    structured_tables = evidence.get("structured_tables") or []
    if structured_tables:
        parts.append(
            "\nStructured VLM table rows/cells (PRIMARY VLM evidence; classify rows using header/cell/neighbour context):"
        )
        for table in structured_tables[:6]:
            ctx = table.get("table_context") or {}
            ctx_small = {
                k: ctx.get(k)
                for k in (
                    "header_index",
                    "header_cells",
                    "tax_header_indices",
                    "first_tax_header_index",
                    "first_total_row_index",
                    "explicit_tax_table",
                    "item_table",
                    "qty_column_table",
                    "amount_col_indices",
                    "description_col_indices",
                    "qty_col_indices",
                    "tax_col_indices",
                )
            }
            parts.append(
                f"TABLE {table.get('id')} source={table.get('source_path')} bbox={table.get('block_bbox')} rows={table.get('row_count')} context={json.dumps(ctx_small, ensure_ascii=False)}"
            )
            for row in (table.get("rows") or [])[:80]:
                amts = row.get("amounts") or []
                amt_txt = " ".join(f"{a.get('raw')}=>{a.get('value')}" for a in amts[:4])
                hint_txt = ",".join(row.get("role_hints") or [])
                reason_txt = "; ".join((row.get("hint_reasons") or [])[:2])
                cells_txt = " || ".join(row.get("cells") or [])
                parts.append(
                    f"- {row.get('id')}: cells=[{cells_txt}]"
                    + (f" amounts=[{amt_txt}]" if amt_txt else "")
                    + (f" hints=[{hint_txt}]" if hint_txt else "")
                    + (f" reason={reason_txt}" if reason_txt else "")
                )
                if "possible_quantity_or_unit_price_note" in (row.get("role_hints") or []):
                    if row.get("prev_row_text"):
                        parts.append(f"  prev: {row.get('prev_row_text')}")
                    if row.get("next_row_text"):
                        parts.append(f"  next: {row.get('next_row_text')}")

    def add_section(title: str, rows: list[dict[str, Any]], limit: int) -> None:
        if not rows:
            return
        parts.append(f"\n{title}:")
        for row in rows[:limit]:
            amts = row.get("amounts") or []
            amt_txt = " ".join(f"{a.get('raw')}=>{a.get('value')}" for a in amts[:4])
            tag_txt = ",".join(row.get("tags") or [])
            parts.append(
                f"- {row.get('id')}: {row.get('text')}"
                + (f" [{amt_txt}]" if amt_txt else "")
                + (f" {{{tag_txt}}}" if tag_txt else "")
            )

    add_section(
        "Non-table payment/change-like visual lines", evidence.get("payment_change_lines") or [], 30
    )
    if not structured_tables:
        add_section("Tax-like visual lines", evidence.get("tax_like_lines") or [], 30)
        add_section("Item/price-like visual lines", evidence.get("item_price_like_lines") or [], 60)
        add_section("Other amount-bearing visual lines", evidence.get("amount_lines") or [], 60)
    else:
        parts.append(
            "\nFlattened VLM item/tax/amount rows were suppressed because structured table rows are available and less ambiguous."
        )
    return "\n".join(parts)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
