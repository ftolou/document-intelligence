from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

MONEY_QUANTUM = Decimal("0.01")
_ROW_PATTERN = re.compile(r"^(R\d{4})\s*::\s*(.*)$")
_MONEY_PATTERN = re.compile(
    r"(?<![0-9])[-+]?(?:[0-9]{1,3}(?:[ .][0-9]{3})*|[0-9]+)[.,][0-9]{2}(?![0-9])"
)


def parse_rows(transcription: str) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for raw_line in transcription.splitlines():
        line = raw_line.strip()
        if not line or line in {"BEGIN_RECEIPT", "END_RECEIPT"}:
            continue
        match = _ROW_PATTERN.match(line)
        if not match:
            warnings.append(f"Unparsed line: {line}")
            continue
        row_id, text = match.groups()
        if row_id in seen:
            warnings.append(f"Duplicate row ID: {row_id}")
        seen.add(row_id)
        rows.append({"row_id": row_id, "text": text.strip()})
    if not rows:
        raise ValueError("Source evidence contains no parseable R#### :: rows")
    return rows, warnings


def row_map(transcription: str) -> tuple[dict[str, str], list[str]]:
    rows, warnings = parse_rows(transcription)
    return {row["row_id"]: row["text"] for row in rows}, warnings


def literal_occurs(value: Any, row_text: str) -> bool:
    return isinstance(value, str) and bool(value) and value in row_text


def parse_decimal_literal(value: Any, *, percent: bool = False) -> Decimal | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if percent:
        text = text.replace("%", "").strip()
    matches = _MONEY_PATTERN.findall(text)
    if not matches and percent:
        simple = re.findall(r"[-+]?[0-9]+(?:[.,][0-9]+)?", text)
        if len(simple) == 1:
            matches = [simple[0]]
    if len(matches) != 1:
        return None
    token = matches[0].replace(" ", "")
    if "," in token and "." in token:
        if token.rfind(",") > token.rfind("."):
            token = token.replace(".", "").replace(",", ".")
        else:
            token = token.replace(",", "")
    else:
        token = token.replace(",", ".")
    try:
        number = Decimal(token)
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite():
        return None
    return number if percent else number.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)


def money_float(value: Decimal) -> float:
    return float(value.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP))
