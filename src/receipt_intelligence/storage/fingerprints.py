"""Receipt fingerprints and duplicate-scoring helpers."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

from receipt_intelligence.receipt_compat import (
    item_line_total,
    receipt_date,
    receipt_grand_total,
    receipt_time,
)

from .normalization import (
    as_float,
    as_str,
    extract_item_description,
    first_present,
    normalize_merchant_name,
    normalize_text,
)


def file_sha256(path: Path | str | None) -> str | None:
    if not path:
        return None
    try:
        file_path = Path(path)
        if not file_path.exists() or not file_path.is_file():
            return None
        digest = hashlib.sha256()
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None


def receipt_core(receipt: dict[str, Any]) -> dict[str, Any]:
    merchant = receipt.get("merchant") if isinstance(receipt.get("merchant"), dict) else {}
    items = receipt.get("items") if isinstance(receipt.get("items"), list) else []
    merchant_name = as_str(first_present(merchant.get("name"), receipt.get("merchant_name")))
    merchant_normalized = normalize_merchant_name(merchant_name)
    grand_total = as_float(receipt_grand_total(receipt))
    item_parts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = normalize_text(extract_item_description(item))
        amount = as_float(item_line_total(item))
        if name and amount is not None:
            item_parts.append(f"{name}:{amount:.2f}")
    item_signature = "|".join(item_parts[:80])
    fingerprint = "|".join(
        [
            merchant_normalized or "",
            as_str(receipt_date(receipt)) or "",
            as_str(receipt_time(receipt)) or "",
            f"{grand_total:.2f}" if grand_total is not None else "",
            str(len(item_parts)),
        ]
    )
    return {
        "merchant_name": merchant_name,
        "merchant_normalized": merchant_normalized,
        "receipt_date": as_str(receipt_date(receipt)),
        "receipt_time": as_str(receipt_time(receipt)),
        "grand_total": grand_total,
        "item_count": len(item_parts),
        "item_signature": item_signature,
        "content_fingerprint": fingerprint,
    }


def item_overlap(signature_a: str | None, signature_b: str | None) -> float:
    def parts(signature: str | None) -> set[str]:
        output = set()
        for part in (signature or "").split("|"):
            name = part.split(":", 1)[0].strip()
            if name:
                output.add(name)
        return output

    left, right = parts(signature_a), parts(signature_b)
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def duplicate_score_against_row(
    core: dict[str, Any],
    file_hash: str | None,
    row: sqlite3.Row,
) -> tuple[float, list[str]]:
    def row_value(key: str, default: Any = None) -> Any:
        try:
            return row[key] if key in row.keys() else default
        except Exception:
            return default

    score = 0.0
    reasons: list[str] = []
    if file_hash and row_value("file_sha256") == file_hash:
        score += 100
        reasons.append("exact_file_hash_match")
    if core.get("merchant_normalized") and row_value("merchant_normalized") == core.get(
        "merchant_normalized"
    ):
        score += 20
        reasons.append("same_merchant")
    if (
        core.get("grand_total") is not None
        and row_value("grand_total") is not None
        and abs(float(row_value("grand_total")) - float(core["grand_total"])) <= 0.01
    ):
        score += 25
        reasons.append("same_total")
    if core.get("receipt_date") and row_value("receipt_date") == core.get("receipt_date"):
        score += 20
        reasons.append("same_date")
    if core.get("receipt_time") and row_value("receipt_time") == core.get("receipt_time"):
        score += 15
        reasons.append("same_time")
    overlap = item_overlap(core.get("item_signature"), row_value("item_signature"))
    if overlap >= 0.75:
        score += 30
        reasons.append(f"high_item_overlap:{overlap:.2f}")
    elif overlap >= 0.45:
        score += 15
        reasons.append(f"medium_item_overlap:{overlap:.2f}")
    return min(score, 100.0), reasons
