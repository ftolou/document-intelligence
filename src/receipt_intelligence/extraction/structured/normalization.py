"""Conservative normalization for schema-constrained Gemma task answers."""

from __future__ import annotations

import copy
import re
from decimal import Decimal, InvalidOperation
from typing import Any

_NULL_PLACEHOLDERS = frozenset({"", "-", "?", "n/a", "na", "none", "null", "unknown"})
_CURRENCY = re.compile(r"^[A-Z]{3}$")
_AGGREGATE_DISCOUNT = re.compile(
    r"\b(?:rabatt(?:e)?\s*(?:gesamt|summe)|gesamt(?:er|e|es)?\s*rabatt|gesamtrabatt|"
    r"discount\s*(?:total|sum)|total\s*discount|total\s*savings|"
    r"ersparnis\s*(?:gesamt|summe)|sie\s+sparen)\b",
    re.IGNORECASE,
)


def normalize_task_answer(
    *,
    task_name: str,
    answer: dict[str, Any],
    schema: dict[str, Any],
    evidence: str,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Normalize typed placeholders and reject unsupported aggregate discount claims.

    This function never calculates values. It only normalizes schema-nullable placeholders,
    canonicalizes valid ISO-style currencies, and removes receipt-level discount values when
    the transcription has no explicit aggregate-discount label.
    """

    normalized = copy.deepcopy(answer)
    changes: list[str] = []
    _normalize_node(normalized, schema, schema, path="$", changes=changes)

    if task_name == "discount_total" and normalized.get("discount_total") is not None:
        if not _has_supported_aggregate_discount(
            evidence, normalized.get("discount_total")
        ):
            normalized["discount_total"] = None
            changes.append("discount_total_removed_without_explicit_aggregate_evidence")

    return normalized, tuple(changes)


_MONEY_TOKEN = re.compile(
    r"(?<![0-9])[-+]?(?:[0-9]{1,3}(?:[ .][0-9]{3})*|[0-9]+)[.,][0-9]{2}(?![0-9])"
)


def _has_supported_aggregate_discount(evidence: str, value: Any) -> bool:
    expected = _decimal(value)
    if expected is None:
        return False
    for line in evidence.splitlines():
        if not _AGGREGATE_DISCOUNT.search(line):
            continue
        for token in _MONEY_TOKEN.findall(line):
            printed = _decimal(token)
            if printed is not None and abs(abs(printed) - abs(expected)) <= Decimal("0.01"):
                return True
    return False


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    token = str(value).strip().replace(" ", "")
    if not token:
        return None
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
    return number if number.is_finite() else None


def _normalize_node(
    value: Any,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    *,
    path: str,
    changes: list[str],
) -> Any:
    resolved = _resolve_schema(schema, root_schema)
    if isinstance(value, dict):
        properties = resolved.get("properties")
        if isinstance(properties, dict):
            for key, child_schema in properties.items():
                if key not in value or not isinstance(child_schema, dict):
                    continue
                child_path = f"{path}.{key}"
                child = value[key]
                child_resolved = _resolve_schema(child_schema, root_schema)
                if isinstance(child, str) and _allows_null(child_resolved):
                    stripped = child.strip()
                    if stripped.casefold() in _NULL_PLACEHOLDERS:
                        value[key] = None
                        changes.append(f"{child_path}:placeholder_to_null")
                        continue
                if key == "currency" and isinstance(child, str):
                    currency = child.strip().upper()
                    if _CURRENCY.fullmatch(currency):
                        if currency != child:
                            changes.append(f"{child_path}:currency_canonicalized")
                        value[key] = currency
                    elif _allows_null(child_resolved):
                        value[key] = None
                        changes.append(f"{child_path}:invalid_currency_to_null")
                        continue
                value[key] = _normalize_node(
                    value[key],
                    child_schema,
                    root_schema,
                    path=child_path,
                    changes=changes,
                )
        return value
    if isinstance(value, list):
        item_schema = resolved.get("items")
        if isinstance(item_schema, dict):
            for index, child in enumerate(value):
                value[index] = _normalize_node(
                    child,
                    item_schema,
                    root_schema,
                    path=f"{path}[{index}]",
                    changes=changes,
                )
    return value


def _resolve_schema(schema: dict[str, Any], root_schema: dict[str, Any]) -> dict[str, Any]:
    reference = schema.get("$ref")
    if not isinstance(reference, str) or not reference.startswith("#/"):
        return schema
    current: Any = root_schema
    for token in reference[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or token not in current:
            return schema
        current = current[token]
    return current if isinstance(current, dict) else schema


def _allows_null(schema: dict[str, Any]) -> bool:
    declared = schema.get("type")
    if declared == "null":
        return True
    if isinstance(declared, list) and "null" in declared:
        return True
    for keyword in ("oneOf", "anyOf"):
        alternatives = schema.get(keyword)
        if isinstance(alternatives, list) and any(
            isinstance(candidate, dict) and _allows_null(candidate)
            for candidate in alternatives
        ):
            return True
    return False


__all__ = ["normalize_task_answer"]
