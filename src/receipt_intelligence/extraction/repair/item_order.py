"""Stable printed-order reconstruction for multi-source item recovery.

The recovery layers may combine rows from the main parser, region re-OCR,
table arbitration, and right-column price recovery.  Those sources can use
different line-id namespaces, so sorting by one namespace or by product name is
not reliable.  This module merges the relative item sequences emitted by each
source and uses intrinsic y/line hints only as a fallback.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

_ALPHA_NUM_RE = re.compile(r"[^A-Z0-9]+")
_SOURCE_NUMBER_RE = re.compile(r"(?:^|_)(?:line|row)_(\d+)(?:$|_)", re.IGNORECASE)
_TRAILING_NUMBER_RE = re.compile(r"(?:^|_)(\d+)$")


def normalized_item_key(value: Any) -> str:
    """Return a source-independent key for a product row."""
    if isinstance(value, dict):
        value = (
            value.get("product_description")
            or value.get("description_candidate")
            or value.get("description")
            or value.get("raw_description")
            or value.get("text")
            or ""
        )
    text = str(value or "").upper()
    text = text.replace("Ä", "AE").replace("Ö", "OE").replace("Ü", "UE").replace("ß", "SS")
    return re.sub(r"\s+", " ", _ALPHA_NUM_RE.sub(" ", text)).strip()


def _source_ids(record: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for value in record.get("source_line_ids") or []:
        if value is not None:
            values.append(str(value))
    for field in ("table_interpretation_source_row_id", "row_id", "layout_row_id", "id"):
        value = record.get(field)
        if value is not None and str(value).strip():
            values.append(str(value))
    return values


def source_position(record: dict[str, Any]) -> tuple[int, float] | None:
    """Return an intrinsic printed-position hint when a source provides one."""
    for field in ("printed_order", "source_order", "recovery_layout_index", "layout_index"):
        try:
            if record.get(field) is not None:
                return (0, float(record[field]))
        except (TypeError, ValueError):
            pass

    for field in (
        "recovery_product_y_center",
        "source_y_center",
        "y_center",
        "recovery_amount_y_center",
    ):
        try:
            if record.get(field) is not None:
                return (1, float(record[field]))
        except (TypeError, ValueError):
            pass

    numbers: list[int] = []
    for value in _source_ids(record):
        match = _SOURCE_NUMBER_RE.search(value)
        if match is None:
            match = _TRAILING_NUMBER_RE.search(value)
        if match is not None:
            numbers.append(int(match.group(1)))
    if numbers:
        return (2, float(min(numbers)))
    return None


def sort_records_by_source_position(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable-sort records with explicit source positions, preserving unknowns."""
    prepared = list(records)

    def key(pair: tuple[int, dict[str, Any]]) -> tuple[int, int, float, int]:
        index, record = pair
        position = source_position(record)
        if position is None:
            return (1, 9, float(index), index)
        return (0, position[0], position[1], index)

    return [record for _, record in sorted(enumerate(prepared), key=key)]


def _sequence_tokens(sequence: Sequence[dict[str, Any]]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    tokens: list[tuple[str, int]] = []
    for record in sequence:
        key = normalized_item_key(record)
        if not key:
            continue
        occurrence = counts.get(key, 0)
        counts[key] = occurrence + 1
        tokens.append((key, occurrence))
    return tokens


def merge_printed_sequences(
    sequences: Iterable[Sequence[dict[str, Any]]],
) -> list[tuple[str, int]]:
    """Merge relative source sequences into one stable canonical item sequence.

    The first non-empty sequence is the dominant source.  Unseen items from
    later sources are inserted next to the closest already-known predecessor or
    successor, rather than being appended or alphabetically sorted.
    """
    canonical: list[tuple[str, int]] = []
    for sequence in sequences:
        tokens = _sequence_tokens(sequence)
        if not tokens:
            continue
        if not canonical:
            canonical.extend(tokens)
            continue
        for token_index, token in enumerate(tokens):
            if token in canonical:
                continue
            predecessor = next(
                (
                    candidate
                    for candidate in reversed(tokens[:token_index])
                    if candidate in canonical
                ),
                None,
            )
            successor = next(
                (candidate for candidate in tokens[token_index + 1 :] if candidate in canonical),
                None,
            )
            if predecessor is not None and successor is not None:
                predecessor_index = canonical.index(predecessor) + 1
                successor_index = canonical.index(successor)
                canonical.insert(min(predecessor_index, successor_index), token)
            elif predecessor is not None:
                canonical.insert(canonical.index(predecessor) + 1, token)
            elif successor is not None:
                canonical.insert(canonical.index(successor), token)
            else:
                canonical.append(token)
    return canonical


def sort_items_by_printed_order(
    items: Sequence[dict[str, Any]],
    *,
    sequences: Iterable[Sequence[dict[str, Any]]] = (),
) -> list[dict[str, Any]]:
    """Return item copies in the best available printed receipt order."""
    prepared = list(items)
    source_sequences = [list(sequence) for sequence in sequences if sequence]
    if not source_sequences:
        source_sequences = [sort_records_by_source_position(prepared)]
    canonical = merge_printed_sequences(source_sequences)
    rank = {token: index for index, token in enumerate(canonical)}

    counts: dict[str, int] = {}
    decorated: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for index, item in enumerate(prepared):
        name = normalized_item_key(item)
        occurrence = counts.get(name, 0)
        counts[name] = occurrence + 1
        token = (name, occurrence)
        if token in rank:
            order_key: tuple[Any, ...] = (0, rank[token], index)
        else:
            intrinsic = source_position(item)
            if intrinsic is not None:
                order_key = (1, intrinsic[0], intrinsic[1], index)
            else:
                order_key = (2, index)
        decorated.append((order_key, item))
    decorated.sort(key=lambda row: row[0])
    return [item for _, item in decorated]
