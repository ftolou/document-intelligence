from __future__ import annotations

import copy
import re
from typing import Any

from .evidence import parse_rows

_ROW_REFERENCE_WITH_TEXT = re.compile(r"^\s*(R\d{4})\s*::(?:\s*.*)?$")
_ITEM_UNRESOLVED_DECORATIVE_FIELDS = {"name", "line_amount", "unit_price"}


def _source_row_ids(transcription: str) -> set[str]:
    try:
        rows, _ = parse_rows(transcription)
    except Exception:
        return set()
    return {row["row_id"] for row in rows}


def _record_replace(
    operations: list[dict[str, Any]],
    *,
    rule: str,
    path: str,
    before: Any,
    after: Any,
) -> None:
    operations.append(
        {
            "rule": rule,
            "path": path,
            "before": copy.deepcopy(before),
            "after": copy.deepcopy(after),
        }
    )


def _normalize_row_reference(
    value: Any,
    *,
    source_ids: set[str],
    path: str,
    operations: list[dict[str, Any]],
) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    candidate = stripped if stripped in source_ids else None
    if candidate is None:
        match = _ROW_REFERENCE_WITH_TEXT.match(stripped)
        if match and match.group(1) in source_ids:
            candidate = match.group(1)
    if candidate is None or candidate == value:
        return value
    _record_replace(
        operations,
        rule="canonical_source_row_reference",
        path=path,
        before=value,
        after=candidate,
    )
    return candidate


def _normalize_row_reference_list(
    value: Any,
    *,
    source_ids: set[str],
    path: str,
    operations: list[dict[str, Any]],
) -> Any:
    if not isinstance(value, list):
        return value
    normalized = [
        _normalize_row_reference(
            entry,
            source_ids=source_ids,
            path=f"{path}/{index}",
            operations=operations,
        )
        for index, entry in enumerate(value)
    ]
    return normalized


def _normalize_unresolved_container(
    payload: dict[str, Any],
    *,
    operations: list[dict[str, Any]],
) -> None:
    value = payload.get("unresolved_candidate_rows")
    if isinstance(value, dict):
        normalized = [copy.deepcopy(value)]
        _record_replace(
            operations,
            rule="singleton_object_to_array",
            path="/unresolved_candidate_rows",
            before=value,
            after=normalized,
        )
        payload["unresolved_candidate_rows"] = normalized


def _normalize_item_evidence(
    payload: dict[str, Any],
    *,
    source_ids: set[str],
    operations: list[dict[str, Any]],
) -> None:
    _normalize_unresolved_container(payload, operations=operations)

    blocks = payload.get("item_blocks")
    if isinstance(blocks, list):
        for index, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            block["source_rows"] = _normalize_row_reference_list(
                block.get("source_rows"),
                source_ids=source_ids,
                path=f"/item_blocks/{index}/source_rows",
                operations=operations,
            )

    unresolved = payload.get("unresolved_candidate_rows")
    if not isinstance(unresolved, list):
        return
    for index, group in enumerate(unresolved):
        if not isinstance(group, dict):
            continue
        group["source_rows"] = _normalize_row_reference_list(
            group.get("source_rows"),
            source_ids=source_ids,
            path=f"/unresolved_candidate_rows/{index}/source_rows",
            operations=operations,
        )
        removable = sorted(key for key in group if key in _ITEM_UNRESOLVED_DECORATIVE_FIELDS)
        if removable and "source_rows" in group:
            before = copy.deepcopy(group)
            for key in removable:
                group.pop(key, None)
            _record_replace(
                operations,
                rule="drop_known_unresolved_item_decorative_fields",
                path=f"/unresolved_candidate_rows/{index}",
                before=before,
                after=group,
            )


def _normalize_vat_evidence(
    payload: dict[str, Any],
    *,
    source_ids: set[str],
    operations: list[dict[str, Any]],
) -> None:
    _normalize_unresolved_container(payload, operations=operations)

    blocks = payload.get("vat_evidence_blocks")
    if isinstance(blocks, list):
        for index, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            block["context_rows"] = _normalize_row_reference_list(
                block.get("context_rows"),
                source_ids=source_ids,
                path=f"/vat_evidence_blocks/{index}/context_rows",
                operations=operations,
            )
            block["source_row"] = _normalize_row_reference(
                block.get("source_row"),
                source_ids=source_ids,
                path=f"/vat_evidence_blocks/{index}/source_row",
                operations=operations,
            )

    unresolved = payload.get("unresolved_candidate_rows")
    if not isinstance(unresolved, list):
        return
    for index, group in enumerate(unresolved):
        if not isinstance(group, dict):
            continue
        for key in ("context_rows", "source_rows"):
            group[key] = _normalize_row_reference_list(
                group.get(key),
                source_ids=source_ids,
                path=f"/unresolved_candidate_rows/{index}/{key}",
                operations=operations,
            )


def _normalize_final_total_evidence(
    payload: dict[str, Any],
    *,
    source_ids: set[str],
    operations: list[dict[str, Any]],
) -> None:
    for key in ("label_row", "source_row"):
        payload[key] = _normalize_row_reference(
            payload.get(key),
            source_ids=source_ids,
            path=f"/{key}",
            operations=operations,
        )


def normalize_source_evidence(
    strategy_id: str,
    answer: Any,
    transcription: str,
) -> tuple[Any, dict[str, Any]]:
    """Normalize only approved response-shape variations before existing validation.

    This function never derives receipt values, chooses evidence, or mutates the
    receipt. The existing strategy validators remain the semantic authority.
    """
    if not isinstance(answer, dict):
        return answer, {
            "status": "not_applicable",
            "strategy_id": strategy_id,
            "operation_count": 0,
            "operations": [],
        }

    normalized = copy.deepcopy(answer)
    operations: list[dict[str, Any]] = []
    source_ids = _source_row_ids(transcription)

    if strategy_id == "item_sum_source_blocks_v3":
        _normalize_item_evidence(
            normalized,
            source_ids=source_ids,
            operations=operations,
        )
    elif strategy_id == "vat_source_evidence_v9":
        _normalize_vat_evidence(
            normalized,
            source_ids=source_ids,
            operations=operations,
        )
    elif strategy_id == "final_total_source_evidence_v2_4":
        _normalize_final_total_evidence(
            normalized,
            source_ids=source_ids,
            operations=operations,
        )

    return normalized, {
        "status": "normalized" if operations else "unchanged",
        "strategy_id": strategy_id,
        "operation_count": len(operations),
        "operations": operations,
    }
