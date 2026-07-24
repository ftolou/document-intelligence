"""Compatibility metadata for the geometry-only spatial evidence stage."""

from __future__ import annotations

from typing import Any

JsonObject = dict[str, Any]


def _confidence(value: Any) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 4)
    except (TypeError, ValueError):
        return 0.0


def _line_ids(values: Any, valid_ids: set[str]) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value) in valid_ids]


def normalize_spatial_overview(value: JsonObject, document_map: JsonObject) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError("Spatial overview must be a JSON object.")
    valid_ids = {
        str(row.get("line_id"))
        for row in document_map.get("rows") or []
        if isinstance(row, dict) and row.get("line_id")
    }
    sections: list[JsonObject] = []
    for index, raw in enumerate(value.get("sections") or []):
        if not isinstance(raw, dict):
            continue
        y_start = max(0.0, min(1.0, float(raw.get("y_start") or 0.0)))
        y_end = max(y_start, min(1.0, float(raw.get("y_end") or y_start)))
        sections.append(
            {
                "section_id": str(raw.get("section_id") or f"section_{index:03d}"),
                "type": str(raw.get("type") or "unknown"),
                "y_start": round(y_start, 5),
                "y_end": round(y_end, 5),
                "source_line_ids": _line_ids(raw.get("source_line_ids"), valid_ids),
                "confidence": _confidence(raw.get("confidence")),
            }
        )

    tables: list[JsonObject] = []
    for table_index, raw_table in enumerate(value.get("tables") or []):
        if not isinstance(raw_table, dict):
            continue
        columns: list[JsonObject] = []
        for column_index, raw_column in enumerate(raw_table.get("columns") or []):
            if not isinstance(raw_column, dict):
                continue
            x_start = max(0.0, min(1.0, float(raw_column.get("x_start") or 0.0)))
            x_end = max(x_start, min(1.0, float(raw_column.get("x_end") or x_start)))
            columns.append(
                {
                    "column_id": str(
                        raw_column.get("column_id") or f"column_{column_index:03d}"
                    ),
                    "role": str(raw_column.get("role") or "unknown"),
                    "x_start": round(x_start, 5),
                    "x_end": round(x_end, 5),
                    "header_text": raw_column.get("header_text"),
                    "confidence": _confidence(raw_column.get("confidence")),
                }
            )
        row_groups: list[JsonObject] = []
        for row_index, raw_row in enumerate(raw_table.get("row_groups") or []):
            if not isinstance(raw_row, dict):
                continue
            continuation = raw_row.get("continuation_of")
            row_groups.append(
                {
                    "row_group_id": str(
                        raw_row.get("row_group_id") or f"row_group_{row_index:03d}"
                    ),
                    "source_line_ids": _line_ids(raw_row.get("source_line_ids"), valid_ids),
                    "row_type_hint": str(raw_row.get("row_type_hint") or "unknown"),
                    "continuation_of": str(continuation) if continuation else None,
                    "confidence": _confidence(raw_row.get("confidence")),
                }
            )
        tables.append(
            {
                "table_id": str(raw_table.get("table_id") or f"table_{table_index:03d}"),
                "section_id": str(raw_table.get("section_id") or ""),
                "header_line_ids": _line_ids(raw_table.get("header_line_ids"), valid_ids),
                "columns": columns,
                "row_groups": row_groups,
                "confidence": _confidence(raw_table.get("confidence")),
            }
        )

    annotations: list[JsonObject] = []
    for raw in value.get("line_annotations") or []:
        if not isinstance(raw, dict):
            continue
        line_id = str(raw.get("line_id") or "")
        if line_id not in valid_ids:
            continue
        attached = str(raw.get("attach_to_line_id") or "")
        annotations.append(
            {
                "line_id": line_id,
                "row_type": str(raw.get("row_type") or "unknown"),
                "attach_to_line_id": attached if attached in valid_ids else None,
                "confidence": _confidence(raw.get("confidence")),
            }
        )

    status = str(value.get("status") or "partial").lower()
    if status not in {"ok", "partial", "failed"}:
        status = "partial"
    return {
        "schema_version": "spatial_overview_1",
        "status": status,
        "sections": sections,
        "tables": tables,
        "line_annotations": annotations,
        "warnings": [str(item) for item in (value.get("warnings") or []) if str(item).strip()],
        "overall_confidence": _confidence(value.get("overall_confidence")),
    }


def build_geometry_only_overview(document_map: JsonObject) -> JsonObject:
    """Describe the spatial stage without performing a second model call."""

    groups = [
        group
        for group in (document_map.get("geometric_row_groups") or [])
        if isinstance(group, dict)
    ]
    return {
        "schema_version": "spatial_overview_1",
        "status": "geometry_only",
        "mode": "deterministic_geometry",
        "llm_call_performed": False,
        "sections": [],
        "tables": [],
        "line_annotations": [],
        "warnings": [],
        "overall_confidence": None,
        "geometric_row_group_count": len(groups),
        "duration_seconds": 0.0,
        "model_metrics": None,
        "prompt": "",
        "raw_output": "",
    }


def run_spatial_overview(*, document_map: JsonObject, **_: Any) -> JsonObject:
    """Backward-compatible alias for callers from the first experiment patch."""

    return build_geometry_only_overview(document_map)


__all__ = [
    "build_geometry_only_overview",
    "normalize_spatial_overview",
    "run_spatial_overview",
]
