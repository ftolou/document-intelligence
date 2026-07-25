"""Metadata for the deterministic spatial geometry stage."""

from __future__ import annotations

from typing import Any

JsonObject = dict[str, Any]


def build_geometry_only_overview(document_map: JsonObject) -> JsonObject:
    """Summarize the spatial artifact without performing another model call."""

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
        "geometric_row_group_count": len(groups),
        "warnings": [],
    }


__all__ = ["build_geometry_only_overview"]
