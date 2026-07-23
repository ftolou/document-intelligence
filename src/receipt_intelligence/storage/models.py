"""Typed storage-layer result models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReceiptImportResult:
    receipt_db_id: int
    job_id: str
    item_count: int
    inserted_at: str
