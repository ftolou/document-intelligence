"""Human-review application service and approved-receipt import workflow."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from receipt_intelligence.services.artifact_service import artifact_url
from receipt_intelligence.storage.job_store import JobStore
from receipt_intelligence.storage.receipt_db import ReceiptDatabase


def _deep_copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _set_if_present(container: dict[str, Any], key: str, value: Any) -> bool:
    if value is None:
        return False
    container[key] = value
    return True


def _number_or_original(value: Any) -> Any:
    if value is None or value == "":
        return value
    try:
        return float(value)
    except Exception:
        return value


def _bool_or_original(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return value


def _category_path_from_group_key(group: Any, key: Any) -> str | None:
    group_text = str(group or "").strip()
    key_text = str(key or "").strip()
    if not group_text and not key_text:
        return None
    if group_text and key_text:
        return f"{group_text}/{key_text}"
    return group_text or key_text


def apply_human_review(
    receipt: dict[str, Any],
    fields: dict[str, Any],
    item_corrections: list[dict[str, Any]],
    review: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Apply header and item corrections without conflating row and spend categories."""
    updated = _deep_copy_json(receipt)
    changed: list[str] = []

    merchant = updated.setdefault("merchant", {})
    totals = updated.setdefault("totals", {})
    validation = updated.setdefault("validation", {})

    mapping: list[tuple[str, dict[str, Any], str, bool]] = [
        ("merchant_name", merchant, "name", False),
        ("merchant_address", merchant, "address", False),
        ("date", updated, "date", False),
        ("time", updated, "time", False),
        ("currency", updated, "currency", False),
        ("document_type", updated, "document_type", False),
        ("receipt_category", updated, "receipt_category", False),
        ("receipt_business_category", updated, "receipt_business_category", False),
        ("subtotal", totals, "subtotal", True),
        ("tax_total", totals, "tax_total", True),
        ("grand_total", totals, "grand_total", True),
        ("paid_total", totals, "paid_total", True),
        ("change", totals, "change", True),
        ("import_decision", validation, "import_decision", False),
    ]
    for incoming_key, target, target_key, numeric in mapping:
        if incoming_key not in fields:
            continue
        value = fields.get(incoming_key)
        if numeric:
            value = _number_or_original(value)
        if _set_if_present(target, target_key, value):
            changed.append(incoming_key)

    items = updated.get("items") if isinstance(updated.get("items"), list) else []
    item_text_fields = {
        "description",
        "raw_description",
        "product_description",
        "line_note",
        "promotion_note",
        "raw_name",
        "normalized_name",
        "category",
        "parser_item_type",
        "receipt_row_type",
        "category_group",
        "category_key",
        "category_path",
        "category_source",
        "category_reason",
        "semantic_description",
        "unit",
        "vat_rate",
        "tax_code",
        "table_interpretation_source_row_id",
        "review_status",
    }
    item_numeric_fields = {
        "quantity",
        "unit_price",
        "original_price",
        "gross_unit_price",
        "discount_amount",
        "line_total",
        "confidence",
        "category_confidence",
    }
    item_bool_fields = {"category_review_required"}

    for correction in item_corrections or []:
        if not isinstance(correction, dict):
            continue
        try:
            index = int(correction.get("index"))
        except Exception:
            continue
        if index < 0 or index >= len(items) or not isinstance(items[index], dict):
            continue
        item = items[index]
        for key in sorted(item_text_fields | item_numeric_fields | item_bool_fields):
            if key not in correction:
                continue
            value = correction.get(key)
            if key in item_numeric_fields:
                value = _number_or_original(value)
            elif key in item_bool_fields:
                value = _bool_or_original(value)

            if key == "description":
                item["description"] = value
                item.setdefault("raw_name", value)
                changed.append(f"items[{index}].description")
            elif key == "product_description":
                item["product_description"] = value
                if not item.get("description"):
                    item["description"] = value
                changed.append(f"items[{index}].product_description")
            elif key == "raw_description":
                item["raw_description"] = value
                changed.append(f"items[{index}].raw_description")
            elif key == "parser_item_type":
                item["category"] = value
                item["parser_item_type"] = value
                changed.append(f"items[{index}].parser_item_type")
                changed.append(f"items[{index}].category")
            else:
                item[key] = value
                changed.append(f"items[{index}].{key}")

        if "category_group" in correction or "category_key" in correction:
            category_path = _category_path_from_group_key(
                item.get("category_group"), item.get("category_key")
            )
            if category_path:
                item["category_path"] = category_path
        if "review_status" not in correction:
            item.setdefault("review_status", review.get("status") or "reviewed")

    reviewed_at = datetime.now(UTC).isoformat(timespec="seconds")
    updated["human_review"] = {
        "status": review.get("status") or "needs_review",
        "reviewer": review.get("reviewer") or "",
        "notes": review.get("notes") or "",
        "reviewed_at": reviewed_at,
        "changed_fields": changed,
    }
    return updated, changed


class ReviewService:
    def __init__(self, store: JobStore, receipt_db: ReceiptDatabase) -> None:
        self.store = store
        self.receipt_db = receipt_db

    def artifact_path_from_url(self, job_id: str, artifact_value: str | None) -> Path | None:
        if not artifact_value:
            return None
        try:
            parsed = urlparse(str(artifact_value))
            filename = Path(unquote(parsed.path)).name
        except Exception:
            filename = Path(str(artifact_value)).name
        if not filename:
            return None
        job_dir = self.store.job_dir(job_id).resolve()
        candidate = (job_dir / filename).resolve()
        try:
            candidate.relative_to(job_dir)
        except ValueError:
            return None
        return candidate if candidate.exists() else None

    def final_receipt_path(self, job_id: str) -> Path | None:
        job = self.store.get(job_id) or {}
        result = job.get("result") if isinstance(job.get("result"), dict) else {}
        artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
        for key in (
            "final_receipt",
            "final_receipt_categorized",
            "final_receipt_reconciled",
        ):
            path = self.artifact_path_from_url(job_id, artifacts.get(key))
            if path is not None:
                return path

        job_dir = self.store.job_dir(job_id)
        candidates = sorted(
            job_dir.glob("*receipt_final*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def review_record_path(self, job_id: str) -> Path:
        return self.store.job_dir(job_id) / "human_review_record.json"

    def approved_receipt_path(self, job_id: str) -> Path:
        return self.store.job_dir(job_id) / "approved_receipt.json"

    def safe_job_artifact_path(
        self,
        job_id: str,
        value: str | Path | None,
    ) -> Path | None:
        """Resolve a stored artifact only when it remains inside its job directory."""

        if not value or not job_id:
            return None
        root = self.store.results_dir.resolve()
        job_dir = self.store.job_dir(job_id).resolve()
        try:
            job_dir.relative_to(root)
        except ValueError:
            return None
        candidate = Path(str(value)).expanduser()
        if not candidate.is_absolute():
            candidate = job_dir / candidate
        try:
            resolved = candidate.resolve()
            resolved.relative_to(job_dir)
        except (OSError, ValueError):
            return None
        return resolved if resolved.exists() and resolved.is_file() else None

    def preferred_receipt_path(
        self,
        job_id: str,
        *,
        stored_approved_path: str | Path | None = None,
        stored_source_path: str | Path | None = None,
    ) -> Path | None:
        """Prefer approved review data before falling back to the original result."""

        candidates = [
            stored_approved_path,
            self.approved_receipt_path(job_id),
            stored_source_path,
            self.final_receipt_path(job_id),
        ]
        seen: set[Path] = set()
        for value in candidates:
            candidate = self.safe_job_artifact_path(job_id, value)
            if candidate is None or candidate in seen:
                continue
            seen.add(candidate)
            return candidate
        return None

    @staticmethod
    def read_receipt_json(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("receipt JSON root must be an object")
        return value

    @staticmethod
    def read_database_receipt(record: dict[str, Any]) -> dict[str, Any]:
        value = json.loads(str(record.get("raw_json") or "{}"))
        if not isinstance(value, dict):
            raise ValueError("stored receipt JSON root must be an object")
        return value

    def load_review_record(
        self,
        job_id: str,
        receipt: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        path = self.review_record_path(job_id)
        if path.exists():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    return value
            except Exception:
                pass
        human_review = receipt.get("human_review") if isinstance(receipt, dict) else None
        return dict(human_review) if isinstance(human_review, dict) else None

    def database_image_url(self, record: dict[str, Any]) -> str | None:
        job_id = str(record.get("job_id") or "").strip()
        path = self.safe_job_artifact_path(job_id, record.get("image_path"))
        if path is None:
            return self.review_image_url(job_id) if job_id else None
        return artifact_url(job_id, path)

    def job_image_path(self, job_id: str) -> Path | None:
        job = self.store.get(job_id) or {}
        image_path = job.get("image_path")
        if not image_path:
            return None
        path = Path(str(image_path))
        return path if path.exists() else None

    def review_image_url(self, job_id: str) -> str | None:
        image_path = self.job_image_path(job_id)
        if image_path is None:
            return None
        job_dir = self.store.job_dir(job_id).resolve()
        try:
            image_path.resolve().relative_to(job_dir)
        except ValueError:
            return None
        return artifact_url(job_id, image_path)

    def import_reviewed_receipt(
        self,
        job_id: str,
        receipt: dict[str, Any],
        approved_path: Path,
        source_path: Path | None = None,
    ) -> dict[str, Any]:
        image_path = self.job_image_path(job_id)
        if image_path is None:
            existing = self.receipt_db.get_receipt_review_record_by_job_id(job_id) or {}
            image_path = self.safe_job_artifact_path(job_id, existing.get("image_path"))
        import_result = self.receipt_db.import_receipt(
            job_id=job_id,
            receipt=receipt,
            approved_receipt_path=approved_path,
            source_receipt_path=source_path,
            image_path=image_path,
        )
        return {
            "receipt_db_id": import_result.receipt_db_id,
            "job_id": import_result.job_id,
            "item_count": import_result.item_count,
            "inserted_at": import_result.inserted_at,
        }

    def register_job_for_review(
        self,
        job_id: str,
        report: dict[str, Any],
        final_receipt_path: Path,
    ) -> dict[str, Any]:
        receipt = json.loads(final_receipt_path.read_text(encoding="utf-8"))
        issues = report.get("issues") if isinstance(report.get("issues"), list) else []
        return self.receipt_db.upsert_review_queue(
            job_id=job_id,
            receipt=receipt,
            decision=report.get("import_decision"),
            balanced=report.get("balanced"),
            difference=report.get("difference"),
            issue_count=len(issues),
            image_path=self.job_image_path(job_id),
            final_receipt_path=final_receipt_path,
        )
