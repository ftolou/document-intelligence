"""Human-review application service and approved-receipt import workflow."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from receipt_intelligence.domain.categorization_taxonomy import fashion_category_path
from receipt_intelligence.extraction.contracts.validation import ValidationRequest
from receipt_intelligence.extraction.validation.engine import DeterministicValidationEngine
from receipt_intelligence.receipt_compat import (
    apply_review_field,
    apply_review_item_field,
    is_next_receipt,
    item_description,
    item_line_total,
    receipt_grand_total,
    to_canonical_validation_document,
    to_review_document,
    validation_for_review,
)
from receipt_intelligence.services.artifact_service import artifact_resource
from receipt_intelligence.services.semantic_index_service import SemanticIndexUpdater
from receipt_intelligence.storage.job_store import JobStore
from receipt_intelligence.storage.receipt_db import ReceiptDatabase


def _deep_copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


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


_NUMERIC_REVIEW_FIELDS = {
    "subtotal",
    "tax_total",
    "grand_total",
    "paid_total",
    "change",
    "quantity",
    "unit_price",
    "original_price",
    "gross_unit_price",
    "discount_amount",
    "line_total",
    "confidence",
    "category_confidence",
}


def _review_values_equal(key: str, left: Any, right: Any) -> bool:
    """Compare values using the semantics exposed by the review UI."""

    if left in (None, "") and right in (None, ""):
        return True
    if key in _NUMERIC_REVIEW_FIELDS:
        try:
            return float(left) == float(right)
        except (TypeError, ValueError):
            pass
    return left == right


def _review_header_value(receipt: dict[str, Any], key: str) -> Any:
    projected = to_review_document(receipt)
    merchant = projected.get("merchant") if isinstance(projected.get("merchant"), dict) else {}
    totals = projected.get("totals") if isinstance(projected.get("totals"), dict) else {}
    if key == "merchant_name":
        return merchant.get("name")
    if key == "merchant_address":
        return merchant.get("address")
    if key in {"subtotal", "tax_total", "grand_total", "paid_total", "change"}:
        return totals.get(key)
    return projected.get(key)


def _review_item_value(item: dict[str, Any], key: str) -> Any:
    if key in {"description", "product_description"}:
        return item_description(item)
    if key == "line_total":
        return item_line_total(item)
    if key == "parser_item_type":
        for candidate in ("parser_item_type", "receipt_row_type", "line_type", "category"):
            value = item.get(candidate)
            if value not in (None, ""):
                return value
        return None
    return item.get(key)


def _category_path_from_group_key(group: Any, key: Any) -> str | None:
    group_text = str(group or "").strip()
    key_text = str(key or "").strip()
    canonical_path = fashion_category_path(key_text)
    if canonical_path:
        return canonical_path
    if not group_text and not key_text:
        return None
    if group_text and key_text:
        return f"{group_text}/{key_text}"
    return group_text or key_text


def _add_review_core_checks(report: dict[str, Any], receipt: dict[str, Any]) -> None:
    """Retain approval safeguards for both historical and canonical saved receipts."""

    merchant = receipt.get("merchant")
    merchant_name = merchant.get("name") if isinstance(merchant, dict) else None
    items = receipt.get("items")
    requirements = (
        ("MISSING_MERCHANT", bool(str(merchant_name or "").strip()), "Merchant name is missing."),
        ("MISSING_TOTAL", receipt_grand_total(receipt) is not None, "Final total is missing."),
        ("NO_ITEMS", bool(items) if isinstance(items, list) else False, "No items are present."),
    )
    missing = [(code, message) for code, present, message in requirements if not present]
    if not missing:
        return
    checks = report.setdefault("checks", [])
    if not isinstance(checks, list):
        checks = []
        report["checks"] = checks
    checks.extend(
        {
            "code": code,
            "status": "failed",
            "severity": "error",
            "message": message,
        }
        for code, message in missing
    )
    report["status"] = "invalid"
    summary = report.setdefault("summary", {})
    if isinstance(summary, dict):
        summary["failed"] = int(summary.get("failed") or 0) + len(missing)
        summary["error_count"] = int(summary.get("error_count") or 0) + len(missing)


def apply_human_review(
    receipt: dict[str, Any],
    fields: dict[str, Any],
    item_corrections: list[dict[str, Any]],
    review: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Apply review edits to the canonical legacy or next receipt schema."""

    updated = _deep_copy_json(receipt)
    changed: list[str] = []
    numeric_header_fields = {"subtotal", "tax_total", "grand_total", "paid_total", "change"}
    for key, raw_value in fields.items():
        value = _number_or_original(raw_value) if key in numeric_header_fields else raw_value
        if value is None:
            continue
        before_value = _review_header_value(updated, key)
        applied = apply_review_field(updated, key, value)
        after_value = _review_header_value(updated, key)
        if applied and not _review_values_equal(key, before_value, after_value):
            changed.append(key)

    items = updated.get("items") if isinstance(updated.get("items"), list) else []
    next_schema = is_next_receipt(updated)
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
    ignored_fields = {"index", "item_id", "_db_item_id"}

    for correction in item_corrections or []:
        if not isinstance(correction, dict):
            continue
        try:
            index = int(correction.get("index"))
        except (TypeError, ValueError):
            continue
        if index < 0 or index >= len(items) or not isinstance(items[index], dict):
            continue
        item = items[index]
        for key, raw_value in correction.items():
            if key in ignored_fields:
                continue
            before_value = _review_item_value(item, key)
            value = raw_value
            if key in item_numeric_fields:
                value = _number_or_original(value)
            elif key in item_bool_fields:
                value = _bool_or_original(value)
            canonical_key = apply_review_item_field(
                item,
                key,
                value,
                next_schema=next_schema,
            )
            after_value = _review_item_value(item, key)
            item_changed = not _review_values_equal(key, before_value, after_value)
            if item_changed:
                changed.append(f"items[{index}].{canonical_key}")
            if item_changed and not next_schema and key == "parser_item_type":
                changed.append(f"items[{index}].category")

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
    _APPROVAL_BLOCKING_CODES = {
        "LLM_PARSE_FAILED",
        "MISSING_MERCHANT",
        "MISSING_TOTAL",
        "NO_ITEMS",
    }

    def __init__(
        self,
        store: JobStore,
        receipt_db: ReceiptDatabase,
        *,
        semantic_index_updater: SemanticIndexUpdater | None = None,
    ) -> None:
        self.store = store
        self.receipt_db = receipt_db
        self.semantic_index_updater = semantic_index_updater

    def ocr_context_path(self, job_id: str) -> Path | None:
        """Resolve the extraction OCR context used for post-HITL validation."""

        if not str(job_id or "").strip():
            return None
        job = self.store.get(job_id) or {}
        result = job.get("result") if isinstance(job.get("result"), dict) else {}
        artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
        path = self.artifact_path_from_url(job_id, artifacts.get("ocr_context"))
        if path is not None:
            return path

        job_dir = self.store.job_dir(job_id)
        candidates = sorted(
            [
                *job_dir.glob("*_v14_ocr_context.json"),
                *job_dir.glob("latest_v14_ocr_context.json"),
            ],
            key=lambda candidate: candidate.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def load_ocr_context(self, job_id: str) -> tuple[dict[str, Any], str]:
        path = self.ocr_context_path(job_id)
        if path is None:
            return {"lines": [], "layout_rows": []}, "unavailable"
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return {"lines": [], "layout_rows": []}, "unreadable"
        if not isinstance(value, dict):
            return {"lines": [], "layout_rows": []}, "invalid"
        return value, path.name

    def finalize_human_review(
        self,
        job_id: str,
        receipt: dict[str, Any],
        *,
        requested_status: str | None,
    ) -> dict[str, Any]:
        """Revalidate edited data and derive the effective review/import state.

        Human approval can resolve non-blocking validation warnings, but it may
        not override missing core receipt data or high/critical validation issues.
        A manually completed receipt can recover from an original LLM failure: the
        machine parse status is preserved in review metadata and validation is run
        against the reviewed document.
        """

        requested = str(requested_status or "needs_review").strip().lower()
        if requested not in {"approved", "needs_review", "rejected"}:
            raise ValueError(f"Unsupported human review status: {requested}")

        updated = _deep_copy_json(receipt)
        previous_validation = (
            dict(updated.get("validation")) if isinstance(updated.get("validation"), dict) else {}
        )
        original_parse_status = str(updated.get("parse_status") or "partial")
        if requested == "approved" and original_parse_status == "failed":
            # The reviewed document, not the failed model attempt, is now the input
            # to validation. Structural validation still blocks incomplete edits.
            updated["parse_status"] = "partial"

        _ocr_context, context_source = self.load_ocr_context(job_id)
        validation_input = to_canonical_validation_document(updated)
        items = validation_input.get("items")
        item_contract = {
            "status": "valid" if isinstance(items, list) else "invalid",
            "warnings": [],
            "errors": [] if isinstance(items, list) else ["items must be an array"],
        }
        report = (
            DeterministicValidationEngine()
            .validate(
                ValidationRequest(
                    receipt=validation_input,
                    item_contract=item_contract,
                    item_pipeline_enabled=True,
                    selected_scalar_tasks=(),
                )
            )
            .to_dict()
        )
        _add_review_core_checks(report, updated)
        report = validation_for_review(report)
        deterministic_decision = str(report.get("import_decision") or "needs_review")
        issues = [issue for issue in report.get("issues") or [] if isinstance(issue, dict)]
        blocking_issues = [
            issue
            for issue in issues
            if str(issue.get("severity") or "").lower() in {"critical", "high"}
            or str(issue.get("code") or "") in self._APPROVAL_BLOCKING_CODES
        ]

        approval_blocked = False
        validation_override = False
        if requested == "approved":
            if blocking_issues or deterministic_decision in {"reject", "llm_failed"}:
                effective_status = "needs_review"
                effective_decision = (
                    deterministic_decision
                    if deterministic_decision in {"reject", "llm_failed"}
                    else "reject"
                )
                approval_blocked = True
            else:
                effective_status = "approved"
                effective_decision = "import"
                validation_override = deterministic_decision != "import"
        elif requested == "rejected":
            effective_status = "rejected"
            effective_decision = "reject"
        else:
            effective_status = "needs_review"
            effective_decision = "needs_review"

        items = updated.get("items") if isinstance(updated.get("items"), list) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            item_status = str(item.get("review_status") or "").strip().lower()
            if effective_status == "approved" and item_status != "rejected":
                item["review_status"] = "corrected" if item_status == "corrected" else "approved"
            elif approval_blocked and item_status == "approved":
                item["review_status"] = "needs_review"

        report["deterministic_import_decision"] = deterministic_decision
        report["pre_review_import_decision"] = previous_validation.get("import_decision")
        report["import_decision"] = effective_decision
        report["review_resolution"] = {
            "requested_status": requested,
            "effective_status": effective_status,
            "approval_blocked": approval_blocked,
            "human_override": validation_override,
            "blocking_issue_codes": [str(issue.get("code") or "") for issue in blocking_issues],
            "ocr_context_source": context_source,
        }
        updated["validation"] = report

        human_review = (
            dict(updated.get("human_review"))
            if isinstance(updated.get("human_review"), dict)
            else {}
        )
        human_review.update(
            {
                "requested_status": requested,
                "status": effective_status,
                "approval_blocked": approval_blocked,
                "validation_override": validation_override,
                "original_parse_status": original_parse_status,
                "remaining_issue_codes": [str(issue.get("code") or "") for issue in issues],
                "blocking_issue_codes": [str(issue.get("code") or "") for issue in blocking_issues],
            }
        )
        updated["human_review"] = human_review
        return {
            "receipt": updated,
            "validation": report,
            "requested_status": requested,
            "effective_status": effective_status,
            "queue_status": effective_status,
            "import_allowed": effective_status == "approved" and effective_decision == "import",
            "approval_blocked": approval_blocked,
            "validation_override": validation_override,
        }

    def sync_review_queue(
        self,
        job_id: str,
        receipt: dict[str, Any],
        *,
        receipt_path: Path,
        queue_status: str,
        receipt_db_id: int | None = None,
    ) -> dict[str, Any]:
        validation = (
            receipt.get("validation") if isinstance(receipt.get("validation"), dict) else {}
        )
        validation = validation_for_review(validation)
        issues = validation.get("issues") if isinstance(validation.get("issues"), list) else []
        result = self.receipt_db.upsert_review_queue(
            job_id=job_id,
            receipt=receipt,
            decision=validation.get("import_decision"),
            balanced=validation.get("balanced"),
            difference=validation.get("difference"),
            issue_count=len(issues),
            image_path=self.job_image_path(job_id),
            final_receipt_path=receipt_path,
            queue_status=queue_status,
        )
        if receipt_db_id is not None:
            result.update(
                self.receipt_db.update_review_status(
                    job_id,
                    queue_status,
                    receipt_db_id=receipt_db_id,
                )
            )
        return result

    def index_receipt_items(self, receipt_id: int) -> dict[str, Any]:
        item_ids = self.receipt_db.list_receipt_item_ids(receipt_id)
        if self.semantic_index_updater is None:
            return {
                "status": "not_configured",
                "requested_item_ids": item_ids,
                "message": "No semantic index updater is configured for this process.",
            }
        return self.semantic_index_updater.index_item_ids(item_ids)

    def index_item_ids(self, item_ids: list[int]) -> dict[str, Any]:
        if self.semantic_index_updater is None:
            return {
                "status": "not_configured",
                "requested_item_ids": sorted({int(value) for value in item_ids}),
                "message": "No semantic index updater is configured for this process.",
            }
        return self.semantic_index_updater.index_item_ids(item_ids)

    def artifact_path_from_url(
        self,
        job_id: str,
        artifact_value: Mapping[str, Any] | str | None,
    ) -> Path | None:
        if not artifact_value:
            return None
        if isinstance(artifact_value, Mapping):
            reference_job_id = str(artifact_value.get("job_id") or job_id).strip()
            if reference_job_id != job_id:
                return None
            filename = Path(str(artifact_value.get("filename") or "")).name
        else:
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
        # The canonical SQLite draft/database receipt wins. The JSON record is
        # retained only as a backward-compatible audit mirror.
        human_review = receipt.get("human_review") if isinstance(receipt, dict) else None
        if isinstance(human_review, dict):
            return dict(human_review)
        path = self.review_record_path(job_id)
        if path.exists():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    return value
            except Exception:
                pass
        return None

    def database_image_reference(self, record: dict[str, Any]) -> dict[str, str] | None:
        job_id = str(record.get("job_id") or "").strip()
        path = self.safe_job_artifact_path(job_id, record.get("image_path"))
        if path is None:
            return self.review_image_reference(job_id) if job_id else None
        return artifact_resource(job_id, path)

    def job_image_path(self, job_id: str) -> Path | None:
        job = self.store.get(job_id) or {}
        image_path = job.get("image_path")
        if not image_path:
            return None
        path = Path(str(image_path))
        return path if path.exists() else None

    def review_image_reference(self, job_id: str) -> dict[str, str] | None:
        image_path = self.job_image_path(job_id)
        if image_path is None:
            return None
        job_dir = self.store.job_dir(job_id).resolve()
        try:
            image_path.resolve().relative_to(job_dir)
        except ValueError:
            return None
        return artifact_resource(job_id, image_path)

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
        report = validation_for_review(report)
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
