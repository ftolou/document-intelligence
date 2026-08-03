"""Background receipt and batch processing independent of Flask routes."""

from __future__ import annotations

import csv
import json
import shutil
import traceback
from pathlib import Path
from typing import Any

import receipt_intelligence.settings as settings
from receipt_intelligence.application.ports import OcrEngine, OcrRequest
from receipt_intelligence.extraction import ExtractionRequest
from receipt_intelligence.pipeline.integrated_receipt_pipeline import run_receipt_extraction
from receipt_intelligence.receipt_compat import validation_for_review
from receipt_intelligence.services.artifact_service import artifact_resource
from receipt_intelligence.services.review_service import ReviewService
from receipt_intelligence.storage.job_store import JobStore
from receipt_intelligence.storage.receipt_db import ReceiptDatabase
from receipt_intelligence.utils.filenames import safe_filename


class JobProcessingService:
    def __init__(
        self,
        store: JobStore,
        receipt_db: ReceiptDatabase,
        *,
        ocr_engine: OcrEngine,
    ) -> None:
        self.store = store
        self.receipt_db = receipt_db
        self.review_service = ReviewService(store, receipt_db)
        self.ocr_engine = ocr_engine

    def allowed_file(self, filename: str) -> bool:
        return "." in filename and filename.rsplit(".", 1)[1].lower() in settings.ALLOWED_EXTENSIONS

    def progress_for(self, job_id: str):
        def callback(event: dict[str, Any]) -> None:
            self.store.add_event(job_id, event)

        return callback

    def run_job(self, job_id: str, image_path: Path, options: dict[str, Any]) -> None:
        job_dir = self.store.job_dir(job_id)
        progress = self.progress_for(job_id)
        try:
            progress(
                {
                    "stage": "upload",
                    "status": "done",
                    "message": "Image saved.",
                    "details": {"image": image_path.name},
                }
            )

            ocr_json_path = job_dir / f"{job_id}_ocr_full_image.json"
            ocr_json = self.ocr_engine.recognize(
                OcrRequest(
                    image_path=image_path,
                    out_json_path=ocr_json_path,
                    work_dir=job_dir,
                    lang=options["ocr_lang"],
                    device=options["ocr_device"],
                    max_side_length=options["ocr_max_side_limit"],
                    detect_orientation=options["ocr_use_angle_cls"],
                    detection_max_side_length=options["ocr_det_limit_side_len"],
                    progress_callback=progress,
                )
            )

            extraction_request = ExtractionRequest(
                ocr_json_path=ocr_json_path,
                result_dir=job_dir,
                run_id=job_id,
                ollama_url=options["ollama_url"],
                model=options["model"],
                tolerance=options["validation_tolerance"],
                spatial_canvas_width=options["spatial_canvas_width"],
                ocr_lang=options["ocr_lang"],
                ocr_device=options["ocr_device"],
                max_lines_for_llm=options["max_lines_for_llm"],
                num_ctx=options["num_ctx"],
                num_predict=options["num_predict"],
                keep_alive=options["keep_alive"],
                llm_timeout_seconds=options["llm_timeout_seconds"],
                json_retry_count=options["json_retry_count"],
                format_json=options["format_json"],
                source_image_path=image_path,
                vlm_backend=options["vlm_backend"],
                vlm_service_url=options["vlm_service_url"],
                vlm_timeout_seconds=options["vlm_timeout_seconds"],
                vlm_max_chars=options["vlm_max_chars"],
                correction_enabled=options["vlm_correction_enabled"],
                gpu_orchestration=options["gpu_orchestration"],
                unload_llm_before_vlm=options["unload_llm_before_vlm"],
                reload_llm_after_vlm=options["reload_llm_after_vlm"],
                ollama_control_mode=options["ollama_control_mode"],
                ollama_control_timeout_seconds=options["ollama_control_timeout_seconds"],
                ollama_unload_command=options["ollama_unload_command"],
                ollama_start_command=options["ollama_start_command"],
                ollama_reload_prompt=options["ollama_reload_prompt"],
                ollama_gpu_handoff_wait_seconds=options["ollama_gpu_handoff_wait_seconds"],
                categorization_enabled=options["categorization_enabled"],
                categorization_model=options["categorization_model"],
                categorization_num_ctx=options["categorization_num_ctx"],
                categorization_num_predict=options["categorization_num_predict"],
                categorization_timeout_seconds=options["categorization_timeout_seconds"],
                categorization_format_json=options["categorization_format_json"],
                progress_callback=progress,
            )
            result = run_receipt_extraction(extraction_request)
            report = result["report"]
            paths = {key: str(value) for key, value in result.get("paths", {}).items()}
            key_artifacts = self._build_key_artifacts(
                job_id,
                image_path=image_path,
                ocr_json_path=ocr_json_path,
                paths=paths,
            )
            self.store.update(
                job_id,
                result={
                    "report": report,
                    "artifacts": key_artifacts,
                    "ocr": {
                        "line_count": ocr_json.get("line_count"),
                        "word_count": ocr_json.get("word_count"),
                    },
                },
            )

            try:
                final_for_review = self._final_result_path(paths)
                review_queue = self.review_service.register_job_for_review(
                    job_id,
                    report,
                    final_for_review,
                )
                self.store.update(job_id, review_queue=review_queue)
                self.store.add_event(
                    job_id,
                    {
                        "stage": "review_queue",
                        "status": "done",
                        "message": "Receipt added to the review queue.",
                        "details": {
                            "queue_status": review_queue.get("queue_status"),
                            "duplicate_score": review_queue.get("duplicate_score"),
                        },
                    },
                )
            except Exception as exc:
                self.store.add_event(
                    job_id,
                    {
                        "stage": "review_queue",
                        "status": "error",
                        "message": f"Could not add receipt to review queue: {exc}",
                    },
                )
        except Exception as exc:
            traceback_text = traceback.format_exc()
            progress(
                {
                    "stage": "pipeline",
                    "status": "error",
                    "message": str(exc),
                    "details": {"traceback": traceback_text[-4000:]},
                }
            )
            raise

    def _final_result_path(self, paths: dict[str, str]) -> Path:
        categorized = paths.get("receipt_final_categorized")
        if categorized and Path(categorized).exists():
            return Path(categorized)
        return Path(paths["receipt_final_reconciled"])

    def _build_key_artifacts(
        self,
        job_id: str,
        *,
        image_path: Path,
        ocr_json_path: Path,
        paths: dict[str, str],
    ) -> dict[str, Any]:
        final_path = self._final_result_path(paths)

        self.store.register_artifact(job_id, "receipt_image", image_path, category="input")
        self.store.register_artifact(job_id, "ocr_json", ocr_json_path, category="ocr")
        for key, value in paths.items():
            path = Path(value)
            if path.exists():
                self.store.register_artifact(job_id, key, path)

        def optional(name: str) -> dict[str, str] | None:
            value = paths.get(name)
            if not value:
                return None
            path = Path(value)
            return artifact_resource(job_id, path) if path.exists() else None

        artifacts = {
            "receipt_image": artifact_resource(job_id, image_path) if image_path.exists() else None,
            "ocr_json": artifact_resource(job_id, ocr_json_path)
            if ocr_json_path.exists()
            else None,
            "final_receipt": artifact_resource(job_id, final_path),
            "final_receipt_reconciled": optional("receipt_final_reconciled"),
            "final_receipt_categorized": optional("receipt_final_categorized"),
            "validation_report": optional("validation_report"),
            "llm_prompt": optional("llm_main_prompt"),
            "llm_raw": optional("llm_main_raw"),
            "ocr_context": optional("ocr_context"),
            "pipeline_meta": optional("pipeline_meta"),
            "stage_trace": optional("stage_trace"),
            "extraction_metrics": optional("extraction_metrics"),
            "visual_evidence": optional("visual_evidence"),
            "region_reocr": optional("region_reocr"),
            "right_column_reocr": optional("right_column_reocr"),
            "correction_patch_prompt": optional("correction_patch_prompt"),
            "correction_patch_raw": optional("correction_patch_raw"),
            "correction_patch_result": optional("correction_patch_result"),
            "corrected_receipt": optional("receipt_patch_corrected"),
            "categorization_result": optional("categorization_result"),
            "categorization_prompt": optional("categorization_prompt"),
            "categorization_raw": optional("categorization_raw"),
        }
        return {key: value for key, value in artifacts.items() if value}

    def resolve_batch_folder(self, folder_text: str | None) -> Path:
        raw = (folder_text or "").strip()
        allowed_root = settings.BATCH_INPUT_DIR.resolve()

        if not raw:
            return allowed_root

        supplied = Path(raw)
        path = supplied.resolve() if supplied.is_absolute() else (allowed_root / supplied).resolve()
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Batch folder does not exist or is not a directory: {path}")
        if not settings.BATCH_ALLOW_ANY_PATH and not self._is_relative_to(path, allowed_root):
            raise ValueError(
                f"Batch folder must be inside {allowed_root}; "
                "set BATCH_ALLOW_ANY_PATH=1 to allow other paths"
            )
        return path

    @staticmethod
    def _is_relative_to(child: Path, parent: Path) -> bool:
        try:
            child.resolve().relative_to(parent.resolve())
            return True
        except Exception:
            return False

    def list_batch_images(
        self,
        folder: Path,
        *,
        recursive: bool,
        max_files: int,
    ) -> list[Path]:
        candidates = folder.rglob("*") if recursive else folder.iterdir()
        images = [path for path in candidates if path.is_file() and self.allowed_file(path.name)]
        images.sort(key=lambda path: str(path).lower())
        return images[: max(0, max_files)]

    def _batch_item_from_job(self, child_id: str, source_path: Path) -> dict[str, Any]:
        child = self.store.get(child_id) or {}
        result = child.get("result") if isinstance(child.get("result"), dict) else {}
        report = result.get("report") if isinstance(result.get("report"), dict) else {}
        report = validation_for_review(report)
        artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
        return {
            "child_job_id": child_id,
            "filename": source_path.name,
            "source_path": str(source_path),
            "state": child.get("state", "unknown"),
            "decision": report.get("import_decision"),
            "balanced": report.get("balanced"),
            "difference": report.get("difference"),
            "issue_count": len(report.get("issues") or []),
            "final_receipt": artifacts.get("final_receipt"),
            "validation_report": artifacts.get("validation_report"),
            "error": child.get("error"),
        }

    def write_batch_summaries(
        self,
        batch_id: str,
        items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        batch_dir = self.store.job_dir(batch_id)
        summary_json = batch_dir / "batch_summary.json"
        summary_csv = batch_dir / "batch_summary.csv"
        payload = {"batch_id": batch_id, "items": items}
        summary_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        fieldnames = [
            "child_job_id",
            "filename",
            "state",
            "decision",
            "balanced",
            "difference",
            "issue_count",
            "final_receipt",
            "validation_report",
            "source_path",
        ]
        with summary_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for item in items:
                writer.writerow({key: item.get(key) for key in fieldnames})
        self.store.register_artifact(batch_id, "batch_summary_json", summary_json, category="batch")
        self.store.register_artifact(batch_id, "batch_summary_csv", summary_csv, category="batch")
        return {
            "batch_summary_json": artifact_resource(batch_id, summary_json),
            "batch_summary_csv": artifact_resource(batch_id, summary_csv),
        }

    def run_batch_job(
        self,
        batch_id: str,
        image_paths: list[Path],
        options: dict[str, Any],
    ) -> None:
        items: list[dict[str, Any]] = []
        try:
            self.store.update(
                batch_id,
                total=len(image_paths),
                completed=0,
                failed=0,
                items=items,
            )
            self.store.add_event(
                batch_id,
                {
                    "stage": "batch",
                    "status": "running",
                    "message": f"Batch started with {len(image_paths)} image(s).",
                    "details": {"serial": True},
                },
            )
            for index, source_path in enumerate(image_paths, start=1):
                child_id = f"{batch_id}_{index:03d}"
                child_dir = self.store.job_dir(child_id)
                child_dir.mkdir(parents=True, exist_ok=True)
                safe_name = safe_filename(source_path.name, fallback=f"receipt_{index:03d}.jpg")
                image_copy = child_dir / safe_name
                shutil.copy2(source_path, image_copy)
                self.store.add_event(
                    batch_id,
                    {
                        "stage": "batch_item",
                        "status": "running",
                        "message": (f"Running {index}/{len(image_paths)}: {source_path.name}"),
                        "details": {"child_job_id": child_id},
                    },
                )
                self.store.create(
                    child_id,
                    {
                        "type": "single_in_batch",
                        "batch_id": batch_id,
                        "filename": safe_name,
                        "source_path": str(source_path),
                        "image_path": str(image_copy),
                        "options": options,
                    },
                )
                self.store.begin_attempt(child_id)
                try:
                    self.run_job(child_id, image_copy, options)
                except Exception as child_error:
                    self.store.fail(
                        child_id,
                        {
                            "message": str(child_error),
                            "type": type(child_error).__name__,
                            "traceback": traceback.format_exc(),
                        },
                    )
                else:
                    self.store.complete(child_id)
                item = self._batch_item_from_job(child_id, source_path)
                items.append(item)
                failed = self._failed_batch_count(items)
                artifacts = self.write_batch_summaries(batch_id, items)
                self.store.update(
                    batch_id,
                    completed=len(items),
                    failed=failed,
                    items=items,
                    result={"artifacts": artifacts, "items": items},
                )
                self.store.add_event(
                    batch_id,
                    {
                        "stage": "batch_item",
                        "status": "done",
                        "message": (f"Finished {index}/{len(image_paths)}: {source_path.name}"),
                        "details": {
                            "child_job_id": child_id,
                            "decision": item.get("decision"),
                            "balanced": item.get("balanced"),
                        },
                    },
                )

            failed = self._failed_batch_count(items)
            artifacts = self.write_batch_summaries(batch_id, items)
            self.store.update(
                batch_id,
                completed=len(items),
                failed=failed,
                items=items,
                result={
                    "artifacts": artifacts,
                    "items": items,
                    "total": len(items),
                    "failed": failed,
                },
            )
            self.store.add_event(
                batch_id,
                {
                    "stage": "batch",
                    "status": "done",
                    "message": (
                        f"Batch finished: {len(items)} image(s), {failed} failed/rejected."
                    ),
                },
            )
        except Exception as exc:
            traceback_text = traceback.format_exc()
            artifacts = self.write_batch_summaries(batch_id, items) if items else {}
            self.store.add_event(
                batch_id,
                {
                    "stage": "batch",
                    "status": "error",
                    "message": str(exc),
                    "details": {"traceback": traceback_text[-4000:]},
                },
            )
            self.store.update(
                batch_id,
                result={"artifacts": artifacts, "items": items},
            )
            raise

    @staticmethod
    def _failed_batch_count(items: list[dict[str, Any]]) -> int:
        return sum(
            1
            for item in items
            if item.get("state") in {"failed", "error"}
            or item.get("decision") in {"reject", "llm_failed"}
        )
