"""Receipt upload, batch execution, job status, and artifact endpoints."""

from __future__ import annotations

import threading
import uuid

from flask import Blueprint, jsonify, request, send_file
from werkzeug.utils import secure_filename

import receipt_intelligence.settings as settings
from receipt_intelligence.services.artifact_service import safe_artifact_path
from receipt_intelligence.services.job_processing import JobProcessingService
from receipt_intelligence.web.dependencies import get_app_services
from receipt_intelligence.web.request_parsing import (
    as_bool,
    build_options_from_request,
    form_int,
)

jobs_bp = Blueprint("jobs", __name__)


def _processor() -> JobProcessingService:
    services = get_app_services()
    return JobProcessingService(services.job_store, services.receipt_db)


@jobs_bp.post("/api/upload")
def upload():
    services = get_app_services()
    processor = _processor()
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"error": "Missing file field named 'file'."}), 400
    if not processor.allowed_file(file.filename):
        return jsonify(
            {"error": (f"Unsupported file type. Allowed: {sorted(settings.ALLOWED_EXTENSIONS)}")}
        ), 400

    job_id = uuid.uuid4().hex[:12]
    job_dir = services.job_store.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    filename = secure_filename(file.filename) or f"receipt_{job_id}.jpg"
    image_path = job_dir / filename
    file.save(image_path)

    options = build_options_from_request()
    services.job_store.create(
        job_id,
        {
            "filename": filename,
            "image_path": str(image_path),
            "options": options,
        },
    )

    thread = threading.Thread(
        target=processor.run_job,
        args=(job_id, image_path, options),
        daemon=True,
    )
    thread.start()
    return jsonify({"job_id": job_id, "status_url": f"/api/status/{job_id}"})


@jobs_bp.post("/api/batch/start")
def batch_start():
    services = get_app_services()
    processor = _processor()
    try:
        folder = processor.resolve_batch_folder(request.form.get("folder_path"))
        recursive = as_bool(
            request.form.get("recursive"),
            settings.BATCH_RECURSIVE_DEFAULT,
        )
        max_files = form_int("max_files", settings.BATCH_MAX_FILES)
        image_paths = processor.list_batch_images(
            folder,
            recursive=recursive,
            max_files=max_files,
        )
        if not image_paths:
            return jsonify({"error": f"No supported receipt images found in {folder}."}), 400

        options = build_options_from_request()
        batch_id = "batch_" + uuid.uuid4().hex[:10]
        batch_dir = services.job_store.job_dir(batch_id)
        batch_dir.mkdir(parents=True, exist_ok=True)
        services.job_store.create(
            batch_id,
            {
                "type": "batch",
                "folder_path": str(folder),
                "recursive": recursive,
                "max_files": max_files,
                "total": len(image_paths),
                "completed": 0,
                "failed": 0,
                "items": [],
                "options": options,
            },
        )
        thread = threading.Thread(
            target=processor.run_batch_job,
            args=(batch_id, image_paths, options),
            daemon=True,
        )
        thread.start()
        return jsonify(
            {
                "batch_id": batch_id,
                "job_id": batch_id,
                "status_url": f"/api/status/{batch_id}",
                "total": len(image_paths),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@jobs_bp.get("/api/status/<job_id>")
def status(job_id: str):
    job = get_app_services().job_store.get(job_id)
    if job is None:
        return jsonify({"error": "job not found"}), 404
    return jsonify(job)


@jobs_bp.get("/api/jobs")
def jobs():
    return jsonify({"jobs": get_app_services().job_store.list_recent(limit=25)})


@jobs_bp.get("/api/jobs/<job_id>/manifest")
def job_manifest(job_id: str):
    store = get_app_services().job_store
    job = store.get(job_id)
    if job is None:
        return jsonify({"error": "job not found"}), 404
    manifest = store.get_manifest(job_id, create_if_missing=True)
    if manifest is None:
        return jsonify({"error": "manifest not found"}), 404
    return jsonify(manifest)


@jobs_bp.get("/api/artifact/<job_id>/<path:filename>")
def artifact(job_id: str, filename: str):
    store = get_app_services().job_store
    requested = safe_artifact_path(store, job_id, filename)
    if requested is None:
        return jsonify({"error": "invalid artifact path"}), 400
    if not requested.exists():
        return jsonify({"error": "artifact not found"}), 404
    return send_file(requested, as_attachment=False)
