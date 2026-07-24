"""Receipt upload, batch execution, job status, and artifact endpoints."""

from __future__ import annotations

from flask import Blueprint, jsonify, request, send_file

import receipt_intelligence.settings as settings
from receipt_intelligence.application.errors import ApplicationError
from receipt_intelligence.application.use_cases.jobs import (
    StartBatchCommand,
    SubmitReceiptCommand,
)
from receipt_intelligence.web.dependencies import get_app_services
from receipt_intelligence.web.errors import application_error_response, unexpected_error_response
from receipt_intelligence.web.presentation import present_job_submission, present_resources
from receipt_intelligence.web.request_parsing import as_bool, build_options_from_request, form_int

jobs_bp = Blueprint("jobs", __name__)


@jobs_bp.post("/api/upload")
def upload():
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"error": "Missing file field named 'file'."}), 400
    try:
        result = get_app_services().jobs.submit_receipt(
            SubmitReceiptCommand(
                filename=file.filename,
                stream=file.stream,
                options=build_options_from_request(),
            )
        )
        return jsonify(present_job_submission(result))
    except ApplicationError as exc:
        return application_error_response(exc)
    except Exception as exc:
        return unexpected_error_response(exc)


@jobs_bp.post("/api/batch/start")
def batch_start():
    try:
        result = get_app_services().jobs.start_batch(
            StartBatchCommand(
                folder_path=request.form.get("folder_path"),
                recursive=as_bool(
                    request.form.get("recursive"),
                    settings.BATCH_RECURSIVE_DEFAULT,
                ),
                max_files=form_int("max_files", settings.BATCH_MAX_FILES),
                options=build_options_from_request(),
            )
        )
        return jsonify(present_job_submission(result))
    except ApplicationError as exc:
        return application_error_response(exc)
    except Exception as exc:
        return unexpected_error_response(exc)


@jobs_bp.get("/api/status/<job_id>")
def status(job_id: str):
    try:
        return jsonify(present_resources(get_app_services().jobs.get_job(job_id)))
    except ApplicationError as exc:
        return application_error_response(exc)


@jobs_bp.get("/api/jobs")
def jobs():
    return jsonify(
        {"jobs": present_resources(get_app_services().jobs.list_jobs(limit=25))}
    )


@jobs_bp.get("/api/jobs/<job_id>/manifest")
def job_manifest(job_id: str):
    try:
        return jsonify(present_resources(get_app_services().jobs.get_manifest(job_id)))
    except ApplicationError as exc:
        return application_error_response(exc)


@jobs_bp.get("/api/artifact/<job_id>/<path:filename>")
def artifact(job_id: str, filename: str):
    try:
        return send_file(
            get_app_services().jobs.get_artifact(job_id, filename),
            as_attachment=False,
        )
    except ApplicationError as exc:
        return application_error_response(exc)
