"""Root, health, and runtime-configuration endpoints."""

from __future__ import annotations

from flask import Blueprint, Response, current_app, jsonify

from receipt_intelligence.web.dependencies import get_app_services

core_bp = Blueprint("core", __name__)


@core_bp.get("/")
def index():
    """Serve the static shell with small optional dashboard extensions."""

    static_folder = current_app.static_folder
    if not static_folder:
        return current_app.send_static_file("index.html")
    index_path = current_app.open_resource(str(static_folder) + "/index.html")
    with index_path as handle:
        html = handle.read().decode("utf-8")
    marker = '<script src="/app.js"></script>'
    extension = marker + '\n  <script src="/model_pricing.js"></script>'
    if marker in html and "/model_pricing.js" not in html:
        html = html.replace(marker, extension, 1)
    return Response(html, mimetype="text/html")


@core_bp.get("/health")
def health():
    return jsonify(get_app_services().runtime.health())


@core_bp.get("/api/readiness")
def readiness():
    payload = get_app_services().runtime.readiness()
    return jsonify(payload), 200 if payload["ready"] else 503


@core_bp.get("/api/config")
def config():
    return jsonify(get_app_services().runtime.configuration())
