"""Root, health, and runtime-configuration endpoints."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify

from receipt_intelligence.web.dependencies import get_app_services

core_bp = Blueprint("core", __name__)


@core_bp.get("/")
def index():
    """Serve the static application shell."""

    return current_app.send_static_file("index.html")


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
