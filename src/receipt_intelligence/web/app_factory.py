"""Flask application factory for Receipt Intelligence."""

from __future__ import annotations

from flask import Flask

import receipt_intelligence.settings as settings
from receipt_intelligence.application.use_cases.query import ReceiptQueryExecutor
from receipt_intelligence.runtime.paths import RuntimePaths
from receipt_intelligence.storage.job_store import JobStore
from receipt_intelligence.storage.receipt_db import ReceiptDatabase
from receipt_intelligence.web.dependencies import init_app_services
from receipt_intelligence.web.routes.core import core_bp
from receipt_intelligence.web.routes.jobs import jobs_bp
from receipt_intelligence.web.routes.query import query_bp
from receipt_intelligence.web.routes.receipts import receipts_bp
from receipt_intelligence.web.routes.review import review_bp


def create_app(
    *,
    job_store: JobStore | None = None,
    receipt_db: ReceiptDatabase | None = None,
    runtime_paths: RuntimePaths | None = None,
    receipt_query_service: ReceiptQueryExecutor | None = None,
    testing: bool = False,
) -> Flask:
    app = Flask(
        __name__,
        static_folder=str(settings.STATIC_DIR),
        static_url_path="",
    )
    app.config.update(
        MAX_CONTENT_LENGTH=settings.MAX_UPLOAD_MB * 1024 * 1024,
        TESTING=testing,
    )

    init_app_services(
        app,
        job_store=job_store,
        receipt_db=receipt_db,
        runtime_paths=runtime_paths,
        receipt_query_service=receipt_query_service,
    )

    app.register_blueprint(core_bp)
    app.register_blueprint(jobs_bp)
    app.register_blueprint(review_bp)
    app.register_blueprint(receipts_bp)
    app.register_blueprint(query_bp)
    return app
