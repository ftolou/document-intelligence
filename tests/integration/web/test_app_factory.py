"""Verify the Flask application factory preserves the public API contract."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

from receipt_intelligence.application.use_cases.query import ReceiptQueryExecutor
from receipt_intelligence.runtime.paths import RuntimePaths
from receipt_intelligence.storage.job_store import JobStore
from receipt_intelligence.storage.receipt_db import ReceiptDatabase
from receipt_intelligence.web.app_factory import create_app

EXPECTED_API_RULES = {
    "/api/model-pricing",
    "/api/model-calls/summary",
    "/api/model-calls",
    "/api/artifact/<job_id>/<path:filename>",
    "/api/ask-receipts",
    "/api/batch/start",
    "/api/config",
    "/api/jobs",
    "/api/jobs/<job_id>/manifest",
    "/api/readiness",
    "/api/receipt-db/delete-all",
    "/api/receipt-db/receipts",
    "/api/receipt-db/receipts/<int:receipt_id>",
    "/api/receipt-db/receipts/<int:receipt_id>/review",
    "/api/receipt-db/receipts/<receipt_id>",
    "/api/receipt-db/summary",
    "/api/receipts/import/<job_id>",
    "/api/review-queue",
    "/api/review-queue/<job_id>/status",
    "/api/review/<job_id>",
    "/api/status/<job_id>",
    "/api/upload",
}


@dataclass
class _StubQueryService:
    response: dict[str, Any] | None = None
    error: Exception | None = None
    calls: list[tuple[str, int]] = field(default_factory=list)

    def execute(self, question: str, *, limit: int = 25) -> dict[str, Any]:
        self.calls.append((question, limit))
        if self.error is not None:
            raise self.error
        return dict(
            self.response
            or {
                "strategy": "rag_sql",
                "question": question,
                "status": "completed",
                "answer": "ok",
            }
        )


def _test_app(tmp_path: Path, *, receipt_query_service: ReceiptQueryExecutor | None = None):
    runtime_paths = RuntimePaths.from_environment(tmp_path, environ={})
    runtime_paths.ensure_directories()
    store = JobStore(runtime_paths.jobs_dir)
    database = ReceiptDatabase(runtime_paths.receipt_db_path)
    return (
        create_app(
            job_store=store,
            receipt_db=database,
            runtime_paths=runtime_paths,
            receipt_query_service=receipt_query_service or _StubQueryService(),
            testing=True,
        ),
        database,
        store,
    )


def test_factory_registers_existing_api_routes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        app, _, _ = _test_app(Path(directory))
        rules = {rule.rule for rule in app.url_map.iter_rules() if rule.rule.startswith("/api/")}
        assert rules == EXPECTED_API_RULES


def test_health_config_and_database_summary_are_available() -> None:
    with tempfile.TemporaryDirectory() as directory:
        app, _, _ = _test_app(Path(directory))
        client = app.test_client()

        health = client.get("/health")
        assert health.status_code == 200
        assert health.get_json()["ok"] is True

        config = client.get("/api/config")
        assert config.status_code == 200
        assert "app_version" in config.get_json()
        assert config.get_json()["query_engine"] == {
            "name": "rag_sql",
            "orchestrator": "langgraph",
            "graph_version": "rag_sql_graph_v2",
        }

        with (
            patch(
                "receipt_intelligence.services.runtime_information.settings.READINESS_PROBE_OLLAMA",
                False,
            ),
            patch(
                "receipt_intelligence.services.runtime_information.settings.READINESS_PROBE_VLM",
                False,
            ),
        ):
            readiness = client.get("/api/readiness")
        assert readiness.status_code == 200
        assert readiness.get_json()["ready"] is True
        assert readiness.get_json()["checks"]["database"]["status"] == "ok"

        summary = client.get("/api/receipt-db/summary")
        assert summary.status_code == 200
        assert summary.get_json()["receipt_count"] == 0


def test_query_route_uses_rag_sql_service_and_clamps_limit() -> None:
    with tempfile.TemporaryDirectory() as directory:
        service = _StubQueryService(
            response={
                "strategy": "rag_sql",
                "question": "How much at REWE?",
                "status": "completed",
                "answer": "Result: 20.00 EUR.",
            }
        )
        app, _, _ = _test_app(
            Path(directory),
            receipt_query_service=service,  # type: ignore[arg-type]
        )
        response = app.test_client().post(
            "/api/ask-receipts",
            json={"question": "How much at REWE?", "limit": 500},
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["strategy"] == "rag_sql"
        assert payload["status"] == "completed"
        assert payload["answer"] == "Result: 20.00 EUR."
        assert payload["strategy"] == "rag_sql"
        assert service.calls == [("How much at REWE?", 100)]


def test_query_route_rejects_removed_request_fields() -> None:
    with tempfile.TemporaryDirectory() as directory:
        service = _StubQueryService()
        app, _, _ = _test_app(
            Path(directory),
            receipt_query_service=service,  # type: ignore[arg-type]
        )
        response = app.test_client().post(
            "/api/ask-receipts",
            json={"question": "Test", "unexpected": "value"},
        )

        assert response.status_code == 400
        payload = response.get_json()
        assert payload["error_code"] == "unsupported_request_field"
        assert service.calls == []


def test_query_route_returns_single_engine_failure() -> None:
    with tempfile.TemporaryDirectory() as directory:
        service = _StubQueryService(error=RuntimeError("RAG-SQL failed"))
        app, _, _ = _test_app(
            Path(directory),
            receipt_query_service=service,  # type: ignore[arg-type]
        )
        response = app.test_client().post(
            "/api/ask-receipts",
            json={"question": "How much for shoes?"},
        )

        assert response.status_code == 500
        payload = response.get_json()
        assert payload["status"] == "error"
        assert payload["strategy"] == "rag_sql"
        assert payload["error_code"] == "query_execution_failed"
        assert service.calls == [("How much for shoes?", 25)]


def test_query_route_rejects_empty_question() -> None:
    with tempfile.TemporaryDirectory() as directory:
        app, _, _ = _test_app(Path(directory))
        response = app.test_client().post("/api/ask-receipts", json={"question": "  "})
        assert response.status_code == 400
        payload = response.get_json()
        assert payload["status"] == "error"
        assert payload["error_code"] == "missing_question"
        assert payload["answer"] == "Enter a question about your approved receipts."
        assert payload["error"] == "Missing question."


def test_receipt_navigation_endpoint_returns_safe_job_metadata() -> None:
    with tempfile.TemporaryDirectory() as directory:
        app, database, _ = _test_app(Path(directory))
        imported = database.import_receipt(
            job_id="receipt-job-1",
            receipt={
                "merchant": {"name": "LIDL"},
                "date": "2026-07-20",
                "currency": "EUR",
                "totals": {"grand_total": 12.85, "paid_total": 12.85},
                "items": [{"description": "Vittel", "category": "item", "line_total": 5.10}],
                "human_review": {"status": "approved"},
            },
        )

        response = app.test_client().get(f"/api/receipt-db/receipts/{imported.receipt_db_id}")

        assert response.status_code == 200
        receipt = response.get_json()["receipt"]
        assert receipt["id"] == imported.receipt_db_id
        assert receipt["job_id"] == "receipt-job-1"
        assert receipt["merchant_name"] == "LIDL"
        assert receipt["grand_total"] == 12.85
        assert receipt["item_count"] == 1
        assert "approved_receipt_path" not in receipt
        assert "image_path" not in receipt


def test_receipt_navigation_endpoint_returns_404_for_missing_receipt() -> None:
    with tempfile.TemporaryDirectory() as directory:
        app, _, _ = _test_app(Path(directory))
        response = app.test_client().get("/api/receipt-db/receipts/99999")
        assert response.status_code == 404
        assert response.get_json() == {"error": "receipt not found"}


def test_database_review_endpoint_uses_authoritative_database_even_with_artifact() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        app, database, store = _test_app(root)
        job_dir = store.job_dir("approved-job")
        job_dir.mkdir(parents=True, exist_ok=True)
        approved_path = job_dir / "approved_receipt.json"
        approved_receipt = {
            "merchant": {"name": "APPROVED LIDL"},
            "date": "2026-07-20",
            "currency": "EUR",
            "totals": {"grand_total": 12.85},
            "items": [{"description": "Vittel", "category": "item", "line_total": 5.10}],
            "human_review": {"status": "approved", "reviewer": "tester"},
        }
        approved_path.write_text(json.dumps(approved_receipt), encoding="utf-8")
        imported = database.import_receipt(
            job_id="approved-job",
            receipt={**approved_receipt, "merchant": {"name": "DATABASE SNAPSHOT"}},
            approved_receipt_path=approved_path,
        )

        response = app.test_client().get(
            f"/api/receipt-db/receipts/{imported.receipt_db_id}/review"
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["receipt"]["merchant"]["name"] == "DATABASE SNAPSHOT"
        assert payload["source"] == "database"
        assert payload["editable"] is True
        assert payload["save_url"] == (f"/api/receipt-db/receipts/{imported.receipt_db_id}/review")
        assert payload["save_method"] == "PUT"
        assert payload["read_only_reason"] is None


def test_database_review_endpoint_is_editable_without_any_job_artifact() -> None:
    with tempfile.TemporaryDirectory() as directory:
        app, database, _ = _test_app(Path(directory))
        imported = database.import_receipt(
            job_id="snapshot-only-job",
            receipt={
                "merchant": {"name": "REWE"},
                "currency": "EUR",
                "totals": {"grand_total": 8.5},
                "items": [],
                "human_review": {"status": "approved"},
            },
        )

        response = app.test_client().get(
            f"/api/receipt-db/receipts/{imported.receipt_db_id}/review"
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["receipt"]["merchant"]["name"] == "REWE"
        assert payload["source"] == "database"
        assert payload["editable"] is True
        assert payload["save_url"] == (f"/api/receipt-db/receipts/{imported.receipt_db_id}/review")
        assert payload["save_method"] == "PUT"
        assert payload["read_only_reason"] is None


def test_database_review_put_updates_sqlite_without_artifacts() -> None:
    with tempfile.TemporaryDirectory() as directory:
        app, database, _ = _test_app(Path(directory))
        imported = database.import_receipt(
            job_id="database-edit-job",
            receipt={
                "merchant": {"name": "REWE"},
                "date": "2026-07-20",
                "currency": "EUR",
                "totals": {"grand_total": 3.0, "paid_total": 3.0},
                "items": [
                    {
                        "product_description": "VITTEL",
                        "normalized_name": "vittel",
                        "category": "item",
                        "category_group": "Drinks",
                        "category_key": "water",
                        "line_total": 3.0,
                    }
                ],
                "human_review": {"status": "approved"},
            },
        )

        with patch(
            "receipt_intelligence.services.database_receipt_editor.settings.RAG_EMBEDDING_ENABLED",
            False,
        ):
            response = app.test_client().put(
                f"/api/receipt-db/receipts/{imported.receipt_db_id}/review",
                json={
                    "fields": {"merchant_name": "REWE CITY", "grand_total": 3.5},
                    "items": [
                        {
                            "index": 0,
                            "product_description": "VITTEL CLASSIC",
                            "category_group": "Beverages",
                            "category_key": "mineral_water",
                            "line_total": 3.5,
                        }
                    ],
                    "review": {"status": "approved", "reviewer": "tester"},
                },
            )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["receipt"]["merchant"]["name"] == "REWE CITY"
        assert payload["receipt"]["items"][0]["product_description"] == "VITTEL CLASSIC"
        assert payload["database_update"]["semantic_item_ids"]
        assert payload["semantic_index"]["status"] == "pending"
        stored = database.get_receipt_edit_document(imported.receipt_db_id)
        assert stored is not None
        assert stored["totals"]["grand_total"] == 3.5


def test_job_review_prefers_and_can_update_approved_data_without_job_status() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        app, database, store = _test_app(root)
        job_dir = store.job_dir("durable-job")
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "receipt_final.json").write_text(
            json.dumps({"merchant": {"name": "ORIGINAL"}, "totals": {}, "items": []}),
            encoding="utf-8",
        )
        approved_path = job_dir / "approved_receipt.json"
        approved_path.write_text(
            json.dumps(
                {
                    "merchant": {"name": "APPROVED"},
                    "currency": "EUR",
                    "totals": {"grand_total": 10.0},
                    "items": [],
                    "human_review": {"status": "approved"},
                }
            ),
            encoding="utf-8",
        )
        database.import_receipt(
            job_id="durable-job",
            receipt=json.loads(approved_path.read_text(encoding="utf-8")),
            approved_receipt_path=approved_path,
            source_receipt_path=job_dir / "receipt_final.json",
        )

        get_response = app.test_client().get("/api/review/durable-job")
        assert get_response.status_code == 200
        assert get_response.get_json()["receipt"]["merchant"]["name"] == "APPROVED"

        save_response = app.test_client().post(
            "/api/review/durable-job",
            json={
                "fields": {"merchant_name": "CORRECTED"},
                "items": [],
                "review": {"status": "approved", "reviewer": "tester"},
            },
        )
        assert save_response.status_code == 200
        assert save_response.get_json()["receipt"]["merchant"]["name"] == "CORRECTED"
        saved_receipt = json.loads(approved_path.read_text(encoding="utf-8"))
        assert saved_receipt["merchant"]["name"] == "CORRECTED"


def test_job_manifest_endpoint_returns_generated_manifest() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        app, _, store = _test_app(root)
        image = store.job_dir("job1") / "receipt.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"image")
        store.create(
            "job1",
            {"filename": image.name, "image_path": str(image)},
        )

        response = app.test_client().get("/api/jobs/job1/manifest")

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["schema_version"] == "job_manifest_v1"
        assert payload["artifacts"]["input_image"]["path"] == "receipt.jpg"
