"""Focused tests for transport-neutral application contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from receipt_intelligence.adapters.observability import AskReceiptsJsonLogWriter
from receipt_intelligence.application.errors import InvalidRequestError
from receipt_intelligence.application.query_diagnostics import record_query_diagnostic
from receipt_intelligence.application.resources import artifact_reference
from receipt_intelligence.application.use_cases.query import AskReceipts
from receipt_intelligence.web.presentation import present_resources, present_review


class _QueryService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def execute(self, question: str, *, limit: int = 25) -> dict:
        self.calls.append((question, limit))
        return {"question": question, "limit": limit}


def test_query_use_case_validates_and_clamps_without_flask() -> None:
    service = _QueryService()
    use_case = AskReceipts(service)

    result = use_case.execute({"question": "  total at REWE? ", "limit": 999})

    assert result == {"question": "total at REWE?", "limit": 100}
    assert service.calls == [("total at REWE?", 100)]


def test_query_use_case_rejects_transport_payload_errors() -> None:
    use_case = AskReceipts(_QueryService())

    with pytest.raises(InvalidRequestError) as error:
        use_case.execute({"question": "test", "unexpected": True})

    assert error.value.code == "unsupported_request_field"


class _DiagnosticQueryService:
    def execute(self, question: str, *, limit: int = 25) -> dict:
        record_query_diagnostic(
            "test.stage",
            {"question": question, "limit": limit, "raw_output": "model output"},
        )
        return {
            "question": question,
            "status": "completed",
            "execution": {"query_id": "q_diagnostic"},
        }


class _FailingDiagnosticQueryService:
    def execute(self, question: str, *, limit: int = 25) -> dict:
        record_query_diagnostic("test.stage", {"question": question})
        raise RuntimeError("candidate resolution failed")


def test_query_use_case_writes_opt_in_json_diagnostics(tmp_path: Path) -> None:
    log_dir = tmp_path / "ask_receipts"
    use_case = AskReceipts(
        _DiagnosticQueryService(),
        log_writer=AskReceiptsJsonLogWriter(log_dir),
    )

    result = use_case.execute({"question": "Vittel", "limit": 25, "save_json_log": True})

    assert result["diagnostic_log"]["saved"] is True
    log_path = log_dir / result["diagnostic_log"]["filename"]
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert payload["query_id"] == "q_diagnostic"
    assert payload["request"] == {"question": "Vittel", "limit": 25}
    assert payload["diagnostic_events"][0]["raw_output"] == "model output"


def test_query_use_case_writes_failure_log_and_preserves_exception(tmp_path: Path) -> None:
    log_dir = tmp_path / "ask_receipts"
    use_case = AskReceipts(
        _FailingDiagnosticQueryService(),
        log_writer=AskReceiptsJsonLogWriter(log_dir),
    )

    with pytest.raises(RuntimeError, match="candidate resolution failed") as error:
        use_case.execute({"question": "Vittel", "save_json_log": True})

    metadata = error.value.diagnostic_log
    assert metadata["saved"] is True
    payload = json.loads((log_dir / metadata["filename"]).read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert payload["exception"]["type"] == "RuntimeError"
    assert "candidate resolution failed" in payload["exception"]["traceback"]
    assert payload["diagnostic_events"][0]["event"] == "test.stage"


def test_artifact_references_are_transport_neutral_until_presented() -> None:
    reference = artifact_reference("job-1", Path("approved_receipt.json"))

    assert "/api/" not in str(reference)
    assert present_resources(reference) == "/api/artifact/job-1/approved_receipt.json"


def test_review_presenter_adds_http_edit_link() -> None:
    payload = present_review(
        {
            "receipt_db_id": 42,
            "source": "database",
            "editable": True,
            "artifacts": {},
        }
    )

    assert payload["save_url"] == "/api/receipt-db/receipts/42/review"
    assert payload["save_method"] == "PUT"


class _JobStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.created: list[tuple[str, dict]] = []
        self.failures: list[tuple[str, dict]] = []

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def create(self, job_id: str, payload: dict) -> dict:
        self.created.append((job_id, payload))
        return payload

    def get(self, _job_id: str):
        return None

    def list_recent(self, *, limit: int):
        return []

    def fail(self, job_id: str, error: dict) -> None:
        self.failures.append((job_id, error))


class _Processor:
    def allowed_file(self, filename: str) -> bool:
        return filename.lower().endswith(".jpg")

    def run_job(self, *_args) -> None:
        return None


class _Dispatcher:
    def __init__(self) -> None:
        self.requests = []

    def submit(self, request) -> None:
        self.requests.append(request)

    def recover_pending(self) -> int:
        return 0

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        return None


def test_job_submission_is_orchestrated_outside_http(tmp_path: Path) -> None:
    from io import BytesIO

    from receipt_intelligence.application.use_cases.jobs import (
        JobUseCases,
        SubmitReceiptCommand,
    )

    store = _JobStore(tmp_path)
    dispatcher = _Dispatcher()
    use_cases = JobUseCases(store, _Processor(), dispatcher)  # type: ignore[arg-type]

    result = use_cases.submit_receipt(
        SubmitReceiptCommand(
            filename="receipt image.jpg",
            stream=BytesIO(b"image"),
            options={"test": True},
        )
    )

    assert result["state"] == "queued"
    assert "status_url" not in result
    _, payload = store.created[0]
    assert payload["filename"] == "receipt_image.jpg"
    assert Path(payload["image_path"]).read_bytes() == b"image"
    assert payload["dispatch"]["kind"] == "receipt"
    assert dispatcher.requests[0].job_id == result["job_id"]


def test_job_submission_persists_queue_rejection(tmp_path: Path) -> None:
    from io import BytesIO

    from receipt_intelligence.application.errors import ServiceUnavailableError
    from receipt_intelligence.application.ports.jobs import JobQueueFullError
    from receipt_intelligence.application.use_cases.jobs import (
        JobUseCases,
        SubmitReceiptCommand,
    )

    class _FullDispatcher(_Dispatcher):
        def submit(self, request) -> None:
            raise JobQueueFullError("Background job queue is full.")

    store = _JobStore(tmp_path)
    use_cases = JobUseCases(store, _Processor(), _FullDispatcher())  # type: ignore[arg-type]

    with pytest.raises(ServiceUnavailableError) as error:
        use_cases.submit_receipt(
            SubmitReceiptCommand(
                filename="receipt.jpg",
                stream=BytesIO(b"image"),
                options={},
            )
        )

    assert error.value.code == "job_queue_full"
    assert store.failures[0][1]["code"] == "job_queue_full"
