"""Focused tests for transport-neutral application contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from receipt_intelligence.application.errors import InvalidRequestError
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

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def create(self, job_id: str, payload: dict) -> dict:
        self.created.append((job_id, payload))
        return payload

    def get(self, _job_id: str):
        return None

    def list_recent(self, *, limit: int):
        return []


class _Processor:
    def allowed_file(self, filename: str) -> bool:
        return filename.lower().endswith(".jpg")

    def run_job(self, *_args) -> None:
        return None


class _ImmediateThread:
    def __init__(self, *, target, args, daemon) -> None:
        self.target = target
        self.args = args
        self.daemon = daemon

    def start(self) -> None:
        self.target(*self.args)


def test_job_submission_is_orchestrated_outside_http(monkeypatch, tmp_path: Path) -> None:
    from io import BytesIO

    from receipt_intelligence.application.use_cases.jobs import (
        JobUseCases,
        SubmitReceiptCommand,
    )

    monkeypatch.setattr(
        "receipt_intelligence.application.use_cases.jobs.threading.Thread",
        _ImmediateThread,
    )
    store = _JobStore(tmp_path)
    use_cases = JobUseCases(store, _Processor())  # type: ignore[arg-type]

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
