from __future__ import annotations

import json
from pathlib import Path

from receipt_intelligence.adapters.observability import AskReceiptsJsonLogWriter


def test_ask_receipts_json_writer_creates_one_valid_json_file(tmp_path: Path) -> None:
    writer = AskReceiptsJsonLogWriter(tmp_path / "ask_receipts")

    filename = writer.write(
        {
            "schema_version": "ask_receipts_diagnostic_v1",
            "request": {"question": "Welche Quittung enthält Vittel?"},
        },
        log_id="q_test/unsafe",
    )

    assert filename is not None
    assert "/" not in filename
    path = tmp_path / "ask_receipts" / filename
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["request"]["question"] == "Welche Quittung enthält Vittel?"
    assert not list(path.parent.glob("*.tmp"))


def test_disabled_ask_receipts_json_writer_does_not_touch_disk(tmp_path: Path) -> None:
    writer = AskReceiptsJsonLogWriter(tmp_path / "ask_receipts", enabled=False)

    assert writer.write({"status": "completed"}, log_id="q_test") is None
    assert not (tmp_path / "ask_receipts").exists()
