from __future__ import annotations

from types import SimpleNamespace

from receipt_intelligence.pipeline import integrated_receipt_pipeline as entrypoint


def test_result_adapter_preserves_application_contract() -> None:
    final = SimpleNamespace(
        as_application_result=lambda: {
            "receipt": {"items": []},
            "report": {"status": "valid"},
            "paths": {},
            "logs": [],
            "pipeline_meta": {"workflow": {"name": "ReceiptExtractionWorkflow"}},
            "observability": {},
        }
    )
    context = SimpleNamespace(
        require_finalized=lambda: SimpleNamespace(result=final),
        paths={"receipt_final": "final.json", "extraction_metrics": "metrics.json"},
        logs=[{"stage": "pipeline", "status": "done"}],
        stage_trace=[{"stage": "prepare", "status": "done"}],
    )

    result = entrypoint._application_result(context)

    assert result["report"]["status"] == "valid"
    assert result["paths"]["receipt_final"] == "final.json"
    assert result["observability"]["metrics_path"] == "metrics.json"
