from __future__ import annotations

from pathlib import Path


def test_model_dashboard_has_usage_cost_and_pricing_controls() -> None:
    root = Path(__file__).resolve().parents[3]
    html = (root / "static" / "index.html").read_text(encoding="utf-8")
    javascript = (root / "static" / "app.js").read_text(encoding="utf-8")

    assert 'data-tab-target="models"' in html
    assert 'id="model-call-summary"' in html
    assert 'id="model-call-table"' in html
    assert 'id="model-pricing-form"' in html
    assert "/api/model-calls/summary" in javascript
    assert "/api/model-pricing" in javascript
