from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_review_ui_uses_compact_table_editor() -> None:
    javascript = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    styles = (ROOT / "static" / "styles.css").read_text(encoding="utf-8")

    assert 'class="review-item-table"' in javascript
    assert "data-review-item-details=" in javascript
    assert "data-review-item-reject=" in javascript
    assert "Reject excludes a row from approval" in javascript
    assert ".review-item-table" in styles
    assert ".review-control-dirty" in styles


def test_review_validation_can_navigate_to_editable_fields() -> None:
    javascript = (ROOT / "static" / "app.js").read_text(encoding="utf-8")

    assert "function reviewIssueFocusTarget" in javascript
    assert "data-review-focus=" in javascript
    assert "control.scrollIntoView" in javascript


def test_review_json_is_hidden_under_technical_details() -> None:
    javascript = (ROOT / "static" / "app.js").read_text(encoding="utf-8")

    assert 'class="review-technical-details"' in javascript
    assert "Current review document JSON" in javascript
    assert "JSON.stringify(receipt, null, 2)" in javascript
