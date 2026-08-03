from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "static" / "receipt_process_view.js"


def _normalize(receipt: dict) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the process-view compatibility test.")
    script = r"""
const adapter = require(process.argv[1]);
const receipt = JSON.parse(process.argv[2]);
const before = JSON.stringify(receipt);
const normalized = adapter.normalizeReceiptForProcessView(receipt);
process.stdout.write(JSON.stringify({ normalized, unchanged: before === JSON.stringify(receipt) }));
"""
    result = subprocess.run(
        [node, "-e", script, str(ADAPTER), json.dumps(receipt)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_modepark_next_receipt_is_projected_for_process_view() -> None:
    receipt = {
        "merchant": {
            "name": "Modepark Röther",
            "address": {
                "street": "Josef-Landes-Straße 44",
                "postal_code": "87600",
                "city": "Kaufbeuren",
                "country": None,
            },
        },
        "receipt_metadata": {
            "date": None,
            "time": None,
            "receipt_number": "8620",
            "currency": "EUR",
        },
        "items": [
            {
                "name": "KRAWATTE",
                "quantity": 1,
                "final_price": 14.24,
                "original_price": 14.99,
            },
            {
                "name": "Louie Winter",
                "quantity": 1,
                "final_price": 96.0,
                "original_price": 100.0,
                "discount_amount": 2.4,
            },
        ],
        "totals": {
            "final_purchase_total": {"final_purchase_total": 110.24, "currency": "EUR"},
            "net_amount": {"net_amount": 92.64, "currency": "EUR"},
        },
        "payment": {
            "payment_method": "Lastschrift",
            "payment_received": {"payment_received": 110.24, "currency": "EUR"},
            "change_returned": None,
        },
        "tax": {
            "vat_amount": {"vat_amount": 17.6, "currency": None},
            "vat_lines": [
                {
                    "source_rows": ["R0027"],
                    "rate_percent": 19,
                    "net_amount": None,
                    "vat_amount": None,
                }
            ],
        },
        "validation": {
            "status": "review_required",
            "metrics": {
                "item_sum": 110.24,
                "final_purchase_total": 110.24,
            },
            "checks": [
                {
                    "code": "ITEM_DISCOUNT_ARITHMETIC",
                    "status": "failed",
                    "severity": "review",
                    "message": "Item discount arithmetic is inconsistent.",
                },
                {
                    "code": "ITEM_SUM_RECONCILIATION",
                    "status": "passed",
                    "severity": "info",
                    "message": "Item sum matches total.",
                    "values": {"difference": 0.0},
                },
            ],
        },
    }

    result = _normalize(receipt)
    view = result["normalized"]

    assert result["unchanged"] is True
    assert view["merchant"]["address"] == "Josef-Landes-Straße 44\n87600 Kaufbeuren"
    assert view["receipt_number"] == "8620"
    assert view["currency"] == "EUR"
    assert view["totals"] == {
        **receipt["totals"],
        "subtotal": 92.64,
        "tax_total": 17.6,
        "grand_total": 110.24,
        "paid_total": 110.24,
        "change": None,
    }
    assert [item["description"] for item in view["items"]] == ["KRAWATTE", "Louie Winter"]
    assert [item["line_total"] for item in view["items"]] == [14.24, 96.0]
    assert view["payments"] == [
        {"method": "Lastschrift", "amount": 110.24, "source_line_ids": []}
    ]
    assert view["taxes"] == [
        {"rate": 19, "net": None, "tax": None, "gross": None, "source_line_ids": ["R0027"]}
    ]
    assert view["validation"]["import_decision"] == "review_required"
    assert view["validation"]["balanced"] is True
    assert view["validation"]["difference"] == 0.0
    assert view["validation"]["calculated_item_total"] == 110.24
    assert [issue["code"] for issue in view["validation"]["issues"]] == [
        "ITEM_DISCOUNT_ARITHMETIC"
    ]


def test_legacy_receipt_fields_remain_available() -> None:
    receipt = {
        "currency": "EUR",
        "merchant": {"name": "Legacy", "address": "Example Street 1"},
        "date": "2026-01-01",
        "totals": {
            "subtotal": 10.0,
            "tax_total": 1.9,
            "grand_total": 11.9,
            "paid_total": 11.9,
            "change": 0.0,
        },
        "items": [{"description": "Item", "line_total": 11.9}],
        "payments": [{"method": "card", "amount": 11.9}],
        "taxes": [{"rate": 19, "net": 10.0, "tax": 1.9, "gross": 11.9}],
        "validation": {
            "import_decision": "import",
            "balanced": True,
            "difference": 0.0,
            "issues": [],
        },
    }

    view = _normalize(receipt)["normalized"]

    assert view["merchant"]["address"] == "Example Street 1"
    assert view["totals"]["grand_total"] == 11.9
    assert view["items"][0]["description"] == "Item"
    assert view["items"][0]["line_total"] == 11.9
    assert view["payments"][0]["method"] == "card"
    assert view["validation"]["import_decision"] == "import"
