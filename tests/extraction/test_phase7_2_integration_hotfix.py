from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from receipt_intelligence.extraction.contracts.presentation import CategorizationRequest
from receipt_intelligence.extraction.correction.acceptance import evaluate_candidate
from receipt_intelligence.extraction.correction.invocation import _strip_code_fences
from receipt_intelligence.extraction.correction.patching import apply_patch, target_for_strategy
from receipt_intelligence.extraction.correction.profile import load_correction_profile
from receipt_intelligence.extraction.correction.strategies.item_sum import (
    build_item_sum_patch,
    validate_item_sum_evidence,
)
from receipt_intelligence.extraction.correction.strategies.vat import build_vat_patch
from receipt_intelligence.extraction.presentation.categorization import (
    ExistingReceiptCategorizationService,
)
from receipt_intelligence.extraction.settings import DEFAULT_SCALAR_TASKS
from receipt_intelligence.extraction.stages.publish import _completed_finalization_trace
from receipt_intelligence.extraction.structured.normalization import normalize_task_answer


@dataclass
class FakeGateway:
    def generate(self, request):  # pragma: no cover - categorizer stub owns the call
        raise AssertionError("gateway should not be called directly")


def test_next_item_contract_is_adapted_for_legacy_categorizer() -> None:
    receipt = {
        "items": [
            {
                "name": "KRAWATTE",
                "final_price": 14.24,
                "original_price": 14.99,
                "discount_amount": 0.75,
            }
        ],
        "totals": {"final_purchase_total": {"final_purchase_total": 14.24}},
    }
    captured = {}

    def categorizer(value, **kwargs):
        captured["receipt"] = value
        return {
            "status": "ok",
            "receipt": {
                **value,
                "items": [{**value["items"][0], "category_key": "clothing_shoes"}],
            },
            "categories": [{"item_index": 0, "category_key": "clothing_shoes"}],
            "merchant_classification": {},
        }

    service = ExistingReceiptCategorizationService(
        llm_gateway=FakeGateway(),
        ollama_url="http://ollama",
        model="gemma4",
        categorizer=categorizer,
    )
    result = service.categorize(CategorizationRequest(run_id="r1", receipt=receipt))

    adapted_item = captured["receipt"]["items"][0]
    assert adapted_item["description"] == "KRAWATTE"
    assert adapted_item["product_description"] == "KRAWATTE"
    assert adapted_item["line_total"] == 14.24
    assert "description" not in receipt["items"][0]
    assert "line_total" not in result.receipt["items"][0]
    assert result.receipt["items"][0]["category_key"] == "clothing_shoes"


def test_scalar_normalization_removes_invalid_currency_and_unsupported_discount() -> None:
    currency_schema = {
        "type": "object",
        "properties": {"currency": {"type": ["string", "null"]}},
    }
    answer, changes = normalize_task_answer(
        task_name="vat_amount",
        answer={"currency": "?"},
        schema=currency_schema,
        evidence="R0001 :: enthaltene MWST 19 % 17,60",
    )
    assert answer["currency"] is None
    assert "$.currency:placeholder_to_null" in changes

    discount_schema = {
        "type": "object",
        "properties": {
            "discount_total": {"type": ["number", "null"]},
            "currency": {"type": ["string", "null"]},
        },
    }
    answer, changes = normalize_task_answer(
        task_name="discount_total",
        answer={"discount_total": 24.0, "currency": "eur"},
        schema=discount_schema,
        evidence="R0001 :: Louie Winter\nR0002 :: Rabatt: 20 %\nR0003 :: -24,00",
    )
    assert answer == {"discount_total": None, "currency": "EUR"}
    assert "discount_total_removed_without_explicit_aggregate_evidence" in changes


def test_explicit_aggregate_discount_is_retained() -> None:
    schema = {
        "type": "object",
        "properties": {"discount_total": {"type": ["number", "null"]}},
    }
    answer, changes = normalize_task_answer(
        task_name="discount_total",
        answer={"discount_total": 24.75},
        schema=schema,
        evidence="R0001 :: Rabatt Gesamt 24,75",
    )
    assert answer["discount_total"] == 24.75
    assert not changes

    answer, changes = normalize_task_answer(
        task_name="discount_total",
        answer={"discount_total": 24.0},
        schema=schema,
        evidence="R0001 :: Rabatt Gesamt 24,75",
    )
    assert answer["discount_total"] is None
    assert changes


def test_explicit_receipt_level_coupon_discount_is_retained() -> None:
    schema = {
        "type": "object",
        "properties": {"discount_total": {"type": ["number", "null"]}},
    }
    answer, changes = normalize_task_answer(
        task_name="discount_total",
        answer={"discount_total": 0.63},
        schema=schema,
        evidence="R0029 :: ZU BEZAHLEN 38,02\nR0030 :: Rabatt-Coupon EUR 0,63",
    )
    assert answer["discount_total"] == 0.63
    assert not changes


def test_default_scalar_tasks_cover_review_relevant_receipt_fields() -> None:
    assert {
        "receipt_number",
        "net_amount",
        "payment_method",
        "payment_received",
        "change_returned",
    } <= set(DEFAULT_SCALAR_TASKS)


def test_item_discount_arithmetic_routes_to_extended_item_evidence_strategy() -> None:
    profile = load_correction_profile(
        Path("src/receipt_intelligence/extraction/correction/config/production.json")
    )
    assert profile.profile_version == "1.2.0"
    assert profile.routes["ITEM_DISCOUNT_ARITHMETIC"] == ("item_sum_source_blocks_v3",)
    assert profile.routes["NET_PLUS_VAT_RECONCILIATION"] == (
        "vat_source_evidence_v9",
        "final_total_source_evidence_v2_4",
    )
    assert profile.strategies["item_sum_source_blocks_v3"].prompt_version == "1.1.0"


def test_item_evidence_builds_bounded_discount_arithmetic_patch() -> None:
    transcription = "\n".join(
        [
            "R0001 :: KRAWATTE",
            "R0002 :: 14,99",
            "R0003 :: Ihr Preis: 14,24",
            "R0004 :: -0,75",
            "R0005 :: Louie Winter",
            "R0006 :: 120,00",
            "R0007 :: Ihr Preis: 96,00",
            "R0008 :: Rabatt: 20 %",
            "R0009 :: -24,00",
        ]
    )
    answer = {
        "item_blocks": [
            {
                "source_rows": ["R0001", "R0002", "R0003", "R0004"],
                "name": "KRAWATTE",
                "line_amount": "14,24",
                "unit_price": None,
                "original_price": "14,99",
                "discount_amount": "-0,75",
            },
            {
                "source_rows": ["R0005", "R0006", "R0007", "R0008", "R0009"],
                "name": "Louie Winter",
                "line_amount": "96,00",
                "unit_price": None,
                "original_price": "120,00",
                "discount_amount": "-24,00",
            },
        ],
        "unresolved_candidate_rows": [],
    }
    receipt = {
        "items": [
            {
                "name": "KRAWATTE",
                "final_price": 14.24,
                "original_price": 14.99,
                "discount_amount": None,
            },
            {
                "name": "Louie Winter",
                "final_price": 96.0,
                "original_price": 100.0,
                "discount_amount": 2.4,
            },
        ]
    }

    evidence_validation = validate_item_sum_evidence(answer, transcription)
    assert evidence_validation["status"] == "valid"
    patch, diagnostics = build_item_sum_patch(answer, receipt)
    by_path = {entry["path"]: entry["value"] for entry in patch["patches"]}
    assert by_path == {
        "/items/0/discount_amount": 0.75,
        "/items/1/original_price": 120.0,
        "/items/1/discount_amount": 24.0,
    }
    assert diagnostics["status"] == "patch_built"

    target = target_for_strategy(
        "item_sum_source_blocks_v3",
        {"code": "ITEM_DISCOUNT_ARITHMETIC"},
        {"checks": []},
        receipt,
        max_patches=8,
    )
    assert set(by_path) <= set(target["permitted_value_paths"])


def test_ikea_missing_items_update_truncated_unpriced_item_without_duplicate() -> None:
    receipt = {
        "items": [
            {"name": "GLIMMA N Teel dn", "final_price": 2.49},
            {"name": "TRATT Kerzlösch Alum", "final_price": 1.99},
            {"name": "GUBBRÖRA Backpins", "final_price": 0.69},
            {"name": "KOPPLA Mfstd 3/S", "final_price": 4.99},
            {"name": "SKOGHALL Hak selbstk", "final_price": 5.99},
            {"name": "FIXA Schr/Dü", "final_price": 4.99},
            {"name": "FENOMEN Bikerze dn", "final_price": 3.49},
            {"name": "OFTAST Desstel", "final_price": None},
        ]
    }
    names_and_amounts = [
        ("GLIMMA N Teel dn 100", "2,49"),
        ("TRATT Kerzlösch Alum", "1,99"),
        ("GUBBRÖRA Backpins", "0,69"),
        ("KOPPLA Mfstd 3/S", "4,99"),
        ("SKOGHALL Hak selbstk", "5,99"),
        ("FIXA Schr/Dü 260", "4,99"),
        ("FENOMEN Bikerze dn 5", "3,49"),
        ("OFTAST Desstel 19 we", "5,88"),
        ("LACK N Wareg 30x26 w", "11,98"),
        ("PERSBY Wareg 79x26 w", "17,98"),
    ]
    answer = {
        "item_blocks": [
            {
                "source_rows": [f"R{index:04d}"],
                "name": name,
                "line_amount": amount,
                "unit_price": None,
            }
            for index, (name, amount) in enumerate(names_and_amounts, start=1)
        ],
        "unresolved_candidate_rows": [],
    }

    patch, diagnostics = build_item_sum_patch(answer, receipt)
    by_path = {entry["path"]: entry for entry in patch["patches"]}
    assert by_path["/items/7/final_price"]["value"] == 5.88
    insertions = [entry for entry in patch["patches"] if entry["op"] == "insert_array_element"]
    assert [entry["value"]["name"] for entry in insertions] == [
        "LACK N Wareg 30x26 w",
        "PERSBY Wareg 79x26 w",
    ]
    assert diagnostics["matches"][7]["method"] == "ordered_token_prefix"

    corrected = apply_patch(receipt, patch)
    assert len(corrected["items"]) == 10
    assert all(item["final_price"] is not None for item in corrected["items"])
    assert round(sum(item["final_price"] for item in corrected["items"]), 2) == 60.47


def test_ikea_item_recovery_is_retained_as_bounded_partial_improvement() -> None:
    def validation(difference: float) -> dict:
        return {
            "summary": {
                "error_count": 0,
                "review_count": 2,
                "failed": 2,
                "skipped": 3,
            },
            "checks": [
                {
                    "code": "ITEM_SUM_RECONCILIATION",
                    "status": "failed",
                    "severity": "review",
                    "values": {
                        "direct_difference": difference,
                        "absolute_direct_difference": abs(difference),
                    },
                },
                {
                    "code": "VAT_LINES_GROSS_RECONCILIATION",
                    "status": "failed",
                    "severity": "review",
                    "values": {"difference": -0.03},
                },
            ],
        }

    accepted, reasons = evaluate_candidate(
        validation(-35.87),
        validation(-0.03),
        targeted_codes={"ITEM_SUM_RECONCILIATION"},
        allow_partial_improvement=True,
    )
    assert accepted is True
    assert reasons == []

    accepted, reasons = evaluate_candidate(
        validation(-35.87),
        validation(-0.03),
        targeted_codes={"ITEM_SUM_RECONCILIATION"},
    )
    assert accepted is False
    assert reasons == ["target_not_resolved:ITEM_SUM_RECONCILIATION:status=failed"]


def test_single_vat_row_can_repair_explicit_aggregate_net_amount() -> None:
    answer = {
        "vat_evidence_blocks": [
            {
                "context_rows": ["R0036"],
                "source_row": "R0037",
                "row_label": "0",
                "fields": [
                    {"role": "rate_percent", "value": "19,0"},
                    {"role": "net_amount", "value": "50,82"},
                    {"role": "vat_amount", "value": "9,65"},
                ],
            }
        ],
        "unresolved_candidate_rows": [],
    }
    receipt = {
        "receipt_metadata": {"currency": "EUR"},
        "tax": {
            "vat_amount": {"vat_amount": 9.65, "currency": "EUR"},
            "vat_lines": [],
        },
        "totals": {"net_amount": {"net_amount": 9.65, "currency": "EUR"}},
    }
    patch, diagnostics = build_vat_patch(answer, receipt)
    by_path = {entry["path"]: entry["value"] for entry in patch["patches"]}
    assert by_path["/totals/net_amount/net_amount"] == 50.82
    assert diagnostics["aggregate_net_replaced"] is True


def test_markdown_json_fences_are_removed_without_repair() -> None:
    payload = '{"status":"resolved"}'
    assert _strip_code_fences(f"```json\n{payload}\n```") == payload


def test_final_total_schema_patterns_are_backend_compatible() -> None:
    path = Path(
        "src/receipt_intelligence/prompts/gemma/correction/"
        "final_total_source_evidence/v1.0.0/schema.json"
    )
    schema = json.loads(path.read_text(encoding="utf-8"))
    resolved = schema["oneOf"][0]["properties"]
    for field in ("label_text", "value_text"):
        pattern = resolved[field]["pattern"]
        assert pattern.startswith("^") and pattern.endswith("$")


def test_finalization_metadata_snapshot_marks_final_stage_done() -> None:
    trace = [
        {"stage": "next_categorization", "status": "done"},
        {"stage": "next_finalize", "status": "running"},
    ]
    snapshot = _completed_finalization_trace(trace)
    assert snapshot[-1]["status"] == "done"
    assert trace[-1]["status"] == "running"
