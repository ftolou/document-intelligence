from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from receipt_intelligence.extraction.contracts.presentation import CategorizationRequest
from receipt_intelligence.extraction.correction.patching import target_for_strategy
from receipt_intelligence.extraction.correction.profile import load_correction_profile
from receipt_intelligence.extraction.correction.strategies.item_sum import (
    build_item_sum_patch,
    validate_item_sum_evidence,
)
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


def test_default_scalar_tasks_cover_review_relevant_receipt_fields() -> None:
    assert {
        "receipt_number",
        "net_amount",
        "payment_method",
        "payment_received",
    } <= set(DEFAULT_SCALAR_TASKS)


def test_item_discount_arithmetic_routes_to_extended_item_evidence_strategy() -> None:
    profile = load_correction_profile(
        Path("src/receipt_intelligence/extraction/correction/config/production.json")
    )
    assert profile.profile_version == "1.1.0"
    assert profile.routes["ITEM_DISCOUNT_ARITHMETIC"] == ("item_sum_source_blocks_v3",)
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


def test_finalization_metadata_snapshot_marks_final_stage_done() -> None:
    trace = [
        {"stage": "next_categorization", "status": "done"},
        {"stage": "next_finalize", "status": "running"},
    ]
    snapshot = _completed_finalization_trace(trace)
    assert snapshot[-1]["status"] == "done"
    assert trace[-1]["status"] == "running"
