from __future__ import annotations

from receipt_intelligence.rag_sql.formatter import format_rag_sql_answer, format_rag_sql_outcome
from receipt_intelligence.rag_sql.models import (
    ResolvedSemanticEntity,
    SqlExecutionResult,
    ValidatedSqlPlan,
)


def _plan(
    *,
    result_shape: str,
    result_entity: str = "result",
    display_columns: list[str] | None = None,
    instruction: str = "Report result.",
) -> ValidatedSqlPlan:
    return ValidatedSqlPlan(
        sql="SELECT 1",
        parameters={},
        result_shape=result_shape,  # type: ignore[arg-type]
        result_entity=result_entity,
        display_columns=display_columns or [],
        answer_instruction=instruction,
    )


def _execution(rows: list[dict[str, object]], *, truncated: bool = False) -> SqlExecutionResult:
    columns = list(rows[0]) if rows else []
    return SqlExecutionResult(
        columns=columns,
        rows=rows,
        row_count=len(rows),
        truncated=truncated,
    )


def _shoes_entity() -> ResolvedSemanticEntity:
    return ResolvedSemanticEntity(
        entity_id="e001",
        search_text="Schuhe",
        status="resolved",
        selected_item_ids=[84, 126],
    )


def _vittel_entity() -> ResolvedSemanticEntity:
    return ResolvedSemanticEntity(
        entity_id="e001",
        search_text="Vittel",
        status="resolved",
        selected_item_ids=[127],
    )


def test_formats_single_currency_spending_with_resolved_entity_in_german() -> None:
    answer = format_rag_sql_answer(
        _execution([{"value": 220.69, "currency": "EUR"}]),
        _plan(
            result_shape="grouped_rows",
            result_entity="spending_amount",
            display_columns=["value", "currency"],
            instruction="Summiere die Ausgaben für Schuhe nach Währung.",
        ),
        language="de",
        question="Wie viel habe ich für Schuhe ausgegeben?",
        resolved_entities=[_shoes_entity()],
    )

    assert answer == "Du hast insgesamt 220,69 € für „Schuhe“ ausgegeben."


def test_formats_general_single_currency_spending_without_entity() -> None:
    answer = format_rag_sql_answer(
        _execution([{"value": 1234.5, "currency": "EUR"}]),
        _plan(
            result_shape="scalar",
            result_entity="spending_amount",
            display_columns=["value", "currency"],
            instruction="Report total spending.",
        ),
        language="de",
        question="Wie viel habe ich insgesamt ausgegeben?",
    )

    assert answer == "Du hast insgesamt 1.234,50 € ausgegeben."


def test_formats_multiple_currency_groups_without_dropping_rows() -> None:
    answer = format_rag_sql_answer(
        _execution(
            [
                {"value": 220.69, "currency": "EUR"},
                {"value": 45.0, "currency": "USD"},
            ]
        ),
        _plan(
            result_shape="grouped_rows",
            result_entity="spending_amount",
            display_columns=["value", "currency"],
            instruction="Report spending by currency.",
        ),
        language="de",
        question="Wie viel habe ich für Schuhe ausgegeben?",
        resolved_entities=[_shoes_entity()],
    )

    assert answer == ("Deine Ausgaben für „Schuhe“ betragen nach Währung: 220,69 €; 45,00 $.")


def test_formats_receipt_lookup_as_receipts_not_spending() -> None:
    answer = format_rag_sql_answer(
        _execution(
            [
                {
                    "receipt_id": 19,
                    "receipt_date": "2017-08-31",
                    "merchant": "Modepark Röther",
                    "grand_total": 86.2,
                    "currency": "EUR",
                }
            ]
        ),
        _plan(
            result_shape="rows",
            result_entity="receipt",
            display_columns=[
                "receipt_id",
                "receipt_date",
                "merchant",
                "grand_total",
                "currency",
            ],
            instruction="List the receipts containing Vittel.",
        ),
        language="de",
        question="Zeige mir die Quittung mit Vittel.",
        resolved_entities=[_vittel_entity()],
    )

    assert answer == (
        "1 passende Quittung mit „Vittel“ gefunden: "
        "2017-08-31 — Modepark Röther — Gesamtbetrag 86,20 € — Beleg-ID 19."
    )


def test_formats_non_monetary_scalar_with_german_number_separators() -> None:
    answer = format_rag_sql_answer(
        _execution([{"value": 1234}]),
        _plan(
            result_shape="scalar",
            result_entity="receipt_count",
            display_columns=["value"],
        ),
        language="de",
    )

    assert answer == "Ergebnis: 1.234."


def test_keeps_generic_summary_for_non_monetary_grouped_rows() -> None:
    answer = format_rag_sql_answer(
        _execution([{"merchant": "REWE"}, {"merchant": "LIDL"}]),
        _plan(
            result_shape="grouped_rows",
            result_entity="merchant",
            display_columns=["merchant"],
        ),
        language="de",
    )

    assert answer == "Die Abfrage lieferte 2 Gruppen."


def test_formats_english_spending_answer() -> None:
    answer = format_rag_sql_answer(
        _execution([{"value": 80.74, "currency": "EUR"}]),
        _plan(
            result_shape="scalar",
            result_entity="spending_amount",
            display_columns=["value", "currency"],
            instruction="Report spending.",
        ),
        language="en",
        question="How much did I spend on shoes?",
        resolved_entities=[_shoes_entity()],
    )

    assert answer == "You spent a total of €80.74 on “Schuhe”."


def test_describe_product_uses_reviewed_semantic_description() -> None:
    status, answer = format_rag_sql_outcome(
        _execution(
            [
                {
                    "item_id": 127,
                    "description": "VITTEL 1,5L",
                    "normalized_name": "Vittel",
                    "semantic_description": "Vittel is a brand of bottled mineral water",
                    "category": "beverages_water",
                    "category_reason": "The item is bottled water.",
                }
            ]
        ),
        _plan(
            result_shape="rows",
            result_entity="product_description",
            display_columns=[
                "item_id",
                "description",
                "normalized_name",
                "semantic_description",
                "category",
                "category_reason",
            ],
        ),
        language="en",
        requested_operation="describe_product",
        resolved_entities=[_vittel_entity()],
    )

    assert status == "completed"
    assert answer == "Vittel is a brand of bottled mineral water."


def test_identify_brand_never_uses_merchant_and_requires_explicit_metadata() -> None:
    status, answer = format_rag_sql_outcome(
        _execution(
            [
                {
                    "item_id": 127,
                    "description": "VITTEL 1,5L",
                    "normalized_name": "Vittel",
                    "semantic_description": None,
                    "category": "beverages_water",
                    "category_reason": "Mineral water sold by REWE.",
                    "merchant": "REWE",
                }
            ]
        ),
        _plan(
            result_shape="rows",
            result_entity="product_brand",
            display_columns=["item_id", "description", "semantic_description", "category_reason"],
        ),
        language="en",
        requested_operation="identify_brand",
        resolved_entities=[_vittel_entity()],
    )

    assert status == "insufficient_info"
    assert "enough product information" in answer


def test_empty_descriptive_query_is_not_found() -> None:
    status, _answer = format_rag_sql_outcome(
        _execution([]),
        _plan(
            result_shape="rows", result_entity="product_description", display_columns=["item_id"]
        ),
        language="en",
        requested_operation="describe_product",
    )
    assert status == "not_found"


def test_identify_brand_uses_explicit_category_reason_even_when_search_label_is_generic() -> None:
    status, answer = format_rag_sql_outcome(
        _execution(
            [
                {
                    "item_id": 127,
                    "description": "Vittel",
                    "normalized_name": "Vittel",
                    "semantic_description": None,
                    "category": "beverages",
                    "category_reason": "Vittel is a brand of mineral water.",
                }
            ]
        ),
        _plan(
            result_shape="rows",
            result_entity="product_brand",
            display_columns=[
                "item_id",
                "description",
                "normalized_name",
                "semantic_description",
                "category",
                "category_reason",
            ],
        ),
        language="en",
        requested_operation="identify_brand",
        resolved_entities=[
            ResolvedSemanticEntity(
                entity_id="e001",
                search_text="bottled water",
                status="resolved",
                selected_item_ids=[127],
            )
        ],
    )

    assert status == "completed"
    assert answer == "The brand named in the reviewed product data is “Vittel”."


def test_identify_brand_uses_only_explicit_reviewed_brand_statement() -> None:
    status, answer = format_rag_sql_outcome(
        _execution(
            [
                {
                    "item_id": 127,
                    "description": "VITTEL 1,5L",
                    "normalized_name": "Vittel",
                    "semantic_description": "Vittel is a bottled mineral water brand.",
                    "category": "beverages_water",
                    "category_reason": None,
                }
            ]
        ),
        _plan(
            result_shape="rows",
            result_entity="product_brand",
            display_columns=["item_id", "description", "semantic_description"],
        ),
        language="en",
        requested_operation="identify_brand",
        resolved_entities=[_vittel_entity()],
    )

    assert status == "completed"
    assert answer == "The brand named in the reviewed product data is “Vittel”."


def test_identify_brand_accepts_explicit_subject_inside_variant_product_label() -> None:
    rows = [
        {
            "item_id": item_id,
            "description": "*SENSEO CLASSIC 1",
            "normalized_name": "*SENSEO CLASSIC 1",
            "semantic_description": None,
            "category": "groceries_food",
            "category_reason": (
                "Senseo is a brand of coffee/coffee pods, which falls under general packaged food."
            ),
        }
        for item_id in (175, 176, 177, 178)
    ]
    status, answer = format_rag_sql_outcome(
        _execution(rows),
        _plan(
            result_shape="rows",
            result_entity="product_description",
            display_columns=[
                "item_id",
                "description",
                "normalized_name",
                "semantic_description",
                "category",
                "category_reason",
            ],
        ),
        language="en",
        requested_operation="identify_brand",
        resolved_entities=[
            ResolvedSemanticEntity(
                entity_id="e001",
                search_text="coffee",
                status="resolved",
                selected_item_ids=[175, 176, 177, 178],
            )
        ],
    )

    assert status == "completed"
    assert answer == "The brand named in the reviewed product data is “Senseo”."


def test_identify_brand_deduplicates_and_lists_multiple_reviewed_brands() -> None:
    status, answer = format_rag_sql_outcome(
        _execution(
            [
                {
                    "item_id": 1,
                    "description": "SENSEO CLASSIC",
                    "normalized_name": "Senseo Classic",
                    "semantic_description": None,
                    "category": "groceries_food",
                    "category_reason": "Senseo is a brand of coffee pods.",
                },
                {
                    "item_id": 2,
                    "description": "JACOBS KRÖNUNG",
                    "normalized_name": "Jacobs Krönung",
                    "semantic_description": None,
                    "category": "groceries_food",
                    "category_reason": "Jacobs is a brand of ground coffee.",
                },
            ]
        ),
        _plan(
            result_shape="rows",
            result_entity="product_description",
            display_columns=[
                "item_id",
                "description",
                "normalized_name",
                "semantic_description",
                "category",
                "category_reason",
            ],
        ),
        language="en",
        requested_operation="identify_brand",
    )

    assert status == "completed"
    assert answer == ("The brands named in the reviewed product data are “Senseo” and “Jacobs”.")


def test_identify_brand_rejects_nonmatching_merchant_subject() -> None:
    status, _answer = format_rag_sql_outcome(
        _execution(
            [
                {
                    "item_id": 1,
                    "description": "SENSEO CLASSIC",
                    "normalized_name": "Senseo Classic",
                    "semantic_description": None,
                    "category": "groceries_food",
                    "category_reason": "REWE is a brand associated with this purchase.",
                }
            ]
        ),
        _plan(
            result_shape="rows",
            result_entity="product_description",
            display_columns=[
                "item_id",
                "description",
                "normalized_name",
                "semantic_description",
                "category",
                "category_reason",
            ],
        ),
        language="en",
        requested_operation="identify_brand",
    )

    assert status == "insufficient_info"


def test_ambiguous_reviewed_brand_evidence_is_routed_to_hybrid_formatter() -> None:
    from receipt_intelligence.rag_sql.formatter import classify_rag_sql_outcome

    decision = classify_rag_sql_outcome(
        _execution(
            [
                {
                    "item_id": 175,
                    "description": "STARBUCKS NESPRESSO CAPSULES",
                    "normalized_name": "Starbucks Nespresso Capsules",
                    "semantic_description": None,
                    "category": "groceries_food",
                    "category_reason": (
                        "The packaging identifies Starbucks as the product brand and "
                        "Nespresso as the compatible system."
                    ),
                }
            ]
        ),
        _plan(
            result_shape="rows",
            result_entity="product_brand",
            display_columns=[
                "item_id",
                "description",
                "normalized_name",
                "semantic_description",
                "category",
                "category_reason",
            ],
        ),
        language="en",
        requested_operation="identify_brand",
    )

    assert decision.status == "ambiguous"
    assert decision.reason == "reviewed_evidence_requires_semantic_normalization"
    assert decision.supporting_item_ids == (175,)


def test_brand_question_without_explicit_reviewed_brand_evidence_skips_llm() -> None:
    from receipt_intelligence.rag_sql.formatter import classify_rag_sql_outcome

    decision = classify_rag_sql_outcome(
        _execution(
            [
                {
                    "item_id": 5,
                    "description": "CLASSIC COFFEE PADS",
                    "normalized_name": "Classic Coffee Pads",
                    "semantic_description": None,
                    "category": "groceries_food",
                    "category_reason": "Coffee pads compatible with Senseo systems.",
                }
            ]
        ),
        _plan(
            result_shape="rows",
            result_entity="product_brand",
            display_columns=["item_id", "description", "category_reason"],
        ),
        language="en",
        requested_operation="identify_brand",
    )

    assert decision.status == "no_evidence"
    assert decision.response_status == "insufficient_info"
