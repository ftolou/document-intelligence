from receipt_intelligence.web.query_response import (
    normalize_query_response,
    query_error_payload,
    user_message_for_error,
)


def test_normalize_query_response_preserves_rag_sql_contract() -> None:
    payload = normalize_query_response(
        {
            "strategy": "rag_sql",
            "question": "How much?",
            "status": "completed",
            "answer": "10.00 EUR",
            "execution": {"engine": "rag_sql", "orchestrator": "langgraph"},
        }
    )

    assert payload["strategy"] == "rag_sql"
    assert payload["status"] == "completed"
    assert payload["answer"] == "10.00 EUR"
    assert payload["execution"]["orchestrator"] == "langgraph"


def test_normalize_query_response_fills_optional_fields() -> None:
    payload = normalize_query_response({"status": "completed", "answer": "Done"})

    assert payload == {
        "strategy": "rag_sql",
        "status": "completed",
        "answer": "Done",
        "data": None,
        "clarification_question": None,
        "error_code": None,
        "error": None,
    }


def test_query_error_payload_is_single_engine_contract() -> None:
    payload = query_error_payload(
        error_code="sql_validation_failed",
        error="unsafe SQL",
    )

    assert payload == {
        "strategy": "rag_sql",
        "status": "error",
        "answer": "The generated query did not pass the safety checks.",
        "data": None,
        "clarification_question": None,
        "error_code": "sql_validation_failed",
        "error": "unsafe SQL",
    }


def test_unknown_error_uses_generic_message() -> None:
    assert user_message_for_error("unknown") == "The receipt query could not be completed."
