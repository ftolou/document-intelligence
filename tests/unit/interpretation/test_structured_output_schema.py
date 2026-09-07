from __future__ import annotations

from typing import Any

from jsonschema.validators import validator_for

from receipt_intelligence.adapters.llm.openai_responses import _strict_transport_schema
from receipt_intelligence.interpretation import LiteralType, LiteralValue, NormalizationStatus
from receipt_intelligence.interpretation.workflow import _GeneratedInterpretation


def _validation_errors(schema: dict[str, Any], value: dict[str, Any]) -> list[Any]:
    validator_class = validator_for(schema)
    validator_class.check_schema(schema)
    return list(validator_class(schema).iter_errors(value))


def _assert_static_objects_are_explicit(schema: Any) -> None:
    if isinstance(schema, list):
        for item in schema:
            _assert_static_objects_are_explicit(item)
        return
    if not isinstance(schema, dict):
        return

    properties = schema.get("properties")
    if isinstance(properties, dict):
        assert schema.get("additionalProperties") is False
        assert set(schema.get("required", [])) == set(properties)
    for value in schema.values():
        _assert_static_objects_are_explicit(value)


def test_literal_schema_exposes_type_specific_normalization_and_metadata() -> None:
    schema = LiteralValue.model_json_schema()
    amount = LiteralValue(
        literal_type=LiteralType.AMOUNT,
        observed="12,50",
        normalization_status=NormalizationStatus.NORMALIZED,
        normalized="12.50",
        currency="EUR",
    ).model_dump(mode="json")

    assert _validation_errors(schema, amount) == []

    numeric_amount = dict(amount, normalized=12.5)
    assert _validation_errors(schema, numeric_amount)

    number_with_unit = LiteralValue(
        literal_type=LiteralType.NUMBER,
        observed="12",
        normalization_status=NormalizationStatus.NORMALIZED,
        normalized=12,
    ).model_dump(mode="json")
    number_with_unit["unit"] = "kg"
    assert _validation_errors(schema, number_with_unit)


def test_generated_interpretation_schema_is_strict_transport_compatible() -> None:
    schema = _GeneratedInterpretation.model_json_schema()

    _assert_static_objects_are_explicit(schema)
    assert "oneOf" not in str(schema)
    assert "discriminator" not in str(schema)

    transport = _strict_transport_schema(schema)
    _assert_static_objects_are_explicit(transport)
