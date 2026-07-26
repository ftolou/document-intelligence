"""Deterministic answer extraction for RAG-SQL results.

Clear result shapes are formatted without another model call. Descriptive
questions additionally expose a structured classification so LangGraph can
route evidence-rich but ambiguous rows to the bounded answer formatter.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from receipt_intelligence.rag_sql.models import (
    ResolvedQueryFilter,
    ResolvedSemanticEntity,
    SqlExecutionResult,
    ValidatedSqlPlan,
)

_DESCRIPTIVE_OPERATIONS = frozenset({"describe_product", "identify_brand", "identify_product_type"})
APPROVED_EVIDENCE_FIELDS = frozenset(
    {
        "item_id",
        "description",
        "normalized_name",
        "semantic_description",
        "category",
        "category_reason",
        "reviewed_brand",
    }
)


@dataclass(frozen=True, slots=True)
class DeterministicAnswerDecision:
    """Evidence classification used by the LangGraph formatting router."""

    status: Literal["resolved", "ambiguous", "no_evidence", "not_found"]
    response_status: Literal["completed", "not_found", "insufficient_info"]
    answer: str
    reason: str
    evidence_rows: tuple[dict[str, Any], ...] = ()
    supporting_item_ids: tuple[int, ...] = ()

    def diagnostics(self) -> dict[str, object]:
        return {
            "status": self.status,
            "response_status": self.response_status,
            "reason": self.reason,
            "evidence_row_count": len(self.evidence_rows),
            "supporting_item_ids": list(self.supporting_item_ids),
        }


def classify_rag_sql_outcome(
    execution: SqlExecutionResult,
    plan: ValidatedSqlPlan,
    *,
    language: str,
    question: str | None = None,
    requested_operation: str | None = None,
    resolved_entities: Sequence[ResolvedQueryFilter | ResolvedSemanticEntity] = (),
) -> DeterministicAnswerDecision:
    """Classify whether the result is resolved, ambiguous, or lacks evidence."""

    german = language == "de"
    if not execution.rows:
        return DeterministicAnswerDecision(
            status="not_found",
            response_status="not_found",
            answer=(
                "Keine passenden geprüften Belegdaten gefunden."
                if german
                else "No matching reviewed receipt data was found."
            ),
            reason="no_rows",
        )

    operation = str(requested_operation or "").casefold()
    if operation in _DESCRIPTIVE_OPERATIONS:
        answer = _format_descriptive_answer(
            execution.rows,
            operation=operation,
            german=german,
            entity_labels=_resolved_entity_labels(resolved_entities),
        )
        evidence_rows = tuple(_sanitize_evidence_rows(execution.rows))
        item_ids = tuple(_supporting_item_ids(evidence_rows))
        if answer is not None:
            return DeterministicAnswerDecision(
                status="resolved",
                response_status="completed",
                answer=answer,
                reason="deterministic_evidence_match",
                evidence_rows=evidence_rows,
                supporting_item_ids=item_ids,
            )
        insufficient = (
            "Die geprüften Belegdaten enthalten dafür nicht genügend Produktinformationen."
            if german
            else "The reviewed receipt data does not contain enough product information for that answer."
        )
        if _has_operation_evidence(evidence_rows, operation=operation):
            return DeterministicAnswerDecision(
                status="ambiguous",
                response_status="insufficient_info",
                answer=insufficient,
                reason="reviewed_evidence_requires_semantic_normalization",
                evidence_rows=evidence_rows,
                supporting_item_ids=item_ids,
            )
        return DeterministicAnswerDecision(
            status="no_evidence",
            response_status="insufficient_info",
            answer=insufficient,
            reason="reviewed_evidence_absent",
            evidence_rows=evidence_rows,
            supporting_item_ids=item_ids,
        )

    return DeterministicAnswerDecision(
        status="resolved",
        response_status="completed",
        answer=format_rag_sql_answer(
            execution,
            plan,
            language=language,
            question=question,
            resolved_entities=resolved_entities,
        ),
        reason="deterministic_result_shape",
        supporting_item_ids=tuple(_supporting_item_ids(execution.rows)),
    )


def format_rag_sql_outcome(
    execution: SqlExecutionResult,
    plan: ValidatedSqlPlan,
    *,
    language: str,
    question: str | None = None,
    requested_operation: str | None = None,
    resolved_entities: Sequence[ResolvedQueryFilter | ResolvedSemanticEntity] = (),
) -> tuple[Literal["completed", "not_found", "insufficient_info"], str]:
    """Return the deterministic terminal result without invoking the LLM fallback."""

    decision = classify_rag_sql_outcome(
        execution,
        plan,
        language=language,
        question=question,
        requested_operation=requested_operation,
        resolved_entities=resolved_entities,
    )
    return decision.response_status, decision.answer


def _format_descriptive_answer(
    rows: Sequence[dict[str, Any]],
    *,
    operation: str,
    german: bool,
    entity_labels: Sequence[str],
) -> str | None:
    # Product semantics must be anchored to reviewed SQL rows. The resolved
    # entity label can be a generic search phrase and is not itself proof of a
    # product name, product type, or brand.
    label = _first_text(rows, "normalized_name", "description")
    if not label and entity_labels:
        label = entity_labels[0]
    semantic = _first_text(rows, "semantic_description")
    reason = _first_text(rows, "category_reason")
    category = _useful_category(_first_text(rows, "category"))

    if operation == "describe_product":
        evidence = semantic or reason
        if evidence:
            return evidence if evidence.endswith((".", "!", "?")) else f"{evidence}."
        if category and label:
            return (
                f"„{label}“ ist als {_humanize_category(category)} kategorisiert."
                if german
                else f"“{label}” is categorized as {_humanize_category(category)}."
            )
        return None

    if operation == "identify_product_type":
        if category and label:
            return (
                f"„{label}“ ist als Produkttyp „{_humanize_category(category)}“ erfasst."
                if german
                else f"“{label}” is recorded as product type “{_humanize_category(category)}”."
            )
        evidence = semantic or reason
        if evidence:
            return evidence if evidence.endswith((".", "!", "?")) else f"{evidence}."
        return None

    if operation == "identify_brand":
        brands = _collect_explicit_brands(rows)
        if not brands:
            return None
        return format_descriptive_values(brands, operation=operation, german=german)
    return None


def format_descriptive_values(
    values: Sequence[str],
    *,
    operation: str,
    german: bool,
) -> str:
    """Render validated structured values without free-form LLM prose."""

    unique: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = " ".join(str(raw or "").split()).strip()
        if not value or value.casefold() in seen:
            continue
        seen.add(value.casefold())
        unique.append(value)
    if not unique:
        raise ValueError("At least one validated value is required.")

    formatted = _format_quoted_values(unique, german=german)
    if operation == "identify_brand":
        if len(unique) == 1:
            return (
                f"Die in den geprüften Produktdaten genannte Marke ist {formatted}."
                if german
                else f"The brand named in the reviewed product data is {formatted}."
            )
        return (
            f"Die in den geprüften Produktdaten genannten Marken sind {formatted}."
            if german
            else f"The brands named in the reviewed product data are {formatted}."
        )
    if operation == "identify_product_type":
        if len(unique) == 1:
            return (
                f"Der geprüfte Produkttyp ist {formatted}."
                if german
                else f"The reviewed product type is {formatted}."
            )
        return (
            f"Die geprüften Produkttypen sind {formatted}."
            if german
            else f"The reviewed product types are {formatted}."
        )
    return (
        f"Die geprüften Produktdaten beschreiben: {formatted}."
        if german
        else f"The reviewed product data describes: {formatted}."
    )


def _collect_explicit_brands(rows: Sequence[dict[str, Any]]) -> list[str]:
    brands: list[str] = []
    seen: set[str] = set()
    for row in rows:
        reviewed_brand = _row_text(row, "reviewed_brand")
        if reviewed_brand:
            key = reviewed_brand.casefold()
            if key not in seen:
                seen.add(key)
                brands.append(reviewed_brand)
            continue
        label = _row_text(row, "normalized_name", "description")
        evidence = " ".join(
            value
            for value in (
                _row_text(row, "semantic_description"),
                _row_text(row, "category_reason"),
            )
            if value
        ).strip()
        if not evidence or not re.search(r"\b(?:brand|marke)\b", evidence, re.IGNORECASE):
            continue
        explicit = _extract_explicit_brand(evidence, label)
        if not explicit:
            continue
        key = explicit.casefold()
        if key in seen:
            continue
        seen.add(key)
        brands.append(explicit)
    return brands


def _extract_explicit_brand(evidence: str, label: str) -> str | None:
    direct = re.search(
        r"\b(?:brand|marke)\s*(?:is|ist|:|=)\s*[\"'„“]?"
        r"([A-Za-zÀ-ÖØ-öø-ÿ0-9][A-Za-zÀ-ÖØ-öø-ÿ0-9&.' -]{0,60}?)"
        r"(?=\s*(?:[.,;!?]|$))",
        evidence,
        re.IGNORECASE,
    )
    if direct:
        value = direct.group(1).strip(" .,:;!?\"'„“")
        if value and (not label or _brand_subject_matches_label(value, label)):
            return value
        return None

    subject_statement = re.search(
        r"^\s*([A-Za-zÀ-ÖØ-öø-ÿ0-9][A-Za-zÀ-ÖØ-öø-ÿ0-9&.' -]{0,60}?)"
        r"\s+(?:is|ist)\s+(.{0,80}?)\b(?:brand|marke)\b",
        evidence,
        re.IGNORECASE,
    )
    if not subject_statement:
        return None
    predicate = subject_statement.group(2).casefold()
    if any(
        forbidden in predicate.split()
        for forbidden in ("from", "by", "of", "sold", "seller", "merchant", "von", "der")
    ):
        return None
    subject = subject_statement.group(1).strip(" .,:;!?\"'„“")
    if label and not _brand_subject_matches_label(subject, label):
        return None
    return subject or None


def _brand_subject_matches_label(subject: str, label: str) -> bool:
    subject_tokens = _semantic_tokens(subject)
    label_tokens = _semantic_tokens(label)
    if not subject_tokens or not label_tokens:
        return False
    if subject_tokens == label_tokens:
        return True
    width = len(subject_tokens)
    return any(
        label_tokens[index : index + width] == subject_tokens
        for index in range(0, len(label_tokens) - width + 1)
    )


def _semantic_tokens(value: str) -> list[str]:
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", value.casefold())


def _row_text(row: dict[str, Any], *columns: str) -> str:
    for column in columns:
        value = " ".join(str(row.get(column) or "").split()).strip()
        if value:
            return value
    return ""


def _sanitize_evidence_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for row in rows[:100]:
        compact: dict[str, Any] = {}
        for field in APPROVED_EVIDENCE_FIELDS:
            if field not in row:
                continue
            value = row.get(field)
            if field == "item_id":
                if isinstance(value, int) and value > 0:
                    compact[field] = value
                continue
            text = " ".join(str(value or "").split()).strip()
            if text:
                compact[field] = text[:2000]
        if compact:
            sanitized.append(compact)
    return sanitized


def _supporting_item_ids(rows: Sequence[dict[str, Any]]) -> list[int]:
    return sorted(
        {
            int(row["item_id"])
            for row in rows
            if isinstance(row.get("item_id"), int) and int(row["item_id"]) > 0
        }
    )


def _has_operation_evidence(rows: Sequence[dict[str, Any]], *, operation: str) -> bool:
    if operation == "identify_brand":
        for row in rows:
            if _row_text(row, "reviewed_brand"):
                return True
            semantic = " ".join(
                value
                for value in (
                    _row_text(row, "semantic_description"),
                    _row_text(row, "category_reason"),
                )
                if value
            )
            if re.search(r"\b(?:brand|marke|branding|hersteller)\b", semantic, re.IGNORECASE):
                return True
        return False
    if operation == "identify_product_type":
        return any(
            _useful_category(_row_text(row, "category"))
            or _row_text(row, "semantic_description", "category_reason")
            for row in rows
        )
    if operation == "describe_product":
        return any(
            _row_text(row, "semantic_description", "category_reason")
            or _useful_category(_row_text(row, "category"))
            for row in rows
        )
    return False


def _format_quoted_values(values: Sequence[str], *, german: bool) -> str:
    opening, closing = ("„", "“") if german else ("“", "”")
    quoted = [f"{opening}{value}{closing}" for value in values]
    if len(quoted) == 1:
        return quoted[0]
    conjunction = " und " if german else " and "
    if len(quoted) == 2:
        return conjunction.join(quoted)
    return ", ".join(quoted[:-1]) + conjunction + quoted[-1]


def _useful_category(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized in {"", "unknown", "item", "product", "purchase_item", "purchased_product"}:
        return ""
    return value


def _first_text(rows: Sequence[dict[str, Any]], *columns: str) -> str:
    seen: set[str] = set()
    for row in rows:
        for column in columns:
            value = " ".join(str(row.get(column) or "").split()).strip()
            if value and value.casefold() not in seen:
                return value
            if value:
                seen.add(value.casefold())
    return ""


def _humanize_category(value: str) -> str:
    return " ".join(value.replace("/", " ").replace("_", " ").split())


_CURRENCY_SYMBOLS = {
    "EUR": "€",
    "USD": "$",
    "GBP": "£",
    "JPY": "¥",
}


def format_rag_sql_answer(
    execution: SqlExecutionResult,
    plan: ValidatedSqlPlan,
    *,
    language: str,
    question: str | None = None,
    resolved_entities: Sequence[ResolvedQueryFilter | ResolvedSemanticEntity] = (),
) -> str:
    """Render a concise answer from a validated SQL result.

    Monetary queries use the planner's stable ``value`` and ``currency``
    aliases. Resolved semantic labels are used only for presentation; item IDs
    and calculations remain entirely determined by the validated SQL plan.
    """

    german = language == "de"
    if not execution.rows:
        return "Die Abfrage lieferte kein Ergebnis." if german else "The query returned no result."

    entity_labels = _resolved_entity_labels(resolved_entities)

    if plan.result_entity == "receipt" and _looks_like_receipt_rows(execution.rows):
        return _format_receipt_rows(
            execution.rows,
            german=german,
            entity_labels=entity_labels,
            truncated=execution.truncated,
        )

    monetary_rows = _monetary_rows(execution.rows)
    spending_intent = _is_spending_intent(question, plan.answer_instruction)

    if monetary_rows:
        return _format_monetary_answer(
            monetary_rows,
            german=german,
            spending_intent=spending_intent,
            entity_labels=entity_labels,
            truncated=execution.truncated,
        )

    if plan.result_shape == "scalar":
        row = execution.rows[0]
        value = row.get("value")
        if value is None and execution.columns:
            value = row.get(execution.columns[0])
        formatted = _format_scalar(value, german=german)
        if entity_labels:
            label_text = _format_entity_labels(entity_labels, german=german)
            return (
                f"Ergebnis für {label_text}: {formatted}."
                if german
                else f"Result for {label_text}: {formatted}."
            )
        return f"Ergebnis: {formatted}." if german else f"Result: {formatted}."

    if plan.result_shape == "row":
        return "Die Abfrage lieferte eine Zeile." if german else "The query returned one row."

    noun = "Gruppen" if plan.result_shape == "grouped_rows" else "Zeilen"
    if german:
        suffix = " (gekürzt)" if execution.truncated else ""
        return f"Die Abfrage lieferte {execution.row_count} {noun}{suffix}."
    noun_en = "groups" if plan.result_shape == "grouped_rows" else "rows"
    suffix_en = " (truncated)" if execution.truncated else ""
    return f"The query returned {execution.row_count} {noun_en}{suffix_en}."


def _format_monetary_answer(
    rows: Sequence[tuple[Any, str]],
    *,
    german: bool,
    spending_intent: bool,
    entity_labels: Sequence[str],
    truncated: bool,
) -> str:
    amounts = [_format_money(value, currency, german=german) for value, currency in rows]
    label_text = _format_entity_labels(entity_labels, german=german) if entity_labels else ""

    if len(amounts) == 1:
        amount = amounts[0]
        if spending_intent:
            if label_text:
                return (
                    f"Du hast insgesamt {amount} für {label_text} ausgegeben."
                    if german
                    else f"You spent a total of {amount} on {label_text}."
                )
            return (
                f"Du hast insgesamt {amount} ausgegeben."
                if german
                else f"You spent a total of {amount}."
            )
        if label_text:
            return (
                f"Ergebnis für {label_text}: {amount}."
                if german
                else f"Result for {label_text}: {amount}."
            )
        return f"Ergebnis: {amount}." if german else f"Result: {amount}."

    amount_list = "; ".join(amounts)
    truncated_suffix = " (gekürzt)" if german and truncated else " (truncated)" if truncated else ""
    if spending_intent:
        if label_text:
            return (
                (
                    f"Deine Ausgaben für {label_text} betragen nach Währung: "
                    f"{amount_list}{truncated_suffix}."
                )
                if german
                else (
                    f"Your spending on {label_text} by currency is: "
                    f"{amount_list}{truncated_suffix}."
                )
            )
        return (
            f"Deine Ausgaben betragen nach Währung: {amount_list}{truncated_suffix}."
            if german
            else f"Your spending by currency is: {amount_list}{truncated_suffix}."
        )
    return (
        f"Ergebnisse nach Währung: {amount_list}{truncated_suffix}."
        if german
        else f"Results by currency: {amount_list}{truncated_suffix}."
    )


def _looks_like_receipt_rows(rows: Sequence[dict[str, Any]]) -> bool:
    return bool(rows) and all("receipt_id" in row for row in rows)


def _format_receipt_rows(
    rows: Sequence[dict[str, Any]],
    *,
    german: bool,
    entity_labels: Sequence[str],
    truncated: bool,
) -> str:
    count = len(rows)
    label_text = _format_entity_labels(entity_labels, german=german) if entity_labels else ""
    if german:
        noun = "passende Quittung" if count == 1 else "passende Quittungen"
        context = f" mit {label_text}" if label_text else ""
        prefix = f"{count} {noun}{context} gefunden"
    else:
        noun = "matching receipt" if count == 1 else "matching receipts"
        context = f" containing {label_text}" if label_text else ""
        prefix = f"Found {count} {noun}{context}"

    details = [_format_receipt_row(row, german=german) for row in rows[:5]]
    details = [detail for detail in details if detail]
    suffix = " (gekürzt)" if german and truncated else " (truncated)" if truncated else ""
    if not details:
        return f"{prefix}{suffix}."
    return f"{prefix}{suffix}: {'; '.join(details)}."


def _format_receipt_row(row: dict[str, Any], *, german: bool) -> str:
    parts: list[str] = []
    date_value = str(row.get("receipt_date") or "").strip()
    merchant = str(row.get("merchant") or row.get("merchant_name") or "").strip()
    if date_value:
        parts.append(date_value)
    if merchant:
        parts.append(merchant)

    total = row.get("grand_total")
    if total is None:
        total = row.get("paid_total")
    currency = str(row.get("currency") or "").strip().upper()
    if total is not None and currency and _to_decimal(total) is not None:
        amount = _format_money(total, currency, german=german)
        parts.append(f"Gesamtbetrag {amount}" if german else f"total {amount}")

    receipt_id = row.get("receipt_id")
    if receipt_id is not None:
        parts.append(f"Beleg-ID {receipt_id}" if german else f"receipt ID {receipt_id}")
    return " — ".join(parts)


def _monetary_rows(rows: Sequence[dict[str, Any]]) -> list[tuple[Any, str]]:
    monetary: list[tuple[Any, str]] = []
    for row in rows:
        if "value" not in row or "currency" not in row:
            return []
        value = row.get("value")
        currency = str(row.get("currency") or "").strip().upper()
        if value is None or not currency:
            return []
        if _to_decimal(value) is None:
            return []
        monetary.append((value, currency))
    return monetary


def _resolved_entity_labels(
    entities: Sequence[ResolvedQueryFilter | ResolvedSemanticEntity],
) -> list[str]:
    labels: list[str] = []
    for entity in entities:
        if entity.status != "resolved":
            continue
        if isinstance(entity, ResolvedQueryFilter):
            if entity.field not in {"product", "merchant", "category"}:
                continue
            raw = entity.original_value
            raw_values = raw if isinstance(raw, list) else [raw]
            candidates = [str(value) for value in raw_values]
        else:
            candidates = [entity.search_text]
        for candidate in candidates:
            label = " ".join(candidate.split()).strip()
            if label and label.casefold() not in {existing.casefold() for existing in labels}:
                labels.append(label)
    return labels


def _format_entity_labels(labels: Sequence[str], *, german: bool) -> str:
    quoted = [f"„{label}“" if german else f"“{label}”" for label in labels]
    if len(quoted) <= 1:
        return quoted[0] if quoted else ""
    conjunction = " und " if german else " and "
    if len(quoted) == 2:
        return conjunction.join(quoted)
    separator = ", "
    return separator.join(quoted[:-1]) + conjunction + quoted[-1]


def _is_spending_intent(question: str | None, instruction: str | None) -> bool:
    text = " ".join(part for part in (question, instruction) if part).casefold()
    markers = (
        "ausgegeben",
        "ausgabe",
        "ausgaben",
        "bezahlt",
        "spending",
        "spent",
        "how much did i spend",
        "total spent",
    )
    return any(marker in text for marker in markers)


def _format_money(value: Any, currency: str, *, german: bool) -> str:
    number = _format_decimal(value, german=german, decimals=2)
    symbol = _CURRENCY_SYMBOLS.get(currency)
    if german:
        return f"{number} {symbol or currency}"
    if symbol:
        return f"{symbol}{number}"
    return f"{number} {currency}"


def _format_scalar(value: Any, *, german: bool) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return ("wahr" if value else "falsch") if german else ("true" if value else "false")
    if isinstance(value, int):
        return _format_decimal(value, german=german, decimals=0)
    if isinstance(value, float | Decimal):
        return _format_decimal(value, german=german, decimals=2)
    return str(value)


def _format_decimal(value: Any, *, german: bool, decimals: int) -> str:
    decimal_value = _to_decimal(value)
    if decimal_value is None:
        return str(value)
    rendered = f"{decimal_value:,.{decimals}f}"
    if german:
        rendered = rendered.replace(",", "\u0000").replace(".", ",").replace("\u0000", ".")
    return rendered


def _to_decimal(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


__all__ = ["format_rag_sql_answer"]
