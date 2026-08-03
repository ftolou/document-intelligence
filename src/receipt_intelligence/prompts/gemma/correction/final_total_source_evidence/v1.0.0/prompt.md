# FINAL PURCHASE TOTAL RECOVERY — V2.4

You receive a receipt transcription with stable row identifiers.

Your task is source selection, not receipt reconciliation.
Find the one explicitly printed final amount the customer must pay for the whole purchase.
Return exactly one JSON object matching OUTPUT_SCHEMA.

## DECISION ORDER

1. Search only for an explicit final-purchase-total label and its associated printed amount.
2. If one unambiguous explicit final-purchase-total label and amount are present, return them immediately.
3. Do not reject, reinterpret, or replace that amount because another amount appears later or because other printed values seem inconsistent.
4. A currency name, currency code, or currency symbol only identifies the currency. It does not identify an amount as the final purchase total.
5. Amounts identified as subtotal, tax, net amount, discount, savings, payment, tender, cash received, card payment, or change are not the final purchase total.
6. An unlabeled or currency-only amount associated with payment or change is not the final purchase total.
7. Do not add, subtract, compare, reconcile, verify, or infer amounts. Do not use item sums, discounts, VAT, payment minus change, or any other arithmetic.
8. Return `unresolved` only when no explicit final-purchase-total label exists, or when multiple explicit final-purchase-total labels contain conflicting amounts and the source does not distinguish which one is final.
9. If `Summe 60,47` is followed by a currency-only payment line such as `Euro 60,50` and `Rückgeld 0,03`, select the explicitly labelled `Summe 60,47`; the later currency-only amount is tendered payment, not the purchase total.

## OUTPUT RULES

For `resolved`:

- `label_row` is the row containing the authoritative final-purchase-total label;
- `source_row` is the row containing its associated printed amount;
- `label_text` is only the literal label phrase copied from `label_row` and does not include the amount;
- `value_text` is only the literal printed amount copied from `source_row`;
- the label and amount may be in the same row.

For `unresolved`, set `label_row`, `source_row`, `label_text`, and `value_text` to null.

Use only literal source text and source row identifiers.
Do not normalize the amount. Do not propose a patch. Do not explain the decision.
Do not return Markdown fences or text outside the JSON object.

## SOURCE_EVIDENCE

$source_evidence

## OUTPUT_SCHEMA

$output_schema
