# Phase 7.3 — Review and SQL compatibility

Phase 7.3 keeps the next-pipeline receipt JSON canonical while adapting it at the
legacy review and relational-storage boundaries.

## Canonical next fields

- `receipt_metadata.date`, `receipt_metadata.time`, `receipt_metadata.currency`
- `totals.net_amount.net_amount`
- `tax.vat_amount.vat_amount`
- `totals.final_purchase_total.final_purchase_total`
- `payment.payment_received.payment_received`
- `payment.payment_method`
- `items[].name`, `items[].final_price`
- validation `checks[]`

## Compatibility projection

`receipt_intelligence.receipt_compat` is the only compatibility authority. It:

1. exposes legacy review aliases without mutating the canonical receipt;
2. applies review edits back to canonical next fields;
3. projects next validation checks into review issues;
4. supplies normalized values to fingerprints and SQLite repositories;
5. retains support for legacy receipts.

## Relational projection

No database migration is required for the immediate integration. Existing columns
are populated as follows:

| Canonical field | Relational column |
|---|---|
| `receipt_metadata.date` | `receipts.receipt_date` |
| `receipt_metadata.time` | `receipts.receipt_time` |
| `receipt_metadata.currency` | `receipts.currency` |
| `totals.net_amount.net_amount` | `receipts.subtotal` |
| `tax.vat_amount.vat_amount` | `receipts.tax_total` |
| `totals.final_purchase_total.final_purchase_total` | `receipts.grand_total` |
| `payment.payment_received.payment_received` | `receipts.paid_total` |
| `payment.payment_method` | `receipts.payment_method` |
| `items[].final_price` | `receipt_items.line_total` |

The full canonical receipt and item objects remain stored in `raw_json`.

## Existing records

Review-queue summaries are recalculated from `raw_json` when read, so existing
next-pipeline queue rows immediately show the correct total, item count, and failed
validation codes. Receipts that were already imported with null relational amounts
must be re-imported or resaved after applying this patch.
