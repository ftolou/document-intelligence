# Human Review

The human-review feature turns AI extraction into a controlled business workflow. Review is deliberately available in exactly one place: the dedicated **Review** tab. The Run, Batch, and Receipt data tabs may link into that workspace, but they do not expose a second editor.

## Unified UI behavior

The Review tab is a queue-based workbench:

- queue summary and filters for pending, duplicate, rejected, and approved records;
- searchable receipt cards in a persistent left sidebar;
- original receipt image and deterministic validation issues beside the editor;
- grouped merchant, transaction, totals, and item controls;
- compact item cards with advanced semantic/category fields behind expandable details;
- sticky actions for **Save draft**, **Reject**, **Approve**, and **Approve & next**.

After a receipt job reaches `done`, the Run tab only confirms that the result entered the queue and provides **Open in Review**. All corrections and review decisions are made in the Review tab.

## One source of truth

The existing SQLite database is used instead of a separate unreviewed-receipt database:

- `review_queue.extraction_json` is the immutable extraction snapshot captured when a receipt first enters the queue;
- `review_queue.draft_json` is the canonical mutable review draft;
- `review_queue.raw_json` remains a compatibility mirror of the current draft for code and databases created before the split;
- `receipts` and `receipt_items` are the canonical approved records used by analytics, RAG, and RAG-SQL;
- `receipt_review_history` stores an immutable snapshot for every saved review revision;
- `approved_receipt.json` and `human_review_record.json` are compatibility/audit mirrors, not authoritative application state.

Each queue record carries an optimistic `review_revision`. A stale browser save is rejected and the reviewer must reload, preventing one review session from silently overwriting another.

The state transition is deliberately explicit:

```text
immutable extraction_json
        |
        v
mutable draft_json  -- review revisions --> receipt_review_history
        |
        v
approved receipts + receipt_items
```

For queue rows created before this schema split, migration 11 initializes both new columns from the previous `raw_json` value because the original extraction snapshot cannot always be reconstructed from database state alone. New queue records preserve `extraction_json` on subsequent upserts and review saves.

## Editable receipt header fields

- merchant name
- merchant address
- date
- time
- currency
- subtotal
- tax total
- grand total
- paid total
- change
- current validation decision (read-only; recalculated on save)

## Editable item fields

Each extracted item row can be corrected before database import:

- raw item / description
- normalized item name
- category, for example `personal_care/shampoo`
- quantity
- unit
- unit price
- line total
- VAT rate
- confidence
- review status

This is important because semantic search questions such as “Which receipt had the shampoo I bought at dm?” depend on reliable item names and categories.

## API

Read current review context:

```http
GET /api/review/<job_id>
```

The response includes the structured receipt, prior review record if available, artifacts, and `receipt_image` when available.

Save review:

```http
POST /api/review/<job_id>
Content-Type: application/json
```

Example payload:

```json
{
  "fields": {
    "merchant_name": "dm-drogerie markt",
    "date": "2026-07-07",
    "grand_total": 23.45
  },
  "items": [
    {
      "index": 0,
      "description": "HEAD&SHOULDERS CLASSIC",
      "normalized_name": "Head & Shoulders Classic Shampoo",
      "category": "personal_care/shampoo",
      "quantity": 1,
      "unit_price": 3.95,
      "line_total": 3.95,
      "review_status": "corrected"
    }
  ],
  "review": {
    "reviewer": "FT",
    "status": "approved",
    "notes": "Corrected item category and approved total."
  }
}
```

## Database-authoritative review of approved receipts

Already approved receipts are persistent database records, not temporary processing jobs.
The **Receipt data** tab opens them through:

```http
GET /api/receipt-db/receipts/<receipt_id>/review
```

The response is reconstructed from the authoritative `receipts` and `receipt_items` rows
and remains editable even when the original job manifest, approved JSON, final JSON, or
receipt image has been removed. The durable save endpoint is:

```http
PUT /api/receipt-db/receipts/<receipt_id>/review
Content-Type: application/json
```

The request body uses the same `fields`, `items`, and `review` structure as the job review
endpoint. Saving performs one SQLite transaction that updates the receipt header, item
rows, raw JSON snapshot, review queue link, and lexical/FTS index. Existing item database
IDs are preserved.

### Selective semantic reindexing

The semantic vector document contains product identity and reviewed product semantics. Therefore:

- product description, normalized product name, parser item type, reviewed category path,
  category reason, or semantic description changed: invalidate the embedding for that
  item ID and selectively re-embed it;
- merchant, date, quantity, price, VAT, totals, or review metadata changed: update SQLite
  and lexical metadata without generating a new vector;
- embedding provider failure: keep the committed database edit, leave the stale vector
  removed, and report the item as pending/failed so the incremental indexer can retry.

SQLite is authoritative. When an existing `approved_receipt.json` still exists inside the
linked job directory, it is updated only as a best-effort mirror after the database commit.
A missing or unwritable artifact never blocks the database save. A missing image only
removes visual evidence; it does not make the receipt read-only.

The legacy `GET/POST /api/review/<job_id>` endpoints remain for pending job-based review.
When a job already has a linked database receipt, they delegate to the same durable
receipt-ID editor.

## Audit mirrors

Saving a review updates these best-effort compatibility artifacts:

```text
approved_receipt.json
human_review_record.json
```

`approved_receipt.json` contains the corrected receipt plus a `human_review` block.

`human_review_record.json` contains:

- job id
- source receipt artifact
- approved receipt artifact
- review status
- reviewer
- notes
- reviewed timestamp
- changed fields
- submitted field values
- submitted item corrections
- receipt database import result

## Post-review validation, state transition, and database import

Saving a review no longer preserves the extraction-time validation state. The application:

1. applies the human corrections;
2. reloads the job's OCR context when available;
3. reruns deterministic receipt validation;
4. replaces the stale `receipt.validation` block;
5. derives the effective review and queue state;
6. imports only an effectively approved receipt;
7. creates semantic embeddings after the database transaction commits.

Human approval can resolve non-blocking warnings. Missing core data such as merchant, total, or priced items, and any remaining high/critical issue, blocks approval and leaves the effective state at `needs_review`. The submitted corrections are still saved so the reviewer can continue editing.

The validation report preserves both decisions:

- `deterministic_import_decision`: result of the post-edit validator;
- `import_decision`: effective decision after the human-review policy;
- `pre_review_import_decision`: stale extraction-time decision retained only for audit.

When a receipt is approved for the first time, every persisted purchase-item ID is submitted to the incremental embedding indexer. The indexer embeds only eligible purchase rows and skips item rows marked `rejected` or `needs_review`. On later edits, only semantic product changes are re-embedded. An embedding-provider failure never rolls back the approved receipt; the API returns `pending` or `failed` so indexing can be retried.

## Why this matters

This demonstrates that the system does not blindly trust the LLM. It creates a controlled handover from AI extraction to human approval, with visual evidence, item-level correction, audit artifacts, and structured database import.

## Two category concepts in review

The review UI now separates two different concepts that must not be mixed:

| Field | Meaning | Example | Used for |
|---|---|---|---|
| OCR row type / parser item type | What kind of receipt line the OCR/parser found | `item`, `discount`, `deposit`, `refund` | Receipt math, validation, parser quality checks |
| Product group / product key | What the purchased product is | `Personal Care` / `personal_care` | Spending analytics, Ask Your Receipts, RAG evidence |

The original parser output often stores the receipt-row type in `item.category`. That field is kept for backward compatibility, but during review it is shown as **OCR row type**. Product categorization is edited separately through `category_group`, `category_key`, `category_confidence`, `category_review_required`, and `category_reason`.

For approved purchase items, `category_reason` also acts as the reviewed semantic description used by
the RAG embedding policy. It should state what the item is or does, for example `Vittel is a brand of
mineral water.` Editing the category path or this reason selectively invalidates and re-embeds only the
affected item. Price, quantity, VAT, date, and merchant edits remain SQL-only updates.

This matters because a line such as `PFAND 0.25` should be treated as a `deposit` row type, while a line such as `HEAD&SHOULDERS CLASSIC` should be treated as a normal `item` row type with product category `Personal Care / personal_care` or a more specific derived category such as `personal_care/shampoo` in the database/RAG layer.


## Item pricing fields

The geometry-first main parser may populate additional item-level pricing fields such as:

- `original_price`
- `discount_amount`
- `tax_code`

These fields separate product identity from pricing context for receipt analytics and later RAG/DB import. Product categories should be reviewed against the clean product description, not against discount, coupon or loyalty text.
