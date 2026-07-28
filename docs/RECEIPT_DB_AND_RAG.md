# Receipt Database and RAG Design

## Why a database is required

Questions such as “How much did I spend on shampoo?” require exact arithmetic.
The correct source of truth is the structured database, not an LLM response.

The LLM/retrieval layer is useful for understanding and finding semantically
related items, for example:

- `HEAD&SHOULDERS` → shampoo
- `ELVITAL` → shampoo / hair care
- `ELMEX` → toothpaste / hygiene
- `DOVE DEO` → deodorant / hygiene

## Tables

### `receipts`

Stores receipt-level facts:

- `job_id`
- `merchant_name`
- `merchant_normalized`
- `receipt_date`
- `currency`
- `grand_total`
- `paid_total`
- `review_status`
- artifact paths
- full approved JSON

### `receipt_items`

Stores item-level facts:

- `raw_name`
- `normalized_name`
- `category`
- `quantity`
- `unit_price`
- `line_total`
- `vat_rate`
- `confidence`
- `embedding_text`
- full item JSON

### `receipt_item_fts`

SQLite FTS5 virtual table over enriched item context.
If FTS5 is not available in the local SQLite build, the app falls back to a
lexical scorer over `receipt_items.embedding_text`.

## What “RAG” means here

This is a local, dependency-light RAG/search layer:

1. User asks a question.
2. The app parses merchant/date/category intent.
3. SQL applies exact filters.
4. FTS/lexical retrieval searches enriched item text.
5. The result includes matched item evidence.
6. Any sum is computed deterministically.

The RAG embedding foundation now includes a tested Ollama embedding client,
canonical item-document generation, and SQLite storage tables for rebuildable
item vectors. Vector generation and retrieval are not connected to the query
engine yet. See `docs/RAG_EMBEDDING_STORAGE.md`.

## Why this is portfolio-relevant

For AI Manager / AI Automation roles, the important point is not only model
usage. The important point is controlled AI operation:

- facts are stored
- results are reviewable
- retrieval is explainable
- financial calculations are deterministic
- answers show evidence
- approved data can be audited later

## Category separation for RAG and analytics

The database import intentionally separates parser/OCR row type from product/spending category:

- `receipt_items.parser_item_type` stores the receipt-line type, for example `item`, `discount`, `deposit`, or `refund`.
- `receipt_items.category` stores the product/spending category used by analytics and Ask Your Receipts.
- `receipt_items.category_group` and `receipt_items.category_key` store the human-reviewed item taxonomy fields.

The import does **not** use parser row types as spending categories. This prevents bad analytics such as every normal product being categorized only as `item`. For specific questions such as “Which receipt had shampoo?”, the import first uses product aliases and item text to derive specific categories such as `personal_care/shampoo`; otherwise it falls back to the reviewed product category key/group.


## Query Planner upgrade

The current Ask path uses the RAG-SQL LangGraph engine. Semantic product concepts are resolved to item IDs, an LLM proposes a typed read-only SQL plan, and deterministic validation/execution enforce the approved analytics scope. See `docs/RAG_SQL_ENGINE.md`.
