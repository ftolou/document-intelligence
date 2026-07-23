# Step 7.3.4 — Reviewed product semantics

This step adds deterministic descriptive product answers to the isolated RAG-SQL path.

## Supported operations

- `describe_product`
- `identify_product_type`
- `identify_brand`

Product names are first resolved through the existing item RAG layer. SQL then reads only the resolved `item_id` rows from `analytics_purchase_items`. Descriptive answers are formatted without a second LLM call.

## Evidence contract

`semantic_description` and `category_reason` are editable, reviewed item fields. `category` may support a product-type answer. Missing evidence produces `insufficient_info`; zero matching SQL rows produce `not_found`.

`merchant` and `merchant_name` identify the receipt seller. They are never product-brand evidence. Both the planner prompt and deterministic SQL validator reject seller-as-brand behavior.

## Database migration

Schema version 6 adds `category_reason` and `semantic_description` to `receipt_items`, backfills them from existing item `raw_json`, and exposes them in `analytics_purchase_items`. Changes to `semantic_description` selectively invalidate/rebuild the affected item embedding.
