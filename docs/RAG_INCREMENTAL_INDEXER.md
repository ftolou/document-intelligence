# RAG incremental item indexer

The item embedding index is derived from approved purchase-item rows in SQLite. Receipt data remains
authoritative; failed embedding generation never changes or removes approved receipt data.

## Canonical semantic document

Embedding policy `approved_product_semantics_v3` hashes and embeds:

- product description,
- normalized product description when materially different,
- reviewed category path,
- reviewed category reason / semantic description,
- parser item type through the document-type label.

It deliberately excludes price, quantity, date, merchant, currency, VAT, and receipt totals.

## Build or refresh the index

```powershell
python scripts/rebuild_rag_item_index.py
```

The indexer skips an item when both its canonical semantic `content_hash` and embedding model are
unchanged. The policy version is included in the hash source, so applying v1.28.1 causes one automatic
incremental refresh of existing eligible vectors. `--force` is not required for that upgrade.

```powershell
python scripts/rebuild_rag_item_index.py --force
```

Use `--force` only when the same source documents must be regenerated intentionally.

By default only product rows linked to approved receipts are indexed. Discounts, deposits, refunds,
fees, totals, and unapproved receipts are excluded.

## Post-review selective indexing

The database receipt editor compares exactly the fields used by the embedding policy. A product name,
category, or category-reason edit invalidates only that item's stored vector and calls
`index_item_ids(...)`. Analytical-only edits do not invoke Ollama.

The JSON report includes `eligible_items`, `embedded`, `unchanged`, `failed`, `dimension`, and failed-
batch messages. Index state is also written to `rag_index_state` for diagnostics and later retry.
