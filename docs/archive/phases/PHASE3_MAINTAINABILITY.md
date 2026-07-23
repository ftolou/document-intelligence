# Phase 3: Storage and Migration Refactor

Phase 3 replaces the 1,540-line `storage/receipt_db.py` implementation with a
small compatibility facade over focused repositories.

## Structure

```text
storage/
├── connection.py
├── models.py
├── normalization.py
├── fingerprints.py
├── migrations/
│   ├── runner.py
│   └── sql/
│       ├── 001_initial_schema.sql
│       └── 002_indexes.sql
├── repositories/
│   ├── analytics.py
│   ├── catalog.py
│   ├── items.py
│   ├── receipts.py
│   ├── review.py
│   └── search.py
└── receipt_db.py
```

`ReceiptDatabase` remains the public API used by the web, review, import, and
query layers. Existing callers do not need to change immediately. New code can
depend on a narrower repository where that improves clarity and testing.

## Migration behavior

`ReceiptDatabase` applies migrations during initialization. The migration runner:

1. creates `schema_migrations`,
2. applies missing migrations in order,
3. adopts databases created by older application versions,
4. adds only missing compatibility columns,
5. creates FTS5 when supported, and
6. keeps the deterministic lexical search fallback when FTS5 is unavailable.

Existing receipt and review data is not deleted or rebuilt.

Create a backup before the first Phase 3 startup:

```powershell
.\scripts\backup_receipt_db.ps1
```

Apply migrations explicitly with:

```powershell
python scripts/migrate_receipt_db.py
```

Inside the application container:

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec receipt-app `
  python /app/scripts/migrate_receipt_db.py
```

## Repository responsibilities

- `ReceiptRepository`: approved receipt import, listing, and deletion.
- `ItemRepository`: item and FTS-row insertion inside the receipt transaction.
- `AnalyticsRepository`: counts, summaries, filters, aggregations, grouping, and
  planner context.
- `SearchRepository`: FTS5 and lexical item retrieval.
- `ReviewRepository`: review queue and duplicate-candidate persistence.
- `CatalogRepository`: product alias reference data.

## Compatibility

The following imports remain valid:

```python
from receipt_intelligence.storage.receipt_db import ReceiptDatabase
from receipt_intelligence.storage.receipt_db import normalize_text
```

The facade also exposes repositories for gradual migration:

```python
database.analytics.aggregate_receipts(...)
database.search.search_items(...)
database.review.list_review_queue(...)
```
