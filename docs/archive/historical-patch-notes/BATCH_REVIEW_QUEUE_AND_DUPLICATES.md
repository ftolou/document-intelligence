# Phase 2.6 — Batch Review Queue, Duplicate Detection and DB Management

This phase changes the batch workflow from immediate manual review to an operational queue:

```text
Batch run -> processed receipt artifacts -> SQLite review_queue -> human review -> approved receipt DB / Ask Your Receipts
```

## Why this design

Batch processing should run unattended. The user should not be forced to approve every receipt during extraction. Instead, each processed receipt is stored as a queue entry with a status:

- `needs_review`
- `auto_validated`
- `rejected`
- `duplicate_candidate`
- `approved`
- `imported`
- `duplicate_confirmed`

The `review_queue` is separate from the trusted `receipts` and `receipt_items` tables used by Ask Your Receipts. This prevents unreviewed or duplicate data from polluting spending analytics.

## UI changes

The app now has four main tabs:

1. **Run receipt** — single receipt processing and detailed side-by-side human review.
2. **Batch runner** — serial folder processing.
3. **Review Queue** — pending, rejected, duplicate and approved queue items.
4. **Ask Your Receipts** — RAG/analytics over trusted approved/imported receipts.

## Review Queue behavior

After each single or batch child job completes, the app registers the final receipt JSON in the SQLite `review_queue` table. The queue row stores:

- job ID
- merchant, date/time, total and item count
- validation decision, balance result and difference
- original image path and final receipt artifact path
- duplicate score and duplicate candidate evidence
- raw final receipt JSON for auditability

A user can later open a row from the Review Queue and review it in the existing side-by-side Human Review screen.

## Duplicate detection

Duplicate detection is intentionally score-based, not a brittle yes/no rule. The current local implementation checks:

- exact file SHA-256 hash
- same merchant
- same grand total
- same date
- same time
- item-name overlap where available

Scores of 70 or higher become `duplicate_candidate` queue entries. The UI shows the evidence and lets the user mark the receipt as duplicate or dismiss the duplicate flag.

## DB management

The Ask Your Receipts tab now includes database management controls:

- list approved/imported DB receipts
- delete a single receipt record from the analytics/RAG database
- delete all approved/imported receipt records
- optionally clear the review queue as well

Deleting a receipt from the DB removes it from Ask Your Receipts, but it does not delete the original output artifacts unless the review queue is also cleared manually or via the delete-all option.

## Safety rule

Only approved/imported receipts should be used for Ask Your Receipts. Review queue entries are staging data and should remain out of analytics/RAG until the user approves them.
