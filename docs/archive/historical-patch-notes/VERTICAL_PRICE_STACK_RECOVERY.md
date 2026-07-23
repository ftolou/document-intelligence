# Vertical Price-Stack Recovery

Phase `v1.13.2-phase2-vertical-stack-balanced-fusion-fix` extends a guarded fallback for receipts where the product text is visible on the left but the right-side item prices were missed, split or shifted by OCR/layout reconstruction.

## Problem it addresses

Some receipts show a clear visual item table, but OCR may split it like this:

```text
KART. VORW. FESTK.
EIER FH. G.M-L
H-MILCH 3,8%        2,58 B
```

while the actual receipt contains a right-side vertical price stack:

```text
1,99 B
2,99 B
2,58 B
```

Small per-row right-column crops can fail because the crop is too narrow or cuts off digits. The vertical price-stack recovery crops the whole right-side amount band in the item region and OCRs the prices as a stack.

## Processing logic

```text
existing extraction + validation
→ if item sum is balanced: skip
→ if item sum is mismatched:
    detect item region before SUMME/payment footer
    collect left-side product rows
    crop full right-side price column
    OCR original + 2x upscaled crop
    extract ordered amount tokens with y positions
    pair product rows and amount rows by y/order
    apply only if printed-total validation improves
    mark recovered rows as review-required
```

## Safety rules

- Balanced receipts are never modified.
- Recovery only runs for item-total mismatch or no-items cases.
- Any recovered item is marked `requires_review=true`.
- Applied results are routed to Human Review before DB/RAG import.
- The recovery is generic layout logic. It does not use merchant-specific rules.

## Artifacts

For each attempted run, the pipeline writes:

```text
latest_v14_22_vertical_price_stack_recovery.json
latest_v14_22_receipt_vertical_price_stack_recovered.json
latest_v14_22_validation_report_vertical_price_stack_recovered.json
<run_id>_v14_22_vertical_price_stack_crops/price_stack_column.jpg
<run_id>_v14_22_vertical_price_stack_crops/price_stack_column_2x.jpg
```

The recovery JSON includes:

- product rows found on the left side
- crop bounds
- OCR variants
- extracted amount stack
- candidate item rows
- before/after reconciliation difference

## Human review behavior

If recovery applies changes, the receipt remains review-required even if the item sum balances. This prevents recovered rows from entering analytics or RAG without explicit approval.


## v1.13.1 parser fix

The first vertical price-stack implementation proved that the crop strategy was correct, but it could still fail when PaddleOCR returned only a few clean amount strings from a narrow price column.  This update keeps the same validation-gated recovery design but strengthens the amount extraction layer.

Changes:

- crops a slightly wider right-side amount band while avoiding middle unit-price columns;
- generates multiple OCR-friendly crop variants: original, 2x, padded 3x, high-contrast 4x and binary 4x;
- lowers the crop OCR confidence threshold because small receipt digits often have useful low-confidence OCR;
- reconstructs amount rows from OCR word/line boxes by y-position;
- parses tolerant amount formats such as `1,99`, `1, 99`, `1 . 99`, `459`, and `4,59A`;
- falls back to full-image right-column OCR boxes if crop OCR is incomplete;
- evaluates every evidence source against the printed total and selects the one that best improves reconciliation.

The recovery still never applies to already balanced receipts, and all applied rows remain `requires_review=true`.


## v1.13.2: balanced-fusion safety fix

The previous parser fix proved the crop strategy but still allowed one unsafe partial mutation: noisy OCR text such as `1, 58 8` could be parsed both as `1.58` and as digit-only `5.88`. That improved the numeric mismatch but produced a wrong item row.

This version changes the policy:

1. **Balanced-only mutation** — vertical price-stack recovery may save diagnostics for partial candidates, but it mutates the final receipt only if the reconstructed item table matches the printed total within tolerance.
2. **No unsafe digit-only parsing** — digit-only parsing is allowed only when the OCR row has no comma/dot decimal candidate.
3. **Fused evidence first** — the module evaluates region re-OCR preferred item rows plus table-arbitration item candidates before using crop OCR.
4. **Residual assignment is gated** — if all evidence rows sum close to the total and exactly one unmatched product row remains, the exact residual may be assigned to that row, marked as review-required.
5. **Review-only outcome** — any applied recovery remains human-review required before database/RAG import.

For the REWE 13-Mar-2020 failure mode, the expected recovered table is 14 rows and sums to `32.22`, moving the receipt from `reject` to `needs_review` rather than automatic import.
