# Patch-only LLM correction

Version: `v1.13.3-phase2-patch-only-correction`

## Problem

The previous pipeline first attempted a compact correction patch, but when that patch did not improve validation it asked the LLM to regenerate the complete receipt JSON. Long receipts could hit the model output limit and stop inside a string, producing errors such as `Unterminated string` or `Invalid/incomplete JSON`.

## New policy

The correction stage is now patch-only:

1. The LLM receives a compact receipt projection, compact validation report, and bounded evidence summary.
2. It may return at most eight whitelisted operations.
3. The backend applies the operations to a copy of the receipt.
4. Deterministic validation compares the patched copy with the original.
5. The patched receipt is selected only when validation improves.
6. A failed, empty, or non-improving patch leaves the original receipt unchanged.
7. The pipeline never asks the LLM to rewrite the complete receipt JSON.

## Reliability controls

- Output is limited to a compact JSON object under roughly 2,000 characters.
- The model may use only `replace_field`, `replace_payments`, `remove_items`, `update_item`, and `add_item`.
- Unsupported operations are ignored.
- Reasons are length-limited.
- Malformed patch JSON is retried once with an even stricter compact-output instruction.
- Full receipt rewrite artifacts are retained as compatibility files with status `skipped` and reason `full_receipt_rewrite_disabled`.

## Expected log behavior

Instead of:

```text
error llm_correction Correction pass failed; original LLM output kept.
```

the pipeline now emits:

```text
skipped llm_correction Full receipt JSON correction rewrite disabled; patch-only result did not improve validation, so the original receipt was kept.
```

This is a controlled non-error outcome.
