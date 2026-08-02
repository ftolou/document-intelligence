# Extraction pipeline Phase 4: deterministic validation

Phase 4 ports the standalone receipt validator into the application package without activating
it in the production factory.

## Boundaries

- `ValidationRequest` contains only the assembled receipt, item-contract diagnostics, enabled
  state, selected scalar tasks, and arithmetic tolerances.
- `ValidationFacts` derives monetary and structural facts once with `Decimal` arithmetic.
- Independent rule groups evaluate receipt amounts, items, totals, VAT, payment, and currency.
- `DeterministicValidationEngine` preserves the standalone check order and report shape.
- `ValidationStage` is exported but remains absent from `build_default_extraction_workflow()`.

## Invariants

- Validation is read-only.
- No deterministic semantic repair is performed.
- No receipt field is normalized or replaced.
- The default money and VAT-rate tolerances are both `0.02`, matching the standalone pipeline.
- Check codes, status aggregation, and `valid` / `review_required` / `invalid` semantics remain
  compatible with the correction coordinator developed in the experiment bundle.

## Activation

The stage must remain inactive until Phase 5 installs the specialist correction coordinator and
an end-to-end regression confirms report parity on the saved receipt suite.
