Resolve exactly one supplied failed constraint and return only a minimal patch list.
The validation domain and document structure are not predefined. Infer semantics only from
SOURCE_EVIDENCE, CURRENT_STRUCTURED_RESULT, TARGET_FAILED_CHECK, and VALIDATOR_CONTEXT.
Do not use retailer-, layout-, field-family-, or validation-code-specific assumptions.

Treat every value, residual, difference, index, path, tolerance, and status supplied by the
validator as authoritative under its own data model. Do not replace validator calculations
with a new calculation model. When several residuals are supplied, use their names and the
failed-check message to determine which failed relation each residual represents.

Use this generic procedure:
1. Restate the failed relation internally using only validator-provided facts.
2. Compare source evidence with the current structured result.
3. Consider only four generic defect classes:
   - an existing value is unsupported or assigned the wrong semantic role;
   - an existing array element is unsupported or contains wrong values;
   - an independently supported element is missing from an array;
   - one or more existing array elements are unsupported duplicates or fragments.
4. Reject hypotheses requiring invented values, hidden assumptions, or unrelated changes.
5. Prefer one directly supported mutation over a combination of indirect changes.
6. Simulate the proposed mutation against the failed validator relation.
7. Return a patch only when it is expected to resolve the target. Otherwise return
   {"patches": []}.

Patch rules:
- Use JSON Pointer paths exactly as listed in CORRECTION_TARGET.
- Use only permitted operations and paths.
- Preserve all unrelated values.
- Never replace a supported value merely because another arithmetic combination balances.
- Never replace evidence with null to make a check unavailable.
- For inserted or replaced objects, populate only properties directly supported by evidence;
  preserve explicit nulls for unsupported optional properties when the current structure uses them.
- Preserve source order for array insertions.
- A calculation, modifier, continuation, subtotal, summary, or annotation must not become an
  independent array element unless the source identifies it independently.
- Mention the strongest supporting source identifiers or exact evidence fragment in reason.
- Keep every reason under 240 characters.
- Return only JSON matching the supplied schema.
