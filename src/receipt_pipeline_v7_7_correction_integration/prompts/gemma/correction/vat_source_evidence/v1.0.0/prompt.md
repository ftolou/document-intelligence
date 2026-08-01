# VAT SOURCE-EVIDENCE EXTRACTION — V9

You receive a receipt transcription whose rows have stable identifiers.

Extract only VAT evidence explicitly printed in the source.

Return exactly one JSON object matching the supplied schema.

Do not correct an existing result. Do not calculate, reconcile, normalize,
infer residuals, or invent values.

## OUTPUT

Return:

- `vat_evidence_blocks`: completed VAT evidence;
- `unresolved_candidate_rows`: likely VAT evidence that cannot be assigned
  safely.

## ONE VALUE-BEARING ROW PER BLOCK

Every completed evidence block must contain exactly one `source_row`.

Create a separate block for every independent VAT data row governed by the
same header. Reuse the applicable header only through `context_rows`.

Never combine values from two independent source rows into one completed
block.

An explicitly labelled aggregate VAT row is also a separate completed block,
even when it has no printed rate value.

## ROLE ASSIGNMENT

Supported field roles are:

- `vat_amount`
- `gross_amount`
- `rate_percent`
- `net_amount`

This list defines vocabulary only. Its order is never an output order.

A VAT row may contain any explicit subset of these roles. A header is complete
for the subset it explicitly names. Do not introduce a role absent from the
applicable header or labels.

Use explicit printed labels and positional structure:

- a printed tax-rate or percentage label establishes `rate_percent`;
- a printed amount-before-tax label establishes `net_amount`;
- a printed tax-amount label establishes `vat_amount`;
- a printed amount-including-tax label establishes `gross_amount`.

Recognize equivalent wording in the language used by the receipt.

Punctuation, separators, and mathematical symbols between role labels do not
invalidate a positional header and do not change its printed left-to-right
order.

When an applicable positional header exists, it remains authoritative for the
following aligned VAT rows. Descriptive text inside a value row does not
replace, add, or reorder the header roles.

Do not use arithmetic plausibility, customary receipt layouts, expected tax
relationships, or the role-list order to assign fields.

## BUILDING A COMPLETED BLOCK

### `context_rows`

Return the header or nearby label rows that establish the roles.

Use an empty array only when the role labels occur directly in `source_row`.

### `source_row`

Return exactly one row identifier. That row must literally contain every
returned field value and the optional `row_label`.

### `row_label`

Copy an explicit aggregate or descriptive row label literally from
`source_row`. Otherwise use JSON null.

A row governed by an active VAT header may be completed without a rate value.
An explicit aggregate row may therefore contain only the net, VAT, gross, or
other subset printed under that header.

### `fields`

Return fields in the printed value order of `source_row`.

Each field contains:

- `role`: one supported VAT role;
- `value`: the literal printed value.

Requirements:

1. Keep each role and value together in one field object.
2. Return each role at most once in one block.
3. Do not reorder fields into a canonical role sequence.
4. Do not repeat one printed occurrence under multiple roles.
5. For a positional row, follow the active header from left to right.
6. For directly labelled evidence, follow labels and values in printed order.
7. Omit roles that are not explicitly printed or established by the header.

Tax-category markers are outside this extraction task. Do not return them as
fields or row labels. Do not treat a category marker as a rate, net, VAT, or
gross value.

## SOURCE-GROUNDING RULES

1. Every non-null returned string must occur literally in `source_row`, except
   role tokens and row identifiers.
2. Preserve decimal separators, signs, percentage notation, spaces,
   punctuation, and text as printed.
3. Do not convert strings to numbers.
4. Do not calculate missing or derived values.
5. Do not reconcile VAT evidence with receipt totals.
6. Do not infer missing net, VAT, gross, or rate values.
7. Printed labels and position remain authoritative even when values appear
   arithmetically inconsistent.
8. Context rows may support multiple independent completed blocks.
9. Do not use one `source_row` in more than one completed block.
10. Receipt totals, payments, change, discounts, item prices, and
    quantity-price calculations are not VAT evidence merely because they
    contain numbers.

## UNRESOLVED CANDIDATES

Use `unresolved_candidate_rows` only when rows are likely VAT evidence but
cannot be assigned safely because:

- no applicable header or explicit labels establish the roles;
- value alignment conflicts with the active header;
- explicit labels conflict; or
- the VAT evidence boundary is genuinely ambiguous.

Rows outside a plausible VAT context are not unresolved VAT candidates merely
because they contain percentages or monetary values.

Return empty arrays when no corresponding evidence exists.

Do not include Markdown fences, commentary, explanations, or text outside the
JSON object.

## SOURCE_EVIDENCE

$source_evidence

## OUTPUT_SCHEMA

$output_schema
