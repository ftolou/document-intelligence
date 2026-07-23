# Phase 2.3 table interpretation refinements

This update focuses on making the dedicated table interpretation stage actually usable as an intermediate artifact.

## Main fixes

1. **Compact table interpretation output**
   - Removed verbose per-row explanations from the schema.
   - Kept only canonical row fields, row type, source row id, confidence, and settlement fields.
   - Reduced input evidence size by removing long hint reasons and neighbour text from the table-interpreter prompt.

2. **Prompt files separated from Python code**
   - Table interpreter, main parser, correction pass, and categorizer prompts now live in `src/receipt_intelligence/prompts/`.

3. **Headerless receipt table handling**
   - Prompt explicitly instructs the LLM to infer columns from repeated row layout, amount order, signs, tax suffixes, quantity rows and arithmetic consistency.

4. **Product text vs context text**
   - Prompt preserves complete product identity in `product_description`.
   - Coupon/promotion/context wording belongs in `line_note` or `promotion_note`, not in the clean product name.

5. **Quantity-note handling**
   - Rows such as `2 Stk x 1,39` should become quantity/unit-price evidence for nearby product rows when the product line total matches, not standalone items.

6. **Percent-in-product handling**
   - A product row like `MOZZARELLA 40%` should not be treated as tax merely because it contains `%`.
   - Tax interpretation requires tax-context words such as `MwSt`, `USt`, `VAT`, `Tax`, `Netto`, `Brutto`, or a tax-table section.

7. **Settlement/tender modeling**
   - The table interpreter and main parser prompts now explicitly model amount due, tenders/credits, coupon/voucher payments and change.

## Expected impact

The table interpreter should be less likely to fail with `Unterminated string` or incomplete JSON because both input and output are more compact. The main parser receives a smaller, clearer intermediate artifact and should be less likely to flatten discount/coupon context into item descriptions.
