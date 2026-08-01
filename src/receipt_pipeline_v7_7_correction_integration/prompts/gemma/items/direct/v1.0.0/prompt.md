Extract every top-level separately purchased or separately charged item from the
complete receipt transcription.

For each item return:
- name
- final_price
- quantity
- unit
- discount_amount
- original_price


Rules:

1. A named product or service row identifies a candidate purchased item.
2. A monetary amount on a named product row is only a price candidate. 
Do not assign final_price until all contiguous rows belonging to that item have been examined.
3. Letters such as A, B, E, F, O, or V following an amount may be VAT category
   signs and are not part of the price or currency.
4. When the named row has no clear final amount and an immediately following
   quantity, weight, or calculation row belongs to it, use the final amount from
   that continuation row as final_price.
5. A quantity, weight, or "N x unit-price" row immediately following a named
   product and before the next named product MUST be attached to that preceding
   product.
6. Do not create a separate item from a quantity, weight, unit-price, article-ID,
   discount, or other continuation row.
7. For a row shaped as "N x unit-price final" or "N * unit-price final":
   - quantity = N;
   - final_price = final.
8. For "weight unit x price/unit final", set quantity to the printed weight,
   unit to the printed unit, and final_price to final.
9. When quantity multiplied by unit price equals the item's final line price
   within normal currency rounding, this confirms the relationship.
10. Never use the unit price as final_price when a separate final amount exists.
11. Included menu or bundle components without a separate charge are not
    separate items.
12. Separately charged deposits, bags, and services are items.
13. Do not create items from totals, VAT, payment, change, headers, metadata,
    receipt-wide discounts, or footer rows.
14. When no explicit quantity is printed, return null. Never assume quantity 1.
15. Return null for a missing unit. Never return an empty string.
16. Preserve the printed OCR product text. Do not silently correct or translate
    product names.
17. Return numeric JSON values using a decimal point.
18. Return only JSON matching the supplied schema.
19. An item block starts with a named product or service and continues through related quantity, 
identifier, variant, size, unit-price, price-adjustment, discount and explanatory rows. 
Intervening non-product detail rows do not end the block. 
The block ends at the next named purchased item or at an unambiguous receipt-level total, 
payment, tax or footer section.
20. When an item block contains multiple amounts, classify all of them before producing the item.
Use signs, percentages, row order, semantic labels and arithmetic relationships to distinguish original price,
discount or surcharge, unit price and final charged price.
A later supported effective amount may supersede an earlier price candidate.
21. A named row with its own monetary amount remains a purchased-item candidate even when
its text is generic, abbreviated, truncated, uppercase, hyphenated, category-like, or unfamiliar.
Do not discard such a row solely because its name is semantically ambiguous.
22. Before returning, compare all named amount rows with the extracted item list. Every omitted
named amount row must be explainable as a total, tax, payment, discount, metadata, or footer row.
