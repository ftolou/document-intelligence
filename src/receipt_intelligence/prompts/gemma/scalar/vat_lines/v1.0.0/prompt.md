Extract only the printed VAT rows.

For each VAT row return:
- source_rows: row IDs supporting the VAT row;
- rate_percent;
- net_amount;
- vat_amount.

Do not include the gross receipt total as VAT.
Do not reverse net and VAT columns.
Use null when the column relationship is unclear.
