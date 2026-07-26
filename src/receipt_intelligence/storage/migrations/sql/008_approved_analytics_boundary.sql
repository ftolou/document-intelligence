DROP VIEW IF EXISTS analytics_purchase_items;
DROP VIEW IF EXISTS analytics_receipts;

-- Embeddings are derived data. Remove vectors that were created while a
-- receipt/item was approved but are no longer eligible after review.
DELETE FROM rag_item_embeddings
WHERE item_id IN (
    SELECT i.id
    FROM receipt_items AS i
    JOIN receipts AS r ON r.id = i.receipt_id
    WHERE lower(COALESCE(r.review_status, '')) NOT IN (
              'approved', 'accepted', 'complete', 'completed'
          )
       OR lower(COALESCE(i.review_status, '')) IN ('rejected', 'needs_review')
);

CREATE VIEW analytics_receipts AS
SELECT
    r.id AS receipt_id,
    r.job_id AS job_id,
    r.merchant_name AS merchant_name,
    COALESCE(NULLIF(r.merchant_normalized, ''), r.merchant_name) AS merchant,
    r.receipt_date AS receipt_date,
    CASE
        WHEN r.receipt_date GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-*'
        THEN substr(r.receipt_date, 1, 7)
        ELSE NULL
    END AS receipt_month,
    r.receipt_time AS receipt_time,
    r.currency AS currency,
    r.subtotal AS subtotal,
    r.tax_total AS tax_total,
    r.grand_total AS grand_total,
    r.paid_total AS paid_total,
    r.payment_method AS payment_method,
    (
        SELECT COUNT(*)
        FROM receipt_items AS i
        WHERE i.receipt_id = r.id
          AND lower(COALESCE(i.parser_item_type, 'item')) IN (
              'item', 'product', 'purchase_item', 'purchased_product'
          )
          AND lower(COALESCE(i.review_status, '')) <> 'rejected'
    ) AS item_count
FROM receipts AS r
WHERE lower(COALESCE(r.review_status, '')) IN (
    'approved', 'accepted', 'complete', 'completed'
);

CREATE VIEW analytics_purchase_items AS
SELECT
    i.id AS item_id,
    i.receipt_id AS receipt_id,
    r.job_id AS job_id,
    i.item_index AS item_index,
    r.merchant_name AS merchant_name,
    COALESCE(NULLIF(r.merchant_normalized, ''), r.merchant_name) AS merchant,
    r.receipt_date AS receipt_date,
    CASE
        WHEN r.receipt_date GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-*'
        THEN substr(r.receipt_date, 1, 7)
        ELSE NULL
    END AS receipt_month,
    r.currency AS currency,
    i.raw_name AS description,
    i.normalized_name AS normalized_name,
    COALESCE(NULLIF(i.category_key, ''), NULLIF(i.category_group, ''), i.category) AS category,
    i.category_key AS category_key,
    i.category_reason AS category_reason,
    i.semantic_description AS semantic_description,
    i.parser_item_type AS parser_item_type,
    i.quantity AS quantity,
    i.unit AS unit,
    i.unit_price AS unit_price,
    i.original_price AS original_price,
    i.discount_amount AS discount_amount,
    i.line_total AS line_total,
    i.tax_code AS tax_code,
    i.confidence AS confidence
FROM receipt_items AS i
JOIN receipts AS r ON r.id = i.receipt_id
WHERE lower(COALESCE(r.review_status, '')) IN (
        'approved', 'accepted', 'complete', 'completed'
    )
  AND lower(COALESCE(i.review_status, '')) <> 'rejected'
  AND lower(COALESCE(i.parser_item_type, 'item')) IN (
      'item', 'product', 'purchase_item', 'purchased_product'
  );
