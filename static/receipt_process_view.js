(function exposeReceiptProcessView(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.ReceiptProcessView = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function createReceiptProcessView() {
  function isObject(value) {
    return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
  }

  function firstDefined(...values) {
    for (const value of values) {
      if (value !== null && value !== undefined && value !== '') return value;
    }
    return null;
  }

  function numeric(value) {
    if (value === null || value === undefined || value === '' || typeof value === 'boolean') {
      return null;
    }
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function unwrapScalar(value, keys = []) {
    if (!isObject(value)) return value;
    for (const key of [...keys, 'value', 'amount']) {
      if (value[key] !== null && value[key] !== undefined && value[key] !== '') {
        return value[key];
      }
    }
    return null;
  }

  function scalarCurrency(value) {
    return isObject(value) ? firstDefined(value.currency, value.currency_code) : null;
  }

  function formatAddress(value) {
    if (!isObject(value)) return value;
    const direct = firstDefined(value.formatted, value.full_address, value.address_text);
    if (direct) return direct;

    const lines = [];
    const street = firstDefined(value.street, value.address_line_1, value.line1);
    const locality = [
      firstDefined(value.postal_code, value.zip, value.zip_code),
      firstDefined(value.city, value.locality, value.town),
    ].filter(Boolean).join(' ');
    const region = firstDefined(value.region, value.state, value.province);
    const country = firstDefined(value.country, value.country_code);
    if (street) lines.push(String(street));
    if (locality) lines.push(locality);
    if (region) lines.push(String(region));
    if (country) lines.push(String(country));
    return lines.join('\n') || null;
  }

  function normalizeItems(receipt) {
    const items = Array.isArray(receipt.items) ? receipt.items : [];
    return items.map((item) => {
      if (!isObject(item)) return item;
      return {
        ...item,
        description: firstDefined(
          item.description,
          item.product_description,
          item.clean_description,
          item.name,
          item.raw_name,
          item.text,
        ),
        line_total: firstDefined(item.line_total, item.final_price, item.total, item.amount),
        tax_rate: firstDefined(item.tax_rate, item.tax_rate_percent, item.vat_rate),
      };
    });
  }

  function normalizePayments(receipt) {
    if (Array.isArray(receipt.payments) && receipt.payments.length) {
      return receipt.payments.map((payment) => (isObject(payment) ? { ...payment } : payment));
    }
    const payment = isObject(receipt.payment) ? receipt.payment : {};
    const method = firstDefined(payment.method, payment.payment_method);
    const amount = firstDefined(
      payment.amount,
      unwrapScalar(payment.payment_received, ['payment_received']),
    );
    if (method === null && amount === null) return [];
    return [{
      method,
      amount,
      source_line_ids: firstDefined(payment.source_line_ids, payment.source_rows, []),
    }];
  }

  function normalizeTaxes(receipt) {
    if (Array.isArray(receipt.taxes) && receipt.taxes.length) {
      return receipt.taxes.map((tax) => (isObject(tax) ? { ...tax } : tax));
    }
    const tax = isObject(receipt.tax) ? receipt.tax : {};
    const vatLines = Array.isArray(tax.vat_lines) ? tax.vat_lines : [];
    if (vatLines.length) {
      return vatLines.map((line) => ({
        rate: firstDefined(line.rate, line.rate_percent, line.vat_rate, line.tax_rate),
        net: firstDefined(line.net, line.net_amount),
        tax: firstDefined(line.tax, line.vat_amount, line.tax_amount),
        gross: firstDefined(line.gross, line.gross_amount),
        source_line_ids: firstDefined(line.source_line_ids, line.source_rows, []),
      }));
    }
    const vatAmount = unwrapScalar(tax.vat_amount, ['vat_amount', 'tax_amount']);
    if (vatAmount === null || vatAmount === undefined || vatAmount === '') return [];
    return [{
      rate: 'VAT total',
      net: null,
      tax: vatAmount,
      gross: null,
      source_line_ids: [],
    }];
  }

  function failedChecks(validation) {
    if (Array.isArray(validation.issues) && validation.issues.length) {
      return validation.issues.map((issue) => (isObject(issue) ? { ...issue } : issue));
    }
    const checks = Array.isArray(validation.checks) ? validation.checks : [];
    return checks
      .filter((check) => isObject(check) && String(check.status || '').toLowerCase() === 'failed')
      .map((check) => ({
        code: check.code,
        severity: firstDefined(check.severity, 'review'),
        message: check.message,
        details: check.details,
      }));
  }

  function normalizeValidation(receipt, items) {
    const validation = isObject(receipt.validation) ? receipt.validation : {};
    const checks = Array.isArray(validation.checks) ? validation.checks : [];
    const metrics = isObject(validation.metrics) ? validation.metrics : {};
    const itemSumCheck = checks.find((check) => check && check.code === 'ITEM_SUM_RECONCILIATION');
    const checkValues = isObject(itemSumCheck?.values) ? itemSumCheck.values : {};

    const itemSum = firstDefined(
      validation.calculated_item_total,
      metrics.item_sum,
      checkValues.item_sum,
      items.reduce((sum, item) => sum + (numeric(item?.line_total) || 0), 0),
    );
    const finalTotal = firstDefined(
      metrics.final_purchase_total,
      checkValues.final_purchase_total,
      unwrapScalar(receipt.totals?.final_purchase_total, ['final_purchase_total', 'grand_total']),
      receipt.totals?.grand_total,
    );
    const derivedDifference = numeric(itemSum) !== null && numeric(finalTotal) !== null
      ? Math.round((numeric(itemSum) - numeric(finalTotal)) * 100) / 100
      : null;
    const difference = firstDefined(validation.difference, checkValues.difference, derivedDifference);

    let balanced = validation.balanced;
    if (balanced === null || balanced === undefined || balanced === '') {
      const itemStatus = String(itemSumCheck?.status || '').toLowerCase();
      if (itemStatus === 'passed') balanced = true;
      else if (itemStatus === 'failed') balanced = false;
      else if (numeric(difference) !== null) balanced = Math.abs(numeric(difference)) <= 0.02;
    }

    return {
      ...validation,
      import_decision: firstDefined(validation.import_decision, validation.status, receipt.parse_status),
      balanced,
      difference,
      calculated_item_total: itemSum,
      issues: failedChecks(validation),
    };
  }

  function normalizeReceiptForProcessView(receipt) {
    if (!isObject(receipt)) return receipt;
    const metadata = isObject(receipt.receipt_metadata) ? receipt.receipt_metadata : {};
    const merchant = isObject(receipt.merchant) ? receipt.merchant : {};
    const totals = isObject(receipt.totals) ? receipt.totals : {};
    const tax = isObject(receipt.tax) ? receipt.tax : {};
    const payment = isObject(receipt.payment) ? receipt.payment : {};
    const items = normalizeItems(receipt);

    const currency = firstDefined(
      receipt.currency,
      metadata.currency,
      scalarCurrency(totals.final_purchase_total),
      scalarCurrency(totals.net_amount),
      scalarCurrency(payment.payment_received),
      scalarCurrency(tax.vat_amount),
      'EUR',
    );

    const normalized = {
      ...receipt,
      currency,
      date: firstDefined(receipt.date, metadata.date),
      time: firstDefined(receipt.time, metadata.time),
      receipt_number: firstDefined(receipt.receipt_number, metadata.receipt_number),
      merchant: {
        ...merchant,
        address: formatAddress(merchant.address),
      },
      totals: {
        ...totals,
        subtotal: firstDefined(
          totals.subtotal,
          unwrapScalar(totals.net_amount, ['net_amount', 'subtotal']),
        ),
        tax_total: firstDefined(
          totals.tax_total,
          unwrapScalar(tax.vat_amount, ['vat_amount', 'tax_amount']),
        ),
        grand_total: firstDefined(
          totals.grand_total,
          unwrapScalar(totals.final_purchase_total, ['final_purchase_total', 'grand_total']),
        ),
        paid_total: firstDefined(
          totals.paid_total,
          unwrapScalar(payment.payment_received, ['payment_received', 'paid_total']),
        ),
        change: firstDefined(
          totals.change,
          unwrapScalar(payment.change_returned, ['change_returned', 'change']),
        ),
      },
      items,
      payments: normalizePayments(receipt),
      taxes: normalizeTaxes(receipt),
    };
    normalized.validation = normalizeValidation(normalized, items);
    return normalized;
  }

  return {
    formatAddress,
    normalizeReceiptForProcessView,
  };
});
