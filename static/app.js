const form = document.getElementById('uploadForm');
const statusCard = document.getElementById('statusCard');
const jobIdEl = document.getElementById('jobId');
const stateEl = document.getElementById('state');
const eventsEl = document.getElementById('events');
const artifactsEl = document.getElementById('artifacts');
const previewEl = document.getElementById('preview');
const summaryEl = document.getElementById('live-status');
const receiptSummaryEl = document.getElementById('receiptSummary');
const progressFillEl = document.getElementById('progress-bar-fill');
const batchForm = document.getElementById('batchForm');
const batchStatusCard = document.getElementById('batchStatusCard');
const batchJobIdEl = document.getElementById('batchJobId');
const batchStateEl = document.getElementById('batch-state');
const batchEventsEl = document.getElementById('batchEvents');
const batchArtifactsEl = document.getElementById('batchArtifacts');
const batchSummaryEl = document.getElementById('batchSummary');
const batchSummaryStatusEl = document.getElementById('batch-live-status');
const batchProgressFillEl = document.getElementById('batch-progress-bar-fill');
const humanReviewPanelEl = document.getElementById('humanReviewPanel');
const reviewHandoffPanelEl = document.getElementById('reviewHandoffPanel');
const reviewSelectionHeaderEl = document.getElementById('reviewSelectionHeader');
const askReceiptsForm = document.getElementById('askReceiptsForm');
const askReceiptsQuestionEl = document.getElementById('askReceiptsQuestion');
const askReceiptsSaveJsonLogEl = document.getElementById('askReceiptsSaveJsonLog');
const askReceiptsLogHelpEl = document.getElementById('askReceiptsLogHelp');
const askReceiptsEngineBadgeEl = document.getElementById('askReceiptsEngineBadge');
const askReceiptsResultEl = document.getElementById('askReceiptsResult');
const askQueryProgressEl = document.getElementById('askQueryProgress');
const askQueryProgressMessageEl = document.getElementById('askQueryProgressMessage');
const askQueryElapsedEl = document.getElementById('askQueryElapsed');
const askResultStatusEl = document.getElementById('askResultStatus');
const askQueryExampleButtons = Array.from(document.querySelectorAll('[data-query-example]'));
const receiptDbSummaryEl = document.getElementById('receiptDbSummary');
const receiptDbBadgeEl = document.getElementById('receiptDbBadge');
const refreshReceiptDbButton = document.getElementById('refresh-receipt-db-button');
const reviewQueueResultEl = document.getElementById('reviewQueueResult');
const reviewQueueBadgeEl = document.getElementById('reviewQueueBadge');
const reviewQueueFilterEl = document.getElementById('reviewQueueFilter');
const reviewQueueSearchEl = document.getElementById('reviewQueueSearch');
const reviewQueueSummaryEl = document.getElementById('reviewQueueSummary');
const refreshReviewQueueButton = document.getElementById('refresh-review-queue-button');
const receiptDbListEl = document.getElementById('receiptDbList');
const refreshReceiptListButton = document.getElementById('refresh-receipt-list-button');
const deleteAllReceiptsButton = document.getElementById('delete-all-receipts-button');
let pollTimer = null;
let batchPollTimer = null;
let currentJobId = null;
let currentReceiptDbId = null;
let currentArtifacts = {};
let currentReviewSaveUrl = null;
let currentReviewSaveMethod = 'POST';
let currentReviewEditable = true;
let currentReviewIdentity = null;
let currentReviewQueueItems = [];
let currentReviewAdvanceAfterSave = false;
let appConfig = {};
let queryProgressTimer = null;
let queryProgressStartedAt = null;

const tabButtons = Array.from(document.querySelectorAll('[data-tab-target]'));
const tabPanels = Array.from(document.querySelectorAll('[data-tab-panel]'));

function setActiveTab(tabName) {
  const target = tabPanels.find((panel) => panel.dataset.tabPanel === tabName) ? tabName : 'run';
  for (const button of tabButtons) {
    const active = button.dataset.tabTarget === target;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', active ? 'true' : 'false');
  }
  for (const panel of tabPanels) {
    const active = panel.dataset.tabPanel === target;
    panel.classList.toggle('active', active);
    panel.hidden = !active;
  }
  if (target === 'data') {
    refreshReceiptDbSummary().catch(() => {});
    refreshReceiptDbList().catch(() => {});
  }
  if (target === 'review') {
    loadReviewQueue().catch(() => {});
  }
  if (target === 'models' && typeof loadModelDashboard === 'function') {
    loadModelDashboard().catch(() => {});
  }
}

function initializeTabs() {
  if (!tabButtons.length || !tabPanels.length) return;
  for (const button of tabButtons) {
    button.addEventListener('click', () => {
      const target = button.dataset.tabTarget || 'run';
      setActiveTab(target);
      if (window.location.hash !== `#${target}`) {
        history.replaceState(null, '', `#${target}`);
      }
    });
  }
  const initialTab = (window.location.hash || '').replace('#', '') || 'run';
  setActiveTab(initialTab);
}


async function loadConfig() {
  const res = await fetch('/api/config');
  if (!res.ok) throw new Error(`Config load failed: ${res.status}`);
  appConfig = await res.json();

  for (const key of ['ollama_url', 'model', 'transcription_model', 'num_ctx', 'num_predict', 'ocr_lang', 'ocr_device', 'max_crops', 'categorization_model', 'categorization_num_ctx', 'categorization_num_predict']) {
    const input = document.getElementById(key);
    if (input && appConfig[key] !== undefined) input.value = appConfig[key];
  }
  const batchFolder = document.getElementById('batch_folder_path');
  if (batchFolder && appConfig.batch_input_dir !== undefined) batchFolder.value = appConfig.batch_input_dir;
  const batchMax = document.getElementById('batch_max_files');
  if (batchMax && appConfig.batch_max_files !== undefined) batchMax.value = appConfig.batch_max_files;
  const batchRecursive = document.getElementById('batch_recursive');
  if (batchRecursive && appConfig.batch_recursive_default !== undefined) batchRecursive.checked = Boolean(appConfig.batch_recursive_default);
  if (askReceiptsLogHelpEl && appConfig.ask_receipts_json_log_dir) {
    askReceiptsLogHelpEl.textContent = `Off by default. Enabled queries are saved to ${appConfig.ask_receipts_json_log_dir}. Logs contain full prompts, model responses, diagnostics, and errors.`;
  }
  syncCategorizationCheckboxFromConfig();
  const correction = document.getElementById('correction_enabled');
  if (correction && appConfig.correction_enabled !== undefined) {
    correction.checked = Boolean(appConfig.correction_enabled);
  }
  updateAppVersionFromConfig();
}

function queryEngineLabel() {
  return 'RAG + SQL';
}

function updateAppVersionFromConfig() {
  const version = appConfig.app_version || appConfig.version || 'unknown';
  const title = document.getElementById('appTitle');
  const heading = document.getElementById('appHeading');
  const badge = document.getElementById('appVersionBadge');
  const runButton = document.getElementById('run-button');
  if (title) title.textContent = `Receipt Intelligence ${version}`;
  if (heading) heading.textContent = 'Receipt Intelligence';
  if (badge) badge.textContent = version;
  if (runButton) runButton.textContent = `Run ${version} parser`; 
}

function syncCategorizationCheckboxFromConfig() {
  const cat = document.getElementById('categorization_enabled');
  if (cat && appConfig.categorization_enabled !== undefined) {
    cat.checked = Boolean(appConfig.categorization_enabled);
  }
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}


function asNumber(value) {
  if (value === null || value === undefined || value === '') return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function formatMoney(value, currency) {
  const n = asNumber(value);
  if (n === null) return '—';
  const suffix = currency ? ` ${escapeHtml(currency)}` : '';
  return `${n.toFixed(2)}${suffix}`;
}

function formatLocalizedMoney(value, currency = 'EUR') {
  const n = asNumber(value);
  if (n === null) return '—';
  try {
    return new Intl.NumberFormat('de-DE', {
      style: 'currency',
      currency: String(currency || 'EUR').toUpperCase(),
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(n);
  } catch (_) {
    return `${n.toFixed(2)} ${String(currency || '').trim()}`.trim();
  }
}

function formatLocalizedDate(value) {
  if (!value) return '—';
  const raw = String(value);
  const normalized = /^\d{4}-\d{2}-\d{2}$/.test(raw) ? `${raw}T00:00:00` : raw;
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) return raw;
  return new Intl.DateTimeFormat('de-DE', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  }).format(date);
}

function queryStatusLabel(status) {
  const labels = {
    completed: 'Completed',
    needs_clarification: 'Clarification needed',
    not_found: 'No matches',
    insufficient_info: 'Insufficient information',
    error: 'Error',
    running: 'Running',
    ready: 'Ready',
  };
  return labels[status] || formatPlain(status);
}

function queryStatusBadgeClass(status) {
  if (status === 'completed') return 'badge ok';
  if (status === 'error') return 'badge bad';
  if (status === 'running' || status === 'needs_clarification') return 'badge warn';
  return 'badge neutral';
}

function setAskResultStatus(status) {
  if (!askResultStatusEl) return;
  askResultStatusEl.textContent = queryStatusLabel(status);
  askResultStatusEl.className = queryStatusBadgeClass(status);
}

function setQueryProgressStep(elapsedSeconds) {
  const thresholds = [
    ['analysis', 0, 'Understanding the request'],
    ['retrieval', 6, 'Searching approved receipt data'],
    ['resolution', 12, 'Resolving matching products'],
    ['planning', 24, 'Preparing a safe database query'],
    ['execution', 36, 'Validating and executing the query'],
  ];
  let currentIndex = 0;
  thresholds.forEach((entry, index) => {
    if (elapsedSeconds >= entry[1]) currentIndex = index;
  });
  const items = Array.from(document.querySelectorAll('[data-query-step]'));
  items.forEach((item, index) => {
    item.classList.toggle('done', index < currentIndex);
    item.classList.toggle('active', index === currentIndex);
  });
  if (askQueryProgressMessageEl) askQueryProgressMessageEl.textContent = thresholds[currentIndex][2];
}

function setQueryLoadingState(isLoading) {
  const button = document.getElementById('ask-receipts-button');
  const buttonLabel = button?.querySelector('.button-label');
  const controls = [askReceiptsQuestionEl, askReceiptsSaveJsonLogEl, ...askQueryExampleButtons];
  controls.forEach((control) => {
    if (control) control.disabled = Boolean(isLoading);
  });
  if (button) button.disabled = Boolean(isLoading);
  if (buttonLabel) buttonLabel.textContent = isLoading ? 'Processing…' : 'Ask receipts';

  if (!askQueryProgressEl) return;
  if (!isLoading) {
    askQueryProgressEl.classList.add('hidden');
    if (queryProgressTimer) window.clearInterval(queryProgressTimer);
    queryProgressTimer = null;
    queryProgressStartedAt = null;
    return;
  }

  askQueryProgressEl.classList.remove('hidden');
  queryProgressStartedAt = Date.now();
  if (askQueryElapsedEl) askQueryElapsedEl.textContent = '0s';
  const title = document.getElementById('askQueryProgressTitle');
  if (title) title.textContent = `${queryEngineLabel()} is processing your question…`;
  setQueryProgressStep(0);
  if (queryProgressTimer) window.clearInterval(queryProgressTimer);
  queryProgressTimer = window.setInterval(() => {
    const elapsed = Math.max(0, Math.floor((Date.now() - queryProgressStartedAt) / 1000));
    if (askQueryElapsedEl) askQueryElapsedEl.textContent = `${elapsed}s`;
    setQueryProgressStep(elapsed);
  }, 1000);
}

function formatPlain(value) {
  if (value === null || value === undefined || value === '') return '—';
  if (Array.isArray(value)) return value.length ? value.map(String).join(', ') : '—';
  return String(value);
}

function inputValue(value) {
  if (value === null || value === undefined || value === '—') return '';
  return String(value);
}

function firstDefined(...values) {
  for (const v of values) {
    if (v !== null && v !== undefined && v !== '') return v;
  }
  return null;
}

function rowHtml(cells) {
  return `<tr>${cells.map((cell) => `<td>${cell}</td>`).join('')}</tr>`;
}

function kvRows(rows) {
  return rows.map(([k, v]) => `<tr><th>${escapeHtml(k)}</th><td>${v}</td></tr>`).join('');
}

function renderReceiptSummary(receipt) {
  if (!receiptSummaryEl) return;
  if (!receipt || typeof receipt !== 'object') {
    receiptSummaryEl.className = 'receipt-summary empty';
    receiptSummaryEl.textContent = 'No structured receipt JSON available yet.';
    return;
  }

  const processView = globalThis.ReceiptProcessView;
  if (processView && typeof processView.normalizeReceiptForProcessView === 'function') {
    receipt = processView.normalizeReceiptForProcessView(receipt);
  }

  const currency = receipt.currency || 'EUR';
  const merchant = receipt.merchant || {};
  const totals = receipt.totals || {};
  const validation = receipt.validation || {};
  const items = Array.isArray(receipt.items) ? receipt.items : [];
  const payments = Array.isArray(receipt.payments) ? receipt.payments : [];
  const taxes = Array.isArray(receipt.taxes) ? receipt.taxes : [];
  const issues = Array.isArray(validation.issues) ? validation.issues : [];
  const warnings = Array.isArray(receipt.warnings) ? receipt.warnings : [];
  const unresolved = Array.isArray(receipt.unresolved_rows) ? receipt.unresolved_rows : [];

  const itemSum = items.reduce((acc, item) => acc + (asNumber(item.line_total) ?? 0), 0);
  const paidSum = payments.reduce((acc, payment) => acc + (asNumber(payment.amount) ?? 0), 0);

  const decision = validation.import_decision || receipt.parse_status || 'n/a';
  const decisionClass = decision === 'import' ? 'ok-cell' : (decision === 'reject' ? 'bad-cell' : 'muted');

  const overview = `
    <div class="summary-grid">
      <div class="summary-card">
        <h4>Merchant</h4>
        <table class="kv-table"><tbody>${kvRows([
          ['Name', escapeHtml(formatPlain(merchant.name))],
          ['Address', escapeHtml(formatPlain(merchant.address)).replaceAll('\n', '<br>')],
          ['Tax ID', escapeHtml(formatPlain(merchant.tax_id))],
          ['Receipt no.', escapeHtml(formatPlain(receipt.receipt_number))],
          ['Date', escapeHtml(formatPlain(receipt.date))],
          ['Time', escapeHtml(formatPlain(receipt.time))],
          ['Currency', escapeHtml(formatPlain(currency))],
        ])}</tbody></table>
      </div>
      <div class="summary-card">
        <h4>Totals</h4>
        <table class="kv-table"><tbody>${kvRows([
          ['Subtotal', formatMoney(totals.subtotal, currency)],
          ['Tax total', formatMoney(totals.tax_total, currency)],
          ['Grand total', formatMoney(totals.grand_total, currency)],
          ['Paid total', formatMoney(firstDefined(totals.paid_total, paidSum || null), currency)],
          ['Change', formatMoney(totals.change, currency)],
          ['Item sum', formatMoney(validation.calculated_item_total ?? itemSum, currency)],
        ])}</tbody></table>
      </div>
      <div class="summary-card">
        <h4>Validation</h4>
        <table class="kv-table"><tbody>${kvRows([
          ['Decision', `<span class="${decisionClass}">${escapeHtml(formatPlain(decision))}</span>`],
          ['Balanced', escapeHtml(formatPlain(validation.balanced))],
          ['Difference', formatMoney(validation.difference, currency)],
          ['Issues', escapeHtml(String(issues.length))],
          ['Items', escapeHtml(String(items.length))],
          ['Payments', escapeHtml(String(payments.length))],
          ['Categorization', escapeHtml(formatPlain((receipt.categorization || {}).status))],
          ['Cat. review items', escapeHtml(String((receipt.categorization || {}).category_review_count ?? items.filter(i => i && i.category_review_required).length))],
        ])}</tbody></table>
      </div>
    </div>`;

  const itemRows = items.length ? items.map((item, idx) => {
    const qty = firstDefined(item.quantity, '—');
    const tax = firstDefined(item.tax_rate, item.tax_rate_percent, '—');
    const conf = asNumber(item.confidence);
    const catConf = asNumber(item.category_confidence);
    const categoryText = item.category_key ? `${formatPlain(item.category_group)} / ${formatPlain(item.category_key)}` : '—';
    const reviewReasons = Array.isArray(item.category_review_reasons) ? item.category_review_reasons.join(', ') : '';
    const reviewBadge = item.category_review_required ? '<br><span class="badge warn">Review category</span>' : '';
    const rawConf = asNumber(item.category_confidence_raw);
    const rawConfText = rawConf !== null && catConf !== null && Math.abs(rawConf - catConf) > 0.001 ? ` raw ${rawConf.toFixed(2)}` : '';
    const catReasonParts = [];
    if (item.category_reason) catReasonParts.push(escapeHtml(item.category_reason));
    if (reviewReasons) catReasonParts.push(`review: ${escapeHtml(reviewReasons)}`);
    if (rawConfText) catReasonParts.push(escapeHtml(rawConfText));
    const catReason = catReasonParts.length ? `<br><span class="muted">${catReasonParts.join(' | ')}</span>` : '';
    return `<tr>
      <td class="numeric">${idx + 1}</td>
      <td>${escapeHtml(formatPlain(item.description))}${item.notes ? `<br><span class="muted">${escapeHtml(item.notes)}</span>` : ''}</td>
      <td class="numeric">${escapeHtml(formatPlain(qty))}</td>
      <td>${escapeHtml(formatPlain(item.unit))}</td>
      <td class="numeric">${formatMoney(item.unit_price, currency)}</td>
      <td class="numeric">${formatMoney(item.line_total, currency)}</td>
      <td class="numeric">${escapeHtml(formatPlain(tax))}</td>
      <td>${escapeHtml(categoryText)}${reviewBadge}${catReason}</td>
      <td class="numeric">${catConf === null ? '—' : catConf.toFixed(2)}</td>
      <td class="numeric">${conf === null ? '—' : conf.toFixed(2)}</td>
    </tr>`;
  }).join('') : `<tr><td colspan="10" class="muted">No items extracted.</td></tr>`;

  const paymentRows = payments.length ? payments.map((payment) => rowHtml([
    escapeHtml(formatPlain(payment.method)),
    `<span class="numeric">${formatMoney(payment.amount, currency)}</span>`,
    escapeHtml(formatPlain(payment.source_line_ids)),
  ])).join('') : `<tr><td colspan="3" class="muted">No payments extracted.</td></tr>`;

  const taxRows = taxes.length ? taxes.map((tax) => rowHtml([
    escapeHtml(formatPlain(tax.rate)),
    `<span class="numeric">${formatMoney(tax.net, currency)}</span>`,
    `<span class="numeric">${formatMoney(tax.tax, currency)}</span>`,
    `<span class="numeric">${formatMoney(tax.gross, currency)}</span>`,
    escapeHtml(formatPlain(tax.source_line_ids)),
  ])).join('') : `<tr><td colspan="5" class="muted">No tax table extracted.</td></tr>`;

  const issueRows = issues.length ? issues.map((issue) => {
    const sev = escapeHtml(formatPlain(issue.severity));
    const code = escapeHtml(formatPlain(issue.code));
    const msg = escapeHtml(formatPlain(issue.message));
    return `<tr>
      <td class="issue-severity-${sev.toLowerCase()}">${sev}</td>
      <td>${code}</td>
      <td>${msg}</td>
    </tr>`;
  }).join('') : `<tr><td colspan="3" class="ok-cell">No validation issues.</td></tr>`;

  const extraNoticeParts = [];
  if (warnings.length) extraNoticeParts.push(`${warnings.length} warning(s): ${escapeHtml(warnings.map(formatPlain).join(' | '))}`);
  if (unresolved.length) extraNoticeParts.push(`${unresolved.length} unresolved row(s).`);
  const extraNotice = extraNoticeParts.length ? `<div class="notice">${extraNoticeParts.join('<br>')}</div>` : '';

  receiptSummaryEl.className = 'receipt-summary';
  receiptSummaryEl.innerHTML = `
    ${overview}
    <div>
      <div class="receipt-section-title"><h4>Items</h4><small>Extracted receipt charge/credit lines</small></div>
      <div class="table-scroll"><table class="extracted-table items-table"><thead><tr>
        <th class="numeric">#</th><th>Description</th><th class="numeric">Qty</th><th>Unit</th><th class="numeric">Unit price</th><th class="numeric">Line total</th><th class="numeric">Tax</th><th>Category</th><th class="numeric">Cat. conf.</th><th class="numeric">Parse conf.</th>
      </tr></thead><tbody>${itemRows}</tbody></table></div>
    </div>
    <div>
      <div class="receipt-section-title"><h4>Payments</h4><small>Payment methods and tender amounts</small></div>
      <div class="table-scroll"><table class="extracted-table"><thead><tr><th>Method</th><th class="numeric">Amount</th><th>Source lines</th></tr></thead><tbody>${paymentRows}</tbody></table></div>
    </div>
    <div>
      <div class="receipt-section-title"><h4>Taxes</h4><small>VAT/MwSt table if available</small></div>
      <div class="table-scroll"><table class="extracted-table"><thead><tr><th>Rate</th><th class="numeric">Net</th><th class="numeric">Tax</th><th class="numeric">Gross</th><th>Source lines</th></tr></thead><tbody>${taxRows}</tbody></table></div>
    </div>
    <div>
      <div class="receipt-section-title"><h4>Validation issues</h4><small>Why the result imports, needs review, or is rejected</small></div>
      <div class="table-scroll"><table class="extracted-table"><thead><tr><th>Severity</th><th>Code</th><th>Message</th></tr></thead><tbody>${issueRows}</tbody></table></div>
    </div>
    ${extraNotice}
  `;
}


function reviewIssueMarkup(validation) {
  const issues = Array.isArray(validation?.issues) ? validation.issues : [];
  if (!issues.length) {
    return '<div class="review-validation-ok">No validation issues remain.</div>';
  }
  return `<div class="review-issue-list">${issues.map((issue) => {
    const severity = String(issue?.severity || 'medium').toLowerCase();
    return `<article class="review-issue review-issue-${escapeHtml(severity)}">
      <span>${escapeHtml(severity)}</span>
      <div><strong>${escapeHtml(formatPlain(issue?.code || 'VALIDATION_ISSUE'))}</strong><p>${escapeHtml(formatPlain(issue?.message || issue?.detail || 'Review required.'))}</p></div>
    </article>`;
  }).join('')}</div>`;
}

function reviewItemEditor(item, idx) {
  const raw = firstDefined(item.raw_description, item.description, item.raw_name, item.name, item.text, '');
  const productDescription = firstDefined(item.product_description, item.clean_description, item.normalized_name, item.description, item.name, '');
  const lineNote = firstDefined(item.line_note, item.promotion_note, '');
  const normalized = firstDefined(item.normalized_name, item.product_description, item.name, productDescription, '');
  const parserType = firstDefined(item.parser_item_type, item.receipt_row_type, item.line_type, item.category, 'item');
  const productCategoryGroup = firstDefined(item.category_group, '');
  const productCategoryKey = firstDefined(item.category_key, item.product_category, item.spending_category, '');
  const vatRate = firstDefined(item.vat_rate, item.tax_rate, item.tax_rate_percent, '');
  const extractionConfidence = firstDefined(item.confidence, '');
  const categoryConfidence = firstDefined(item.category_confidence, item.category_confidence_calibrated, '');
  const categoryReviewRequired = Boolean(item.category_review_required);
  const categoryReason = firstDefined(item.category_reason, '');
  const semanticDescription = firstDefined(item.semantic_description, '');
  const itemId = escapeHtml(inputValue(item._db_item_id));
  const selectOptions = (values, current) => values.map((value) => `<option value="${value}" ${String(current).toLowerCase() === value ? 'selected' : ''}>${value}</option>`).join('');
  return `<article class="review-item-card" data-review-item-row="${idx}" data-review-item-id="${itemId}">
    <header class="review-item-card-header">
      <span class="review-item-number">${idx + 1}</span>
      <div class="field review-item-product"><label>Product / printed item</label><input data-review-item-index="${idx}" data-review-item-field="product_description" value="${escapeHtml(inputValue(productDescription))}" /></div>
      <div class="field review-item-price"><label>Line total</label><input data-review-item-index="${idx}" data-review-item-field="line_total" type="number" step="0.01" value="${escapeHtml(inputValue(firstDefined(item.line_total, item.total, item.amount)))}" /></div>
      <div class="field review-item-type"><label>Row type</label><select data-review-item-index="${idx}" data-review-item-field="parser_item_type">${selectOptions(['item', 'discount', 'deposit', 'refund', 'fee', 'tax', 'info', 'unknown'], parserType)}</select></div>
    </header>
    <div class="review-item-core-grid">
      <div class="field"><label>Quantity</label><input data-review-item-index="${idx}" data-review-item-field="quantity" type="number" step="0.001" value="${escapeHtml(inputValue(item.quantity))}" /></div>
      <div class="field"><label>Unit</label><input data-review-item-index="${idx}" data-review-item-field="unit" value="${escapeHtml(inputValue(item.unit))}" /></div>
      <div class="field"><label>Unit price</label><input data-review-item-index="${idx}" data-review-item-field="unit_price" type="number" step="0.01" value="${escapeHtml(inputValue(item.unit_price))}" /></div>
      <div class="field"><label>Category</label><input data-review-item-index="${idx}" data-review-item-field="category_key" placeholder="groceries" value="${escapeHtml(inputValue(productCategoryKey))}" /></div>
      <div class="field"><label>Review state</label><select data-review-item-index="${idx}" data-review-item-field="review_status">${selectOptions(['approved', 'corrected', 'needs_review', 'rejected'], item.review_status || 'needs_review')}</select></div>
    </div>
    <details class="review-item-details">
      <summary>More item details</summary>
      <div class="review-item-detail-grid">
        <div class="field full-width"><label>Raw printed row</label><input data-review-item-index="${idx}" data-review-item-field="raw_description" value="${escapeHtml(inputValue(raw))}" /></div>
        <div class="field full-width"><label>Line note / promotion context</label><input data-review-item-index="${idx}" data-review-item-field="line_note" value="${escapeHtml(inputValue(lineNote))}" /></div>
        <div class="field"><label>Normalized item</label><input data-review-item-index="${idx}" data-review-item-field="normalized_name" value="${escapeHtml(inputValue(normalized))}" /></div>
        <div class="field"><label>Category group</label><input data-review-item-index="${idx}" data-review-item-field="category_group" value="${escapeHtml(inputValue(productCategoryGroup))}" /></div>
        <div class="field"><label>Category confidence</label><input data-review-item-index="${idx}" data-review-item-field="category_confidence" type="number" step="0.01" min="0" max="1" value="${escapeHtml(inputValue(categoryConfidence))}" /></div>
        <label class="checkline review-checkbox"><input data-review-item-index="${idx}" data-review-item-field="category_review_required" type="checkbox" ${categoryReviewRequired ? 'checked' : ''} /> Category requires review</label>
        <div class="field full-width"><label>Category reason</label><input data-review-item-index="${idx}" data-review-item-field="category_reason" value="${escapeHtml(inputValue(categoryReason))}" /></div>
        <div class="field full-width"><label>Semantic description</label><textarea rows="2" data-review-item-index="${idx}" data-review-item-field="semantic_description">${escapeHtml(inputValue(semanticDescription))}</textarea></div>
        <div class="field"><label>Original price</label><input data-review-item-index="${idx}" data-review-item-field="original_price" type="number" step="0.01" value="${escapeHtml(inputValue(firstDefined(item.original_price, item.gross_unit_price, '')))}" /></div>
        <div class="field"><label>Discount amount</label><input data-review-item-index="${idx}" data-review-item-field="discount_amount" type="number" step="0.01" value="${escapeHtml(inputValue(item.discount_amount))}" /></div>
        <div class="field"><label>Tax code</label><input data-review-item-index="${idx}" data-review-item-field="tax_code" value="${escapeHtml(inputValue(item.tax_code))}" /></div>
        <div class="field"><label>VAT rate</label><input data-review-item-index="${idx}" data-review-item-field="vat_rate" value="${escapeHtml(inputValue(vatRate))}" /></div>
        <div class="field"><label>OCR confidence</label><input data-review-item-index="${idx}" data-review-item-field="confidence" type="number" step="0.01" min="0" max="1" value="${escapeHtml(inputValue(extractionConfidence))}" /></div>
      </div>
    </details>
  </article>`;
}

function renderHumanReview(receipt, artifacts = {}, options = {}) {
  if (!humanReviewPanelEl) return;
  if (!receipt || typeof receipt !== 'object') {
    humanReviewPanelEl.className = 'review-panel empty';
    humanReviewPanelEl.textContent = 'Select a receipt from the queue to inspect and review it.';
    return;
  }

  const merchant = receipt.merchant || {};
  const totals = receipt.totals || {};
  const validation = receipt.validation || {};
  const items = Array.isArray(receipt.items) ? receipt.items : [];
  const decision = validation.import_decision || receipt.parse_status || 'needs_review';
  const editable = options.editable !== false;
  const persistedReview = options.review && typeof options.review === 'object'
    ? options.review
    : (receipt.human_review && typeof receipt.human_review === 'object' ? receipt.human_review : {});
  const imageUrl = artifacts?.receipt_image || artifacts?.image || artifacts?.source_image || '';
  const receiptImage = imageUrl
    ? `<a href="${escapeHtml(imageUrl)}" target="_blank" rel="noopener noreferrer"><img class="review-receipt-image" src="${escapeHtml(imageUrl)}" alt="Original receipt image" /></a>`
    : '<div class="review-image-missing">No original receipt image is available. The canonical draft can still be reviewed.</div>';
  const queueRevision = options.queueRecord?.review_revision ?? currentReviewIdentity?.review_revision ?? '—';
  const itemEditors = items.length
    ? items.map((item, idx) => reviewItemEditor(item, idx)).join('')
    : '<div class="review-empty-items">No item rows were extracted. Header fields can still be corrected, but item-level analytics will have no evidence.</div>';

  humanReviewPanelEl.className = 'review-panel';
  humanReviewPanelEl.innerHTML = `
    <div class="review-source-notice ${editable ? '' : 'review-read-only-notice'}">
      <div><strong>${options.source === 'database' ? 'Approved database receipt' : 'Canonical review draft'}</strong><span>${options.source === 'database' ? 'Changes commit to SQLite and selectively refresh semantic embeddings.' : 'Draft changes and every review revision are stored in SQLite.'}</span></div>
      <span class="badge neutral">revision ${escapeHtml(formatPlain(queueRevision))}</span>
    </div>
    <div class="review-layout">
      <aside class="review-image-panel">
        <div class="receipt-section-title"><h4>Receipt evidence</h4><small>Open image for full size</small></div>
        ${receiptImage}
        <div class="review-validation-panel">
          <div class="receipt-section-title"><h4>Validation</h4><small>${escapeHtml(formatPlain(decision))}</small></div>
          ${reviewIssueMarkup(validation)}
        </div>
      </aside>
      <div class="review-edit-panel">
        <section class="review-form-section">
          <div class="receipt-section-title"><h4>Merchant and transaction</h4><small>Core receipt identity</small></div>
          <div class="review-grid">
            <div class="field"><label>Merchant name</label><input data-review-field="merchant_name" value="${escapeHtml(inputValue(merchant.name))}" /></div>
            <div class="field"><label>Date</label><input data-review-field="date" value="${escapeHtml(inputValue(receipt.date))}" /></div>
            <div class="field"><label>Time</label><input data-review-field="time" value="${escapeHtml(inputValue(receipt.time))}" /></div>
            <div class="field"><label>Currency</label><input data-review-field="currency" value="${escapeHtml(inputValue(receipt.currency || 'EUR'))}" /></div>
            <div class="field"><label>Document type</label><input data-review-field="document_type" value="${escapeHtml(inputValue(firstDefined(receipt.document_type, receipt.type, 'receipt')))}" /></div>
            <div class="field"><label>Receipt category</label><input data-review-field="receipt_category" value="${escapeHtml(inputValue(firstDefined(receipt.receipt_category, receipt.document_category, '')))}" /></div>
            <div class="field"><label>Business category</label><input data-review-field="receipt_business_category" value="${escapeHtml(inputValue(firstDefined(receipt.receipt_business_category, receipt.business_category, '')))}" /></div>
            <div class="field full-width"><label>Merchant address</label><textarea data-review-field="merchant_address" rows="2">${escapeHtml(inputValue(merchant.address))}</textarea></div>
          </div>
        </section>

        <section class="review-form-section">
          <div class="receipt-section-title"><h4>Totals</h4><small>Validation is recalculated on save</small></div>
          <div class="review-totals-grid">
            <div class="field"><label>Subtotal</label><input data-review-field="subtotal" type="number" step="0.01" value="${escapeHtml(inputValue(totals.subtotal))}" /></div>
            <div class="field"><label>Tax total</label><input data-review-field="tax_total" type="number" step="0.01" value="${escapeHtml(inputValue(totals.tax_total))}" /></div>
            <div class="field"><label>Grand total</label><input data-review-field="grand_total" type="number" step="0.01" value="${escapeHtml(inputValue(totals.grand_total))}" /></div>
            <div class="field"><label>Paid total</label><input data-review-field="paid_total" type="number" step="0.01" value="${escapeHtml(inputValue(totals.paid_total))}" /></div>
            <div class="field"><label>Change</label><input data-review-field="change" type="number" step="0.01" value="${escapeHtml(inputValue(totals.change))}" /></div>
          </div>
        </section>

        <section class="review-form-section">
          <div class="receipt-section-title"><h4>Items</h4><small>${items.length} extracted row(s)</small></div>
          <div class="review-item-list">${itemEditors}</div>
        </section>

        <section class="review-form-section review-meta">
          <div class="review-grid">
            <div class="field"><label>Reviewer</label><input id="reviewerName" placeholder="Name or initials" value="${escapeHtml(inputValue(persistedReview.reviewer))}" /></div>
            <div class="field full-width"><label>Review notes</label><textarea id="reviewNotes" rows="3" placeholder="Corrections, uncertainty, or rejection reason">${escapeHtml(inputValue(persistedReview.notes))}</textarea></div>
          </div>
        </section>
      </div>
    </div>
    <div class="review-action-bar">
      <div><strong>Review decision</strong><small id="humanReviewMessage">Save a draft or finalize this receipt. Approval is blocked when core validation errors remain.</small></div>
      <div class="review-action-buttons">
        <button type="button" class="secondary" data-review-action="needs_review">Save draft</button>
        <button type="button" class="danger-secondary" data-review-action="rejected">Reject</button>
        <button type="button" data-review-action="approved">Approve</button>
        <button type="button" data-review-action="approved" data-review-advance="true">Approve &amp; next</button>
      </div>
    </div>`;

  for (const button of humanReviewPanelEl.querySelectorAll('[data-review-action]')) {
    button.disabled = !editable;
    if (editable) {
      button.addEventListener('click', () => saveHumanReview(
        button.dataset.reviewAction || 'needs_review',
        button.dataset.reviewAdvance === 'true',
      ));
    }
  }
  if (!editable) {
    for (const control of humanReviewPanelEl.querySelectorAll('input, select, textarea')) control.disabled = true;
    const message = document.getElementById('humanReviewMessage');
    if (message) message.textContent = options.readOnlyReason || 'This receipt is read-only.';
  }
}

function collectHumanReviewPayload(status = 'needs_review') {
  const fields = {};
  for (const el of humanReviewPanelEl.querySelectorAll('[data-review-field]')) {
    let value = el.type === 'checkbox' ? el.checked : el.value;
    if (el.type === 'number' && value !== '') value = Number(value);
    fields[el.dataset.reviewField] = value;
  }

  const itemMap = new Map();
  for (const el of humanReviewPanelEl.querySelectorAll('[data-review-item-index][data-review-item-field]')) {
    const index = Number(el.dataset.reviewItemIndex);
    if (!Number.isInteger(index)) continue;
    const field = el.dataset.reviewItemField;
    let value = el.type === 'checkbox' ? el.checked : el.value;
    if (el.type === 'number' && value !== '') value = Number(value);
    if (!itemMap.has(index)) {
      const row = el.closest('[data-review-item-row]');
      const itemId = Number(row?.dataset.reviewItemId || '');
      itemMap.set(index, Number.isInteger(itemId) && itemId > 0 ? { index, item_id: itemId } : { index });
    }
    itemMap.get(index)[field] = value;
  }
  return {
    identity: currentReviewIdentity || {},
    fields,
    items: Array.from(itemMap.values()).sort((a, b) => a.index - b.index),
    review: {
      reviewer: document.getElementById('reviewerName')?.value || '',
      status,
      notes: document.getElementById('reviewNotes')?.value || '',
    },
  };
}

async function saveHumanReview(status = 'needs_review', advance = false) {
  const msg = document.getElementById('humanReviewMessage');
  const buttons = Array.from(humanReviewPanelEl?.querySelectorAll('[data-review-action]') || []);
  if (!currentReviewEditable || !currentReviewSaveUrl) {
    if (msg) msg.textContent = 'This receipt is available in read-only mode.';
    return;
  }
  buttons.forEach((button) => { button.disabled = true; });
  if (msg) msg.textContent = `Saving ${status === 'needs_review' ? 'draft' : status} decision…`;
  const savedJobId = currentJobId;
  try {
    const res = await fetch(currentReviewSaveUrl, {
      method: currentReviewSaveMethod || 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(collectHumanReviewPayload(status)),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Human review save failed');
    applyReviewPayload(data);
    const effectiveStatus = data.receipt?.human_review?.status || data.review?.status || status;
    const revision = data.review_queue?.revision || data.review_revision?.revision || data.review?.review_revision || '—';
    const newMessage = document.getElementById('humanReviewMessage');
    if (newMessage) {
      newMessage.textContent = `Saved revision ${revision}. Effective status: ${effectiveStatus}. Changed fields: ${(data.review?.changed_fields || []).join(', ') || 'none'}.`;
    }
    refreshReceiptDbSummary().catch(() => {});
    refreshReceiptDbList().catch(() => {});
    await loadReviewQueue();
    if (advance) {
      const next = currentReviewQueueItems.find((item) => item.job_id !== savedJobId && ['needs_review', 'duplicate_candidate', 'rejected'].includes(String(item.queue_status || '')));
      if (next) await openReviewJob(next.job_id);
    }
  } catch (err) {
    const currentMessage = document.getElementById('humanReviewMessage');
    if (currentMessage) currentMessage.textContent = err.message || String(err);
  } finally {
    for (const button of Array.from(humanReviewPanelEl?.querySelectorAll('[data-review-action]') || [])) {
      button.disabled = !currentReviewEditable;
    }
  }
}


function renderReceiptDbSummary(summary) {
  if (!receiptDbSummaryEl) return;
  if (!summary || typeof summary !== 'object') {
    receiptDbSummaryEl.className = 'receipt-summary empty';
    receiptDbSummaryEl.textContent = 'Receipt database summary is not available.';
    if (receiptDbBadgeEl) receiptDbBadgeEl.textContent = 'DB unavailable';
    return;
  }
  if (receiptDbBadgeEl) receiptDbBadgeEl.textContent = `${summary.receipt_count || 0} receipt(s), ${summary.item_count || 0} item(s)`;
  const merchantRows = (summary.top_merchants || []).length ? summary.top_merchants.map((m) => rowHtml([
    escapeHtml(formatPlain(m.merchant)),
    `<span class="numeric">${escapeHtml(formatPlain(m.receipt_count))}</span>`,
    `<span class="numeric">${formatMoney(m.total_amount, 'EUR')}</span>`,
  ])).join('') : '<tr><td colspan="3" class="muted">No merchants yet.</td></tr>';
  const categoryRows = (summary.top_categories || []).length ? summary.top_categories.map((c) => rowHtml([
    escapeHtml(formatPlain(c.category)),
    `<span class="numeric">${escapeHtml(formatPlain(c.item_count))}</span>`,
    `<span class="numeric">${formatMoney(c.total_amount, 'EUR')}</span>`,
  ])).join('') : '<tr><td colspan="3" class="muted">No categories yet.</td></tr>';

  receiptDbSummaryEl.className = 'receipt-summary';
  receiptDbSummaryEl.innerHTML = `
    <div class="summary-grid">
      <div class="summary-card">
        <h4>Local receipt DB</h4>
        <table class="kv-table"><tbody>${kvRows([
          ['Path', `<code>${escapeHtml(formatPlain(summary.db_path))}</code>`],
          ['Receipts', escapeHtml(formatPlain(summary.receipt_count))],
          ['Items', escapeHtml(formatPlain(summary.item_count))],
          ['Schema', escapeHtml(formatPlain(summary.schema_version))],
        ])}</tbody></table>
      </div>
    </div>
    <div>
      <div class="receipt-section-title"><h4>Top merchants</h4><small>Approved receipt data only</small></div>
      <div class="table-scroll"><table class="extracted-table"><thead><tr><th>Merchant</th><th class="numeric">Receipts</th><th class="numeric">Receipt total</th></tr></thead><tbody>${merchantRows}</tbody></table></div>
    </div>
    <div>
      <div class="receipt-section-title"><h4>Top product categories</h4><small>Derived from item categorization and alias rules; parser/OCR row types are stored separately</small></div>
      <div class="table-scroll"><table class="extracted-table"><thead><tr><th>Product category</th><th class="numeric">Items</th><th class="numeric">Item total</th></tr></thead><tbody>${categoryRows}</tbody></table></div>
    </div>
  `;
}

async function refreshReceiptDbSummary() {
  if (!receiptDbSummaryEl) return;
  const res = await fetch('/api/receipt-db/summary');
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'DB summary failed');
  renderReceiptDbSummary(data);
}

function queueStatusBadge(status) {
  const value = String(status || '').toLowerCase();
  if (['approved', 'imported', 'auto_validated'].includes(value)) return 'badge ok';
  if (['rejected', 'duplicate_confirmed'].includes(value)) return 'badge bad';
  if (['needs_review', 'duplicate_candidate'].includes(value)) return 'badge warn';
  return 'badge neutral';
}

function renderReviewQueueSummary(summary = {}) {
  if (!reviewQueueSummaryEl) return;
  const values = {
    needs_review: summary.needs_review || 0,
    duplicate_candidate: summary.duplicate_candidate || 0,
    rejected: summary.rejected || 0,
    approved: (summary.approved || 0) + (summary.imported || 0),
  };
  for (const button of reviewQueueSummaryEl.querySelectorAll('[data-review-summary-filter]')) {
    const status = button.dataset.reviewSummaryFilter;
    const strong = button.querySelector('strong');
    if (strong) strong.textContent = String(values[status] || 0);
  }
}

async function loadReviewQueue() {
  if (!reviewQueueResultEl) return;
  const status = reviewQueueFilterEl?.value || 'all';
  reviewQueueResultEl.className = 'review-queue-list empty';
  reviewQueueResultEl.textContent = 'Loading review queue…';
  const res = await fetch(`/api/review-queue?status=${encodeURIComponent(status)}&limit=200`);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Review queue failed');
  currentReviewQueueItems = Array.isArray(data.items) ? data.items : [];
  renderReviewQueueSummary(data.summary || {});
  renderReviewQueue(currentReviewQueueItems);
}

function queueSearchText(item) {
  return [
    item.job_id,
    item.merchant_name,
    item.merchant_normalized,
    item.receipt_date,
    item.queue_status,
    item.decision,
    ...(Array.isArray(item.reason_codes) ? item.reason_codes : []),
  ].filter(Boolean).join(' ').toLowerCase();
}

function renderReviewQueue(items) {
  if (!reviewQueueResultEl) return;
  const query = String(reviewQueueSearchEl?.value || '').trim().toLowerCase();
  const visibleItems = query ? items.filter((item) => queueSearchText(item).includes(query)) : items;
  if (reviewQueueBadgeEl) reviewQueueBadgeEl.textContent = `${visibleItems.length} shown / ${items.length} loaded`;
  if (!visibleItems.length) {
    reviewQueueResultEl.className = 'review-queue-list empty';
    reviewQueueResultEl.textContent = query ? 'No receipts match this search.' : 'No receipts match this queue filter.';
    return;
  }
  reviewQueueResultEl.className = 'review-queue-list';
  reviewQueueResultEl.innerHTML = visibleItems.map((item) => {
    const reasons = Array.isArray(item.reason_codes) ? item.reason_codes : [];
    const active = String(item.job_id) === String(currentJobId || '');
    const validation = item.balanced === 1 ? 'balanced' : item.balanced === 0 ? 'not balanced' : 'balance unknown';
    return `<article class="review-queue-card ${active ? 'selected' : ''}" data-queue-card-job="${escapeHtml(item.job_id)}">
      <button type="button" class="review-queue-open" data-review-open="${escapeHtml(item.job_id)}">
        <span class="review-queue-card-top"><span class="${queueStatusBadge(item.queue_status)}">${escapeHtml(formatPlain(item.queue_status))}</span><time>${escapeHtml(formatPlain(item.receipt_date || 'no date'))}</time></span>
        <strong>${escapeHtml(formatPlain(item.merchant_name || item.merchant_normalized || 'Unknown merchant'))}</strong>
        <span class="review-queue-card-total">${formatMoney(item.grand_total, 'EUR')} · ${escapeHtml(formatPlain(item.item_count || 0))} item(s)</span>
        <span class="review-queue-card-validation">${escapeHtml(validation)} · ${escapeHtml(formatPlain(item.issue_count || 0))} issue(s)</span>
        ${reasons.length ? `<span class="review-reason-tags">${reasons.slice(0, 3).map((reason) => `<em>${escapeHtml(reason)}</em>`).join('')}</span>` : ''}
        <code>${escapeHtml(formatPlain(item.job_id))}</code>
      </button>
      ${item.duplicate_status === 'duplicate_candidate' || item.queue_status === 'duplicate_candidate' ? `<div class="review-queue-duplicate-actions">
        <button type="button" class="danger-secondary" data-queue-status="duplicate_confirmed" data-queue-job="${escapeHtml(item.job_id)}">Confirm duplicate</button>
        <button type="button" class="secondary" data-queue-status="dismissed_duplicate" data-queue-job="${escapeHtml(item.job_id)}">Not duplicate</button>
      </div>` : ''}
    </article>`;
  }).join('');
  for (const button of reviewQueueResultEl.querySelectorAll('[data-review-open]')) {
    button.addEventListener('click', () => {
      button.disabled = true;
      openReviewJob(button.dataset.reviewOpen)
        .catch((error) => renderReviewLoadError(error.message || String(error)))
        .finally(() => { button.disabled = false; });
    });
  }
  for (const button of reviewQueueResultEl.querySelectorAll('[data-queue-status][data-queue-job]')) {
    button.addEventListener('click', () => updateQueueStatus(button.dataset.queueJob, button.dataset.queueStatus));
  }
}

function renderReviewSelectionHeader(data) {
  if (!reviewSelectionHeaderEl) return;
  const receipt = data?.receipt || {};
  const merchant = receipt?.merchant?.name || 'Unknown merchant';
  const status = data?.queue_record?.queue_status || receipt?.human_review?.status || 'needs_review';
  const revision = data?.queue_record?.review_revision ?? data?.review_queue?.revision ?? currentReviewIdentity?.review_revision ?? '—';
  reviewSelectionHeaderEl.className = 'review-selection-header';
  reviewSelectionHeaderEl.innerHTML = `<div><p class="section-kicker">Selected receipt</p><h3>${escapeHtml(formatPlain(merchant))}</h3><span>${escapeHtml(formatPlain(receipt.date || 'No date'))} · ${formatMoney(receipt?.totals?.grand_total, receipt.currency || 'EUR')}</span></div><div class="review-selection-meta"><span class="${queueStatusBadge(status)}">${escapeHtml(formatPlain(status))}</span><code>${escapeHtml(formatPlain(data.job_id || data.receipt_db_id || 'stored'))}</code><small>revision ${escapeHtml(formatPlain(revision))}</small></div>`;
}

function renderReviewLoadError(message) {
  currentJobId = null;
  currentReceiptDbId = null;
  currentReviewSaveUrl = null;
  currentReviewSaveMethod = 'POST';
  currentReviewEditable = false;
  currentReviewIdentity = null;
  setActiveTab('review');
  if (reviewSelectionHeaderEl) {
    reviewSelectionHeaderEl.className = 'review-selection-header empty';
    reviewSelectionHeaderEl.textContent = 'Receipt review could not be opened.';
  }
  if (humanReviewPanelEl) {
    humanReviewPanelEl.className = 'review-panel query-error-state';
    humanReviewPanelEl.innerHTML = `<div class="query-terminal-icon" aria-hidden="true">!</div><div><h4>Receipt review could not be opened</h4><p>${escapeHtml(message || 'The review source is unavailable.')}</p></div>`;
  }
}

function applyReviewPayload(data) {
  const receipt = data?.receipt;
  if (!receipt || typeof receipt !== 'object') throw new Error('The server returned no receipt review data.');
  currentJobId = data.job_id || null;
  currentReceiptDbId = data.receipt_id || data.receipt_db_id || null;
  currentArtifacts = data.artifacts || {};
  currentReviewSaveUrl = data.save_url || (currentJobId ? `/api/review/${encodeURIComponent(currentJobId)}` : null);
  currentReviewSaveMethod = String(data.save_method || 'POST').toUpperCase();
  currentReviewEditable = data.editable !== false && Boolean(currentReviewSaveUrl);
  currentReviewIdentity = data.review_identity || null;
  setActiveTab('review');
  renderReviewSelectionHeader(data);
  renderHumanReview(receipt, currentArtifacts, {
    editable: currentReviewEditable,
    readOnlyReason: data.read_only_reason,
    review: data.review,
    source: data.source,
    queueRecord: data.queue_record || data.review_queue,
  });
  renderReviewQueue(currentReviewQueueItems);
}

async function openReviewJob(jobId) {
  if (!jobId) return;
  const res = await fetch(`/api/review/${encodeURIComponent(jobId)}`);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Review load failed');
  applyReviewPayload(data);
}

async function updateQueueStatus(jobId, status) {
  if (!jobId || !status) return;
  const label = status === 'duplicate_confirmed' ? 'confirm this receipt as a duplicate' : 'dismiss the duplicate warning';
  if (!confirm(`Really ${label}?`)) return;
  const res = await fetch(`/api/review-queue/${encodeURIComponent(jobId)}/status`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ status }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Status update failed');
  await loadReviewQueue();
}


function renderReceiptDbList(receipts) {
  if (!receiptDbListEl) return;
  if (!receipts.length) {
    receiptDbListEl.className = 'receipt-summary empty';
    receiptDbListEl.textContent = 'No approved/imported receipts in the analytics database.';
    return;
  }
  const rows = receipts.map((r) => `<tr>
    <td class="numeric">${escapeHtml(formatPlain(r.id))}</td>
    <td>${escapeHtml(formatPlain(r.merchant_name || r.merchant_normalized))}<br><span class="muted">${escapeHtml(formatPlain(r.receipt_date))} ${escapeHtml(formatPlain(r.receipt_time))}</span></td>
    <td class="numeric">${formatMoney(r.grand_total, r.currency || 'EUR')}<br><span class="muted">${escapeHtml(formatPlain(r.item_count))} item(s)</span></td>
    <td>${escapeHtml(formatPlain(r.review_status))}<br><code>${escapeHtml(formatPlain(r.job_id))}</code></td>
    <td>
      <div class="receipt-data-actions">
        <button type="button" class="button-secondary" data-review-receipt-id="${escapeHtml(formatPlain(r.id))}">Review</button>
        <button type="button" class="danger-button" data-delete-receipt-id="${escapeHtml(formatPlain(r.id))}">Delete from DB</button>
      </div>
    </td>
  </tr>`).join('');
  receiptDbListEl.className = 'receipt-summary';
  receiptDbListEl.innerHTML = `<div class="table-scroll"><table class="extracted-table"><thead><tr><th class="numeric">DB ID</th><th>Receipt</th><th class="numeric">Total</th><th>Status/job</th><th>Actions</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  for (const btn of receiptDbListEl.querySelectorAll('[data-review-receipt-id]')) {
    btn.addEventListener('click', () => {
      btn.disabled = true;
      openReceiptFromQuery(btn.getAttribute('data-review-receipt-id'))
        .catch((error) => renderReviewLoadError(error.message || String(error)))
        .finally(() => { btn.disabled = false; });
    });
  }
  for (const btn of receiptDbListEl.querySelectorAll('[data-delete-receipt-id]')) {
    btn.addEventListener('click', () => deleteReceiptFromDb(btn.getAttribute('data-delete-receipt-id')));
  }
}

async function openReceiptFromQuery(receiptId) {
  if (!receiptId) return;
  const res = await fetch(`/api/receipt-db/receipts/${encodeURIComponent(receiptId)}/review`);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Receipt is no longer available.');
  applyReviewPayload(data);
}

async function refreshReceiptDbList() {
  if (!receiptDbListEl) return;
  receiptDbListEl.className = 'receipt-summary empty';
  receiptDbListEl.textContent = 'Loading approved/imported receipts...';
  const res = await fetch('/api/receipt-db/receipts?limit=300');
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Receipt list failed');
  renderReceiptDbList(data.receipts || []);
}

async function deleteReceiptFromDb(receiptId) {
  if (!receiptId) return;
  if (!confirm(`Delete receipt DB record ${receiptId}? This removes it from Ask Your Receipts/RAG analytics.`)) return;
  const res = await fetch(`/api/receipt-db/receipts/${encodeURIComponent(receiptId)}`, { method: 'DELETE' });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Delete receipt failed');
  await refreshReceiptDbSummary();
  await refreshReceiptDbList();
}

async function deleteAllReceiptsFromDb() {
  const typed = prompt('Type DELETE_ALL_RECEIPTS to remove all approved/imported receipt records from the analytics DB. Review queue entries are kept.');
  if (typed !== 'DELETE_ALL_RECEIPTS') return;
  const includeReviewQueue = confirm('Also clear the Review Queue? Choose Cancel to keep pending/rejected/duplicate review tasks.');
  const res = await fetch('/api/receipt-db/delete-all', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ confirm: 'DELETE_ALL_RECEIPTS', include_review_queue: includeReviewQueue }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Delete all failed');
  await refreshReceiptDbSummary();
  await refreshReceiptDbList();
  if (includeReviewQueue) await loadReviewQueue();
}

function renderReceiptChartSpec(chart) {
  if (!chart || !Array.isArray(chart.data) || !chart.data.length) return '';
  const rows = chart.data
    .map((row) => ({ label: formatPlain(row.label), value: Number(row.value || 0) }))
    .filter((row) => Number.isFinite(row.value));
  if (!rows.length) return '';

  const currency = chart.currency || '';
  const metric = chart.metric || '';
  const formattedValue = (value) => {
    if (metric === 'item_count' || metric === 'receipt_count') return String(Math.round(value));
    if (metric === 'percentage_change') return `${value.toFixed(2)}%`;
    return currency ? formatMoney(value, currency) : value.toFixed(2);
  };

  if (chart.chart_type === 'line' && rows.length > 1) {
    const width = 760;
    const height = 250;
    const padX = 52;
    const padY = 30;
    const values = rows.map((row) => row.value);
    const min = Math.min(...values, 0);
    const max = Math.max(...values, 0);
    const span = max - min || 1;
    const points = rows.map((row, index) => {
      const x = padX + (index * (width - 2 * padX)) / Math.max(1, rows.length - 1);
      const y = height - padY - ((row.value - min) / span) * (height - 2 * padY);
      return { ...row, x, y };
    });
    const polyline = points.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(' ');
    const pointMarkup = points.map((point) => `
      <circle cx="${point.x.toFixed(1)}" cy="${point.y.toFixed(1)}" r="4"></circle>
      <text x="${point.x.toFixed(1)}" y="${height - 8}" text-anchor="middle">${escapeHtml(point.label)}</text>
      <title>${escapeHtml(point.label)}: ${escapeHtml(formattedValue(point.value))}</title>
    `).join('');
    return `
      <div class="receipt-chart-card">
        <div class="receipt-section-title"><h4>${escapeHtml(formatPlain(chart.title || 'Chart'))}</h4><small>${escapeHtml(formatPlain(chart.y_label || metric))}</small></div>
        <div class="receipt-line-chart-scroll">
          <svg class="receipt-line-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(formatPlain(chart.title || 'Receipt chart'))}">
            <line x1="${padX}" y1="${height - padY}" x2="${width - padX}" y2="${height - padY}"></line>
            <polyline points="${polyline}"></polyline>
            ${pointMarkup}
          </svg>
        </div>
      </div>`;
  }

  const maxAbs = Math.max(...rows.map((row) => Math.abs(row.value)), 1);
  const bars = rows.map((row) => {
    const pct = Math.max(1, Math.round((Math.abs(row.value) / maxAbs) * 100));
    return `<div class="receipt-chart-row">
      <div class="receipt-chart-label">${escapeHtml(row.label)}</div>
      <div class="receipt-chart-track"><div class="receipt-chart-bar ${row.value < 0 ? 'negative' : ''}" style="width:${pct}%"></div></div>
      <div class="receipt-chart-value">${escapeHtml(formattedValue(row.value))}</div>
    </div>`;
  }).join('');
  return `
    <div class="receipt-chart-card">
      <div class="receipt-section-title"><h4>${escapeHtml(formatPlain(chart.title || 'Chart'))}</h4><small>${escapeHtml(formatPlain(chart.chart_type || 'bar'))}</small></div>
      <div class="receipt-bar-chart">${bars}</div>
    </div>`;
}

function formatQueryCell(column, row) {
  const value = row?.[column];
  if (value === null || value === undefined || value === '') return '—';
  const normalizedColumn = String(column || '').toLowerCase();
  if (normalizedColumn.includes('date')) return formatLocalizedDate(value);
  if (typeof value === 'number') {
    if (row?.currency && /(value|total|amount|price|spend|cost)/.test(normalizedColumn)) {
      return formatLocalizedMoney(value, row.currency);
    }
    return new Intl.NumberFormat('de-DE', { maximumFractionDigits: 2 }).format(value);
  }
  return formatPlain(value);
}

function renderQueryGenericTable(columns, rows, title = 'Query result') {
  if (!columns.length) return '';
  const header = columns.map((column) => `<th>${escapeHtml(formatPlain(column).replaceAll('_', ' '))}</th>`).join('');
  const body = rows.length ? rows.map((row) => `<tr>${columns.map((column) => {
    const numericClass = typeof row?.[column] === 'number' ? ' class="numeric"' : '';
    return `<td${numericClass}>${escapeHtml(formatQueryCell(column, row))}</td>`;
  }).join('')}</tr>`).join('') : `<tr><td colspan="${Math.max(1, columns.length)}" class="muted">No rows returned.</td></tr>`;
  return `
    <section class="query-data-section">
      <div class="receipt-section-title"><h4>${escapeHtml(title)}</h4><small>${rows.length} row(s)</small></div>
      <div class="table-scroll"><table class="extracted-table query-generic-table"><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table></div>
    </section>`;
}

function renderQueryReceiptCards(rows) {
  if (!rows.length) return '';
  return `
    <section class="query-data-section">
      <div class="receipt-section-title"><h4>Matching receipts</h4><small>${rows.length} approved receipt(s)</small></div>
      <div class="query-receipt-grid">
        ${rows.map((row) => `
          <article class="query-receipt-card">
            <div class="query-receipt-card-header">
              <div>
                <span class="query-card-label">Merchant</span>
                <h5>${escapeHtml(formatPlain(row.merchant || row.merchant_name || 'Unknown merchant'))}</h5>
              </div>
              <span class="badge neutral">#${escapeHtml(formatPlain(row.receipt_id))}</span>
            </div>
            <dl class="query-card-facts">
              <div><dt>Date</dt><dd>${escapeHtml(formatLocalizedDate(row.receipt_date))}</dd></div>
              <div><dt>Total</dt><dd>${escapeHtml(formatLocalizedMoney(row.grand_total, row.currency || 'EUR'))}</dd></div>
            </dl>
            <button type="button" class="button-secondary query-open-receipt" data-open-receipt-id="${escapeHtml(formatPlain(row.receipt_id))}">Open receipt</button>
          </article>`).join('')}
      </div>
    </section>`;
}

function renderQueryMetrics(columns, rows, resultEntity) {
  if (!rows.length) return '';
  const hasCurrencyValue = columns.includes('currency') && columns.includes('value');
  if (hasCurrencyValue) {
    return `<div class="query-metric-grid">${rows.map((row) => `
      <div class="query-metric-card">
        <span>${escapeHtml(formatPlain(resultEntity || 'Result').replaceAll('_', ' '))}</span>
        <strong>${escapeHtml(formatLocalizedMoney(row.value, row.currency || 'EUR'))}</strong>
        <small>${escapeHtml(formatPlain(row.currency || ''))}</small>
      </div>`).join('')}</div>`;
  }
  if (columns.length === 1 && columns[0] === 'value') {
    const value = rows[0]?.value;
    return `<div class="query-metric-grid"><div class="query-metric-card">
      <span>${escapeHtml(formatPlain(resultEntity || 'Result').replaceAll('_', ' '))}</span>
      <strong>${escapeHtml(formatPlain(value))}</strong>
    </div></div>`;
  }
  return '';
}

function renderRagDiagnostics(diagnostics) {
  const stages = Array.isArray(diagnostics.stages) ? diagnostics.stages : [];
  const resolvedEntities = Array.isArray(diagnostics.resolved_entities) ? diagnostics.resolved_entities : [];
  const validatedSql = diagnostics.validated_sql || null;
  const ollama = diagnostics.model_call_summary || diagnostics.ollama_summary || {};
  const stageRows = stages.map((stage, index) => `<tr>
    <td class="numeric">${index + 1}</td>
    <td><code>${escapeHtml(formatPlain(stage.name))}</code></td>
    <td>${escapeHtml(formatPlain(stage.status))}</td>
    <td class="numeric">${escapeHtml(formatPlain(asNumber(stage.duration_ms)?.toFixed(1) ?? stage.duration_ms))}</td>
    <td>${escapeHtml(formatPlain(stage.error || stage.model || '—'))}</td>
  </tr>`).join('');
  const entityRows = resolvedEntities.map((entity) => `<tr>
    <td><code>${escapeHtml(formatPlain(entity.entity_id))}</code></td>
    <td>${escapeHtml(formatPlain(entity.search_text))}</td>
    <td>${escapeHtml(formatPlain(entity.status))}</td>
    <td><code>${escapeHtml(JSON.stringify(entity.selected_item_ids || []))}</code></td>
    <td><code>${escapeHtml(JSON.stringify(entity.uncertain_item_ids || []))}</code></td>
  </tr>`).join('');

  return `
    <details class="query-diagnostics-panel">
      <summary>
        <span>Technical details</span>
        <small>Stages, resolved entities, validated SQL, and model timing</small>
      </summary>
      <div class="query-diagnostics-content">
        <div class="diagnostic-metric-grid">
          <div><span>Total duration</span><strong>${escapeHtml(formatPlain(asNumber(diagnostics.duration_ms)?.toFixed(0)))} ms</strong></div>
          <div><span>Model calls</span><strong>${escapeHtml(formatPlain(ollama.call_count ?? '—'))}</strong></div>
          <div><span>Model loading</span><strong>${escapeHtml(formatPlain(asNumber(ollama.total_load_duration_ms)?.toFixed(0)))} ms</strong></div>
          <div><span>SQL execution</span><strong>${escapeHtml(formatPlain(stages.find((stage) => stage.name === 'execute_sql')?.duration_ms ?? '—'))} ms</strong></div>
        </div>
        ${stages.length ? `<details open><summary>Execution stages</summary><div class="table-scroll"><table class="extracted-table"><thead><tr><th>#</th><th>Stage</th><th>Status</th><th>Duration ms</th><th>Model/error</th></tr></thead><tbody>${stageRows}</tbody></table></div></details>` : ''}
        ${resolvedEntities.length ? `<details><summary>Resolved product entities</summary><div class="table-scroll"><table class="extracted-table"><thead><tr><th>ID</th><th>Search text</th><th>Status</th><th>Selected IDs</th><th>Uncertain IDs</th></tr></thead><tbody>${entityRows}</tbody></table></div></details>` : ''}
        ${validatedSql ? `<details><summary>Validated read-only SQL</summary><pre>${escapeHtml(formatPlain(validatedSql.sql))}</pre><pre>${escapeHtml(JSON.stringify(validatedSql.parameters || {}, null, 2))}</pre></details>` : ''}
        <details><summary>Raw diagnostic JSON</summary><pre>${escapeHtml(JSON.stringify(diagnostics, null, 2))}</pre></details>
      </div>
    </details>`;
}

const ASK_QUERY_ERROR_MESSAGES = {
  missing_question: 'Enter a question about your approved receipts.',
  embedding_unavailable: 'Semantic receipt search is currently unavailable.',
  question_analysis_failed: 'The question could not be interpreted reliably.',
  candidate_resolution_failed: 'Matching receipt products could not be resolved.',
  sql_planning_failed: 'A database query could not be created for this question.',
  sql_validation_failed: 'The generated query did not pass the safety checks.',
  sql_execution_failed: 'The validated query could not be executed.',
  invalid_server_response: 'The server returned an unreadable query response.',
  network_error: 'The receipt query service could not be reached.',
  unsupported_request_field: 'The request contains an unsupported field.',
  query_execution_failed: 'The receipt query could not be completed.',
};

function queryStateBanner(status) {
  return `<div class="query-engine-banner">
    <strong>Executed with ${escapeHtml(queryEngineLabel())}</strong>
    <span class="${queryStatusBadgeClass(status)}">${escapeHtml(queryStatusLabel(status))}</span>
  </div>`;
}

function queryStateAction(label = 'Edit question') {
  return `<div class="query-state-actions">
    <button type="button" class="button-secondary" data-query-action="focus-input">${escapeHtml(label)}</button>
  </div>`;
}

function renderDiagnosticLogNotice(data) {
  const log = data?.diagnostic_log;
  if (!log || !log.enabled) return '';
  if (log.saved && log.filename) {
    return `<div class="query-log-notice query-log-saved">Diagnostic JSON saved as <code>${escapeHtml(log.filename)}</code>.</div>`;
  }
  return '<div class="query-log-notice query-log-failed">Diagnostic logging was enabled, but the JSON file could not be saved.</div>';
}

function renderAskNotFoundState(data) {
  const diagnostics = data.diagnostics && typeof data.diagnostics === 'object' ? data.diagnostics : {};
  const message = data.answer || 'No approved receipt data matched the request.';
  askReceiptsResultEl.className = 'query-state-card query-state-not_found';
  askReceiptsResultEl.innerHTML = `
    ${queryStateBanner('not_found')}
    <div class="query-state-content">
      <div class="query-state-icon" aria-hidden="true">0</div>
      <div>
        <h4>No matching receipt data found</h4>
        <p>${escapeHtml(formatPlain(message))}</p>
        <small>Try a product name, merchant, date, or a broader description. The query used the approved RAG-SQL engine.</small>
      </div>
    </div>
    ${queryStateAction('Edit question')}
    ${renderDiagnosticLogNotice(data)}
    ${Object.keys(diagnostics).length ? renderRagDiagnostics(diagnostics) : ''}`;
  setAskResultStatus('not_found');
}

function renderAskInsufficientInfoState(data) {
  const diagnostics = data.diagnostics && typeof data.diagnostics === 'object' ? data.diagnostics : {};
  const message = data.answer || 'The reviewed receipt data does not contain enough product information for this answer.';
  askReceiptsResultEl.className = 'query-state-card query-state-insufficient_info';
  askReceiptsResultEl.innerHTML = `
    ${queryStateBanner('insufficient_info')}
    <div class="query-state-content">
      <div class="query-state-icon" aria-hidden="true">i</div>
      <div>
        <h4>Not enough reviewed product information</h4>
        <p>${escapeHtml(formatPlain(message))}</p>
        <small>Add or correct the item semantic description/category reason in Receipt Data, then save the review.</small>
      </div>
    </div>
    ${queryStateAction('Edit question')}
    ${renderDiagnosticLogNotice(data)}
    ${Object.keys(diagnostics).length ? renderRagDiagnostics(diagnostics) : ''}`;
  setAskResultStatus('insufficient_info');
}

function renderAskErrorState(data) {
  const diagnostics = data.diagnostics && typeof data.diagnostics === 'object' ? data.diagnostics : {};
  const errorCode = String(data.error_code || 'query_execution_failed');
  const message = data.answer || ASK_QUERY_ERROR_MESSAGES[errorCode] || 'The receipt query could not be completed.';
  const technicalError = data.error ? formatPlain(data.error) : 'No additional technical details were returned.';
  askReceiptsResultEl.className = 'query-state-card query-state-error';
  askReceiptsResultEl.innerHTML = `
    ${queryStateBanner('error')}
    <div class="query-state-content">
      <div class="query-state-icon" aria-hidden="true">!</div>
      <div>
        <h4>The query could not be completed</h4>
        <p>${escapeHtml(formatPlain(message))}</p>
        <small>You can revise the question and try again.</small>
      </div>
    </div>
    ${queryStateAction('Revise question')}
    ${renderDiagnosticLogNotice(data)}
    <details class="query-error-details">
      <summary>Technical error</summary>
      <div class="query-error-code"><span>Error code</span><code>${escapeHtml(errorCode)}</code></div>
      <pre>${escapeHtml(technicalError)}</pre>
    </details>
    ${Object.keys(diagnostics).length ? renderRagDiagnostics(diagnostics) : ''}`;
  setAskResultStatus('error');
}

function renderAskClarificationState(data) {
  const diagnostics = data.diagnostics && typeof data.diagnostics === 'object' ? data.diagnostics : {};
  const message = data.clarification_question || data.answer || 'Please reformulate the question with the intended scope.';
  askReceiptsResultEl.className = 'query-state-card query-state-needs_clarification';
  askReceiptsResultEl.innerHTML = `
    ${queryStateBanner('needs_clarification')}
    <div class="query-state-content">
      <div class="query-state-icon" aria-hidden="true">?</div>
      <div>
        <h4>More information is needed</h4>
        <p>${escapeHtml(formatPlain(message))}</p>
        <small>Reformulate the question with the intended scope and submit it again.</small>
      </div>
    </div>
    ${queryStateAction('Clarify question')}
    ${renderDiagnosticLogNotice(data)}
    ${Object.keys(diagnostics).length ? renderRagDiagnostics(diagnostics) : ''}`;
  setAskResultStatus('needs_clarification');
}

function renderQueryTerminalState(data) {
  const status = data.status || 'error';
  if (status === 'not_found') {
    renderAskNotFoundState(data);
    return;
  }
  if (status === 'needs_clarification') {
    renderAskClarificationState(data);
    return;
  }
  if (status === 'insufficient_info') {
    renderAskInsufficientInfoState(data);
    return;
  }
  renderAskErrorState({ ...data, status: 'error' });
}

function renderRagSqlResult(data) {
  if (!askReceiptsResultEl) return;
  const status = data.status || data.execution?.status || 'completed';
  if (status !== 'completed') {
    renderQueryTerminalState(data);
    return;
  }

  const executionData = data.data && typeof data.data === 'object' ? data.data : {};
  const columns = Array.isArray(executionData.columns) ? executionData.columns : [];
  const rows = Array.isArray(executionData.rows) ? executionData.rows : [];
  const diagnostics = data.diagnostics && typeof data.diagnostics === 'object' ? data.diagnostics : {};
  const resultEntity = diagnostics.validated_sql?.result_entity || diagnostics.sql_plan?.result_entity || '';
  const isReceiptResult = resultEntity === 'receipt' || (
    !resultEntity
    && columns.includes('receipt_id')
    && columns.includes('grand_total')
    && columns.some((column) => ['merchant', 'merchant_name'].includes(column))
  );
  const metrics = renderQueryMetrics(columns, rows, resultEntity);
  const structuredResult = isReceiptResult
    ? renderQueryReceiptCards(rows)
    : (metrics || renderQueryGenericTable(columns, rows));

  askReceiptsResultEl.className = 'query-completed-result';
  askReceiptsResultEl.innerHTML = `
    <div class="query-engine-banner">
      <strong>Executed with ${escapeHtml(queryEngineLabel())}</strong>
      <span class="badge ok">Completed</span>
    </div>
    <section class="query-answer-card">
      <span class="query-answer-label">Answer</span>
      <p>${escapeHtml(formatPlain(data.answer || 'The query completed successfully.'))}</p>
    </section>
    ${structuredResult}
    ${renderDiagnosticLogNotice(data)}
    ${renderRagDiagnostics(diagnostics)}`;
  setAskResultStatus('completed');
}

function renderAskReceiptsResult(data) {
  if (!askReceiptsResultEl) return;
  if (!data || typeof data !== 'object') {
    askReceiptsResultEl.className = 'query-empty-state';
    askReceiptsResultEl.innerHTML = '<div class="empty-state-icon" aria-hidden="true">?</div><strong>Ask a question about your approved receipts</strong><span>Results and diagnostics will appear here.</span>';
    setAskResultStatus('ready');
    return;
  }
  renderRagSqlResult(data);
}

function isCompletedJobState(state) {
  return state === 'completed' || state === 'done';
}

function isFailedJobState(state) {
  return state === 'failed' || state === 'error';
}

function isTerminalJobState(state) {
  return isCompletedJobState(state) || isFailedJobState(state);
}

function stateClass(state) {
  if (isCompletedJobState(state)) return 'badge ok';
  if (isFailedJobState(state)) return 'badge bad';
  return 'badge neutral';
}

function updateProgress(job) {
  const events = job.events || [];
  let pct = 0;
  if (job.state === 'queued') pct = 8;
  else if (job.state === 'running') pct = Math.min(90, 20 + events.length * 12);
  else if (isTerminalJobState(job.state)) pct = 100;
  if (progressFillEl) progressFillEl.style.width = `${pct}%`;
}

function renderEvents(events) {
  eventsEl.innerHTML = '';
  for (const ev of events || []) {
    const li = document.createElement('li');
    const status = ev.status || '';
    li.className = `event ${escapeHtml(status)}`;
    const details = ev.details ? `<small>${escapeHtml(JSON.stringify(ev.details))}</small>` : '';
    li.innerHTML = `
      <span class="event-time">${escapeHtml(ev.time || '')}</span>
      <span class="event-status">${escapeHtml(status)}</span>
      <strong>${escapeHtml(ev.stage || 'pipeline')}</strong>
      <span>${escapeHtml(ev.message || '')}</span>
      ${details}
    `;
    eventsEl.appendChild(li);
  }
}

function renderArtifacts(artifacts) {
  artifactsEl.innerHTML = '';
  if (!artifacts) return;
  for (const [name, url] of Object.entries(artifacts)) {
    const a = document.createElement('a');
    a.href = url;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.textContent = name;
    artifactsEl.appendChild(a);
  }
}

function renderReviewHandoff(receipt = null) {
  if (!reviewHandoffPanelEl) return;
  if (!receipt || !currentJobId) {
    reviewHandoffPanelEl.className = 'review-handoff empty';
    reviewHandoffPanelEl.textContent = 'Review is performed only in the dedicated Review tab. Completed receipts appear in its canonical queue.';
    return;
  }
  const status = receipt?.validation?.import_decision || receipt?.parse_status || 'needs_review';
  reviewHandoffPanelEl.className = 'review-handoff';
  reviewHandoffPanelEl.innerHTML = `<div><strong>Receipt added to the review queue</strong><span>Status: ${escapeHtml(formatPlain(status))}. Editing is intentionally disabled on the Run tab.</span></div><button type="button" id="openCurrentReviewButton">Open in Review</button>`;
  document.getElementById('openCurrentReviewButton')?.addEventListener('click', () => {
    openReviewJob(currentJobId).catch((error) => renderReviewLoadError(error.message || String(error)));
  });
}

async function fetchPreview(artifacts) {
  previewEl.textContent = '';
  currentArtifacts = artifacts || {};
  renderReceiptSummary(null);
  renderReviewHandoff(null);
  if (!artifacts || !artifacts.final_receipt) return;
  try {
    const res = await fetch(artifacts.final_receipt);
    const data = await res.json();
    previewEl.textContent = JSON.stringify(data, null, 2);
    renderReceiptSummary(data);
    renderReviewHandoff(data);
  } catch (e) {
    previewEl.textContent = String(e);
    renderReceiptSummary(null);
    renderReviewHandoff(null);
  }
}

async function poll(jobId) {
  const res = await fetch(`/api/status/${jobId}`);
  const job = await res.json();
  if (!res.ok) throw new Error(job.error || `Status request failed: ${res.status}`);

  stateEl.textContent = job.state || 'unknown';
  stateEl.className = stateClass(job.state);
  updateProgress(job);
  renderEvents(job.events);

  if (job.result) {
    const report = job.result.report || {};
    summaryEl.textContent = `Decision: ${report.import_decision || 'n/a'} | Balanced: ${report.balanced} | Difference: ${report.difference ?? 'n/a'} | Issues: ${(report.issues || []).length}`;
    renderArtifacts(job.result.artifacts);
    await fetchPreview(job.result.artifacts);
  } else if (job.error) {
    summaryEl.textContent = job.error.message || 'Error';
  } else {
    const last = (job.events || []).at(-1);
    summaryEl.textContent = last ? last.message : 'Waiting for job...';
  }

  if (isTerminalJobState(job.state)) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}


function collectParserOptionsForBatch() {
  const data = new FormData();
  for (const el of uploadForm.querySelectorAll('input[name], select[name]')) {
    if (el.name === 'file') continue;
    if (el.type === 'checkbox') {
      if (el.checked) data.append(el.name, 'on');
      continue;
    }
    data.append(el.name, el.value ?? '');
  }
  return data;
}

function updateBatchProgress(job) {
  const total = Number(job.total || 0);
  const completed = Number(job.completed || 0);
  let pct = 0;
  if (job.state === 'queued') pct = 5;
  else if (job.state === 'running') pct = total > 0 ? Math.max(5, Math.min(95, (completed / total) * 100)) : 20;
  else if (isTerminalJobState(job.state)) pct = 100;
  if (batchProgressFillEl) batchProgressFillEl.style.width = `${pct}%`;
}

function renderBatchEvents(events) {
  if (!batchEventsEl) return;
  batchEventsEl.innerHTML = '';
  for (const ev of events || []) {
    const li = document.createElement('li');
    const status = ev.status || '';
    li.className = `event ${escapeHtml(status)}`;
    const details = ev.details ? `<small>${escapeHtml(JSON.stringify(ev.details))}</small>` : '';
    li.innerHTML = `
      <span class="event-time">${escapeHtml(ev.time || '')}</span>
      <span class="event-status">${escapeHtml(status)}</span>
      <strong>${escapeHtml(ev.stage || 'batch')}</strong>
      <span>${escapeHtml(ev.message || '')}</span>
      ${details}
    `;
    batchEventsEl.appendChild(li);
  }
}

function renderBatchArtifacts(artifacts) {
  if (!batchArtifactsEl) return;
  batchArtifactsEl.innerHTML = '';
  if (!artifacts) return;
  for (const [name, url] of Object.entries(artifacts)) {
    const a = document.createElement('a');
    a.href = url;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.textContent = name;
    batchArtifactsEl.appendChild(a);
  }
}

function renderBatchSummary(job) {
  if (!batchSummaryEl) return;
  const result = job.result || {};
  const items = Array.isArray(job.items) ? job.items : (Array.isArray(result.items) ? result.items : []);
  const rows = items.length ? items.map((item, idx) => {
    const decision = item.decision || item.state || 'n/a';
    const cls = decision === 'import' ? 'ok-cell' : (decision === 'reject' || decision === 'llm_failed' || isFailedJobState(item.state) ? 'bad-cell' : 'muted');
    const finalLink = item.final_receipt ? `<a href="${escapeHtml(item.final_receipt)}" target="_blank" rel="noopener noreferrer">final JSON</a>` : '—';
    const reportLink = item.validation_report ? `<a href="${escapeHtml(item.validation_report)}" target="_blank" rel="noopener noreferrer">report</a>` : '—';
    return `<tr>
      <td class="numeric">${idx + 1}</td>
      <td>${escapeHtml(formatPlain(item.filename))}</td>
      <td class="${cls}">${escapeHtml(formatPlain(decision))}</td>
      <td>${escapeHtml(formatPlain(item.balanced))}</td>
      <td class="numeric">${escapeHtml(formatPlain(item.difference))}</td>
      <td class="numeric">${escapeHtml(formatPlain(item.issue_count))}</td>
      <td><code>${escapeHtml(formatPlain(item.child_job_id))}</code></td>
      <td>${finalLink} ${reportLink}</td>
    </tr>`;
  }).join('') : `<tr><td colspan="8" class="muted">No batch items have finished yet.</td></tr>`;

  batchSummaryEl.className = 'receipt-summary';
  batchSummaryEl.innerHTML = `
    <div class="summary-grid">
      <div class="summary-card">
        <h4>Batch</h4>
        <table class="kv-table"><tbody>${kvRows([
          ['Folder', escapeHtml(formatPlain(job.folder_path))],
          ['Total', escapeHtml(formatPlain(job.total))],
          ['Completed', escapeHtml(formatPlain(job.completed))],
          ['Failed/rejected', escapeHtml(formatPlain(job.failed))],
          ['Recursive', escapeHtml(formatPlain(job.recursive))],
        ])}</tbody></table>
      </div>
    </div>
    <div class="table-scroll"><table class="extracted-table batch-table"><thead><tr>
      <th class="numeric">#</th><th>File</th><th>Decision</th><th>Balanced</th><th class="numeric">Diff.</th><th class="numeric">Issues</th><th>Job</th><th>Artifacts</th>
    </tr></thead><tbody>${rows}</tbody></table></div>
  `;
}

async function pollBatch(batchId) {
  const res = await fetch(`/api/status/${batchId}`);
  const job = await res.json();
  if (!res.ok) throw new Error(job.error || `Batch status request failed: ${res.status}`);

  batchStateEl.textContent = job.state || 'unknown';
  batchStateEl.className = stateClass(job.state);
  updateBatchProgress(job);
  renderBatchEvents(job.events);
  renderBatchSummary(job);

  const artifacts = (job.result || {}).artifacts || null;
  renderBatchArtifacts(artifacts);
  if (job.error) {
    batchSummaryStatusEl.textContent = job.error.message || 'Batch error';
  } else {
    batchSummaryStatusEl.textContent = `Batch: ${job.completed || 0}/${job.total || 0} completed | Failed/rejected: ${job.failed || 0}`;
  }

  if (isTerminalJobState(job.state)) {
    clearInterval(batchPollTimer);
    batchPollTimer = null;
  }
}

if (batchForm) {
  batchForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const submitButton = document.getElementById('batch-run-button') || batchForm.querySelector('button');
    submitButton.disabled = true;
    renderBatchArtifacts(null);
    if (batchSummaryEl) {
      batchSummaryEl.className = 'receipt-summary empty';
      batchSummaryEl.textContent = 'Starting batch...';
    }
    if (batchSummaryStatusEl) batchSummaryStatusEl.textContent = 'Submitting batch...';
    if (batchProgressFillEl) batchProgressFillEl.style.width = '5%';

    try {
      const data = collectParserOptionsForBatch();
      const folderPath = document.getElementById('batch_folder_path')?.value || '';
      const maxFiles = document.getElementById('batch_max_files')?.value || '';
      const recursive = document.getElementById('batch_recursive')?.checked;
      data.append('folder_path', folderPath);
      data.append('max_files', maxFiles);
      if (recursive) data.append('recursive', 'on');

      const res = await fetch('/api/batch/start', { method: 'POST', body: data });
      const payload = await res.json();
      if (!res.ok) throw new Error(payload.error || 'Batch start failed');

      batchStatusCard.classList.remove('hidden');
      batchStatusCard.hidden = false;
      batchJobIdEl.textContent = payload.batch_id || payload.job_id;
      batchStateEl.textContent = 'queued';
      batchStateEl.className = 'badge neutral';

      await pollBatch(payload.batch_id || payload.job_id);
      batchPollTimer = setInterval(() => {
        pollBatch(payload.batch_id || payload.job_id).catch((err) => {
          batchSummaryStatusEl.textContent = err.message || String(err);
          clearInterval(batchPollTimer);
          batchPollTimer = null;
        });
      }, 1500);
    } catch (err) {
      batchSummaryStatusEl.textContent = err.message || String(err);
      alert(err.message || String(err));
    } finally {
      submitButton.disabled = false;
    }
  });
}

if (refreshReceiptDbButton) {
  refreshReceiptDbButton.addEventListener('click', () => {
    refreshReceiptDbSummary().catch((err) => {
      if (receiptDbSummaryEl) {
        receiptDbSummaryEl.className = 'receipt-summary empty';
        receiptDbSummaryEl.textContent = err.message || String(err);
      }
    });
  });
}

if (refreshReviewQueueButton) {
  refreshReviewQueueButton.addEventListener('click', () => {
    loadReviewQueue().catch((err) => {
      if (reviewQueueResultEl) {
        reviewQueueResultEl.className = 'review-queue-list empty';
        reviewQueueResultEl.textContent = err.message || String(err);
      }
    });
  });
}

if (reviewQueueFilterEl) {
  reviewQueueFilterEl.addEventListener('change', () => {
    loadReviewQueue().catch(() => {});
  });
}

if (reviewQueueSearchEl) {
  reviewQueueSearchEl.addEventListener('input', () => renderReviewQueue(currentReviewQueueItems));
}

if (reviewQueueSummaryEl) {
  for (const button of reviewQueueSummaryEl.querySelectorAll('[data-review-summary-filter]')) {
    button.addEventListener('click', () => {
      if (reviewQueueFilterEl) reviewQueueFilterEl.value = button.dataset.reviewSummaryFilter || 'all';
      loadReviewQueue().catch(() => {});
    });
  }
}

document.addEventListener('keydown', (event) => {
  const reviewPanel = document.querySelector('[data-tab-panel="review"]');
  if (!reviewPanel || reviewPanel.hidden || !currentReviewEditable) return;
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
    event.preventDefault();
    saveHumanReview('needs_review', false);
  }
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
    event.preventDefault();
    saveHumanReview('approved', true);
  }
});

if (refreshReceiptListButton) {
  refreshReceiptListButton.addEventListener('click', () => {
    refreshReceiptDbList().catch((err) => {
      if (receiptDbListEl) {
        receiptDbListEl.className = 'receipt-summary empty';
        receiptDbListEl.textContent = err.message || String(err);
      }
    });
  });
}

if (deleteAllReceiptsButton) {
  deleteAllReceiptsButton.addEventListener('click', () => {
    deleteAllReceiptsFromDb().catch((err) => alert(err.message || String(err)));
  });
}

for (const button of askQueryExampleButtons) {
  button.addEventListener('click', () => {
    if (!askReceiptsQuestionEl) return;
    askReceiptsQuestionEl.value = button.dataset.queryExample || '';
    askReceiptsQuestionEl.focus();
  });
}

if (askReceiptsQuestionEl) {
  askReceiptsQuestionEl.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      if (!event.repeat) askReceiptsForm?.requestSubmit();
    }
  });
}

if (askReceiptsResultEl) {
  askReceiptsResultEl.addEventListener('click', (event) => {
    const receiptButton = event.target.closest('[data-open-receipt-id]');
    if (receiptButton) {
      const receiptId = receiptButton.getAttribute('data-open-receipt-id');
      receiptButton.disabled = true;
      openReceiptFromQuery(receiptId)
        .catch((error) => renderReviewLoadError(error.message || String(error)))
        .finally(() => { receiptButton.disabled = false; });
      return;
    }

    const queryAction = event.target.closest('[data-query-action]');
    if (queryAction?.getAttribute('data-query-action') === 'focus-input') {
      askReceiptsQuestionEl?.focus();
      askReceiptsQuestionEl?.select();
    }
  });
}

if (askReceiptsForm) {
  askReceiptsForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const question = askReceiptsQuestionEl?.value || '';
    if (!question.trim()) {
      renderQueryTerminalState({
        strategy: 'rag_sql',
        status: 'error',
        error_code: 'missing_question',
        answer: 'Enter a question about your approved receipts.',
        error: 'Missing question.',
      });
      askReceiptsQuestionEl?.focus();
      return;
    }

    setAskResultStatus('running');
    if (askReceiptsResultEl) {
      askReceiptsResultEl.className = 'query-empty-state query-loading-placeholder';
      askReceiptsResultEl.innerHTML = '<div class="empty-state-icon" aria-hidden="true">…</div><strong>Preparing your result</strong><span>RAG-SQL is resolving evidence and validating a read-only query.</span>';
    }
    setQueryLoadingState(true);

    try {
      const res = await fetch('/api/ask-receipts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: question.trim(),
          limit: 25,
          save_json_log: Boolean(askReceiptsSaveJsonLogEl?.checked),
        }),
      });
      let data;
      try {
        data = await res.json();
      } catch (_) {
        data = {
          strategy: 'rag_sql',
          status: 'error',
          error_code: 'invalid_server_response',
          answer: ASK_QUERY_ERROR_MESSAGES.invalid_server_response,
          error: `HTTP ${res.status}`,
        };
      }
      if (!res.ok && (!data.status || data.status === 'completed')) {
        data = {
          ...data,
          strategy: 'rag_sql',
          status: 'error',
          error_code: data.error_code || 'query_execution_failed',
          answer: data.answer || ASK_QUERY_ERROR_MESSAGES.query_execution_failed,
          error: data.error || `HTTP ${res.status}`,
        };
      }
      renderAskReceiptsResult(data);
    } catch (error) {
      renderQueryTerminalState({
        strategy: 'rag_sql',
        status: 'error',
        error_code: 'network_error',
        answer: ASK_QUERY_ERROR_MESSAGES.network_error,
        error: error.message || String(error),
      });
    } finally {
      setQueryLoadingState(false);
    }
  });
}


form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const submitButton = form.querySelector('button');
  submitButton.disabled = true;
  previewEl.textContent = '';
  artifactsEl.innerHTML = '';
  renderReceiptSummary(null);
  renderReviewHandoff(null);
  currentJobId = null;
  currentReceiptDbId = null;
  currentArtifacts = {};
  currentReviewSaveUrl = null;
  currentReviewSaveMethod = 'POST';
  currentReviewEditable = true;
  currentReviewIdentity = null;
  summaryEl.textContent = 'Uploading image...';
  if (progressFillEl) progressFillEl.style.width = '5%';

  try {
    const data = new FormData(form);
    const res = await fetch('/api/upload', { method: 'POST', body: data });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.error || 'Upload failed');

    statusCard.classList.remove('hidden');
    statusCard.hidden = false;
    jobIdEl.textContent = payload.job_id;
    currentJobId = payload.job_id;
    currentReceiptDbId = null;
    currentReviewSaveUrl = `/api/review/${encodeURIComponent(payload.job_id)}`;
    currentReviewSaveMethod = 'POST';
    currentReviewEditable = true;
    stateEl.textContent = 'queued';
    stateEl.className = 'badge neutral';

    await poll(payload.job_id);
    pollTimer = setInterval(() => {
      poll(payload.job_id).catch((err) => {
        summaryEl.textContent = err.message || String(err);
        clearInterval(pollTimer);
        pollTimer = null;
      });
    }, 1200);
  } catch (err) {
    summaryEl.textContent = err.message || String(err);
    alert(err.message || String(err));
  } finally {
    submitButton.disabled = false;
  }
});

const modelDashboardRefreshButton = document.getElementById('refresh-model-dashboard');
const modelPricingForm = document.getElementById('model-pricing-form');

function modelDashboardQuery() {
  const params = new URLSearchParams();
  for (const [id, name] of [
    ['model-call-hours', 'hours'], ['model-call-provider', 'provider'],
    ['model-call-model', 'model'], ['model-call-operation', 'operation'],
    ['model-call-status', 'status'],
  ]) {
    const value = document.getElementById(id)?.value?.trim();
    if (value) params.set(name, value);
  }
  return params.toString();
}

function compactNumber(value) {
  return new Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 2 }).format(Number(value || 0));
}

function metricCard(label, value) {
  return `<article class="metric-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong></article>`;
}

function formatEstimatedCost(value, currency) {
  if (value === null || value === undefined) return 'unpriced';
  return new Intl.NumberFormat(undefined, { style: 'currency', currency: currency || 'EUR', minimumFractionDigits: 4, maximumFractionDigits: 8 }).format(Number(value));
}

async function loadModelDashboard() {
  const query = modelDashboardQuery();
  const [summaryRes, callsRes, pricingRes] = await Promise.all([
    fetch(`/api/model-calls/summary?${query}`),
    fetch(`/api/model-calls?${query}&limit=200`),
    fetch('/api/model-pricing'),
  ]);
  if (!summaryRes.ok || !callsRes.ok || !pricingRes.ok) throw new Error('Model dashboard could not be loaded.');
  const summary = await summaryRes.json();
  const calls = (await callsRes.json()).calls || [];
  const pricing = (await pricingRes.json()).pricing || [];
  document.getElementById('model-call-summary').innerHTML = [
    metricCard('Calls', compactNumber(summary.call_count)),
    metricCard('Input tokens', compactNumber(summary.input_tokens)),
    metricCard('Output tokens', compactNumber(summary.output_tokens)),
    metricCard('Estimated cost', formatEstimatedCost(summary.estimated_cost, summary.currency)),
    metricCard('Average duration', summary.average_duration_ms == null ? '—' : `${(summary.average_duration_ms / 1000).toFixed(2)} s`),
    metricCard('P95 duration', summary.p95_duration_ms == null ? '—' : `${(summary.p95_duration_ms / 1000).toFixed(2)} s`),
    metricCard('Generation speed', summary.average_generated_tokens_per_second == null ? '—' : `${summary.average_generated_tokens_per_second.toFixed(1)} tok/s`),
    metricCard('Unpriced calls', compactNumber(summary.unpriced_call_count)),
  ].join('');
  const maxTokens = Math.max(1, ...(summary.by_operation || []).map((row) => Number(row.input_tokens || 0) + Number(row.output_tokens || 0)));
  document.getElementById('model-operation-summary').innerHTML = `<table><thead><tr><th>Operation</th><th>Calls</th><th>Tokens</th><th>Avg.</th><th>Cost</th></tr></thead><tbody>${(summary.by_operation || []).map((row) => {
    const tokens = Number(row.input_tokens || 0) + Number(row.output_tokens || 0);
    return `<tr><td><code>${escapeHtml(row.name)}</code><div class="usage-bar"><span style="width:${Math.max(2, tokens / maxTokens * 100).toFixed(1)}%"></span></div></td><td>${row.call_count}</td><td>${compactNumber(tokens)}</td><td>${(Number(row.average_duration_ms || 0) / 1000).toFixed(2)} s</td><td>${formatEstimatedCost(row.estimated_cost, summary.currency)}</td></tr>`;
  }).join('')}</tbody></table>`;
  document.getElementById('model-pricing-list').innerHTML = pricing.length ? `<table><thead><tr><th>Provider/model</th><th>Input / 1M</th><th>Output / 1M</th></tr></thead><tbody>${pricing.map((row) => `<tr><td>${escapeHtml(row.provider)}/${escapeHtml(row.model)}<br><small>${escapeHtml(row.currency)}</small></td><td>${row.input_price_per_million}</td><td>${row.output_price_per_million}</td></tr>`).join('')}</tbody></table>` : '<p class="sub small">No prices configured. Token metrics are still recorded.</p>';
  document.getElementById('model-call-count-badge').textContent = `${calls.length} calls`;
  document.getElementById('model-call-table').innerHTML = `<table class="model-call-table"><thead><tr><th>Time</th><th>Operation</th><th>Provider/model</th><th>Input</th><th>Output</th><th>Total</th><th>Prompt eval</th><th>Generation</th><th>tok/s</th><th>Cost</th><th>Status</th></tr></thead><tbody>${calls.map((call) => `<tr><td>${escapeHtml(new Date(call.recorded_at).toLocaleString())}</td><td><code>${escapeHtml(call.operation)}</code>${call.attempt > 1 ? `<br><small>attempt ${call.attempt}</small>` : ''}</td><td>${escapeHtml(call.provider)}/${escapeHtml(call.model || 'unknown')}</td><td>${call.input_tokens ?? '—'}</td><td>${call.output_tokens ?? '—'}</td><td>${(Number(call.duration_ms || 0) / 1000).toFixed(2)} s</td><td>${call.prompt_evaluation_duration_ms == null ? '—' : `${(call.prompt_evaluation_duration_ms / 1000).toFixed(2)} s`}</td><td>${call.generation_duration_ms == null ? '—' : `${(call.generation_duration_ms / 1000).toFixed(2)} s`}</td><td>${call.generated_tokens_per_second ?? '—'}</td><td>${formatEstimatedCost(call.estimated_cost, call.currency)}</td><td><span class="badge ${call.status === 'completed' ? 'ok' : 'bad'}">${escapeHtml(call.status)}</span></td></tr>`).join('')}</tbody></table>`;
}

if (modelDashboardRefreshButton) modelDashboardRefreshButton.addEventListener('click', () => loadModelDashboard().catch((error) => alert(error.message)));
for (const id of ['model-call-hours', 'model-call-provider', 'model-call-model', 'model-call-operation', 'model-call-status']) {
  document.getElementById(id)?.addEventListener('change', () => loadModelDashboard().catch(() => {}));
}
if (modelPricingForm) modelPricingForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const payload = {
    provider: document.getElementById('pricing-provider').value,
    model: document.getElementById('pricing-model').value,
    currency: document.getElementById('pricing-currency').value,
    input_price_per_million: Number(document.getElementById('pricing-input').value),
    output_price_per_million: Number(document.getElementById('pricing-output').value),
  };
  const response = await fetch('/api/model-pricing', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  const body = await response.json();
  if (!response.ok) return alert(body.error || 'Pricing could not be saved.');
  await loadModelDashboard();
});

initializeTabs();
loadReviewQueue().catch(() => {});

loadConfig().catch((err) => {
  console.error(err);
  summaryEl.textContent = err.message || String(err);
});
