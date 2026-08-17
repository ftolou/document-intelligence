(() => {
  const state = { models: [], pricing: [] };
  const form = document.getElementById('model-pricing-form');
  const list = document.getElementById('model-pricing-list');
  if (!form || !list) return;

  function escapeHtml(value) {
    return String(value ?? '')
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#039;');
  }

  function modelKey(provider, model) {
    return `${provider}\u0000${model}`;
  }

  function selectedIdentity() {
    const select = document.getElementById('pricing-model-key');
    const index = Number(select?.value);
    return Number.isInteger(index) && state.models[index] ? state.models[index] : null;
  }

  function matchingPricing(model) {
    if (!model) return null;
    return state.pricing.find(
      (row) => row.provider === model.provider && row.model === model.model,
    ) || null;
  }

  function installForm() {
    form.innerHTML = `
      <label class="field full" style="grid-column:1/-1">
        <span>Model</span>
        <select id="pricing-model-key" required></select>
        <small id="pricing-canonical-id">Select a model observed by telemetry.</small>
      </label>
      <input id="pricing-provider" type="hidden" />
      <input id="pricing-model" type="hidden" />
      <label class="field">
        <span>Currency</span>
        <input id="pricing-currency" value="USD" maxlength="3" required />
      </label>
      <label class="field">
        <span>Input / 1M</span>
        <input id="pricing-input" type="number" step="0.000001" min="0" required />
      </label>
      <label class="field">
        <span>Cached read / 1M</span>
        <input id="pricing-cached-input" type="number" step="0.000001" min="0" />
      </label>
      <label class="field">
        <span>Cache write / 1M</span>
        <input id="pricing-cache-write-input" type="number" step="0.000001" min="0" />
      </label>
      <label class="field">
        <span>Output / 1M</span>
        <input id="pricing-output" type="number" step="0.000001" min="0" required />
      </label>
      <label class="field">
        <span>Effective from</span>
        <input id="pricing-effective-from" type="date" />
      </label>
      <button type="submit" style="grid-column:1/-1">Save pricing</button>
    `;
    document.getElementById('pricing-model-key')?.addEventListener('change', syncSelectedModel);
    form.addEventListener('submit', savePricing, true);
  }

  function preferredModelIndex(previousKey) {
    if (previousKey) {
      const previous = state.models.findIndex(
        (row) => modelKey(row.provider, row.model) === previousKey,
      );
      if (previous >= 0) return previous;
    }
    const observedUnpriced = state.models.findIndex((row) => row.observed && !row.has_pricing);
    if (observedUnpriced >= 0) return observedUnpriced;
    const observed = state.models.findIndex((row) => row.observed);
    return observed >= 0 ? observed : (state.models.length ? 0 : -1);
  }

  function populateModelSelect(previousKey = null) {
    const select = document.getElementById('pricing-model-key');
    if (!select) return;
    if (!state.models.length) {
      select.innerHTML = '<option value="">No observed or configured models</option>';
      select.disabled = true;
      syncSelectedModel();
      return;
    }
    select.disabled = false;
    select.innerHTML = state.models.map((row, index) => {
      const suffix = row.observed ? '' : ' (configured)';
      return `<option value="${index}">${escapeHtml(row.display_name || `${row.provider}/${row.model}`)}${suffix}</option>`;
    }).join('');
    const preferred = preferredModelIndex(previousKey);
    if (preferred >= 0) select.value = String(preferred);
    syncSelectedModel();
  }

  function setNumber(id, value) {
    const input = document.getElementById(id);
    if (input) input.value = value === null || value === undefined ? '' : String(value);
  }

  function syncSelectedModel() {
    const model = selectedIdentity();
    const providerInput = document.getElementById('pricing-provider');
    const modelInput = document.getElementById('pricing-model');
    const canonical = document.getElementById('pricing-canonical-id');
    if (!model) {
      if (providerInput) providerInput.value = '';
      if (modelInput) modelInput.value = '';
      if (canonical) canonical.textContent = 'No canonical model selected.';
      return;
    }
    if (providerInput) providerInput.value = model.provider;
    if (modelInput) modelInput.value = model.model;
    if (canonical) canonical.innerHTML = `Canonical ID: <code>${escapeHtml(model.provider)}/${escapeHtml(model.model)}</code>`;

    const pricing = matchingPricing(model);
    const currency = document.getElementById('pricing-currency');
    if (currency) currency.value = pricing?.currency || (model.provider === 'openai' ? 'USD' : 'EUR');
    setNumber('pricing-input', pricing?.input_price_per_million);
    setNumber('pricing-cached-input', pricing?.cached_input_price_per_million);
    setNumber('pricing-cache-write-input', pricing?.cache_write_input_price_per_million);
    setNumber('pricing-output', pricing?.output_price_per_million);
    const effective = document.getElementById('pricing-effective-from');
    if (effective) effective.value = pricing?.effective_from || '';
  }

  function renderPricingList() {
    if (!state.pricing.length) {
      list.innerHTML = '<p class="sub small">No prices configured. Token metrics are still recorded.</p>';
      return;
    }
    list.innerHTML = `
      <table>
        <thead><tr><th>Provider/model</th><th>Input</th><th>Cached read</th><th>Cache write</th><th>Output</th><th>Effective</th></tr></thead>
        <tbody>${state.pricing.map((row) => `
          <tr>
            <td>${escapeHtml(row.display_name || `${row.provider}/${row.model}`)}<br><small><code>${escapeHtml(row.provider)}/${escapeHtml(row.model)}</code> · ${escapeHtml(row.currency || '')}</small></td>
            <td>${row.input_price_per_million ?? '—'}</td>
            <td>${row.cached_input_price_per_million ?? '—'}</td>
            <td>${row.cache_write_input_price_per_million ?? '—'}</td>
            <td>${row.output_price_per_million ?? '—'}</td>
            <td>${escapeHtml(row.effective_from || '—')}<br><small>${escapeHtml(row.pricing_source || 'manual')}</small></td>
          </tr>`).join('')}
        </tbody>
      </table>`;
  }

  async function refreshPricingEnhancements() {
    const select = document.getElementById('pricing-model-key');
    const selected = selectedIdentity();
    const previousKey = selected ? modelKey(selected.provider, selected.model) : null;
    const response = await fetch('/api/model-pricing');
    if (!response.ok) throw new Error(`Pricing catalog load failed: ${response.status}`);
    const payload = await response.json();
    state.models = Array.isArray(payload.models) ? payload.models : [];
    state.pricing = Array.isArray(payload.pricing) ? payload.pricing : [];
    populateModelSelect(previousKey);
    renderPricingList();
    if (select && select.disabled && state.models.length) select.disabled = false;
  }

  async function savePricing(event) {
    event.preventDefault();
    event.stopImmediatePropagation();
    const model = selectedIdentity();
    if (!model) {
      alert('Select a canonical model first.');
      return;
    }
    const optionalNumber = (id) => {
      const raw = document.getElementById(id)?.value?.trim();
      return raw === '' || raw == null ? null : Number(raw);
    };
    const payload = {
      provider: model.provider,
      model: model.model,
      currency: document.getElementById('pricing-currency').value,
      input_price_per_million: Number(document.getElementById('pricing-input').value),
      cached_input_price_per_million: optionalNumber('pricing-cached-input'),
      cache_write_input_price_per_million: optionalNumber('pricing-cache-write-input'),
      output_price_per_million: Number(document.getElementById('pricing-output').value),
      effective_from: document.getElementById('pricing-effective-from').value || null,
      pricing_source: 'manual',
    };
    const response = await fetch('/api/model-pricing', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) {
      alert(body.error || 'Pricing could not be saved.');
      return;
    }
    if (typeof globalThis.loadModelDashboard === 'function') {
      await globalThis.loadModelDashboard();
    } else {
      await refreshPricingEnhancements();
    }
  }

  installForm();

  const originalLoadModelDashboard = globalThis.loadModelDashboard;
  if (typeof originalLoadModelDashboard === 'function') {
    globalThis.loadModelDashboard = async function enhancedLoadModelDashboard(...args) {
      const result = await originalLoadModelDashboard(...args);
      await refreshPricingEnhancements();
      return result;
    };
  }

  refreshPricingEnhancements().catch((error) => {
    console.warn('Pricing catalog enhancement failed', error);
  });
})();
