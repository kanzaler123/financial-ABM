'use strict';
// Greenwashing policy desk: page logic. Model arithmetic lives in core.js (cross-checked
// against Python) and the desk's drawing in desk.js; this file reads the levers, calls the
// core and hands the results to the page.
(() => {
  const core = globalThis.SandboxCore;
  const desk = globalThis.SandboxDesk;
  const TEXT = globalThis.SandboxText;
  const $ = id => document.getElementById(id);
  const SVG = 'http://www.w3.org/2000/svg';
  const REPO = 'https://github.com/kanzaler123/financial-ABM';

  let payload = null;
  try { payload = JSON.parse($('sandbox-data').textContent); } catch (error) { payload = null; }
  if (!payload || !payload.firms || !core || !desk || !TEXT) { $('load-error').hidden = false; return; }
  const { firms: firmData, stage_f: stageF, simplification } = payload;
  const domains = firmData.domains;
  const F1 = domains.esg.baseline_policy.fixed_fine;     // CNY 10 m in model money units
  const YUAN_PER_UNIT = 1e7 / F1;                        // one unit = the CSI 300 mean market cap
  const BP = 1e-4;
  const SNAPSHOT_DAY = firmData.snapshot_day;
  const css = (name, fallback) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
  const COLOR = {
    improve: css('--improve', '#2a78d6'), wash: css('--wash', '#eb6834'), neutral: css('--neutral', '#898781'),
    idle: css('--idle', '#d8d6cd'), ink: css('--ink', '#0b0b0b'), surface: css('--surface', '#fcfcfb'),
    violet: css('--regulator', '#6250d6'), investor: css('--investor', '#1baf7a'), axis: css('--axis', '#c3c2b7'),
  };
  const PALETTE = [COLOR.improve, COLOR.wash, COLOR.neutral, COLOR.idle];
  const IDLE = 3;
  const ACTIONS = [
    { key: 'actionImprove', color: 'var(--improve)' },
    { key: 'actionWash', color: 'var(--wash)' },
    { key: 'actionMaintain', color: 'var(--neutral)' },
  ];
  const SIZES = ['sizeLarge', 'sizeMid', 'sizeSmall'];
  const SERIES = { policy: { color: 'var(--ink)' }, baseline: { color: 'var(--faint)' } };
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

  // -- language ------------------------------------------------------------------------
  // English first; a stored or linked choice of Chinese wins.
  const query = new URLSearchParams(location.search);
  const storage = { get: () => { try { return localStorage.getItem('abm-lang'); } catch (e) { return null; } },
    set: value => { try { localStorage.setItem('abm-lang', value); } catch (e) { /* file:// may refuse */ } } };
  let lang = ['zh', 'en'].includes(query.get('lang')) ? query.get('lang') : storage.get() === 'zh' ? 'zh' : 'en';
  const t = (key, vars = {}) => String(TEXT[lang][key] ?? TEXT.en[key] ?? key)
    .replace(/\{(\w+)\}/g, (_, name) => (name in vars ? String(vars[name]) : `{${name}}`));
  const minus = s => s.replace('-', '−');
  const fixed = (x, d) => minus(Number(x).toFixed(d));
  const signed = (x, d = 3) => (x > 0 ? '+' : x < 0 ? '−' : '±') + Math.abs(x).toFixed(d);
  const pct = (x, d = 1) => `${(100 * x).toFixed(d)}%`;
  const grouped = x => Math.round(x).toLocaleString(lang === 'zh' ? 'zh-CN' : 'en-US');
  const wan = v => { const a = Math.abs(v); return a >= 100 ? fixed(v, 0) : a >= 10 ? fixed(v, 1) : a >= 0.01 || a === 0 ? fixed(v, 2) : minus(v.toPrecision(2)); };
  const sci = p => {
    if (p >= 1e-3) return p.toFixed(3);
    const exponent = Math.floor(Math.log10(p));
    const digits = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
    return `${(p / 10 ** exponent).toFixed(0)}×10${String(exponent).replace(/./g, c => digits[c])}`;
  };
  const money = yuan => {
    const [big, mid] = lang === 'zh' ? [1e8, 1e4] : [1e9, 1e6];
    if (yuan >= big) return t('moneyBig', { x: (yuan / big).toFixed(2) });
    const v = yuan / mid;
    return t('moneyMid', { x: v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2) });
  };
  const firmId = i => String(i + 1).padStart(3, '0');
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

  // -- small DOM and SVG helpers -------------------------------------------------------------
  const node = (tag, className, text) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = String(text);
    return element;
  };
  const svgNode = (tag, attrs = {}, parent = null, text) => {
    const element = document.createElementNS(SVG, tag);
    for (const [name, value] of Object.entries(attrs)) element.setAttribute(name, String(value));
    if (text !== undefined) element.textContent = String(text);
    if (parent) parent.append(element);
    return element;
  };
  const scale = (d0, d1, r0, r1) => value => r0 + (value - d0) / (d1 - d0 || 1) * (r1 - r0);
  const niceStep = (span, count) => {
    const raw = span / Math.max(count, 1);
    const magnitude = 10 ** Math.floor(Math.log10(raw || 1));
    return [1, 2, 2.5, 5, 10].map(m => m * magnitude).find(step => step >= raw) ?? 10 * magnitude;
  };
  const ticks = (lo, hi, count = 5) => {
    const step = niceStep(hi - lo, count);
    const start = Math.ceil(lo / step - 1e-9) * step;
    const values = [];
    for (let v = start; v <= hi + step * 1e-9; v += step) values.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return { values, step };
  };
  const decimals = step => Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  const frame = (container, width, height, margin) => {
    container.replaceChildren();
    const svg = svgNode('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img' }, container);
    return { svg, left: margin[3], right: width - margin[1], top: margin[0], bottom: height - margin[2], width, height };
  };
  const yAxis = (f, y, lo, hi, count, format) => {
    const grid = svgNode('g', { class: 'grid' }, f.svg);
    const { values, step } = ticks(lo, hi, count);
    for (const v of values) {
      svgNode('line', { x1: f.left, x2: f.right, y1: y(v), y2: y(v) }, grid);
      svgNode('text', { x: f.left - 8, y: y(v) + 4, 'text-anchor': 'end' }, f.svg, format ? format(v) : fixed(v, decimals(step)));
    }
  };
  const xAxis = (f, x, values, format) => {
    for (const v of values) svgNode('text', { x: x(v), y: f.bottom + 16, 'text-anchor': 'middle' }, f.svg, format(v));
    svgNode('line', { x1: f.left, x2: f.right, y1: f.bottom, y2: f.bottom, class: 'baseline' }, f.svg);
  };
  const pointer = (svg, event) => {
    const box = svg.getBoundingClientRect();
    const view = svg.viewBox.baseVal;
    return { x: (event.clientX - box.left) * view.width / box.width, y: (event.clientY - box.top) * view.height / box.height };
  };
  const legend = (container, items) => {
    container.replaceChildren();
    for (const item of items) {
      const entry = node('span');
      const key = node('i', item.ring ? 'key-ring' : item.line ? 'key-line' : item.band ? 'key-band' : 'key-dot');
      if (!item.ring) key.style.background = item.color;
      entry.append(key, document.createTextNode(item.label));
      container.append(entry);
    }
  };
  const table = (headers, rows, numeric = []) => {
    const element = node('table');
    const head = element.createTHead().insertRow();
    headers.forEach((label, i) => { const th = node('th', numeric.includes(i) ? 'num' : '', label); th.scope = 'col'; head.append(th); });
    const body = element.createTBody();
    for (const row of rows) {
      const tr = body.insertRow();
      row.forEach((cell, i) => {
        const td = tr.insertCell();
        if (numeric.includes(i)) td.className = 'num';
        if (cell instanceof Node) td.append(cell); else td.textContent = String(cell);
      });
    }
    const wrapper = node('div', 'table-scroll');
    wrapper.append(element);
    return wrapper;
  };

  // -- tooltip -------------------------------------------------------------------------
  const tip = $('tooltip');
  const showTip = (event, title, rows) => {
    tip.replaceChildren(node('div', 'tt-title', title));
    for (const row of rows) {
      const line = node('div', 'tt-row');
      if (row.color) { const key = node('i', row.ring ? 'key-ring' : 'key-line'); if (!row.ring) key.style.background = row.color; line.append(key); }
      line.append(node('strong', '', row.value), node('span', '', row.label ?? ''));
      tip.append(line);
    }
    tip.hidden = false;
    const box = tip.getBoundingClientRect();
    const x = event.clientX ?? (event.target.getBoundingClientRect().right);
    const y = event.clientY ?? (event.target.getBoundingClientRect().top);
    tip.style.left = `${Math.min(x + 14, innerWidth - box.width - 8)}px`;
    tip.style.top = `${Math.min(y + 14, innerHeight - box.height - 8)}px`;
  };
  const hideTip = () => { tip.hidden = true; };

  // -- levers ----------------------------------------------------------------------------
  const state = { domain: 'esg', levers: null, firm: null, seed: 1, k: -1 };
  const domain = () => domains[state.domain];
  const baselineRegime = d => ({ ...d.baseline_policy, disclosure_delay_days: null });
  const policyLevers = d => ({ tau: d.baseline_policy.transparency, fixed: d.baseline_policy.fixed_fine / F1,
    prop: d.baseline_policy.penalty, audit: d.baseline_policy.audit_probability });
  const PRESETS = {
    B: () => ({}), I: () => ({ tau: 0.6 }), F: () => ({ fixed: 4 }), IF: () => ({ tau: 0.6, fixed: 4 }), zero: () => ({ fixed: 0, prop: 0 }),
  };
  const presetLevers = name => ({ ...policyLevers(domain()), ...PRESETS[name]() });
  const regime = () => ({ audit_probability: state.levers.audit, penalty: state.levers.prop, transparency: state.levers.tau,
    subsidy: 0, fixed_fine: state.levers.fixed * F1, disclosure_delay_days: null });
  // The proportional fine slider is logarithmic: 0, then 0.1 to 100 basis points per SD.
  const propFromPosition = k => (k <= 0 ? 0 : 1e-5 * 10 ** ((k - 1) * 3 / 89));
  const positionFromProp = p => (p <= 0 ? 0 : Math.min(90, Math.max(1, Math.round(1 + Math.log10(p / 1e-5) * 89 / 3))));
  const sameLevers = (a, b) => ['tau', 'fixed', 'prop', 'audit'].every(key => Math.abs(a[key] - b[key]) < 1e-15);
  const leverText = l => ({
    tau: l.tau.toFixed(2), fixed: l.fixed % 1 ? l.fixed.toFixed(1) : l.fixed.toFixed(0),
    prop: l.prop ? Number((l.prop / BP).toPrecision(2)).toString() : '0', audit: (100 * l.audit).toFixed(1), decay: l.decay.toFixed(2),
  });
  const LEVERS = ['tau', 'fixed', 'prop', 'audit', 'decay'];

  // -- shareable links -----------------------------------------------------------------
  // The address carries the domain, the levers and the selected firm, so a copied link opens
  // the same desk. Values that are missing, malformed or outside a slider's range are ignored.
  const LINK_DOMAINS = { esg: 'esg', ai: 'ai_washing' };
  const linkNumber = (name, lo, hi) => {
    const raw = query.get(name);
    if (raw === null || raw.trim() === '') return null;
    const v = Number(raw);
    return Number.isFinite(v) && v >= lo && v <= hi ? v : null;
  };
  const linkedState = () => {
    const range = id => [Number($(id).min), Number($(id).max)];
    const levers = {};
    for (const key of ['tau', 'fixed', 'audit', 'decay']) {
      const v = linkNumber(key, ...range(`lever-${key}`));
      if (v !== null) levers[key] = v;
    }
    // The proportional fine travels in basis points per SD and snaps to the slider's steps.
    const bp = linkNumber('prop', 0, propFromPosition(Number($('lever-prop').max)) / BP);
    if (bp !== null) levers.prop = propFromPosition(positionFromProp(bp * BP));
    const firm = linkNumber('firm', 1, 200);
    return { domain: LINK_DOMAINS[query.get('d')] ?? null, levers, firm: firm !== null && firm % 1 === 0 ? firm - 1 : null };
  };
  const linkParams = () => {
    const l = state.levers, text = leverText(l), d = domain();
    const opening = { ...presetLevers('I'), decay: d.natural_decay };
    const params = new URLSearchParams();
    if (query.has('lang')) params.set('lang', lang);
    const untouched = state.domain === 'esg' && sameLevers(l, opening) && Math.abs(l.decay - opening.decay) < 1e-12
      && state.firm === defaultFirm(d);
    if (!untouched) {
      params.set('d', state.domain === 'esg' ? 'esg' : 'ai');
      params.set('tau', text.tau); params.set('fixed', text.fixed); params.set('prop', text.prop);
      params.set('audit', String(Number(l.audit.toFixed(4)))); params.set('decay', text.decay);
      params.set('firm', String(state.firm + 1));
    }
    return params;
  };
  const shareUrl = () => {
    const search = linkParams().toString();
    return `${location.href.split(/[?#]/)[0]}${search ? `?${search}` : ''}`;
  };
  const writeLink = () => {
    const search = linkParams().toString();
    if (`?${search}` === location.search || (!search && !location.search)) return;
    try { history.replaceState(null, '', `${location.pathname}${search ? `?${search}` : ''}${location.hash}`); } catch (e) { /* file:// may refuse */ }
  };

  const syncLevers = () => {
    const l = state.levers, text = leverText(l);
    $('lever-tau').value = l.tau; $('lever-fixed').value = l.fixed; $('lever-prop').value = positionFromProp(l.prop);
    $('lever-audit').value = l.audit; $('lever-decay').value = l.decay;
    for (const key of LEVERS) {
      const input = $(`lever-${key}`);
      input.style.setProperty('--fill', `${100 * (input.value - input.min) / (input.max - input.min)}%`);
    }
    $('out-tau').textContent = text.tau;
    $('out-fixed').textContent = t('leverFixedValue', { x: text.fixed });
    $('out-prop').textContent = t('leverPropValue', { x: text.prop });
    $('out-audit').textContent = t('leverAuditValue', { x: text.audit });
    $('out-decay').textContent = t('leverDecayValue', { x: text.decay });
    const star = core.tauStar(domain(), domain().market, regime());
    const tick = $('tau-star-tick');
    tick.hidden = !(star > 0 && star < 1);
    tick.style.left = `${100 * star}%`;
    $('tau-star-note').textContent = tick.hidden ? '' : t('leverTauStar', { tau: star.toFixed(2) });
    for (const button of document.querySelectorAll('[data-preset]')) {
      button.setAttribute('aria-pressed', String(sameLevers(l, presetLevers(button.dataset.preset))));
    }
    for (const button of document.querySelectorAll('[data-domain]')) button.setAttribute('aria-pressed', String(button.dataset.domain === state.domain));
    $('ai-note').hidden = state.domain !== 'ai_washing';
    $('policy-summary').textContent = t('policySummary', text);
    renderPolicyAgents(star);
  };
  const flash = index => {
    sphere.boost(index);
    const lever = document.querySelectorAll('.lever')[index];
    lever.classList.add('active');
    clearTimeout(lever.flashTimer);
    lever.flashTimer = setTimeout(() => lever.classList.remove('active'), 1400);
  };
  const onLever = (key, apply) => $(`lever-${key}`).addEventListener('input', event => {
    apply(Number(event.target.value));
    flash(LEVERS.indexOf(key));
    update();
  });
  onLever('tau', v => { state.levers.tau = v; });
  onLever('fixed', v => { state.levers.fixed = v; });
  onLever('prop', v => { state.levers.prop = propFromPosition(v); });
  onLever('audit', v => { state.levers.audit = v; });
  onLever('decay', v => { state.levers.decay = v; });
  for (const button of document.querySelectorAll('[data-preset]')) {
    button.addEventListener('click', () => {
      state.levers = { ...presetLevers(button.dataset.preset), decay: state.levers.decay };
      LEVERS.forEach((_, k) => sphere.boost(k));
      update();
    });
  }
  for (const button of document.querySelectorAll('[data-domain]')) {
    button.addEventListener('click', () => { if (button.dataset.domain !== state.domain) setDomain(button.dataset.domain); });
  }

  // -- panel 1: one decision period ----------------------------------------------------------
  const snapshotState = d => core.createState(d.firms.goodness, d.firms.inflation, d.firms.cost_scale, d.firms.pending);
  const decideAll = (d, r) => core.decide(snapshotState(d), d.firms, d, d.market, r, SNAPSHOT_DAY);
  const capYi = (d, i) => d.firms.market_cap[i] * YUAN_PER_UNIT / 1e8;
  const toWan = (d, i, fraction) => fraction * d.firms.market_cap[i] * YUAN_PER_UNIT / 1e4;
  const agreement = {};
  const defaultFirm = d => {
    const washer = d.firms.full_model_decision.actions.findIndex((a, i) => a === core.INFLATE && d.firms.size_class[i] === 0);
    return washer >= 0 ? washer : Math.max(0, d.firms.size_class.indexOf(0));
  };
  const firmOption = (d, i) => t('firmOption', { id: firmId(i), size: t(SIZES[d.firms.size_class[i]]), cap: grouped(capYi(d, i)) });
  let decision = null;

  const renderScatter = (d, choices) => {
    const container = $('scatter');
    const f = frame(container, 640, 330, [12, 16, 40, 54]);
    f.svg.setAttribute('aria-label', t('scatterTitle'));
    f.svg.setAttribute('tabindex', '0');
    const caps = d.firms.market_cap.map((_, i) => Math.log10(capYi(d, i)));
    const benefits = d.firms.benefit.map(b => b / BP);
    const xLo = Math.floor(Math.min(...caps) * 4) / 4, xHi = Math.ceil(Math.max(...caps) * 4) / 4;
    const yHi = ticks(0, Math.max(...benefits) * 1.05, 5).values.at(-1) || 1;
    const x = scale(xLo, xHi, f.left, f.right), y = scale(0, yHi, f.bottom, f.top);
    yAxis(f, y, 0, yHi, 5);
    const decades = [];
    for (let k = Math.ceil(xLo); k <= xHi; k++) decades.push(k);
    xAxis(f, x, decades, k => grouped(10 ** k));
    svgNode('text', { x: f.right, y: f.bottom + 34, 'text-anchor': 'end', class: 'axis-title' }, f.svg, t('scatterX'));
    svgNode('text', { x: f.left, y: f.top - 2, class: 'axis-title' }, f.svg, t('scatterY'));
    const points = caps.map((c, i) => ({ i, px: x(c), py: y(benefits[i]) }));
    const dots = svgNode('g', {}, f.svg);
    for (const p of points) {
      const dot = svgNode('circle', { cx: p.px, cy: p.py, r: 4.5, class: 'dot' }, dots);
      dot.style.fill = ACTIONS[choices[p.i].action].color;
    }
    const s = points[state.firm];
    svgNode('circle', { cx: s.px, cy: s.py, r: 8.5, class: 'selected-ring' }, f.svg);
    const nearest = event => {
      const at = pointer(f.svg, event);
      let best = null, distance = 24 * 24;
      for (const p of points) {
        const dd = (p.px - at.x) ** 2 + (p.py - at.y) ** 2;
        if (dd < distance) { distance = dd; best = p; }
      }
      return best;
    };
    f.svg.addEventListener('pointermove', event => {
      const p = nearest(event);
      if (!p) { hideTip(); f.svg.style.cursor = 'default'; return; }
      f.svg.style.cursor = 'pointer';
      const c = choices[p.i];
      showTip(event, firmOption(d, p.i), [
        { color: ACTIONS[c.action].color, value: c.action === core.MAINTAIN ? t('actionMaintain') : t('factAmount', { action: t(ACTIONS[c.action].key), x: c.amount.toFixed(3) }) },
        { value: (d.firms.benefit[p.i] / BP).toFixed(2), label: t('factBenefit') },
      ]);
    });
    f.svg.addEventListener('pointerleave', hideTip);
    f.svg.addEventListener('click', event => { const p = nearest(event); if (p) selectFirm(p.i); });
    f.svg.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
      event.preventDefault();
      selectFirm((state.firm + (event.key === 'ArrowRight' ? 1 : -1) + points.length) % points.length);
      $('scatter').querySelector('svg').focus();
    });
  };

  const renderSizeBars = (d, choices) => {
    const container = $('size-bars');
    container.replaceChildren();
    SIZES.forEach((key, k) => {
      const members = choices.filter((_, i) => d.firms.size_class[i] === k);
      const counts = [0, 0, 0];
      for (const c of members) counts[c.action] += 1;
      const row = node('div', 'size-row');
      const bar = node('div', 'bar');
      counts.forEach((count, a) => {
        if (!count) return;
        const segment = node('span');
        segment.style.flex = String(count);
        segment.style.background = ACTIONS[a].color;
        segment.title = `${t(ACTIONS[a].key)} ${count}`;
        bar.append(segment);
      });
      row.append(node('span', '', `${t(key)} · ${members.length}`), bar,
        node('span', 'counts', counts.map((count, a) => `${t(ACTIONS[a].key)} ${count}`).filter((_, a) => counts[a] || a < 2).join(' · ')));
      container.append(row);
    });
  };

  const valueCurves = (d, i, r, factors, room, probability) => {
    const b = d.firms.benefit[i];
    const mc = d.firms.cost_scale[i] * Math.exp(d.cost_level_elasticity * d.firms.goodness[i]);
    const cap = d.max_change_per_period;
    const improveCap = Math.min(Math.max(room.improve[i], 0), cap);
    const inflateCap = Math.min(Math.max(room.inflate[i], 0), cap);
    const washCost = d.wash_cost_ratio * mc;
    const gain = Math.max(b * (1 - r.transparency) * factors.wash - r.penalty * factors.fine, 0);
    const newCase = probability ? r.fixed_fine / d.firms.market_cap[i] * probability[i] : 0;
    const improve = delta => (delta <= inflateCap
      ? b * factors.real * delta
      : b * factors.real * inflateCap + b * factors.displaced * (delta - inflateCap)) + r.subsidy * delta - mc * delta * delta;
    const wash = delta => gain * delta - washCost * delta * delta - (delta > 0 ? newCase * factors.fine : 0);
    return { improve, wash, improveCap, inflateCap, newCaseValue: newCase * factors.fine };
  };

  const renderFirm = (d, choices, factors, room, probability) => {
    const i = state.firm, c = choices[i], r = regime();
    const facts = $('firm-facts');
    facts.replaceChildren();
    const fact = (label, value) => { const item = node('div'); item.append(node('dt', '', label), node('dd', '', value)); facts.append(item); };
    fact(t('factSize'), t(SIZES[d.firms.size_class[i]]));
    fact(t('factCap'), t('factCapValue', { x: grouped(capYi(d, i)) }));
    fact(t('factBenefit'), t('factBenefitValue', { x: (d.firms.benefit[i] / BP).toFixed(2) }));
    fact(t('factCost'), d.firms.cost_scale[i].toFixed(4));
    fact(t('factGoodness'), t('factSdValue', { x: signed(d.firms.goodness[i], 2) }));
    fact(t('factInflation'), t('factSdValue', { x: d.firms.inflation[i].toFixed(2) }));
    fact(t('factChoice'), c.action === core.MAINTAIN ? t('actionMaintain') : t('factAmount', { action: t(ACTIONS[c.action].key), x: c.amount.toFixed(3) }));

    const curves = valueCurves(d, i, r, factors, room, probability);
    const container = $('npv-chart');
    // The top margin holds the y-axis title on its own row, so optimum labels near the top never meet it.
    const f = frame(container, 560, 270, [36, 18, 40, 58]);
    f.svg.setAttribute('aria-label', t('npvTitle'));
    const xHi = Math.max(curves.improveCap, curves.inflateCap, 0.01);
    const samples = 120;
    const series = [
      { key: 'improve', cap: curves.improveCap, fn: curves.improve, color: ACTIONS[0].color, best: c.improveAmount, label: t('npvImproveBest') },
      { key: 'wash', cap: curves.inflateCap, fn: curves.wash, color: ACTIONS[1].color, best: c.inflateAmount, label: t('npvWashBest') },
    ].map(s => ({ ...s, points: Array.from({ length: samples + 1 }, (_, k) => { const delta = s.cap * k / samples; return [delta, toWan(d, i, s.fn(delta))]; }) }));
    const values = series.flatMap(s => s.points.slice(1).map(p => p[1])).concat([0]);
    let yLo = Math.min(...values), yHi = Math.max(...values);
    if (yHi - yLo < 1e-9) { yHi = 1; yLo = -1; }
    const pad = (yHi - yLo) * 0.08;
    const yTicks = ticks(yLo - pad, yHi + pad, 5);
    yLo = Math.min(yLo - pad, yTicks.values[0]); yHi = Math.max(yHi + pad, yTicks.values.at(-1));
    // Δ = 0 sits a little right of the tick labels, so a marker there does not cover them.
    const x0 = f.left + 12;
    const x = scale(0, xHi, x0, f.right), y = scale(yLo, yHi, f.bottom, f.top);
    const yStep = ticks(yLo, yHi, 5).step;
    yAxis(f, y, yLo, yHi, 5, v => (yStep >= 1 ? minus(grouped(v)) : fixed(v, decimals(yStep))));
    xAxis(f, x, ticks(0, xHi, 5).values, v => v.toFixed(decimals(niceStep(xHi, 5))));
    svgNode('line', { x1: f.left, x2: f.right, y1: y(0), y2: y(0), class: 'baseline' }, f.svg);
    svgNode('text', { x: f.right, y: f.bottom + 34, 'text-anchor': 'end', class: 'axis-title' }, f.svg, t('npvX'));
    const yTitle = svgNode('text', { x: f.left, y: 14, class: 'axis-title' }, f.svg, t('npvY'));
    for (const s of series) {
      if (s.cap <= 0) continue;
      // A fixed fine is a jump at Δ > 0: the curve starts just right of zero.
      const path = s.points.map(([delta, value], k) => `${k ? 'L' : 'M'}${x(delta).toFixed(1)},${y(k === 0 && s.key === 'wash' ? toWan(d, i, s.fn(1e-12)) : value).toFixed(1)}`).join('');
      const line = svgNode('path', { d: path, fill: 'none', 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, f.svg);
      line.style.stroke = s.color;
    }
    // Optima: markers first, then each label at the first spot around its point that stays inside the plot
    // and clear of the axis title, the markers and the labels already placed. Optima often sit together
    // at Δ ≈ 0; the halo keeps a curve that passes behind a label from striking through it.
    const taken = [yTitle.getBBox()];
    const marks = [];
    for (const s of series) {
      if (s.best <= 0) continue;
      const value = toWan(d, i, s.fn(s.best));
      const chosen = (s.key === 'improve' && c.action === core.IMPROVE) || (s.key === 'wash' && c.action === core.INFLATE);
      const px = x(s.best), py = y(value);
      const dot = svgNode('circle', { cx: px, cy: py, r: chosen ? 5.5 : 4.5, class: 'dot' }, f.svg);
      dot.style.fill = s.color;
      if (chosen) svgNode('circle', { cx: px, cy: py, r: 9, class: 'selected-ring' }, f.svg);
      marks.push({ s, value, chosen, px, py });
    }
    // Two optima on top of each other are labelled as one cluster: the labels stack beside it, higher point first.
    const [first, second] = marks;
    if (second && Math.abs(first.px - second.px) < 24 && Math.abs(first.py - second.py) < 24) {
      first.group = second.group = { left: Math.min(first.px, second.px), right: Math.max(first.px, second.px) };
    }
    for (const m of marks) taken.push({ x: m.px - 11, y: m.py - 11, width: 22, height: 22, owner: m.group ?? m });
    // A label's spots are offsets from its own marker (or cluster), so only the other markers count against it.
    const clear = (b, owner) => b.x >= f.left && b.x + b.width <= f.right && b.y >= f.top - 16 && b.y + b.height <= f.bottom
      && taken.every(o => o.owner === owner || b.x > o.x + o.width + 3 || o.x > b.x + b.width + 3 || b.y > o.y + o.height + 1 || o.y > b.y + b.height + 1);
    const place = (text, className, px, py, spots, owner = null) => {
      const label = svgNode('text', { class: className }, f.svg, text);
      const at = ([dx, dy, anchor]) => {
        label.setAttribute('x', px + dx); label.setAttribute('y', py + dy); label.setAttribute('text-anchor', anchor);
        return label.getBBox();
      };
      taken.push(at(spots.find(spot => clear(at(spot), owner)) ?? spots[0]));
    };
    for (const m of marks.sort((a, b) => b.chosen - a.chosen || a.py - b.py)) {
      const side = m.px > (f.left + f.right) / 2 ? -1 : 1;
      const near = side > 0 ? 'start' : 'end', far = side > 0 ? 'end' : 'start';
      const px = m.group ? (side > 0 ? m.group.right : m.group.left) : m.px;
      place(`${m.s.label} Δ=${m.s.best.toFixed(3)} · ${wan(m.value)}`, 'label halo', px, m.py, [
        [12 * side, -10, near], [12 * side, 20, near], [-12 * side, -10, far], [-12 * side, 20, far],
        [12 * side, 36, near], [12 * side, -26, near], [12 * side, 52, near],
      ], m.group ?? m);
    }
    place(t('npvMaintain'), 'halo', f.right, y(0), [[0, -6, 'end'], [0, 14, 'end']]);
    const hit = svgNode('rect', { x: f.left, y: f.top, width: f.right - f.left, height: f.bottom - f.top, class: 'hit' }, f.svg);
    const cross = svgNode('line', { y1: f.top, y2: f.bottom, class: 'crosshair', visibility: 'hidden' }, f.svg);
    hit.addEventListener('pointermove', event => {
      const delta = Math.min(Math.max((pointer(f.svg, event).x - x0) / (f.right - x0) * xHi, 0), xHi);
      cross.setAttribute('x1', x(delta)); cross.setAttribute('x2', x(delta)); cross.setAttribute('visibility', 'visible');
      showTip(event, `Δ = ${delta.toFixed(3)} SD`, series.filter(s => delta <= s.cap + 1e-12 && s.cap > 0)
        .map(s => ({ color: s.color, value: wan(toWan(d, i, s.fn(delta))), label: t(s.key === 'improve' ? 'actionImprove' : 'actionWash') })));
    });
    hit.addEventListener('pointerleave', () => { cross.setAttribute('visibility', 'hidden'); hideTip(); });

    const best = { v: toWan(d, i, c.values[0]), w: toWan(d, i, c.values[1]) };
    const why = $('firm-why');
    why.replaceChildren();
    const main = c.action === core.IMPROVE ? t('whyImprove', { a: c.amount.toFixed(3), v: wan(best.v), w: wan(best.w) })
      : c.action === core.INFLATE ? t('whyWash', { a: c.amount.toFixed(3), v: wan(best.v), w: wan(best.w) }) : t('whyMaintain');
    why.append(document.createTextNode(main));
    const notes = [];
    const star = core.tauStar(d, d.market, r);
    if (r.transparency >= star && c.action !== core.INFLATE) notes.push(t('whyTau', { tau: r.transparency.toFixed(2), star: star.toFixed(2) }));
    if (r.fixed_fine > 0 && d.firms.inflation[i] > 0) notes.push(t('whyJoin'));
    else if (r.fixed_fine > 0 && curves.newCaseValue > 0) notes.push(t('whyFine', { f: wan(toWan(d, i, curves.newCaseValue)) }));
    if (curves.inflateCap < 0.05) notes.push(t('whyCap', { x: curves.inflateCap.toFixed(3) }));
    for (const note of notes) why.append(node('span', '', note));
  };

  const fillSelect = (select, d, n) => {
    if (select.options.length !== n || select.dataset.lang !== lang || select.dataset.domain !== state.domain) {
      select.replaceChildren(...Array.from({ length: n }, (_, i) => {
        const option = node('option', '', firmOption(d, i));
        option.value = String(i);
        return option;
      }));
      select.dataset.lang = lang; select.dataset.domain = state.domain;
    }
    select.value = String(state.firm);
  };

  const renderDecision = () => {
    const d = domain(), r = regime();
    const { choices, factors, room, probability } = decideAll(d, r);
    decision = { choices, factors, room, probability };
    const counts = [0, 0, 0];
    for (const c of choices) counts[c.action] += 1;
    $('decision-summary').textContent = t('p1Summary', { improve: counts[0], wash: counts[1], maintain: counts[2] });
    legend($('action-legend'), ACTIONS.map(a => ({ label: t(a.key), color: a.color })));
    legend($('npv-legend'), ACTIONS.slice(0, 2).map(a => ({ label: t(a.key), color: a.color, line: true })));
    if (!(state.domain in agreement)) {
      const base = decideAll(d, baselineRegime(d)).choices;
      const recorded = d.firms.full_model_decision;
      agreement[state.domain] = base.filter((c, i) => c.action !== recorded.actions[i]
        || Math.abs(c.amount - recorded.amounts[i]) > 1e-12 * Math.max(Math.abs(recorded.amounts[i]), 1e-300)).length;
    }
    const check = $('agreement');
    check.replaceChildren();
    const mismatches = agreement[state.domain];
    check.append(node('span', mismatches ? 'bad' : 'ok', mismatches ? '!' : '✓'),
      document.createTextNode(mismatches ? t('p1Disagreement', { n: mismatches }) : t('p1Agreement')));
    fillSelect($('firm-select'), d, choices.length);
    renderScatter(d, choices);
    renderSizeBars(d, choices);
    renderFirm(d, choices, factors, room, probability);
  };

  // -- CSV export of panel 1 decisions ---------------------------------------------------
  // RFC 4180: a field is quoted when it holds a comma, a quote or a line break, and embedded
  // quotes are doubled. Every column here is a number or a fixed token, but escaping keeps the
  // file correct for any future column.
  const csvCell = value => {
    const text = String(value);
    return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  const csvNumber = x => String(Number(x.toFixed(9)));
  const csvRow = cells => cells.map(csvCell).join(',');
  const SIZE_NAMES = ['large', 'mid', 'small'];
  const ACTION_NAMES = ['improve', 'inflate', 'maintain'];
  const exportCsv = () => {
    const d = domain(), r = regime(), text = leverText(state.levers);
    const choices = decision ? decision.choices : decideAll(d, r).choices;
    const lines = [
      '# Financial ABM policy desk - Panel 1 - one decision period',
      `# domain: ${state.domain}`,
      `# snapshot day: ${SNAPSHOT_DAY}`,
      `# transparency tau: ${text.tau}`,
      `# fixed fine: ${text.fixed}x baseline (CNY 10 m)`,
      `# proportional penalty: ${text.prop} bp/SD`,
      `# audit probability: ${text.audit}%`,
      '# subsidy: 0',
      '# columns: id, size_class, market_cap (CNY 100 m), action, amount (SD)',
      csvRow(['id', 'size_class', 'market_cap', 'action', 'amount']),
    ];
    for (let i = 0; i < choices.length; i++) {
      const c = choices[i];
      lines.push(csvRow([i + 1, SIZE_NAMES[d.firms.size_class[i]], csvNumber(capYi(d, i)), ACTION_NAMES[c.action], csvNumber(c.amount)]));
    }
    // Built in the page with a Blob and an object URL: no network request.
    const blob = new Blob([lines.join('\r\n') + '\r\n'], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `financial-abm-panel1-${state.domain}.csv`;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  };
  $('export-csv').addEventListener('click', exportCsv);

  const selectFirm = i => {
    state.firm = i;
    $('firm-select').value = String(i);
    $('ticket-select').value = String(i);
    sphere.select(i);
    renderScatter(domain(), decision.choices);
    renderFirm(domain(), decision.choices, decision.factors, decision.room, decision.probability);
    if (replay) { renderTicket(); drawSpectro(); }
    writeLink();
  };
  $('firm-select').addEventListener('change', event => selectFirm(Number(event.target.value)));
  $('ticket-select').addEventListener('change', event => selectFirm(Number(event.target.value)));

  // -- panel 2 and the desk replay: simplified dynamics -----------------------------------------
  const baselineRuns = new Map();
  const simulate = (d, intervention, decay, seed, onPeriod = null) => core.simulate(d.firms, d, d.market,
    intervention ? { baseline: baselineRegime(d), intervention, start_day: d.market.burn_in_days + 1 } : { baseline: baselineRegime(d), intervention: null, start_day: 0 },
    { naturalDecay: decay, auditSeed: seed, onPeriod });
  const baselineRun = (d, decay, seed) => {
    const key = `${state.domain}|${decay}|${seed}`;
    if (!baselineRuns.has(key)) {
      if (baselineRuns.size > 24) baselineRuns.clear();
      baselineRuns.set(key, simulate(d, null, decay, seed));
    }
    return baselineRuns.get(key);
  };

  // Firms by size class (large first), largest market cap first within each class.
  const sizeGroups = d => [0, 1, 2].map(k => d.firms.size_class.map((c, i) => (c === k ? i : -1)).filter(i => i >= 0)
    .sort((a, b) => d.firms.market_cap[b] - d.firms.market_cap[a]));

  const buildEvents = (d, run) => {
    const events = [{ day: 0, type: 'start' }];
    const switchAt = d.market.burn_in_days + 1;
    let switched = false, caughtTotal = 0;
    run.decisionDays.forEach((day, k) => {
      if (!switched && day >= switchAt) { events.push({ day: switchAt, k, type: 'switch' }); switched = true; }
      const caught = run.detections[k];
      for (let i = 0; i < caught.length; i++) {
        if (caught[i]) { events.push({ day, k, type: 'caught', i, fine: run.fines[k][i] * d.firms.market_cap[i] * YUAN_PER_UNIT }); caughtTotal += 1; }
      }
      const now = run.actions[k];
      if (k === 0) {
        const counts = [0, 0, 0];
        for (const a of now) counts[a] += 1;
        events.push({ day, k, type: 'first', counts });
        return;
      }
      const before = run.actions[k - 1], groups = new Map();
      for (let i = 0; i < now.length; i++) {
        if (now[i] === before[i]) continue;
        const key = before[i] * 3 + now[i];
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(i);
      }
      for (const [key, firms] of groups) events.push({ day, k, type: 'turn', from: Math.floor(key / 3), to: key % 3, firms });
    });
    events.push({ day: d.market.trading_days, type: 'end', total: caughtTotal });
    return events;
  };

  let replay = null;
  const computeReplay = () => {
    const d = domain(), r = regime(), decay = state.levers.decay, seed = state.seed;
    const goodness = [], inflation = [];
    const run = simulate(d, r, decay, seed, (day, s) => {
      goodness.push(Float32Array.from(s.goodness));
      inflation.push(Float32Array.from(s.inflation));
    });
    const base = baselineRun(d, decay, seed);
    const groups = sizeGroups(d);
    const counts = run.actions.map(actions => {
      const c = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      actions.forEach((a, i) => { c[d.firms.size_class[i]][a] += 1; });
      return c;
    });
    const order = Int32Array.from(groups.flat());
    const rowOf = new Int32Array(order.length);
    order.forEach((i, j) => { rowOf[i] = j; });
    const start = d.market.burn_in_days, end = d.market.trading_days;
    return {
      d, run, base, goodness, inflation, counts, groups, order, rowOf,
      caughtPerPeriod: run.detections.map(c => c.reduce((s, v) => s + v, 0)),
      switchK: run.decisionDays.findIndex(day => day > start),
      events: buildEvents(d, run),
      m: core.windowMetrics(run, d.firms, start, end), mb: core.windowMetrics(base, d.firms, start, end),
    };
  };

  const lineChart = (container, series, label, format) => {
    const f = frame(container, 560, 230, [14, 16, 34, 50]);
    f.svg.setAttribute('aria-label', label);
    f.svg.setAttribute('tabindex', '0');
    const days = series[0].values.length - 1;
    const step = 5;
    const all = series.flatMap(s => Array.from(s.values));
    let lo = Math.min(...all), hi = Math.max(...all);
    const pad = (hi - lo || 1) * 0.08;
    const yT = ticks(lo - pad, hi + pad, 4);
    lo = Math.min(lo - pad, yT.values[0]); hi = Math.max(hi + pad, yT.values.at(-1));
    const x = scale(0, days, f.left, f.right), y = scale(lo, hi, f.bottom, f.top);
    yAxis(f, y, lo, hi, 4);
    const years = [];
    for (let year = 0; year * 252 <= days; year += 2) years.push(year * 252);
    xAxis(f, x, years, v => String(v / 252));
    svgNode('text', { x: f.right, y: f.bottom + 30, 'text-anchor': 'end', class: 'axis-title' }, f.svg, t('chartYears'));
    const switchDay = domain().market.burn_in_days + 1;
    svgNode('line', { x1: x(switchDay), x2: x(switchDay), y1: f.top, y2: f.bottom, class: 'annotation' }, f.svg);
    svgNode('text', { x: x(switchDay) + 6, y: f.top + 10, class: 'annotation-text' }, f.svg, t('chartSwitch'));
    for (const s of [...series].reverse()) {
      const sampled = [];
      for (let day = 0; day < days; day += step) sampled.push(day);
      sampled.push(days);
      const path = sampled.map((day, k) => `${k ? 'L' : 'M'}${x(day).toFixed(1)},${y(s.values[day]).toFixed(1)}`).join('');
      const line = svgNode('path', { d: path, fill: 'none', 'stroke-width': 2, 'stroke-linejoin': 'round' }, f.svg);
      line.style.stroke = s.color;
    }
    const end = series[0];
    svgNode('circle', { cx: x(days), cy: y(end.values[days]), r: 4, class: 'dot' }, f.svg).style.fill = end.color;
    svgNode('text', { x: x(days) - 8, y: y(end.values[days]) - 9, 'text-anchor': 'end', class: 'label' }, f.svg, format(end.values[days]));
    const cross = svgNode('line', { y1: f.top, y2: f.bottom, class: 'crosshair', visibility: 'hidden' }, f.svg);
    const hit = svgNode('rect', { x: f.left, y: f.top, width: f.right - f.left, height: f.bottom - f.top, class: 'hit' }, f.svg);
    let focusDay = days;
    const show = (event, day) => {
      cross.setAttribute('x1', x(day)); cross.setAttribute('x2', x(day)); cross.setAttribute('visibility', 'visible');
      showTip(event, `${t('chartYears')} ${(day / 252).toFixed(1)} · day ${day}`, series.map(s => ({ color: s.color, value: format(s.values[day]), label: s.label })));
    };
    hit.addEventListener('pointermove', event => {
      const day = Math.round(Math.min(Math.max((pointer(f.svg, event).x - f.left) / (f.right - f.left), 0), 1) * days);
      show(event, day);
    });
    hit.addEventListener('pointerleave', () => { cross.setAttribute('visibility', 'hidden'); hideTip(); });
    f.svg.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
      event.preventDefault();
      focusDay = Math.min(days, Math.max(0, focusDay + (event.key === 'ArrowRight' ? 60 : -60)));
      const box = f.svg.getBoundingClientRect();
      show({ clientX: box.left + (x(focusDay) / f.width) * box.width, clientY: box.top + 20 }, focusDay);
    });
    f.svg.addEventListener('blur', () => { cross.setAttribute('visibility', 'hidden'); hideTip(); });
  };

  const renderDynamics = () => {
    const { run, base } = replay, end = domain().market.trading_days;
    const policy = { label: t('legendPolicy'), ...SERIES.policy }, baseline = { label: t('legendBaseline'), ...SERIES.baseline };
    legend($('dynamics-legend'), [{ ...policy, line: true }, { ...baseline, line: true }]);
    lineChart($('chart-goodness'), [{ ...policy, values: run.meanGoodness }, { ...baseline, values: base.meanGoodness }], t('chartGoodness'), v => signed(v, 2));
    lineChart($('chart-inflation'), [{ ...policy, values: run.meanInflation }, { ...baseline, values: base.meanInflation }], t('chartInflation'), v => v.toFixed(2));
    const rows = [];
    for (let year = 0; year * 252 <= end; year++) {
      const day = year * 252;
      rows.push([year, signed(run.meanGoodness[day], 3), signed(base.meanGoodness[day], 3), run.meanInflation[day].toFixed(3), base.meanInflation[day].toFixed(3)]);
    }
    rows.push([(end / 252).toFixed(1), signed(run.meanGoodness[end], 3), signed(base.meanGoodness[end], 3), run.meanInflation[end].toFixed(3), base.meanInflation[end].toFixed(3)]);
    $('dynamics-table').replaceChildren(table([t('tableYear'), `${t('chartGoodness')} · ${policy.label}`, baseline.label,
      `${t('chartInflation')} · ${policy.label}`, baseline.label], rows, [1, 2, 3, 4]));
  };
  $('audit-seed').addEventListener('change', event => {
    const value = Math.floor(Number(event.target.value));
    state.seed = Number.isFinite(value) && value >= 1 && value <= 4294967295 ? value : 1;
    event.target.value = state.seed;
    update();
  });
  $('seed-next').addEventListener('click', () => { state.seed = state.seed % 4294967295 + 1; $('audit-seed').value = state.seed; update(); });

  const renderCost = () => {
    const container = $('cost-tables');
    container.replaceChildren();
    const range = s => { const cell = node('span', '', signed(s.mean, 3)); cell.append(node('span', 'range', `${signed(s.min, 3)} … ${signed(s.max, 3)}`)); return cell; };
    const rows = Object.values(simplification.rows).map(row => [
      row.natural_decay.toString(), row.policy, range(row.simplified.true_change_sd), signed(row.full_same_seed.true_change_sd, 3),
      signed(row.stage_f_mean.true_change_sd, 3), pct(row.simplified.wash_share.mean), pct(row.full_same_seed.wash_share), pct(row.stage_f_mean.wash_share),
    ]);
    container.append(table([t('costDecay'), t('costScenario'), `${t('costMetric')} · ${t('costSimplified')}`, t('costFull'), t('costStageF'),
      `${t('costWash')} · ${t('costSimplified')}`, t('costFull'), t('costStageF')], rows, [2, 3, 4, 5, 6, 7]));
    const contrasts = Object.values(simplification.contrasts).map(row => [
      row.natural_decay.toString(), row.contrast.replace('_minus_', ' − '), range(row.simplified), signed(row.full_same_seed, 3),
      `${signed(row.stage_f.mean, 3)} [${signed(row.stage_f.ci_low, 3)}, ${signed(row.stage_f.ci_high, 3)}]`,
    ]);
    container.append(table([t('costDecay'), t('costContrast'), t('costSimplified'), t('costFull'), t('costStageF')], contrasts, [2, 3, 4]));
  };

  // -- the desk ----------------------------------------------------------------------------------
  const sphere = desk.createSphere($('sphere'), {
    palette: PALETTE, ink: COLOR.ink, violet: COLOR.violet, pink: '#e87ba4', surface: COLOR.surface, reduced,
    orbiters: { regulator: $('orbit-regulator'), investors: $('orbit-investors') },
    onHover: (i, event) => {
      if (i < 0 || !replay || !event) { hideTip(); return; }
      const d = replay.d, k = state.k;
      const a = k >= 0 ? replay.run.actions[k][i] : IDLE;
      const g = k >= 0 ? replay.goodness[k][i] : d.firms.initial_goodness[i];
      showTip(event, firmOption(d, i), [
        { color: PALETTE[a], value: a === IDLE ? t('sphereIdle') : a === core.MAINTAIN ? t('actionMaintain') : t('factAmount', { action: t(ACTIONS[a].key), x: replay.run.amounts[k][i].toFixed(3) }) },
        { value: t('factSdValue', { x: signed(g, 2) }), label: t('ticketTrue') },
      ]);
    },
    onSelect: i => selectFirm(i),
  });
  const avatarColors = { ink: '#24232b', regulator: COLOR.violet, investor: COLOR.investor };
  $('orbit-regulator').innerHTML = desk.avatar('regulator', 'open', avatarColors);
  $('orbit-investors').insertAdjacentHTML('afterbegin', desk.avatar('investors', 'sleep', avatarColors));
  const setAvatar = (element, kind, mood) => {
    const key = `${kind}|${mood}`;
    if (element.dataset.avatar === key) return;
    element.dataset.avatar = key;
    element.innerHTML = desk.avatar(kind, mood, avatarColors);
  };
  setAvatar($('agent-investors').querySelector('.avatar'), 'investors', 'sleep');
  setAvatar($('agent-regulator').querySelector('.avatar'), 'regulator', 'open');

  const play = { day: 0, playing: !reduced, speed: 1, hold: 0, cursor: 0 };
  const DAYS_PER_SECOND = 60;     // one decision period per second at 1×
  const periodIndex = day => Math.min(replay.run.decisionDays.length - 1, Math.floor(day / domain().market.firm_decision_interval) - 1);

  const renderPolicyAgents = star => {
    const l = state.levers, text = leverText(l);
    const regulatorCard = $('agent-regulator');
    regulatorCard.querySelector('.agent-stat').textContent = t('agentRegulatorStat', { tau: text.tau, audit: text.audit, fixed: text.fixed });
    const card = $('agent-threshold');
    const has = star > 0 && star < 1;
    card.querySelector('.gauge').innerHTML = desk.gauge(l.tau, star, { wash: COLOR.wash, improve: COLOR.improve, ink: COLOR.ink, axis: COLOR.axis });
    card.querySelector('.agent-stat').textContent = has ? t('agentThresholdStat', { tau: text.tau, star: star.toFixed(2) }) : `τ ${text.tau}`;
    card.querySelector('.agent-sub').textContent = !has ? t('agentThresholdNone') : t(l.tau >= star ? 'agentThresholdAbove' : 'agentThresholdBelow');
  };

  const renderAgents = () => {
    const rp = replay, k = state.k;
    const caught = k >= 0 ? rp.caughtPerPeriod.slice(0, k + 1).reduce((s, v) => s + v, 0) : 0;
    const regulatorCard = $('agent-regulator');
    regulatorCard.querySelector('.agent-sub').textContent = t('agentCaught', { n: caught });
    desk.spark(regulatorCard.querySelector('.spark'), rp.caughtPerPeriod, { color: COLOR.ink, current: k, bars: true, max: Math.max(3, ...rp.caughtPerPeriod) });
    ['large', 'mid', 'small'].forEach((key, c) => {
      const card = $(`agent-${key}`), members = rp.groups[c].length;
      const counts = k >= 0 ? rp.counts[k][c] : null;
      card.querySelector('.agent-tag').textContent = t('agentFirmTag', { n: members });
      card.querySelector('.agent-stat').textContent = !counts ? t('agentFirmIdle')
        : counts.map((n, a) => (n || a < 2 ? `${t(ACTIONS[a].key)} ${n}` : '')).filter(Boolean).join(' · ');
      const mood = !counts ? 'open' : counts[1] / members >= 0.5 ? 'shifty' : counts[0] / members >= 0.6 ? 'happy' : 'open';
      setAvatar(card.querySelector('.avatar'), key, mood);
      const bar = card.querySelector('.agent-bar');
      const parts = counts ? counts.map((n, a) => [n, PALETTE[a]]) : [[1, COLOR.idle]];
      if (bar.children.length !== parts.length) bar.replaceChildren(...parts.map(() => node('span')));
      parts.forEach(([n, color], a) => { bar.children[a].style.flexGrow = String(n); bar.children[a].style.background = color; });
      desk.spark(card.querySelector('.spark'), rp.counts.map(p => p[c][1] / members), { color: COLOR.wash, current: k, max: 1 });
    });
  };

  // Ticket: the selected firm through the replay.
  let ticketChart = null;
  const renderTicketChart = () => {
    const rp = replay, d = rp.d, i = state.firm, days = rp.run.decisionDays;
    const xs = [0, ...days];
    const truth = [d.firms.initial_goodness[i], ...rp.goodness.map(g => g[i])];
    const report = truth.map((g, k) => g + (k ? rp.inflation[k - 1][i] : 0));
    const f = frame($('ticket-chart'), 300, 128, [8, 8, 20, 34]);
    f.svg.setAttribute('aria-label', t('ticketChartLabel'));
    const all = truth.concat(report);
    let lo = Math.min(...all), hi = Math.max(...all);
    const pad = (hi - lo || 1) * 0.12;
    lo -= pad; hi += pad;
    const x = scale(0, d.market.trading_days, f.left, f.right), y = scale(lo, hi, f.bottom, f.top);
    yAxis(f, y, lo, hi, 3, v => fixed(v, 1));
    const years = [];
    for (let year = 0; year * 252 <= d.market.trading_days; year += 4) years.push(year * 252);
    xAxis(f, x, years, v => String(v / 252));
    const sw = d.market.burn_in_days + 1;
    svgNode('line', { x1: x(sw), x2: x(sw), y1: f.top, y2: f.bottom, class: 'annotation' }, f.svg);
    const path = values => values.map((v, k) => `${k ? 'L' : 'M'}${x(xs[k]).toFixed(1)},${y(v).toFixed(1)}`).join('');
    const area = `${path(report)}${truth.map((v, k) => `L${x(xs[truth.length - 1 - k]).toFixed(1)},${y(truth[truth.length - 1 - k]).toFixed(1)}`).join('')}Z`;
    svgNode('path', { d: area }, f.svg).style.fill = 'var(--wash-wash)';
    svgNode('path', { d: path(report), fill: 'none', 'stroke-width': 1.6, 'stroke-linejoin': 'round' }, f.svg).style.stroke = 'var(--wash)';
    svgNode('path', { d: path(truth), fill: 'none', 'stroke-width': 2, 'stroke-linejoin': 'round' }, f.svg).style.stroke = 'var(--ink)';
    const head = svgNode('line', { y1: f.top, y2: f.bottom, class: 'playhead' }, f.svg);
    const dot = svgNode('circle', { r: 3.6, class: 'dot' }, f.svg);
    dot.style.fill = 'var(--ink)';
    ticketChart = {
      move: k => {
        const at = k + 1;
        head.setAttribute('x1', x(xs[at])); head.setAttribute('x2', x(xs[at]));
        dot.setAttribute('cx', x(xs[at])); dot.setAttribute('cy', y(truth[at]));
      },
    };
    legend($('ticket-legend'), [{ label: t('ticketTrue'), color: 'var(--ink)', line: true }, { label: t('ticketReport'), color: 'var(--wash)', line: true }]);
  };
  const renderStrip = () => {
    const rp = replay, i = state.firm, k = state.k, strip = $('ticket-strip');
    if (strip.children.length !== rp.run.actions.length) strip.replaceChildren(...rp.run.actions.map(() => node('span')));
    rp.run.actions.forEach((actions, p) => {
      const cell = strip.children[p];
      cell.style.background = PALETTE[actions[i]];
      cell.className = `${p > k ? 'future' : ''}${rp.run.detections[p][i] ? ' caught' : ''}${p === k ? ' now' : ''}`;
      cell.title = `${rp.run.decisionDays[p]} · ${t(ACTIONS[actions[i]].key)}`;
    });
  };
  $('ticket-strip').addEventListener('click', event => {
    const p = [...$('ticket-strip').children].indexOf(event.target);
    if (p >= 0) { setPlaying(false); setDay(replay.run.decisionDays[p], true); }
  });
  const renderTicket = () => {
    const rp = replay, d = rp.d, i = state.firm, k = state.k;
    fillSelect($('ticket-select'), d, d.firms.size_class.length);
    const a = k >= 0 ? rp.run.actions[k][i] : IDLE;
    const choice = $('ticket-choice');
    const left = node('div');
    const what = node('p', 'what');
    const key = node('i');
    key.style.background = PALETTE[a];
    what.append(key, document.createTextNode(a === IDLE ? t('sphereIdle') : t(ACTIONS[a].key)));
    left.append(node('p', 'when', k >= 0 ? t('ticketNow', { day: rp.run.decisionDays[k] }) : t('ticketIdle')), what);
    choice.replaceChildren(left, node('span', 'amount', a === IDLE || a === core.MAINTAIN ? '' : `Δ ${rp.run.amounts[k][i].toFixed(3)} SD`));
    let caught = 0, fines = 0;
    for (let p = 0; p <= k; p++) if (rp.run.detections[p][i]) { caught += 1; fines += rp.run.fines[p][i] * d.firms.market_cap[i] * YUAN_PER_UNIT; }
    const facts = $('ticket-facts');
    facts.replaceChildren();
    const fact = (label, value) => { const item = node('div'); item.append(node('dt', '', label), node('dd', '', value)); facts.append(item); };
    fact(t('factSize'), t(SIZES[d.firms.size_class[i]]));
    fact(t('factCap'), t('factCapValue', { x: grouped(capYi(d, i)) }));
    fact(t('factBenefit'), t('factBenefitValue', { x: (d.firms.benefit[i] / BP).toFixed(2) }));
    fact(t('ticketTrue'), t('factSdValue', { x: signed(k >= 0 ? rp.goodness[k][i] : d.firms.initial_goodness[i], 2) }));
    fact(t('factInflation'), t('factSdValue', { x: (k >= 0 ? rp.inflation[k][i] : 0).toFixed(2) }));
    fact(t('ticketCaught'), `${t('ticketCaughtValue', { n: caught })}${caught ? ` · ${money(fines)}` : ''}`);
    if (!ticketChart || ticketChart.firm !== i || ticketChart.replay !== rp || ticketChart.lang !== lang) {
      renderTicketChart();
      Object.assign(ticketChart, { firm: i, replay: rp, lang });
    }
    ticketChart.move(k);
    renderStrip();
  };

  // Policy window: hero figure, tiles and the mean true level with a playhead.
  let windowChart = null;
  const renderWindow = () => {
    const { m, mb, run, base, d } = replay;
    const tiles = $('dynamics-tiles');
    tiles.replaceChildren();
    const tile = (label, value, sub) => {
      const item = node('div', 'tile');
      item.append(node('p', 'tile-label', label), node('p', 'tile-value', value), node('p', 'tile-sub', sub));
      tiles.append(item);
    };
    tile(t('tileChange'), `${signed(m.true_change_sd, 2)} SD`,
      `${t('tileVsBase', { x: signed(mb.true_change_sd, 2) })} · ${t('tileDelta', { x: signed(m.true_change_sd - mb.true_change_sd, 3) })}`);
    tile(t('tileWash'), pct(m.wash_share), t('tileVsBase', { x: pct(mb.wash_share) }));
    tile(t('tileInflation'), `${m.inflation_mean_sd.toFixed(2)} SD`, t('tileVsBase', { x: mb.inflation_mean_sd.toFixed(2) }));
    const policy = { label: t('legendPolicy'), ...SERIES.policy }, baseline = { label: t('legendBaseline'), ...SERIES.baseline };
    legend($('window-legend'), [{ ...policy, line: true }, { ...baseline, line: true }]);
    const f = frame($('window-chart'), 440, 156, [10, 12, 26, 40]);
    f.svg.setAttribute('aria-label', t('chartGoodness'));
    const days = d.market.trading_days;
    const all = [...run.meanGoodness, ...base.meanGoodness];
    let lo = Math.min(...all), hi = Math.max(...all);
    const pad = (hi - lo || 1) * 0.1;
    lo -= pad; hi += pad;
    const x = scale(0, days, f.left, f.right), y = scale(lo, hi, f.bottom, f.top);
    yAxis(f, y, lo, hi, 3, v => fixed(v, 1));
    const years = [];
    for (let year = 0; year * 252 <= days; year += 2) years.push(year * 252);
    xAxis(f, x, years, v => String(v / 252));
    const sw = d.market.burn_in_days + 1;
    svgNode('rect', { x: x(sw), y: f.top, width: f.right - x(sw), height: f.bottom - f.top }, f.svg).style.fill = '#6250d60a';
    svgNode('line', { x1: x(sw), x2: x(sw), y1: f.top, y2: f.bottom, class: 'annotation' }, f.svg);
    svgNode('text', { x: x(sw) + 5, y: f.top + 9, class: 'annotation-text' }, f.svg, t('chartSwitch'));
    for (const s of [{ ...baseline, values: base.meanGoodness }, { ...policy, values: run.meanGoodness }]) {
      let path = '';
      for (let day = 0; day <= days; day += 5) path += `${day ? 'L' : 'M'}${x(day).toFixed(1)},${y(s.values[day]).toFixed(1)}`;
      path += `L${x(days).toFixed(1)},${y(s.values[days]).toFixed(1)}`;
      svgNode('path', { d: path, fill: 'none', 'stroke-width': 2, 'stroke-linejoin': 'round' }, f.svg).style.stroke = s.color;
    }
    svgNode('text', { x: f.right, y: y(run.meanGoodness[days]) - 8, 'text-anchor': 'end', class: 'label' }, f.svg, signed(run.meanGoodness[days], 2));
    const head = svgNode('line', { y1: f.top, y2: f.bottom, class: 'playhead' }, f.svg);
    const dot = svgNode('circle', { r: 4, class: 'dot' }, f.svg);
    dot.style.fill = 'var(--ink)';
    const cross = svgNode('line', { y1: f.top, y2: f.bottom, class: 'crosshair', visibility: 'hidden' }, f.svg);
    const hit = svgNode('rect', { x: f.left, y: f.top, width: f.right - f.left, height: f.bottom - f.top, class: 'hit' }, f.svg);
    hit.addEventListener('pointermove', event => {
      const day = Math.round(clamp((pointer(f.svg, event).x - f.left) / (f.right - f.left), 0, 1) * days);
      cross.setAttribute('x1', x(day)); cross.setAttribute('x2', x(day)); cross.setAttribute('visibility', 'visible');
      showTip(event, `${t('chartYears')} ${(day / 252).toFixed(1)} · day ${day}`, [
        { color: policy.color, value: signed(run.meanGoodness[day], 2), label: policy.label },
        { color: baseline.color, value: signed(base.meanGoodness[day], 2), label: baseline.label }]);
    });
    hit.addEventListener('pointerleave', () => { cross.setAttribute('visibility', 'hidden'); hideTip(); });
    hit.addEventListener('click', event => {
      setDay(Math.round(clamp((pointer(f.svg, event).x - f.left) / (f.right - f.left), 0, 1) * days), true);
    });
    windowChart = {
      move: day => {
        const at = Math.round(clamp(day, 0, days));
        head.setAttribute('x1', x(at)); head.setAttribute('x2', x(at));
        dot.setAttribute('cx', x(at)); dot.setAttribute('cy', y(run.meanGoodness[at]));
      },
    };
  };

  // Decision spectrogram.
  let spectroHit = null;
  const drawSpectro = () => {
    const rp = replay;
    spectroHit = desk.spectrogram($('spectro'), {
      order: rp.order, groupStarts: [0, rp.groups[0].length, rp.groups[0].length + rp.groups[1].length],
      groupLabels: SIZES.map(key => t(key)), actions: rp.run.actions, detections: rp.run.detections,
      palette: PALETTE, ink: COLOR.ink, surface: COLOR.surface, violet: COLOR.violet, current: state.k, switchK: rp.switchK,
      switchLabel: t('chartSwitch'), selectedRow: rp.rowOf[state.firm], days: rp.d.market.trading_days,
      interval: rp.d.market.firm_decision_interval, yearLabel: t('chartYears'),
    });
    $('spectro').setAttribute('aria-label', t('spectroAria', { day: Math.round(play.day) }));
  };
  const spectroAt = event => { const box = $('spectro').getBoundingClientRect(); return spectroHit?.hit(event.clientX - box.left, event.clientY - box.top); };
  $('spectro').addEventListener('pointermove', event => {
    const at = spectroAt(event);
    if (!at) { hideTip(); return; }
    const rp = replay, d = rp.d, a = rp.run.actions[at.k][at.i];
    const rows = [{ color: PALETTE[a], value: a === core.MAINTAIN ? t('actionMaintain') : t('factAmount', { action: t(ACTIONS[a].key), x: rp.run.amounts[at.k][at.i].toFixed(3) }) }];
    if (rp.run.detections[at.k][at.i]) rows.push({ ring: true, color: COLOR.ink, value: money(rp.run.fines[at.k][at.i] * d.firms.market_cap[at.i] * YUAN_PER_UNIT), label: t('spectroCaught') });
    showTip(event, `${firmOption(d, at.i)} · ${t('traceDay')} ${rp.run.decisionDays[at.k]}`, rows);
  });
  $('spectro').addEventListener('pointerleave', hideTip);
  $('spectro').addEventListener('click', event => {
    const at = spectroAt(event);
    if (!at) return;
    setPlaying(false);
    selectFirm(at.i);
    setDay(replay.run.decisionDays[at.k], true);
  });
  $('spectro').addEventListener('keydown', event => {
    const keys = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    if (!(event.key in keys)) return;
    event.preventDefault();
    const [dk, dr] = keys[event.key], rp = replay;
    if (dr) selectFirm(rp.order[clamp(rp.rowOf[state.firm] + dr, 0, rp.order.length - 1)]);
    if (dk) { setPlaying(false); setDay(rp.run.decisionDays[clamp(state.k + dk, 0, rp.run.decisionDays.length - 1)], true); }
  });
  const renderSpectroTable = () => {
    const { run } = replay;
    legend($('spectro-legend'), [...ACTIONS.map(a => ({ label: t(a.key), color: a.color, band: true })), { label: t('spectroCaught'), ring: true }]);
    const rows = run.decisionDays.map((day, k) => {
      const c = [0, 0, 0];
      for (const a of run.actions[k]) c[a] += 1;
      return [day, c[0], c[1], c[2], replay.caughtPerPeriod[k]];
    });
    $('spectro-table').replaceChildren(table([t('spectroPeriod'), t('actionImprove'), t('actionWash'), t('actionMaintain'), t('spectroCaughtCount')], rows, [0, 1, 2, 3, 4]));
  };

  // Live wire and trace.
  const sizeName = i => t(SIZES[replay.d.firms.size_class[i]]);
  const eventItems = e => {
    const text = l => leverText(l);
    if (e.type === 'start') return [{ text: t('evStart'), mark: 'neutral' }];
    if (e.type === 'end') return [{ text: t('evEnd', { n: e.total }), mark: 'neutral' }];
    if (e.type === 'switch') return [{ text: t('evSwitch', text(state.levers)), mark: 'policy' }];
    if (e.type === 'first') return [{ text: t('agentFirmStat', { improve: e.counts[0], wash: e.counts[1], maintain: e.counts[2] }), mark: 'neutral' }];
    if (e.type === 'caught') return [{ text: t('evCaught', { id: firmId(e.i), size: sizeName(e.i), fine: money(e.fine) }), mark: 'caught' }];
    const from = t(ACTIONS[e.from].key), to = t(ACTIONS[e.to].key);
    if (e.firms.length <= 2) return e.firms.map(i => ({ text: t('evTurn', { id: firmId(i), size: sizeName(i), from, to }), color: PALETTE[e.to] }));
    return [{ text: t('evTurnMany', { n: e.firms.length, from, to }), color: PALETTE[e.to] }];
  };
  const WIRE_MAX = 14, TRACE_MAX = 60;
  const wireNode = (e, item) => {
    const li = node('li');
    const mark = node('i', `mark ${item.mark ?? ''}`);
    if (item.color) mark.style.background = item.color;
    if (item.mark === 'neutral') mark.style.background = COLOR.axis;
    li.append(node('span', 'day', String(e.day).padStart(4, '0')), mark, document.createTextNode(item.text));
    return li;
  };
  const traceRows = e => {
    if (e.type === 'caught') return [[e.day, `${firmId(e.i)} · ${sizeName(e.i)}`, ['caught', t('traceCaught')], money(e.fine)]];
    if (e.type === 'switch') return [[e.day, '—', ['policy', t('traceSwitch')], `τ ${state.levers.tau.toFixed(2)}`]];
    if (e.type === 'turn') {
      const detail = `${t(ACTIONS[e.from].key)} → ${t(ACTIONS[e.to].key)}`;
      if (e.firms.length <= 2) return e.firms.map(i => [e.day, `${firmId(i)} · ${sizeName(i)}`, [PALETTE[e.to], t('traceTurn')], detail]);
      return [[e.day, `× ${e.firms.length}`, [PALETTE[e.to], t('traceTurn')], detail]];
    }
    return [];
  };
  const traceNode = (cells, fresh) => {
    const tr = node('tr', fresh ? 'fresh' : '');
    cells.forEach((cell, c) => {
      const td = node('td', c === 3 ? 'num' : '');
      if (c === 2) {
        const [kind, label] = cell;
        const ev = node('span', 'ev');
        const mark = node('i', kind === 'caught' || kind === 'policy' ? kind : '');
        if (kind !== 'caught' && kind !== 'policy') mark.style.background = kind;
        ev.append(mark, document.createTextNode(label));
        td.append(ev);
      } else td.textContent = String(cell);
      tr.append(td);
    });
    return tr;
  };
  const traceTotals = () => {
    let n = 0, total = 0;
    for (let c = 0; c < play.cursor; c++) { const e = replay.events[c]; if (e.type === 'caught') { n += 1; total += e.fine; } }
    $('trace-total').textContent = t('traceTotal', { n, x: money(total) });
    $('trace-empty').hidden = n > 0;
  };
  const rebuildFeeds = () => {
    const events = replay.events;
    let cursor = 0;
    while (cursor < events.length && events[cursor].day <= play.day) cursor += 1;
    play.cursor = cursor;
    const wire = [], trace = [];
    for (let c = cursor - 1; c >= 0 && (wire.length < WIRE_MAX || trace.length < TRACE_MAX); c--) {
      const e = events[c];
      if (wire.length < WIRE_MAX) for (const item of eventItems(e).reverse()) if (wire.length < WIRE_MAX) wire.push(wireNode(e, item));
      if (trace.length < TRACE_MAX) for (const cells of traceRows(e).reverse()) if (trace.length < TRACE_MAX) trace.push(traceNode(cells, false));
    }
    $('wire-feed').replaceChildren(...wire);
    $('trace-body').replaceChildren(...trace);
    traceTotals();
  };
  const pushEvents = until => {
    const events = replay.events, wire = $('wire-feed'), trace = $('trace-body');
    let added = false;
    while (play.cursor < events.length && events[play.cursor].day <= until) {
      const e = events[play.cursor];
      play.cursor += 1;
      for (const item of eventItems(e)) wire.prepend(wireNode(e, item));
      for (const cells of traceRows(e)) trace.prepend(traceNode(cells, true));
      added = true;
    }
    if (!added) return;
    while (wire.children.length > WIRE_MAX) wire.lastElementChild.remove();
    while (trace.children.length > TRACE_MAX) trace.lastElementChild.remove();
    traceTotals();
  };

  // One decision period on every desk view.
  const applyPeriod = (k, { ripple = false, beams = false } = {}) => {
    const rp = replay;
    state.k = k;
    const n = rp.d.firms.size_class.length;
    sphere.setColors(k >= 0 ? rp.run.actions[k] : new Array(n).fill(IDLE), { ripple });
    if (beams && k >= 0) {
      const caught = [];
      rp.run.detections[k].forEach((v, i) => { if (v) caught.push(i); });
      if (caught.length) sphere.beam(caught);
    }
    const counts = [0, 0, 0];
    if (k >= 0) for (const a of rp.run.actions[k]) counts[a] += 1;
    $('sphere').setAttribute('aria-label', t('sphereAria', { day: Math.round(play.day), improve: counts[0], wash: counts[1], maintain: counts[2] }));
    renderAgents();
    renderTicket();
    drawSpectro();
  };
  let lastClock = -1;
  const updateClock = () => {
    const day = Math.round(play.day);
    if (day === lastClock) return;
    lastClock = day;
    const d = replay.d, total = d.market.trading_days;
    $('clock-day').textContent = grouped(day);
    $('clock-sub').textContent = t('clockSub', { y: (day / 252).toFixed(1), total: grouped(total) });
    const after = day > d.market.burn_in_days;
    const chip = $('clock-phase');
    chip.textContent = after ? t('legendPolicy') : t('legendBaseline');
    chip.classList.toggle('after', after);
    $('scrub').value = String(day);
    $('scrub').style.setProperty('--fill', `${100 * day / total}%`);
    windowChart?.move(day);
    $('window-now').textContent = t('windowNow', { day, x: signed(replay.run.meanGoodness[day], 2) });
  };
  const setDay = (day, jump = false) => {
    const previous = play.day;
    play.day = clamp(day, 0, replay.d.market.trading_days);
    const k = periodIndex(play.day);
    if (jump || play.day < previous) {
      rebuildFeeds();
      applyPeriod(k);
    } else {
      pushEvents(play.day);
      if (k !== state.k) applyPeriod(k, { ripple: true, beams: true });
    }
    updateClock();
  };
  const setPlaying = playing => {
    play.playing = playing;
    play.hold = 0;
    const button = $('play');
    button.classList.toggle('playing', playing);
    button.setAttribute('aria-label', t(playing ? 'pauseLabel' : 'playLabel'));
    wake();
  };
  $('play').addEventListener('click', () => {
    if (!play.playing && play.day >= replay.d.market.trading_days) setDay(0, true);
    setPlaying(!play.playing);
  });
  $('restart').addEventListener('click', () => { setDay(0, true); setPlaying(true); });
  $('scrub').addEventListener('input', event => setDay(Number(event.target.value), true));
  for (const button of document.querySelectorAll('[data-speed]')) {
    button.addEventListener('click', () => {
      play.speed = Number(button.dataset.speed);
      for (const other of document.querySelectorAll('[data-speed]')) other.setAttribute('aria-pressed', String(other === button));
    });
  }

  // Geometry of the stage for the canvas: sphere between the HUD rows, lever ports, ticket anchor.
  const layoutStage = () => {
    const box = $('sphere').getBoundingClientRect();
    const rel = element => { const r = element.getBoundingClientRect(); return { x: r.left - box.left, y: r.top - box.top, w: r.width, h: r.height }; };
    const hud = rel(document.querySelector('.hud'));
    const top = rel(document.querySelector('.hud-top')), bottom = rel(document.querySelector('.hud-bottom'));
    const areaTop = top.y + top.h + 6, areaBottom = bottom.y - 10;
    const r = Math.max(70, Math.min(hud.w * (hud.w < 560 ? 0.42 : 0.34), (areaBottom - areaTop) / 2, 232));
    const ports = [...document.querySelectorAll('.lever .port')].map(rel).filter(p => p.w > 0).map(p => ({ x: p.x + p.w / 2, y: p.y + p.h / 2 }));
    const ticket = rel(document.querySelector('.ticket')), choice = rel($('ticket-choice'));
    const cx = hud.x + hud.w / 2;
    sphere.setGeometry({ cx, cy: (areaTop + areaBottom) / 2, r, ports,
      ticket: ticket.x > cx ? { x: ticket.x, y: choice.y + choice.h / 2 } : null,
      right: ticket.x > cx ? ticket.x - 8 : hud.x + hud.w });
    const scrub = $('scrub');
    $('scrub-switch').style.left = `${100 * (replay ? replay.d.market.burn_in_days + 1 : 1201) / Number(scrub.max)}%`;
  };
  new ResizeObserver(() => layoutStage()).observe($('stage'));

  // The animation loop runs only while the stage is on screen.
  let raf = 0, visible = true, lastNow = 0;
  const loop = now => {
    // Book the next frame first, so one failed frame cannot stop the desk.
    raf = visible && !document.hidden ? requestAnimationFrame(loop) : 0;
    const dt = lastNow ? Math.min(0.1, (now - lastNow) / 1000) : 0;
    lastNow = now;
    if (replay && play.playing) {
      const end = replay.d.market.trading_days;
      if (play.hold) {
        if (now >= play.hold) { play.hold = 0; setDay(0, true); }
      } else {
        const next = play.day + dt * DAYS_PER_SECOND * play.speed;
        if (next >= end) { setDay(end); play.hold = now + 2600; } else setDay(next);
      }
    }
    sphere.frame(now);
  };
  const wake = () => { if (!raf && visible && !document.hidden) { lastNow = 0; raf = requestAnimationFrame(loop); } };
  new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; wake(); }).observe($('stage'));
  document.addEventListener('visibilitychange', wake);
  const policyBar = $('policy-bar');
  new IntersectionObserver(([entry]) => { policyBar.hidden = entry.isIntersecting || entry.boundingClientRect.top > 0; },
    { rootMargin: '-52px 0px 0px 0px' }).observe($('levers'));

  const renderDesk = () => {
    legend($('sphere-legend'), ACTIONS.map(a => ({ label: t(a.key), color: a.color })));
    sphere.setBandLabels(replay.groups.map((g, k) => t('bandLabel', { size: t(SIZES[k]), n: g.length })));
    $('scrub').max = String(replay.d.market.trading_days);
    renderWindow();
    renderSpectroTable();
    rebuildFeeds();
    applyPeriod(periodIndex(play.day), { ripple: true });
    lastClock = -1;
    updateClock();
    layoutStage();
  };

  // -- panel 3: stage F results --------------------------------------------------------------
  const primary = name => stageF.primary.find(row => row.name === name);
  const bracketRows = () => [
    { key: 'contrastIB', low: stageF.secondary.decay0_I_minus_B, high: stageF.secondary.I_minus_B },
    { key: 'contrastFB', low: stageF.secondary.decay0_F_minus_B, high: stageF.secondary.F_minus_B },
    { key: 'contrastIF', low: stageF.secondary.decay0_I_minus_F, high: primary('P2_I_minus_F') },
  ];
  const ends = [{ key: 'bracketLow', field: 'low', color: 'var(--mixed)' }, { key: 'bracketHigh', field: 'high', color: 'var(--improve)' }];

  const renderBracket = () => {
    const rows = bracketRows();
    legend($('bracket-legend'), ends.map(e => ({ label: t(e.key), color: e.color })));
    const f = frame($('bracket-chart'), 600, 64 + rows.length * 62, [10, 26, 34, 180]);
    f.svg.setAttribute('aria-label', t('bracketTitle'));
    const hi = Math.max(...rows.map(r => r.high.ci_high)) * 1.12;
    const x = scale(0, hi, f.left, f.right);
    const grid = svgNode('g', { class: 'grid' }, f.svg);
    const xt = ticks(0, hi, 5);
    for (const v of xt.values) {
      svgNode('line', { x1: x(v), x2: x(v), y1: f.top, y2: f.bottom }, grid);
      svgNode('text', { x: x(v), y: f.bottom + 16, 'text-anchor': 'middle' }, f.svg, v.toFixed(decimals(xt.step)));
    }
    svgNode('line', { x1: x(0), x2: x(0), y1: f.top, y2: f.bottom, class: 'baseline' }, f.svg);
    svgNode('text', { x: f.right, y: f.bottom + 32, 'text-anchor': 'end', class: 'axis-title' }, f.svg, 'SD');
    rows.forEach((row, k) => {
      const cy = f.top + 30 + k * 62;
      svgNode('text', { x: 8, y: cy + 4, class: 'label' }, f.svg, t(row.key));
      svgNode('line', { x1: x(row.low.mean), x2: x(row.high.mean), y1: cy, y2: cy, 'stroke-width': 2 }, f.svg).style.stroke = 'var(--axis)';
      for (const e of ends) {
        const v = row[e.field];
        svgNode('line', { x1: x(v.ci_low), x2: x(v.ci_high), y1: cy, y2: cy, 'stroke-width': 1.5 }, f.svg).style.stroke = e.color;
        for (const edge of [v.ci_low, v.ci_high]) svgNode('line', { x1: x(edge), x2: x(edge), y1: cy - 5, y2: cy + 5, 'stroke-width': 1.5 }, f.svg).style.stroke = e.color;
        const dot = svgNode('circle', { cx: x(v.mean), cy, r: 5, class: 'dot' }, f.svg);
        dot.style.fill = e.color;
        svgNode('text', { x: x(v.mean), y: cy + (e.field === 'low' ? 22 : -12), 'text-anchor': 'middle', class: 'label' }, f.svg, signed(v.mean, 3));
        const hit = svgNode('circle', { cx: x(v.mean), cy, r: 13, class: 'hit' }, f.svg);
        const tipRows = () => [{ color: e.color, value: `${signed(v.mean, 3)} [${signed(v.ci_low, 3)}, ${signed(v.ci_high, 3)}]`, label: t(e.key) }];
        hit.addEventListener('pointermove', event => showTip(event, t(row.key), tipRows()));
        hit.addEventListener('pointerleave', hideTip);
      }
    });
    $('bracket-table').replaceChildren(table([t('costContrast'), t('bracketLow'), t('bracketHigh')], rows.map(row => [t(row.key),
      ...ends.map(e => `${signed(row[e.field].mean, 3)} [${signed(row[e.field].ci_low, 3)}, ${signed(row[e.field].ci_high, 3)}]`)]), [1, 2]));
  };

  const renderPrimary = () => {
    $('primary-table').replaceChildren(table([t('primaryQuestion'), t('primaryEstimate'), t('primaryHolm')], stageF.primary.map(row => [
      t(row.name), `${signed(row.mean, 3)} [${signed(row.ci_low, 3)}, ${signed(row.ci_high, 3)}]`, sci(row.holm_p_value),
    ]), [1, 2]));
  };

  const doseChart = (container, title, points, yLabel) => {
    container.replaceChildren(node('h4', '', title));
    const holder = node('div', 'chart');
    container.append(holder);
    const f = frame(holder, 360, 200, [16, 14, 34, 46]);
    f.svg.setAttribute('aria-label', title);
    const xs = points.map(p => p.x), ys = points.flatMap(p => [p.lo ?? p.y, p.hi ?? p.y, 0]);
    const xLo = Math.min(...xs), xHi = Math.max(...xs);
    let yLo = Math.min(...ys), yHi = Math.max(...ys);
    const pad = (yHi - yLo) * 0.12;
    yLo -= pad; yHi += pad;
    const yT = ticks(yLo, yHi, 4);
    const xPad = (xHi - xLo) * 0.06;
    const x = scale(xLo - xPad, xHi + xPad, f.left, f.right), y = scale(yLo, yHi, f.bottom, f.top);
    yAxis(f, y, yLo, yHi, 4, v => v.toFixed(decimals(yT.step)));
    xAxis(f, x, xs, v => points.find(p => p.x === v).label);
    svgNode('line', { x1: f.left, x2: f.right, y1: y(0), y2: y(0), class: 'baseline' }, f.svg);
    svgNode('text', { x: f.left, y: f.top - 4, class: 'axis-title' }, f.svg, yLabel);
    const path = points.map((p, k) => `${k ? 'L' : 'M'}${x(p.x).toFixed(1)},${y(p.y).toFixed(1)}`).join('');
    svgNode('path', { d: path, fill: 'none', 'stroke-width': 2 }, f.svg).style.stroke = 'var(--improve)';
    for (const p of points) {
      if (p.lo !== undefined) svgNode('line', { x1: x(p.x), x2: x(p.x), y1: y(p.lo), y2: y(p.hi), 'stroke-width': 1.5 }, f.svg).style.stroke = 'var(--improve)';
      svgNode('circle', { cx: x(p.x), cy: y(p.y), r: 4.5, class: 'dot' }, f.svg).style.fill = 'var(--improve)';
      const hit = svgNode('circle', { cx: x(p.x), cy: y(p.y), r: 13, class: 'hit' }, f.svg);
      hit.addEventListener('pointermove', event => showTip(event, `${title}: ${p.label}`, [
        { color: 'var(--improve)', value: `${signed(p.y, 3)}${p.lo !== undefined ? ` [${signed(p.lo, 3)}, ${signed(p.hi, 3)}]` : ''}`, label: 'SD' },
        { value: pct(p.wash), label: t('doseWash') }]));
      hit.addEventListener('pointerleave', hideTip);
    }
    container.append(table([title, 'SD', t('doseWash')], points.map(p => [p.label, signed(p.y, 3), pct(p.wash)]), [1, 2]));
  };

  const renderDose = () => {
    const s = stageF.secondary, sc = stageF.scenarios;
    const ci = row => ({ y: row.mean, lo: row.ci_low, hi: row.ci_high });
    const grid = $('dose-grid');
    grid.replaceChildren();
    const cells = [node('div'), node('div'), node('div')];
    grid.append(...cells);
    doseChart(cells[0], t('doseTau'), [
      { x: 0.1, label: '0.1', y: 0, wash: sc.B.wash },
      ...[['0.3', 'I_0p3'], ['0.5', 'I_0p5'], ['0.6', 'I'], ['0.7', 'I_0p7'], ['0.8', 'I_0p8']].map(([label, name]) => ({
        x: Number(label), label, ...ci(s[`${name}_minus_B`]), wash: sc[name].wash })),
    ], t('doseY'));
    doseChart(cells[1], t('doseFixed'), [
      { x: 0, label: '0×', ...ci(s.F_0x_minus_B), wash: sc.F_0x.wash }, { x: 1, label: '1×', y: 0, wash: sc.B.wash },
      { x: 2, label: '2×', ...ci(s.F_2x_minus_B), wash: sc.F_2x.wash }, { x: 4, label: '4×', ...ci(s.F_minus_B), wash: sc.F.wash },
      { x: 8, label: '8×', ...ci(s.F_8x_minus_B), wash: sc.F_8x.wash },
    ], t('doseY'));
    doseChart(cells[2], t('doseProp'), [1, 2, 4].map(k => ({ x: k, label: `${k}×`, ...ci(s[`P_${k}x_minus_fixed_${k}x`]), wash: sc[`P_${k}x`].wash })), t('doseYProp'));
  };

  const CLASS = {
    real_improvement: ['classReal', 'class-real'], washing_trap: ['classTrap', 'class-trap'], low_low: ['classLow', 'class-low'],
    mixed: ['classMixed', 'class-mixed'], uncertain: ['classUncertain', 'class-uncertain'],
  };
  const phaseTable = (rowsLabel, colsLabel, rowValues, colValues, cell) => {
    const element = node('table', 'phase-table');
    const head = element.createTHead().insertRow();
    const corner = node('th', '', `${rowsLabel} \\ ${colsLabel}`);
    head.append(corner);
    for (const c of colValues) { const th = node('th', '', String(c)); th.scope = 'col'; head.append(th); }
    const body = element.createTBody();
    for (const r of rowValues) {
      const tr = body.insertRow();
      const th = node('th', '', String(r)); th.scope = 'row'; tr.append(th);
      for (const c of colValues) tr.append(cell(r, c));
    }
    return element;
  };
  const renderPhase = () => {
    const star = Object.fromEntries(stageF.tau_star_check.map(row => [`${row.w}|${row.tau}`, row]));
    $('phase2').replaceChildren(phaseTable(t('phaseW'), t('phaseTau'), [0.05, 0.1, 0.2, 0.4], [0.1, 0.3, 0.6, 0.8], (w, tau) => {
      const entry = stageF.phase_map_2[`${w}|${tau}`];
      const check = star[`${w}|${tau}`];
      const dominant = entry.decay_adjusted.dominant;
      const td = node('td', CLASS[dominant][1], t(CLASS[dominant][0]));
      const previous = [0.1, 0.3, 0.6, 0.8][[0.1, 0.3, 0.6, 0.8].indexOf(tau) - 1];
      if (!check.predicted_washing && (previous === undefined || star[`${w}|${previous}`].predicted_washing)) td.classList.add('threshold');
      const correct = (check.wash_share > 0) === check.predicted_washing;
      td.append(node('small', '', `${pct(check.wash_share)} · ${t(check.predicted_washing ? 'predictedYes' : 'predictedNo')} ${correct ? '✓' : '✗'}`));
      td.title = `τ*(${w}) = ${check.tau_star.toFixed(2)}`;
      return td;
    }));
    // Keys as written by the stage F analysis: taste|initial share.
    $('phase1').replaceChildren(phaseTable(t('phaseTaste'), t('phaseShare'), ['0.0', '0.0362', '0.0724', '0.1448'], ['0.0', '0.2', '0.5', '0.8'], (taste, share) => {
      const entry = stageF.phase_map_1[`${taste}|${share}`];
      const dominant = entry.decay_adjusted.dominant;
      const td = node('td', CLASS[dominant][1], t(CLASS[dominant][0]));
      const counts = entry.decay_adjusted.counts;
      td.append(node('small', '', `${counts.real_improvement}/${counts.washing_trap}/${counts.low_low}/${counts.mixed}`));
      return td;
    }));
  };

  const renderProvenance = () => {
    const source = stageF.source;
    const list = $('provenance');
    list.replaceChildren();
    const fact = (label, value, href) => {
      const item = node('div');
      const dd = node('dd');
      if (href) { const link = node('a', '', value); link.href = href; dd.append(link); } else dd.textContent = value;
      item.append(node('dt', '', label), dd);
      list.append(item);
    };
    fact(t('provExperiment'), `experiment_id ${source.experiment_id.slice(0, 12)}…${source.experiment_id.slice(-4)}`);
    fact(t('provRuns'), t('provRunsValue', { runs: source.runs, planned: source.planned, seeds: source.seeds }));
    fact(t('provCommit'), source.commit.slice(0, 7));
    fact(t('provFile'), `${source.path} · sha256 ${source.sha256.slice(0, 12)}…`, `${REPO}/blob/master/${source.path}`);
    fact(t('provReport'), source.report, `${REPO}/blob/master/${source.report}`);
    fact(t('provContract'), 'docs/disclosure_research_contract.md §20', `${REPO}/blob/master/docs/disclosure_research_contract.md`);
  };

  const renderResults = () => {
    const ai = state.domain === 'ai_washing';
    $('results-ai').hidden = !ai;
    $('results-body').hidden = ai;
    if (ai) return;
    renderBracket(); renderPrimary(); renderDose(); renderPhase(); renderProvenance();
  };

  // -- page text and orchestration -------------------------------------------------------------
  const renderHero = () => {
    const w02 = stageF.tau_star_check.find(row => row.w === 0.2);
    $('finding2-value').textContent = t('finding2Value', { tau: w02.tau_star.toFixed(2) });
    $('finding3-value').textContent = t('finding3Value', {
      lo: stageF.secondary.decay0_I_minus_B.mean.toFixed(2), hi: stageF.secondary.I_minus_B.mean.toFixed(2) });
  };
  const applyText = () => {
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
    document.title = t('pageTitle');
    for (const element of document.querySelectorAll('[data-i18n]')) element.textContent = t(element.dataset.i18n);
    for (const element of document.querySelectorAll('[data-i18n-aria]')) element.setAttribute('aria-label', t(element.dataset.i18nAria));
    // Carry the language to the stage 1 exhibit, which shares the 'abm-lang' preference.
    for (const element of document.querySelectorAll('[data-lang-link]')) element.setAttribute('href', `${element.dataset.langLink}?lang=${lang}`);
    $('lang-toggle').setAttribute('lang', lang === 'zh' ? 'en' : 'zh');
    $('play').setAttribute('aria-label', t(play.playing ? 'pauseLabel' : 'playLabel'));
    renderHero();
  };
  // Coalesce slider events into one render per frame; the timer covers hidden tabs, where
  // animation frames do not fire.
  let pending = false;
  const flush = () => {
    if (!pending) return;
    pending = false;
    renderDecision();
    replay = computeReplay();
    renderDesk();
    renderDynamics();
    writeLink();
  };
  const update = () => {
    syncLevers();
    if (pending) return;
    pending = true;
    requestAnimationFrame(flush);
    setTimeout(flush, 120);
  };
  const setDomain = (key, preset = null, linked = { levers: {}, firm: null }) => {
    state.domain = key;
    const d = domain();
    state.levers = { ...(preset ? presetLevers(preset) : policyLevers(d)), decay: d.natural_decay, ...linked.levers };
    state.firm = linked.firm ?? defaultFirm(d);
    const groups = sizeGroups(d);
    const caps = d.firms.market_cap, low = Math.min(...caps);
    sphere.setFirms(groups, caps.map(c => clamp(2.1 + 1.3 * Math.log10(c / low), 2.1, 5.8)));
    sphere.select(state.firm);
    ticketChart = null;
    renderResults();
    update();
  };
  $('lang-toggle').addEventListener('click', () => {
    lang = lang === 'zh' ? 'en' : 'zh';
    storage.set(lang);
    applyText(); renderCost(); renderResults(); update();
  });
  $('share-link').addEventListener('click', async () => {
    const url = shareUrl(), status = $('share-status');
    const field = $('share-field');
    field.value = url;
    let copied = false;
    try { await navigator.clipboard.writeText(url); copied = true; } catch (e) {
      // Embedded or file:// pages may refuse the clipboard API; fall back to a selected field.
      field.hidden = false; field.select();
      try { copied = document.execCommand('copy'); } catch (error) { copied = false; }
    }
    status.textContent = copied ? t('shareCopied') : t('shareManual');
    clearTimeout(status.timer);
    status.timer = setTimeout(() => { status.textContent = ''; }, 4000);
  });
  document.querySelectorAll('.segmented').forEach(group => group.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    const buttons = [...group.querySelectorAll('button')];
    const i = buttons.indexOf(document.activeElement);
    if (i < 0) return;
    event.preventDefault();
    buttons[(i + (event.key === 'ArrowRight' ? 1 : -1) + buttons.length) % buttons.length].focus();
  }));
  addEventListener('scroll', hideTip, { passive: true });

  applyText();
  renderCost();
  // The desk opens on the information policy a few periods before the switch, so the first
  // replay reaches it within seconds; without motion it rests on the finished run instead.
  play.day = reduced ? domains.esg.market.trading_days : domains.esg.market.burn_in_days - 360;
  // A shared link opens on its own policy; without one the desk opens on the information policy.
  const linked = linkedState();
  setDomain(linked.domain ?? 'esg', 'I', linked);
  setPlaying(play.playing);
})();
