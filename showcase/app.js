'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const repo = 'https://github.com/kanzaler123/financial-ABM';
  const txt = (id, value) => { $(id).textContent = String(value); };
  const node = (tag, className, text) => {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text !== undefined) el.textContent = String(text);
    return el;
  };
  const fmt = (v, percent = false) => Number.isFinite(v)
    ? percent ? `${(v * 100).toFixed(2)}%` : v.toFixed(3)
    : '未记录';
  const steps = [
    ['信息有两条路径，不是一条答案。', '公开新闻可直接进入报价；投资者对信息的异步处理，则通过主观估值和订单形成另一条路径。做市商不持续锚定潜在真实基本面。', 'harness.py'],
    ['相同市场，不同的决策逻辑。', '价值、趋势、噪声策略拥有不同的信号与目标。异步参与和个体参数差异，让决策不会整齐划一地发生。', 'policies.py'],
    ['交易的是仓位变化，不是重复的目标。', '先产生平滑后的目标仓位，再减去当前持仓得到增量订单，结合现金、头寸和卖空约束控制可执行规模。', 'policies.py'],
    ['订单流遇上有限流动性。', '聚合型准订单簿结合深度、价差和订单流，分离永久与暂时价格冲击。它不是逐笔连续撮合的完整限价订单簿。', 'microstructure.py'],
    ['每一笔钱和股份，都要有去处。', '交易者与做市商统一结算；检查合法账户状态，并记录交易成本、成交和股份变化。强制回补也进入同批次现金计算。', 'settlement.py'],
    ['把结果变成可检查的证据。', '配置、随机种子、版本信息、每日审计和遥测被保存。观察者队列不应改变仿真；冻结报告与证据矩阵用于追溯实验。', 'harness.py'],
  ];
  document.querySelectorAll('[data-step]').forEach(button => button.addEventListener('click', () => {
    const i = Number(button.dataset.step);
    document.querySelectorAll('[data-step]').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    txt('step-number', String(i + 1).padStart(2, '0'));
    txt('step-title', steps[i][0]); txt('step-description', steps[i][1]);
    $('step-source').href = `${repo}/blob/master/src/abm/${steps[i][2]}`;
  }));
  document.querySelectorAll('.pipeline, .segmented').forEach(group => group.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    const buttons = [...group.querySelectorAll('button')];
    const i = buttons.indexOf(document.activeElement);
    if (i < 0) return;
    event.preventDefault();
    const j = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1
      : (i + (event.key === 'ArrowRight' ? 1 : -1) + buttons.length) % buttons.length;
    buttons[j].focus(); buttons[j].click();
  }));
  // Decorative topology only: these are not sampled agents or telemetry.
  const svgNS = 'http://www.w3.org/2000/svg';
  const colors = ['#c6f590', '#75d8df', '#bca5eb'];
  for (let i = 0; i < 48; i++) {
    const a = i * 2.3999632297, r = 97 + (i % 9) * 12;
    const x = 280 + Math.cos(a) * r * 1.06, y = 247 + Math.sin(a) * r * .88;
    if (i % 3 === 0) {
      const line = document.createElementNS(svgNS, 'line');
      for (const [k, v] of Object.entries({x1:280, y1:247, x2:x, y2:y, stroke:colors[i % 3], 'stroke-opacity':.13})) line.setAttribute(k, String(v));
      $('field').append(line);
    }
    const dot = document.createElementNS(svgNS, 'circle');
    for (const [k, v] of Object.entries({cx:x, cy:y, r:i % 6 === 0 ? 3.5 : 2, fill:colors[i % 3], class:'field-dot'})) dot.setAttribute(k, String(v));
    dot.style.animationDelay = `${-(i % 7)}s`;
    $('field').append(dot);
  }
  const motion = $('motion-toggle');
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const syncMotion = () => {
    if (reduced.matches) { document.body.classList.add('motion-paused'); motion.disabled = true; motion.textContent = '已减少动效'; motion.setAttribute('aria-pressed', 'true'); }
    else { motion.disabled = false; motion.textContent = document.body.classList.contains('motion-paused') ? '开启动效' : '暂停动效'; }
  };
  syncMotion(); reduced.addEventListener('change', syncMotion);
  motion.addEventListener('click', () => {
    const paused = document.body.classList.toggle('motion-paused');
    motion.setAttribute('aria-pressed', String(paused)); motion.textContent = paused ? '开启动效' : '暂停动效';
  });
  $('copy-command').addEventListener('click', async () => {
    try {
      if (!navigator.clipboard) throw new Error('clipboard unavailable');
      await navigator.clipboard.writeText($('preview-command').textContent);
      txt('copy-feedback', '命令已复制。');
    } catch {
      const range = document.createRange(); range.selectNodeContents($('preview-command'));
      const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
      txt('copy-feedback', '浏览器限制自动复制，命令已选中，可手动复制。');
    }
  });
  const metrics = {
    daily_volatility: ['日收益波动率', true, '衡量日收益变化幅度。区间展示不同种子的结果分布，不是估计值的置信区间。'],
    absolute_return_acf_1: ['绝对收益 ACF(1)', false, '检验大波动是否倾向于接着大波动。单个滞后的数值不能替代完整的波动聚集衰减验收。'],
    excess_kurtosis: ['超额峰度', false, '描述尾部相对高斯分布的厚度。外生 GARCH-t 对照以紫色标注，不作为内生涌现证据。'],
    mean_absolute_log_price_gap: ['平均绝对对数价格偏离', false, '衡量价格相对基本面的平均绝对对数差。该指标描述偏离，不是预测准确率。'],
  };
  const scenarios = {
    full: ['完整模型', '所有基线机制开启，固定策略人口；是消融场景的参照。'],
    no_direct_news: ['移除直接新闻', '移除公开新闻直接进入报价的路径，保留 Agent 信息通道。'],
    no_agent_information: ['移除个体信息', '关闭 Agent 信息通道，保留公开新闻的直接报价路径。'],
    no_information_channels: ['移除双信息通道', '同时移除两条信息通道，用来诊断信息对价格发现的作用。'],
    value_only: ['仅价值策略', '只保留价值型策略，检查单一策略市场是否仍活跃。'],
    trend_only: ['仅趋势策略', '只保留趋势型策略，检查动量交易的独立行为。'],
    noise_only: ['仅噪声策略', '只保留噪声型策略，观察缺少价值与趋势交易时的市场。'],
    no_logit: ['关闭 Logit', '显式关闭策略模仿；基线本就关闭学习时，两者可能重合。'],
    no_activity_persistence: ['移除活跃持续性', '移除活跃状态的持续性，观察交易参与过程的作用。'],
    independent_signals: ['独立策略信号', '移除策略内共同信号，比较相关决策的影响。'],
    fixed_participation: ['固定参与机制', '使用固定参与的对照设置，比较状态依赖参与的影响。'],
    fixed_liquidity: ['固定流动性', '关闭动态流动性反馈，检验风险与流动性的内生联系。'],
    liquidity_stress: ['流动性压力', '使用流动性压力场景，观察风险放大与波动聚集。'],
    concentrated_wealth: ['财富集中', '使用财富更集中的初始状态，比较账户异质性的影响。'],
    garch_t_control: ['GARCH-t 外生对照', '明确的外生压力对照，不能据此声称 Agent 交互产生了厚尾。'],
    logit_learning: ['开启 Logit 学习', '启用策略模仿机制，观察策略切换及其市场影响。'],
  };
  const checks = {
    cash_conservation: ['现金守恒', '检查整个系统现金记账的一致性。'],
    share_conservation: ['股份守恒', '检查交易者与做市商之间的股份总量守恒。'],
    full_runs_without_price_cap: ['不依赖价格截断', '完整模型不应靠触碰价格变动上限维持稳定。'],
    full_volatility_in_daily_range: ['日波动率范围', '检验日收益波动率是否处于协议规定区间。'],
    full_return_autocorrelation_near_zero_across_lags: ['多滞后收益自相关', '检查收益在多个滞后上是否存在过强的可预测性。'],
    full_volatility_clustering_has_decay: ['波动聚集与衰减', '要求跨种子稳定呈现波动聚集及合理的滞后衰减，不只看总体中位数。'],
    full_heavier_than_gaussian: ['厚尾稳定性', '检查收益厚尾是否在足够多的随机种子下成立。'],
    endogenous_liquidity_feedback_increases_tail_weight: ['内生流动性与厚尾', '以配对种子比较完整模型和固定流动性对照，检验反馈对尾部的贡献。'],
    full_volume_volatility_relation_positive: ['量价波动关系', '检验交易量与绝对收益之间的正相关。'],
    full_price_discovery_bounded: ['价格发现有界', '检查价格相对基本面的偏离是否满足门槛。'],
    public_news_channel_has_paired_price_response: ['公开新闻通道', '用配对场景检验公开新闻的独立与边际价格响应。'],
    agent_information_channel_has_paired_price_response: ['Agent 信息通道', '用配对场景检验个体信息的独立与边际价格响应。'],
    empty_information_baseline_is_weakest_discovery: ['无信息基线', '比较双通道关闭后价格发现的相对表现。'],
    single_strategy_markets_remain_active: ['单策略市场活跃', '价值、趋势、噪声单一策略市场都要满足各自活跃度要求。'],
    ablations_avoid_pathological_return_predictability: ['消融不产生病态预测性', '不同消融场景的收益自相关也不能出现异常。'],
    fixed_population_has_no_strategy_turnover: ['固定人口无策略切换', '关闭学习时，策略人口不能自行变化。'],
    logit_imitation_fires_and_tracks_fitness: ['学习触发与适应度', '学习场景需产生策略切换，并检验适应度响应。'],
    logit_market_remains_active_and_bounded: ['学习市场活跃有界', '打开学习之后，市场仍需满足活跃性和边界检查。'],
    persistent_order_flow_is_observable: ['订单流持续性', '检查订单流不平衡的持续性是否可观测。'],
    spread_and_depth_are_finite: ['价差与深度有效', '价差与市场深度不能产生非有限值，并需满足相应范围。'],
    liquidity_stress_increases_clustering: ['压力与波动聚集', '通过配对压力场景检查流动性压力是否增加聚集。'],
    garch_control_is_labelled_exogenous: ['外生对照明确标注', 'GARCH-t 只能作为外生对照，不能替代内生机制证据。'],
  };
  let snapshot, current, selectedScenario = 'full', selectedGate = null;
  try {
    snapshot = JSON.parse($('showcase-data').textContent);
    if (!Array.isArray(snapshot.reports) || snapshot.reports.length !== 3) throw new Error('missing reports');
    current = snapshot.reports.find(r => r.id === 'formal');
    if (!current) throw new Error('missing formal report');
  } catch (error) {
    $('load-error').hidden = false;
    document.querySelectorAll('[data-report], #metric-select, #failed-only, #download-data').forEach(e => { e.disabled = true; });
    return;
  }
  function fact(parent, label, value) {
    const row = node('div', 'gate-fact'); row.append(node('span', '', label), node('b', '', value)); parent.append(row);
  }
  function sourceUrl(name) { return `${repo}/blob/${snapshot.source_commit}/${current.path}/${name}`; }
  function renderScenarioDetail() {
    const key = $('metric-select').value, [label, percent] = metrics[key];
    const stats = current.summary[selectedScenario][key];
    const detail = $('scenario-detail'); detail.replaceChildren();
    detail.append(node('strong', '', `${scenarios[selectedScenario]?.[0] || selectedScenario} / ${label}`));
    detail.append(node('div', '', `${scenarios[selectedScenario]?.[1] || ''} 中位数 ${fmt(stats.median, percent)}；种子间 Q10–Q90：${fmt(stats.q10, percent)} 至 ${fmt(stats.q90, percent)}。`));
  }
  function renderChart() {
    const key = $('metric-select').value, [, percent, explanation] = metrics[key];
    txt('metric-explanation', explanation);
    const rows = Object.keys(scenarios).filter(s => current.summary[s]);
    const min = Math.min(0, ...rows.map(s => current.summary[s][key].q10));
    const max = Math.max(0, ...rows.map(s => current.summary[s][key].q90));
    const width = max - min || 1;
    const pos = v => Math.max(0, Math.min(100, (v - min) / width * 100));
    txt('scale-min', fmt(min, percent)); txt('scale-max', fmt(max, percent));
    const chart = $('scenario-chart'); chart.replaceChildren();
    for (const name of rows) {
      const stats = current.summary[name][key];
      const row = node('button', `scenario-row${name === 'garch_t_control' ? ' exogenous' : ''}`);
      row.type = 'button'; row.dataset.scenario = name;
      row.setAttribute('aria-pressed', String(name === selectedScenario));
      row.setAttribute('aria-label', `${scenarios[name][0]}，中位数 ${fmt(stats.median, percent)}，10% 至 90% 分位 ${fmt(stats.q10, percent)} 至 ${fmt(stats.q90, percent)}`);
      const distribution = node('span', 'distribution'); distribution.setAttribute('aria-hidden', 'true');
      const interval = node('span', 'interval'); interval.style.left = `${pos(stats.q10)}%`; interval.style.width = `${pos(stats.q90) - pos(stats.q10)}%`;
      const median = node('span', 'median'); median.style.left = `${pos(stats.median)}%`;
      if (min < 0 && max > 0) { const zero = node('span', 'zero'); zero.style.left = `${pos(0)}%`; distribution.append(zero); }
      distribution.append(interval, median);
      row.append(node('span', 'scenario-name', scenarios[name][0]), distribution, node('span', 'numeric', fmt(stats.median, percent)));
      row.addEventListener('click', () => {
        selectedScenario = name;
        chart.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(b === row)));
        renderScenarioDetail();
      });
      chart.append(row);
    }
    renderScenarioDetail();
  }
  function renderGateDetail() {
    if (!selectedGate) return;
    const passed = current.checks[selectedGate];
    const [title, description] = checks[selectedGate] || [selectedGate, '请追溯原始报告中的完整判据。'];
    txt('gate-detail-label', passed ? 'CHECK / PASSED' : 'CHECK / NOT PASSED');
    txt('gate-title', title); txt('gate-key', selectedGate); txt('gate-description', description);
    const facts = $('gate-facts'); facts.replaceChildren();
    const evidence = current.evidence[selectedGate];
    if (evidence) {
      if (Number.isFinite(evidence.pass_rate)) fact(facts, '跨种子通过率', `${(evidence.pass_rate * 100).toFixed(0)}%`);
      if (evidence.observed !== undefined) fact(facts, '报告 observed', typeof evidence.observed === 'number' ? fmt(evidence.observed) : JSON.stringify(evidence.observed));
      if (evidence.threshold !== undefined) fact(facts, '报告 threshold', JSON.stringify(evidence.threshold));
      if (Array.isArray(evidence.failed_seeds)) fact(facts, '未过种子数量', String(evidence.failed_seeds.length));
    } else {
      fact(facts, '原始检查结果', passed ? 'PASS' : 'FAIL');
      facts.append(node('p', '', '该项未提供同名独立证据行，可能由多个子检查共同决定。请查看原始报告，不据此虚构通过率。'));
    }
    $('gate-source').href = sourceUrl('mechanism_report.json');
  }
  function renderGates() {
    const entries = Object.entries(current.checks);
    const filtered = entries.filter(([, passed]) => !$('failed-only').checked || !passed);
    if (!filtered.some(([key]) => key === selectedGate)) selectedGate = filtered.find(([, passed]) => !passed)?.[0] || filtered[0]?.[0] || null;
    const matrix = $('gate-matrix'); matrix.replaceChildren();
    filtered.forEach(([key, passed]) => {
      const b = node('button', passed ? '' : 'failed'); b.type = 'button';
      b.append(node('span', '', String(entries.findIndex(([k]) => k === key) + 1).padStart(2, '0')), node('span', 'check-symbol', passed ? '✓' : '×'));
      b.setAttribute('aria-label', `${checks[key]?.[0] || key}：${passed ? '通过' : '未通过'}`);
      b.setAttribute('aria-pressed', String(key === selectedGate)); b.title = checks[key]?.[0] || key;
      b.addEventListener('click', () => { selectedGate = key; matrix.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', String(x === b))); renderGateDetail(); });
      matrix.append(b);
    });
    $('gate-empty').hidden = filtered.length !== 0;
    document.querySelector('.gate-detail').hidden = !selectedGate;
    renderGateDetail();
  }
  function renderReport() {
    const passed = Object.values(current.checks).filter(Boolean).length, total = Object.keys(current.checks).length;
    const failed = total - passed;
    document.querySelectorAll('[data-report]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.report === current.id)));
    txt('kpi-gates', `${passed} / ${total}`); txt('kpi-decision', current.id === 'formal' ? (failed ? '正式归档：未通过' : '正式归档：通过') : '仅开发 / 留出验证证据');
    txt('kpi-seeds', current.seeds); txt('kpi-days', current.config.trading_days - current.config.burn_in_days);
    txt('kpi-burn', `另含 ${current.config.burn_in_days} 日预热期`);
    txt('kpi-scenarios', Object.keys(current.summary).length); txt('kpi-runs', `${current.seeds * Object.keys(current.summary).length} 次归档模拟`);
    txt('gate-count', `${passed}/${total}`); $('gate-ring').style.setProperty('--progress', `${passed / total * 100}%`);
    txt('gate-status', failed ? `${failed} 项仍未通过` : '该归档检查全部通过'); $('gate-status').classList.toggle('warn', failed > 0);
    txt('gate-summary', current.id === 'formal' ? '正式归档尚未支持进入 Stage 2。点击方格查看每项检查。' : '开发或未见种子结果不替代正式验收，不能单独宣告 Stage 1 完成。');
    $('report-source').href = sourceUrl('mechanism_report.md');
    txt('report-version', current.protocol); txt('report-revision', `实验代码版本：${current.code_revision}`);
    txt('report-hash', `源报告 SHA-256：${current.sha256}`);
    const config = $('config-info'); config.replaceChildren();
    fact(config, '交易者', current.config.population_size);
    fact(config, '策略占比（价值 / 趋势 / 噪声）', ['value', 'trend', 'noise'].map(k => fmt(current.config.strategy_shares[k], true)).join(' / '));
    fact(config, '基线学习', current.config.learning_enabled ? '开启' : '关闭');
    fact(config, '报告类别', current.stage);
    fact(config, '统计口径', '跨种子分位数');
    renderChart(); renderGates();
  }
  document.querySelectorAll('[data-report]').forEach(button => button.addEventListener('click', () => {
    current = snapshot.reports.find(r => r.id === button.dataset.report);
    selectedGate = null; renderReport();
  }));
  $('metric-select').addEventListener('change', renderChart);
  $('failed-only').addEventListener('change', renderGates);
  $('download-data').addEventListener('click', () => {
    const blob = new Blob([JSON.stringify(snapshot, null, 2)], {type:'application/json'});
    const url = URL.createObjectURL(blob), link = node('a');
    link.href = url; link.download = 'financial-abm-archived-showcase.json';
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  renderReport();
})();
