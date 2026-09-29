import { useEffect, useMemo, useState } from 'react'
import type { TelemetryEvent } from './api'
import { Chart } from './Chart'
import { useLanguage } from './i18n'

const palette = ['#377dff', '#b477ff', '#f77fb3', '#08b8c8', '#ffb020']
interface Delivery {
  source_id: string; target_id: string; event_time: string; available_at: string
  credibility: number; path: string[]; message_id: string
}
interface Frame {
  day: number; nodes: string[]; edges: { source_id: string; target_id: string; weight: number; trust: number; delay_days: number }[]
  trader_ids: string[]; company_ids: string[]; trader_actions: number[]; trader_types: number[]
  trader_states: number[]; trader_features: number[][]; factor_preferences: number[]; risk_aversion: number[]
  trader_probabilities: number[][]; bandit_values: number[][][]; q_values: number[][][]
  wealth: number[]; positions: number[]; submitted_orders: number[]; executed_orders: number[]
  visible_news: number[]; subjective_values: number[]; company_groups: number[]; company_states: number[]
  company_actions: number[]; company_features: number[][]; company_cash: number[]; arrivals: Delivery[]
  trader_rewards: { reward: number[]; gross_return: number[]; transaction_cost: number[]; volatility_penalty: number[]; drawdown_penalty: number[]; start_day: number; end_day: number } | null
  company_rewards: { names: string[]; components: number[][]; reward: number[]; start_day: number; end_day: number } | null
}

function Network({ frame, agent, company, source }: { frame: Frame; agent: string; company: string; source: string }) {
  const { t } = useLanguage()
  const neighbors = new Set([agent, company])
  frame.edges.forEach((e) => { if (e.source_id === agent) neighbors.add(e.target_id); if (e.target_id === agent) neighbors.add(e.source_id) })
  const nodes = frame.nodes.length <= 60 ? frame.nodes : frame.nodes.filter((n) => neighbors.has(n))
  const points = new Map(nodes.map((n, i) => [n, [300 + 205 * Math.cos(i * Math.PI * 2 / nodes.length), 210 + 155 * Math.sin(i * Math.PI * 2 / nodes.length)]]))
  const paths = new Set<string>()
  frame.arrivals.filter((d) => source === 'all' || d.source_id === source).forEach((d) => d.path.slice(1).forEach((n, i) => paths.add(`${d.path[i]}|${n}`)))
  return <section className="chart-card"><header><div><h3>{t('Information network', '信息传播网络')}</h3><p>{nodes.length} / {frame.nodes.length} {t('nodes · orange paths arrived today', '个节点 · 橙色路径表示当日送达')}</p></div></header>
    {nodes.length ? <svg viewBox="0 0 600 420" role="img" aria-label={t('Directed information graph', '有向信息传播图')}><defs><marker id="arrow" viewBox="0 0 10 10" refX="18" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke" /></marker></defs>
      {frame.edges.filter((e) => points.has(e.source_id) && points.has(e.target_id)).map((e) => <line key={`${e.source_id}|${e.target_id}`} x1={points.get(e.source_id)![0]} y1={points.get(e.source_id)![1]} x2={points.get(e.target_id)![0]} y2={points.get(e.target_id)![1]} stroke={paths.has(`${e.source_id}|${e.target_id}`) ? '#eb6834' : '#99a4af'} strokeWidth={paths.has(`${e.source_id}|${e.target_id}`) ? 3 : 1.5} markerEnd="url(#arrow)"><title>{`${e.source_id} → ${e.target_id}; ${t('delay', '延迟')} ${e.delay_days}${t('d', '天')}; ${t('trust', '信任度')} ${e.trust.toFixed(3)}; ${t('weight', '权重')} ${e.weight.toFixed(3)}`}</title></line>)}
      {nodes.map((n) => <g key={n}><circle cx={points.get(n)![0]} cy={points.get(n)![1]} r={n === agent || n === company ? 10 : 7} fill={n.startsWith('company') ? '#eda100' : '#2a78d6'} /><text x={points.get(n)![0]} y={points.get(n)![1] - 16} textAnchor="middle" fill="currentColor" fontSize="12">{n}</text></g>)}
    </svg> : <p>{t('Graph disabled: direct public broadcast.', '信息图未启用：直接使用公共广播。')}</p>}
  </section>
}

export function Stage2Panel({ events }: { events: TelemetryEvent[] }) {
  const { t } = useLanguage()
  const traderActions = [t('Value', '价值策略'), t('Trend', '趋势策略'), t('Contrarian', '逆向策略'), t('Reduce', '减仓')]
  const companyActions = [t('Real investment', '实体投资'), t('Disclosure', '信息披露'), t('Maintain', '维持现状')]
  const rewardNames: Record<string, string> = {
    gross_return: t('Gross return', '毛收益'), transaction_cost: t('Transaction cost', '交易成本'),
    volatility_penalty: t('Volatility penalty', '波动惩罚'), drawdown_penalty: t('Drawdown penalty', '回撤惩罚'), reward: t('Total reward', '总奖励'),
    company_value_change: t('Company value change', '公司价值变动'), financing_cost_change: t('Financing cost change', '融资成本变动'),
    real_investment_cost: t('Real investment cost', '实体投资成本'), disclosure_cost: t('Disclosure cost', '披露成本'), verification_penalty: t('Verification penalty', '核验惩罚'),
  }
  const frames = useMemo(() => events.filter((e) => typeof e.payload.stage2_json === 'string').map((e) => JSON.parse(e.payload.stage2_json as string) as Frame), [events])
  const [trader, setTrader] = useState(0)
  const [company, setCompany] = useState(0)
  const [source, setSource] = useState('all')
  const [from, setFrom] = useState(1)
  const [to, setTo] = useState<number | null>(null)
  const [day, setDay] = useState<number | null>(null)
  const [playing, setPlaying] = useState(false)
  const latest = frames.at(-1)
  const history = frames.filter((f) => f.day >= from && f.day <= (to ?? latest?.day ?? 1) && f.day <= (day ?? latest?.day ?? 1))
  const frame = history.at(-1)
  useEffect(() => {
    if (!playing) return
    const timer = setInterval(() => setDay((current) => Math.min((current ?? from) + 1, to ?? latest?.day ?? 1)), 200)
    return () => clearInterval(timer)
  }, [playing, from, to, latest?.day])
  if (!latest) return <section className="chart-card"><h3>{t('No Stage 2 telemetry in this run.', '当前实验暂无第二阶段记录。')}</h3></section>
  const i = Math.min(trader, latest.trader_ids.length - 1), j = Math.min(company, latest.company_ids.length - 1)
  const reward = history.filter((f) => f.trader_rewards).at(-1)?.trader_rewards
  const companyReward = history.filter((f) => f.company_rewards).at(-1)?.company_rewards
  const rows = history.map((f) => {
    const q = f.q_values[f.company_groups[j]][f.company_states[j]]
    const values = [...f.wealth].sort((a, b) => a - b)
    return { day: f.day, p0: f.trader_probabilities[i]?.[0] ?? 0, p1: f.trader_probabilities[i]?.[1] ?? 0, p2: f.trader_probabilities[i]?.[2] ?? 0, p3: f.trader_probabilities[i]?.[3] ?? 0,
      q0: q[0], q1: q[1], q2: q[2], w10: values[Math.floor(values.length * .1)], w50: values[Math.floor(values.length * .5)], w90: values[Math.floor(values.length * .9)],
      a0: f.trader_actions.filter((a) => a === 0).length / values.length, a1: f.trader_actions.filter((a) => a === 1).length / values.length, a2: f.trader_actions.filter((a) => a === 2).length / values.length, a3: f.trader_actions.filter((a) => a === 3).length / values.length }
  })
  const deliveries = history.flatMap((f) => f.arrivals).filter((d) => d.target_id === latest.trader_ids[i] && (source === 'all' || d.source_id === source))
  const series = (names: string[], prefix: string) => names.map((name, k) => ({ name, field: `${prefix}${k}`, color: palette[k] }))
  return <div className="stage2-panel">
    <div className="controls"><label>{t('Trader', '交易者')}<select value={i} onChange={(e) => setTrader(Number(e.target.value))}>{latest.trader_ids.map((id, k) => <option key={id} value={k}>{id}</option>)}</select></label>
      <label>{t('Company', '公司')}<select value={j} onChange={(e) => setCompany(Number(e.target.value))}>{latest.company_ids.map((id, k) => <option key={id} value={k}>{id}</option>)}</select></label>
      <label>{t('Source', '消息来源')}<select value={source} onChange={(e) => setSource(e.target.value)}><option value="all">{t('All sources', '全部来源')}</option>{latest.company_ids.map((id) => <option key={id}>{id}</option>)}</select></label>
      <label>{t('From', '起始日')}<input type="number" min="1" max={latest.day} value={from} onChange={(e) => setFrom(Math.max(1, Number(e.target.value)))} /></label>
      <label>{t('To', '结束日')}<input type="number" min={from} max={latest.day} value={to ?? latest.day} onChange={(e) => setTo(Number(e.target.value))} /></label>
      <button onClick={() => { setDay(from); setPlaying(true) }}>{t('Play log', '播放记录')}</button><button onClick={() => setPlaying(false)}>{t('Pause log', '暂停回放')}</button><button onClick={() => { setPlaying(false); setDay((current) => Math.min((current ?? from) + 1, latest.day)) }}>{t('Step log', '单步回放')}</button><button onClick={() => { setPlaying(false); setDay(null); setTo(null) }}>{t('Follow latest', '跟随最新')}</button>
    </div>
    <label className="replay-slider">{t('Recorded day', '记录天数')} {frame?.day ?? '—'}<input type="range" min={from} max={to ?? latest.day} value={day ?? latest.day} onChange={(e) => { setPlaying(false); setDay(Number(e.target.value)) }} /></label>
    {!frame ? <p>{t('No frame in this time range.', '该时间范围内没有记录。')}</p> : <div className="chart-grid">
      <Network frame={frame} agent={frame.trader_ids[i]} company={frame.company_ids[j]} source={source} />
      <section className="chart-card"><header><div><h3>{t('Observation, action & position', '观测、行动与持仓')}</h3><p>{t('Day', '第')} {frame.day}{t(' · company private state is researcher-only', '天 · 公司私有状态仅供研究者查看')}</p></div></header><pre>{JSON.stringify({ [t('Trader', '交易者')]: frame.trader_ids[i], [t('State at decision', '决策时状态')]: frame.trader_states[i], [t('Features before trade', '交易前特征')]: frame.trader_features[i], [t('Arrived news', '已送达消息')]: frame.visible_news[i], [t('Subjective value at close', '收盘时主观价值')]: frame.subjective_values[i], [t('Action', '行动')]: traderActions[frame.trader_actions[i]] ?? t('Fixed rule', '固定规则'), [t('Submitted order', '提交订单')]: frame.submitted_orders[i], [t('Executed order', '成交订单')]: frame.executed_orders[i], [t('Position', '持仓')]: frame.positions[i], [t('Wealth', '财富')]: frame.wealth[i], [t('Stable preference', '稳定偏好')]: frame.factor_preferences[i], [t('Risk aversion', '风险厌恶')]: frame.risk_aversion[i], [t('Company', '公司')]: frame.company_ids[j], [t('Company action', '公司行动')]: companyActions[frame.company_actions[j]], [t('Company cash', '公司现金')]: frame.company_cash[j], [t('Company private features', '公司私有特征')]: frame.company_features[j] }, null, 2)}</pre></section>
      <Chart title={t('Bandit choice probabilities', 'Bandit 行动选择概率')} rows={rows} series={series(traderActions, 'p')} />
      <Chart title={t('Company Q values · decision state', '公司 Q 值 · 决策状态')} rows={rows} series={series(companyActions, 'q')} />
      <Chart title={t('Action shares', '行动占比')} rows={rows} series={series(traderActions, 'a')} />
      <Chart title={t('Wealth distribution · quantiles', '财富分布 · 分位数')} rows={rows} series={series([t('10th percentile', '10% 分位数'), t('Median', '中位数'), t('90th percentile', '90% 分位数')], 'w').map((s, k) => ({ ...s, field: ['w10', 'w50', 'w90'][k] }))} />
      <section className="chart-card"><h3>{t('Last completed rewards', '最近完成的奖励结算')}</h3><p>{t('Unfinished windows receive no update.', '未完成的结算窗口不更新奖励。')}</p><table><thead><tr><th>{t('Component', '奖励组成')}</th><th>{t('Reward', '奖励值')}</th></tr></thead><tbody>{reward && (['gross_return', 'transaction_cost', 'volatility_penalty', 'drawdown_penalty', 'reward'] as const).map((key) => <tr key={key}><td>{t('Trader', '交易者')} {reward.start_day}–{reward.end_day} · {rewardNames[key]}</td><td>{reward[key][i].toPrecision(6)}</td></tr>)}{companyReward?.names.map((name, k) => <tr key={name}><td>{t('Company', '公司')} {companyReward.start_day}–{companyReward.end_day} · {rewardNames[name] ?? name}</td><td>{companyReward.components[j][k].toPrecision(6)}</td></tr>)}</tbody></table></section>
      <section className="chart-card"><h3>{t('Arrived messages', '已送达消息')} · {deliveries.length}</h3><div className="table-scroll"><table><thead><tr><th>{t('Source / path', '来源 / 路径')}</th><th>{t('Available', '送达时间')}</th><th>{t('Delay', '延迟')}</th><th>{t('Trust', '信任度')}</th></tr></thead><tbody>{deliveries.map((d) => <tr key={`${d.message_id}|${d.target_id}`}><td>{d.source_id}<br />{d.path.join(' → ')}</td><td>{d.available_at.slice(0, 10)}</td><td>{(Date.parse(d.available_at) - Date.parse(d.event_time)) / 86400000}{t('d', '天')}</td><td>{d.credibility.toFixed(4)}</td></tr>)}</tbody></table></div></section>
    </div>}
  </div>
}
