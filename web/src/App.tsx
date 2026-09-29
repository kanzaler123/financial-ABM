import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import { api, type ConfigRecord, type ReportRecord, type RunRecord, type TelemetryEvent, streamTelemetry } from './api'
import { Chart, type SeriesSpec } from './Chart'
import { Stage2Panel } from './Stage2Panel'
import { useLanguage } from './i18n'
import './App.css'

const colors = { blue: '#377dff', violet: '#b477ff', pink: '#f77fb3', green: '#00b880', aqua: '#08b8c8', orange: '#ffb020' }
type Page = 'dashboard' | 'experiments' | 'market' | 'agents' | 'analysis' | 'reports' | 'settings'
type Translate = (english: string, chinese: string) => string
interface ReportDetail { gate: { checks: Record<string, boolean> } }
interface AgentFrame { trader_types: number[]; wealth: number[]; company_ids: string[] }

function Icon({ name, size = 22 }: { name: string; size?: number }) {
  const paths: Record<string, string> = {
    dashboard: 'M3 3h7v9H3z M14 3h7v5h-7z M3 16h7v5H3z M14 12h7v9h-7z',
    experiments: 'M9 3H5v18h14V3h-4 M9 2h6v4H9z M8 15l3-3 3 2 3-5',
    market: 'M3 18h18 M5 15V9 M10 15V4 M15 15V7 M20 15v-4 M3 7l7-5 10 6',
    agents: 'M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2 M16 3a4 4 0 0 1 0 8 M22 21v-2a4 4 0 0 0-3-3.87 M13 7a4 4 0 1 1-8 0 4 4 0 0 1 8 0',
    analysis: 'M3 21V11h4v10 M10 21V7h4v14 M17 21V3h4v18 M2 8l5-4 5 1 8-4',
    reports: 'M14 2H5v20h14V7z M14 2v5h5 M8 11h8 M8 15h8 M8 18h5',
    settings: 'M9 3l1-2h4l1 2 2 1 2-.2 2 3-1 2v3l1 2-2 3-2-.2-2 1-1 2h-4l-1-2-2-1-2 .2-2-3 1-2V9L3 7l2-3 2 .2z M15 10a3 3 0 1 1-6 0 3 3 0 0 1 6 0',
    calendar: 'M4 5h16v16H4z M8 2v6 M16 2v6 M4 10h16 M8 14h2 M14 14h2 M8 18h2',
    seed: 'M8 5l8 4 M6 8l-2 8 M8 18l9-5 M18 14l2 5 M9 5a3 3 0 1 1-6 0 3 3 0 0 1 6 0 M21 10a3 3 0 1 1-6 0 3 3 0 0 1 6 0 M7 19a3 3 0 1 1-6 0 3 3 0 0 1 6 0',
    layers: 'M12 2l10 6-10 6L2 8z M2 12l10 6 10-6 M2 16l10 6 10-6',
    play: 'M7 4l14 8-14 8z', pause: 'M8 5v14 M16 5v14', step: 'M4 5l10 7-10 7z M18 5v14',
    reset: 'M3 10a9 9 0 1 1 2 8 M3 3v7h7', search: 'M16 16l5 5 M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0',
    arrow: 'M4 12h16 M15 7l5 5-5 5', check: 'M5 12l4 4L19 6', plus: 'M12 5v14 M5 12h14',
    sun: 'M12 1v2 M12 21v2 M1 12h2 M21 12h2 M4 4l2 2 M18 18l2 2 M4 20l2-2 M18 6l2-2 M17 12a5 5 0 1 1-10 0 5 5 0 0 1 10 0',
    moon: 'M21 13A9 9 0 0 1 11 3a9 9 0 1 0 10 10',
  }
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]} /></svg>
}

function numeric(event: TelemetryEvent): Record<string, number> {
  const row: Record<string, number> = { day: event.sim_time }
  Object.entries(event.payload).forEach(([key, value]) => {
    if (typeof value === 'number') row[key] = value
    if (typeof value === 'boolean') row[key] = Number(value)
  })
  return row
}

function stateLabel(state: string | undefined, t: Translate) {
  return ({ running: t('Running', '运行中'), paused: t('Paused', '已暂停'), completed: t('Completed', '已完成'), failed: t('Failed', '运行失败'), created: t('Ready', '就绪'), starting: t('Starting', '启动中') })[state ?? ''] ?? t('Loading', '加载中')
}

function checkLabels(t: Translate): Record<string, string> {
  return {
    full_price_discovery_bounded: t('Price discovery', '价格发现'),
    full_volatility_clustering_has_decay: t('Volatility clustering', '波动聚集'),
    spread_and_depth_are_finite: t('Spread & market depth', '价差与市场深度'),
    persistent_order_flow_is_observable: t('Persistent order flow', '订单流持续性'),
    public_news_channel_has_paired_price_response: t('Public news response', '公共消息响应'),
    full_return_autocorrelation_near_zero_across_lags: t('Return autocorrelation', '收益自相关'),
    ablations_avoid_pathological_return_predictability: t('Ablation predictability', '消融场景收益可预测性'),
    agent_information_channel_has_paired_price_response: t('Agent information response', '智能体信息响应'),
    cash_conservation: t('Cash conservation', '现金守恒'),
    empty_information_baseline_is_weakest_discovery: t('No-information baseline', '无信息基准'),
    endogenous_liquidity_feedback_increases_tail_weight: t('Liquidity feedback & tails', '流动性反馈与尾部'),
    fixed_population_has_no_strategy_turnover: t('Fixed strategy identities', '固定策略身份'),
    full_heavier_than_gaussian: t('Heavy-tailed returns', '收益厚尾'),
    full_runs_without_price_cap: t('No price cap activation', '无价格上限触发'),
    full_volatility_in_daily_range: t('Daily volatility range', '日波动率范围'),
    full_volume_volatility_relation_positive: t('Volume–volatility relation', '量价波动关系'),
    garch_control_is_labelled_exogenous: t('Exogenous GARCH control', '外生 GARCH 对照'),
    liquidity_stress_increases_clustering: t('Liquidity stress response', '流动性压力响应'),
    logit_imitation_fires_and_tracks_fitness: t('Logit fitness tracking', 'Logit 适应度跟踪'),
    logit_market_remains_active_and_bounded: t('Logit market stability', 'Logit 市场稳定性'),
    share_conservation: t('Share conservation', '股份守恒'),
    single_strategy_markets_remain_active: t('Single-strategy activity', '单策略市场活跃度'),
  }
}

function App() {
  const { language, setLanguage, t, format } = useLanguage()
  const queryClient = useQueryClient()
  const [selectedRun, setSelectedRun] = useState<string | null>(null)
  const [events, setEvents] = useState<TelemetryEvent[]>([])
  const [range, setRange] = useState('all')
  const [page, setPage] = useState<Page>('dashboard')
  const [marketTab, setMarketTab] = useState('prices')
  const [theme, setTheme] = useState<'light' | 'dark'>(() => localStorage.getItem('abm-theme') === 'dark' ? 'dark' : 'light')
  const [runName, setRunName] = useState(`run-${new Date().toISOString().slice(0, 10)}`)
  const [configName, setConfigName] = useState('stage2')
  const [search, setSearch] = useState('')
  const [replayIndex, setReplayIndex] = useState<number | null>(null)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(5)
  const [loadError, setLoadError] = useState(false)
  const [reportId, setReportId] = useState<string | null>(null)
  const cursor = useRef(-1)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('abm-theme', theme)
  }, [theme])

  const runs = useQuery({ queryKey: ['runs'], queryFn: () => api<RunRecord[]>('/api/runs'), refetchInterval: 1000 })
  const configs = useQuery({ queryKey: ['configs'], queryFn: () => api<ConfigRecord[]>('/api/configs') })
  const reports = useQuery({ queryKey: ['reports'], queryFn: () => api<ReportRecord[]>('/api/reports') })
  const selected = runs.data?.find((run) => run.id === selectedRun) ?? runs.data?.find((run) => run.active && ['running', 'paused', 'starting'].includes(run.state)) ?? runs.data?.[0]
  const runId = selected?.id
  const formalReport = reports.data?.find((report) => report.stage1_complete && !report.superseded)
    ?? reports.data?.find((report) => report.protocol_stage === 'formal' && !report.superseded)
  const selectedReport = reports.data?.find((report) => report.id === reportId) ?? formalReport ?? reports.data?.[0]
  const report = useQuery({ queryKey: ['report', selectedReport?.id], enabled: !!selectedReport,
    queryFn: () => api<ReportDetail>(`/api/reports/${selectedReport!.id.split('/').map(encodeURIComponent).join('/')}`) })
  const formal = useQuery({ queryKey: ['report', formalReport?.id], enabled: !!formalReport,
    queryFn: () => api<ReportDetail>(`/api/reports/${formalReport!.id.split('/').map(encodeURIComponent).join('/')}`) })
  const stage1Complete = reports.data?.some((item) => item.stage1_complete) ?? false
  const chooseRun = (id: string) => { setSelectedRun(id); setReplayIndex(null); setPlaying(false) }
  const openRunDialog = () => (document.getElementById('run-dialog') as HTMLDialogElement).showModal()

  const startRun = useMutation({
    mutationFn: () => api<RunRecord>('/api/runs', { method: 'POST', body: JSON.stringify({ name: runName, config_name: configName }) }),
    onSuccess: (run) => {
      (document.getElementById('run-dialog') as HTMLDialogElement).close()
      chooseRun(run.id)
      setPage('dashboard')
      queryClient.invalidateQueries({ queryKey: ['runs'] })
    },
  })
  const command = useMutation({
    mutationFn: ({ action, value }: { action: string; value?: number }) => api(`/api/runs/${encodeURIComponent(runId ?? '')}/commands`, { method: 'POST', body: JSON.stringify({ action, value }) }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['runs'] }),
  })

  useEffect(() => {
    setEvents([])
    setLoadError(false)
    cursor.current = -1
    if (!runId) return
    const controller = new AbortController()
    api<TelemetryEvent[]>(`/api/runs/${encodeURIComponent(runId)}/events?after=-1&limit=10000`, { signal: controller.signal }).then((loaded) => {
      if (controller.signal.aborted) return
      setEvents(loaded)
      cursor.current = loaded.at(-1)?.sequence_no ?? -1
      if (selected?.active) {
        streamTelemetry(runId, cursor.current, controller.signal, (event) => {
          cursor.current = event.sequence_no
          setEvents((current) => current.some((item) => item.sequence_no === event.sequence_no) ? current : [...current, event])
        }).catch((error) => { if (error.name !== 'AbortError') setLoadError(true) })
      }
    }).catch((error) => { if (error.name !== 'AbortError') setLoadError(true) })
    return () => controller.abort()
  }, [runId, selected?.active])

  const allRows = useMemo(() => events.map(numeric), [events])
  const currentIndex = Math.min(replayIndex ?? allRows.length - 1, allRows.length - 1)
  const isPlaying = playing && currentIndex < allRows.length - 1
  useEffect(() => {
    if (!isPlaying) return
    const timer = setInterval(() => setReplayIndex((index) => Math.min((index ?? 0) + 1, allRows.length - 1)), 1000 / speed)
    return () => clearInterval(timer)
  }, [isPlaying, speed, allRows.length])
  const rows = useMemo(() => {
    const values = allRows.slice(0, currentIndex + 1)
    return range === 'all' ? values : values.slice(-Number(range))
  }, [allRows, currentIndex, range])
  const latest = allRows[currentIndex] ?? {}
  const previous = allRows[currentIndex - 1] ?? {}
  const agentFrame = useMemo(() => {
    const value = events[currentIndex]?.payload.stage2_json
    return typeof value === 'string' ? JSON.parse(value) as AgentFrame : null
  }, [events, currentIndex])
  const priceSeries: SeriesSpec[] = [
    { name: t('Market price', '市场价格'), field: 'mid_price', color: colors.blue },
    { name: t('Fundamental value', '基本价值'), field: 'fundamental_value', color: colors.violet, dashed: true },
  ]
  const strategySeries: SeriesSpec[] = [
    { name: t('Value', '价值型'), field: 'value_share', color: colors.blue },
    { name: t('Trend', '趋势型'), field: 'trend_share', color: colors.violet },
    { name: t('Noise', '噪声型'), field: 'noise_share', color: colors.pink },
  ]
  const navigation: { id: Page; label: string }[] = [
    { id: 'dashboard', label: t('Dashboard', '仪表盘') }, { id: 'experiments', label: t('Experiments', '实验管理') },
    { id: 'market', label: t('Market View', '市场行情') }, { id: 'agents', label: t('Agents', '智能体') },
    { id: 'analysis', label: t('Analysis', '分析') }, { id: 'reports', label: t('Reports', '验证报告') }, { id: 'settings', label: t('Settings', '设置') },
  ]
  const rangeButtons = <div className="range-buttons" aria-label={t('Time range', '时间范围')}>{['30', '90', '250', 'all'].map((value) => <button key={value} aria-pressed={range === value} onClick={() => setRange(value)}>{value === 'all' ? t('All', '全部') : `${value}${t('D', '天')}`}</button>)}</div>
  const delta = (field: string, percent = false) => {
    if (!Number.isFinite(previous[field]) || !Number.isFinite(latest[field])) return t('Awaiting previous day', '等待前一日数据')
    const difference = (latest[field] - previous[field]) * (percent ? 100 : 1)
    return `${difference >= 0 ? '+' : ''}${format(difference, percent ? 2 : field === 'spread' ? 4 : 2)}${percent ? t(' pp', ' 个百分点') : ''} ${t('vs prior day', '较前一日')}`
  }
  const live = !!selected?.active && ['running', 'paused', 'starting'].includes(selected.state)
  const togglePlayback = () => {
    if (live) command.mutate({ action: selected?.state === 'paused' ? 'resume' : 'pause' })
    else { if (currentIndex >= allRows.length - 1) setReplayIndex(0); setPlaying(!isPlaying) }
  }

  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#dashboard" onClick={(event) => { event.preventDefault(); setPage('dashboard') }} aria-label={t('ABM dashboard', 'ABM 仪表盘')}><span className="brand-mark"><Icon name="seed" size={37} /></span><span><strong>ABM</strong><small>{t('Financial Market', '金融市场')}<br />{t('Agent-Based Model', '多智能体模型')}</small></span></a>
      <nav className="main-nav" aria-label={t('Main navigation', '主导航')}>{navigation.map((item) => <button key={item.id} aria-label={item.label} title={item.label} className={page === item.id ? 'selected' : ''} aria-current={page === item.id ? 'page' : undefined} onClick={() => setPage(item.id)}><Icon name={item.id} /><span>{item.label}</span></button>)}</nav>
      <blockquote className="sidebar-quote">{t('“Better models', '“以更好的模型，')}<br />{t('for a deeper', '探索更深层的')}<br />{t('understanding.”', '市场规律。”')}<span /></blockquote>
      <div className="system-card"><strong>ABM {t('Laboratory', '实验室')}</strong><p className={runs.isSuccess ? 'connected' : 'muted'}><i className="dot" />{runs.isSuccess ? t('Local service connected', '本地服务已连接') : t('Connecting to service', '正在连接服务')}</p><div><i className={`dot ${loadError ? 'warning-dot' : ''}`} />{t('Telemetry', '仿真记录')}<span>{format(events.length, 0)}</span></div><div><i className={`dot ${stage1Complete ? '' : 'warning-dot'}`} />{t('Stage 1 validation', '第一阶段验证')}<span>{stage1Complete ? t('Pass', '通过') : t('Pending', '待验证')}</span></div></div>
    </aside>

    <main>
      <div className="mountain-header">
        <header className="topbar"><p className="tagline">{t('Explore. Simulate. Understand.', '探索 · 仿真 · 理解')}<em>{t('A more realistic financial world.', '构建更真实的金融世界。')}</em></p><nav className="top-nav" aria-label={t('Quick navigation', '快捷导航')}><button onClick={() => setPage('dashboard')}>{t('Simulation', '仿真')}</button><button onClick={() => setPage('analysis')}>{t('Analysis', '分析')}</button><button onClick={() => setPage('reports')}>{t('Research', '研究')}</button><button onClick={() => (document.getElementById('guide-dialog') as HTMLDialogElement).showModal()}>{t('Documentation', '使用指南')}</button></nav><div className="top-actions"><button className="icon-button search-button" onClick={() => setPage('experiments')} aria-label={t('Search experiments', '搜索实验')}><Icon name="search" size={20} /></button><div className="language-switch" role="group" aria-label={t('Interface language', '界面语言')}><button lang="zh-CN" aria-pressed={language === 'zh'} onClick={() => setLanguage('zh')}>中文</button><button lang="en" aria-pressed={language === 'en'} onClick={() => setLanguage('en')}>EN</button></div><button className="avatar" onClick={() => setPage('settings')} aria-label={t('Open settings', '打开设置')}>A</button></div></header>
        <section className="hero"><div><h1>{page === 'dashboard' ? t('Market Simulation', '金融市场仿真') : navigation.find((item) => item.id === page)?.label}</h1><p>{t('Agent-Based Financial Market Laboratory', '基于多智能体的金融市场实验室')}</p></div><blockquote>{t('Markets are systems,', '市场是一个系统，')}<br />{t('not just numbers.', '不止是一串数字。')}<cite>— ABM</cite></blockquote></section>
      </div>

      <div className="workspace">
        {(runs.isError || loadError) && <p className="error alert" role="alert">{t('Unable to load simulation data. Check the local service, then reload.', '无法加载仿真数据，请检查本地服务后刷新页面。')}</p>}
        <section className="status-strip" aria-label={t('Current experiment', '当前实验')}>
          <div className="run-status"><span className={`status-label ${selected?.state ?? ''}`}><span><Icon name={selected?.state === 'completed' ? 'check' : selected?.state === 'paused' ? 'pause' : 'play'} size={16} /></span>{stateLabel(selected?.state, t)}</span><div className="run-selector"><select aria-label={t('Select experiment', '选择实验')} value={runId ?? ''} onChange={(event) => chooseRun(event.target.value)}>{!runs.data?.length && <option value="">{t('No experiments yet', '暂无实验')}</option>}{runs.data?.map((run) => <option value={run.id} key={run.id}>{run.id}</option>)}</select><small>{agentFrame ? t('Stage 2 · Adaptive agents', '第二阶段 · 自适应智能体') : t('Stage 1 · Market mechanisms', '第一阶段 · 市场机制')}</small></div></div>
          <div className="status-cell"><span className="cell-icon"><Icon name="calendar" /></span><div><small>{t('Day', '仿真天数')}</small><strong>{format(latest.day ?? selected?.current_day ?? 0, 0)} <em>/ {format(selected?.trading_days ?? 0, 0)}</em></strong></div></div>
          <div className="status-cell seed-cell"><span className="cell-icon"><Icon name="seed" /></span><div><small>{t('Seed', '随机种子')}</small><strong>{selected?.seed ?? '—'}</strong></div></div>
          <div className="status-cell scenario-cell"><span className="cell-icon"><Icon name="layers" /></span><div><small>{t('Scenario', '仿真场景')}</small><strong>{agentFrame ? t('Adaptive market', '自适应市场') : t('Baseline', '基准市场')}</strong><span className="scenario-note">{t('1 asset · 3 agent types', '1 种资产 · 3 类智能体')}</span></div></div>
          <button className="new-experiment" onClick={openRunDialog}><Icon name="plus" size={16} />{t('New experiment', '新建实验')}</button>
        </section>

        {page === 'dashboard' && <div className="dashboard-grid">
          <div className="dashboard-main">
            <section className="kpis">
              <Kpi label={t('Price', '市场价格')} value={format(latest.mid_price, 2)} detail={delta('mid_price')} values={rows.map((row) => row.mid_price)} color={colors.green} />
              <Kpi label={t('Fundamental Value', '基本价值')} value={format(latest.fundamental_value, 2)} detail={delta('fundamental_value')} values={rows.map((row) => row.fundamental_value)} color={colors.blue} />
              <Kpi label={t('Volatility (20d)', '波动率（20天）')} value={Number.isFinite(latest.volatility_20d) ? `${format(latest.volatility_20d * 100, 2)}%` : '—'} detail={delta('volatility_20d', true)} values={rows.map((row) => row.volatility_20d)} color={colors.violet} />
              <Kpi label={t('Trading Volume', '成交量')} value={format(latest.volume, 2)} detail={delta('volume')} values={rows.map((row) => row.volume)} color={colors.orange} />
              <Kpi label={t('Bid-Ask Spread', '买卖价差')} value={format(latest.spread, 4)} detail={delta('spread')} values={rows.map((row) => row.spread)} color={colors.aqua} />
            </section>
            <div className="price-chart"><Chart title={t('Price vs Fundamental Value', '市场价格与基本价值')} rows={rows} series={[...priceSeries, { name: t('Trading volume', '成交量'), field: 'volume', color: '#84b2ff', type: 'bar', secondary: true }]} yLabel={t('Price', '价格')} secondaryLabel={t('Volume', '成交量')} toolbar={rangeButtons} /></div>
            <div className="dashboard-bottom">
              <section className="panel mechanism-card"><header><h3>{t('Mechanism Verification (Stage 1)', '机制验证（第一阶段）')}</h3><span className={`badge ${formalReport?.passed ? 'success' : ''}`}>{formalReport ? `${formalReport.passed_checks} / ${formalReport.total_checks} ${t('passed', '通过')}` : t('Pending', '待验证')}</span></header><table><thead><tr><th>{t('Mechanism', '机制')}</th><th>{t('Status', '状态')}</th><th>{t('Evidence', '证据')}</th></tr></thead><tbody>{Object.entries(checkLabels(t)).slice(0, 6).map(([key, label]) => <tr key={key}><td>{label}</td><td><span className={formal.data?.gate.checks[key] ? 'pass' : 'muted'}><i className={`dot ${formal.data?.gate.checks[key] ? '' : 'warning-dot'}`} />{formal.data?.gate.checks[key] == null ? '—' : formal.data.gate.checks[key] ? t('PASS', '通过') : t('FAIL', '未通过')}</span></td><td><button className="table-button" onClick={() => { setReportId(formalReport?.id ?? null); setPage('reports') }}>{t('View', '查看')}</button></td></tr>)}</tbody></table><p className="panel-note">{t('Formal acceptance suite · independent of the selected run', '正式验收集 · 独立于当前实验')}</p></section>
              <section className="panel events-card"><header><h3>{t('Recent Events', '最近事件')}</h3><button className="text-button" onClick={() => setPage('analysis')}>{t('View All', '查看全部')}<Icon name="arrow" size={15} /></button></header><div className="event-list">{events.slice(0, currentIndex + 1).slice(-6).reverse().map((event, index) => <div className="event-row" key={event.sequence_no}><i className="dot" style={{ background: [colors.violet, colors.green, colors.orange, colors.pink][index % 4] }} /><span>{t('Day', '第')} {event.sim_time}{t('', '天')}</span><strong>{t('Market close recorded', '收盘行情已记录')}</strong><small>{t('Price', '价格')}: {format(Number(event.payload.mid_price), 2)}</small></div>)}{!events.length && <p className="empty">{t('Waiting for simulation events…', '等待仿真事件…')}</p>}</div></section>
            </div>
          </div>
          <div className="dashboard-side">
            <Composition latest={latest} population={selected?.population_size ?? 0} frame={agentFrame} />
            <section className="panel simulation-controls"><header><h3>{live ? t('Simulation Controls', '仿真控制') : t('Replay Controls', '回放控制')}</h3><span className="mini-label">{live ? t('LIVE', '实时') : t('RECORDED', '已录制')}</span></header><div className="control-buttons"><button className="primary" disabled={!events.length || command.isPending} onClick={togglePlayback}><Icon name={live ? selected?.state === 'paused' ? 'play' : 'pause' : isPlaying ? 'pause' : 'play'} size={15} />{live ? selected?.state === 'paused' ? t('Resume', '继续') : t('Pause', '暂停') : isPlaying ? t('Pause', '暂停') : t('Play', '播放')}</button><button disabled={!events.length || command.isPending} onClick={() => { if (live) command.mutate({ action: 'step' }); else { setPlaying(false); setReplayIndex(Math.min(currentIndex + 1, allRows.length - 1)) } }}><Icon name="step" size={14} />{t('Step', '单步')}</button><button disabled={!events.length} onClick={() => { setPlaying(false); setReplayIndex(live ? null : 0) }}><Icon name="reset" size={14} />{live ? t('Latest', '最新') : t('Reset', '重置')}</button></div><div className="control-options"><label><span>{live ? t('Simulation speed', '仿真速度') : t('Replay speed', '回放速度')}</span><input type="range" min="1" max="10" value={speed} onChange={(event) => { const value = Number(event.target.value); setSpeed(value); if (live) command.mutate({ action: 'speed', value }) }} /><strong>{speed}×</strong></label><label><span>{t('Recorded day', '回放天数')}</span><input aria-label={t('Replay day', '回放天数')} type="range" min="0" max={Math.max(allRows.length - 1, 0)} value={Math.max(currentIndex, 0)} disabled={!events.length} onChange={(event) => { setPlaying(false); setReplayIndex(Number(event.target.value)) }} /><strong>{latest.day ?? '—'}</strong></label><div className="telemetry-state"><span>{t('Telemetry', '仿真记录')}</span><span><i className="dot" />{format(events.length, 0)} {t('frames saved', '帧已保存')}</span></div></div>{command.isError && <p className="error" role="alert">{t('Command failed. Check the experiment state and try again.', '操作失败，请检查实验状态后重试。')}</p>}</section>
            <section className="landscape-quote"><blockquote>{t('“Understanding markets', '“通过智能体、数据与证据，')}<br />{t('through agents, data, and evidence.”', '理解市场。”')}<span /><cite>ABM</cite></blockquote></section>
          </div>
        </div>}

        {page === 'experiments' && <section className="panel experiments-panel"><header><div><h3>{t('Your Experiments', '实验列表')}</h3><p>{t('Open a live run or replay a completed simulation.', '打开运行中的实验，或回放已完成的仿真。')}</p></div><label className="search-field"><Icon name="search" size={17} /><input aria-label={t('Search experiments', '搜索实验')} placeholder={t('Search by name or seed…', '按名称或种子搜索…')} value={search} onChange={(event) => setSearch(event.target.value)} /></label></header><div className="experiment-list">{runs.data?.filter((run) => `${run.id} ${run.seed}`.toLowerCase().includes(search.toLowerCase())).map((run) => <button key={run.id} className={runId === run.id ? 'selected' : ''} onClick={() => { chooseRun(run.id); setPage('dashboard') }}><span className="cell-icon"><Icon name="experiments" /></span><span><strong>{run.id}</strong><small>{format(run.population_size, 0)} {t('agents', '位智能体')} · {t('Seed', '种子')} {run.seed}</small></span><span>{format(run.current_day, 0)} / {format(run.trading_days, 0)} {t('days', '天')}</span><span className={`status-text ${run.state}`}>{stateLabel(run.state, t)}</span><Icon name="arrow" size={18} /></button>)}</div>{!runs.data?.length && <p className="empty">{t('Create your first experiment to begin.', '新建第一个实验以开始探索。')}</p>}</section>}

        {page === 'market' && <><div className="section-toolbar"><div className="tabs"><button className={marketTab === 'prices' ? 'active' : ''} onClick={() => setMarketTab('prices')}>{t('Market overview', '市场概览')}</button><button className={marketTab === 'micro' ? 'active' : ''} onClick={() => setMarketTab('micro')}>{t('Microstructure', '市场微观结构')}</button></div>{rangeButtons}</div><section className="chart-grid">{marketTab === 'prices' ? <><Chart title={t('Price discovery', '价格发现')} rows={rows} series={priceSeries} yLabel={t('Price', '价格')} /><Chart title={t('Daily return', '日收益率')} rows={rows} series={[{ name: t('Return', '收益率'), field: 'return', color: colors.blue }]} valueFormatter={(value) => `${format(value * 100, 2)}%`} /><Chart title={t('20-day volatility', '20日波动率')} rows={rows} series={[{ name: t('Volatility', '波动率'), field: 'volatility_20d', color: colors.violet }]} valueFormatter={(value) => `${format(value * 100, 2)}%`} /><Chart title={t('Executed volume', '实际成交量')} rows={rows} series={[{ name: t('Volume', '成交量'), field: 'volume', color: colors.aqua }]} yLabel={t('Shares', '股数')} /></> : <><Chart title={t('Bid-ask spread', '买卖价差')} rows={rows} series={[{ name: t('Spread', '价差'), field: 'spread', color: colors.pink }]} /><Chart title={t('Quoted depth', '报价深度')} rows={rows} series={[{ name: t('Depth', '深度'), field: 'depth', color: colors.aqua }]} /><Chart title={t('Order-flow information', '订单流信息')} rows={rows} series={[{ name: t('Imbalance', '不平衡度'), field: 'order_flow_imbalance', color: colors.blue }, { name: t('Surprise', '意外冲击'), field: 'order_flow_surprise', color: colors.orange }]} /><Chart title={t('Impact decomposition', '冲击分解')} rows={rows} series={[{ name: t('Permanent', '永久冲击'), field: 'permanent_impact', color: colors.blue }, { name: t('Transient state', '暂时冲击'), field: 'transient_impact', color: colors.violet }, { name: t('Public news', '公共消息'), field: 'public_news_impact', color: colors.green }]} yLabel={t('Log impact', '对数冲击')} /></>}</section></>}
        {page === 'agents' && <Stage2Panel key={runId} events={events} />}
        {page === 'analysis' && <><div className="section-toolbar"><h2>{t('Strategy & Accounting', '策略与守恒审计')}</h2>{rangeButtons}</div><section className="chart-grid"><Chart title={t('Fixed strategy identities', '固定策略身份')} rows={rows} series={strategySeries} valueFormatter={(value) => `${format(value * 100, 1)}%`} /><Chart title={t('Accounting conservation', '会计守恒')} rows={rows} series={[{ name: t('Cash error', '现金误差'), field: 'cash_relative_error', color: colors.blue }, { name: t('Share error', '股份误差'), field: 'share_relative_error', color: colors.violet }]} yLabel={t('Relative error', '相对误差')} /><Chart title={t('Daily market record', '每日行情记录')} rows={rows} series={[...priceSeries, { name: t('Volume', '成交量'), field: 'volume', color: colors.aqua, secondary: true }]} secondaryLabel={t('Volume', '成交量')} /><section className="panel audit-card"><h3>{t('Run provenance', '实验溯源')}</h3><dl><dt>{t('Experiment', '实验名称')}</dt><dd>{runId ?? '—'}</dd><dt>{t('Result fingerprint', '结果指纹')}</dt><dd>{selected?.fingerprint ?? t('Available after completion', '完成后生成')}</dd><dt>{t('Cash relative error', '现金相对误差')}</dt><dd>{latest.cash_relative_error?.toExponential(3) ?? '—'}</dd><dt>{t('Share relative error', '股份相对误差')}</dt><dd>{latest.share_relative_error?.toExponential(3) ?? '—'}</dd></dl></section></section></>}
        {page === 'reports' && <><section className="panel report-detail"><header><div><h3>{t('Mechanism Evidence', '机制验证证据')}</h3><p>{t('Formal acceptance and development results are shown separately.', '正式验收与开发验证分别标识。')}</p></div><span className={`badge ${selectedReport?.stage1_complete ? 'success' : ''}`}>{selectedReport ? reportLabel(selectedReport, t) : t('Pending', '待验证')}</span></header><label className="report-select">{t('Report', '报告')}<select value={selectedReport?.id ?? ''} onChange={(event) => setReportId(event.target.value)}>{reports.data?.map((item) => <option value={item.id} key={item.id}>{item.id} · {reportLabel(item, t)}</option>)}</select></label><div className="check-grid">{Object.entries(report.data?.gate.checks ?? {}).map(([key, passed]) => <div key={key}><i className={`dot ${passed ? '' : 'warning-dot'}`} /><span>{checkLabels(t)[key] ?? key}</span><strong className={passed ? 'pass' : 'fail'}>{passed ? t('PASS', '通过') : t('FAIL', '未通过')}</strong></div>)}</div>{report.isError && <p className="error">{t('Unable to load this report.', '无法加载此报告。')}</p>}</section><Validation reports={reports.data ?? []} onSelect={setReportId} /></>}
        {page === 'settings' && <section className="panel settings-panel"><h3>{t('Display Preferences', '界面偏好')}</h3><p>{t('Preferences are saved automatically on this browser.', '偏好会自动保存在当前浏览器中。')}</p><div className="setting-row"><div><strong>{t('Interface language', '界面语言')}</strong><small>{t('Applies to navigation, charts, and simulation controls.', '适用于导航、图表和仿真操作。')}</small></div><div className="language-switch"><button lang="zh-CN" aria-pressed={language === 'zh'} onClick={() => setLanguage('zh')}>中文</button><button lang="en" aria-pressed={language === 'en'} onClick={() => setLanguage('en')}>English</button></div></div><div className="setting-row"><div><strong>{t('Appearance', '外观')}</strong><small>{t('Choose a light or dark workspace.', '选择浅色或深色工作区。')}</small></div><button onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}><Icon name={theme === 'light' ? 'moon' : 'sun'} size={17} />{theme === 'light' ? t('Switch to dark', '切换深色') : t('Switch to light', '切换浅色')}</button></div></section>}
      </div>
    </main>

    <dialog id="run-dialog"><form method="dialog" onSubmit={(event) => { event.preventDefault(); startRun.mutate() }}><header><h2>{t('Start a Market Experiment', '新建市场仿真实验')}</h2><button type="button" aria-label={t('Close', '关闭')} onClick={() => (document.getElementById('run-dialog') as HTMLDialogElement).close()}>×</button></header><label>{t('Experiment name', '实验名称')}<input value={runName} onChange={(event) => setRunName(event.target.value)} pattern="[A-Za-z0-9][A-Za-z0-9._-]{0,63}" title={t('1–64 letters, numbers, dots, underscores or hyphens; start with a letter or number.', '使用 1–64 位英文字母、数字、点、下划线或连字符，以字母或数字开头。')} required /><small>{t('Letters, numbers, dots, underscores and hyphens.', '可使用英文字母、数字、点、下划线和连字符。')}</small></label><label>{t('Configuration', '仿真配置')}<select value={configName} onChange={(event) => setConfigName(event.target.value)}>{configs.data?.map((config) => <option key={config.id} value={config.id}>{config.id} · {format(config.population_size, 0)} {t('agents', '智能体')} · {format(config.trading_days, 0)} {t('days', '天')}</option>)}</select></label>{startRun.error && <p className="error" role="alert">{t('Unable to start. Use a unique name and an available configuration.', '启动失败，请使用未占用的名称和可用配置。')}</p>}<button className="primary" type="submit" disabled={startRun.isPending || !configs.data?.length}>{startRun.isPending ? t('Starting…', '启动中…') : t('Start simulation', '开始仿真')}</button></form></dialog>
    <dialog id="guide-dialog"><div className="guide-content"><header><h2>{t('Using the Laboratory', '实验室使用指南')}</h2><button type="button" aria-label={t('Close guide', '关闭指南')} onClick={() => (document.getElementById('guide-dialog') as HTMLDialogElement).close()}>×</button></header><h3>{t('01 · Run an experiment', '01 · 运行实验')}</h3><p>{t('Select an existing experiment or create one from a configuration. The same random seed and configuration reproduce the same simulation.', '选择已有实验，或使用配置新建实验。相同的随机种子与配置可复现同一仿真。')}</p><h3>{t('02 · Explore the market', '02 · 观察市场')}</h3><p>{t('Follow prices, volume and agent composition on the dashboard. Pause or step a live simulation; use the replay slider to explore recorded days.', '在仪表盘观察价格、成交量和智能体组成。运行时可暂停或单步执行；回放时可通过滑块选择已记录的天数。')}</p><h3>{t('03 · Inspect agents and evidence', '03 · 检查智能体与证据')}</h3><p>{t('The Agents page shows information paths, decisions and learning rewards for Stage 2. Reports distinguish formal validation from development checks.', '智能体页面显示第二阶段的信息路径、决策和学习奖励。验证报告会区分正式验收与开发检查。')}</p></div></dialog>
  </div>
}

function Kpi({ label, value, detail, values, color }: { label: string; value: string; detail: string; values: number[]; color: string }) {
  const data = values.slice(-40).filter(Number.isFinite)
  const low = Math.min(...data), high = Math.max(...data)
  const points = data.map((number, index) => `${index / Math.max(data.length - 1, 1) * 160},${32 - (number - low) / (high - low || 1) * 27}`).join(' ')
  return <article className="kpi" style={{ '--accent': color } as CSSProperties}><span>{label}</span><strong>{value}</strong><small>{detail}</small><svg viewBox="0 0 160 40" preserveAspectRatio="none" aria-hidden="true"><polygon points={`0,40 ${points} 160,40`} fill="currentColor" opacity=".09" /><polyline points={points} fill="none" stroke="currentColor" strokeWidth="1.8" vectorEffect="non-scaling-stroke" /></svg></article>
}

function Composition({ latest, population, frame }: { latest: Record<string, number>; population: number; frame: AgentFrame | null }) {
  const { t, format } = useLanguage()
  const shares = [latest.value_share, latest.trend_share, latest.noise_share].map((value) => Number.isFinite(value) ? value : 0)
  const labels = [t('Value', '价值型'), t('Trend', '趋势型'), t('Noise', '噪声型')]
  const palette = [colors.blue, colors.violet, colors.pink]
  const means = [0, 1, 2].map((type) => {
    const values = frame?.wealth.filter((_, index) => frame.trader_types[index] === type) ?? []
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null
  })
  let offset = 0
  return <section className="panel composition-card"><header><h3>{t('Agent Composition', '智能体组成')}</h3><span className="mini-label">{t('TRADERS', '交易者')}</span></header><div className="composition-main"><div className="donut"><svg viewBox="0 0 180 180" role="img" aria-label={labels.map((label, index) => `${label} ${format(shares[index] * 100, 1)}%`).join(', ')}><circle cx="90" cy="90" r="70" fill="none" stroke="var(--track)" strokeWidth="29" />{shares.map((share, index) => { const start = offset; offset += share * 100; return <circle key={index} cx="90" cy="90" r="70" fill="none" stroke={palette[index]} strokeWidth="29" pathLength="100" strokeDasharray={`${Math.max(0, share * 100 - .35)} ${100 - Math.max(0, share * 100 - .35)}`} strokeDashoffset={-start} transform="rotate(-90 90 90)" /> })}</svg><div><strong>{format(population, 0)}</strong><small>{t('Agents', '智能体')}</small></div></div><ul className="composition-legend">{labels.map((label, index) => <li key={label}><i className="dot" style={{ background: palette[index] }} /><span>{label} <small>({format(shares[index] * 100, 1)}%)</small></span></li>)}</ul></div><div className="agent-performance"><h4>{t('Agent Wealth', '智能体财富')} <span>{t('(mean by type)', '（类型均值）')}</span></h4>{labels.map((label, index) => <div className="performance-row" key={label}><span>{label}</span><span className="bar-track"><i style={{ width: means[index] === null ? '0%' : `${Math.max(0, means[index]! / Math.max(...means.map((value) => value ?? 0), 1) * 100)}%`, background: palette[index] }} /></span><strong>{means[index] === null ? '—' : format(means[index]!, 0)}</strong></div>)}{!frame && <p className="panel-note">{t('Individual wealth is available in Stage 2 runs.', '第二阶段实验提供个体财富数据。')}</p>}</div></section>
}

function reportLabel(report: ReportRecord, t: Translate) {
  return { development_pass: t('Development pass', '开发验证通过'), formal_pass: t('Formal pass', '正式验收通过'), formal_fail: t('Formal fail', '正式验收未通过'), failed: t('Fail', '未通过'), superseded: t('Superseded', '已被替代') }[report.scientific_status]
}

function Validation({ reports, onSelect }: { reports: ReportRecord[]; onSelect: (id: string) => void }) {
  const { t } = useLanguage()
  return <section className="validation-card"><header><div><h3>{t('Validation History', '验证历史')}</h3><p>{t('Select a report to inspect its individual checks above.', '选择报告，在上方查看各项检查结果。')}</p></div></header><div className="report-table"><table><thead><tr><th>{t('Report', '报告')}</th><th>{t('Protocol', '验证协议')}</th><th>{t('Checks', '检查项')}</th><th>{t('Decision', '结论')}</th></tr></thead><tbody>{reports.map((report) => <tr key={report.id}><td><button className="report-link" onClick={() => { onSelect(report.id); window.scrollTo({ top: 0, behavior: 'smooth' }) }}>{report.id}</button></td><td>{report.protocol ?? t('Legacy', '旧版')}</td><td>{report.passed_checks ?? '—'} / {report.total_checks ?? '—'}</td><td><span className={report.passed && !report.superseded ? 'pass' : 'muted'}>{reportLabel(report, t)}</span></td></tr>)}</tbody></table></div></section>
}

export default App
