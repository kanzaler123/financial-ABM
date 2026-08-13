import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState } from 'react'
import { api, type ConfigRecord, type ReportRecord, type RunRecord, type TelemetryEvent, streamTelemetry } from './api'
import { Chart, type SeriesSpec } from './Chart'
import './App.css'

const colors = {
  blue: '#2a78d6', orange: '#eb6834', aqua: '#1baf7a', yellow: '#eda100',
  magenta: '#e87ba4', green: '#008300', violet: '#4a3aa7', red: '#e34948',
}

const priceSeries: SeriesSpec[] = [
  { name: 'Mid price', field: 'mid_price', color: colors.blue },
  { name: 'Fundamental', field: 'fundamental_value', color: colors.orange },
]
const strategySeries: SeriesSpec[] = [
  { name: 'Value', field: 'value_share', color: colors.blue },
  { name: 'Trend', field: 'trend_share', color: colors.orange },
  { name: 'Noise', field: 'noise_share', color: colors.aqua },
]

function numeric(event: TelemetryEvent): Record<string, number> {
  const row: Record<string, number> = { day: event.sim_time }
  Object.entries(event.payload).forEach(([key, value]) => {
    if (typeof value === 'number') row[key] = value
    if (typeof value === 'boolean') row[key] = Number(value)
  })
  return row
}

function format(value: number, digits = 4) {
  return Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : '—'
}

function App() {
  const queryClient = useQueryClient()
  const [selectedRun, setSelectedRun] = useState<string | null>(null)
  const [events, setEvents] = useState<TelemetryEvent[]>([])
  const [range, setRange] = useState('all')
  const [tab, setTab] = useState<'market' | 'micro' | 'strategy' | 'validation'>('market')
  const [theme, setTheme] = useState<'light' | 'dark'>(() => localStorage.getItem('abm-theme') === 'dark' ? 'dark' : 'light')
  const [runName, setRunName] = useState(`run-${new Date().toISOString().slice(0, 10)}`)
  const [configName, setConfigName] = useState('stage1_structural_dev_v5')
  const cursor = useRef(-1)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('abm-theme', theme)
  }, [theme])

  const runs = useQuery({ queryKey: ['runs'], queryFn: () => api<RunRecord[]>('/api/runs'), refetchInterval: 1000 })
  const configs = useQuery({ queryKey: ['configs'], queryFn: () => api<ConfigRecord[]>('/api/configs') })
  const reports = useQuery({ queryKey: ['reports'], queryFn: () => api<ReportRecord[]>('/api/reports') })
  const selected = runs.data?.find((run) => run.id === selectedRun)
  const stage1Complete = reports.data?.some((report) => report.stage1_complete) ?? false

  const startRun = useMutation({
    mutationFn: () => api<RunRecord>('/api/runs', { method: 'POST', body: JSON.stringify({ name: runName, config_name: configName }) }),
    onSuccess: (run) => {
      setSelectedRun(run.id)
      queryClient.invalidateQueries({ queryKey: ['runs'] })
    },
  })
  const sendCommand = (action: string, value?: number) => api(`/api/runs/${encodeURIComponent(selectedRun ?? '')}/commands`, { method: 'POST', body: JSON.stringify({ action, value }) })

  useEffect(() => {
    if (!selectedRun) return
    setEvents([])
    cursor.current = -1
    const controller = new AbortController()
    api<TelemetryEvent[]>(`/api/runs/${encodeURIComponent(selectedRun)}/events?after=-1&limit=10000`).then((loaded) => {
      setEvents(loaded)
      cursor.current = loaded.at(-1)?.sequence_no ?? -1
      if (selected?.active) {
        streamTelemetry(selectedRun, cursor.current, controller.signal, (event) => {
          cursor.current = event.sequence_no
          setEvents((current) => current.some((item) => item.sequence_no === event.sequence_no) ? current : [...current, event])
        }).catch((error) => { if (error.name !== 'AbortError') console.error(error) })
      }
    }).catch(console.error)
    return () => controller.abort()
  }, [selectedRun, selected?.active])

  const rows = useMemo(() => {
    const values = events.map(numeric)
    const limits: Record<string, number> = { '30': 30, '90': 90, '250': 250 }
    return range === 'all' ? values : values.slice(-limits[range])
  }, [events, range])
  const latest = rows.at(-1) ?? {}

  return (
    <div className="app-shell">
      <aside>
        <div className="brand"><span className="brand-mark">A</span><div><strong>Auditable Market</strong><small>Stage 1 laboratory</small></div></div>
        <button className="primary new-run" type="button" onClick={() => (document.getElementById('run-dialog') as HTMLDialogElement).showModal()}>＋ New run</button>
        <h2>Experiments</h2>
        <nav className="run-list" aria-label="Simulation runs">
          {runs.data?.map((run) => <button type="button" className={run.id === selectedRun ? 'selected' : ''} key={run.id} onClick={() => setSelectedRun(run.id)}><span className={`status ${run.state}`}>●</span><span><strong>{run.id}</strong><small>{run.current_day}/{run.trading_days} · seed {run.seed}</small></span></button>)}
          {!runs.data?.length && <p className="empty">No runs yet.</p>}
        </nav>
        <div className="sidebar-footer"><span>Formal acceptance</span><strong>{stage1Complete ? 'Stage 1 complete' : 'Frozen pending'}</strong></div>
      </aside>

      <main>
        <header className="topbar">
          <div><p className="eyebrow">STAGE 1 / {selectedRun ?? 'OVERVIEW'}</p><h1>{selectedRun ?? 'Market experiments'}</h1></div>
          <button className="icon-button" type="button" onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')} aria-label="Toggle theme">{theme === 'light' ? '◐' : '◑'}</button>
        </header>

        {!selectedRun ? <Welcome reports={reports.data ?? []} /> : <>
          <section className="status-strip"><span className={`status-label ${selected?.state}`}><b>●</b> {selected?.state}</span><span>Day {selected?.current_day ?? latest.day ?? 0} of {selected?.trading_days ?? 0}</span><span>Seed {selected?.seed}</span><span>Fingerprint {selected?.fingerprint?.slice(0, 12) ?? 'pending'}</span></section>
          <section className="controls">
            <div className="button-group"><button onClick={() => sendCommand('pause')}>Pause</button><button onClick={() => sendCommand('resume')}>Resume</button><button onClick={() => sendCommand('step')}>Step</button><button onClick={() => sendCommand('speed', 5)}>5×</button></div>
            <label>Time range<select value={range} onChange={(event) => setRange(event.target.value)}><option value="all">All days</option><option value="250">Last 250</option><option value="90">Last 90</option><option value="30">Last 30</option></select></label>
          </section>
          <section className="kpis">
            <Kpi label="Mid price" value={format(latest.mid_price, 2)} detail={`Fundamental ${format(latest.fundamental_value, 2)}`} />
            <Kpi label="Daily return" value={`${format((latest.return ?? 0) * 100, 2)}%`} detail="close-to-close" />
            <Kpi label="20D volatility" value={`${format((latest.volatility_20d ?? 0) * 100, 2)}%`} detail="rolling standard deviation" />
            <Kpi label="Volume" value={format(latest.volume, 0)} detail="shares executed" />
            <Kpi label="Spread" value={format(latest.spread, 4)} detail={`Depth ${format(latest.depth, 0)}`} />
            <Kpi label="Conservation" value={(latest.cash_relative_error ?? 0) < 1e-8 ? '✓ Pass' : '✕ Review'} detail={`cash ${Number(latest.cash_relative_error ?? 0).toExponential(1)}`} status />
          </section>
          <div className="tabs" role="tablist"><button className={tab === 'market' ? 'active' : ''} onClick={() => setTab('market')}>Market</button><button className={tab === 'micro' ? 'active' : ''} onClick={() => setTab('micro')}>Microstructure</button><button className={tab === 'strategy' ? 'active' : ''} onClick={() => setTab('strategy')}>Strategy & audit</button><button className={tab === 'validation' ? 'active' : ''} onClick={() => setTab('validation')}>Mechanism validation</button></div>
          <section className="chart-grid">
            {tab === 'market' && <><Chart title="Price discovery" rows={rows} series={priceSeries} yLabel="Price" valueFormatter={(v) => format(v, 2)} /><Chart title="Daily return" rows={rows} series={[{ name: 'Return', field: 'return', color: colors.blue }]} yLabel="Return" valueFormatter={(v) => `${format(v * 100, 2)}%`} /><Chart title="20-day volatility" rows={rows} series={[{ name: 'Volatility', field: 'volatility_20d', color: colors.orange }]} yLabel="Volatility" /><Chart title="Executed volume" rows={rows} series={[{ name: 'Volume', field: 'volume', color: colors.aqua }]} yLabel="Shares" /></>}
            {tab === 'micro' && <><Chart title="Bid-ask spread" rows={rows} series={[{ name: 'Spread', field: 'spread', color: colors.magenta }]} yLabel="Price units" /><Chart title="Quoted depth" rows={rows} series={[{ name: 'Depth', field: 'depth', color: colors.aqua }]} yLabel="Shares" /><Chart title="Order-flow information" rows={rows} series={[{ name: 'OFI', field: 'order_flow_imbalance', color: colors.blue }, { name: 'Surprise', field: 'order_flow_surprise', color: colors.yellow }]} /><Chart title="Impact decomposition" rows={rows} series={[{ name: 'Permanent', field: 'permanent_impact', color: colors.blue }, { name: 'Transient state', field: 'transient_impact', color: colors.orange }, { name: 'Public news', field: 'public_news_impact', color: colors.aqua }]} yLabel="Log impact" /></>}
            {tab === 'strategy' && <><Chart title="Fixed strategy identities" rows={rows} series={strategySeries} yLabel="Population share" valueFormatter={(v) => `${format(v * 100, 1)}%`} /><Chart title="Accounting conservation" rows={rows} series={[{ name: 'Cash error', field: 'cash_relative_error', color: colors.blue }, { name: 'Share error', field: 'share_relative_error', color: colors.orange }]} yLabel="Relative error" /></>}
            {tab === 'validation' && <Validation reports={reports.data ?? []} />}
          </section>
        </>}
      </main>

      <dialog id="run-dialog"><form method="dialog" onSubmit={(event) => { event.preventDefault(); startRun.mutate(); (event.currentTarget.parentElement as HTMLDialogElement).close() }}><header><h2>Start a Stage 1 run</h2><button value="cancel" aria-label="Close">×</button></header><label>Run name<input value={runName} onChange={(event) => setRunName(event.target.value)} pattern="[A-Za-z0-9][A-Za-z0-9._-]{0,63}" required /></label><label>Configuration<select value={configName} onChange={(event) => setConfigName(event.target.value)}>{configs.data?.map((config) => <option key={config.id}>{config.id}</option>)}</select></label>{startRun.error && <p className="error">{startRun.error.message}</p>}<button className="primary" type="submit">Start simulation</button></form></dialog>
    </div>
  )
}

function Kpi({ label, value, detail, status = false }: { label: string; value: string; detail: string; status?: boolean }) { return <article className="kpi"><span>{label}</span><strong className={status ? 'good' : ''}>{value}</strong><small>{detail}</small></article> }
function reportLabel(report: ReportRecord) {
  const labels: Record<ReportRecord['scientific_status'], string> = {
    development_pass: 'Development pass',
    formal_pass: 'Formal pass',
    formal_fail: 'Formal fail',
    failed: 'Fail',
    superseded: 'Superseded',
  }
  return labels[report.scientific_status]
}

function Welcome({ reports }: { reports: ReportRecord[] }) { const complete = reports.some((report) => report.stage1_complete); return <section className="welcome"><p className="eyebrow">AUDITABLE ARTIFICIAL MARKET</p><h2>Observe the mechanism, not just the output.</h2><p>Start a deterministic run or choose a completed experiment. Live and replay use the same telemetry contract; visual observers never mutate the simulation.</p><div className="milestones"><article><span>01</span><strong>Dual-channel discovery</strong><p>Public news and Agent order flow remain separately auditable.</p></article><article><span>02</span><strong>Deterministic replay</strong><p>Every completed run carries a result fingerprint and full event stream.</p></article><article><span>03</span><strong>Scientific status</strong><p>{reports.length} validation reports discovered. {complete ? 'Formal acceptance passed.' : 'Formal acceptance remains pending.'}</p></article></div></section> }
function Validation({ reports }: { reports: ReportRecord[] }) { const complete = reports.some((report) => report.stage1_complete); return <section className="validation-card"><header><div><h3>Mechanism evidence</h3><p>Development evidence is not formal acceptance.</p></div><span className="pending-badge">{complete ? 'Formal pass' : 'Frozen pending'}</span></header><table><thead><tr><th>Report</th><th>Protocol</th><th>Checks</th><th>Decision</th></tr></thead><tbody>{reports.map((report) => <tr key={report.id}><td>{report.id}</td><td>{report.protocol ?? 'legacy'}</td><td>{report.passed_checks ?? '—'} / {report.total_checks ?? '—'}</td><td><span className={report.scientific_status === 'formal_pass' || report.scientific_status === 'development_pass' ? 'pass' : 'fail'}>{reportLabel(report)}</span></td></tr>)}</tbody></table></section> }

export default App
