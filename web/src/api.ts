export type JsonScalar = string | number | boolean | null

export interface RunRecord {
  id: string
  state: string
  active: boolean
  current_day: number
  trading_days: number
  seed: number
  population_size: number
  latest_sequence_no: number
  fingerprint: string | null
  published_frames: number
  dropped_frames: number
  error: string | null
}

export interface ConfigRecord {
  id: string
  population_size: number
  trading_days: number
  seed: number
}

export interface TelemetryEvent {
  run_id: string
  sim_time: number
  sequence_no: number
  event_type: string
  payload: Record<string, JsonScalar>
}

export interface ReportRecord {
  id: string
  passed: boolean | null
  passed_checks: number | null
  total_checks: number | null
  protocol: string | null
  protocol_stage: 'development' | 'formal'
  scientific_status: 'development_pass' | 'formal_pass' | 'formal_fail' | 'failed' | 'superseded'
  stage1_complete: boolean
  freeze_verified: boolean
  superseded: boolean
}

let token: string | null = sessionStorage.getItem('abm-token')

export async function initializeSession(): Promise<void> {
  if (token) return
  const response = await fetch('/api/session')
  if (!response.ok) throw new Error('Unable to initialize the local Web session.')
  const session = (await response.json()) as { token: string }
  token = session.token
  sessionStorage.setItem('abm-token', token)
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  await initializeSession()
  const headers = new Headers(init?.headers)
  headers.set('Authorization', `Bearer ${token}`)
  if (init?.body) headers.set('Content-Type', 'application/json')
  const response = await fetch(path, { ...init, headers })
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `${response.status} ${response.statusText}`)
  }
  return response.json() as Promise<T>
}

export async function streamTelemetry(
  runId: string,
  after: number,
  signal: AbortSignal,
  onEvent: (event: TelemetryEvent) => void,
): Promise<void> {
  await initializeSession()
  const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/stream`, {
    headers: {
      Accept: 'text/event-stream',
      Authorization: `Bearer ${token}`,
      'Last-Event-ID': String(after),
    },
    signal,
  })
  if (!response.ok || !response.body) throw new Error('Telemetry stream unavailable.')
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += value
    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''
    for (const frame of frames) {
      const eventType = frame.match(/^event:\s*(.+)$/m)?.[1]
      const data = frame.match(/^data:\s*(.+)$/m)?.[1]
      if (eventType === 'telemetry' && data) onEvent(JSON.parse(data) as TelemetryEvent)
    }
  }
}
