import assert from 'node:assert/strict'
import { test } from 'node:test'

let serial = 0
async function freshApi(initialToken = null) {
  const data = new Map(initialToken ? [['abm-token', initialToken]] : [])
  globalThis.sessionStorage = {
    getItem: key => data.get(key) ?? null,
    setItem: (key, value) => data.set(key, value),
    removeItem: key => data.delete(key),
  }
  return import(`../src/api.ts?test=${serial++}`)
}
const json = (value, status = 200) => new Response(JSON.stringify(value), {
  status, headers: { 'Content-Type': 'application/json' },
})

test('concurrent startup requests share one session request', async () => {
  const client = await freshApi()
  let sessions = 0
  globalThis.fetch = async (url, init) => {
    if (url === '/api/session') { sessions++; await new Promise(resolve => setTimeout(resolve, 10)); return json({ token: 'new' }) }
    assert.equal(init.headers.get('Authorization'), 'Bearer new')
    return json([])
  }
  await Promise.all([client.api('/api/runs'), client.api('/api/configs')])
  assert.equal(sessions, 1)
})

test('stale tokens are renewed once after a server restart', async () => {
  const client = await freshApi('old')
  let sessions = 0, requests = 0
  globalThis.fetch = async (url, init) => {
    if (url === '/api/session') { sessions++; return json({ token: 'new' }) }
    requests++
    return init.headers.get('Authorization') === 'Bearer old' ? json({}, 401) : json({ ok: true })
  }
  assert.deepEqual(await client.api('/api/runs'), { ok: true })
  assert.equal(sessions, 1)
  assert.equal(requests, 2)
})

test('repeated authorization failure does not loop forever', async () => {
  const client = await freshApi('old')
  let requests = 0
  globalThis.fetch = async url => url === '/api/session'
    ? json({ token: 'new' }) : (requests++, json({ error: 'denied' }, 401))
  await assert.rejects(client.api('/api/runs'), /denied/)
  assert.equal(requests, 2)
})

test('replay fetches all pages instead of truncating after 10000 days', async () => {
  const client = await freshApi('valid')
  const requested = []
  globalThis.fetch = async url => {
    const after = Number(new URL(url, 'http://localhost').searchParams.get('after'))
    requested.push(after)
    return json(after === -1
      ? Array.from({ length: 10000 }, (_, sequence_no) => ({ sequence_no }))
      : [{ sequence_no: 10000 }, { sequence_no: 10001 }])
  }
  const events = await client.loadRunEvents('long-run', new AbortController().signal)
  assert.equal(events.length, 10002)
  assert.deepEqual(requested, [-1, 9999])
})

test('a broken pagination cursor fails instead of spinning forever', async () => {
  const client = await freshApi('valid')
  globalThis.fetch = async () => json(Array.from({ length: 10000 }, (_, sequence_no) => ({ sequence_no })))
  await assert.rejects(client.loadRunEvents('bad', new AbortController().signal), /did not advance/)
})

test('replay requests propagate cancellation when switching runs', async () => {
  const client = await freshApi('valid')
  const controller = new AbortController()
  controller.abort()
  globalThis.fetch = async (_url, init) => { init.signal.throwIfAborted(); return json([]) }
  await assert.rejects(client.loadRunEvents('old-run', controller.signal), { name: 'AbortError' })
})

test('stream handles split frames and stops at the complete event', async () => {
  const client = await freshApi('valid')
  const chunks = ['event: telemetry\r\nda', 'ta: {"sequence_no":1}\r\n\r\n',
                  'event: complete\r\ndata: {}\r\n\r\n', 'event: telemetry\ndata: {"sequence_no":2}\n\n']
  globalThis.fetch = async () => new Response(new ReadableStream({
    start(controller) { for (const part of chunks) controller.enqueue(new TextEncoder().encode(part)); controller.close() },
  }))
  const events = []
  await client.streamTelemetry('run', 0, new AbortController().signal, event => events.push(event))
  assert.deepEqual(events, [{ sequence_no: 1 }])
})
