import { afterEach, describe, expect, it, vi } from 'vitest'
import { ask, pdfUrl, type AskEvent } from './api'

function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder()
  return new ReadableStream({
    start(controller) {
      chunks.forEach((c) => controller.enqueue(enc.encode(c)))
      controller.close()
    },
  })
}

afterEach(() => vi.unstubAllGlobals())

describe('ask', () => {
  it('delivers events in order even when frames are split across network chunks', async () => {
    const body = streamOf([
      'data: {"type":"token","text":"Reve',
      'nue"}\n\ndata: {"type":"token","text":" up"}\n\n',
      'data: {"type":"done","answered":true,"cited":[1],"gated":false,"total_ms":12}\n\n',
    ])
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, body })
    vi.stubGlobal('fetch', fetchMock)

    const events: AskEvent[] = []
    await ask('Infosys revenue?', [{ question: 'q', answer: 'a' }], (e) => events.push(e))

    expect(events.map((e) => e.type)).toEqual(['token', 'token', 'done'])
    expect(events[0]).toEqual({ type: 'token', text: 'Revenue' })
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toMatch(/\/api\/ask$/)
    expect(JSON.parse(init.body)).toEqual({ question: 'Infosys revenue?', history: [{ question: 'q', answer: 'a' }] })
  })

  it('surfaces the API error detail', async () => {
    const res = { ok: false, status: 429, json: async () => ({ detail: 'Rate limit: 30 questions per hour.' }) }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(res))
    await expect(ask('x y z', [], () => {})).rejects.toThrow('Rate limit: 30 questions per hour.')
  })

  it('falls back to a status message when the error body is not JSON', async () => {
    const res = { ok: false, status: 502, json: async () => Promise.reject(new Error('html')) }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(res))
    await expect(ask('x y z', [], () => {})).rejects.toThrow('Request failed (502)')
  })
})

describe('pdfUrl', () => {
  it('encodes the document id and appends the page anchor', () => {
    expect(pdfUrl('a b', 12)).toMatch(/\/api\/documents\/a%20b\/pdf#page=12$/)
    expect(pdfUrl('doc')).toMatch(/\/api\/documents\/doc\/pdf$/)
  })
})
