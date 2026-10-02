import { describe, expect, it } from 'vitest'
import { parseSseFrames } from './sse'

describe('parseSseFrames', () => {
  it('parses complete frames and keeps the unfinished remainder', () => {
    const { events, rest } = parseSseFrames<{ n: number }>('data: {"n":1}\n\ndata: {"n":2}\n\ndata: {"n"')
    expect(events).toEqual([{ n: 1 }, { n: 2 }])
    expect(rest).toBe('data: {"n"')
  })

  it('reassembles a frame split across chunks', () => {
    const first = parseSseFrames<{ text: string }>('data: {"text":"Rev')
    expect(first.events).toEqual([])
    const second = parseSseFrames<{ text: string }>(first.rest + 'enue"}\n\n')
    expect(second.events).toEqual([{ text: 'Revenue' }])
    expect(second.rest).toBe('')
  })

  it('handles CRLF line endings and ignores non-data lines', () => {
    const { events } = parseSseFrames<{ ok: boolean }>(': keep-alive\r\nevent: x\r\ndata: {"ok":true}\r\n\r\n')
    expect(events).toEqual([{ ok: true }])
  })

  it('returns nothing for an empty buffer', () => {
    expect(parseSseFrames('')).toEqual({ events: [], rest: '' })
  })
})
