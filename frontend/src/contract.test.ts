import { describe, expect, it } from 'vitest'
import contract from '../../contracts/ask-events.json'
import { ANSWER_EVENTS, source } from '../e2e/mock-api'

// The browser tests mock the API. This keeps those mocks honest: they must have exactly the event shapes
// the backend is tested to emit (backend/tests/test_contract.py reads the same contracts/ask-events.json).
describe('ask event contract', () => {
  it.each(ANSWER_EVENTS.map((e) => [e.type, e] as const))('mock "%s" event matches the contract', (type, event) => {
    const spec = contract.events[type as keyof typeof contract.events]
    expect(spec, `unknown event type ${type}`).toBeDefined()
    const keys = Object.keys(event)
    expect(keys).toEqual(expect.arrayContaining(spec.required))
    expect([...spec.required, ...spec.optional]).toEqual(expect.arrayContaining(keys))
  })

  it('mock sources have exactly the contract fields', () => {
    const s = source(1, 1)
    expect(Object.keys(s).sort()).toEqual([...contract.source.required].sort())
    expect(Object.keys(s.scores).sort()).toEqual([...contract.source.scores].sort())
  })

  it('covers every event type the UI handles', () => {
    expect(Object.keys(contract.events).sort()).toEqual(['analysis', 'done', 'error', 'rewrite', 'sources', 'token', 'tool'])
  })
})
