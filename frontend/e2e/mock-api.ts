import type { Page, Route } from '@playwright/test'

/** A minimal one-page PDF, enough for the source viewer's iframe to load. */
const TINY_PDF =
  '%PDF-1.1\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n' +
  '3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF'

export const DOCUMENTS = {
  companies: ['Cognizant', 'Infosys', 'Wipro'],
  documents: [
    { id: 'infosys-q1', company: 'Infosys', doc_type: 'earnings', period: 'Q1 FY2027', title: 'Infosys Q1 FY27 earnings', source: 'https://sec.gov/x', pages: 12, chunks: 40, tables: 6 },
    { id: 'wipro-q1', company: 'Wipro', doc_type: 'earnings', period: 'Q1 FY2027', title: 'Wipro Q1 FY27 earnings', source: 'https://sec.gov/y', pages: 10, chunks: 30, tables: 4 },
  ],
}

export const EVAL = {
  generated_at: '2026-09-26',
  corpus: { documents: 8, chunks: 3168, pages: 937 },
  questions: { answerable: 51, unanswerable: 12 },
  retrieval: [
    { config: 'BM25 only', description: 'keyword search', recall_at_5: 0.598, recall_at_8: 0.726, mrr: 0.532, latency_ms: 32 },
    { config: 'Hybrid + filters + rerank', description: 'production', recall_at_5: 0.853, recall_at_8: 0.961, mrr: 0.683, latency_ms: 3879 },
  ],
  notes: ['Each retrieval stage adds recall.'],
}

const sse = (events: object[]) => events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join('')

export const source = (n: number, page: number) => ({
  n,
  doc_id: 'infosys-q1',
  title: 'Infosys Q1 FY27 earnings',
  company: 'Infosys',
  period: 'Q1 FY2027',
  doc_type: 'earnings',
  page,
  section: 'Results',
  kind: 'text' as const,
  text: 'Infosys revenue for Q1 was $5,082 million, up from $4,941 million a year ago.',
  scores: { dense: 0.81, bm25: 12.4, fused: 0.0328, rerank: 7.5, dense_rank: 1, bm25_rank: 2 },
})

export const ANSWER_EVENTS = [
  { type: 'analysis', companies: ['Infosys'], doc_type: 'earnings', comparison: false, best_similarity: 0.81, retrieval_ms: 42 },
  { type: 'sources', sources: [source(1, 2), source(2, 5)] },
  { type: 'tool', name: 'calculate', expression: '(5082-4941)/4941*100', result: 2.853674 },
  { type: 'token', text: 'Revenue was $5,082 million [1], ' },
  { type: 'token', text: 'up 2.85% year over year [1][2].' },
  { type: 'done', answered: true, cited: [1, 2], gated: false, total_ms: 1800 },
]

export interface MockOptions {
  /** Called with the parsed body of every POST /api/ask; returns the SSE events to stream back. */
  onAsk?: (body: { question: string; history: { question: string; answer: string }[] }) => object[] | { status: number; detail: string }
  documentsFail?: boolean
}

/** Intercepts every /api call the app makes, so the browser test runs without a backend. */
export async function mockApi(page: Page, opts: MockOptions = {}) {
  const asks: { question: string; history: { question: string; answer: string }[] }[] = []
  const json = (route: Route, body: unknown, status = 200) =>
    route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })

  await page.route('**/api/documents', (route) => (opts.documentsFail ? json(route, { detail: 'down' }, 503) : json(route, DOCUMENTS)))
  await page.route('**/api/eval', (route) => json(route, EVAL))
  await page.route('**/api/documents/*/pdf', (route) => route.fulfill({ status: 200, contentType: 'application/pdf', body: TINY_PDF }))
  await page.route('**/api/ask', async (route) => {
    const body = route.request().postDataJSON()
    asks.push(body)
    const reply = opts.onAsk ? opts.onAsk(body) : ANSWER_EVENTS
    if (!Array.isArray(reply)) return json(route, { detail: reply.detail }, reply.status)
    await route.fulfill({ status: 200, contentType: 'text/event-stream', body: sse(reply) })
  })
  return { asks }
}
