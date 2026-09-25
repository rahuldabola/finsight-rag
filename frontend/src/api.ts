export const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000').replace(/\/$/, '')

export interface DocumentInfo {
  id: string
  company: string
  doc_type: 'annual' | 'earnings' | 'other'
  period: string
  title: string
  source: string
  pages: number
  chunks: number
  tables: number
}

export interface Source {
  n: number
  doc_id: string
  title: string
  company: string
  period: string
  doc_type: string
  page: number
  section: string
  kind: 'text' | 'table'
  text: string
  scores: {
    dense: number
    bm25: number
    fused: number
    rerank: number | null
    dense_rank: number | null
    bm25_rank: number | null
  }
}

export interface Analysis {
  companies: string[]
  doc_type: string | null
  comparison: boolean
  best_similarity: number
  retrieval_ms: number
}

export interface ToolCall {
  name: string
  expression: string
  result?: number
  error?: string
}

export type AskEvent =
  | ({ type: 'analysis' } & Analysis)
  | { type: 'sources'; sources: Source[] }
  | { type: 'token'; text: string }
  | ({ type: 'tool' } & ToolCall)
  | { type: 'done'; answered: boolean; cited: number[]; gated: boolean; total_ms: number }
  | { type: 'error'; message: string }

export const pdfUrl = (docId: string, page?: number) =>
  `${API_BASE}/api/documents/${encodeURIComponent(docId)}/pdf${page ? `#page=${page}` : ''}`

async function errorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json()
    return typeof body.detail === 'string' ? body.detail : `Request failed (${res.status})`
  } catch {
    return `Request failed (${res.status})`
  }
}

export async function fetchDocuments(): Promise<{ documents: DocumentInfo[]; companies: string[] }> {
  const res = await fetch(`${API_BASE}/api/documents`)
  if (!res.ok) throw new Error(await errorMessage(res))
  return res.json()
}

export async function fetchEval(): Promise<EvalResults> {
  const res = await fetch(`${API_BASE}/api/eval`)
  if (!res.ok) throw new Error(await errorMessage(res))
  return res.json()
}

/** POST a question and parse the Server-Sent Events stream as it arrives. */
export async function ask(question: string, onEvent: (e: AskEvent) => void, signal?: AbortSignal): Promise<void> {
  const res = await fetch(`${API_BASE}/api/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
    signal,
  })
  if (!res.ok || !res.body) throw new Error(await errorMessage(res))

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let sep: number
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      for (const line of frame.split('\n')) {
        if (line.startsWith('data: ')) onEvent(JSON.parse(line.slice(6)) as AskEvent)
      }
    }
  }
}

export async function uploadDocument(form: FormData, password: string): Promise<{ id: string; status: string }> {
  const res = await fetch(`${API_BASE}/api/documents`, {
    method: 'POST',
    headers: { 'X-Admin-Password': password },
    body: form,
  })
  if (!res.ok) throw new Error(await errorMessage(res))
  return res.json()
}

export async function jobStatus(id: string): Promise<{ status: string; error?: string; document?: DocumentInfo }> {
  const res = await fetch(`${API_BASE}/api/jobs/${id}`)
  if (!res.ok) throw new Error(await errorMessage(res))
  return res.json()
}

export interface RetrievalRow {
  config: string
  description: string
  recall_at_5: number
  recall_at_8: number
  mrr: number
  latency_ms: number
}

export interface EvalResults {
  generated_at: string
  corpus: { documents: number; chunks: number; pages: number }
  questions: { answerable: number; unanswerable: number }
  retrieval: RetrievalRow[]
  answers?: {
    evaluated: number
    exact_figure_accuracy: number
    citation_page_accuracy: number
    citation_support_rate: number
    calculator_used: number
    refusal_rate_unanswerable: number
    false_refusal_rate: number
    avg_latency_ms: number
    model: string
  }
  notes: string[]
}
