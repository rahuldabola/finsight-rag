import { AlertTriangle, ArrowUp, Calculator, CornerDownRight, Filter, Loader2, Plus, Sparkles, Square } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { ask, type Analysis, type HistoryTurn, type Source, type ToolCall } from '../api'
import { Answer } from './Answer'
import { SourcePanel } from './SourcePanel'

interface Turn {
  id: number
  question: string
  /** Standalone form of a follow-up, as rewritten by the backend. */
  rewritten?: string
  answer: string
  analysis?: Analysis
  sources: Source[]
  tools: ToolCall[]
  cited: number[]
  status: 'retrieving' | 'answering' | 'done' | 'error'
  answered?: boolean
  error?: string
  totalMs?: number
}

const EXAMPLES = [
  'How did Infosys revenue and operating margin change in Q1 FY27?',
  'Compare the latest quarterly revenue growth of Wipro, Infosys and Cognizant.',
  "What are Accenture's revenues by geographic market in Q3 FY26?",
  "What was Wipro's voluntary attrition in Q1 FY27?",
  'By what percentage did Cognizant revenue grow in Q2 2026 versus Q2 2025?',
  'What risks does Infosys highlight about generative AI?',
]

// Earlier turns sent with each question so follow-ups ("and Wipro?") resolve.
const HISTORY_TURNS = 3

export function AskView() {
  const [turns, setTurns] = useState<Turn[]>([])
  const [input, setInput] = useState('')
  const [activeTurn, setActiveTurn] = useState<number | null>(null)
  const [activeSource, setActiveSource] = useState<number | null>(null)
  const [viewer, setViewer] = useState<{ docId: string; page: number } | null>(null)
  const [mobilePanel, setMobilePanel] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const busy = turns.some((t) => t.status === 'retrieving' || t.status === 'answering')

  const current = turns.find((t) => t.id === activeTurn) ?? turns[turns.length - 1]

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [turns.length, current?.answer.length])

  const update = (id: number, fn: (t: Turn) => Turn) => setTurns((ts) => ts.map((t) => (t.id === id ? fn(t) : t)))

  async function submit(q: string) {
    const question = q.trim()
    if (question.length < 3 || busy) return
    const id = Date.now()
    const history: HistoryTurn[] = turns
      .filter((t) => t.status === 'done' && t.answer)
      .slice(-HISTORY_TURNS)
      .map((t) => ({ question: t.rewritten ?? t.question, answer: t.answer.slice(0, 2000) }))
    setTurns((ts) => [
      ...ts,
      { id, question, answer: '', sources: [], tools: [], cited: [], status: 'retrieving' },
    ])
    setActiveTurn(id)
    setActiveSource(null)
    setViewer(null)
    setInput('')
    const controller = new AbortController()
    abortRef.current = controller
    try {
      await ask(
        question,
        history,
        (e) => {
          switch (e.type) {
            case 'rewrite':
              update(id, (t) => ({ ...t, rewritten: e.question }))
              break
            case 'analysis': {
              const { type: _t, ...analysis } = e
              void _t
              update(id, (t) => ({ ...t, analysis }))
              break
            }
            case 'sources':
              update(id, (t) => ({ ...t, sources: e.sources, status: 'answering' }))
              break
            case 'token':
              update(id, (t) => ({ ...t, answer: t.answer + e.text }))
              break
            case 'tool':
              update(id, (t) => ({ ...t, tools: [...t.tools, e] }))
              break
            case 'done':
              update(id, (t) => ({ ...t, status: 'done', cited: e.cited, answered: e.answered, totalMs: e.total_ms }))
              break
            case 'error':
              update(id, (t) => ({ ...t, status: 'error', error: e.message }))
              break
          }
        },
        controller.signal,
      )
      update(id, (t) => (t.status === 'answering' || t.status === 'retrieving' ? { ...t, status: 'done' } : t))
    } catch (err) {
      const aborted = (err as Error).name === 'AbortError'
      update(id, (t) => ({ ...t, status: aborted ? 'done' : 'error', error: aborted ? undefined : (err as Error).message }))
    }
  }

  const newChat = () => {
    abortRef.current?.abort()
    setTurns([])
    setActiveTurn(null)
    setActiveSource(null)
    setViewer(null)
    setMobilePanel(false)
  }

  const selectCitation = (turnId: number, n: number) => {
    setActiveTurn(turnId)
    setActiveSource(n)
    setViewer(null)
    setMobilePanel(true)
  }

  return (
    <div className="flex h-full min-h-0">
      {/* Conversation */}
      <section className="flex min-w-0 flex-1 flex-col">
        <div className="scroll-thin min-h-0 flex-1 overflow-y-auto">
          <div className="mx-auto max-w-3xl px-4 py-6 sm:px-6">
            {turns.length === 0 && <Welcome onPick={submit} />}
            {turns.map((t) => (
              <article
                key={t.id}
                onClick={() => setActiveTurn(t.id)}
                className={`mb-8 rounded-xl transition ${
                  turns.length > 1 && current?.id === t.id ? 'ring-1 ring-ink-700' : ''
                } p-1`}
              >
                <h2 className="mb-3 text-[1.05rem] font-semibold leading-snug text-white">{t.question}</h2>
                {t.rewritten && (
                  <div className="-mt-1.5 mb-3 flex items-start gap-1.5 text-xs text-ink-400">
                    <CornerDownRight size={12} className="mt-0.5 shrink-0" />
                    <span>
                      Searched as <span className="text-ink-100">{t.rewritten}</span>
                    </span>
                  </div>
                )}
                {t.analysis && (
                  <div className="mb-3 flex flex-wrap items-center gap-1.5 text-xs text-ink-400">
                    <Filter size={12} />
                    {t.analysis.companies.length ? (
                      <span>
                        Filtered to <span className="text-ink-100">{t.analysis.companies.join(', ')}</span>
                      </span>
                    ) : (
                      <span>All companies</span>
                    )}
                    {t.analysis.doc_type && (
                      <span>
                        · prefers <span className="text-ink-100">{t.analysis.doc_type}</span> docs
                      </span>
                    )}
                    <span>· retrieval {t.analysis.retrieval_ms} ms</span>
                    {t.totalMs != null && <span>· total {(t.totalMs / 1000).toFixed(1)} s</span>}
                  </div>
                )}
                {t.tools.map((tool, i) => (
                  <div
                    key={i}
                    className="mb-2 inline-flex max-w-full items-center gap-2 rounded-md border border-ink-700 bg-ink-900 px-2.5 py-1 font-mono text-xs text-ink-300"
                  >
                    <Calculator size={12} className="shrink-0 text-amber-300" />
                    <span className="truncate">{tool.expression}</span>
                    <span className="text-ink-400">=</span>
                    <span className="text-amber-300">{tool.error ?? tool.result}</span>
                  </div>
                ))}
                {t.status === 'retrieving' && (
                  <div className="flex items-center gap-2 text-sm text-ink-400">
                    <Loader2 size={15} className="animate-spin" /> Searching filings: hybrid search + rerank…
                  </div>
                )}
                {t.status === 'answering' && !t.answer && (
                  <div className="flex items-center gap-2 text-sm text-ink-400">
                    <Loader2 size={15} className="animate-spin" /> Reading {t.sources.length} passages…
                  </div>
                )}
                {t.answer && (
                  <Answer
                    text={t.answer}
                    streaming={t.status === 'answering'}
                    activeSource={current?.id === t.id ? activeSource : null}
                    onCite={(n) => selectCitation(t.id, n)}
                  />
                )}
                {t.status === 'error' && (
                  <div className="mt-2 flex items-start gap-2 rounded-lg border border-red-400/30 bg-red-400/10 px-3 py-2 text-sm text-red-200">
                    <AlertTriangle size={16} className="mt-0.5 shrink-0" /> {t.error}
                  </div>
                )}
                {t.status === 'done' && t.sources.length > 0 && (
                  <button
                    onClick={() => {
                      setActiveTurn(t.id)
                      setViewer(null)
                      setMobilePanel(true)
                    }}
                    className="mt-3 text-xs text-ink-400 hover:text-mint-300 lg:hidden"
                  >
                    View {t.sources.length} sources →
                  </button>
                )}
              </article>
            ))}
            <div ref={bottomRef} />
          </div>
        </div>

        {/* Composer */}
        <div className="border-t border-ink-800 bg-ink-950/80 px-4 py-3 backdrop-blur sm:px-6">
          <form
            className="mx-auto flex max-w-3xl items-end gap-2 rounded-xl border border-ink-700 bg-ink-900 p-2 focus-within:border-mint-500/60"
            onSubmit={(e) => {
              e.preventDefault()
              submit(input)
            }}
          >
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  submit(input)
                }
              }}
              rows={1}
              maxLength={600}
              placeholder={turns.length ? 'Ask a follow-up, e.g. "and Wipro?"' : 'Ask about revenue, margins, risks, headcount… across Infosys, Wipro, Cognizant, Accenture'}
              className="max-h-40 min-h-[2.5rem] flex-1 resize-none bg-transparent px-2 py-2 text-sm text-ink-100 outline-none placeholder:text-ink-400"
            />
            {busy ? (
              <button
                type="button"
                onClick={() => abortRef.current?.abort()}
                className="flex h-9 w-9 items-center justify-center rounded-lg bg-ink-700 text-ink-100 hover:bg-ink-600"
                title="Stop"
              >
                <Square size={14} />
              </button>
            ) : (
              <button
                type="submit"
                disabled={input.trim().length < 3}
                className="flex h-9 w-9 items-center justify-center rounded-lg bg-mint-400 text-ink-950 transition hover:bg-mint-300 disabled:bg-ink-700 disabled:text-ink-400"
                title="Ask"
              >
                <ArrowUp size={18} />
              </button>
            )}
          </form>
          <div className="mx-auto mt-1.5 flex max-w-3xl items-center justify-center gap-3 text-[0.7rem] text-ink-400">
            {turns.length > 0 && (
              <button onClick={newChat} className="flex shrink-0 items-center gap-1 hover:text-mint-300" title="Start a new conversation">
                <Plus size={11} /> New chat
              </button>
            )}
            <p className="text-center">
              Follow-ups keep context. Answers only use the indexed SEC filings and cite the page.
            </p>
          </div>
        </div>
      </section>

      {/* Sources / PDF */}
      <aside
        className={`${
          mobilePanel ? 'fixed inset-0 z-30 flex' : 'hidden'
        } w-full flex-col border-l border-ink-800 bg-ink-900 lg:static lg:flex lg:w-[44%] lg:max-w-[640px]`}
      >
        <div className="flex items-center justify-between border-b border-ink-800 px-4 py-2 lg:hidden">
          <span className="text-sm font-medium">Sources</span>
          <button onClick={() => setMobilePanel(false)} className="text-sm text-mint-300">
            Close
          </button>
        </div>
        <div className="min-h-0 flex-1">
          <SourcePanel
            sources={current?.sources ?? []}
            cited={current?.cited ?? []}
            active={activeSource}
            onSelect={setActiveSource}
            viewer={viewer}
            onOpenPdf={(docId, page) => setViewer({ docId, page })}
            onCloseViewer={() => setViewer(null)}
          />
        </div>
      </aside>
    </div>
  )
}

function Welcome({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="pt-6 sm:pt-12">
      <div className="mb-2 inline-flex items-center gap-1.5 rounded-full border border-mint-500/30 bg-mint-500/10 px-2.5 py-0.5 text-xs text-mint-300">
        <Sparkles size={12} /> RAG over 937 pages of real SEC filings
      </div>
      <h1 className="mb-3 text-2xl font-semibold tracking-tight text-white sm:text-3xl">
        Ask the annual reports.
        <br />
        <span className="text-ink-400">Get answers you can check.</span>
      </h1>
      <p className="mb-7 max-w-xl text-sm leading-relaxed text-ink-300">
        FinSight searches the latest annual filings (20-F / 10-K) and quarterly earnings releases of Infosys, Wipro,
        Cognizant and Accenture with hybrid search (dense vectors + BM25) and a cross-encoder reranker. It answers only from
        what it finds and cites every figure down to the page.
      </p>
      <div className="grid gap-2 sm:grid-cols-2">
        {EXAMPLES.map((q) => (
          <button
            key={q}
            onClick={() => onPick(q)}
            className="rounded-lg border border-ink-700 bg-ink-900 px-3.5 py-3 text-left text-sm text-ink-300 transition hover:border-mint-500/50 hover:text-ink-100"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  )
}
