import { ExternalLink, FileText, Table2, X } from 'lucide-react'
import { useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { pdfUrl, type Source } from '../api'

interface Props {
  sources: Source[]
  cited: number[]
  active: number | null
  onSelect: (n: number | null) => void
  viewer: { docId: string; page: number } | null
  onOpenPdf: (docId: string, page: number) => void
  onCloseViewer: () => void
}

const COMPANY_COLORS: Record<string, string> = {
  Infosys: 'bg-sky-400/15 text-sky-300',
  Wipro: 'bg-violet-400/15 text-violet-300',
  Cognizant: 'bg-amber-400/15 text-amber-300',
  Accenture: 'bg-pink-400/15 text-pink-300',
}

export const companyClass = (c: string) => COMPANY_COLORS[c] ?? 'bg-ink-600/60 text-ink-100'

function Score({ label, value }: { label: string; value: string }) {
  return (
    <span className="rounded bg-ink-800 px-1.5 py-0.5 font-mono text-[0.65rem] text-ink-300">
      {label} <span className="text-ink-100">{value}</span>
    </span>
  )
}

export function SourcePanel({ sources, cited, active, onSelect, viewer, onOpenPdf, onCloseViewer }: Props) {
  const refs = useRef<Record<number, HTMLDivElement | null>>({})

  useEffect(() => {
    if (active != null) refs.current[active]?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [active])

  if (viewer) {
    return (
      <div className="flex h-full flex-col">
        <div className="flex items-center justify-between gap-2 border-b border-ink-700 px-4 py-2.5">
          <div className="min-w-0 truncate text-sm text-ink-300">
            <span className="text-ink-100">{viewer.docId}</span> · page {viewer.page}
          </div>
          <div className="flex items-center gap-1">
            <a
              href={pdfUrl(viewer.docId, viewer.page)}
              target="_blank"
              rel="noreferrer"
              className="rounded p-1.5 text-ink-300 hover:bg-ink-800 hover:text-ink-100"
              title="Open in new tab"
            >
              <ExternalLink size={16} />
            </a>
            <button
              onClick={onCloseViewer}
              className="rounded p-1.5 text-ink-300 hover:bg-ink-800 hover:text-ink-100"
              title="Back to sources"
            >
              <X size={16} />
            </button>
          </div>
        </div>
        {/* key forces a reload so #page= is honoured when only the page changes */}
        <iframe
          key={`${viewer.docId}-${viewer.page}`}
          src={pdfUrl(viewer.docId, viewer.page)}
          title="Source document"
          className="min-h-0 w-full flex-1 bg-white"
        />
      </div>
    )
  }

  if (!sources.length) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 px-8 text-center text-sm text-ink-400">
        <FileText size={28} className="text-ink-600" />
        <p>
          The passages retrieved for your question appear here, with their retrieval scores. Click a citation to jump
          to the exact page in the filing.
        </p>
      </div>
    )
  }

  return (
    <div className="scroll-thin h-full space-y-2.5 overflow-y-auto p-4">
      <div className="flex items-baseline justify-between text-xs text-ink-400">
        <span>
          {sources.length} passages retrieved · {cited.length} cited
        </span>
        <span className="font-mono">hybrid → rerank</span>
      </div>
      {sources.map((s) => {
        const isActive = active === s.n
        const isCited = cited.includes(s.n)
        return (
          <div
            key={s.n}
            ref={(el) => {
              refs.current[s.n] = el
            }}
            onClick={() => onSelect(isActive ? null : s.n)}
            className={`cursor-pointer rounded-lg border p-3 transition ${
              isActive
                ? 'border-mint-400/70 bg-ink-800'
                : 'border-ink-700 bg-ink-900 hover:border-ink-600'
            } ${!isCited && cited.length ? 'opacity-60' : ''}`}
          >
            <div className="mb-1.5 flex flex-wrap items-center gap-1.5 text-xs">
              <span
                className={`inline-flex h-5 min-w-5 items-center justify-center rounded px-1 font-mono font-semibold ${
                  isCited ? 'bg-mint-400 text-ink-950' : 'bg-ink-700 text-ink-300'
                }`}
              >
                {s.n}
              </span>
              <span className={`rounded px-1.5 py-0.5 font-medium ${companyClass(s.company)}`}>{s.company}</span>
              <span className="text-ink-300">{s.period}</span>
              <span className="text-ink-600">·</span>
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  onOpenPdf(s.doc_id, s.page)
                }}
                className="font-medium text-mint-300 hover:underline"
              >
                p. {s.page}
              </button>
              {s.kind === 'table' && (
                <span className="inline-flex items-center gap-1 text-amber-300">
                  <Table2 size={12} /> table
                </span>
              )}
            </div>
            {s.section && <div className="mb-1 truncate text-xs text-ink-400">{s.section}</div>}
            <div
              className={`source-text text-[0.8rem] leading-relaxed text-ink-300 ${
                isActive ? '' : 'line-clamp-3'
              } ${s.kind === 'table' && isActive ? 'overflow-x-auto' : ''}`}
            >
              {isActive && s.kind === 'table' ? (
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{s.text}</ReactMarkdown>
              ) : (
                <span className="whitespace-pre-line">{s.text}</span>
              )}
            </div>
            {isActive && (
              <div className="mt-2 flex flex-wrap gap-1">
                <Score label="cos" value={s.scores.dense.toFixed(3)} />
                <Score label="bm25" value={s.scores.bm25.toFixed(2)} />
                <Score label="rrf" value={s.scores.fused.toFixed(4)} />
                {s.scores.rerank != null && <Score label="rerank" value={s.scores.rerank.toFixed(2)} />}
                <Score label="ranks" value={`d${s.scores.dense_rank ?? '–'} / k${s.scores.bm25_rank ?? '–'}`} />
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
