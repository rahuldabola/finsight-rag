import { CheckCircle2, ExternalLink, FileText, Loader2, Lock, Upload } from 'lucide-react'
import { useState } from 'react'
import { jobStatus, pdfUrl, uploadDocument, type DocumentInfo } from '../api'
import { companyClass } from './SourcePanel'

interface Props {
  documents: DocumentInfo[]
  companies: string[]
  onChanged: () => void
}

export function Library({ documents, companies, onChanged }: Props) {
  const totals = documents.reduce(
    (acc, d) => ({ pages: acc.pages + d.pages, chunks: acc.chunks + d.chunks, tables: acc.tables + d.tables }),
    { pages: 0, chunks: 0, tables: 0 },
  )
  return (
    <div className="scroll-thin h-full overflow-y-auto">
      <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6">
        <h1 className="text-xl font-semibold text-white">Document library</h1>
        <p className="mt-1 text-sm text-ink-400">
          {documents.length} documents · {totals.pages.toLocaleString()} pages · {totals.chunks.toLocaleString()} chunks (
          {totals.tables.toLocaleString()} table chunks). The seed documents are public SEC filings, printed to PDF so
          every citation resolves to a real page.
        </p>

        <div className="mt-6 grid gap-3 sm:grid-cols-2">
          {documents.map((d) => (
            <div key={d.id} className="rounded-xl border border-ink-700 bg-ink-900 p-4">
              <div className="mb-2 flex items-center gap-2 text-xs">
                <span className={`rounded px-1.5 py-0.5 font-medium ${companyClass(d.company)}`}>{d.company}</span>
                <span className="rounded bg-ink-800 px-1.5 py-0.5 text-ink-300">
                  {d.doc_type === 'annual' ? 'Annual filing' : d.doc_type === 'earnings' ? 'Earnings release' : 'Other'}
                </span>
                <span className="text-ink-400">{d.period}</span>
              </div>
              <div className="mb-3 text-sm font-medium leading-snug text-ink-100">{d.title}</div>
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-400">
                <span>{d.pages} pages</span>
                <span>{d.chunks} chunks</span>
                <span>{d.tables} tables</span>
                <a
                  href={pdfUrl(d.id)}
                  target="_blank"
                  rel="noreferrer"
                  className="ml-auto inline-flex items-center gap-1 text-mint-300 hover:underline"
                >
                  <FileText size={12} /> PDF
                </a>
                {d.source.startsWith('http') && (
                  <a
                    href={d.source}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1 text-ink-300 hover:text-ink-100"
                  >
                    <ExternalLink size={12} /> SEC
                  </a>
                )}
              </div>
            </div>
          ))}
        </div>

        <UploadForm companies={companies} onDone={onChanged} />
      </div>
    </div>
  )
}

function UploadForm({ companies, onDone }: { companies: string[]; onDone: () => void }) {
  const [file, setFile] = useState<File | null>(null)
  const [company, setCompany] = useState('')
  const [period, setPeriod] = useState('')
  const [docType, setDocType] = useState('annual')
  const [password, setPassword] = useState('')
  const [status, setStatus] = useState<{ kind: 'idle' | 'working' | 'ok' | 'err'; text?: string }>({ kind: 'idle' })

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    if (!file) return
    const form = new FormData()
    form.append('file', file)
    form.append('company', company)
    form.append('period', period)
    form.append('doc_type', docType)
    setStatus({ kind: 'working', text: 'Uploading…' })
    try {
      const job = await uploadDocument(form, password)
      setStatus({ kind: 'working', text: 'Parsing, chunking and embedding… (about a minute per 100 pages)' })
      for (;;) {
        await new Promise((r) => setTimeout(r, 2500))
        const s = await jobStatus(job.id)
        if (s.status === 'done') {
          setStatus({ kind: 'ok', text: `Indexed ${s.document?.pages} pages into ${s.document?.chunks} chunks.` })
          onDone()
          break
        }
        if (s.status === 'failed') throw new Error(s.error || 'Indexing failed')
      }
    } catch (err) {
      setStatus({ kind: 'err', text: (err as Error).message })
    }
  }

  const input =
    'w-full rounded-lg border border-ink-700 bg-ink-950 px-3 py-2 text-sm text-ink-100 outline-none focus:border-mint-500/60'

  return (
    <form onSubmit={submit} className="mt-10 rounded-xl border border-dashed border-ink-600 p-5">
      <div className="mb-1 flex items-center gap-2 font-medium text-white">
        <Upload size={16} /> Add a PDF
      </div>
      <p className="mb-4 text-xs text-ink-400">
        Uploads are indexed live on the server (local ONNX embeddings, no API cost). Uploading needs the admin password
        so the public demo stays clean; asking questions doesn't.
      </p>
      <div className="grid gap-3 sm:grid-cols-2">
        <input type="file" accept="application/pdf" required onChange={(e) => setFile(e.target.files?.[0] ?? null)} className={`${input} file:mr-3 file:rounded file:border-0 file:bg-ink-700 file:px-2 file:py-1 file:text-ink-100`} />
        <input list="companies" required placeholder="Company (e.g. TCS)" value={company} onChange={(e) => setCompany(e.target.value)} className={input} />
        <datalist id="companies">
          {companies.map((c) => (
            <option key={c} value={c} />
          ))}
        </datalist>
        <input required placeholder="Period (e.g. FY2026 or Q1 FY2027)" value={period} onChange={(e) => setPeriod(e.target.value)} className={input} />
        <select value={docType} onChange={(e) => setDocType(e.target.value)} className={input}>
          <option value="annual">Annual report</option>
          <option value="earnings">Earnings release</option>
          <option value="other">Other</option>
        </select>
        <div className="relative sm:col-span-2">
          <Lock size={14} className="absolute top-1/2 left-3 -translate-y-1/2 text-ink-400" />
          <input type="password" required placeholder="Admin password" value={password} onChange={(e) => setPassword(e.target.value)} className={`${input} pl-8`} />
        </div>
      </div>
      <div className="mt-4 flex flex-wrap items-center gap-3">
        <button
          disabled={status.kind === 'working'}
          className="rounded-lg bg-mint-400 px-4 py-2 text-sm font-medium text-ink-950 hover:bg-mint-300 disabled:opacity-50"
        >
          Upload & index
        </button>
        {status.kind !== 'idle' && (
          <span
            className={`inline-flex items-center gap-1.5 text-sm ${
              status.kind === 'err' ? 'text-red-300' : status.kind === 'ok' ? 'text-mint-300' : 'text-ink-300'
            }`}
          >
            {status.kind === 'working' && <Loader2 size={14} className="animate-spin" />}
            {status.kind === 'ok' && <CheckCircle2 size={14} />}
            {status.text}
          </span>
        )}
      </div>
    </form>
  )
}
