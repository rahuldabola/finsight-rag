import { Loader2 } from 'lucide-react'
import { useEffect, useState } from 'react'
import { fetchEval, type EvalResults } from '../api'

const pct = (v: number) => `${(v * 100).toFixed(1)}%`

export function Evaluation() {
  const [data, setData] = useState<EvalResults | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchEval().then(setData, (e: Error) => setError(e.message))
  }, [])

  if (error) return <div className="p-8 text-sm text-red-300">{error}</div>
  if (!data)
    return (
      <div className="flex items-center gap-2 p-8 text-sm text-ink-400">
        <Loader2 size={15} className="animate-spin" /> Loading evaluation…
      </div>
    )

  const best = Math.max(...data.retrieval.map((r) => r.recall_at_5))

  return (
    <div className="scroll-thin h-full overflow-y-auto">
      <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6">
        <h1 className="text-xl font-semibold text-white">Evaluation</h1>
        <p className="mt-1 max-w-3xl text-sm leading-relaxed text-ink-400">
          {data.questions.answerable} hand-labelled questions, each tied to the page(s) of the filing that answer it, plus{' '}
          {data.questions.unanswerable} questions the corpus cannot answer. Corpus: {data.corpus.documents} documents,{' '}
          {data.corpus.pages} pages, {data.corpus.chunks.toLocaleString()} chunks. Generated {data.generated_at}.
        </p>

        <h2 className="mt-8 mb-3 text-sm font-semibold tracking-wide text-ink-300 uppercase">Retrieval ablation</h2>
        <div className="overflow-x-auto rounded-xl border border-ink-700">
          <table className="w-full min-w-[640px] text-sm">
            <thead className="bg-ink-850 text-left text-xs text-ink-400">
              <tr>
                <th className="px-4 py-2.5 font-medium">Configuration</th>
                <th className="px-4 py-2.5 font-medium">Recall@5</th>
                <th className="px-4 py-2.5 font-medium">Recall@8</th>
                <th className="px-4 py-2.5 font-medium">MRR</th>
                <th className="px-4 py-2.5 font-medium">Latency</th>
              </tr>
            </thead>
            <tbody>
              {data.retrieval.map((r) => (
                <tr key={r.config} className="border-t border-ink-800">
                  <td className="px-4 py-3">
                    <div className="font-medium text-ink-100">{r.config}</div>
                    <div className="text-xs text-ink-400">{r.description}</div>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 w-24 overflow-hidden rounded-full bg-ink-800">
                        <div
                          className={`h-full rounded-full ${r.recall_at_5 === best ? 'bg-mint-400' : 'bg-ink-400'}`}
                          style={{ width: pct(r.recall_at_5) }}
                        />
                      </div>
                      <span className={`font-mono ${r.recall_at_5 === best ? 'text-mint-300' : 'text-ink-100'}`}>
                        {pct(r.recall_at_5)}
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-3 font-mono text-ink-100">{pct(r.recall_at_8)}</td>
                  <td className="px-4 py-3 font-mono text-ink-100">{r.mrr.toFixed(3)}</td>
                  <td className="px-4 py-3 font-mono text-ink-300">{r.latency_ms} ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-xs text-ink-400">
          Recall@k: share of questions where a gold page is among the top k passages. MRR: mean reciprocal rank of the first
          gold passage. Latency measured on a laptop CPU.
        </p>

        {data.answers && (
          <>
            <h2 className="mt-10 mb-3 text-sm font-semibold tracking-wide text-ink-300 uppercase">
              End-to-end answers ({data.answers.model})
            </h2>
            <div className="grid gap-3 sm:grid-cols-3">
              {[
                ['Exact-figure accuracy', data.answers.exact_figure_accuracy, 'answer contains the expected figure'],
                ['Citations support the figure', data.answers.citation_support_rate, 'a cited page prints the expected figure'],
                ['Refusal on unanswerable', data.answers.refusal_rate_unanswerable, 'says "not found" instead of guessing'],
              ].map(([label, value, hint]) => (
                <div key={label as string} className="rounded-xl border border-ink-700 bg-ink-900 p-4">
                  <div className="text-xs text-ink-400">{label}</div>
                  <div className="mt-1 font-mono text-2xl text-white">{pct(value as number)}</div>
                  <div className="mt-1 text-xs text-ink-400">{hint}</div>
                </div>
              ))}
            </div>
            <p className="mt-2 text-xs text-ink-400">
              {data.answers.evaluated} questions · cited a labelled gold page {pct(data.answers.citation_page_accuracy)} ·
              calculator used on {data.answers.calculator_used} · false-refusal rate {pct(data.answers.false_refusal_rate)} · average{' '}
              {(data.answers.avg_latency_ms / 1000).toFixed(1)} s per answer.
            </p>
          </>
        )}

        {data.notes.length > 0 && (
          <>
            <h2 className="mt-10 mb-3 text-sm font-semibold tracking-wide text-ink-300 uppercase">Findings</h2>
            <ul className="list-disc space-y-1.5 pl-5 text-sm leading-relaxed text-ink-300">
              {data.notes.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
          </>
        )}
      </div>
    </div>
  )
}
