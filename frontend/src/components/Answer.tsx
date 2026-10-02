import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { citationNumber, linkCitations } from '../lib/citations'

interface Props {
  text: string
  streaming: boolean
  activeSource: number | null
  onCite: (n: number) => void
}

export function Answer({ text, streaming, activeSource, onCite }: Props) {
  return (
    <div className={`answer ${streaming ? 'caret' : ''}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) => {
            const n = citationNumber(href)
            if (n === null) {
              return (
                <a href={href} target="_blank" rel="noreferrer" className="text-mint-300 underline">
                  {children}
                </a>
              )
            }
            const active = activeSource === n
            return (
              <button
                type="button"
                onClick={() => onCite(n)}
                title={`Show source ${n}`}
                className={`mx-0.5 inline-flex h-[1.15rem] min-w-[1.15rem] -translate-y-px items-center justify-center rounded px-1 align-middle font-mono text-[0.68rem] font-semibold transition ${
                  active
                    ? 'bg-mint-400 text-ink-950'
                    : 'bg-mint-500/15 text-mint-300 hover:bg-mint-500/30'
                }`}
              >
                {n}
              </button>
            )
          },
        }}
      >
        {linkCitations(text)}
      </ReactMarkdown>
    </div>
  )
}
