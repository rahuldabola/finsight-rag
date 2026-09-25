import { BarChart3, Library as LibraryIcon, MessageSquareText } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { fetchDocuments, type DocumentInfo } from './api'
import { AskView } from './components/AskView'
import { Evaluation } from './components/Evaluation'
import { Library } from './components/Library'

type Tab = 'ask' | 'library' | 'eval'

const TABS: { id: Tab; label: string; icon: typeof MessageSquareText }[] = [
  { id: 'ask', label: 'Ask', icon: MessageSquareText },
  { id: 'library', label: 'Library', icon: LibraryIcon },
  { id: 'eval', label: 'Evaluation', icon: BarChart3 },
]

const REPO_URL = 'https://github.com/rahuldabola/finsight-rag'

function initialTab(): Tab {
  const hash = window.location.hash.slice(1)
  return hash === 'library' || hash === 'eval' ? hash : 'ask'
}

export default function App() {
  const [tab, setTab] = useState<Tab>(initialTab)
  const [docs, setDocs] = useState<{ documents: DocumentInfo[]; companies: string[] } | null>(null)
  const [apiDown, setApiDown] = useState(false)

  const load = useCallback(() => {
    fetchDocuments().then(
      (d) => {
        setDocs(d)
        setApiDown(false)
      },
      () => setApiDown(true),
    )
  }, [])

  useEffect(load, [load])

  // Keep the tab in sync with #library / #eval links and back/forward navigation.
  useEffect(() => {
    const onHash = () => setTab(initialTab())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const go = (t: Tab) => {
    setTab(t)
    history.replaceState(null, '', t === 'ask' ? ' ' : `#${t}`)
  }

  return (
    <div className="flex h-full flex-col">
      <header className="flex h-14 shrink-0 items-center gap-3 border-b border-ink-800 px-4 sm:px-6">
        <button onClick={() => go('ask')} className="flex items-center gap-2">
          <svg viewBox="0 0 32 32" className="h-7 w-7" aria-hidden>
            <rect width="32" height="32" rx="8" fill="#1fb888" />
            <path d="M8 22 L13 15 L17.5 18.5 L24 9" stroke="#080b12" strokeWidth="3" fill="none" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span className="text-[1.05rem] font-semibold tracking-tight text-white">FinSight</span>
        </button>
        <nav className="ml-2 flex items-center gap-0.5 sm:ml-6">
          {TABS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => go(id)}
              className={`flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm transition ${
                tab === id ? 'bg-ink-800 text-white' : 'text-ink-400 hover:text-ink-100'
              }`}
            >
              <Icon size={15} />
              <span className="hidden sm:inline">{label}</span>
            </button>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-3 text-xs text-ink-400">
          {docs && (
            <span className="hidden md:inline">
              {docs.documents.length} filings · {docs.companies.length} companies
            </span>
          )}
          <a href={REPO_URL} target="_blank" rel="noreferrer" className="rounded p-1.5 hover:bg-ink-800 hover:text-ink-100" title="Source on GitHub">
            <svg viewBox="0 0 16 16" width="17" height="17" fill="currentColor" aria-hidden>
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
            </svg>
          </a>
        </div>
      </header>

      {apiDown && (
        <div className="border-b border-amber-300/20 bg-amber-300/10 px-4 py-2 text-center text-xs text-amber-300">
          Can't reach the API. If the backend was idle it may be waking up; retry in a few seconds.{' '}
          <button onClick={load} className="underline">
            Retry
          </button>
        </div>
      )}

      <main className="min-h-0 flex-1">
        {tab === 'ask' && <AskView />}
        {tab === 'library' && <Library documents={docs?.documents ?? []} companies={docs?.companies ?? []} onChanged={load} />}
        {tab === 'eval' && <Evaluation />}
      </main>
    </div>
  )
}
