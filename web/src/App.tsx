import { useEffect, useRef, useState, type FormEvent } from 'react'
import {
  ApiError,
  ask,
  getHealth,
  search,
  type HealthResponse,
  type Mode,
  type Source,
} from './api.ts'

type Status =
  | { state: 'loading' }
  | { state: 'ready'; health: HealthResponse }
  | { state: 'down'; message: string }

type Action = 'ask' | 'search'

type Outcome =
  | { state: 'idle' }
  | { state: 'loading'; action: Action }
  | { state: 'error'; message: string }
  | { state: 'done'; action: Action; query: string; answer?: string; sources: Source[] }

const MODES: { value: Mode; label: string }[] = [
  { value: 'dense', label: 'Dense (embeddings)' },
  { value: 'bm25', label: 'BM25 (keywords)' },
  { value: 'hybrid', label: 'Hybrid' },
]

/** Render an answer, turning citations like [2] into links to the matching source. */
function Answer({ text, sources }: { text: string; sources: Source[] }) {
  const parts = text.split(/(\[\d+\])/)
  return (
    <p className="answer">
      {parts.map((part, i) => {
        const n = /^\[(\d+)\]$/.exec(part)?.[1]
        if (n && Number(n) >= 1 && Number(n) <= sources.length) {
          return (
            <a key={i} className="cite" href={`#source-${n}`}>
              {part}
            </a>
          )
        }
        return part
      })}
    </p>
  )
}

function SourceList({ sources }: { sources: Source[] }) {
  if (sources.length === 0) return <p className="empty">No matching passages.</p>
  return (
    <ol className="sources">
      {sources.map((s, i) => (
        <li key={i} id={`source-${i + 1}`}>
          <div className="source-head">
            <span className="source-path">{s.source}</span>
            <span className="source-score">{s.score.toFixed(3)}</span>
          </div>
          <p className="source-text">{s.text}</p>
        </li>
      ))}
    </ol>
  )
}

function App() {
  const [status, setStatus] = useState<Status>({ state: 'loading' })
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<Mode>('hybrid')
  const [k, setK] = useState(5)
  const [outcome, setOutcome] = useState<Outcome>({ state: 'idle' })
  const inflight = useRef<AbortController | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    getHealth(controller.signal)
      .then((health) => setStatus({ state: 'ready', health }))
      .catch((e: Error) => {
        if (e.name !== 'AbortError') setStatus({ state: 'down', message: e.message })
      })
    return () => controller.abort()
  }, [])

  // Cancel any request still running when the page unmounts.
  useEffect(() => () => inflight.current?.abort(), [])

  async function run(action: Action) {
    const trimmed = query.trim()
    if (!trimmed) return

    // A newer request supersedes the previous one, so a slow answer can't overwrite it.
    inflight.current?.abort()
    const controller = new AbortController()
    inflight.current = controller
    setOutcome({ state: 'loading', action })

    try {
      const body = { query: trimmed, k, mode }
      if (action === 'ask') {
        const res = await ask(body, controller.signal)
        setOutcome({ state: 'done', action, query: trimmed, answer: res.answer, sources: res.sources })
      } else {
        const res = await search(body, controller.signal)
        setOutcome({ state: 'done', action, query: trimmed, sources: res.results })
      }
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') return
      setOutcome({
        state: 'error',
        message: e instanceof ApiError || e instanceof Error ? e.message : 'Something went wrong',
      })
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault()
    void run('ask')
  }

  const busy = outcome.state === 'loading'
  const canSubmit = query.trim() !== '' && !busy

  return (
    <main>
      <header>
        <h1>TechRAG</h1>
        <p className="tagline">Ask questions about technical documentation.</p>
        <p className={`status ${status.state}`} role="status">
          {status.state === 'loading' && 'Connecting…'}
          {status.state === 'ready' && `${status.health.chunks} chunks indexed`}
          {status.state === 'down' && status.message}
        </p>
      </header>

      <form onSubmit={onSubmit}>
        <label className="visually-hidden" htmlFor="query">
          Question
        </label>
        <textarea
          id="query"
          rows={2}
          placeholder="What is our parental leave policy?"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
              e.preventDefault()
              if (canSubmit) void run('ask')
            }
          }}
        />

        <div className="controls">
          <label>
            Mode
            <select value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
              {MODES.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Passages
            <input
              type="number"
              min={1}
              max={20}
              value={k}
              onChange={(e) => setK(Math.min(20, Math.max(1, Math.trunc(Number(e.target.value)) || 1)))}
            />
          </label>
          <div className="actions">
            <button type="button" className="secondary" disabled={!canSubmit} onClick={() => void run('search')}>
              Search only
            </button>
            <button type="submit" disabled={!canSubmit}>
              Ask
            </button>
          </div>
        </div>
      </form>

      <section className="results" aria-live="polite" aria-busy={busy}>
        {outcome.state === 'loading' && (
          <p className="pending">{outcome.action === 'ask' ? 'Thinking…' : 'Searching…'}</p>
        )}
        {outcome.state === 'error' && (
          <p className="error" role="alert">
            {outcome.message}
          </p>
        )}
        {outcome.state === 'done' && (
          <>
            {outcome.answer !== undefined && (
              <>
                <h2>Answer</h2>
                <Answer text={outcome.answer} sources={outcome.sources} />
              </>
            )}
            <h2>{outcome.answer !== undefined ? 'Sources' : 'Results'}</h2>
            <SourceList sources={outcome.sources} />
          </>
        )}
      </section>
    </main>
  )
}

export default App
