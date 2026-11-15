// Types and calls for the TechRAG API (src/techrag/api.py).

export type Mode = 'dense' | 'bm25' | 'hybrid'

export interface SearchRequest {
  query: string
  k?: number
  mode?: Mode
}

export interface Source {
  /** Path of the document, relative to the data directory. */
  source: string
  text: string
  /** Comparable only within one query and mode. */
  score: number
}

export interface SearchResponse {
  results: Source[]
}

export interface AskResponse {
  answer: string
  /** Numbered by position: a citation like [2] refers to sources[1]. */
  sources: Source[]
}

export interface HealthResponse {
  status: string
  chunks: number
}

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

const BASE = '/api'
const UNREACHABLE = 'Cannot reach the TechRAG server'

/** Turn FastAPI's `{"detail": ...}` error body into a readable message. */
function describe(detail: unknown, fallback: string): string {
  if (typeof detail === 'string') return detail
  // 422 validation errors: [{"loc": [...], "msg": "..."}]
  if (Array.isArray(detail)) {
    return detail.map((d: { msg?: string }) => d.msg ?? '').join('; ') || fallback
  }
  return fallback
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(BASE + path, init)
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e
    throw new ApiError(0, UNREACHABLE)
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    // A gateway error without FastAPI's JSON body comes from the proxy, not the API.
    if (body === null && (res.status === 502 || res.status === 504)) {
      throw new ApiError(res.status, UNREACHABLE)
    }
    throw new ApiError(res.status, describe(body?.detail, res.statusText))
  }
  return res.json()
}

function post<T>(path: string, body: SearchRequest, signal?: AbortSignal): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
}

export const getHealth = (signal?: AbortSignal) => request<HealthResponse>('/healthz', { signal })
export const search = (body: SearchRequest, signal?: AbortSignal) =>
  post<SearchResponse>('/search', body, signal)
export const ask = (body: SearchRequest, signal?: AbortSignal) =>
  post<AskResponse>('/ask', body, signal)
