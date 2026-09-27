export interface HealthResponse {
  status: 'ok'
  version: string
  ollama_available: boolean
}

export async function apiGet<T>(path: string, timeoutMs = 2500): Promise<T> {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), timeoutMs)
  try {
    const response = await fetch(path, { signal: controller.signal, headers: { Accept: 'application/json' } })
    if (!response.ok) throw new Error(`API request failed (${response.status})`)
    return await response.json() as T
  } finally {
    window.clearTimeout(timer)
  }
}

interface ArtifactResponse<T> {
  records: T[]
}

export async function loadLiveBenchmarks(fallback: BenchmarkBundle): Promise<BenchmarkBundle> {
  const endpoints = [
    ['internals', '/api/benchmarks/custom'],
    ['ollamaBaseline', '/api/benchmarks/ollama/baseline'],
    ['contextScaling', '/api/benchmarks/ollama/context'],
    ['outputScaling', '/api/benchmarks/ollama/output'],
    ['modelComparison', '/api/benchmarks/ollama/models'],
    ['concurrency', '/api/benchmarks/ollama/concurrency'],
  ] as const
  const responses = await Promise.allSettled(endpoints.map(([, path]) => apiGet<ArtifactResponse<unknown>>(path)))
  const next: BenchmarkBundle = { ...fallback }
  responses.forEach((result, index) => {
    if (result.status !== 'fulfilled') return
    const key = endpoints[index][0]
    const records = result.value.records
    if (key === 'internals') next.internals = records as BenchmarkBundle['internals']
    if (key === 'ollamaBaseline') next.ollamaBaseline = records as BenchmarkBundle['ollamaBaseline']
    if (key === 'contextScaling') next.contextScaling = records as BenchmarkBundle['contextScaling']
    if (key === 'outputScaling') next.outputScaling = records as BenchmarkBundle['outputScaling']
    if (key === 'modelComparison') next.modelComparison = records as BenchmarkBundle['modelComparison']
    if (key === 'concurrency') next.concurrency = records as BenchmarkBundle['concurrency']
  })
  return next
}
import type { BenchmarkBundle } from './types'
