import { useEffect, useState } from 'react'
import { apiGet, type HealthResponse } from './api'

export default function RuntimeStatus() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  useEffect(() => {
    let active = true
    const update = () => apiGet<HealthResponse>('/api/health').then((value) => {
      if (active) setHealth(value)
    }).catch(() => { if (active) setHealth(null) })
    void update()
    const interval = window.setInterval(() => void update(), 15000)
    return () => { active = false; window.clearInterval(interval) }
  }, [])
  const live = health !== null
  const ollamaReady = health?.ollama_available === true
  const state = !live ? 'runtime-offline' : ollamaReady ? 'runtime-online' : 'runtime-degraded'
  return <div className={`runtime-status ${state}`} title={live ? `FastAPI ${health.version}; Ollama ${ollamaReady ? 'available' : 'unavailable'}` : 'Local API unavailable; recorded results use committed benchmark data'}>
    <i aria-hidden="true" />{!live ? 'RECORDED DATA' : ollamaReady ? 'LIVE LAB' : 'LAB / OLLAMA OFF'}
  </div>
}
