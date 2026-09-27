import { useEffect, useRef, type CSSProperties } from 'react'
import { animate, stagger } from 'animejs'

type CoreProps = { mode?: 'internals' | 'serving'; compact?: boolean; activeStage?: number }

export function InferenceCore({ mode = 'internals', compact = false, activeStage = 0 }: CoreProps) {
  const root = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (reduced || !root.current) return
    const intro = animate(root.current.querySelectorAll('.core-orbit, .core-layer, .core-label'), {
      opacity: [0, 1], scale: [0.92, 1], delay: stagger(85), duration: 700, ease: 'out(3)',
    })
    const orbit = animate(root.current.querySelectorAll('.core-orbit'), {
      rotate: (_el, i) => ((i ?? 0) % 2 === 0 ? 360 : -360), duration: (_el, i) => 30000 + (i ?? 0) * 9000, loop: true, ease: 'linear',
    })
    const tokenMotion = animate(root.current.querySelectorAll('.core-token-path i'), {
      translateX: [-2, 3], opacity: [0.35, 1], delay: stagger(120), duration: 620, alternate: true, loop: true, ease: 'inOutSine',
    })
    return () => { intro.pause(); orbit.pause(); tokenMotion.pause() }
  }, [])

  return <div ref={root} className={`inference-core ${compact ? 'core-compact' : ''} ${mode === 'serving' ? 'core-serving' : ''}`} aria-label="Layered transformer inference core">
    <div className="core-orbit orbit-a" /><div className="core-orbit orbit-b" /><div className="core-orbit orbit-c" />
    <div className="core-ruler" aria-hidden="true">{Array.from({ length: 48 }, (_, i) => <i key={i} />)}</div>
    <div className="core-wafer">
      {Array.from({ length: mode === 'serving' ? 5 : 7 }, (_, i) => <div key={i} className={`core-layer ${i === activeStage % 7 ? 'layer-active' : ''}`} style={{ '--layer': i } as CSSProperties} />)}
      <div className="core-center"><span className="center-cross">×</span><span className="center-caption">{mode === 'serving' ? 'GPU' : 'ATTN'}</span><div className="center-signal" /></div>
    </div>
    <span className="core-label label-top">{mode === 'serving' ? 'CUDA / 01' : 'ROPE / 01'}</span>
    <span className="core-label label-side">{mode === 'serving' ? 'VRAM BUS' : 'KV MEMORY'}</span>
    <div className="core-token-path" aria-hidden="true"><i /><i /><i /><i /><i /></div>
  </div>
}
