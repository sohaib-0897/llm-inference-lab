import { useEffect, useState, type ReactNode } from 'react'
import Lenis from 'lenis'
import 'lenis/dist/lenis.css'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { LenisContext, PageProgressContext } from './lenisContext'

gsap.registerPlugin(ScrollTrigger)
export function LenisProvider({ children }: { children: ReactNode }) {
  const [instance, setInstance] = useState<Lenis | null>(null)
  const [progress, setProgress] = useState(0)
  useEffect(() => {
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const small = window.matchMedia('(max-width: 700px)').matches
    const lenis = new Lenis({ duration: reduced ? 0 : small ? .55 : .82, smoothWheel: !reduced, syncTouch: false, wheelMultiplier: .92, touchMultiplier: 1 })
    const onScroll = (state: Lenis) => {
      ScrollTrigger.update()
      const next = state.limit > 0 ? Math.min(1, Math.max(0, state.scroll / state.limit)) : 0
      setProgress((current) => Math.abs(current - next) > .0005 ? next : current)
    }
    const tick = (time: number) => lenis.raf(time * 1000)
    lenis.on('scroll', onScroll)
    gsap.ticker.add(tick)
    gsap.ticker.lagSmoothing(0)
    const announce = window.setTimeout(() => setInstance(lenis), 0)
    const refresh = window.setTimeout(() => ScrollTrigger.refresh(), 100)
    const onLoad = () => ScrollTrigger.refresh()
    window.addEventListener('load', onLoad)
    return () => {
      window.clearTimeout(refresh)
      window.clearTimeout(announce)
      window.removeEventListener('load', onLoad)
      lenis.off('scroll', onScroll)
      gsap.ticker.remove(tick)
      lenis.destroy()
    }
  }, [])
  return <LenisContext.Provider value={instance}><PageProgressContext.Provider value={progress}>{children}</PageProgressContext.Provider></LenisContext.Provider>
}
