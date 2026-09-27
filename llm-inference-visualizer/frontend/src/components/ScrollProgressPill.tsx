import { useReducedMotion, motion } from 'motion/react'
import { usePageProgress } from '../scroll/lenisContext'

const divisions = Array.from({ length: 36 }, (_, index) => index)

export default function ScrollProgressPill() {
  const progress = usePageProgress()
  const reduced = useReducedMotion()
  const active = Math.round(progress * (divisions.length - 1))

  return <motion.aside
    className="scroll-pill"
    role="progressbar"
    aria-label={`Page progress ${Math.round(progress * 100)} percent`}
    aria-valuemin={0}
    aria-valuemax={100}
    aria-valuenow={Math.round(progress * 100)}
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    whileHover={{ scale: 1.035 }}
    transition={reduced ? { duration: 0 } : { opacity: { duration: .35, delay: .95 }, y: { duration: .35, delay: .95 }, scale: { type: 'spring', stiffness: 360, damping: 28 } }}
  >
    <span className="scroll-pill-caption" aria-hidden="true">SCROLL</span>
    <div className="scroll-pill-track" aria-hidden="true">
      {divisions.map((division) => <i className={division <= active ? 'passed' : ''} key={division} />)}
      <motion.b
        className="scroll-pill-marker"
        animate={{ left: `${progress * 100}%` }}
        transition={reduced ? { duration: 0 } : { type: 'spring', stiffness: 220, damping: 32, mass: .55 }}
      />
    </div>
    <span className="scroll-pill-value" aria-hidden="true">{String(Math.round(progress * 100)).padStart(2, '0')}</span>
  </motion.aside>
}
