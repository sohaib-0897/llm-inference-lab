import type { ReactNode } from 'react'

export function TechnicalLabel({ children, tone = 'cyan' }: { children: ReactNode; tone?: 'cyan' | 'amber' | 'red' }) {
  return <span className={`tech-label tone-${tone}`}><i />{children}</span>
}

export function MetricReadout({ label, value, unit, detail }: { label: string; value: string | number; unit?: string; detail?: string }) {
  return <div className="metric-readout"><span className="metric-label">{label}</span><strong>{value}<small>{unit}</small></strong>{detail && <span className="metric-detail">{detail}</span>}</div>
}

export function SignalChart({ values, labels, color = 'var(--cyan)' }: { values: number[]; labels: string[]; color?: string }) {
  const max = Math.max(...values)
  const points = values.map((v, i) => `${(i / Math.max(values.length - 1, 1)) * 100},${100 - (v / max) * 78 - 8}`).join(' ')
  return <div className="signal-chart">
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="Benchmark measurements plotted in order">
      {[22, 48, 74].map(y => <line key={y} x1="0" x2="100" y1={y} y2={y} className="chart-grid" />)}
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.6" vectorEffect="non-scaling-stroke" />
      {values.map((v, i) => <circle key={i} cx={(i / Math.max(values.length - 1, 1)) * 100} cy={100 - (v / max) * 78 - 8} r="1.5" fill={color} />)}
    </svg>
    <div className="chart-labels">{labels.map((label, i) => <span key={`${label}-${i}`}>{label}</span>)}</div>
  </div>
}
