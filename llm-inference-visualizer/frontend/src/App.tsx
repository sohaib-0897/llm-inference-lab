import { useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { animated, useSpring } from '@react-spring/web'
import { animate as anime } from 'animejs'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import benchmarkData from './data/benchmarks.generated.json'
import type { BenchmarkBundle, ConcurrencyResult, ContextScaling, ModelComparison } from './data/types'
import { useLenis } from './scroll/useLenis'
import ScrollProgressPill from './components/ScrollProgressPill'
import { loadLiveBenchmarks } from './data/api'

gsap.registerPlugin(ScrollTrigger)
let data = benchmarkData as unknown as BenchmarkBundle
const chapters = [
  ['PROMPT', 'A request enters the system.', 'Text arrives as a sequence. Before generation, the prompt becomes one context window.'],
  ['TOKENS', 'Text becomes discrete IDs.', 'The tokenizer maps each text fragment into token IDs the model can process.'],
  ['PREFILL', 'The prompt moves through the stack.', 'Each transformer layer computes representations for the prompt in parallel.'],
  ['ATTENTION', 'Queries meet keys and values.', 'Causal attention combines each position with the context available before it.'],
  ['KV CACHE', 'Past state stays close to compute.', 'Keys and values are retained so each new token can reuse prior work.'],
  ['DECODE', 'One next token is selected.', 'The model appends a selected token, then repeats the decode step.'],
  ['OUTPUT', 'Tokens resolve back into text.', 'The generated sequence is decoded for the caller as it streams.'],
] as const
const sections = [
  ['top', 'OVERVIEW'], ['internals', 'PROMPT'], ['kv-cache', 'KV CACHE'], ['real-model', 'GPU'],
  ['comparison', 'MODELS'], ['concurrency', 'CONCURRENCY'], ['context-length', 'CONTEXT'],
  ['quantization', 'QUANTIZATION'], ['onnx', 'RUNTIME'],
]
const byScenario = (scenario: string) => data.internals.find((row) => row.scenario === scenario)!
const format = (value: number, digits = 1) => Number(value).toFixed(digits)

function Atmosphere({ kind }: { kind: 'hero' | 'gpu' }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const reduce = useReducedMotion()
  useEffect(() => {
    const el = canvas.current
    const gl = el?.getContext('webgl', { alpha: true, antialias: false })
    if (!el || !gl || reduce || window.innerWidth < 900) return
    const vertex = gl.createShader(gl.VERTEX_SHADER)!
    gl.shaderSource(vertex, 'attribute vec2 p; varying vec2 uv; void main(){uv=p*.5+.5;gl_Position=vec4(p,0.,1.);}')
    gl.compileShader(vertex)
    const fragment = gl.createShader(gl.FRAGMENT_SHADER)!
    gl.shaderSource(fragment, 'precision mediump float; varying vec2 uv; uniform float t; uniform vec2 m; uniform vec3 a; uniform vec3 b; void main(){vec2 p=uv-.5; float w=sin(p.x*5.+t*.33+sin(p.y*4.-t*.22))*cos(p.y*4.-t*.27); float d=length(p-vec2(.27,-.07)); float g=exp(-d*3.2); float q=exp(-length(p+vec2(.22,.2))*5.); vec3 c=mix(a,b,smoothstep(-.8,.8,w)); c+=g*a*.48+q*b*.18; c*=.72+.28*sin(t*.18); gl_FragColor=vec4(c, .18);}')
    gl.compileShader(fragment)
    const program = gl.createProgram()!
    gl.attachShader(program, vertex); gl.attachShader(program, fragment); gl.linkProgram(program); gl.useProgram(program)
    const buffer = gl.createBuffer()!; gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,1,1]), gl.STATIC_DRAW)
    const p = gl.getAttribLocation(program, 'p'); gl.enableVertexAttribArray(p); gl.vertexAttribPointer(p, 2, gl.FLOAT, false, 0, 0)
    const t = gl.getUniformLocation(program, 't')!, m = gl.getUniformLocation(program, 'm')!
    const a = gl.getUniformLocation(program, 'a')!, b = gl.getUniformLocation(program, 'b')!
    gl.uniform3f(a, kind === 'hero' ? .42 : .52, kind === 'hero' ? .15 : .63, kind === 'hero' ? .08 : .26)
    gl.uniform3f(b, kind === 'hero' ? .72 : .20, kind === 'hero' ? .34 : .36, kind === 'hero' ? .11 : .40)
    let raf = 0, visible = false
    const started = performance.now()
    const observer = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; if (visible && !document.hidden && !raf) raf = requestAnimationFrame(draw); else if ((!visible || document.hidden) && raf) { cancelAnimationFrame(raf); raf = 0 } })
    observer.observe(el)
    const resize = () => { const ratio = Math.min(devicePixelRatio || 1, 1.35); el.width = el.clientWidth * ratio; el.height = el.clientHeight * ratio; gl.viewport(0, 0, el.width, el.height) }
    const draw = (now: number) => { if (visible && !document.hidden) { gl.uniform1f(t, (now - started) / 1000); gl.uniform2f(m, .5, .5); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4); raf = requestAnimationFrame(draw) } else raf = 0 }
    const visibility = () => { if (document.hidden && raf) { cancelAnimationFrame(raf); raf = 0 } else if (!document.hidden && visible && !raf) raf = requestAnimationFrame(draw) }
    resize(); window.addEventListener('resize', resize); document.addEventListener('visibilitychange', visibility)
    return () => { cancelAnimationFrame(raf); observer.disconnect(); window.removeEventListener('resize', resize); document.removeEventListener('visibilitychange', visibility); gl.getExtension('WEBGL_lose_context')?.loseContext() }
  }, [kind, reduce])
  return <div className={`atmosphere atmosphere-${kind}`}><canvas ref={canvas} aria-hidden="true"/><div className="atmosphere-fallback"/></div>
}

function TokenField({ count = 18 }: { count?: number }) {
  const root = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = root.current
    if (!el || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const targets = el.querySelectorAll<HTMLElement>('.flight-token')
    const animations = [...targets].map((target, i) => anime(target, { translateX: `${110 + (i % 4) * 34}px`, translateY: `${(i % 2 ? -1 : 1) * (18 + i % 5 * 9)}px`, opacity: [{ to: .25, duration: 0 }, { to: .95, duration: 600 }, { to: .15, duration: 1100 }], delay: i * 96, duration: 2100 + i % 5 * 270, loop: true, alternate: true, ease: 'inOutSine' }))
    const observer = new IntersectionObserver(([entry]) => animations.forEach((animation) => entry.isIntersecting ? animation.play() : animation.pause()))
    observer.observe(el)
    return () => { observer.disconnect(); animations.forEach((animation) => animation.cancel()) }
  }, [])
  return <div className="token-field" ref={root} aria-hidden="true">{Array.from({ length: count }, (_, i) => <i className="flight-token" key={i}>{['0','1','KV','→'][i % 4]}</i>)}</div>
}

function HeroMachine({ decodeRate }: { decodeRate: number }) {
  const root = useRef<HTMLDivElement>(null)
  const reduce = useReducedMotion()
  useEffect(() => {
    const element = root.current
    if (!element || reduce) return
    const paths = [...element.querySelectorAll<SVGPathElement>('[data-machine-draw]')]
    const draws = paths.map((path) => {
      const length = path.getTotalLength()
      path.style.strokeDasharray = `${length}`
      path.style.strokeDashoffset = `${length}`
      return anime(path, { strokeDashoffset: [length, 0], duration: 850, delay: 620, ease: 'outQuad' })
    })
    const rings = [...element.querySelectorAll<SVGGElement>('.machine-ring')].map((ring, index) => anime(ring, {
      rotate: index === 0 ? 360 : -360,
      duration: index === 0 ? 52000 : 72000,
      delay: 900,
      loop: true,
      ease: 'linear',
    }))
    const labels = [...element.querySelectorAll<SVGElement>('.machine-annotation')].map((label, index) => anime(label, {
      opacity: [0, 1],
      translateY: [4, 0],
      duration: 430,
      delay: index * 110,
      ease: 'outQuad',
    }))
    const tokens = [...element.querySelectorAll<SVGCircleElement>('.machine-token')].map((token, index) => anime(token, {
      cx: [48, 212],
      opacity: [.25, 1, .25],
      duration: 2100,
      delay: 1320 + index * 430,
      loop: true,
      ease: 'inOutSine',
    }))
    const led = element.querySelector<SVGCircleElement>('.machine-led')
    const pulse = led ? anime(led, { opacity: [.45, 1], scale: [.82, 1.16], duration: 1150, delay: 1850, loop: true, alternate: true, ease: 'inOutSine' }) : null
    return () => [...draws, ...rings, ...labels, ...tokens, ...(pulse ? [pulse] : [])].forEach((animation) => animation.cancel())
  }, [reduce])

  return <div className="hero-machine" ref={root}>
    <svg viewBox="0 0 620 520" role="img" aria-label="A prompt travels through processing rings and retained KV memory before a decoded token emerges">
      <g className="machine-ring machine-ring-outer"><circle cx="310" cy="254" r="202"/><circle cx="310" cy="254" r="187"/></g>
      <g className="machine-ring machine-ring-inner"><circle cx="310" cy="254" r="151"/><circle cx="310" cy="254" r="138"/></g>
      <path className="machine-flow" data-machine-draw d="M24 254H198M420 254H590"/>
      <path className="machine-connector" data-machine-draw d="M310 60V128M310 380V420"/>
      <g className="machine-annotation"><text x="24" y="225">PROMPT</text><text x="250" y="40">PREFILL</text><text x="246" y="462">KV CACHE</text><text x="498" y="225">DECODE</text></g>
      <g className="machine-input-tokens"><rect x="47" y="241" width="25" height="25"/><rect x="81" y="241" width="25" height="25"/><rect x="115" y="241" width="25" height="25"/></g>
      <circle className="machine-token" cx="48" cy="254" r="6"/>
      <g className="machine-core">
        <rect x="245" y="188" width="130" height="132" rx="2"/>
        <rect className="machine-layer" x="263" y="207" width="94" height="14"/>
        <rect className="machine-layer" x="255" y="234" width="110" height="14"/>
        <rect className="machine-layer" x="263" y="261" width="94" height="14"/>
        <rect className="machine-layer" x="255" y="288" width="110" height="14"/>
        <text x="276" y="344">TRANSFORMER</text>
      </g>
      <g className="machine-memory">{Array.from({ length: 12 }, (_, index) => <rect key={index} x={218 + (index % 6) * 31} y={392 + Math.floor(index / 6) * 15} width="22" height="8"/>)}</g>
      <g className="machine-output"><rect x="548" y="238" width="30" height="32" rx="2"/><circle className="machine-led" cx="563" cy="254" r="3"/></g>
      <text className="machine-rate" x="310" y="498" textAnchor="middle">{decodeRate.toFixed(1)} TOK/S · CACHED DECODE</text>
      <path className="machine-tick" d="M310 52V60M512 254H520M310 448V456M100 254H108"/>
    </svg>
  </div>
}

function useScrollScenes() {
  const reduce = useReducedMotion()
  useEffect(() => {
    if (reduce) return
    const contexts: gsap.Context[] = []
    const media = gsap.matchMedia()
    media.add('(min-width: 900px)', () => {
      const ctx = gsap.context(() => {
        const hero = gsap.timeline({ scrollTrigger: { trigger: '.hero', start: 'top top', end: '+=100%', scrub: .65, pin: true, anticipatePin: 1, invalidateOnRefresh: true } })
        hero.to('.hero-scene-macro', { yPercent: -2, scale: 1.08, ease: 'none', duration: .4 }, .2)
          .to('.hero-copy', { yPercent: -3, opacity: .82, ease: 'none', duration: .2 }, .62)
          .to('.hero-copy', { yPercent: -8, opacity: 0, ease: 'power1.in', duration: .18 }, .82)
          .to('.hero-scene-macro', { yPercent: -10, scale: 1.13, opacity: .18, ease: 'power1.in', duration: .18 }, .82)
        const handoff = gsap.timeline({ scrollTrigger: { trigger: '.hero', start: 'top top', end: '+=100%', scrub: .65, invalidateOnRefresh: true } })
        handoff.to('.handoff-token', { autoAlpha: 1, scale: 1.15, ease: 'none', duration: .08 }, .82).to('.handoff-token', { x: '-30vw', y: '-4vh', scale: .62, ease: 'none', duration: .2 }, .9).to('.handoff-token', { autoAlpha: 0, ease: 'none', duration: .06 }, .98)
        const story = gsap.timeline({ scrollTrigger: { trigger: '#internals', start: 'top top', end: '+=1450', scrub: .7, pin: '.pipeline-stage', anticipatePin: 1, invalidateOnRefresh: true, onUpdate: (self) => window.dispatchEvent(new CustomEvent('pipeline-progress', { detail: self.progress })) } })
        story.fromTo('.pipeline-trace', { scaleX: 0 }, { scaleX: 1, ease: 'none' }, 0).to('.pipeline-carrier', { x: () => document.querySelector('.pipeline-track')?.clientWidth ?? 500, ease: 'none' }, 0)
        const emphasis = ['.prompt-node', '.tokens-node', '.attention-web', '.attention-web', '.cache-stack', '.pipeline-decode-holder', '.output-token b']
        gsap.set('.pipeline-node,.attention-web,.cache-stack,.pipeline-decode-holder,.output-token b', { autoAlpha: .18, scale: .92 })
        emphasis.forEach((selector, index) => {
          story.to(selector, { autoAlpha: 1, scale: 1.07, ease: 'none', duration: .42 }, index)
            .to(selector, { autoAlpha: .22, scale: .94, ease: 'none', duration: .58 }, index + .42)
        })
        gsap.fromTo('.kv-history-left .memory-cell', { y: 18, opacity: .25 }, { y: 0, opacity: 1, stagger: .035, duration: .6, scrollTrigger: { trigger: '#kv-cache', start: 'top 72%' } })
        gsap.fromTo('.gpu-loading', { scaleX: 0 }, { scaleX: 1, duration: 1.25, ease: 'power2.out', scrollTrigger: { trigger: '#real-model', start: 'top 65%' } })
        gsap.fromTo('.race-beam', { scaleX: 0 }, { scaleX: 1, stagger: .18, duration: .9, transformOrigin: 'left', scrollTrigger: { trigger: '#onnx', start: 'top 68%' } })
        const context = gsap.timeline({ scrollTrigger: { trigger: '#context-length', start: 'top top', end: '+=900', scrub: .5, pin: '.context-stage', onUpdate: (self) => window.dispatchEvent(new CustomEvent('context-progress', { detail: self.progress })) } })
        context.to('.context-tape', { scaleX: 1, ease: 'none' }, 0)
      })
      contexts.push(ctx)
      return () => ctx.revert()
    })
    return () => { media.revert(); contexts.forEach((ctx) => ctx.revert()) }
  }, [reduce])
}

function Metric({ label, value, unit, note, className = '' }: { label: string; value: string; unit?: string; note?: string; className?: string }) {
  return <motion.div className={`metric ${className}`} whileHover={{ y: -4 }} transition={{ type: 'spring', stiffness: 280, damping: 20 }} title={note}>
    <span className="metric-label">{label}</span><AnimatePresence mode="wait"><motion.div className="metric-value" key={value} initial={{ y: 9, opacity: 0 }} animate={{ y: 0, opacity: 1 }} exit={{ y: -7, opacity: 0 }} transition={{ duration: .24 }}>{value}<small>{unit}</small></motion.div></AnimatePresence>{note && <span className="metric-note">{note}</span>}
  </motion.div>
}

function SectionHeader({ eyebrow, title, copy, light = false }: { eyebrow: string; title: string; copy: string; light?: boolean }) {
  return <header className={`section-head ${light ? 'section-head-light' : ''}`}><span className="section-index mono">{eyebrow}</span><h2>{title}</h2><p>{copy}</p></header>
}

function Pipeline() {
  const lenis = useLenis()
  const reduce = useReducedMotion()
  const stageRef = useRef(0)
  const [stage, setStage] = useState(0)
  useEffect(() => { const update = (event: Event) => { const progress = (event as CustomEvent<number>).detail; const index = Math.min(chapters.length - 1, Math.floor(progress * chapters.length)); if (index !== stageRef.current) { stageRef.current = index; setStage(index) } }; window.addEventListener('pipeline-progress', update); return () => window.removeEventListener('pipeline-progress', update) }, [])
  const chapter = chapters[stage]
  return <section id="internals" className="pipeline-section scene-cyan">
    <div className="pipeline-stage">
      <div className="pipeline-head"><span className="section-index mono">FROM PROMPT TO TOKEN</span><span className="pipeline-readout mono">PROCESS {stage + 1} OF {chapters.length}</span></div>
      <div className="pipeline-layout">
        <div className="pipeline-copy"><AnimatePresence mode="wait"><motion.div key={chapter[0]} initial={{ opacity: 0, x: -22 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 18 }} transition={{ duration: .3 }}><span className="mono pipeline-step-label">{chapter[0]}</span><h2>{chapter[1]}</h2><p>{chapter[2]}</p></motion.div></AnimatePresence><div className="pipeline-dots">{chapters.map((item, index) => <button key={item[0]} aria-label={`Show ${item[0]} stage`} aria-pressed={index === stage} className={index === stage ? 'selected' : ''} onClick={() => { const trigger = ScrollTrigger.getAll().find((entry) => entry.trigger?.id === 'internals'); if (trigger && lenis) lenis.scrollTo(trigger.start + (index / (chapters.length - 1)) * (trigger.end - trigger.start), { duration: .6 }); else setStage(index) }}/>)}</div></div>
        <div className="pipeline-visual">
          <div className="pipeline-track"><div className="pipeline-trace"/><span className="pipeline-carrier">◆</span></div>
          <div className={`pipeline-node prompt-node ${stage >= 0 ? 'lit' : ''}`}><span>PROMPT</span><b>“Explain inference”</b><small>UTF-8 TEXT</small></div>
          <div className={`pipeline-node tokens-node ${stage >= 1 ? 'lit' : ''}`}><span>IDS</span><div className="token-blocks"><i>EX</i><i>PL</i><i>AIN</i></div><small>TOKENIZER</small></div>
          <div className={`attention-web ${stage >= 2 ? 'lit' : ''}`}><svg viewBox="0 0 240 150" aria-hidden="true"><path d="M12 120 74 25 130 99 198 18 228 122M12 120 130 99 228 122M74 25 198 18M74 25 228 122"/><path d="M12 120 198 18M74 25 130 99"/></svg><b>ATTENTION</b></div>
          <div className={`cache-stack ${stage >= 4 ? 'lit' : ''}`}><span>KV STATE</span>{Array.from({ length: 12 }, (_, i) => <i key={i} className={i < Math.max(2, stage * 2) ? 'filled' : ''}/>)}</div>
          <div className="pipeline-decode-holder"><motion.div className={`decode-orbit ${stage >= 5 ? 'lit' : ''}`} animate={{ rotate: !reduce && stage >= 5 ? 360 : 0 }} transition={{ duration: 8, repeat: !reduce && stage >= 5 ? Infinity : 0, ease: 'linear' }}><i/><i/><i/><span>DECODE</span></motion.div></div>
          <div className={`output-token ${stage === 6 ? 'launch' : ''}`}><b>next</b><span>OUTPUT</span></div>
          <TokenField count={10}/>
          <div className="pipeline-baseline mono"><span>PARALLEL PREFILL</span><span>SEQUENTIAL DECODE</span></div>
        </div>
      </div>
      <ol className="mobile-pipeline" aria-label="Prompt to output sequence">
        <li><span>PROMPT</span><p>A request enters as text.</p></li>
        <li><span>TOKENIZATION</span><p>Text becomes discrete token IDs.</p></li>
        <li><span>ATTENTION</span><p>Each layer reads the context available so far.</p></li>
        <li><span>KV CACHE</span><p>Earlier keys and values stay available for reuse.</p></li>
        <li><span>DECODE</span><p>The model selects the next token.</p></li>
        <li><span>OUTPUT</span><p>Generated tokens resolve back into text.</p></li>
      </ol>
    </div>
  </section>
}

function KVCache() {
  const [selectedGen, setSelectedGen] = useState(32)
  const rows = useMemo(() => [16, 32, 64].map((gen) => ({ gen, cached: data.internals.find((r) => r.scenario === 'kv_cache_enabled' && r.gen_len === gen)!, recompute: data.internals.find((r) => r.scenario === 'kv_cache_disabled' && r.gen_len === gen)! })), [])
  const row = rows.find((entry) => entry.gen === selectedGen)!
  const speedup = row.recompute.total_latency_ms / row.cached.total_latency_ms
  return <section id="kv-cache" className="kv-section scene-paper">
    <div className="section-wrap"><SectionHeader eyebrow="RETAINED STATE" title="Remember the prefix." copy="Decoding can reuse keys and values already computed for prior tokens. Full-prefix recomputation processes the growing sequence again." light/>
      <div className="kv-control mono"><span>GENERATED TOKENS</span>{rows.map((entry) => <button key={entry.gen} className={selectedGen === entry.gen ? 'chosen' : ''} onClick={() => setSelectedGen(entry.gen)}>{entry.gen}</button>)}</div>
      <label className="kv-scrub"><span>Scrub the generated sequence</span><input aria-label="Generated token count" type="range" min="0" max="2" step="1" value={rows.findIndex((entry) => entry.gen === selectedGen)} onChange={(event) => setSelectedGen(rows[Number(event.target.value)].gen)}/><output>{selectedGen} tokens</output></label>
      <div className="kv-comparison">
        <div className="kv-path kv-recompute"><div className="path-top"><span className="path-symbol">↻</span><div><span className="mono">MODE A / EVERY STEP</span><h3>Full-prefix recomputation</h3></div></div><div className="history-lane kv-history-left">{Array.from({ length: Math.min(selectedGen, 32) }, (_, i) => <i className="memory-cell" key={i} style={{ '--delay': `${i * 12}ms` } as React.CSSProperties}/>)}</div><p>Earlier tokens are included again with each decode step.</p><Metric label="TOTAL LATENCY" value={format(row.recompute.total_latency_ms, 2)} unit="ms" note={`Measured ${row.gen} generated tokens`}/><Metric label="DECODE RATE" value={format(row.recompute.tokens_per_second, 1)} unit="tok/s"/></div>
        <div className="kv-vs mono">VS</div>
        <div className="kv-path kv-cached"><div className="path-top"><span className="path-symbol">＋</span><div><span className="mono">MODE B / APPEND STATE</span><h3>Cached incremental decode</h3></div></div><div className="history-lane">{Array.from({ length: Math.min(selectedGen, 32) }, (_, i) => <i className="memory-cell retained" key={i}/>)}</div><p>Prior key/value state stays resident; one new token state is appended.</p><Metric label="TOTAL LATENCY" value={format(row.cached.total_latency_ms, 2)} unit="ms" note={`Measured ${row.gen} generated tokens`}/><Metric label="DECODE RATE" value={format(row.cached.tokens_per_second, 1)} unit="tok/s"/></div>
      </div><div className="kv-result"><span className="mono">MEASURED LATENCY RATIO</span><strong>{format(speedup, 2)}<small>×</small></strong><span className="mono">CACHE / RECOMPUTE · {row.gen} OUTPUT TOKENS</span></div><p className="source-line mono">SOURCE · {data.sourceFiles.internals}</p>
    </div>
  </section>
}

function Serving() {
  const model = data.modelComparison.find((entry) => entry.model_name === 'qwen2.5:0.5b')!
  const runtime = data.ollamaBaseline[0]
  const hardware = data.hardware
  return <section id="real-model" className="serving-section scene-gpu"><Atmosphere kind="gpu"/><div className="section-wrap serving-content"><SectionHeader eyebrow="OLLAMA / REAL MODEL" title="Now on hardware." copy="A pretrained model is loaded into GPU memory and served through Ollama. These are runtime observations, not kernel-level profiles."/>
      <div className="gpu-stage"><div className="gpu-model"><span>MODEL / Q4_K_M</span><strong>qwen2.5:<br/><em>0.5b</em></strong><div className="gpu-loading-track"><i className="gpu-loading"/></div><span>LOADING WEIGHTS → VRAM</span></div><div className="gpu-chip"><div className="gpu-ring"><div className="gpu-ring-inner"><span>RESIDENT MODEL STATE</span><b>{format(model.vram_size_mb, 1)}<small> MiB</small></b><span>MEASURED RUNTIME RESIDENCY</span></div></div><div className="gpu-chip-core">RTX 4050</div></div><TokenField count={16}/><div className="gpu-path-label">REQUEST → PREFILL → DECODE → STREAM</div></div>
      <div className="serving-readouts"><Metric label="TIME TO FIRST TOKEN" value={format(runtime.mean_ttft_ms, 2)} unit="ms" note="Mean client TTFT"/><Metric label="SERVER DECODE" value={format(runtime.mean_server_tokens_per_sec, 2)} unit="tok/s" note="Mean Ollama server rate"/><Metric label="CLIENT THROUGHPUT" value={format(runtime.mean_client_tokens_per_sec, 2)} unit="tok/s" note="Mean client-observed rate"/><Metric label="GPU REPORTED" value={hardware.ollamaProcessor.replace('% GPU', '%')} note="Processor field from ollama ps"/></div><div className="gpu-foot mono"><span>GPU / {hardware.gpu}</span><span>RUNTIME / OLLAMA</span><span>MEASURE / RUNTIME RESIDENCY</span></div><p className="source-line mono">SOURCE · {data.sourceFiles.ollamaBaseline} + BENCHMARKS.md · model record {model.model_name}</p>
    </div>
  </section>
}

function ModelMass({ model, index }: { model: ModelComparison; index: number }) {
  const [hovered, setHovered] = useState(false)
  const spring = useSpring({ scale: hovered ? 1.035 : 1, y: hovered ? -8 : 0, config: index === 0 ? { mass: .7, tension: 280, friction: 18 } : { mass: 2.2, tension: 140, friction: 28 } })
  return <animated.article className={`model-mass ${index === 0 ? 'mass-small' : 'mass-large'}`} style={{ transform: spring.scale.to((scale) => `translate3d(0,${spring.y.get()}px,0) scale(${scale})`) }} onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)}>
    <div className="mass-orb"><div className="mass-core"><span className="mono">{model.param_size}</span><b>{model.model_name}</b></div><div className="mass-orbit orbit-a"/><div className="mass-orbit orbit-b"/>{Array.from({ length: index ? 16 : 8 }, (_, i) => <i key={i} style={{ '--i': i } as React.CSSProperties}/>)}</div><div className="mass-data"><span className="mono">{model.quantization} / VRAM {format(model.vram_size_mb, 1)} MiB</span><div className="mass-metrics"><Metric label="TTFT" value={format(model.mean_ttft_ms, 2)} unit="ms"/><Metric label="SERVER DECODE" value={format(model.mean_eval_tokens_per_sec, 2)} unit="tok/s"/><Metric label="TOTAL LATENCY" value={format(model.mean_total_latency_ms, 2)} unit="ms"/></div></div>
  </animated.article>
}

function Comparison() {
  return <section id="comparison" className="compare-section scene-neutral"><div className="section-wrap"><SectionHeader eyebrow="MODEL SCALE / SAME MACHINE" title="Two different masses." copy="The small model feels nimble. The larger module occupies more resident memory and takes longer to produce each response." light/><div className="mass-compare"><ModelMass model={data.modelComparison.find((entry) => entry.model_name === 'qwen2.5:0.5b')!} index={0}/><div className="mass-axis"><span className="mono">PARAMETERS</span><i/><span>0.5B <b>→</b> 4B</span><small>VRAM RESIDENCY / OUTPUT RATE</small></div><ModelMass model={data.modelComparison.find((entry) => entry.model_name === 'qwen3:4b')!} index={1}/></div><p className="source-line mono">SOURCE · {data.sourceFiles.modelComparison} · same GPU serving environment</p></div></section>
}

function PacketStream({ concurrency }: { concurrency: number }) {
  const root = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!root.current || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const animations = [...root.current.querySelectorAll<HTMLElement>('.request-packet')].map((target, i) => anime(target, { translateX: ['0%', '780%'], opacity: [{ to: 1, duration: 100 }, { to: 0, duration: 500 }], delay: (i % concurrency) * 270, duration: 2200, loop: true, ease: 'inOutSine' }))
    const observer = new IntersectionObserver(([entry]) => animations.forEach((animation) => entry.isIntersecting ? animation.play() : animation.pause()))
    observer.observe(root.current)
    return () => { observer.disconnect(); animations.forEach((animation) => animation.cancel()) }
  }, [concurrency])
  return <div className="request-routes" ref={root}>{Array.from({ length: concurrency }, (_, i) => <div className="request-route" key={i}><span className="mono">REQ {String(i + 1).padStart(2, '0')}</span><i className="request-packet"/><span className="route-end mono">GPU</span></div>)}</div>
}

function Concurrency() {
  const [selected, setSelected] = useState(1)
  const result = data.concurrency.find((entry) => entry.concurrency === selected)!
  const baseline = data.concurrency[0]
  const queue = useSpring({ width: `${selected * 17}%`, config: { mass: .8, tension: 190, friction: 22 } })
  const improvement = (data.concurrency[2].aggregate_tokens_per_sec / baseline.aggregate_tokens_per_sec - 1) * 100
  const latencyIncrease = data.concurrency[2].p95_latency_ms / baseline.p95_latency_ms
  return <section id="concurrency" className="concurrency-section scene-night"><div className="section-wrap"><SectionHeader eyebrow="QUEUE PRESSURE" title="More requests. More waiting." copy="Concurrency adds a modest amount of aggregate output while the slowest request waits longer in line."/><div className="concurrency-layout"><div className="queue-scene"><div className="queue-entry mono">INCOMING REQUESTS</div><PacketStream concurrency={selected}/><div className="queue-bin"><span className="mono">WAITING QUEUE</span><animated.i style={{ width: queue.width }}/></div><div className="gpu-queue">GPU / BATCHED EXECUTION</div><div className="queue-complete mono">COMPLETIONS → STREAM</div></div><div className="concurrency-data"><span className="mono">SIMULTANEOUS REQUESTS</span><div className="concurrency-switch">{data.concurrency.map((entry: ConcurrencyResult) => <button key={entry.concurrency} className={selected === entry.concurrency ? 'chosen' : ''} onClick={() => setSelected(entry.concurrency)}>{entry.concurrency}</button>)}</div><div className="p95-dial" style={{ '--dial': `${Math.min(100, result.p95_latency_ms / data.concurrency[2].p95_latency_ms * 100)}%` } as React.CSSProperties}><span className="mono">P95 REQUEST</span><strong>{format(result.p95_latency_ms, 2)}<small>ms</small></strong><i/></div><div className="conc-metrics"><Metric label="AGGREGATE THROUGHPUT" value={format(result.aggregate_tokens_per_sec, 2)} unit="tok/s"/><Metric label="MEAN REQUEST LATENCY" value={format(result.mean_latency_ms, 2)} unit="ms"/></div></div></div><div className="tradeoff-strip mono"><span>AT 4 CONCURRENT REQUESTS</span><b>+{format(improvement, 1)}% aggregate throughput</b><i>·</i><b>{format(latencyIncrease, 1)}× P95 latency</b></div><p className="source-line mono">SOURCE · {data.sourceFiles.concurrency} · measured concurrency 1 / 2 / 4</p></div></section>
}

function ContextLength() {
  const [index, setIndex] = useState(0)
  const root = useRef<HTMLDivElement>(null)
  useEffect(() => { const update = (event: Event) => { const value = (event as CustomEvent<number>).detail; const next = Math.min(data.contextScaling.length - 1, Math.floor(value * data.contextScaling.length)); setIndex((previous) => previous === next ? previous : next) }; window.addEventListener('context-progress', update); return () => window.removeEventListener('context-progress', update) }, [])
  const row: ContextScaling = data.contextScaling[index]
  return <section id="context-length" className="context-section scene-lime"><div className="section-wrap context-stage" ref={root}>
    <div className="context-overview"><SectionHeader eyebrow="PROMPT EVALUATION" title="Context takes up space." copy="The input sequence expands. Prompt evaluation measures the work needed before the first generated token." light/><div className="context-count"><span>ACTUAL PROMPT TOKENS</span><AnimatePresence mode="wait"><motion.strong key={row.actual_prompt_tokens} initial={{ y: 18, opacity: 0 }} animate={{ y: 0, opacity: 1 }} exit={{ y: -12, opacity: 0 }}>{row.actual_prompt_tokens.toLocaleString()}</motion.strong></AnimatePresence><span>{row.target_approx_tokens.toLocaleString()} approximate target</span></div></div>
    <div className="context-tape-wrap"><div className="context-tape" style={{ '--tape-size': `${30 + index * 16}%` } as React.CSSProperties}>{Array.from({ length: 54 }, (_, i) => <i key={i} style={{ opacity: .35 + (i / 54) * .65 }}/>)}</div><div className="context-ruler"><span>START</span><span>INPUT SEQUENCE</span><span>{row.actual_prompt_tokens.toLocaleString()} TOKENS</span></div></div>
    <div className="context-readout-strip"><div className="context-data"><Metric label="PROMPT EVALUATION" value={format(row.mean_prompt_eval_duration_ms, 2)} unit="ms" note="Mean prompt evaluation duration"/><Metric label="CLIENT TTFT" value={format(row.mean_client_ttft_ms, 2)} unit="ms"/><Metric label="TOTAL LATENCY" value={format(row.mean_total_latency_ms, 2)} unit="ms"/></div><div className="context-selector" aria-label="Select measured prompt length">{data.contextScaling.map((entry, i) => <button key={entry.actual_prompt_tokens} aria-pressed={index === i} className={index === i ? 'chosen' : ''} onClick={() => setIndex(i)}>{entry.actual_prompt_tokens.toLocaleString()}</button>)}</div></div>
    <p className="source-line">SOURCE · {data.sourceFiles.contextScaling} · {row.output_tokens} output tokens per run</p></div></section>
}

function Quantization() {
  const fp32 = byScenario('precision_fp32'), int8 = byScenario('precision_int8')
  const ratio = fp32.process_memory_mb / int8.process_memory_mb
  return <section id="quantization" className="quant-section scene-graphite"><div className="section-wrap"><SectionHeader eyebrow="WEIGHT ONLY / CPU" title="Compress the weights." copy="INT8 reduces this measured footprint. In this CPU implementation the smaller representation also decoded more slowly."/><div className="quant-layout"><div className="weight-visual"><div className="weight-label mono">FP32 · {format(fp32.process_memory_mb, 2)} MB</div><div className="weight-matrix matrix-fp32">{Array.from({ length: 48 }, (_, i) => <i key={i} style={{ '--i': i } as React.CSSProperties}/>)}</div><div className="compress-arrow">↓<span className="mono">QUANTIZE</span></div><div className="weight-matrix matrix-int8">{Array.from({ length: 48 }, (_, i) => <i key={i} style={{ '--i': i } as React.CSSProperties}/>)}</div><div className="weight-label mono">INT8 · {format(int8.process_memory_mb, 2)} MB</div><div className="compress-stamp"><strong>{format(ratio, 2)}×</strong><span className="mono">SMALLER FOOTPRINT</span></div></div><div className="quant-data"><div className="mono quant-warning">LOWER MEMORY ≠ LOWER LATENCY</div><div className="quant-metrics"><div><span className="mono">FP32 / CPU</span><Metric label="DECODE RATE" value={format(fp32.tokens_per_second, 2)} unit="tok/s"/><Metric label="TOTAL LATENCY" value={format(fp32.total_latency_ms, 3)} unit="ms"/></div><div><span className="mono">INT8 / CPU</span><Metric label="DECODE RATE" value={format(int8.tokens_per_second, 2)} unit="tok/s"/><Metric label="TOTAL LATENCY" value={format(int8.total_latency_ms, 3)} unit="ms"/></div></div><p>Footprint and decode latency are measured for the committed CPU benchmark. The matrix is an explanatory schematic, not a count of model weights.</p></div></div><p className="source-line mono">SOURCE · {data.sourceFiles.internals} · precision_fp32 / precision_int8</p></div></section>
}

function Onnx() {
  const pytorch = byScenario('engine_pytorch_cpu'), onnx = byScenario('engine_onnxruntime_cpu')
  const ratio = pytorch.total_latency_ms / onnx.total_latency_ms
  return <section id="onnx" className="onnx-section scene-blue"><div className="section-wrap"><SectionHeader eyebrow="CPU EXECUTION TRACE" title="Same graph. Faster path." copy="The ONNX Runtime CPU path completed this measured forward pass sooner than the PyTorch CPU baseline."/><div className="runtime-race"><div className="race-row"><span className="mono">PYTORCH CPU</span><div className="race-track"><i className="race-beam torch-beam"/><b>FORWARD PASS / HOST EXECUTION</b></div><strong>{format(pytorch.total_latency_ms, 3)}<small>ms</small></strong></div><div className="race-row"><span className="mono">ONNX RUNTIME</span><div className="race-track"><i className="race-beam onnx-beam"/><b>FORWARD PASS / OPTIMIZED GRAPH</b></div><strong>{format(onnx.total_latency_ms, 3)}<small>ms</small></strong></div><div className="race-ruler mono"><span>0 ms</span><span>3 ms</span><span>6 ms</span><span>9 ms</span></div><div className="runtime-result"><span className="mono">MEASURED SPEEDUP</span><strong>{format(ratio, 2)}<small>×</small></strong><span className="mono">CPU FORWARD PASS</span></div></div><p className="source-line mono">SOURCE · {data.sourceFiles.internals} · engine_pytorch_cpu / engine_onnxruntime_cpu</p><footer className="endnote mono"><span>LLM INFERENCE LAB / SYSTEMS FIELD NOTES</span><a href="https://github.com/sohaib-0897/llm-inference-lab" target="_blank" rel="noreferrer">SOURCE CODE ↗</a><span>MEASUREMENTS FROM COMMITTED BENCHMARK ARTIFACTS</span></footer></div></section>
}

function Navigation() {
  const lenis = useLenis()
  const [active, setActive] = useState(sections[0][0])
  const [mobileOpen, setMobileOpen] = useState(false)
  useEffect(() => {
    const observer = new IntersectionObserver((entries) => entries.forEach((entry) => { if (entry.isIntersecting) setActive(entry.target.id) }), { rootMargin: '-42% 0px -46% 0px' })
    sections.forEach(([id]) => { const element = document.getElementById(id); if (element) observer.observe(element) })
    return () => observer.disconnect()
  }, [])
  const activeIndex = Math.max(0, sections.findIndex(([id]) => id === active))
  const goTo = (id: string) => {
    if (lenis) lenis.scrollTo(`#${id}`, { offset: 0, duration: .85 })
    else document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
    setMobileOpen(false)
  }
  return <><nav className="nav-rail" aria-label="Story sections">{sections.map(([id, name]) => <button key={id} className={active === id ? 'active' : ''} aria-label={`Go to ${name}`} onClick={() => goTo(id)}><span className="nav-name">{name}</span><i/></button>)}</nav><div className={`mobile-navigation ${active === 'top' ? 'at-top' : ''}`}><button className="mobile-progress" aria-expanded={mobileOpen} aria-haspopup="menu" onClick={() => setMobileOpen((open) => !open)}><span className="mobile-progress-track"><i style={{ width: `${((activeIndex + 1) / sections.length) * 100}%` }}/></span><b>{sections[activeIndex][1]}</b><small>{activeIndex + 1}/{sections.length} <span aria-hidden="true">{mobileOpen ? '−' : '+'}</span></small></button>{mobileOpen && <nav className="mobile-section-menu" aria-label="Story sections" role="menu">{sections.map(([id, name], index) => <button key={id} role="menuitem" aria-current={active === id ? 'location' : undefined} onClick={() => goTo(id)}>{name}<span>{String(index + 1).padStart(2, '0')}</span></button>)}</nav>}</div></>
}
function Hero() {
  const openingRate = data.internals.find((entry) => entry.scenario === 'kv_cache_enabled' && entry.gen_len === 16)!.tokens_per_second
  const hardware = data.hardware.gpu.replace('NVIDIA GeForce ', '')
  const model = data.modelComparison.find((entry) => entry.model_name === 'qwen2.5:0.5b')!.model_name
  const [springs, api] = useSpring(() => ({ x: 0, y: 0, config: { mass: 4, tension: 110, friction: 32 } }))
  return <section id="top" className="hero scene-hero" onPointerMove={(event) => { if (event.pointerType === 'touch' || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return; const rect = event.currentTarget.getBoundingClientRect(); api.start({ x: (event.clientX - rect.left - rect.width / 2) * .012, y: (event.clientY - rect.top - rect.height / 2) * .012 }) }} onPointerLeave={() => api.start({ x: 0, y: 0 })}>
    <div className="hero-layout">
      <motion.div className="hero-copy" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .65, delay: .28, ease: [.2, .8, .2, 1] }}>
        <span className="hero-overline">EXPERIMENTS IN MODEL INFERENCE</span>
        <h1 className="hero-title"><span>LLM</span><span>INFERENCE LAB</span></h1>
        <p>Exploring what happens between a prompt and the next token.</p>
        <div className="hero-topics" aria-label="Topics"><span>KV CACHE</span><i/><span>PREFILL</span><i/><span>DECODE</span><i/><span>QUANTIZATION</span><i/><span>GPU SERVING</span></div>
        <a className="hero-link" href="#internals">TRACE THE REQUEST <span>&#8595;</span></a>
      </motion.div>
      <animated.div className="hero-scene" style={{ transform: springs.x.to((x) => 'translate3d(' + x + 'px,' + springs.y.get() + 'px,0)') }}>
        <div className="hero-scene-macro"><HeroMachine decodeRate={openingRate}/></div>
      </animated.div>
    </div>
    <div className="hero-metadata">
      <div><span>CUSTOM BACKEND</span><strong>PyTorch / ONNX Runtime</strong></div>
      <div><span>REAL MODEL BACKEND</span><strong>Ollama / {model}</strong></div>
      <div><span>GPU</span><strong>{hardware}</strong></div>
    </div>
  </section>
}
export default function App() {
  useScrollScenes()
  const reduce = useReducedMotion()
  const [, setDataRevision] = useState(0)
  useEffect(() => {
    let active = true
    void loadLiveBenchmarks(benchmarkData as unknown as BenchmarkBundle).then((bundle) => {
      if (active) { data = bundle; setDataRevision((revision) => revision + 1) }
    })
    return () => { active = false }
  }, [])
  return <><Navigation/><ScrollProgressPill/><div className="handoff-token" aria-hidden="true"><span>tₙ₊₁</span></div><main><Hero/><Pipeline/><KVCache/><Serving/><Comparison/><Concurrency/><ContextLength/><Quantization/><Onnx/></main><div className="motion-mode mono" aria-hidden="true">{reduce ? 'REDUCED MOTION' : 'SCROLL / SIGNAL'}</div></>
}
