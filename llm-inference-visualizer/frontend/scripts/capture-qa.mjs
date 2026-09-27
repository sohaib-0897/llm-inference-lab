import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { chromium } from 'playwright'

const outputDir = path.resolve('docs/screenshots')
await mkdir(outputDir, { recursive: true })
const browser = await chromium.launch({ headless: true, executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe', args: ['--no-sandbox'] })
const errors = [], overflow = [], positions = []
let heroStyles
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 })
await page.emulateMedia({ forcedColors: 'none' })
page.on('pageerror', error => errors.push(error.message))
page.on('console', message => { if (message.type() === 'error') errors.push(message.text()) })
await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' })
await page.waitForTimeout(1000)

async function capture(name, section, offset = 20) {
  if (section === 'top') await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'instant' }))
  else {
    const top = await page.locator(`#${section}`).evaluate(el => el.getBoundingClientRect().top + window.scrollY)
    await page.evaluate(({ y, delta }) => window.scrollTo({ top: y + delta, behavior: 'instant' }), { y: top, delta: offset })
  }
  await page.waitForTimeout(700)
  if (section === 'top') heroStyles = await page.evaluate(() => ({ background: getComputedStyle(document.querySelector('.hero')).backgroundColor, color: getComputedStyle(document.querySelector('.hero')).color, opacity: getComputedStyle(document.querySelector('.hero-copy')).opacity, forcedColors: matchMedia('(forced-colors: active)').matches, contrast: matchMedia('(prefers-contrast: more)').matches, atmosphere: getComputedStyle(document.querySelector('.atmosphere-fallback')).backgroundImage }))
  await page.screenshot({ path: path.join(outputDir, `overhaul-${name}-1440.png`), fullPage: false })
  positions.push({ name, y: await page.evaluate(() => Math.round(window.scrollY)) })
  const result = await page.evaluate(() => ({ width: document.documentElement.scrollWidth, viewport: innerWidth }))
  if (result.width > result.viewport + 2) overflow.push(`${name}: ${result.width} > ${result.viewport}`)
}

for (const [name, section, offset] of [
  ['hero', 'top', 0], ['prompt-to-token', 'internals', 160], ['kv-cache', 'kv-cache', 10],
  ['real-model-gpu', 'real-model', 10], ['model-comparison', 'comparison', 10], ['concurrency', 'concurrency', 10],
  ['context-length', 'context-length', 150], ['quantization', 'quantization', 10], ['onnx', 'onnx', 10],
]) await capture(name, section, offset)

for (const viewport of [{ width: 1024, height: 768 }, { width: 390, height: 844 }]) {
  await page.setViewportSize(viewport)
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'instant' }))
  await page.waitForTimeout(400)
  const dimensions = await page.evaluate(() => ({ width: document.documentElement.scrollWidth, viewport: innerWidth, height: document.documentElement.scrollHeight }))
  if (dimensions.width > viewport.width + 2) overflow.push(`${viewport.width}px page: ${dimensions.width} > ${viewport.width}`)
  const ids = await page.locator('main section[id]').evaluateAll(nodes => nodes.map(node => node.id))
  const required = ['internals', 'kv-cache', 'real-model', 'comparison', 'concurrency', 'context-length', 'quantization', 'onnx']
  if (required.some(id => !ids.includes(id))) errors.push(`Missing section at ${viewport.width}px`)
  if (viewport.width === 1024) {
    const y = await page.locator('#kv-cache').evaluate(el => el.getBoundingClientRect().top + window.scrollY)
    await page.evaluate(top => window.scrollTo({ top, behavior: 'instant' }), y)
    await page.waitForTimeout(350)
    await page.screenshot({ path: path.join(outputDir, 'overhaul-kv-cache-1024.png') })
  } else {
    for (const section of ['top', 'internals', 'kv-cache', 'comparison', 'concurrency']) {
      if (section === 'top') await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'instant' }))
      else {
        const y = await page.locator(`#${section}`).evaluate(el => el.getBoundingClientRect().top + window.scrollY)
        await page.evaluate(top => window.scrollTo({ top, behavior: 'instant' }), y)
      }
      await page.waitForTimeout(250)
      await page.screenshot({ path: path.join(outputDir, `overhaul-${section}-390.png`) })
    }
  }
}
await page.setViewportSize({ width: 1440, height: 900 })
await page.getByRole('button', { name: '64', exact: true }).click()
await page.waitForTimeout(400)
if (!(await page.locator('.kv-recompute .metric-value').first().innerText()).includes('330.23')) errors.push(`KV selector did not update the real 64-token result: ${await page.locator('.kv-recompute .metric-value').first().innerText()}`)
await page.screenshot({ path: path.join(outputDir, 'overhaul-kv-cache-64-1440.png') })
await page.getByRole('button', { name: '4', exact: true }).click()
if (!(await page.locator('.p95-dial strong').innerText()).includes('279.07')) errors.push('Concurrency selector did not update the real P95 value')
await page.screenshot({ path: path.join(outputDir, 'overhaul-concurrency-4-1440.png') })
await page.getByRole('button', { name: '2,979', exact: true }).click()
await page.waitForTimeout(400)
if (!(await page.locator('.context-count strong').innerText()).includes('2,979')) errors.push(`Context selector did not update the actual token count: ${await page.locator('.context-count strong').innerText()}`)
await page.screenshot({ path: path.join(outputDir, 'overhaul-context-2979-1440.png') })
const gpuTop = await page.locator('#real-model').evaluate(el => el.getBoundingClientRect().top + window.scrollY)
await page.getByRole('button', { name: 'Go to GPU' }).click()
await page.waitForTimeout(1100)
if (Math.abs((await page.evaluate(() => window.scrollY)) - gpuTop) > 40) errors.push('Navigation rail Lenis scrollTo did not reach the GPU section')
await page.emulateMedia({ reducedMotion: 'reduce' })
await page.reload({ waitUntil: 'networkidle' })
await page.waitForTimeout(250)
const reduced = await page.evaluate(() => ({ smoothClass: document.documentElement.classList.contains('lenis'), tokens: getComputedStyle(document.querySelector('.hero-scene .token-field')).display }))
if (reduced.tokens !== 'none') errors.push('Reduced motion did not suppress decorative token stream')
await writeFile(path.join(outputDir, 'qa-report.json'), `${JSON.stringify({ errors, overflow, positions, reduced, heroStyles }, null, 2)}\n`)
console.log(JSON.stringify({ captures: positions.length, errors, overflow, positions, reduced, heroStyles }, null, 2))
await browser.close()
