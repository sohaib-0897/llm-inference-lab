import { mkdir } from 'node:fs/promises'
import path from 'node:path'
import { chromium } from 'playwright'

const outputDir = path.resolve('docs/screenshots/qa')
await mkdir(outputDir, { recursive: true })
const browser = await chromium.launch({ headless: true, executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe', args: ['--no-sandbox'] })
const errors = [], overflow = []
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 })
page.on('pageerror', error => errors.push(error.message))
page.on('console', message => { if (message.type() === 'error') errors.push(message.text()) })
await page.goto('http://localhost:5176/', { waitUntil: 'networkidle' })
await page.waitForTimeout(700)
const desktopWidth = await page.evaluate(() => ({ width: document.documentElement.scrollWidth, viewport: innerWidth }))
if (desktopWidth.width > desktopWidth.viewport + 2) overflow.push(`1440: document width ${desktopWidth.width}`)

async function goTo(top) {
  await page.evaluate(y => window.scrollTo({ top: y, behavior: 'instant' }), top)
  await page.waitForTimeout(650)
}
async function shot(name) { await page.screenshot({ path: path.join(outputDir, `${name}.png`), fullPage: false }) }
async function sectionTop(id) { return page.locator(`#${id}`).evaluate(el => el.getBoundingClientRect().top + window.scrollY) }

await goTo(0); await shot('01-hero-initial')
await goTo(330); await shot('02-hero-mid-scroll')
const promptTop = await sectionTop('internals')
await goTo(promptTop + 790); await shot('03-prompt-active')
await page.getByRole('button', { name: 'Show KV CACHE stage' }).click().catch(() => {})
await page.waitForTimeout(900); await shot('03-prompt-active')
const contextTop = await sectionTop('context-length')
await goTo(contextTop + 20); await shot('04-context-overview')
await goTo(contextTop + 220); await shot('05-context-metrics')
await goTo(0); await shot('06-scroll-pill')

for (const viewport of [{ width: 1024, height: 768 }, { width: 390, height: 844 }]) {
  const responsive = await browser.newPage({ viewport, deviceScaleFactor: 1 })
  responsive.on('pageerror', error => errors.push(error.message))
  responsive.on('console', message => { if (message.type() === 'error') errors.push(message.text()) })
  await responsive.goto('http://localhost:5176/', { waitUntil: 'networkidle' })
  await responsive.waitForTimeout(700)
  const state = await responsive.evaluate(() => ({ width: document.documentElement.scrollWidth, viewport: innerWidth }))
  if (state.width > viewport.width + 2) overflow.push(`${viewport.width}: document width ${state.width}`)
  const hero = await responsive.locator('.hero-copy').evaluate(el => ({ opacity: Number(getComputedStyle(el).opacity), top: el.getBoundingClientRect().top, bottom: el.getBoundingClientRect().bottom }))
  if (hero.opacity < .98 || hero.top < 0 || hero.bottom > viewport.height) errors.push(`Hero copy is not fully visible at ${viewport.width}px`)
  await responsive.screenshot({ path: path.join(outputDir, `${viewport.width === 1024 ? '07-laptop-hero' : '08-mobile-hero'}.png`) })
  const context = await responsive.locator('#context-length').evaluate(el => el.getBoundingClientRect().top + window.scrollY)
  await responsive.evaluate(y => window.scrollTo({ top: y + 20, behavior: 'instant' }), context)
  await responsive.waitForTimeout(700)
  const contextFlow = await responsive.evaluate(() => {
    const box = selector => document.querySelector(selector).getBoundingClientRect()
    return { count: box('.context-count'), tape: box('.context-tape-wrap'), metrics: box('.context-readout-strip') }
  })
  if (contextFlow.count.bottom > contextFlow.tape.top + 2 || contextFlow.tape.bottom > contextFlow.metrics.top + 2) errors.push(`Context metric and visualization regions overlap at ${viewport.width}px`)
  await responsive.screenshot({ path: path.join(outputDir, `${viewport.width === 1024 ? '09-laptop-context' : '10-mobile-context'}.png`) })
  await responsive.close()
}

await page.setViewportSize({ width: 1440, height: 900 })
await goTo(0)
const heroText = await page.locator('.hero-copy').evaluate(el => ({ opacity: getComputedStyle(el).opacity, rect: el.getBoundingClientRect().toJSON(), text: el.innerText }))
if (!heroText.text.includes('Exploring what happens between a prompt and the next token.')) errors.push('Hero subtitle missing or changed')
await page.getByRole('button', { name: '2,979', exact: true }).click()
await page.waitForTimeout(250)
if (!(await page.locator('.context-count strong').innerText()).includes('2,979')) errors.push('Context selector failed')
await page.getByRole('button', { name: 'Show OUTPUT stage' }).click()
await page.waitForTimeout(700)
if (!(await page.locator('.pipeline-step-label').innerText()).includes('OUTPUT')) errors.push('Pipeline stage control failed')
await page.getByRole('button', { name: 'Go to GPU' }).click()
await page.waitForTimeout(1200)
if (Number(await page.locator('.scroll-pill').getAttribute('aria-valuenow')) < 10) errors.push('Scroll pill did not track Lenis navigation')
await page.emulateMedia({ reducedMotion: 'reduce' })
await page.reload({ waitUntil: 'networkidle' })
await page.waitForTimeout(350)
const reduced = await page.evaluate(() => ({ lenis: document.documentElement.classList.contains('lenis'), tokens: getComputedStyle(document.querySelector('.hero-scene .token-field')).display, pins: document.querySelectorAll('.pin-spacer').length }))
if (reduced.tokens !== 'none') errors.push('Reduced-motion token loop remains visible')
if (reduced.pins) errors.push('Reduced-motion mode created pinned scenes')
console.log(JSON.stringify({ errors, overflow, reduced, heroText }, null, 2))
await browser.close()
if (errors.length || overflow.length) process.exitCode = 1
