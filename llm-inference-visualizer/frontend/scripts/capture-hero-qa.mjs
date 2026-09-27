import { mkdir } from 'node:fs/promises'
import path from 'node:path'
import { chromium } from 'playwright'

const output = path.resolve('docs/screenshots/hero-redesign')
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true, executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe', args: ['--no-sandbox'] })
const errors = [], overflow = []
async function open(viewport, reduced = false) {
  const page = await browser.newPage({ viewport, deviceScaleFactor: 1 })
  page.on('pageerror', error => errors.push(error.message))
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()) })
  if (reduced) await page.emulateMedia({ reducedMotion: 'reduce' })
  await page.goto('http://localhost:5176/', { waitUntil: 'networkidle' })
  await page.waitForTimeout(1750)
  return page
}
async function audit(page, width) {
  return page.evaluate(() => {
    const title = document.querySelector('.hero-title').getBoundingClientRect()
    const copy = document.querySelector('.hero-copy').getBoundingClientRect()
    const machine = document.querySelector('.hero-machine svg').getBoundingClientRect()
    const next = document.querySelector('#internals').getBoundingClientRect()
    return { pageWidth: document.documentElement.scrollWidth, viewport: innerWidth, title: title.toJSON(), copy: copy.toJSON(), machine: machine.toJSON(), nextSectionTop: next.top, headline: document.querySelector('.hero-title').innerText, copyOpacity: getComputedStyle(document.querySelector('.hero-copy')).opacity, progress: document.querySelector('.scroll-pill').getAttribute('aria-valuenow') }
  }).then(result => {
    if (result.pageWidth > width + 2) overflow.push(`${width}px: page width ${result.pageWidth}`)
    if (!result.headline.includes('LLM') || Number(result.copyOpacity) < .98) errors.push(`${width}px: hero headline or copy not visible at load`)
    if (result.nextSectionTop < 0) errors.push(`${width}px: prompt section overlaps the hero at load`)
    return result
  })
}

const desktop = await open({ width: 1440, height: 900 })
const desktopAudit = await audit(desktop, 1440)
await desktop.screenshot({ path: path.join(output, 'hero-1440.png') })
await desktop.evaluate(() => window.scrollTo({ top: 450, behavior: 'instant' }))
await desktop.waitForTimeout(900)
const midAudit = await desktop.evaluate(() => ({
  scrollY: Math.round(scrollY),
  titleOpacity: getComputedStyle(document.querySelector('.hero-copy')).opacity,
  sceneTransform: getComputedStyle(document.querySelector('.hero-scene-macro')).transform,
  pill: document.querySelector('.scroll-pill').getAttribute('aria-valuenow'),
}))
await desktop.screenshot({ path: path.join(output, 'hero-1440-mid-scroll.png') })
if (Number(midAudit.titleOpacity) < .65) errors.push('Hero copy fades too early at 50% hero scroll')
if (Number(midAudit.pill) < 1) errors.push('Pill progress did not update during the hero scroll')
await desktop.getByRole('link', { name: /TRACE THE REQUEST/ }).click()
await desktop.waitForTimeout(1500)
const handoff = await desktop.locator('#internals').evaluate(el => ({ top: el.getBoundingClientRect().top, scrollY: Math.round(scrollY) }))
if (Math.abs(handoff.top) > 24) errors.push('Hero CTA did not hand off to the prompt section')
await desktop.close()

for (const viewport of [{ width: 1024, height: 768 }, { width: 390, height: 844 }]) {
  const page = await open(viewport)
  const state = await audit(page, viewport.width)
  await page.screenshot({ path: path.join(output, `hero-${viewport.width}.png`) })
  if (state.machine.width <= 0 || state.machine.height <= 0) errors.push(`${viewport.width}px: inference visualization missing`)
  await page.close()
}

const reduced = await open({ width: 1440, height: 900 }, true)
const reducedAudit = await reduced.evaluate(() => ({
  titleOpacity: getComputedStyle(document.querySelector('.hero-copy')).opacity,
  progress: document.querySelector('.scroll-pill').getAttribute('aria-valuenow'),
  headline: document.querySelector('.hero-title').innerText,
}))
if (Number(reducedAudit.titleOpacity) < .98 || !reducedAudit.headline.includes('INFERENCE LAB')) errors.push('Reduced motion did not preserve the static hero')
await reduced.close()
await browser.close()
console.log(JSON.stringify({ errors, overflow, desktopAudit, midAudit, reducedAudit }, null, 2))
if (errors.length || overflow.length) process.exitCode = 1
