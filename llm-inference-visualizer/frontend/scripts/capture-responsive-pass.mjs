import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { chromium } from 'playwright'

const mode = process.argv[2]
if (!['before', 'after'].includes(mode)) throw new Error('Pass before or after')
const output = path.resolve('docs/screenshots/responsive-pass', mode)
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true, executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe' })
const page = await browser.newPage()
const errors = []
page.on('pageerror', (error) => errors.push(error.message))

async function ready(width, height) {
  await page.setViewportSize({ width, height })
  await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' })
  await page.waitForTimeout(1000)
}

async function save(name) {
  await page.screenshot({ path: path.join(output, `${name}.png`) })
}

for (const [width, height] of [[1440, 900], [1366, 768], [1024, 768]]) {
  await ready(width, height)
  await save(`desktop-${width}x${height}`)
  if (width === 1440) {
    await page.locator('#kv-cache').evaluate((element) => window.scrollTo(0, element.getBoundingClientRect().top + window.scrollY))
    await page.waitForTimeout(700)
    await save('desktop-kv-cache-1440x900')
  }
}

const sections = [
  ['hero', 'top'], ['prompt', 'internals'], ['kv', 'kv-cache'], ['gpu', 'real-model'],
  ['models', 'comparison'], ['concurrency', 'concurrency'], ['context', 'context-length'],
  ['quantization', 'quantization'], ['onnx', 'onnx'],
]
const responsive = []
for (const [width, height] of [[390, 844], [430, 932], [768, 1024]]) {
  await ready(width, height)
  await save(`mobile-${width}x${height}-hero`)
  const state = await page.evaluate(() => ({ viewport: innerWidth, document: document.documentElement.scrollWidth }))
  responsive.push({ width, height, ...state })
  await page.locator('#internals').evaluate((element) => window.scrollTo(0, element.getBoundingClientRect().top + window.scrollY))
  await page.waitForTimeout(900)
  const menu = page.locator('.mobile-progress')
  await menu.click()
  const menuVisible = await page.locator('.mobile-section-menu').isVisible()
  const menuTouchTargets = await page.locator('.mobile-section-menu button').evaluateAll((buttons) => buttons.map((button) => Math.round(button.getBoundingClientRect().height)))
  await page.locator('.mobile-section-menu button').filter({ hasText: 'KV CACHE' }).click()
  await page.waitForTimeout(900)
  const navigatedToKV = !(await page.locator('.mobile-section-menu').isVisible()) && await page.locator('#kv-cache').evaluate((element) => element.getBoundingClientRect().top < innerHeight && element.getBoundingClientRect().bottom > 0)
  const kvButtons = page.locator('#kv-cache .kv-control button')
  const kvTouchTarget = await kvButtons.first().evaluate((button) => Math.round(button.getBoundingClientRect().height))
  await kvButtons.filter({ hasText: '64' }).click()
  const kvResponded = await kvButtons.filter({ hasText: '64' }).evaluate((button) => button.classList.contains('chosen'))
  await kvButtons.filter({ hasText: '32' }).click()
  await page.locator('#concurrency').evaluate((element) => window.scrollTo(0, element.getBoundingClientRect().top + window.scrollY))
  await page.waitForTimeout(900)
  const concurrencyButtons = page.locator('#concurrency .concurrency-switch button')
  const concurrencyTouchTarget = await concurrencyButtons.first().evaluate((button) => Math.round(button.getBoundingClientRect().height))
  await concurrencyButtons.filter({ hasText: '4' }).click()
  const concurrencyResponded = await concurrencyButtons.filter({ hasText: '4' }).evaluate((button) => button.classList.contains('chosen'))
  await concurrencyButtons.filter({ hasText: '1' }).click()
  responsive.push({ width, menuVisible, minMenuTouchTarget: Math.min(...menuTouchTargets), navigatedToKV, kvTouchTarget, kvControlResponded: kvResponded, concurrencyTouchTarget, concurrencyControlResponded: concurrencyResponded })
  if (width === 390) {
    for (const [name, id] of sections.slice(1)) {
      const section = page.locator(`#${id}`)
      await section.evaluate((element) => window.scrollTo(0, element.getBoundingClientRect().top + window.scrollY))
      await page.waitForTimeout(450)
      await save(`mobile-390x844-${name}`)
      const details = await section.evaluate((element) => ({
        id: element.id,
        top: Math.round(element.getBoundingClientRect().top),
        width: Math.round(element.getBoundingClientRect().width),
        scrollWidth: document.documentElement.scrollWidth,
        heading: element.querySelector('h2')?.textContent,
      }))
      responsive.push({ width, height, ...details })
    }
  }
}
await writeFile(path.join(output, 'qa-report.json'), JSON.stringify({ mode, responsive, errors }, null, 2))
await browser.close()
console.log(JSON.stringify({ mode, responsive, errors }, null, 2))
