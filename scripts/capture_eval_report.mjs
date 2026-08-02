import { createRequire } from 'node:module'
import { pathToFileURL } from 'node:url'
import path from 'node:path'

const requireFromWeb = createRequire(new URL('../web/package.json', import.meta.url))
const { chromium } = requireFromWeb('@playwright/test')

const args = process.argv.slice(2)
const readArg = (name, fallback) => {
  const index = args.indexOf(name)
  return index >= 0 ? args[index + 1] : fallback
}

const input = readArg('--input', 'http://127.0.0.1:5173/app/evaluation')
const screenshot = path.resolve(readArg('--screenshot', 'artifacts/evaluation-center.png'))
const pdf = path.resolve(readArg('--pdf', 'artifacts/evaluation-center.pdf'))
const target = /^[a-z]+:\/\//i.test(input) ? input : pathToFileURL(path.resolve(input)).href

const browser = await chromium.launch({ headless: true })
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 }, deviceScaleFactor: 1 })
  // The React console intentionally polls operation status, so networkidle is
  // not a valid readiness signal. Wait for DOM/load plus a short chart render.
  await page.goto(target, { waitUntil: 'domcontentloaded' })
  await page.waitForLoadState('load')
  await page.waitForTimeout(2500)
  await page.screenshot({ path: screenshot, fullPage: true })
  await page.pdf({ path: pdf, format: 'A4', printBackground: true, margin: { top: '12mm', right: '10mm', bottom: '12mm', left: '10mm' } })
  process.stdout.write(JSON.stringify({ input: target, screenshot, pdf }, null, 2) + '\n')
} finally {
  await browser.close()
}
