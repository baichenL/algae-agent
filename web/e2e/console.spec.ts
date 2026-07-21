import { expect, test } from '@playwright/test'

test('普通导航、调试模式和旧路由兼容', async ({ page }, testInfo) => {
  await page.goto('/app')
  await expect(page.getByRole('heading', { name: '实验室运行总览' })).toBeVisible()
  const compact = testInfo.project.name !== 'desktop-1280'
  if (compact) {
    await page.getByRole('button', { name: '打开导航菜单' }).click()
  }
  await expect(page.getByRole('link', { name: '运行中心' })).not.toBeVisible()
  if (compact) {
    await page.keyboard.press('Escape')
  }
  await expect(page.getByText('simulation_only adapters', { exact: true })).toBeVisible()

  await page.goto('/app/settings')
  await page.getByRole('checkbox', { name: '启用调试模式' }).check()
  if (compact) {
    await page.getByRole('button', { name: '打开导航菜单' }).click()
  }
  await expect(page.getByRole('link', { name: '运行中心' })).toBeVisible()
  await expect(page.getByRole('link', { name: 'Agent 调试' })).toBeVisible()

  await page.goto('/app/observability')
  await expect(page).toHaveURL(/\/app\/debug\/agent$/)
})

test('正式传代审批结果可在 3D 页面实时播放', async ({ page, request }, testInfo) => {
  const session = await request.get('/api/v2/session')
  const csrf = (await session.json()).csrf_token
  const pendingResponse = await request.post('/api/v1/workflow/force_subculture', {
    data: { strain_id: 'Chlorella_01' },
  })
  expect(pendingResponse.ok()).toBeTruthy()
  const pending = await pendingResponse.json()
  const approvalResponse = await request.post(`/api/v2/approvals/${pending.pending_id}/decision`, {
    headers: { 'X-CSRF-Token': csrf },
    data: { decision: 'approved', reason: 'Playwright 3D verification' },
  })
  expect(approvalResponse.ok()).toBeTruthy()
  const approved = await approvalResponse.json()
  await expect.poll(async () => {
    const response = await request.get(`/api/v2/runs/${encodeURIComponent(approved.run_id)}`)
    if (!response.ok()) return 'not_ready'
    return (await response.json()).run.status
  }, { timeout: 30_000 }).toBe('succeeded')

  await page.goto(`/app/runs/${encodeURIComponent(approved.run_id)}/simulation`)
  await expect(page.getByRole('heading', { name: '传代可视化仿真' })).toBeVisible()
  const canvas = page.locator('canvas')
  await expect.poll(async () => {
    try {
      return await canvas.evaluate(element => {
        const canvasElement = element as HTMLCanvasElement
        const rect = canvasElement.getBoundingClientRect()
        const gl = canvasElement.getContext('webgl2') || canvasElement.getContext('webgl')
        if (!gl) return false
        const width = canvasElement.width
        const height = canvasElement.height
        const pixels = new Uint8Array(width * height * 4)
        gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels)
        let nonBackground = 0
        let sampled = 0
        for (let index = 0; index < pixels.length; index += 64) {
          sampled += 1
          const red = pixels[index]
          const green = pixels[index + 1]
          const blue = pixels[index + 2]
          if (Math.max(red, green, blue) - Math.min(red, green, blue) > 8 || red < 225) nonBackground += 1
        }
        return rect.width >= 300 && rect.height >= 400 && sampled > 100 && nonBackground > sampled * 0.02
      })
    } catch {
      return false
    }
  }, { timeout: 30_000, intervals: [250, 500, 1000] }).toBeTruthy()
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
  expect(overflow).toBeLessThanOrEqual(1)
  const screenshotPath = process.env.ALGAE_E2E_SCREENSHOT_DIR
    ? `${process.env.ALGAE_E2E_SCREENSHOT_DIR}/simulation-${testInfo.project.name}.png`
    : undefined
  await testInfo.attach(`simulation-${testInfo.project.name}`, {
    body: await page.screenshot({ fullPage: true, path: screenshotPath }),
    contentType: 'image/png',
  })
})

test('隔离科学场景停在审批边界并在平板提供操作抽屉', async ({ page, request }, testInfo) => {
  test.skip(testInfo.project.name !== 'tablet-768', '平板动作抽屉只需在 768px 项目验证')
  const session = await request.get('/api/v2/session')
  const csrf = (await session.json()).csrf_token
  const created = await request.post('/api/v2/test-scenarios/scientific-two-cycle/runs', {
    headers: { 'X-CSRF-Token': csrf },
    data: { parameters: { use_demo_dataset: true } },
  })
  expect(created.ok()).toBeTruthy()
  const payload = await created.json()
  try {
    const initialResponse = await request.get(`/api/v2/runs/${payload.run_id}`)
    const initialDetail = await initialResponse.json()
    const approvalId = initialDetail.run.approvals.find((item: { status: string }) => item.status === 'pending')?.id
    expect(approvalId).toBeTruthy()
    await page.goto(`/app/runs/${encodeURIComponent(payload.run_id)}`)
    await expect(page.getByText('等待审批', { exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: '批准方案' })).toBeVisible()
    await page.getByRole('button', { name: '批准方案' }).click()
    const actionPanel = page.getByRole('heading', { name: '下一步操作' }).locator('..')
    await expect(actionPanel.getByRole('button', { name: '批准方案' })).toBeVisible()
    page.once('dialog', dialog => dialog.accept())
    await actionPanel.getByRole('button', { name: '批准方案' }).click()
    await expect.poll(async () => {
      const response = await request.get(`/api/v2/runs/${payload.run_id}`)
      const detail = await response.json()
      return detail.run.approvals.find((item: { id: number }) => item.id === approvalId)?.execution_status
    }, { timeout: 30_000 }).toBe('succeeded')
  } finally {
    await request.delete(`/api/v2/test-workspaces/${payload.workspace.id}`, { headers: { 'X-CSRF-Token': csrf } })
  }
})
