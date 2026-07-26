import { expect, test } from '@playwright/test'

test('Markdown, task summary and four structured cards survive every viewport', async ({ page, request }, testInfo) => {
  const session = await request.get('/api/v2/session')
  expect(session.ok()).toBeTruthy()
  const csrf = (await session.json()).csrf_token
  const seeded = await request.post('/api/v2/test-scenarios/assistant-presentation', {
    headers: { 'X-CSRF-Token': csrf },
  })
  expect(seeded.ok()).toBeTruthy()
  const scenario = await seeded.json()

  await page.goto(`/app/assistant/${scenario.conversation_id}`)
  await expect(page.getByRole('heading', { name: '调查结论' })).toBeVisible()
  const markdown = page.locator('.markdown-message')
  await expect(markdown.getByRole('list')).toContainText('已生成两个候选')
  await expect(markdown.getByRole('table')).toContainText('风险')
  await expect(markdown.locator('pre code')).toContainText('simulation_only')

  await expect(page.getByText('当前任务')).toBeVisible()
  await expect(page.getByText(/为 Chlamydomonas_01 准备邮件草稿/)).toBeVisible()
  await expect(page.getByText(/还需要.*recipient/)).toBeVisible()
  await expect(page.getByText('developer_only')).not.toBeVisible()

  await expect(page.getByRole('heading', { name: '邮件草稿', exact: true })).toBeVisible()
  await expect(page.getByText('尚未发送', { exact: true }).last()).toBeVisible()
  await expect(page.getByText(/待审批请求.*900001/)).toBeVisible()
  await expect(page.getByRole('heading', { name: '科学结果', exact: true })).toBeVisible()
  await expect(page.getByText(/2 个候选/)).toBeVisible()
  await expect(page.getByText('任务已暂停')).toBeVisible()

  await page.evaluate(() => localStorage.setItem('algae-ui-mode', 'debug'))
  await page.reload()
  const developerDetails = page.locator('details').filter({ hasText: '开发者详情' })
  await expect(developerDetails).toBeVisible()
  await expect(developerDetails).not.toHaveAttribute('open', '')
  await expect(page.getByText('developer_only')).not.toBeVisible()
  await developerDetails.locator('summary').click()
  await expect(developerDetails).toContainText('developer_only')
  await expect(page.getByRole('heading', { name: '调查结论' })).toBeVisible()

  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  )
  expect(overflow).toBeLessThanOrEqual(1)
  await testInfo.attach(`assistant-presentation-${testInfo.project.name}`, {
    body: await page.screenshot({ fullPage: true }),
    contentType: 'image/png',
  })
})
