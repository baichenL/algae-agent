import { expect, test } from '@playwright/test'

test('评估中心展示标准指标、NOT RUN 边界并可打印截图', async ({ page }, testInfo) => {
  await page.goto('/app/evaluation')
  await expect(page.getByRole('heading', { name: '评估中心' })).toBeVisible()
  await expect(page.getByText('Agent Task Success')).toBeVisible()
  await expect(page.getByText('RAG Recall@20')).toBeVisible()
  await expect(page.getByText('NOT RUN').first()).toBeVisible()
  await expect(page.getByText(/不得描述为真实模型质量/)).toBeVisible()
  await expect(page.getByRole('link', { name: '下载 JSON' })).toBeVisible()

  await page.emulateMedia({ media: 'print' })
  await expect(page.getByRole('button', { name: '打印 / PDF' })).toBeHidden()
  await testInfo.attach(`evaluation-${testInfo.project.name}`, {
    body: await page.screenshot({ fullPage: true }),
    contentType: 'image/png',
  })
})
