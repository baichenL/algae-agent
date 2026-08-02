import { describe, expect, it } from 'vitest'

import { testLabRunPath } from './pages'


describe('Test Lab Run 跳转', () => {
  it.each([
    'simulation:happy-run',
    'simulation:pump-fault-run',
    'simulation:manual-run',
  ])('传代场景 %s 自动进入动画页', runId => {
    expect(testLabRunPath(runId)).toBe(`/runs/${encodeURIComponent(runId)}/simulation`)
  })

  it('科学场景仍进入普通 Run 详情页', () => {
    const runId = 'scientific:two-cycle'
    expect(testLabRunPath(runId)).toBe(`/runs/${encodeURIComponent(runId)}`)
  })
})
