import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { ThemeProvider } from '@mui/material'
import { expect, it } from 'vitest'
import { AnswerEnvelopeCard, DeveloperDetails, MarkdownMessage } from './components'
import { theme } from './theme'

function wrapper(children: React.ReactNode) {
  return <ThemeProvider theme={theme}><MemoryRouter>{children}</MemoryRouter></ThemeProvider>
}

it('renders safe GFM markdown instead of plain text or raw HTML', () => {
  const { container } = render(wrapper(<MarkdownMessage>{[
    '# 调查结论',
    '',
    '- 候选 A',
    '- 候选 B',
    '',
    '| 方案 | 风险 |',
    '| --- | --- |',
    '| A | 低 |',
    '',
    '```python',
    'print("safe")',
    '```',
    '',
    '<script>window.__unsafe = true</script>',
  ].join('\n')}</MarkdownMessage>))

  expect(screen.getByRole('heading', { name: '调查结论' })).toBeInTheDocument()
  expect(screen.getByRole('list')).toBeInTheDocument()
  expect(screen.getByRole('table')).toBeInTheDocument()
  expect(container.querySelector('pre code')).toHaveTextContent('print("safe")')
  expect(container.querySelector('script')).not.toBeInTheDocument()
  expect(container).not.toHaveTextContent('window.__unsafe')
})

it('keeps complete task JSON in collapsed developer details', () => {
  render(wrapper(<DeveloperDetails value={{
    goal_text: '调查 Chlamydomonas_01',
    status: 'running',
    secret_internal_field: 'must-not-replace-summary',
  }} />))

  const details = screen.getByText('开发者详情').closest('details')
  expect(details).not.toHaveAttribute('open')
  expect(screen.getByText(/secret_internal_field/)).not.toBeVisible()
  fireEvent.click(screen.getByText('开发者详情'))
  expect(screen.getByText(/secret_internal_field/)).toBeVisible()
})

it('renders all answer-envelope v2 presentation cards as non-executable', () => {
  render(wrapper(<AnswerEnvelopeCard envelope={{
    schema_version: 'answer-envelope/v2',
    outcome_status: 'paused',
    direct_answer: '主回答',
    confirmed_facts: [],
    inferences: [],
    simulation_results: [],
    recommendations: [],
    unknowns: [],
    source_statuses: [],
    completed_work: {},
    remaining_work: ['比较候选'],
    budget: {},
    references: [],
    next_actions: [],
    presentation_blocks: [
      { type: 'email_draft', draft: { subject: '检查提醒', body: '请检查培养物', recipients: ['test@example.invalid'] }, target: 'Chlamydomonas_01', send: false },
      { type: 'approval', pending_id: 19, status: 'pending', executable: false, requires_human_approval: true },
      { type: 'scientific_result', candidate_count: 2, candidates: [{ candidate_id: 'A' }, { candidate_id: 'B' }], validations: [], simulations: [], plan_patches: [{ patch_id: 'P1' }], comparison: {} },
      { type: 'paused_task', task_id: 'task-1', reason: 'budget', completed_work: {}, remaining_work: ['比较候选'], budget: {} },
    ],
  }} />))

  expect(screen.getByText('检查提醒')).toBeInTheDocument()
  expect(screen.getByText(/尚未发送/)).toBeInTheDocument()
  expect(screen.getByText(/19/)).toBeInTheDocument()
  expect(screen.getByText(/候选.*2/)).toBeInTheDocument()
  expect(screen.getByText(/任务已暂停/)).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: /执行|发送|批准/ })).not.toBeInTheDocument()
})
