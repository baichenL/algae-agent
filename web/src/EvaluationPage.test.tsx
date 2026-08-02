import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { expect, test, vi } from 'vitest'
import { EvaluationPage } from './EvaluationPage'
import { api } from './api'

vi.mock('echarts-for-react/lib/core', () => ({
  default: () => <div data-testid="evaluation-chart" />,
}))

const report = {
  schema_version: '1.0' as const,
  report_id: 'eval-test',
  generated_at: '2026-07-30T00:00:00Z',
  git_commit: '0123456789abcdef',
  dataset_versions: { agent: 'v1', rag: 'interview-rag-v1' },
  environment: {
    embedding_backend: 'fake',
    reranker_backend: 'lexical/fake',
    judge_enabled: false,
  },
  suites: [
    {
      key: 'agent',
      display_name: 'Agent 整体',
      status: 'completed' as const,
      sample_count: 50,
      details: {
        router: { labels: ['chat'], confusion_matrix: [[50]] },
        failure_slices: [{ slice: 'chat', total: 50, failed: 0 }],
      },
      metrics: [
        {
          key: 'task_success_rate',
          display_name: 'Task Success Rate',
          value: 0.9,
          unit: 'ratio' as const,
          denominator: 50,
          ci95: [0.79, 0.96] as [number, number],
          standard: 'τ-bench',
          reference_url: 'https://arxiv.org/abs/2406.12045',
          evidence_mode: 'deterministic' as const,
          status: 'completed' as const,
        },
        {
          key: 'router_macro_f1',
          display_name: 'Router Macro-F1',
          value: 0.88,
          unit: 'ratio' as const,
          denominator: 50,
          standard: 'classification',
          reference_url: 'https://example.test',
          evidence_mode: 'deterministic' as const,
          status: 'completed' as const,
        },
      ],
    },
    {
      key: 'rag',
      display_name: 'RAG Retrieval',
      status: 'not_run' as const,
      status_reason: 'pending human review',
      sample_count: 0,
      details: {},
      metrics: [{
        key: 'recall_at_20',
        display_name: 'Recall@20',
        value: null,
        unit: 'ratio' as const,
        standard: 'BEIR',
        reference_url: 'https://openreview.net/',
        evidence_mode: 'deterministic' as const,
        status: 'not_run' as const,
        status_reason: 'pending human review',
      }],
    },
  ],
  comparisons: [],
}

test('renders traceable metrics, confidence interval, not-run state, and backend warning', async () => {
  vi.spyOn(api, 'latestEvaluation').mockResolvedValue(report)
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(<QueryClientProvider client={client}><EvaluationPage /></QueryClientProvider>)
  expect(await screen.findByText('评估中心')).toBeInTheDocument()
  expect(screen.getByText('90.0%')).toBeInTheDocument()
  expect(screen.getAllByText(/95% CI/).length).toBeGreaterThan(0)
  expect(screen.getAllByText('NOT RUN').length).toBeGreaterThan(0)
  expect(screen.getByText(/不得描述为真实模型质量/)).toBeInTheDocument()
  expect(screen.getByRole('link', { name: '下载 JSON' })).toHaveAttribute(
    'href',
    '/api/v2/evaluations/eval-test/download',
  )
})
