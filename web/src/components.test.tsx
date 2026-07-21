import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ThemeProvider } from '@mui/material'
import { describe, expect, it, vi } from 'vitest'
import { AppShell } from './App'
import { ApprovalCard, EmptyState, NextActionPanel, OperationFeedback, ReplanPanel, RunStepper } from './components'
import { theme } from './theme'
import type { Approval, RunDetail } from './types'

const run: RunDetail = {
  id: 'scientific:run-1', kind: 'scientific', title: '科学闭环', workspace_id: 'shared',
  status: 'waiting_approval', phase: 'approval', progress: 0.72, cycle: 1,
  risk_level: 'medium', simulation_only: true, source_ref: { type: 'scientific_run', id: 'run-1' },
  phases: [
    { key: 'data', label: '数据', state: 'completed' },
    { key: 'quality', label: '质量检查', state: 'completed' },
    { key: 'approval', label: '审批', state: 'active' },
    { key: 'execute', label: '执行', state: 'pending' },
  ],
  current_action: { id: 'approve', label: '批准方案', kind: 'approval', approval_id: 9, required_role: 'approver', requires_confirmation: true },
  available_actions: [
    { id: 'approve', label: '批准方案', kind: 'approval', approval_id: 9, required_role: 'approver', requires_confirmation: true },
    { id: 'deny', label: '拒绝', kind: 'approval', approval_id: 9, required_role: 'approver', requires_confirmation: true },
  ],
  artifacts: [], approvals: [], executions: [], assertions: [],
}

function wrapper(children: React.ReactNode) {
  return <ThemeProvider theme={theme}><MemoryRouter>{children}</MemoryRouter></ThemeProvider>
}

describe('统一流程组件', () => {
  it('显示服务端阶段和当前审批边界', () => {
    render(wrapper(<RunStepper run={run} />))
    expect(screen.getByText('数据')).toBeInTheDocument()
    expect(screen.getByText('审批')).toBeInTheDocument()
    expect(screen.getByText('执行')).toBeInTheDocument()
  })

  it('把唯一下一步交给调用方执行', () => {
    const onAction = vi.fn()
    render(wrapper(<NextActionPanel run={run} busy={false} onAction={onAction} />))
    fireEvent.click(screen.getByRole('button', { name: '批准方案' }))
    expect(onAction).toHaveBeenCalledWith(run.available_actions[0])
    expect(screen.getByText(/所需角色：approver/)).toBeInTheDocument()
  })

  it('审批卡展示风险、hash 并保留 Run 返回入口', () => {
    const approval: Approval = {
      id: 9, run_id: run.id, action_type: 'scientific_experiment_plan', status: 'pending',
      risk_level: 'medium', design_hash: 'abcdef1234567890', approval_version: 1,
      requested_by: 'scientist', summary: '数字孪生方案 · 4 条件',
    }
    render(wrapper(<ApprovalCard approval={approval} onDecision={vi.fn()} />))
    expect(screen.getByText('数字孪生方案 · 4 条件')).toBeInTheDocument()
    expect(screen.getByText('abcdef123456')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: '查看 Run' })).toHaveAttribute('href', '/runs/scientific%3Arun-1')
  })

  it('提供明确空状态', () => {
    render(wrapper(<EmptyState title="暂无运行" description="从 Test Lab 启动一个流程。" />))
    expect(screen.getByText('暂无运行')).toBeInTheDocument()
    expect(screen.getByText(/Test Lab/)).toBeInTheDocument()
  })

  it('向普通用户展示 Replan 原因和前后差异', () => {
    render(wrapper(<ReplanPanel replans={[{
      id: 'replan-1', kind: 'agent', trigger_observation: {},
      original_plan: { action_name: 'query_due_subculture' },
      reason: '发现一个到期品系，需要改为审批流程',
      revised_plan: { action_name: 'workflow_subculture' },
      key_changes: ['目标锁定为 Chlorella_01'], from_version: 1, to_version: 2,
      final_action: 'workflow_subculture', safety_boundary: 'approval_required',
    }]} />))
    expect(screen.getByText('已重新规划 1 次')).toBeInTheDocument()
    expect(screen.getByText('计划 v1 → v2')).toBeInTheDocument()
    expect(screen.getByText('目标锁定为 Chlorella_01')).toBeInTheDocument()
  })

  it('后台任务显示当前阶段并提示可离开页面', () => {
    render(wrapper(<OperationFeedback operation={{
      id: 'op-1', kind: 'assistant_message', label: '生成 Assistant 回复', status: 'running',
      phase: 'understanding', message: '正在理解问题', retryable: false,
      created_at: '2026-07-17 12:00:00', updated_at: new Date().toISOString(),
    }} />))
    expect(screen.getByText('正在理解问题')).toBeInTheDocument()
    expect(screen.getByText('运行中')).toBeInTheDocument()
  })
})

describe('演示与调试模式', () => {
  it('切换模式并只在内存状态之外保存非敏感显示偏好', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    localStorage.removeItem('algae-ui-mode')
    render(<QueryClientProvider client={client}><ThemeProvider theme={theme}><MemoryRouter initialEntries={['/settings']}><AppShell session={{ workspace: { id: 'shared', name: '共享实验室', mode: 'simulation_only' }, principal: { name: 'tester', role: 'approver' } }} /></MemoryRouter></ThemeProvider></QueryClientProvider>)
    fireEvent.click(screen.getByRole('checkbox'))
    expect(screen.getByText('启用调试模式')).toBeInTheDocument()
    expect(localStorage.getItem('algae-ui-mode')).toBe('debug')
  })
})
