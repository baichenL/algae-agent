import type { Approval, EvaluationReport, OperationSummary, RunDetail, RunSummary, SessionInfo, SimulationEvent, UserMemory, UserMemoryCandidate, UserMemorySettings } from './types'

let csrfToken: string | undefined
const executionKeys = new Map<string, string>()

export class ApiError extends Error {
  status: number
  detail: unknown
  constructor(status: number, detail: unknown) {
    super(typeof detail === 'string' ? detail : `Request failed (${status})`)
    this.status = status
    this.detail = detail
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  if (init.body && !(init.body instanceof FormData) && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json')
  if (csrfToken && init.method && !['GET', 'HEAD'].includes(init.method)) headers.set('X-CSRF-Token', csrfToken)
  const response = await fetch(path, { ...init, headers, credentials: 'include' })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new ApiError(response.status, body.detail ?? body)
  return body as T
}

export const api = {
  async session() {
    const result = await request<SessionInfo>('/api/v2/session')
    csrfToken = result.csrf_token
    return result
  },
  async login(apiKey: string) {
    const result = await request<SessionInfo>('/api/v2/session/login', { method: 'POST', headers: { 'X-API-Key': apiKey }, body: JSON.stringify({ remember: false }) })
    csrfToken = result.csrf_token
    return result
  },
  logout: () => request('/api/v2/session', { method: 'DELETE' }),
  dashboard: () => request<any>('/api/v2/dashboard'),
  latestEvaluation: () => request<EvaluationReport>('/api/v2/evaluations/latest'),
  evaluation: (reportId: string) => request<EvaluationReport>(`/api/v2/evaluations/${encodeURIComponent(reportId)}`),
  assistantHistory: (sessionId = 'react-assistant') => request<{ messages: { role: 'user' | 'assistant'; content: string }[] }>(`/api/v2/assistant/history?session_id=${encodeURIComponent(sessionId)}`),
  conversations: (status = 'active') => request<any>(`/api/v2/assistant/conversations?status=${status}`),
  createConversation: (title?: string) => request<any>('/api/v2/assistant/conversations', { method: 'POST', body: JSON.stringify({ title }) }),
  conversationMessages: (id: string) => request<any>(`/api/v2/assistant/conversations/${encodeURIComponent(id)}/messages`),
  conversationTasks: (id: string, scope: 'active' | 'recent' = 'recent') => request<any>(`/api/v2/assistant/conversations/${encodeURIComponent(id)}/tasks?scope=${scope}`),
  assistantTask: (id: string) => request<any>(`/api/v2/assistant/tasks/${encodeURIComponent(id)}`),
  cancelAssistantTask: (id: string) => request<any>(`/api/v2/assistant/tasks/${encodeURIComponent(id)}/cancel`, { method: 'POST' }),
  sendConversationMessage: (id: string, content: string, clientMessageId: string) => request<any>(`/api/v2/assistant/conversations/${encodeURIComponent(id)}/messages`, { method: 'POST', body: JSON.stringify({ content, client_message_id: clientMessageId }) }),
  updateConversation: (id: string, values: { title?: string; status?: 'active' | 'archived' }) => request<any>(`/api/v2/assistant/conversations/${encodeURIComponent(id)}`, { method: 'PATCH', body: JSON.stringify(values) }),
  runs: (kind = '', status = '') => request<{ runs: RunSummary[] }>(`/api/v2/runs?kind=${encodeURIComponent(kind)}&status=${encodeURIComponent(status)}`),
  run: (id: string) => request<{ run: RunDetail }>(`/api/v2/runs/${encodeURIComponent(id)}`),
  events: (id: string) => request<{ events: SimulationEvent[] }>(`/api/v2/runs/${encodeURIComponent(id)}/events`),
  operations: (scope: 'active' | 'recent' = 'active') => request<{ operations: OperationSummary[] }>(`/api/v2/operations?scope=${scope}`),
  operation: (id: string) => request<any>(`/api/v2/operations/${encodeURIComponent(id)}`),
  retryOperation: (id: string) => request<any>(`/api/v2/operations/${encodeURIComponent(id)}/retry`, { method: 'POST' }),
  createRun: (kind: string, input: Record<string, unknown>) => request<any>('/api/v2/runs', { method: 'POST', body: JSON.stringify({ kind, input }) }),
  approvals: (status = 'all') => request<{ approvals: Approval[] }>(`/api/v2/approvals?status=${encodeURIComponent(status)}`),
  decideApproval: (id: number, decision: 'approved' | 'denied', reason = '', runId?: string) => request<any>(`/api/v2/approvals/${id}/decision`, { method: 'POST', body: JSON.stringify({ decision, reason, run_id: runId }) }),
  retryApproval: (id: number) => request<any>(`/api/v2/approvals/${id}/retry`, { method: 'POST' }),
  executeRun: (id: string) => {
    const idempotencyKey = executionKeys.get(id) || crypto.randomUUID()
    executionKeys.set(id, idempotencyKey)
    return request<any>(`/api/v2/runs/${encodeURIComponent(id)}/execute`, { method: 'POST', body: JSON.stringify({ idempotency_key: idempotencyKey }) })
  },
  resolveManual: (runId: string, taskId: string, resolution: 'completed' | 'failed') => request<any>(`/api/v2/runs/${encodeURIComponent(runId)}/manual-tasks/${encodeURIComponent(taskId)}/resolve`, { method: 'POST', body: JSON.stringify({ resolution }) }),
  manualCommit: (runId: string, completedAt: string, note: string, idempotencyKey: string) => request<any>(`/api/v2/runs/${encodeURIComponent(runId)}/manual-commit`, { method: 'POST', body: JSON.stringify({ completed_at: completedAt, note, idempotency_key: idempotencyKey }) }),
  scenarios: () => request<any>('/api/v2/test-scenarios'),
  startScenario: (id: string, parameters: Record<string, unknown> = {}) => request<any>(`/api/v2/test-scenarios/${id}/runs`, { method: 'POST', body: JSON.stringify({ parameters }) }),
  resources: (type: string) => request<any>(`/api/v2/resources/${type}`),
  requestStrainChange: (operation: 'add' | 'update' | 'delete', data: Record<string, unknown>) => request<any>('/api/v2/resources/strains/requests', { method: 'POST', body: JSON.stringify({ operation, data }) }),
  importDataset: (form: FormData) => request<any>('/api/v2/resources/datasets/import', { method: 'POST', body: form }),
  knowledge: (question: string) => request<any>('/api/v2/knowledge/query', { method: 'POST', body: JSON.stringify({ question, top_k: 5 }) }),
  knowledgeSources: () => request<any>('/api/v2/knowledge/sources'),
  uploadKnowledge: (form: FormData) => request<any>('/api/v2/knowledge/sources', { method: 'POST', body: form }),
  reindexKnowledge: (id: string) => request<any>(`/api/v2/knowledge/sources/${encodeURIComponent(id)}/reindex`, { method: 'POST' }),
  setKnowledgeArchived: (id: string, archived: boolean) => request<any>(`/api/v2/knowledge/sources/${encodeURIComponent(id)}/${archived ? 'archive' : 'restore'}`, { method: 'POST' }),
  memories: (status: 'active' | 'archived' | 'superseded' = 'active') => request<{ memories: UserMemory[] }>(`/api/v2/memories?status=${status}`),
  memoryCandidates: () => request<{ candidates: UserMemoryCandidate[] }>('/api/v2/memories/candidates'),
  memorySettings: () => request<{ settings: UserMemorySettings; predicates: string[] }>('/api/v2/memory-settings'),
  updateMemorySettings: (settings: Pick<UserMemorySettings, 'enabled' | 'auto_write_low_risk'>) => request<{ settings: UserMemorySettings }>('/api/v2/memory-settings', { method: 'PATCH', body: JSON.stringify(settings) }),
  createMemory: (memory_type: 'preference' | 'profile' | 'note', predicate: string, value: unknown) => request<{ memory: UserMemory }>('/api/v2/memories', { method: 'POST', body: JSON.stringify({ memory_type, predicate, value }) }),
  updateMemory: (id: number, value: unknown, expected_revision: number) => request<{ memory: UserMemory }>(`/api/v2/memories/${id}`, { method: 'PATCH', body: JSON.stringify({ value, expected_revision }) }),
  archiveMemory: (id: number) => request<{ memory: UserMemory }>(`/api/v2/memories/${id}/archive`, { method: 'POST' }),
  restoreMemory: (id: number) => request<{ memory: UserMemory }>(`/api/v2/memories/${id}/restore`, { method: 'POST' }),
  deleteMemory: (id: number) => request<{ status: string }>(`/api/v2/memories/${id}`, { method: 'DELETE' }),
  decideMemoryCandidate: (id: number, accept: boolean) => request<any>(`/api/v2/memories/candidates/${id}/${accept ? 'accept' : 'reject'}`, { method: 'POST' }),
  exportMemories: () => request<any>('/api/v2/memories/export'),
}
