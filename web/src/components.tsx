import { useMemo } from 'react'
import { Link as RouterLink } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeSanitize from 'rehype-sanitize'
import ReactEChartsCore from 'echarts-for-react/lib/core'
import * as echarts from 'echarts/core'
import { LineChart } from 'echarts/charts'
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { Background, Controls, ReactFlow, type Edge, type Node } from '@xyflow/react'
import {
  Alert, Box, Button, Card, CardContent, Chip, CircularProgress, LinearProgress, Link, List, ListItem,
  ListItemText, Stack, Step, StepLabel, Stepper, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, Typography,
} from '@mui/material'
import ArrowForwardRounded from '@mui/icons-material/ArrowForwardRounded'
import CheckCircleRounded from '@mui/icons-material/CheckCircleRounded'
import ErrorRounded from '@mui/icons-material/ErrorRounded'
import PendingRounded from '@mui/icons-material/PendingRounded'
import ScienceRounded from '@mui/icons-material/ScienceRounded'
import AutoFixHighRounded from '@mui/icons-material/AutoFixHighRounded'
import type { ActionDescriptor, AnswerEnvelope, Approval, Artifact, OperationSummary, ReplanSummary, RunDetail, RunSummary } from './types'

echarts.use([LineChart, GridComponent, LegendComponent, TooltipComponent, CanvasRenderer])

const statusMeta: Record<string, { label: string; color: 'default' | 'success' | 'warning' | 'error' | 'info' }> = {
  paused: { label: '已暂停', color: 'warning' },
  partial: { label: '部分完成', color: 'warning' },
  needs_input: { label: '需要输入', color: 'info' },
  success: { label: '已完成', color: 'success' },
  queued: { label: '排队中', color: 'info' }, running: { label: '运行中', color: 'info' },
  waiting_input: { label: '等待人工', color: 'warning' }, waiting_approval: { label: '等待审批', color: 'warning' },
  ready_to_execute: { label: '可执行', color: 'success' }, succeeded: { label: '已完成', color: 'success' },
  failed: { label: '失败', color: 'error' }, blocked: { label: '已阻断', color: 'error' },
  cancelled: { label: '已取消', color: 'default' }, pending: { label: '待审批', color: 'warning' },
  approved: { label: '已批准', color: 'success' }, denied: { label: '已拒绝', color: 'error' },
}

export function StatusChip({ status, size = 'small' }: { status: string; size?: 'small' | 'medium' }) {
  const meta = statusMeta[status] || { label: status || '未知', color: 'default' as const }
  return <Chip size={size} color={meta.color} label={meta.label} variant={meta.color === 'default' ? 'outlined' : 'filled'} />
}

export function MarkdownMessage({ children }: { children: string }) {
  return <Box
    className="markdown-message"
    sx={{
      overflowWrap: 'anywhere',
      '& > :first-of-type': { mt: 0 },
      '& > :last-child': { mb: 0 },
      '& table': { width: '100%', borderCollapse: 'collapse', my: 1 },
      '& th, & td': { border: '1px solid', borderColor: 'divider', px: 1, py: 0.75, textAlign: 'left' },
      '& pre': { overflowX: 'auto', p: 1.25, borderRadius: 1, bgcolor: '#f5f7f7' },
      '& code': { fontFamily: 'ui-monospace, SFMono-Regular, Consolas, monospace' },
      '& a': { color: 'primary.main' },
    }}
  >
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeSanitize]}
      components={{
        a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" />,
      }}
    >
      {children}
    </ReactMarkdown>
  </Box>
}

export function DeveloperDetails({ value }: { value: unknown }) {
  return <Box component="details" sx={{ mt: 1 }}>
    <Box component="summary" sx={{ cursor: 'pointer', color: 'text.secondary', fontSize: 13, fontWeight: 700 }}>
      开发者详情
    </Box>
    <Box mt={1}><DebugJson value={value} /></Box>
  </Box>
}

function compactSummary(value: unknown): string {
  if (value == null) return '—'
  if (typeof value !== 'object') return String(value)
  const item = value as Record<string, any>
  return String(item.title || item.name || item.candidate_id || item.id || item.status || item.conclusion || '结构化结果')
}

export function AnswerEnvelopeCard({
  envelope,
  draft,
  onResume,
}: {
  envelope?: AnswerEnvelope
  draft?: Record<string, any>
  onResume?: () => void
}) {
  if (!envelope) return null
  const limitations = envelope.source_statuses.filter(item => ['failed', 'degraded', 'empty'].includes(item.status))
  const blocks = envelope.presentation_blocks || (draft ? [{ type: 'email_draft', draft, send: false as const }] : [])
  return <Stack spacing={1.25} mt={1.5}>
    <Stack direction="row" spacing={1} alignItems="center">
      <StatusChip status={envelope.outcome_status} />
      <Typography variant="caption" color="text.secondary">结构化结果</Typography>
    </Stack>
    {blocks.map((block: any, index) => {
      if (block.type === 'email_draft') {
        const item = block.draft || {}
        return <Card key={`email-${index}`} variant="outlined"><CardContent sx={{ py: 1.5 }}>
          <Typography variant="subtitle2">邮件草稿</Typography>
          {block.target && <Typography variant="caption" display="block">目标：{block.target}</Typography>}
          <Typography variant="caption" display="block">收件人：{(item.recipients || []).join(', ') || '未设置'}</Typography>
          <Typography fontWeight={700} mt={0.5}>{item.subject}</Typography>
          <Typography variant="body2" whiteSpace="pre-wrap" mt={0.5}>{item.body}</Typography>
          <Chip size="small" color="info" variant="outlined" label="尚未发送" sx={{ mt: 1 }} />
        </CardContent></Card>
      }
      if (block.type === 'approval') return <Alert key={`approval-${index}`} severity="warning" action={<Button component={RouterLink} to="/approvals" size="small">查看审批</Button>}>
        待审批请求 #{block.pending_id} 已创建；必须由人工审批，当前不可执行。
      </Alert>
      if (block.type === 'scientific_result') return <Card key={`science-${index}`} variant="outlined"><CardContent sx={{ py: 1.5 }}>
        <Stack direction="row" spacing={1} alignItems="center"><ScienceRounded color="primary" /><Typography variant="subtitle2">科学结果</Typography><Chip size="small" label={`${block.candidate_count || 0} 个候选`} /></Stack>
        {!!block.candidates?.length && <List dense>{block.candidates.map((item: unknown, itemIndex: number) => <ListItem key={itemIndex} disableGutters><ListItemText primary={compactSummary(item)} secondary={`候选 ${itemIndex + 1}`} /></ListItem>)}</List>}
        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap mt={1}>
          <Chip size="small" variant="outlined" label={`验证 ${block.validations?.length || 0}`} />
          <Chip size="small" variant="outlined" label={`仿真 ${block.simulations?.length || 0}`} />
          <Chip size="small" variant="outlined" label={`PlanPatch ${block.plan_patches?.length || 0}`} />
        </Stack>
      </CardContent></Card>
      if (block.type === 'paused_task') return <Alert key={`paused-${index}`} severity="warning">
        <Typography fontWeight={800}>任务已暂停</Typography>
        <Typography variant="body2">{block.reason || '已保存 checkpoint，可稍后继续。'}</Typography>
        {!!block.remaining_work?.length && <Typography variant="caption">剩余工作：{block.remaining_work.map(compactSummary).join('；')}</Typography>}
      </Alert>
      return null
    })}
    {!!limitations.length && <Alert severity={envelope.outcome_status === 'failed' ? 'error' : 'warning'}>
      {limitations.map(item => `${item.source}: ${item.status}${item.impact ? `（${item.impact}）` : ''}`).join('；')}
    </Alert>}
    {!!envelope.unknowns.length && <Box>
      <Typography variant="caption" fontWeight={800}>限制与未知</Typography>
      {envelope.unknowns.map((item, index) => <Typography key={index} variant="body2">• {item}</Typography>)}
    </Box>}
    <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
      {envelope.references.map((ref, index) => {
        if (ref.type === 'pending') return <Button key={index} component={RouterLink} to="/approvals" size="small">查看审批 #{ref.id}</Button>
        if (ref.type === 'scientific_run') return <Button key={index} component={RouterLink} to={`/runs/${encodeURIComponent(`scientific:${ref.id}`)}`} size="small">Scientific Run</Button>
        if (ref.type === 'agent_trace') return <Button key={index} component={RouterLink} to={`/runs/${encodeURIComponent(`agent:${ref.id}`)}`} size="small">Agent Trace</Button>
        return null
      })}
      {envelope.outcome_status === 'paused' && onResume && <Button size="small" variant="contained" onClick={onResume}>继续任务</Button>}
    </Stack>
  </Stack>
}

export function formatDate(value?: string) {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value.slice(0, 19) : date.toLocaleString('zh-CN', { hour12: false })
}

export function AsyncActionButton({ busy, busyLabel, children, ...props }: React.ComponentProps<typeof Button> & { busy?: boolean; busyLabel: string }) {
  return <Button {...props} disabled={busy || props.disabled} startIcon={busy ? <CircularProgress color="inherit" size={16} /> : props.startIcon}>
    {busy ? busyLabel : children}
  </Button>
}

export function OperationFeedback({ operation, onRetry }: { operation: OperationSummary; onRetry?: () => void }) {
  const active = ['queued', 'running', 'waiting_input'].includes(operation.status)
  const ageSeconds = Math.max(0, (Date.now() - new Date(operation.heartbeat_at || operation.updated_at).getTime()) / 1000)
  const severity = operation.status === 'failed' ? 'error' : operation.status === 'succeeded' ? 'success' : ageSeconds > 60 ? 'warning' : 'info'
  const staleText = active && ageSeconds > 60 ? '耗时较长，任务仍在后台继续。' : active && ageSeconds > 15 ? '仍在处理，可以离开页面。' : null
  return <Alert severity={severity} sx={{ alignItems: 'flex-start' }} action={operation.status === 'failed' && operation.retryable && onRetry ? <Button color="inherit" size="small" onClick={onRetry}>重试</Button> : undefined}>
    <Stack spacing={0.75} sx={{ minWidth: 0 }}>
      <Stack direction="row" spacing={1} alignItems="center"><Typography fontWeight={750}>{operation.label}</Typography><StatusChip status={operation.status} /></Stack>
      <Typography variant="body2">{operation.message || operation.phase}{staleText ? ` · ${staleText}` : ''}</Typography>
      {active && (operation.progress == null ? <LinearProgress /> : <LinearProgress variant="determinate" value={Math.round(operation.progress * 100)} />)}
      {operation.error_message && <Typography variant="caption">原因：{operation.error_message}</Typography>}
      <Typography variant="caption" color="text.secondary">阶段 {operation.phase} · 最近更新 {formatDate(operation.updated_at)}</Typography>
    </Stack>
  </Alert>
}

export function ReplanPanel({ replans = [], debugMode = false }: { replans?: ReplanSummary[]; debugMode?: boolean }) {
  if (!replans.length) return null
  return <Stack spacing={1.5}>
    <Stack direction="row" spacing={1} alignItems="center"><AutoFixHighRounded color="warning" /><Typography variant="h2">重新规划</Typography><Chip size="small" color="warning" label={`已重新规划 ${replans.length} 次`} /></Stack>
    {replans.map(item => <Card key={item.id} variant="outlined"><CardContent>
      <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" spacing={1}>
        <Box><Typography fontWeight={800}>计划 v{item.from_version} → v{item.to_version}</Typography><Typography color="text.secondary" mt={0.5}>{item.reason}</Typography></Box>
        {item.occurred_at && <Typography variant="caption" color="text.secondary">{formatDate(item.occurred_at)}</Typography>}
      </Stack>
      <Box display="grid" gridTemplateColumns={{ xs: '1fr', md: '1fr auto 1fr' }} gap={1.5} alignItems="stretch" mt={2}>
        <Box sx={{ p: 1.5, bgcolor: '#f7f8f8', borderRadius: 1 }}><Typography variant="caption" fontWeight={800}>原计划</Typography><Typography variant="body2" mt={0.75}>{String(item.original_plan?.action_name || item.original_plan?.route_kind || item.original_plan?.design_hash || '按原方案继续')}</Typography></Box>
        <ArrowForwardRounded sx={{ alignSelf: 'center', justifySelf: 'center', transform: { xs: 'rotate(90deg)', md: 'none' } }} />
        <Box sx={{ p: 1.5, bgcolor: '#edf8f5', borderRadius: 1 }}><Typography variant="caption" fontWeight={800} color="primary.main">修订后</Typography><Typography variant="body2" mt={0.75}>{String(item.final_action || item.revised_plan?.action_name || item.revised_plan?.design_hash || '采用修订方案')}</Typography></Box>
      </Box>
      <List dense sx={{ mt: 1 }}>{item.key_changes.map(change => <ListItem key={change} disableGutters><ListItemText primary={change} /></ListItem>)}</List>
      {item.safety_boundary && <Alert severity="warning" sx={{ mt: 1 }}>安全边界：{item.safety_boundary}</Alert>}
      {debugMode && item.debug && <Box mt={1.5}><Typography variant="caption" fontWeight={800}>候选、置信度与拒绝原因</Typography><DebugJson value={item.debug} /></Box>}
    </CardContent></Card>)}
  </Stack>
}

export function PageHeader({ eyebrow, title, description, action }: { eyebrow?: string; title: string; description?: string; action?: React.ReactNode }) {
  return <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" alignItems={{ xs: 'flex-start', sm: 'center' }} spacing={2} mb={3}>
    <Box>
      {eyebrow && <Typography className="eyebrow" mb={0.5}>{eyebrow}</Typography>}
      <Typography variant="h1">{title}</Typography>
      {description && <Typography color="text.secondary" mt={0.75}>{description}</Typography>}
    </Box>
    {action}
  </Stack>
}

export function MetricCard({ label, value, hint, tone = 'primary' }: { label: string; value: string | number; hint?: string; tone?: string }) {
  const colors: Record<string, string> = { primary: '#0f766e', warning: '#b45309', error: '#b91c1c', neutral: '#475569' }
  return <Card sx={{ height: '100%' }}><CardContent>
    <Typography color="text.secondary" variant="body2" fontWeight={650}>{label}</Typography>
    <Typography sx={{ fontSize: 30, lineHeight: 1.25, fontWeight: 760, color: colors[tone] || colors.primary, mt: 1 }}>{value}</Typography>
    {hint && <Typography variant="caption" color="text.secondary">{hint}</Typography>}
  </CardContent></Card>
}

export function RunTable({ runs, compact = false }: { runs: RunSummary[]; compact?: boolean }) {
  if (!runs.length) return <EmptyState title="暂无运行" description="从 Assistant、Test Lab 或科学数据集启动一个流程。" />
  return <TableContainer><Table size={compact ? 'small' : 'medium'}>
    <TableHead><TableRow><TableCell>运行</TableCell><TableCell>类型</TableCell><TableCell>状态</TableCell><TableCell>阶段</TableCell>{!compact && <TableCell>更新时间</TableCell>}<TableCell /></TableRow></TableHead>
    <TableBody>{runs.map(run => <TableRow key={run.id} hover>
      <TableCell sx={{ minWidth: compact ? 220 : 260 }}><Typography fontWeight={700}>{run.title}</Typography><Typography variant="caption" color="text.secondary" className="mono">{run.id}</Typography></TableCell>
      <TableCell><Chip size="small" variant="outlined" label={run.kind} /></TableCell>
      <TableCell><StatusChip status={run.status} /></TableCell>
      <TableCell><Stack spacing={0.5} sx={{ minWidth: 130 }}><Typography variant="body2">{run.phase}</Typography><LinearProgress variant="determinate" value={Math.round(run.progress * 100)} sx={{ height: 5, borderRadius: 4 }} /></Stack></TableCell>
      {!compact && <TableCell sx={{ whiteSpace: 'nowrap' }}>{formatDate(run.updated_at || run.created_at)}</TableCell>}
      <TableCell align="right"><Button component={RouterLink} to={`/runs/${encodeURIComponent(run.id)}`} endIcon={<ArrowForwardRounded />} size="small">详情</Button></TableCell>
    </TableRow>)}</TableBody>
  </Table></TableContainer>
}

export function EmptyState({ title, description }: { title: string; description: string }) {
  return <Box sx={{ py: 7, textAlign: 'center', border: '1px dashed', borderColor: 'divider', borderRadius: 2, bgcolor: '#fbfcfc' }}>
    <ScienceRounded sx={{ color: 'primary.main', fontSize: 36, mb: 1 }} />
    <Typography variant="h3">{title}</Typography><Typography color="text.secondary" mt={0.75}>{description}</Typography>
  </Box>
}

export function RunStepper({ run }: { run: RunDetail }) {
  const active = Math.max(0, run.phases.findIndex(item => item.state === 'active' || item.state === 'blocked'))
  return <Card><CardContent sx={{ py: 2.5, overflowX: 'auto' }}><Stepper activeStep={active} alternativeLabel sx={{ minWidth: Math.max(760, run.phases.length * 100) }}>
    {run.phases.map(phase => <Step key={phase.key} completed={phase.state === 'completed'}>
      <StepLabel error={phase.state === 'blocked'}>{phase.label}</StepLabel>
    </Step>)}
  </Stepper></CardContent></Card>
}

export function ApprovalCard({ approval, busy = false, onDecision, onRetry }: { approval: Approval; busy?: boolean; onDecision?: (decision: 'approved' | 'denied') => void; onRetry?: () => void }) {
  return <Card><CardContent>
    <Stack direction="row" justifyContent="space-between" alignItems="flex-start">
      <Box><Typography className="eyebrow">Approval {approval.id}</Typography><Typography variant="h2" mt={0.5}>{approval.summary}</Typography></Box>
      <StatusChip status={approval.status} />
    </Stack>
    <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap" my={2}>
      <Chip size="small" label={approval.action_type} variant="outlined" />
      <Chip size="small" label={`risk ${approval.risk_level}`} color={approval.risk_level === 'high' ? 'warning' : 'default'} />
      {approval.design_hash && <Chip size="small" className="mono" label={approval.design_hash.slice(0, 12)} />}
    </Stack>
    <Typography variant="body2" color="text.secondary">请求人 {approval.requested_by || 'system'} · {formatDate(approval.created_at)}</Typography>
    {approval.execution_status && approval.status === 'approved' && <Alert severity={approval.execution_status === 'failed' ? 'error' : approval.execution_status === 'succeeded' ? 'success' : 'info'} sx={{ mt: 1.5 }}>执行状态：{approval.execution_status}{approval.execution_error ? ` · ${approval.execution_error}` : ''}</Alert>}
    <Stack direction="row" spacing={1} mt={2}>
      {approval.run_id && <Button component={RouterLink} to={`/runs/${encodeURIComponent(approval.run_id)}`}>查看 Run</Button>}
      {approval.status === 'pending' && onDecision && <>
        <AsyncActionButton variant="contained" busy={busy} busyLabel="正在批准并启动" onClick={() => onDecision('approved')}>批准方案</AsyncActionButton>
        <AsyncActionButton color="error" variant="outlined" busy={busy} busyLabel="正在提交" onClick={() => onDecision('denied')}>拒绝</AsyncActionButton>
      </>}
      {approval.status === 'approved' && approval.execution_status === 'failed' && onRetry && <AsyncActionButton variant="contained" busy={busy} busyLabel="正在重试" onClick={onRetry}>重试执行</AsyncActionButton>}
    </Stack>
  </CardContent></Card>
}

function artifact(run: RunDetail, type: string): Artifact | undefined {
  return [...run.artifacts].reverse().find(item => item.artifact_type === type)
}

function measurementSeries(run: RunDetail) {
  const execution = run.executions?.[run.executions.length - 1] || {}
  const points = execution.measurements || execution.measurements_json || []
  const grouped = new Map<string, Array<[number, number]>>()
  for (const point of points.slice(0, 500)) {
    const name = String(point.batch_id || point.condition_id || point.role || 'virtual result')
    if (!grouped.has(name)) grouped.set(name, [])
    grouped.get(name)!.push([Number(point.elapsed_hours ?? point.time ?? grouped.get(name)!.length), Number(point.value ?? point.biomass ?? 0)])
  }
  return [...grouped.entries()].slice(0, 8).map(([name, data]) => ({ name, type: 'line', showSymbol: false, data: data.sort((a, b) => a[0] - b[0]) }))
}

export function ScientificOverview({ run }: { run: RunDetail }) {
  const design = artifact(run, 'experiment_design')?.payload
  const patch = artifact(run, 'plan_patch')?.payload
  const diagnosis = artifact(run, 'diagnosis_report')?.payload
  const series = measurementSeries(run)
  const conditions = (design?.conditions || []) as any[]
  return <Stack spacing={2}>
    {run.simulation_only && <Alert severity="info">本运行严格标记为 simulation_only；offline_replay 与 model_predicted 均不是湿实验事实。</Alert>}
    <Box display="grid" gridTemplateColumns={{ xs: '1fr', md: '1.25fr .75fr' }} gap={2}>
      <Card><CardContent>
        <Stack direction="row" justifyContent="space-between" alignItems="center" mb={2}><Typography variant="h3">数字孪生结果</Typography><Chip size="small" label={`${run.executions.length} 次结果`} /></Stack>
        {series.length ? <ReactEChartsCore echarts={echarts} style={{ height: 300 }} option={{ tooltip: { trigger: 'axis' }, legend: { type: 'scroll', bottom: 0 }, grid: { left: 48, right: 24, top: 20, bottom: 56 }, xAxis: { type: 'value', name: '时间 / h' }, yAxis: { type: 'value', name: '响应' }, series }} /> : <Box sx={{ height: 260, display: 'grid', placeItems: 'center', bgcolor: '#f8fbfa', borderRadius: 2 }}><Box textAlign="center"><PendingRounded color="primary" /><Typography color="text.secondary" mt={1}>执行完成后在此显示生长曲线与回流结果</Typography></Box></Box>}
      </CardContent></Card>
      <Stack spacing={2}>
        <Card><CardContent><Typography variant="h3" mb={1.5}>诊断结论</Typography><Typography color="text.secondary">{String(diagnosis?.conclusion || '等待诊断 Artifact')}</Typography>
          {!!diagnosis?.hypotheses?.length && <List dense>{diagnosis.hypotheses.slice(0, 3).map((item: any, index: number) => <ListItem key={index} disableGutters><ListItemText primary={item.cause || item.title || `候选 ${index + 1}`} secondary={item.support_level || item.rationale} /></ListItem>)}</List>}
        </CardContent></Card>
        <Card><CardContent><Typography variant="h3" mb={1}>约束修补</Typography>{patch ? <><Chip color="warning" size="small" label="PlanPatch" /><Typography mt={1.5}>{String(patch.reason || '已根据验证结果修补方案')}</Typography></> : <Typography color="text.secondary">当前没有触发方案修补。</Typography>}</CardContent></Card>
      </Stack>
    </Box>
    <Card><CardContent><Stack direction="row" justifyContent="space-between" mb={1.5}><Typography variant="h3">实验方案</Typography>{design?.design_hash && <Typography variant="caption" className="mono">hash {String(design.design_hash).slice(0, 16)}</Typography>}</Stack>
      {conditions.length ? <TableContainer><Table size="small"><TableHead><TableRow><TableCell>角色</TableCell><TableCell>条件</TableCell><TableCell>重复</TableCell><TableCell>预测</TableCell></TableRow></TableHead><TableBody>{conditions.map((condition, index) => <TableRow key={condition.id || index}><TableCell><Chip size="small" label={condition.role || 'candidate'} color={condition.role === 'control' ? 'default' : 'primary'} variant="outlined" /></TableCell><TableCell className="mono">{JSON.stringify(condition.factors || condition.condition || {})}</TableCell><TableCell>{condition.replicates || design?.replicates || 1}</TableCell><TableCell>{condition.predicted_response ?? '—'}</TableCell></TableRow>)}</TableBody></Table></TableContainer> : <EmptyState title="尚无实验方案" description="方案通过验证后将在这里显示条件、重复和预测。" />}
    </CardContent></Card>
  </Stack>
}

export function TraceGraph({ trace }: { trace?: Record<string, any> }) {
  const steps = trace?.steps || []
  const graph = useMemo(() => {
    const graphNodes: Node[] = []
    const graphEdges: Edge[] = []
    let previousId: string | undefined
    steps.forEach((step: any, index: number) => {
      const baseX = index * 520
      const parts = [
        { key: 'observation', label: `Observation\n${step.observation?.status || step.observation?.action || '结果已返回'}`, color: '#eef6ff', border: '#7aa7d8' },
        ...(step.replan_directive?.is_dynamic || step.replan_source === 'dynamic_observation' ? [{ key: 'replan', label: `Replan\n${step.replan_directive?.reason || '根据观察修订'}`, color: '#fff7e8', border: '#d49a35' }] : []),
        { key: 'decision', label: `Decision / Action\n${step.replan_decision?.action_name || step.decision?.action_name || step.action?.action_name || step.decision?.route_kind || '完成'}`, color: '#edf8f5', border: '#5aa89a' },
      ]
      parts.forEach((part, partIndex) => {
        const id = `${index}-${part.key}`
        graphNodes.push({ id, position: { x: baseX + partIndex * 190, y: part.key === 'replan' ? 55 : 0 }, data: { label: part.label }, style: { whiteSpace: 'pre-line', border: `1px solid ${part.border}`, borderRadius: 6, background: part.color, width: 170, fontSize: 12 } })
        if (previousId) graphEdges.push({ id: `e-${previousId}-${id}`, source: previousId, target: id, animated: part.key === 'replan' })
        previousId = id
      })
    })
    return { nodes: graphNodes, edges: graphEdges }
  }, [steps])
  if (!steps.length) return <EmptyState title="暂无 Trace" description="调试模式会在 Agent 执行后显示 Context、Decision、Policy、Tool 与 Observation。" />
  return <Box sx={{ height: 360, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}><ReactFlow nodes={graph.nodes} edges={graph.edges} fitView><Background /><Controls /></ReactFlow></Box>
}

export function NextActionPanel({ run, busy, onAction }: { run: RunDetail; busy: boolean; onAction: (action: ActionDescriptor) => void }) {
  const action = run.current_action
  return <Card sx={{ position: { md: 'sticky' }, top: { md: 88 } }}><CardContent>
    <Typography className="eyebrow">Next action</Typography><Typography variant="h2" mt={0.5}>下一步操作</Typography>
    {!action ? <Box mt={2}>{run.status === 'succeeded' ? <Alert icon={<CheckCircleRounded />} severity="success">流程已经完成，结果与审计记录均已保存。</Alert> : run.status === 'failed' || run.status === 'blocked' ? <Alert icon={<ErrorRounded />} severity="error">当前流程已阻断，请在调试模式查看失败阶段和事件。</Alert> : <Alert severity="info">系统正在推进当前阶段，页面会自动刷新。</Alert>}</Box> : <Stack spacing={1.5} mt={2}>
      <Typography color="text.secondary">当前流程需要明确的人工作用，完成后状态会留在同一个 Run 中继续推进。</Typography>
      {run.available_actions.map(item => <Button key={item.id} fullWidth variant={item.id === 'deny' ? 'outlined' : 'contained'} color={item.id === 'deny' ? 'error' : 'primary'} disabled={busy} onClick={() => onAction(item)}>{item.label}</Button>)}
      <Typography variant="caption" color="text.secondary">所需角色：{action.required_role} · {action.requires_confirmation ? '执行前需要确认' : '可直接执行'}</Typography>
    </Stack>}
  </CardContent></Card>
}

export function DebugJson({ value }: { value: unknown }) {
  return <pre className="json-view">{JSON.stringify(value, null, 2)}</pre>
}
