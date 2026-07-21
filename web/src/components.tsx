import { useMemo } from 'react'
import { Link as RouterLink } from 'react-router-dom'
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
import type { ActionDescriptor, Approval, Artifact, OperationSummary, ReplanSummary, RunDetail, RunSummary } from './types'

echarts.use([LineChart, GridComponent, LegendComponent, TooltipComponent, CanvasRenderer])

const statusMeta: Record<string, { label: string; color: 'default' | 'success' | 'warning' | 'error' | 'info' }> = {
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
