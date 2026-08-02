import { useEffect, useMemo, useState } from 'react'
import { Link as RouterLink, useNavigate, useParams } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Alert, Avatar, Box, Button, Card, CardActionArea, CardContent, Chip, CircularProgress,
  Dialog, DialogActions, DialogContent, DialogTitle, Divider, Drawer, FormControl,
  FormControlLabel, Grid2 as Grid, IconButton, InputLabel, List, ListItem, ListItemAvatar,
  ListItemButton, ListItemText, MenuItem, Paper, Select, Stack, Switch, Tab, Table,
  TableBody, TableCell, TableContainer, TableHead, TableRow, Tabs, TextField, Tooltip, Typography,
} from '@mui/material'
import AddRounded from '@mui/icons-material/AddRounded'
import ArchiveRounded from '@mui/icons-material/ArchiveRounded'
import ApprovalRounded from '@mui/icons-material/ApprovalRounded'
import BiotechRounded from '@mui/icons-material/BiotechRounded'
import BugReportRounded from '@mui/icons-material/BugReportRounded'
import DataObjectRounded from '@mui/icons-material/DataObjectRounded'
import DnsRounded from '@mui/icons-material/DnsRounded'
import ForumRounded from '@mui/icons-material/ForumRounded'
import PlayArrowRounded from '@mui/icons-material/PlayArrowRounded'
import RefreshRounded from '@mui/icons-material/RefreshRounded'
import RestoreRounded from '@mui/icons-material/RestoreRounded'
import SendRounded from '@mui/icons-material/SendRounded'
import UploadFileRounded from '@mui/icons-material/UploadFileRounded'
import EditRounded from '@mui/icons-material/EditRounded'
import DeleteRounded from '@mui/icons-material/DeleteRounded'
import CancelRounded from '@mui/icons-material/CancelRounded'
import { api, ApiError } from './api'
import {
  AnswerEnvelopeCard, ApprovalCard, AsyncActionButton, DebugJson, DeveloperDetails, EmptyState, formatDate, MarkdownMessage, MetricCard, NextActionPanel, PageHeader,
  ReplanPanel, RunStepper, RunTable, ScientificOverview, StatusChip, TraceGraph,
} from './components'
import type { ActionDescriptor, AnswerEnvelope, Conversation, ConversationMessage, ConversationTask, RunDetail, UserMemory } from './types'

function Loading() { return <Box sx={{ minHeight: 320, display: 'grid', placeItems: 'center' }}><CircularProgress /></Box> }
function ErrorPanel({ error }: { error: unknown }) {
  const message = error instanceof ApiError
    ? typeof error.detail === 'object' && error.detail
      ? String((error.detail as any).message || (error.detail as any).code || JSON.stringify(error.detail))
      : String(error.detail)
    : error instanceof Error ? error.message : '加载失败'
  return <Alert severity="error">{message}</Alert>
}

export function OverviewPage() {
  const query = useQuery({ queryKey: ['dashboard'], queryFn: api.dashboard, refetchInterval: 5000 })
  if (query.isLoading) return <Loading />
  if (query.error) return <ErrorPanel error={query.error} />
  const data = query.data
  return <>
    <PageHeader eyebrow="Unified control plane" title="实验室运行总览" description="从数据、审批到数字孪生结果，统一查看当前实验室的运行状态。" action={<Button component={RouterLink} to="/test-lab" variant="contained" startIcon={<PlayArrowRounded />}>启动测试场景</Button>} />
    <Grid container spacing={2} mb={3}>
      <Grid size={{ xs: 12, sm: 6, lg: 3 }}><MetricCard label="活跃运行" value={data.counts.active_runs} hint="正在执行或等待人工" /></Grid>
      <Grid size={{ xs: 12, sm: 6, lg: 3 }}><MetricCard label="待审批" value={data.counts.pending_approvals} tone="warning" hint="需要 approver 处理" /></Grid>
      <Grid size={{ xs: 12, sm: 6, lg: 3 }}><MetricCard label="失败 / 阻断" value={data.counts.failed_runs} tone="error" hint="进入调试模式定位" /></Grid>
      <Grid size={{ xs: 12, sm: 6, lg: 3 }}><MetricCard label="在线设备" value={data.counts.online_devices} tone="neutral" hint="simulation_only adapters" /></Grid>
    </Grid>
    <Stack spacing={2}>
      <Card><CardContent><Typography variant="h2" mb={1}>最近任务</Typography><RunTable runs={data.recent_runs} compact /></CardContent></Card>
      <Grid container spacing={2}>
        <Grid size={{ xs: 12, lg: 7 }}><Card sx={{ height: '100%' }}><CardContent><Stack direction="row" justifyContent="space-between" mb={1}><Typography variant="h2">审批队列</Typography><Button component={RouterLink} to="/approvals" size="small">处理</Button></Stack>
          {data.pending_approvals.length ? <List dense>{data.pending_approvals.map((item: any) => <ListItem key={item.id} component={RouterLink} to={item.run_id ? `/runs/${encodeURIComponent(item.run_id)}` : '/approvals'} disableGutters><ListItemAvatar><Avatar sx={{ bgcolor: '#fff7ed', color: '#b45309' }}><ApprovalRounded /></Avatar></ListItemAvatar><ListItemText primary={item.summary} secondary={`${item.risk_level} · ${formatDate(item.created_at)}`} /></ListItem>)}</List> : <Typography color="text.secondary" py={2}>当前没有待审批项目。</Typography>}
        </CardContent></Card></Grid>
        <Grid size={{ xs: 12, lg: 5 }}><Card sx={{ height: '100%' }}><CardContent><Typography variant="h2" mb={1}>设备边界</Typography>{data.devices.map((device: any) => <Stack key={device.id} direction="row" justifyContent="space-between" alignItems="center" py={1}><Box><Typography fontWeight={700}>{device.name}</Typography><Typography variant="caption" color="text.secondary">{device.adapter_type}</Typography></Box><StatusChip status={device.status} /></Stack>)}</CardContent></Card></Grid>
      </Grid>
    </Stack>
  </>
}

export function AssistantPage() {
  const { conversationId = '' } = useParams()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [text, setText] = useState('')
  const [conversationStatus, setConversationStatus] = useState<'active' | 'archived'>('active')
  const [localMessages, setLocalMessages] = useState<ConversationMessage[]>([])
  const [mobileConversationsOpen, setMobileConversationsOpen] = useState(false)
  const conversations = useQuery({ queryKey: ['assistant-conversations', conversationStatus], queryFn: () => api.conversations(conversationStatus) })
  const messages = useQuery({ queryKey: ['assistant-messages', conversationId], queryFn: () => api.conversationMessages(conversationId), enabled: !!conversationId, refetchInterval: 1500 })
  const activeTask: ConversationTask | undefined = messages.data?.active_task
  const createConversation = useMutation({ mutationFn: () => api.createConversation(), onSuccess: result => {
    setConversationStatus('active')
    localStorage.setItem('algae-current-conversation', result.conversation.id)
    queryClient.invalidateQueries({ queryKey: ['assistant-conversations'] })
    navigate(`/assistant/${result.conversation.id}`)
  } })
  useEffect(() => {
    if (conversationId || conversations.isLoading || createConversation.isPending) return
    const items: Conversation[] = conversations.data?.conversations || []
    const saved = localStorage.getItem('algae-current-conversation')
    const selected = items.find(item => item.id === saved) || items[0]
    if (selected) navigate(`/assistant/${selected.id}`, { replace: true })
    else if (conversationStatus === 'active') createConversation.mutate()
  }, [conversationId, conversationStatus, conversations.data, conversations.isLoading, createConversation.isPending, navigate])
  useEffect(() => { setLocalMessages(messages.data?.messages || []) }, [messages.data])
  useEffect(() => { if (conversationId) localStorage.setItem('algae-current-conversation', conversationId) }, [conversationId])
  const mutation = useMutation({ mutationFn: ({ content, clientId }: { content: string; clientId: string }) => api.sendConversationMessage(conversationId, content, clientId), onSuccess: result => {
    setLocalMessages(current => [...current.filter(item => item.client_message_id !== result.user_message.client_message_id), result.user_message, result.assistant_message])
    queryClient.invalidateQueries({ queryKey: ['assistant-conversations'] })
    queryClient.invalidateQueries({ queryKey: ['assistant-messages', conversationId] })
  } })
  const archive = useMutation({ mutationFn: (id: string) => api.updateConversation(id, { status: 'archived' }), onSuccess: () => {
    localStorage.removeItem('algae-current-conversation')
    setConversationStatus('active')
    queryClient.invalidateQueries({ queryKey: ['assistant-conversations'] })
    navigate('/assistant')
  } })
  const restore = useMutation({ mutationFn: (id: string) => api.updateConversation(id, { status: 'active' }), onSuccess: () => {
    setConversationStatus('active')
    queryClient.invalidateQueries({ queryKey: ['assistant-conversations'] })
    navigate('/assistant')
  } })
  const cancelTask = useMutation({ mutationFn: (id: string) => api.cancelAssistantTask(id), onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: ['assistant-messages', conversationId] })
    queryClient.invalidateQueries({ queryKey: ['assistant-tasks', conversationId] })
  } })
  const submit = () => {
    const content = text.trim()
    if (!content || mutation.isPending || !conversationId) return
    const clientId = crypto.randomUUID()
    setLocalMessages(current => [...current, { id: -Date.now(), role: 'user', content, client_message_id: clientId }])
    setText('')
    mutation.mutate({ content, clientId })
  }
  const conversationItems: Conversation[] = conversations.data?.conversations || []
  const conversationPanel = (mobile = false) => <>
    <Stack direction="row" justifyContent="space-between" alignItems="center" px={1} pb={1}>
      <Typography variant="h3">历史会话</Typography>
      {conversationId && (conversationStatus === 'active'
        ? <Tooltip title="归档当前会话"><IconButton size="small" onClick={() => archive.mutate(conversationId)}><ArchiveRounded fontSize="small" /></IconButton></Tooltip>
        : <Tooltip title="恢复当前会话"><IconButton size="small" onClick={() => restore.mutate(conversationId)}><RestoreRounded fontSize="small" /></IconButton></Tooltip>)}
    </Stack>
    <Tabs value={conversationStatus} onChange={(_, value) => { setConversationStatus(value); navigate('/assistant') }} variant="fullWidth">
      <Tab value="active" label="进行中" />
      <Tab value="archived" label="已归档" />
    </Tabs>
    <Divider />
    <List sx={{ maxHeight: mobile ? 'calc(100vh - 140px)' : 570, overflowY: 'auto' }}>
      {conversationItems.map(item => <ListItemButton key={item.id} selected={item.id === conversationId} onClick={() => { navigate(`/assistant/${item.id}`); if (mobile) setMobileConversationsOpen(false) }} sx={{ borderRadius: 1, mb: 0.5 }}>
        <ListItemText primary={item.title} secondary={`${item.message_count || 0} 条消息 · ${formatDate(item.last_message_at || item.updated_at)}`} primaryTypographyProps={{ noWrap: true, fontWeight: item.id === conversationId ? 750 : 600 }} />
      </ListItemButton>)}
    </List>
  </>
  return <>
    <PageHeader eyebrow="Agent workspace" title="Assistant" description="当前会话会持续保存，只有新建或切换会话才会改变上下文。" action={<Stack direction="row" spacing={1}><Button variant="outlined" startIcon={<ForumRounded />} onClick={() => setMobileConversationsOpen(true)} sx={{ display: { xs: 'inline-flex', lg: 'none' } }}>会话</Button><Button variant="contained" startIcon={<AddRounded />} onClick={() => createConversation.mutate()} disabled={createConversation.isPending}>新对话</Button></Stack>} />
    <Drawer anchor="left" open={mobileConversationsOpen} onClose={() => setMobileConversationsOpen(false)} ModalProps={{ keepMounted: true }} PaperProps={{ sx: { width: 'min(86vw, 340px)', top: { xs: 72, sm: 64 }, height: { xs: 'calc(100% - 72px)', sm: 'calc(100% - 64px)' }, p: 1.5 } }}>{conversationPanel(true)}</Drawer>
    <Grid container spacing={2}><Grid size={{ lg: 3 }} sx={{ display: { xs: 'none', lg: 'block' } }}><Card sx={{ height: 650 }}><CardContent sx={{ p: 1.5 }}>{conversationPanel()}</CardContent></Card></Grid><Grid size={{ xs: 12, lg: 9 }}><Card sx={{ minHeight: 650 }}><CardContent>
      {activeTask && <Paper variant="outlined" sx={{ p: 1.5, mb: 2, borderColor: 'primary.light', bgcolor: '#f4fbf9' }}>
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5} justifyContent="space-between" alignItems={{ sm: 'center' }}>
          <Box sx={{ minWidth: 0 }}><Stack direction="row" spacing={1} alignItems="center"><Typography variant="h3">当前任务</Typography><StatusChip status={activeTask.status} /></Stack><Typography fontWeight={700} mt={0.5} sx={{ overflowWrap: 'anywhere' }}>{activeTask.goal_text}</Typography><Typography variant="caption" color="text.secondary">{activeTask.missing_slots?.length ? `还需要：${activeTask.missing_slots.join('、')}` : `阶段：${activeTask.task_type}`}</Typography></Box>
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>{activeTask.pending_id && <Button component={RouterLink} to="/approvals" size="small">查看审批 #{activeTask.pending_id}</Button>}{activeTask.workflow_run_id && <Button component={RouterLink} to={`/runs/${encodeURIComponent(activeTask.workflow_run_id)}`} size="small">查看运行</Button>}<AsyncActionButton size="small" color="inherit" startIcon={<CancelRounded />} busy={cancelTask.isPending} busyLabel="正在取消" onClick={() => cancelTask.mutate(activeTask.id)}>取消任务</AsyncActionButton></Stack>
        </Stack>
        {localStorage.getItem('algae-ui-mode') === 'debug' && <DeveloperDetails value={activeTask} />}
      </Paper>}
      <Stack spacing={2} sx={{ minHeight: 510, maxHeight: '66vh', overflowY: 'auto', pr: 1 }}>{localMessages.length ? localMessages.map(message => <Box key={message.id || message.client_message_id} alignSelf={message.role === 'user' ? 'flex-end' : 'flex-start'} sx={{ maxWidth: '82%' }}><Paper variant="outlined" sx={{ p: 2, bgcolor: message.role === 'user' ? '#e8f7f3' : '#fff' }}>
        {message.status === 'processing' && <Stack direction="row" spacing={1} alignItems="center" mb={1}><CircularProgress size={16} /><Typography variant="caption" fontWeight={750} color="primary.main">Agent 正在处理，可以离开页面</Typography></Stack>}
        <MarkdownMessage>{message.content}</MarkdownMessage>
        {message.role === 'assistant' && <AnswerEnvelopeCard
          envelope={(message.structured as any)?.answer_envelope as AnswerEnvelope | undefined}
          draft={(message.structured as any)?.draft}
          onResume={() => {
            const domain = (message.structured as any)?.task_spec?.task_domain
            setText(domain === 'email' ? '继续邮件任务' : '继续科学任务')
          }}
        />}
        {message.status === 'failed' && <Alert severity="error" sx={{ mt: 1 }}>任务处理失败，可在后台任务中查看原因并重试。</Alert>}
        {message.run_id && <Button component={RouterLink} to={`/runs/${encodeURIComponent(message.run_id)}`} size="small" endIcon={<PlayArrowRounded />} sx={{ mt: 1 }}>打开任务详情</Button>}
      </Paper></Box>) : <EmptyState title="开始新对话" description="询问品系状态、知识证据，或创建需要审批的实验任务。" />}</Stack>
      {mutation.error && <ErrorPanel error={mutation.error} />}
      <Stack direction="row" spacing={1} mt={2}><TextField fullWidth multiline maxRows={4} value={text} disabled={conversationStatus === 'archived'} onChange={event => setText(event.target.value)} onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); submit() } }} placeholder={conversationStatus === 'archived' ? '恢复会话后可继续发送消息' : '输入实验室问题或任务'} /><AsyncActionButton variant="contained" onClick={submit} busy={mutation.isPending} busyLabel="正在提交" disabled={!text.trim() || !conversationId || conversationStatus === 'archived'} sx={{ minWidth: 118 }} endIcon={<SendRounded />}>发送</AsyncActionButton></Stack>
    </CardContent></Card></Grid></Grid>
  </>
}

export function RunsPage({ initialKind = '' }: { initialKind?: string }) {
  const [kind, setKind] = useState(initialKind)
  const [status, setStatus] = useState('')
  const query = useQuery({ queryKey: ['runs', kind, status], queryFn: () => api.runs(kind, status), refetchInterval: 5000 })
  return <><PageHeader eyebrow="Run registry" title={initialKind === 'agent' ? 'Agent 调试' : '运行中心'} description={initialKind === 'agent' ? '检查 Agent Context、Decision、Policy、Tool、Observation 和 Replan。' : '所有科学闭环、Agent、Workflow 和仿真运行的统一入口。'} action={<Button onClick={() => query.refetch()} startIcon={<RefreshRounded />}>刷新</Button>} />
    <Card><CardContent><Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} mb={2}><FormControl size="small" sx={{ minWidth: 180 }}><InputLabel>流程类型</InputLabel><Select label="流程类型" value={kind} onChange={event => setKind(event.target.value)}><MenuItem value="">全部</MenuItem><MenuItem value="scientific">Scientific</MenuItem><MenuItem value="agent">Agent</MenuItem><MenuItem value="workflow">Workflow</MenuItem><MenuItem value="simulation">Simulation</MenuItem></Select></FormControl><FormControl size="small" sx={{ minWidth: 180 }}><InputLabel>运行状态</InputLabel><Select label="运行状态" value={status} onChange={event => setStatus(event.target.value)}><MenuItem value="">全部</MenuItem>{['running','waiting_input','waiting_approval','ready_to_execute','succeeded','failed','blocked'].map(item => <MenuItem key={item} value={item}>{item}</MenuItem>)}</Select></FormControl></Stack>{query.isLoading ? <Loading /> : query.error ? <ErrorPanel error={query.error} /> : <RunTable runs={query.data?.runs || []} />}</CardContent></Card>
  </>
}

export function RunDetailPage({ debugMode }: { debugMode: boolean }) {
  const { runId = '' } = useParams()
  const id = decodeURIComponent(runId)
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [tab, setTab] = useState(0)
  const [actionOpen, setActionOpen] = useState(false)
  const [manualOpen, setManualOpen] = useState(false)
  const [manualCompletedAt, setManualCompletedAt] = useState(() => {
    const now = new Date(Date.now() - new Date().getTimezoneOffset() * 60000)
    return now.toISOString().slice(0, 16)
  })
  const [manualNote, setManualNote] = useState('')
  const query = useQuery({ queryKey: ['run', id], queryFn: () => api.run(id), refetchInterval: 2500 })
  const events = useQuery({ queryKey: ['events', id], queryFn: () => api.events(id), refetchInterval: debugMode ? 2500 : false })
  useEffect(() => {
    const stream = new EventSource(`/api/v2/runs/${encodeURIComponent(id)}/stream`)
    const refresh = () => {
      queryClient.invalidateQueries({ queryKey: ['run', id] })
      if (debugMode) queryClient.invalidateQueries({ queryKey: ['events', id] })
    }
    stream.addEventListener('run_event', refresh)
    stream.addEventListener('snapshot', refresh)
    return () => stream.close()
  }, [debugMode, id, queryClient])
  const actionMutation = useMutation({ mutationFn: async (action: ActionDescriptor) => {
    if (action.id === 'approve' || action.id === 'deny') return api.decideApproval(action.approval_id!, action.id === 'approve' ? 'approved' : 'denied', '', id)
    if (action.id === 'execute') return api.executeRun(id)
    if (action.id === 'resolve_manual') return api.resolveManual(id, action.task_id!, 'completed')
    if (action.id === 'manual_commit') return api.manualCommit(id, manualCompletedAt, manualNote, crypto.randomUUID())
  }, onSuccess: result => {
    queryClient.invalidateQueries({ queryKey: ['run', id] }); queryClient.invalidateQueries({ queryKey: ['runs'] }); queryClient.invalidateQueries({ queryKey: ['approvals'] }); queryClient.invalidateQueries({ queryKey: ['operations'] })
    if (result?.decision === 'approved' && String(result?.run_id || '').startsWith('workflow:')) navigate(`/runs/${encodeURIComponent(result.run_id)}/simulation`)
  } })
  if (query.isLoading) return <Loading />
  if (query.error || !query.data) return <ErrorPanel error={query.error || new Error('Run 不存在')} />
  const run = query.data.run
  const perform = (action: ActionDescriptor) => {
    if (action.id === 'manual_commit') { setManualOpen(true); return }
    if (!action.requires_confirmation || window.confirm(`确认执行“${action.label}”？`)) actionMutation.mutate(action)
  }
  const visibleTabs = debugMode ? ['工作概览', '时间线', 'Artifacts', 'Trace', '事件', '断言', '原始数据'] : ['工作概览', '时间线', 'Artifacts', '断言']
  return <>
    <PageHeader eyebrow={`${run.kind} run`} title={run.title} description={run.summary || run.id} action={<Stack direction="row" spacing={1}><StatusChip status={run.status} size="medium" /><Chip label={`cycle ${run.cycle}`} variant="outlined" /><Chip label={run.simulation_only ? 'simulation_only' : 'physical'} color={run.simulation_only ? 'info' : 'warning'} /></Stack>} />
    <RunStepper run={run} />
    {!!run.replans?.length && <Box mt={2}><ReplanPanel replans={run.replans} debugMode={debugMode} /></Box>}
    {actionMutation.error && <Box mt={2}><ErrorPanel error={actionMutation.error} /></Box>}
    <Grid container spacing={2} mt={0}><Grid size={{ xs: 12, lg: 9 }}><Card><Tabs value={tab} onChange={(_, value) => setTab(value)} variant="scrollable" scrollButtons="auto" sx={{ px: 2, borderBottom: 1, borderColor: 'divider' }}>{visibleTabs.map(item => <Tab key={item} label={item} />)}</Tabs><CardContent>
      {visibleTabs[tab] === '工作概览' && (run.kind === 'scientific' ? <ScientificOverview run={run} /> : <GenericRunOverview run={run} />)}
      {visibleTabs[tab] === '时间线' && <TimelineView run={run} events={events.data?.events || []} />}
      {visibleTabs[tab] === 'Artifacts' && <ArtifactsView run={run} />}
      {visibleTabs[tab] === 'Trace' && <TraceGraph trace={run.trace} />}
      {visibleTabs[tab] === '事件' && <DebugJson value={events.data?.events || run.raw_events || []} />}
      {visibleTabs[tab] === '断言' && <AssertionsView run={run} />}
      {visibleTabs[tab] === '原始数据' && <DebugJson value={run.raw || run} />}
    </CardContent></Card></Grid><Grid size={{ xs: 12, lg: 3 }} sx={{ display: { xs: 'none', lg: 'block' } }}><NextActionPanel run={run} busy={actionMutation.isPending} onAction={perform} /></Grid></Grid>
    {run.current_action && <Box sx={{ display: { xs: 'block', lg: 'none' } }}>
      <Button variant="contained" size="large" onClick={() => setActionOpen(true)} sx={{ position: 'fixed', right: 20, bottom: 20, zIndex: theme => theme.zIndex.speedDial }}>{run.current_action.label}</Button>
      <Drawer anchor="bottom" open={actionOpen} onClose={() => setActionOpen(false)} PaperProps={{ sx: { borderRadius: '18px 18px 0 0', maxHeight: '72vh' } }}><Box sx={{ p: 2, maxWidth: 680, width: '100%', mx: 'auto' }}><NextActionPanel run={run} busy={actionMutation.isPending} onAction={action => { setActionOpen(false); perform(action) }} /></Box></Drawer>
    </Box>}
    <Dialog open={manualOpen} onClose={() => setManualOpen(false)} fullWidth maxWidth="sm"><DialogTitle>确认已手工完成传代</DialogTitle><DialogContent><Alert severity="warning" sx={{ mb: 2 }}>提交后将增加品系代数并更新最近传代时间。该记录会标记为人工确认，不会被记为硬件执行。</Alert><TextField fullWidth type="datetime-local" label="实际完成时间" value={manualCompletedAt} onChange={event => setManualCompletedAt(event.target.value)} InputLabelProps={{ shrink: true }} /><TextField fullWidth multiline minRows={3} label="操作备注" value={manualNote} onChange={event => setManualNote(event.target.value)} sx={{ mt: 2 }} /></DialogContent><DialogActions><Button onClick={() => setManualOpen(false)}>取消</Button><AsyncActionButton variant="contained" busy={actionMutation.isPending} busyLabel="正在校验并写入" disabled={!manualCompletedAt} onClick={() => { actionMutation.mutate({ id: 'manual_commit', label: '确认手工传代', kind: 'manual_commit', required_role: 'approver', requires_confirmation: true }); setManualOpen(false) }}>确认并更新资源</AsyncActionButton></DialogActions></Dialog>
  </>
}

function GenericRunOverview({ run }: { run: RunDetail }) {
  const raw = run.raw || {}
  return <Stack spacing={2}><Grid container spacing={2}><Grid size={{ xs: 12, sm: 4 }}><MetricCard label="当前阶段" value={run.phase} /></Grid><Grid size={{ xs: 12, sm: 4 }}><MetricCard label="总体进度" value={`${Math.round(run.progress * 100)}%`} tone="neutral" /></Grid><Grid size={{ xs: 12, sm: 4 }}><MetricCard label="风险等级" value={run.risk_level} tone={run.risk_level === 'high' ? 'warning' : 'neutral'} /></Grid></Grid>
    {raw.pending_manual_task && <Alert severity="warning">等待人工任务：{raw.pending_manual_task.instruction || raw.pending_manual_task.step}</Alert>}
    <Card variant="outlined"><CardContent><Typography variant="h3" mb={1}>运行摘要</Typography><Typography color="text.secondary">{run.summary || '当前运行正在推进。'}</Typography>{raw.current_step && <Typography mt={2}><strong>当前步骤：</strong>{raw.current_step}</Typography>}{raw.last_error && <Alert severity="error" sx={{ mt: 2 }}>{raw.last_error.message || JSON.stringify(raw.last_error)}</Alert>}</CardContent></Card>
    {run.kind === 'agent' && <TraceGraph trace={run.trace} />}
  </Stack>
}

function TimelineView({ run, events }: { run: RunDetail; events: any[] }) {
  const items = events.length ? events : run.artifacts.map(item => ({ sequence: item.id, event_type: item.artifact_type, created_at: item.created_at, phase: `v${item.version}` }))
  return items.length ? <List>{items.map((item: any, index: number) => <ListItem key={item.sequence || index} alignItems="flex-start" sx={{ borderLeft: '2px solid #99c9c0', ml: 1, pl: 3 }}><ListItemAvatar><Avatar sx={{ width: 30, height: 30, bgcolor: '#e5f5f1', color: '#0f766e' }}>{index + 1}</Avatar></ListItemAvatar><ListItemText primary={item.event_type || item.phase} secondary={`${item.phase || ''} · ${formatDate(item.created_at)}`} /></ListItem>)}</List> : <EmptyState title="暂无时间线" description="运行事件产生后会按顺序显示。" />
}

function ArtifactsView({ run }: { run: RunDetail }) {
  return run.artifacts.length ? <Stack spacing={1.5}>{run.artifacts.map(item => <Paper key={`${item.artifact_type}-${item.version}`} variant="outlined" sx={{ p: 2 }}><Stack direction="row" justifyContent="space-between" mb={1}><Box><Typography fontWeight={750}>{item.artifact_type}</Typography><Typography variant="caption" className="mono">v{item.version} · {item.content_hash?.slice(0, 12)}</Typography></Box><Typography variant="caption">{formatDate(item.created_at)}</Typography></Stack><DebugJson value={item.payload} /></Paper>)}</Stack> : <EmptyState title="暂无 Artifact" description="结构化产物会在运行过程中逐步生成。" />
}

function AssertionsView({ run }: { run: RunDetail }) {
  return run.assertions.length ? <Stack spacing={1}>{run.assertions.map((item, index) => <Alert key={index} severity={item.status === 'passed' ? 'success' : 'error'}><strong>{item.name}</strong>{item.details ? `：${item.details}` : ''}</Alert>)}</Stack> : <EmptyState title="暂无断言" description="Test Lab 场景和 Verifier 结果会在这里形成可重复检查。" />
}

export function testLabRunPath(runId: string) {
  const encoded = encodeURIComponent(runId)
  return runId.startsWith('simulation:') ? `/runs/${encoded}/simulation` : `/runs/${encoded}`
}

export function TestLabPage() {
  const navigate = useNavigate()
  const query = useQuery({ queryKey: ['scenarios'], queryFn: api.scenarios })
  const datasets = useQuery({ queryKey: ['resources', 'datasets'], queryFn: () => api.resources('datasets') })
  const [sources, setSources] = useState<Record<string, string>>({})
  const [uploadScenario, setUploadScenario] = useState<string | null>(null)
  const mutation = useMutation({
    mutationFn: ({ id, parameters }: { id: string; parameters: Record<string, unknown> }) => api.startScenario(id, parameters),
    onSuccess: result => navigate(testLabRunPath(String(result.run_id))),
  })
  return <><PageHeader eyebrow="Isolated scenario runner" title="Test Lab" description="固定 seed、simulation_only 与安全人工边界，让每类流程都可以重复验证。" action={<Chip icon={<BugReportRounded />} color="info" label="test-default · isolated simulation" />} />
    {mutation.error && <ErrorPanel error={mutation.error} />}{query.isLoading ? <Loading /> : query.error ? <ErrorPanel error={query.error} /> : <Grid container spacing={2}>{query.data.scenarios.map((scenario: any) => {
      const selected = sources[scenario.id] || ''
      const parameters = selected === 'demo' ? { use_demo_dataset: true } : selected ? { dataset_id: selected } : {}
      return <Grid key={scenario.id} data-testid={`test-scenario-${scenario.id}`} size={{ xs: 12, md: 6, xl: 4 }}><Card sx={{ height: '100%' }}><CardContent sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}><Stack direction="row" justifyContent="space-between"><Chip size="small" label={scenario.kind} variant="outlined" />{scenario.requires_dataset && <Chip size="small" label="需要数据集" color="warning" />}</Stack><Typography variant="h2" mt={2}>{scenario.name}</Typography><Typography color="text.secondary" mt={1} mb={2}>{scenario.description}</Typography>{scenario.requires_dataset && <Stack spacing={1.25} mb={2}><FormControl fullWidth size="small"><InputLabel>数据来源</InputLabel><Select label="数据来源" value={selected} onChange={event => setSources({ ...sources, [scenario.id]: event.target.value })}><MenuItem value="demo">使用内置样例数据</MenuItem>{(datasets.data?.items || []).filter((item: any) => item.status === 'ready').map((item: any) => <MenuItem key={item.id} value={item.id}>{item.name} · {item.strain_id || '未关联品系'} · {item.row_count} 行</MenuItem>)}</Select></FormControl><Button size="small" variant="outlined" startIcon={<UploadFileRounded />} onClick={() => setUploadScenario(scenario.id)}>上传新数据集</Button></Stack>}<Divider /><Typography variant="caption" fontWeight={750} mt={2}>预期断言</Typography><List dense sx={{ flex: 1 }}>{scenario.assertions.map((item: string) => <ListItem key={item} disableGutters><ListItemText primary={`✓ ${item}`} /></ListItem>)}</List><AsyncActionButton data-testid={`test-scenario-start-${scenario.id}`} variant="contained" busy={mutation.isPending} busyLabel="正在创建隔离工作区" startIcon={<PlayArrowRounded />} onClick={() => mutation.mutate({ id: scenario.id, parameters })} disabled={mutation.isPending || (scenario.requires_dataset && !selected)}>{scenario.kind === 'simulation' ? '启动并观看动画' : '运行至下一人工边界'}</AsyncActionButton></CardContent></Card></Grid>
    })}</Grid>}
    <DatasetImportDialog open={!!uploadScenario} onClose={() => setUploadScenario(null)} onImported={dataset => { if (uploadScenario) setSources({ ...sources, [uploadScenario]: dataset.id }); setUploadScenario(null); datasets.refetch() }} />
  </>
}

export function ApprovalsPage() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [status, setStatus] = useState('pending')
  const query = useQuery({ queryKey: ['approvals', status], queryFn: () => api.approvals(status), refetchInterval: 4000 })
  const mutation = useMutation({ mutationFn: ({ id, decision, runId }: { id: number; decision: 'approved' | 'denied'; runId?: string }) => api.decideApproval(id, decision, '', runId), onSuccess: result => {
    queryClient.invalidateQueries({ queryKey: ['approvals'] }); queryClient.invalidateQueries({ queryKey: ['runs'] }); queryClient.invalidateQueries({ queryKey: ['operations'] })
    if (result?.decision === 'approved' && String(result?.run_id || '').startsWith('workflow:')) navigate(`/runs/${encodeURIComponent(result.run_id)}/simulation`)
  } })
  const retry = useMutation({ mutationFn: (id: number) => api.retryApproval(id), onSuccess: () => { queryClient.invalidateQueries({ queryKey: ['approvals'] }); queryClient.invalidateQueries({ queryKey: ['runs'] }) } })
  const approvals = query.data?.approvals ?? []
  return <><PageHeader eyebrow="Human-in-the-loop" title="审批中心" description="批准后系统会自动启动对应执行；失败记录可在这里检查并重试。" action={<FormControl size="small" sx={{ minWidth: 150 }}><InputLabel>状态</InputLabel><Select label="状态" value={status} onChange={event => setStatus(event.target.value)}><MenuItem value="pending">待审批</MenuItem><MenuItem value="approved">已批准</MenuItem><MenuItem value="denied">已拒绝</MenuItem><MenuItem value="all">全部</MenuItem></Select></FormControl>} />
    {(mutation.error || retry.error) && <ErrorPanel error={mutation.error || retry.error} />}{query.isLoading ? <Loading /> : query.error ? <ErrorPanel error={query.error} /> : approvals.length ? <Grid container spacing={2}>{approvals.map(item => <Grid key={`${item.run_id || 'shared'}:${item.id}`} size={{ xs: 12, lg: 6 }}><ApprovalCard approval={item} busy={mutation.isPending || retry.isPending} onDecision={decision => mutation.mutate({ id: item.id, decision, runId: item.run_id })} onRetry={() => retry.mutate(item.id)} /></Grid>)}</Grid> : <EmptyState title="当前筛选下没有审批" description="新的高风险操作会自动进入审批中心并关联所属 Run。" />}
  </>
}

export function ResourcesPage() {
  const [tab, setTab] = useState(0)
  const [strainDialog, setStrainDialog] = useState<{ open: boolean; item?: any }>({ open: false })
  const [datasetDialog, setDatasetDialog] = useState(false)
  const [strainForm, setStrainForm] = useState<any>({ strain_id: '', name_cn: '', name_en: '', generation_number: 1, days_since_last_subculture: 0 })
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const types = ['strains', 'datasets', 'devices']
  const query = useQuery({ queryKey: ['resources', types[tab]], queryFn: () => api.resources(types[tab]) })
  const request = useMutation({ mutationFn: ({ operation, data }: any) => api.requestStrainChange(operation, data), onSuccess: () => { setStrainDialog({ open: false }); queryClient.invalidateQueries({ queryKey: ['resources', 'strains'] }); navigate('/approvals') } })
  const openStrain = (item?: any) => {
    setStrainForm(item ? {
      strain_id: item.strain_id,
      name_cn: item.name_cn,
      name_en: item.name_en,
      generation_number: Number(item.generation_number),
      days_since_last_subculture: Number(item.days_since_last_subculture),
    } : { strain_id: '', name_cn: '', name_en: '', generation_number: 1, days_since_last_subculture: 0 })
    setStrainDialog({ open: true, item })
  }
  const removeStrain = (item: any) => {
    const confirmation = window.prompt(`输入 ${item.strain_id} 以确认删除品系`)
    if (confirmation === item.strain_id) request.mutate({ operation: 'delete', data: { strain_id: item.strain_id } })
  }
  const action = tab === 0 ? <Button variant="contained" startIcon={<AddRounded />} onClick={() => openStrain()}>新建品系</Button> : tab === 1 ? <Button variant="contained" startIcon={<UploadFileRounded />} onClick={() => setDatasetDialog(true)}>导入数据集</Button> : undefined
  return <><PageHeader eyebrow="Laboratory facts" title="资源中心" description="品系变更经过审批，数据集可直接导入，设备能力保持只读。" action={action} /><Card><Tabs value={tab} onChange={(_, value) => setTab(value)} sx={{ px: 2, borderBottom: 1, borderColor: 'divider' }}><Tab label="品系" /><Tab label="数据集" /><Tab label="设备" /></Tabs><CardContent>{request.error && <Box mb={2}><ErrorPanel error={request.error} /></Box>}{query.isLoading ? <Loading /> : query.error ? <ErrorPanel error={query.error} /> : <ResourceTable type={types[tab]} items={query.data.items || []} onEdit={openStrain} onDelete={removeStrain} />}</CardContent></Card>
    <Dialog open={strainDialog.open} onClose={() => setStrainDialog({ open: false })} fullWidth maxWidth="sm"><DialogTitle>{strainDialog.item ? '编辑品系' : '新建品系'}</DialogTitle><DialogContent><Stack spacing={2} mt={1}><TextField label="品系编号" value={strainForm.strain_id} disabled={!!strainDialog.item} onChange={event => setStrainForm({ ...strainForm, strain_id: event.target.value })} /><TextField label="中文名称" value={strainForm.name_cn || ''} onChange={event => setStrainForm({ ...strainForm, name_cn: event.target.value })} /><TextField label="英文名称" value={strainForm.name_en || ''} onChange={event => setStrainForm({ ...strainForm, name_en: event.target.value })} /><Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}><TextField fullWidth type="number" label="当前代数" value={strainForm.generation_number} onChange={event => setStrainForm({ ...strainForm, generation_number: Number(event.target.value) })} /><TextField fullWidth type="number" label="距上次传代天数" value={strainForm.days_since_last_subculture} onChange={event => setStrainForm({ ...strainForm, days_since_last_subculture: Number(event.target.value) })} /></Stack></Stack></DialogContent><DialogActions><Button onClick={() => setStrainDialog({ open: false })}>取消</Button><AsyncActionButton variant="contained" busy={request.isPending} busyLabel="正在提交审批" disabled={!strainForm.strain_id || !strainForm.name_cn || !strainForm.name_en} onClick={() => request.mutate({ operation: strainDialog.item ? 'update' : 'add', data: strainForm })}>提交审批</AsyncActionButton></DialogActions></Dialog>
    <DatasetImportDialog open={datasetDialog} onClose={() => setDatasetDialog(false)} onImported={() => { setDatasetDialog(false); queryClient.invalidateQueries({ queryKey: ['resources', 'datasets'] }) }} />
  </>
}

function ResourceTable({ type, items, onEdit, onDelete }: { type: string; items: any[]; onEdit?: (item: any) => void; onDelete?: (item: any) => void }) {
  if (!items.length) return <EmptyState title="暂无资源" description="导入数据或创建设备与品系记录后会显示在这里。" />
  const keys = type === 'strains' ? ['strain_id','name_cn','name_en','generation_number','days_since_last_subculture'] : type === 'datasets' ? ['id','name','strain_id','batch_count','row_count','status'] : ['id','name','adapter_type','mode','status']
  return <TableContainer><Table><TableHead><TableRow>{keys.map(key => <TableCell key={key}>{key}</TableCell>)}{type === 'strains' && <TableCell align="right">操作</TableCell>}</TableRow></TableHead><TableBody>{items.map((item, index) => <TableRow key={item.id || item.strain_id || index} hover>{keys.map(key => <TableCell key={key}>{typeof item[key] === 'object' ? JSON.stringify(item[key]) : String(item[key] ?? '—')}</TableCell>)}{type === 'strains' && <TableCell align="right"><Tooltip title="编辑"><IconButton size="small" onClick={() => onEdit?.(item)}><EditRounded fontSize="small" /></IconButton></Tooltip><Tooltip title="删除"><IconButton size="small" color="error" onClick={() => onDelete?.(item)}><DeleteRounded fontSize="small" /></IconButton></Tooltip></TableCell>}</TableRow>)}</TableBody></Table></TableContainer>
}

function DatasetImportDialog({ open, onClose, onImported }: { open: boolean; onClose: () => void; onImported: (dataset: any) => void }) {
  const [file, setFile] = useState<File | null>(null)
  const [name, setName] = useState('')
  const [strainId, setStrainId] = useState('')
  const [columns, setColumns] = useState<string[]>([])
  const [timeColumn, setTimeColumn] = useState('')
  const [responseColumn, setResponseColumn] = useState('')
  const mutation = useMutation({ mutationFn: async () => {
    const form = new FormData()
    form.append('file', file!)
    if (name) form.append('dataset_name', name)
    if (strainId) form.append('strain_id', strainId)
    if (timeColumn && responseColumn) form.append('mapping_json', JSON.stringify({ time_column: timeColumn, response_column: responseColumn }))
    return api.importDataset(form)
  }, onSuccess: result => onImported(result.dataset), onError: error => {
    if (error instanceof ApiError && typeof error.detail === 'object' && error.detail && (error.detail as any).code === 'mapping_required') {
      const detail = error.detail as any
      setColumns(detail.columns || [])
      setTimeColumn(detail.suggested_mapping?.time_column || '')
      setResponseColumn(detail.suggested_mapping?.response_column || '')
    }
  } })
  return <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm"><DialogTitle>导入科学数据集</DialogTitle><DialogContent><Stack spacing={2} mt={1}><Button component="label" variant="outlined" startIcon={<UploadFileRounded />}>{file ? file.name : '选择 CSV 或 XLSX 文件'}<input hidden type="file" accept=".csv,.xlsx" onChange={event => setFile(event.target.files?.[0] || null)} /></Button><TextField label="数据集名称（可选）" value={name} onChange={event => setName(event.target.value)} /><TextField label="关联品系（可选）" value={strainId} onChange={event => setStrainId(event.target.value)} />{columns.length > 0 && <Alert severity="info">需要确认数据列映射后才能导入。</Alert>}{columns.length > 0 && <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}><FormControl fullWidth><InputLabel>时间列</InputLabel><Select label="时间列" value={timeColumn} onChange={event => setTimeColumn(event.target.value)}>{columns.map(column => <MenuItem key={column} value={column}>{column}</MenuItem>)}</Select></FormControl><FormControl fullWidth><InputLabel>响应列</InputLabel><Select label="响应列" value={responseColumn} onChange={event => setResponseColumn(event.target.value)}>{columns.map(column => <MenuItem key={column} value={column}>{column}</MenuItem>)}</Select></FormControl></Stack>}{mutation.error && <ErrorPanel error={mutation.error} />}</Stack></DialogContent><DialogActions><Button onClick={onClose}>取消</Button><AsyncActionButton variant="contained" busy={mutation.isPending} busyLabel="正在解析数据集" disabled={!file || (columns.length > 0 && (!timeColumn || !responseColumn))} onClick={() => mutation.mutate()}>导入</AsyncActionButton></DialogActions></Dialog>
}

export function KnowledgePage() {
  const [uploadOpen, setUploadOpen] = useState(false)
  const queryClient = useQueryClient()
  const query = useQuery({ queryKey: ['knowledge-sources'], queryFn: api.knowledgeSources, refetchInterval: 5000 })
  const action = useMutation({ mutationFn: ({ id, kind }: { id: string; kind: 'reindex' | 'archive' | 'restore' }) => kind === 'reindex' ? api.reindexKnowledge(id) : api.setKnowledgeArchived(id, kind === 'archive'), onSuccess: () => queryClient.invalidateQueries({ queryKey: ['knowledge-sources'] }) })
  const labels: Record<string, string> = { media_recipe: '培养基配方', manual: 'SOP / 操作手册', paper: '文献', experiment_data: '实验数据', device_document: '设备资料', document: '其他资料' }
  const sources = query.data?.sources || []
  return <><PageHeader eyebrow="Knowledge assets" title="知识库" description="管理 Assistant 检索使用的实验室文件、类别和索引状态。" action={<Button variant="contained" startIcon={<UploadFileRounded />} onClick={() => setUploadOpen(true)}>添加文件</Button>} />
    <Grid container spacing={1.5} mb={2}>{Object.entries(labels).map(([key, label]) => <Grid key={key} size={{ xs: 6, md: 4, xl: 2 }}><Card variant="outlined"><CardContent sx={{ p: '14px !important' }}><Typography variant="caption" color="text.secondary">{label}</Typography><Typography variant="h2" mt={0.5}>{query.data?.category_counts?.[key] || 0}</Typography></CardContent></Card></Grid>)}</Grid>
    {action.error && <Box mb={2}><ErrorPanel error={action.error} /></Box>}
    <Card><CardContent>{query.isLoading ? <Loading /> : query.error ? <ErrorPanel error={query.error} /> : sources.length ? <TableContainer><Table><TableHead><TableRow><TableCell>文件</TableCell><TableCell>类别</TableCell><TableCell>索引状态</TableCell><TableCell>分块</TableCell><TableCell>更新时间</TableCell><TableCell align="right">操作</TableCell></TableRow></TableHead><TableBody>{sources.map((source: any) => <TableRow key={source.source_id} hover><TableCell><Typography fontWeight={700}>{source.file_name}</Typography>{source.last_error && <Typography variant="caption" color="error">{source.last_error}</Typography>}</TableCell><TableCell>{labels[source.doc_type] || source.doc_type}</TableCell><TableCell><StatusChip status={source.lifecycle_status === 'archived' ? 'archived' : source.ingestion_status} /></TableCell><TableCell>{source.chunk_count || 0}</TableCell><TableCell>{formatDate(source.updated_at)}</TableCell><TableCell align="right">{source.ingestion_status === 'failed' && <Button size="small" onClick={() => action.mutate({ id: source.source_id, kind: 'reindex' })}>重试</Button>}<Button size="small" color={source.lifecycle_status === 'archived' ? 'primary' : 'warning'} onClick={() => action.mutate({ id: source.source_id, kind: source.lifecycle_status === 'archived' ? 'restore' : 'archive' })}>{source.lifecycle_status === 'archived' ? '恢复' : '归档'}</Button></TableCell></TableRow>)}</TableBody></Table></TableContainer> : <EmptyState title="知识库中还没有文件" description="添加 SOP、配方、文献、实验数据或设备资料后会在这里显示。" />}</CardContent></Card>
    <KnowledgeUploadDialog open={uploadOpen} onClose={() => setUploadOpen(false)} onUploaded={() => { setUploadOpen(false); queryClient.invalidateQueries({ queryKey: ['knowledge-sources'] }) }} />
  </>
}

function KnowledgeUploadDialog({ open, onClose, onUploaded }: { open: boolean; onClose: () => void; onUploaded: () => void }) {
  const [file, setFile] = useState<File | null>(null)
  const [docType, setDocType] = useState('manual')
  const [title, setTitle] = useState('')
  const [version, setVersion] = useState('')
  const [language, setLanguage] = useState('zh')
  const mutation = useMutation({ mutationFn: () => {
    const form = new FormData()
    form.append('file', file!)
    form.append('doc_type', docType)
    if (title) form.append('title', title)
    if (version) form.append('version', version)
    if (language) form.append('language', language)
    return api.uploadKnowledge(form)
  }, onSuccess: onUploaded })
  return <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm"><DialogTitle>添加知识文件</DialogTitle><DialogContent><Stack spacing={2} mt={1}><Button component="label" variant="outlined" startIcon={<UploadFileRounded />}>{file ? file.name : '选择文件'}<input hidden type="file" accept=".pdf,.docx,.xlsx,.xls,.csv,.md,.txt" onChange={event => setFile(event.target.files?.[0] || null)} /></Button><FormControl fullWidth><InputLabel>知识类别</InputLabel><Select label="知识类别" value={docType} onChange={event => setDocType(event.target.value)}><MenuItem value="media_recipe">培养基配方</MenuItem><MenuItem value="manual">SOP / 操作手册</MenuItem><MenuItem value="paper">文献</MenuItem><MenuItem value="experiment_data">实验数据</MenuItem><MenuItem value="device_document">设备资料</MenuItem><MenuItem value="document">其他资料</MenuItem></Select></FormControl><TextField label="标题（可选）" value={title} onChange={event => setTitle(event.target.value)} /><Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}><TextField fullWidth label="版本（可选）" value={version} onChange={event => setVersion(event.target.value)} /><TextField fullWidth label="语言" value={language} onChange={event => setLanguage(event.target.value)} /></Stack>{mutation.error && <ErrorPanel error={mutation.error} />}</Stack></DialogContent><DialogActions><Button onClick={onClose}>取消</Button><AsyncActionButton variant="contained" busy={mutation.isPending} busyLabel="正在上传文件" disabled={!file} onClick={() => mutation.mutate()}>上传并索引</AsyncActionButton></DialogActions></Dialog>
}

function SystemSettingsSummary({ session, debugMode, onDebugModeChange }: { session: any; debugMode: boolean; onDebugModeChange: (value: boolean) => void }) {
  const dashboard = useQuery({ queryKey: ['dashboard-settings'], queryFn: api.dashboard })
  return <><PageHeader eyebrow="System" title="设置与边界" description="检查当前会话、权限、工作区和安全执行边界。" /><Grid container spacing={2}><Grid size={{ xs: 12, md: 6 }}><Card><CardContent><Typography variant="h2" mb={2}>当前会话</Typography><Table size="small"><TableBody><TableRow><TableCell>用户</TableCell><TableCell>{session.principal.name}</TableCell></TableRow><TableRow><TableCell>角色</TableCell><TableCell><Chip size="small" label={session.principal.role} /></TableCell></TableRow><TableRow><TableCell>工作区</TableCell><TableCell>{session.workspace.name}</TableCell></TableRow><TableRow><TableCell>执行边界</TableCell><TableCell><Chip size="small" color="info" label={session.workspace.mode} /></TableCell></TableRow></TableBody></Table></CardContent></Card></Grid><Grid size={{ xs: 12, md: 6 }}><Card><CardContent><Typography variant="h2" mb={2}>界面模式</Typography><FormControlLabel control={<Switch checked={debugMode} onChange={event => onDebugModeChange(event.target.checked)} />} label="启用调试模式" /><Typography variant="body2" color="text.secondary" mt={1}>开启后显示运行中心、Agent 调试、Trace、事件和原始数据。此设置只影响当前浏览器的显示。</Typography><Divider sx={{ my: 2 }} /><Alert severity="info">当前前端使用 HttpOnly 会话，不在浏览器存储 API Key。</Alert><Alert severity="warning" sx={{ mt: 1 }}>所有设备结果必须保留 simulation_only、offline_replay 或 model_predicted 来源标签。</Alert><Typography mt={2} color="text.secondary">在线设备：{dashboard.data?.counts?.online_devices ?? '—'}</Typography></CardContent></Card></Grid></Grid></>
}

export function SettingsPage(props: { session: any; debugMode: boolean; onDebugModeChange: (value: boolean) => void }) {
  return <><SystemSettingsSummary {...props} /><UserMemoryPanel /></>
}

function UserMemoryPanel() {
  const queryClient = useQueryClient()
  const settings = useQuery({ queryKey: ['memory-settings'], queryFn: api.memorySettings })
  const active = useQuery({ queryKey: ['memories', 'active'], queryFn: () => api.memories('active') })
  const archived = useQuery({ queryKey: ['memories', 'archived'], queryFn: () => api.memories('archived') })
  const candidates = useQuery({ queryKey: ['memory-candidates'], queryFn: api.memoryCandidates })
  const [editing, setEditing] = useState<UserMemory | null>(null)
  const [editValue, setEditValue] = useState('')
  const [note, setNote] = useState('')
  const refresh = () => {
    queryClient.invalidateQueries({ queryKey: ['memories'] })
    queryClient.invalidateQueries({ queryKey: ['memory-candidates'] })
    queryClient.invalidateQueries({ queryKey: ['memory-settings'] })
  }
  const mutation = useMutation({ mutationFn: (action: () => Promise<unknown>) => action(), onSuccess: refresh })
  const renderValue = (value: unknown) => typeof value === 'string' ? value : JSON.stringify(value)
  const changeSettings = (enabled: boolean, autoWrite: boolean) => mutation.mutate(() => api.updateMemorySettings({ enabled, auto_write_low_risk: autoWrite }))
  const exportJson = async () => {
    const data = await api.exportMemories()
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }))
    const link = document.createElement('a')
    link.href = url
    link.download = 'algae-agent-memories.json'
    link.click()
    URL.revokeObjectURL(url)
  }
  const renderList = (items: UserMemory[], isArchived = false) => items.length ? <TableContainer><Table size="small"><TableHead><TableRow><TableCell>谓词 / 内容</TableCell><TableCell>来源与版本</TableCell><TableCell align="right">操作</TableCell></TableRow></TableHead><TableBody>{items.map(memory => <TableRow key={memory.id}><TableCell><Typography fontWeight={700}>{memory.predicate}</Typography><Typography variant="body2">{renderValue(memory.value)}</Typography><Typography variant="caption" color="text.secondary">置信度 {Math.round(memory.confidence * 100)}% · {formatDate(memory.updated_at)}</Typography></TableCell><TableCell><Typography variant="caption">{memory.source_conversation_id || '用户手工创建'} · revision {memory.revision}</Typography></TableCell><TableCell align="right">{!isArchived && <IconButton aria-label="编辑记忆" onClick={() => { setEditing(memory); setEditValue(renderValue(memory.value)) }}><EditRounded /></IconButton>}<IconButton aria-label={isArchived ? '恢复记忆' : '归档记忆'} onClick={() => mutation.mutate(() => isArchived ? api.restoreMemory(memory.id) : api.archiveMemory(memory.id))}>{isArchived ? <RestoreRounded /> : <ArchiveRounded />}</IconButton><IconButton color="error" aria-label="永久删除记忆" onClick={() => window.confirm('永久删除后无法恢复，确认继续？') && mutation.mutate(() => api.deleteMemory(memory.id))}><DeleteRounded /></IconButton></TableCell></TableRow>)}</TableBody></Table></TableContainer> : <Typography color="text.secondary" py={2}>暂无记忆。</Typography>

  if (settings.isLoading || active.isLoading || archived.isLoading || candidates.isLoading) return <Box mt={2}><Loading /></Box>
  const error = settings.error || active.error || archived.error || candidates.error || mutation.error
  if (!settings.data || !active.data || !archived.data || !candidates.data) {
    return <Box mt={2}>{error ? <ErrorPanel error={error} /> : <Loading />}</Box>
  }
  const current = settings.data!.settings
  return <Card sx={{ mt: 2 }}><CardContent><Stack direction={{ xs: 'column', md: 'row' }} justifyContent="space-between" gap={2}><Box><Typography variant="h2">我的记忆</Typography><Typography variant="body2" color="text.secondary" mt={0.5}>用户偏好仅用于建议和展示，不会修改实验事实、Policy、审批结果或工具权限。</Typography></Box><Button onClick={exportJson}>导出 JSON</Button></Stack>{error && <Box mt={2}><ErrorPanel error={error} /></Box>}<Stack mt={2}><FormControlLabel control={<Switch checked={current.enabled} onChange={event => changeSettings(event.target.checked, current.auto_write_low_risk)} />} label="启用跨会话 Memory" /><FormControlLabel control={<Switch checked={current.auto_write_low_risk} disabled={!current.enabled} onChange={event => changeSettings(current.enabled, event.target.checked)} />} label="自动保存高置信、低风险偏好" /></Stack><Divider sx={{ my: 2 }} /><Typography variant="h3">待确认候选</Typography>{candidates.data!.candidates.length ? <List>{candidates.data!.candidates.map(candidate => <ListItem key={candidate.id} secondaryAction={<Stack direction="row"><Button onClick={() => mutation.mutate(() => api.decideMemoryCandidate(candidate.id, true))}>确认</Button><Button color="warning" onClick={() => mutation.mutate(() => api.decideMemoryCandidate(candidate.id, false))}>拒绝</Button></Stack>}><ListItemText primary={`${candidate.predicate}: ${renderValue(candidate.value)}`} secondary={`置信度 ${Math.round(candidate.confidence * 100)}% · ${candidate.reason || '待用户确认'} · ${candidate.source_conversation_id || '无来源会话'}`} /></ListItem>)}</List> : <Typography color="text.secondary" py={2}>没有待确认候选。</Typography>}<Divider sx={{ my: 2 }} /><Stack direction={{ xs: 'column', sm: 'row' }} gap={1}><TextField fullWidth label="新增一般备注（仅作建议）" value={note} onChange={event => setNote(event.target.value)} /><Button variant="outlined" disabled={!note.trim()} onClick={() => mutation.mutate(async () => { await api.createMemory('note', 'general.note', note.trim()); setNote('') })}>添加</Button></Stack><Typography variant="h3" mt={3}>当前记忆</Typography>{renderList(active.data!.memories)}<Typography variant="h3" mt={3}>已归档</Typography>{renderList(archived.data!.memories, true)}</CardContent><Dialog open={!!editing} onClose={() => setEditing(null)} fullWidth maxWidth="sm"><DialogTitle>修正记忆</DialogTitle><DialogContent><TextField fullWidth multiline minRows={3} sx={{ mt: 1 }} label={editing?.predicate} value={editValue} onChange={event => setEditValue(event.target.value)} /></DialogContent><DialogActions><Button onClick={() => setEditing(null)}>取消</Button><Button variant="contained" onClick={() => editing && mutation.mutate(async () => { const value = typeof editing.value === 'string' ? editValue : JSON.parse(editValue); await api.updateMemory(editing.id, value, editing.revision); setEditing(null) })}>保存新版本</Button></DialogActions></Dialog></Card>
}
