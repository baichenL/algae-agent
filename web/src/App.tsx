import { lazy, Suspense, useEffect, useState } from 'react'
import { Navigate, NavLink, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Alert, AppBar, Avatar, Badge, Box, Button, Card, CardContent, Chip, CircularProgress,
  Divider, Drawer, IconButton, List, ListItemButton, ListItemIcon,
  ListItemText, Stack, TextField, Toolbar, Tooltip, Typography, useMediaQuery,
  useTheme,
} from '@mui/material'
import ApprovalRounded from '@mui/icons-material/ApprovalRounded'
import BiotechRounded from '@mui/icons-material/BiotechRounded'
import BugReportRounded from '@mui/icons-material/BugReportRounded'
import DashboardRounded from '@mui/icons-material/DashboardRounded'
import DnsRounded from '@mui/icons-material/DnsRounded'
import ForumRounded from '@mui/icons-material/ForumRounded'
import HubRounded from '@mui/icons-material/HubRounded'
import LogoutRounded from '@mui/icons-material/LogoutRounded'
import MenuRounded from '@mui/icons-material/MenuRounded'
import PsychologyRounded from '@mui/icons-material/PsychologyRounded'
import ScienceRounded from '@mui/icons-material/ScienceRounded'
import SettingsRounded from '@mui/icons-material/SettingsRounded'
import StorageRounded from '@mui/icons-material/StorageRounded'
import ManageHistoryRounded from '@mui/icons-material/ManageHistoryRounded'
import { api, ApiError } from './api'
import { OperationFeedback } from './components'
import type { OperationSummary } from './types'
import {
  ApprovalsPage, AssistantPage, KnowledgePage, OverviewPage, ResourcesPage,
  RunDetailPage, RunsPage, SettingsPage, TestLabPage,
} from './pages'

const SimulationPage = lazy(() => import('./simulation').then(module => ({ default: module.SimulationPage })))

const drawerWidth = 248
const navigation: Array<{ label: string; path: string; icon: React.ReactNode; debug?: boolean }> = [
  { label: '总览', path: '/', icon: <DashboardRounded /> },
  { label: 'Assistant', path: '/assistant', icon: <ForumRounded /> },
  { label: '运行中心', path: '/runs', icon: <HubRounded />, debug: true },
  { label: 'Test Lab', path: '/test-lab', icon: <BugReportRounded /> },
  { label: '审批中心', path: '/approvals', icon: <ApprovalRounded /> },
  { label: '资源', path: '/resources', icon: <StorageRounded /> },
  { label: '知识库', path: '/knowledge', icon: <PsychologyRounded /> },
  { label: 'Agent 调试', path: '/debug/agent', icon: <DnsRounded />, debug: true },
  { label: '设置', path: '/settings', icon: <SettingsRounded /> },
]

function OperationsDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const queryClient = useQueryClient()
  const query = useQuery({ queryKey: ['operations'], queryFn: () => api.operations('recent'), refetchInterval: 5000 })
  const retry = useMutation({ mutationFn: api.retryOperation, onSuccess: () => queryClient.invalidateQueries({ queryKey: ['operations'] }) })
  useEffect(() => {
    if (typeof EventSource === 'undefined') return
    const source = new EventSource('/api/v2/operations/stream', { withCredentials: true })
    source.addEventListener('operations', event => {
      const operations = JSON.parse((event as MessageEvent).data) as OperationSummary[]
      queryClient.setQueryData(['operations'], { operations })
    })
    source.onerror = () => source.close()
    return () => source.close()
  }, [queryClient])
  const items = query.data?.operations || []
  return <Drawer anchor="right" open={open} onClose={onClose} PaperProps={{ sx: { width: { xs: '100%', sm: 430 }, p: 2.5 } }}>
    <Stack direction="row" justifyContent="space-between" alignItems="center" mb={0.5}><Typography variant="h2">后台任务</Typography><Button onClick={onClose}>关闭</Button></Stack>
    <Typography color="text.secondary" variant="body2" mb={2}>活跃任务与最近 24 小时结果。离开当前页面不会中断执行。</Typography>
    <Stack spacing={1.5}>{items.map(item => <OperationFeedback key={item.id} operation={item} onRetry={item.retryable ? () => retry.mutate(item.id) : undefined} />)}{!items.length && !query.isLoading && <Typography color="text.secondary" py={5} textAlign="center">暂无后台任务</Typography>}</Stack>
  </Drawer>
}

function LoginPage({ onSuccess }: { onSuccess: () => void }) {
  const [apiKey, setApiKey] = useState('')
  const mutation = useMutation({ mutationFn: () => api.login(apiKey), onSuccess })
  return <Box sx={{ minHeight: '100vh', display: 'grid', placeItems: 'center', p: 3, background: 'radial-gradient(circle at 20% 20%, #dff7f1 0, #f5f7f7 42%, #eef2f1 100%)' }}>
    <Card sx={{ width: '100%', maxWidth: 440 }}><CardContent sx={{ p: 4 }}>
      <Stack direction="row" spacing={1.5} alignItems="center" mb={3}><Avatar sx={{ bgcolor: 'primary.main' }}><ScienceRounded /></Avatar><Box><Typography variant="h2">Algae Agent</Typography><Typography color="text.secondary">Unified Control Center</Typography></Box></Stack>
      <Typography variant="h1" mb={1}>进入实验室控制台</Typography><Typography color="text.secondary" mb={3}>API Key 仅用于同源会话交换，不会写入浏览器存储或前端构建产物。</Typography>
      <TextField fullWidth type="password" label="API Key" value={apiKey} onChange={event => setApiKey(event.target.value)} onKeyDown={event => { if (event.key === 'Enter' && apiKey) mutation.mutate() }} />
      {mutation.error && <Alert severity="error" sx={{ mt: 2 }}>{mutation.error instanceof ApiError ? String(mutation.error.detail) : '登录失败'}</Alert>}
      <Button variant="contained" fullWidth size="large" sx={{ mt: 2 }} disabled={!apiKey || mutation.isPending} onClick={() => mutation.mutate()}>{mutation.isPending ? '正在验证…' : '创建安全会话'}</Button>
    </CardContent></Card>
  </Box>
}

export function AppShell({ session }: { session: any }) {
  const theme = useTheme()
  const mobile = useMediaQuery(theme.breakpoints.down('md'))
  const [mobileOpen, setMobileOpen] = useState(false)
  const [operationsOpen, setOperationsOpen] = useState(false)
  const [debugMode, setDebugMode] = useState(() => localStorage.getItem('algae-ui-mode') === 'debug')
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const location = useLocation()
  const operationsQuery = useQuery({ queryKey: ['operations'], queryFn: () => api.operations('recent'), refetchInterval: 5000 })
  const activeOperationCount = (operationsQuery.data?.operations || []).filter(item => ['queued', 'running', 'waiting_input'].includes(item.status)).length
  const logout = useMutation({ mutationFn: api.logout, onSuccess: () => { queryClient.clear(); navigate('/') } })
  const setMode = (checked: boolean) => { setDebugMode(checked); localStorage.setItem('algae-ui-mode', checked ? 'debug' : 'demo') }
  const visibleNavigation = navigation.filter(item => !item.debug || debugMode)
  const drawer = <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
    <Toolbar sx={{ gap: 1.5, px: 2.5 }}><Avatar sx={{ width: 34, height: 34, bgcolor: 'primary.main' }}><BiotechRounded fontSize="small" /></Avatar><Box><Typography fontWeight={800} lineHeight={1.1}>Algae Agent</Typography><Typography variant="caption" color="text.secondary">Control Center</Typography></Box></Toolbar>
    <Divider />
    <List sx={{ px: 1.25, py: 2 }}>{visibleNavigation.map(item => <ListItemButton key={item.path} component={NavLink} to={item.path} end={item.path === '/'} onClick={() => setMobileOpen(false)} sx={{ borderRadius: 2, mb: 0.5, '&.active': { bgcolor: '#e4f4f0', color: 'primary.dark', '& .MuiListItemIcon-root': { color: 'primary.main' } } }}><ListItemIcon sx={{ minWidth: 40 }}>{item.icon}</ListItemIcon><ListItemText primary={item.label} primaryTypographyProps={{ fontWeight: 650 }} /></ListItemButton>)}</List>
    <Box sx={{ mt: 'auto', p: 2 }}><Card variant="outlined" sx={{ bgcolor: '#f7fbfa' }}><CardContent sx={{ p: '14px !important' }}><Typography variant="caption" fontWeight={800} color="primary.main">ACTIVE WORKSPACE</Typography><Typography fontWeight={700} mt={0.5}>{session.workspace.name}</Typography><Stack direction="row" spacing={0.5} mt={1}><Chip size="small" label="simulation_only" color="info" /><Chip size="small" label={session.principal.role} variant="outlined" /></Stack></CardContent></Card></Box>
  </Box>
  return <Box sx={{ display: 'flex', minHeight: '100vh' }}>
    <AppBar position="fixed" color="inherit" elevation={0} sx={{ zIndex: theme.zIndex.drawer + 1, borderBottom: 1, borderColor: 'divider', ml: { md: `${drawerWidth}px` }, width: { md: `calc(100% - ${drawerWidth}px)` } }}><Toolbar>
      {mobile && <IconButton edge="start" aria-label="打开导航菜单" onClick={() => setMobileOpen(true)} sx={{ mr: 1 }}><MenuRounded /></IconButton>}
      <Box sx={{ flex: 1 }}><Typography fontWeight={750}>{navigation.find(item => item.path === location.pathname)?.label || (location.pathname.startsWith('/assistant/') ? 'Assistant' : '任务详情')}</Typography><Typography variant="caption" color="text.secondary">状态感知 · 独立验证 · 人工授权 · 数字孪生反馈</Typography></Box>
      <Stack direction="row" alignItems="center" spacing={1}>
        <Tooltip title="后台任务"><IconButton aria-label="后台任务" onClick={() => setOperationsOpen(true)}><Badge color="warning" badgeContent={activeOperationCount}><ManageHistoryRounded fontSize="small" /></Badge></IconButton></Tooltip>
        <Tooltip title={`${session.principal.name} · ${session.principal.role}`}><Avatar sx={{ width: 34, height: 34, bgcolor: '#dcebe8', color: '#0f766e', fontSize: 14 }}>{session.principal.name.slice(0, 2).toUpperCase()}</Avatar></Tooltip>
        <Tooltip title="退出"><IconButton onClick={() => logout.mutate()}><LogoutRounded fontSize="small" /></IconButton></Tooltip>
      </Stack>
    </Toolbar></AppBar>
    <Box component="nav" sx={{ width: { md: drawerWidth }, flexShrink: { md: 0 } }}><Drawer variant="temporary" open={mobileOpen} onClose={() => setMobileOpen(false)} ModalProps={{ keepMounted: true }} sx={{ display: { xs: 'block', md: 'none' }, '& .MuiDrawer-paper': { width: drawerWidth } }}>{drawer}</Drawer><Drawer variant="permanent" open sx={{ display: { xs: 'none', md: 'block' }, '& .MuiDrawer-paper': { width: drawerWidth, borderRightColor: 'divider' } }}>{drawer}</Drawer></Box>
    <Box component="main" sx={{ flexGrow: 1, width: { md: `calc(100% - ${drawerWidth}px)` }, minWidth: 0 }}><Toolbar /><Box sx={{ p: { xs: 2, sm: 3 }, maxWidth: 1600, mx: 'auto' }}>
      <Routes>
        <Route path="/" element={<OverviewPage />} />
        <Route path="/assistant" element={<AssistantPage />} />
        <Route path="/assistant/:conversationId" element={<AssistantPage />} />
        <Route path="/runs" element={debugMode ? <RunsPage /> : <Navigate to="/settings" replace />} />
        <Route path="/runs/:runId/simulation" element={<Suspense fallback={<Box sx={{ minHeight: 520, display: 'grid', placeItems: 'center' }}><CircularProgress /></Box>}><SimulationPage /></Suspense>} />
        <Route path="/runs/:runId" element={<RunDetailPage debugMode={debugMode} />} />
        <Route path="/test-lab" element={<TestLabPage />} />
        <Route path="/approvals" element={<ApprovalsPage />} />
        <Route path="/resources" element={<ResourcesPage />} />
        <Route path="/knowledge" element={<KnowledgePage />} />
        <Route path="/debug/agent" element={debugMode ? <RunsPage initialKind="agent" /> : <Navigate to="/settings" replace />} />
        <Route path="/observability" element={<Navigate to="/debug/agent" replace />} />
        <Route path="/settings" element={<SettingsPage session={session} debugMode={debugMode} onDebugModeChange={setMode} />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Box></Box>
    <OperationsDrawer open={operationsOpen} onClose={() => setOperationsOpen(false)} />
  </Box>
}

export default function App() {
  const query = useQuery({ queryKey: ['session'], queryFn: api.session, retry: false })
  if (query.isLoading) return <Box sx={{ minHeight: '100vh', display: 'grid', placeItems: 'center' }}><CircularProgress /></Box>
  if (query.error || !query.data) return <LoginPage onSuccess={() => query.refetch()} />
  return <AppShell session={query.data} />
}
