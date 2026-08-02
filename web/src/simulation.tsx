import { Component, useEffect, useMemo, useRef, useState } from 'react'
import { Link as RouterLink, useParams } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Canvas } from '@react-three/fiber'
import * as THREE from 'three'
import {
  Alert, Box, Button, Chip, CircularProgress, Divider, IconButton, LinearProgress,
  List, ListItemButton, ListItemText, MenuItem, Select, Stack, Typography,
  useMediaQuery, useTheme,
} from '@mui/material'
import CenterFocusStrongRounded from '@mui/icons-material/CenterFocusStrongRounded'
import FastForwardRounded from '@mui/icons-material/FastForwardRounded'
import PauseRounded from '@mui/icons-material/PauseRounded'
import PlayArrowRounded from '@mui/icons-material/PlayArrowRounded'
import ReplayRounded from '@mui/icons-material/ReplayRounded'
import SkipNextRounded from '@mui/icons-material/SkipNextRounded'
import { api } from './api'
import { formatDate, StatusChip } from './components'
import {
  LabDigitalTwin, SCENE_DEVICES, deriveSceneState, resolveActiveSceneDevice,
  type SceneDeviceId,
} from './simulation-scene'
import type { SimulationEvent, SimulationSnapshot } from './types'
import {
  advanceSimulationClock, buildMotionTimeline, motionFrameAt, timelineProgress,
  totalTimelineDuration,
} from './simulation-motion'

const STEP_LABELS: Record<string, string> = {
  CheckSchedule: '检查传代周期',
  ManualLoadLiquidHandler: '装载培养瓶、培养基和检测孔板',
  PrepareMeasurementPlate: '制备 96 孔检测板',
  ManualLoadSpectrophotometer: '将检测孔板交接到酶标仪',
  BlankSpectrophotometer: '酶标仪空白校准',
  MeasureAbsorbance: '读取 680 nm 吸光度',
  StoreMeasurementPlate: '检测板退出并进入缓存位',
  LoadMaterials: '检查培养基与耗材',
  DispenseMedium: '向目标培养瓶加注培养基',
  TransferSeedCulture: '转移种液',
  MixAndSeal: '混匀并封口',
  ManualMoveToIncubator: '将目标培养瓶交接到培养箱',
  MoveToIncubator: '配置培养箱温度与光照',
  RecordExperiment: '生成实验记录',
  CleanupWorkspace: '清理工作区',
  SafeShutdown: '安全停机',
}

export function selectSimulationFrame(
  events: SimulationEvent[],
  eventIndex: number,
  fallback: { snapshot?: SimulationSnapshot; step?: string; progress?: number } = {},
) {
  const event = events[Math.max(0, eventIndex)]
  const payload = event?.payload || {}
  const snapshot = payload.snapshot || fallback.snapshot || {}
  return {
    event,
    payload,
    snapshot,
    step: payload.step || fallback.step,
    progress: Number(payload.progress ?? fallback.progress ?? 0),
    fault: event?.event_type === 'simulation_fault'
      || event?.event_type === 'simulation_safe_shutdown'
      || Boolean(snapshot.alarm),
  }
}

function hasWebGL() {
  try {
    const canvas = document.createElement('canvas')
    return Boolean(canvas.getContext('webgl2') || canvas.getContext('webgl'))
  } catch {
    return false
  }
}

export function supportsSimulationAnimation(kind?: string) {
  return kind === 'workflow' || kind === 'simulation'
}

export function nextSimulationEventIndex(current: number, eventCount: number) {
  if (eventCount <= 0) return -1
  if (current < 0) return 0
  return Math.min(current + 1, eventCount - 1)
}

class SceneBoundary extends Component<{ children: React.ReactNode; fallback: React.ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  render() { return this.state.failed ? this.props.fallback : this.props.children }
}

function DeviceTopology({ id, active, fault, status }: { id: SceneDeviceId; active?: boolean; fault?: boolean; status?: string }) {
  const info = SCENE_DEVICES[id]
  return <Box
    data-testid={`simulation-device-${id}`}
    sx={{
      bgcolor: active ? '#e8f1eb' : '#f7f8f7',
      border: 1, borderColor: fault ? '#b32632' : active ? '#4d7d5e' : '#b8c2c1',
      borderRadius: 1.5, p: 1.5, minHeight: 82,
    }}
  >
    <Stack direction="row" justifyContent="space-between" spacing={1}>
      <Typography color="#263336" fontWeight={750}>{info.label}</Typography>
      <Typography variant="caption" color={fault ? '#b32632' : active ? '#3f704f' : '#667474'}>{status || 'STANDBY'}</Typography>
    </Stack>
    <Typography color="text.secondary" variant="caption" display="block" mt={0.75}>{info.role}</Typography>
  </Box>
}

function snapshotDeviceStatus(snapshot: SimulationSnapshot, id: SceneDeviceId) {
  if (id === 'plate_reader') return String(snapshot.plate_reader?.status || snapshot.spectrophotometer?.status || 'STANDBY')
  if (id === 'robot_arm') return String(snapshot.robot_arm?.status || 'STANDBY')
  if (id === 'mobile_robot') return String(snapshot.mobile_robot?.status || 'STANDBY')
  if (id === 'liquid_handler') return String(snapshot.liquid_handler?.status || 'STANDBY')
  if (id === 'incubator') return String(snapshot.incubator?.status || 'STANDBY')
  return 'STANDBY'
}

function FallbackView({ snapshot, payload, fault }: { snapshot: SimulationSnapshot; payload: SimulationEvent['payload']; fault: boolean }) {
  const active = resolveActiveSceneDevice(payload.device, payload.action, payload.step)
  const deviceIds = Object.keys(SCENE_DEVICES) as SceneDeviceId[]
  return <Box data-testid="simulation-static-fallback" sx={{ height: '100%', minHeight: 520, bgcolor: '#dfe6e6', p: { xs: 2, md: 3 }, color: '#263336' }}>
    <Alert severity="info" sx={{ mb: 2 }}>当前环境使用二维设备拓扑替代 3D 动画，仿真事件和设备状态仍在实时推进。</Alert>
    <Typography variant="h2" color="inherit" mb={2}>{STEP_LABELS[payload.step || ''] || payload.step || '准备仿真'}</Typography>
    <Box display="grid" gridTemplateColumns={{ xs: '1fr 1fr', md: 'repeat(3, 1fr)' }} gap={1.2}>
      {deviceIds.map(id => <DeviceTopology key={id} id={id} active={active === id} fault={fault && active === id} status={snapshotDeviceStatus(snapshot, id)} />)}
    </Box>
    <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} mt={2}>
      <Chip label={`孔板：${String(snapshot.measurement_plate?.location || 'plate_stack')}`} variant="outlined" />
      <Chip label={`载具：${String(snapshot.transport?.carrier || '待机')}`} variant="outlined" />
      <Chip label={`运动：${String(snapshot.transport?.motion_phase || 'IDLE')}`} variant="outlined" />
      <Chip label={`资源：${Object.keys(snapshot.resource_occupancy || {}).length ? '已预约' : '空闲'}`} variant="outlined" />
    </Stack>
  </Box>
}

export function SimulationPage() {
  const { runId = '' } = useParams()
  const id = decodeURIComponent(runId)
  const queryClient = useQueryClient()
  const theme = useTheme()
  const mobile = useMediaQuery(theme.breakpoints.down('md'))
  const reducedMotion = useMediaQuery('(prefers-reduced-motion: reduce)')
  const [webglAvailable] = useState(hasWebGL)
  const runQuery = useQuery({ queryKey: ['run', id], queryFn: () => api.run(id), refetchInterval: 2500 })
  const eventsQuery = useQuery({ queryKey: ['events', id], queryFn: () => api.events(id), refetchInterval: 2500 })
  const [paused, setPaused] = useState(false)
  const [playheadMs, setPlayheadMs] = useState(() => {
    if (typeof window === 'undefined') return 0
    const saved = Number(window.sessionStorage.getItem(`simulation-playhead:${id}`) || 0)
    return Number.isFinite(saved) ? saved : 0
  })
  const [speed, setSpeed] = useState(1)
  const [catchingUp, setCatchingUp] = useState(false)
  const [selectedDevice, setSelectedDevice] = useState<SceneDeviceId | undefined>()
  const [resetCameraKey, setResetCameraKey] = useState(0)
  const events = eventsQuery.data?.events || []
  const timeline = useMemo(() => buildMotionTimeline(events), [events])
  const totalDurationMs = totalTimelineDuration(timeline)
  const lastAnimationFrame = useRef<number | null>(null)
  const run = runQuery.data?.run
  const completed = ['succeeded', 'failed', 'blocked', 'cancelled'].includes(run?.status || '')
  const expectedEventCount = Number(run?.simulation?.event_count || 0)
  const eventStreamComplete = completed && (expectedEventCount <= 0 || events.length >= expectedEventCount)
  const manualMutation = useMutation({
    mutationFn: (resolution: 'completed' | 'failed') => {
      const taskId = run?.current_action?.task_id
      if (!taskId) throw new Error('当前没有可处理的人工任务')
      return api.resolveManual(id, taskId, resolution)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['events', id] })
      queryClient.invalidateQueries({ queryKey: ['run', id] })
    },
  })

  useEffect(() => {
    const stream = new EventSource(`/api/v2/runs/${encodeURIComponent(id)}/stream`)
    stream.addEventListener('run_event', () => {
      queryClient.invalidateQueries({ queryKey: ['events', id] })
      queryClient.invalidateQueries({ queryKey: ['run', id] })
    })
    stream.addEventListener('snapshot', () => queryClient.invalidateQueries({ queryKey: ['run', id] }))
    return () => stream.close()
  }, [id, queryClient])

  useEffect(() => {
    if (paused || totalDurationMs <= 0 || playheadMs >= totalDurationMs) {
      lastAnimationFrame.current = null
      return
    }
    let frameId = 0
    const tick = (now: number) => {
      const previous = lastAnimationFrame.current ?? now
      lastAnimationFrame.current = now
      setPlayheadMs(current => advanceSimulationClock(current, now - previous, catchingUp ? 4 : speed, totalDurationMs, false))
      frameId = window.requestAnimationFrame(tick)
    }
    frameId = window.requestAnimationFrame(tick)
    return () => {
      window.cancelAnimationFrame(frameId)
      lastAnimationFrame.current = null
    }
  }, [paused, playheadMs >= totalDurationMs, totalDurationMs, speed, catchingUp])

  useEffect(() => {
    const resetFrameClock = () => { lastAnimationFrame.current = null }
    document.addEventListener('visibilitychange', resetFrameClock)
    return () => document.removeEventListener('visibilitychange', resetFrameClock)
  }, [])

  useEffect(() => {
    window.sessionStorage.setItem(`simulation-playhead:${id}`, String(Math.round(playheadMs)))
  }, [id, playheadMs])

  useEffect(() => {
    if (catchingUp && eventStreamComplete && totalDurationMs > 0 && playheadMs >= totalDurationMs) setCatchingUp(false)
  }, [catchingUp, eventStreamComplete, playheadMs, totalDurationMs])

  if (runQuery.isLoading) return <Box sx={{ minHeight: 500, display: 'grid', placeItems: 'center' }}><CircularProgress /></Box>
  if (!run || !supportsSimulationAnimation(run.kind)) return <Alert severity="error">该 Run 不支持传代 3D 仿真。</Alert>
  const motionFrame = motionFrameAt(timeline, playheadMs, run.simulation?.latest_snapshot || {})
  const eventIndex = motionFrame.eventIndex
  const frame = selectSimulationFrame(events, eventIndex, {
    snapshot: run.simulation?.latest_snapshot,
    step: run.simulation?.current_step || run.raw?.current_step,
    progress: run.simulation?.progress ?? run.progress,
  })
  const selectedEvent = frame.event
  const payload = frame.payload
  const snapshot = motionFrame.snapshot && Object.keys(motionFrame.snapshot).length ? motionFrame.snapshot : frame.snapshot
  const step = frame.step
  const progress = timelineProgress(playheadMs, totalDurationMs) || frame.progress
  const fault = frame.fault || frame.step === 'SafeShutdown'
  const sceneState = deriveSceneState(snapshot, payload)
  const activeDevice = sceneState.activeDevice
  const useFallback = reducedMotion || !webglAvailable
  const queuePosition = run.simulation?.queue_position ?? Number(run.raw?.queue_position ?? 0)
  const waitingManual = run.current_action?.id === 'resolve_manual'
  const timelineSteps = [...new Map(events.filter(item => item.payload.step).map(item => [item.payload.step, item])).values()]
  const inspectedDevice = selectedDevice ? SCENE_DEVICES[selectedDevice] : undefined
  const viewingTerminal = eventStreamComplete && (events.length === 0 || playheadMs >= totalDurationMs)

  const resetCamera = () => {
    setSelectedDevice(undefined)
    setResetCameraKey(value => value + 1)
  }

  return <Box sx={{ mx: { xs: -2, sm: -3 }, mt: { xs: -2, sm: -3 }, minHeight: 'calc(100vh - 64px)', bgcolor: '#dfe6e6' }}>
    <Stack direction={{ xs: 'column', md: 'row' }} justifyContent="space-between" spacing={1.5} sx={{ px: { xs: 2, md: 3 }, py: 2, bgcolor: '#fff', borderBottom: 1, borderColor: 'divider' }}>
      <Box><Stack direction="row" spacing={1} alignItems="center"><Typography variant="h1">传代可视化仿真</Typography><Chip size="small" color="info" label="simulation_only" /></Stack><Typography color="text.secondary" mt={0.5}>{run.title} · {id}</Typography></Box>
      <Stack direction="row" spacing={1} alignItems="center"><StatusChip status={run.status} /><Button component={RouterLink} to={`/runs/${encodeURIComponent(id)}`}>查看 Run</Button></Stack>
    </Stack>
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'minmax(0, 1fr) 360px' }, minHeight: { lg: 'calc(100vh - 154px)' } }}>
      <Box sx={{ position: 'relative', minHeight: { xs: 540, lg: 'calc(100vh - 154px)' }, overflow: 'hidden' }}>
        {useFallback
          ? <FallbackView snapshot={snapshot} payload={payload} fault={fault} />
          : <Box data-testid="simulation-3d-scene" sx={{ height: '100%', minHeight: { xs: 540, lg: 'calc(100vh - 154px)' } }}>
            <SceneBoundary fallback={<FallbackView snapshot={snapshot} payload={payload} fault={fault} />}>
              <Canvas shadows={!mobile} dpr={mobile ? [1, 1.2] : [1, 1.6]} camera={{ position: [14.4, 8.4, 16.4], fov: 48 }} gl={{ antialias: !mobile, powerPreference: 'high-performance', toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.05 }}>
                <LabDigitalTwin snapshot={snapshot} payload={payload} motionFrame={motionFrame} fault={fault} paused={paused} selectedDevice={selectedDevice} resetCameraKey={resetCameraKey} onSelectDevice={setSelectedDevice} />
              </Canvas>
            </SceneBoundary>
          </Box>}

        {!useFallback && <Stack direction="row" spacing={0.75} useFlexGap flexWrap="wrap" sx={{ position: 'absolute', left: 14, top: 14, maxWidth: '76%' }}>
          {(Object.keys(SCENE_DEVICES) as SceneDeviceId[]).map(device => <Chip
            key={device}
            data-testid={`simulation-device-${device}`}
            size="small"
            clickable
            onClick={() => setSelectedDevice(device)}
            label={SCENE_DEVICES[device].label}
            sx={{ bgcolor: device === (selectedDevice || activeDevice) ? '#dfece3' : 'rgba(255,255,255,.9)', color: '#263336', border: '1px solid', borderColor: device === (selectedDevice || activeDevice) ? '#568168' : '#b7c0bf' }}
          />)}
        </Stack>}
        {!useFallback && <IconButton data-testid="simulation-camera-reset" aria-label="恢复导览视角" onClick={resetCamera} sx={{ position: 'absolute', right: 14, top: 14, color: '#314044', bgcolor: 'rgba(255,255,255,.92)', border: '1px solid #b7c0bf' }}><CenterFocusStrongRounded /></IconButton>}

        {inspectedDevice && <Box data-testid="simulation-device-inspector" sx={{ position: 'absolute', right: 14, top: 62, width: 250, bgcolor: 'rgba(255,255,255,.95)', color: '#263336', border: '1px solid #b7c0bf', borderRadius: 1.5, p: 1.5 }}>
          <Typography fontWeight={800}>{inspectedDevice.label}</Typography>
          <Typography variant="caption" color="text.secondary">{inspectedDevice.role}</Typography>
          <Typography variant="caption" display="block" mt={1}>状态：{snapshotDeviceStatus(snapshot, selectedDevice!)}</Typography>
        </Box>}

        <Box sx={{ position: 'absolute', left: 16, bottom: 16, right: 16, bgcolor: 'rgba(255,255,255,.94)', border: 1, borderColor: 'divider', borderRadius: 1.5, p: 1.25, backdropFilter: 'blur(8px)' }}>
          <Stack direction="row" spacing={1} alignItems="center">
            <IconButton aria-label={paused ? '继续观看' : '暂停观看'} onClick={() => setPaused(value => !value)}>{paused ? <PlayArrowRounded /> : <PauseRounded />}</IconButton>
            <IconButton aria-label="追赶实时位置" onClick={() => { setPaused(false); setCatchingUp(true) }}><SkipNextRounded /></IconButton>
            {completed && <IconButton aria-label="重新播放" onClick={() => { setPlayheadMs(0); setPaused(false); setCatchingUp(false) }}><ReplayRounded /></IconButton>}
            <LinearProgress variant="determinate" value={Math.round(progress * 100)} sx={{ flex: 1, height: 7 }} />
            <Typography variant="caption" sx={{ minWidth: 42 }}>{Math.round(progress * 100)}%</Typography>
            {completed && <Select size="small" value={speed} onChange={event => setSpeed(Number(event.target.value))}>{[0.5, 1, 2, 4].map(value => <MenuItem key={value} value={value}>{value}×</MenuItem>)}</Select>}
          </Stack>
        </Box>
      </Box>

      <Box sx={{ bgcolor: '#fff', borderLeft: { lg: 1 }, borderTop: { xs: 1, lg: 0 }, borderColor: 'divider', p: 2.5, minWidth: 0 }}>
        <Typography className="eyebrow">实时业务状态</Typography>
        <Typography variant="h2" mt={0.5}>{STEP_LABELS[step || ''] || step || '等待启动'}</Typography>
        <Typography color={fault ? 'error.main' : 'text.secondary'} mt={1}>{payload.message || (completed ? '仿真已结束' : '后台仿真正在推进')}</Typography>
        {motionFrame.phase && <Chip data-testid="simulation-motion-phase" size="small" sx={{ mt: 1 }} label={`动作阶段：${motionFrame.phase.name}`} variant="outlined" />}
        {run.status === 'queued' && <Alert data-testid="simulation-queue-state" severity="info" sx={{ mt: 2 }}>正在等待虚拟工作站{queuePosition > 0 ? `，当前排队位置：${queuePosition}` : ''}。设备可用后会自动开始播放。</Alert>}
        {!completed && <Alert severity="info" sx={{ mt: 2 }}>可以离开页面，仿真会在后台继续；暂停只影响观看。</Alert>}
        {fault && <Alert data-testid="simulation-safe-shutdown" severity="error" sx={{ mt: 2 }}>检测到设备故障，系统正在执行安全停机；品系数据不会修改。</Alert>}
        {waitingManual && <Stack data-testid="simulation-manual-action" spacing={1} mt={2}>
          <Alert severity="warning">{String(run.raw?.pending_manual_task?.instruction || '仿真已暂停，等待人工操作确认。')}</Alert>
          <Chip data-testid="simulation-handoff-zone" label={`交接：${String(run.raw?.pending_manual_task?.source || '操作区')} → ${String(run.raw?.pending_manual_task?.destination || '设备')}`} variant="outlined" />
          {manualMutation.error && <Alert severity="error">{manualMutation.error instanceof Error ? manualMutation.error.message : '人工任务处理失败'}</Alert>}
          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1}>
            <Button variant="contained" disabled={manualMutation.isPending} onClick={() => manualMutation.mutate('completed')}>已完成人工操作</Button>
            <Button color="error" disabled={manualMutation.isPending} onClick={() => manualMutation.mutate('failed')}>执行失败并安全结束</Button>
          </Stack>
        </Stack>}
        <Divider sx={{ my: 2 }} />
        <Stack spacing={1.15}>
          <Stack direction="row" justifyContent="space-between"><Typography color="text.secondary">目标瓶体积</Typography><Typography fontWeight={750}>{Number(snapshot.target_reactor?.volume_ml || 0).toFixed(1)} mL</Typography></Stack>
          <Stack direction="row" justifyContent="space-between"><Typography color="text.secondary">源瓶体积</Typography><Typography fontWeight={750}>{Number(snapshot.source_reactor?.volume_ml || 0).toFixed(1)} mL</Typography></Stack>
          <Stack direction="row" justifyContent="space-between"><Typography color="text.secondary">培养基余量</Typography><Typography fontWeight={750}>{Number(snapshot.media_reservoir?.volume_ml || 0).toFixed(1)} mL</Typography></Stack>
          <Stack direction="row" justifyContent="space-between"><Typography color="text.secondary">检测孔板</Typography><Typography fontWeight={750}>{String(snapshot.measurement_plate?.status || 'EMPTY')}</Typography></Stack>
          <Stack direction="row" justifyContent="space-between"><Typography color="text.secondary">吸光度</Typography><Typography fontWeight={750}>{snapshot.plate_reader?.absorbance ?? snapshot.spectrophotometer?.absorbance ?? '—'}</Typography></Stack>
          <Stack direction="row" justifyContent="space-between"><Typography color="text.secondary">搬运载具</Typography><Typography fontWeight={750}>{String(snapshot.transport?.carrier || '待机')}</Typography></Stack>
          <Stack direction="row" justifyContent="space-between"><Typography color="text.secondary">最近事件</Typography><Typography variant="caption">{formatDate(selectedEvent?.created_at || run.simulation?.last_event_at)}</Typography></Stack>
        </Stack>
        <Divider sx={{ my: 2 }} />
        <Typography variant="h3" mb={1}>步骤时间轴</Typography>
        <List dense sx={{ maxHeight: 280, overflowY: 'auto' }}>{timelineSteps.map(item => <ListItemButton key={`${item.sequence}-${item.payload.step}`} selected={item.sequence === selectedEvent?.sequence} onClick={() => { const entry = timeline.find(candidate => candidate.event.sequence === item.sequence); setPlayheadMs(entry?.startMs || 0); setPaused(true) }}><ListItemText primary={STEP_LABELS[item.payload.step || ''] || item.payload.step} secondary={`${Math.round(Number(item.payload.progress || 0) * 100)}% · ${item.payload.status || item.event_type}`} /></ListItemButton>)}</List>
        {viewingTerminal && !fault && <Stack spacing={1} mt={2}><Alert severity="success">仿真已完成，但不代表真实传代已经发生。</Alert><Button variant="contained" component={RouterLink} to={`/runs/${encodeURIComponent(id)}`}>{run.kind === 'workflow' ? '确认已手工完成并更新资源' : '查看 Run'}</Button><Button startIcon={<ReplayRounded />} onClick={() => { setPlayheadMs(0); setPaused(false); setCatchingUp(false) }}>重新播放</Button></Stack>}
        {viewingTerminal && fault && <Button fullWidth component={RouterLink} to={`/runs/${encodeURIComponent(id)}`} startIcon={<FastForwardRounded />} sx={{ mt: 2 }}>查看故障日志</Button>}
      </Box>
    </Box>
  </Box>
}
