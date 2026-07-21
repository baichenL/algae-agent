import { Component, useEffect, useRef, useState } from 'react'
import { Link as RouterLink, useParams } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Canvas, useFrame } from '@react-three/fiber'
import { Line, OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import {
  Alert, Box, Button, Chip, CircularProgress, Divider, IconButton, LinearProgress,
  List, ListItemButton, ListItemText, MenuItem, Select, Stack, Typography,
  useMediaQuery, useTheme,
} from '@mui/material'
import FastForwardRounded from '@mui/icons-material/FastForwardRounded'
import PauseRounded from '@mui/icons-material/PauseRounded'
import PlayArrowRounded from '@mui/icons-material/PlayArrowRounded'
import ReplayRounded from '@mui/icons-material/ReplayRounded'
import SkipNextRounded from '@mui/icons-material/SkipNextRounded'
import { api } from './api'
import { formatDate, StatusChip } from './components'
import type { SimulationEvent, SimulationSnapshot } from './types'

const STEP_LABELS: Record<string, string> = {
  CheckSchedule: '检查传代周期',
  ManualLoadSpectrophotometer: '装载分光光度计',
  BlankSpectrophotometer: '分光光度计校准',
  MeasureAbsorbance: '测量吸光度',
  ManualLoadLiquidHandler: '装载液体处理器',
  LoadMaterials: '检查培养基与耗材',
  DispenseMedium: '加注培养基',
  TransferSeedCulture: '转移种液',
  MixAndSeal: '混匀并封口',
  ManualMoveToIncubator: '移入培养箱',
  MoveToIncubator: '配置培养箱',
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
    fault: event?.event_type === 'simulation_fault' || event?.event_type === 'simulation_safe_shutdown' || Boolean(snapshot.alarm),
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

class SceneBoundary extends Component<{ children: React.ReactNode; fallback: React.ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  render() { return this.state.failed ? this.props.fallback : this.props.children }
}

function DeviceBox({ position, size, color, active = false, fault = false }: { position: [number, number, number]; size: [number, number, number]; color: string; label: string; active?: boolean; fault?: boolean }) {
  const ref = useRef<THREE.Mesh>(null)
  useFrame(({ clock }) => {
    if (ref.current && active) (ref.current.material as THREE.MeshStandardMaterial).opacity = 0.72 + Math.sin(clock.elapsedTime * 5) * 0.18
  })
  return <group position={position}>
    <mesh ref={ref as any} castShadow receiveShadow>
      <boxGeometry args={size} />
      <meshStandardMaterial color={fault ? '#c62828' : color} transparent opacity={0.9} roughness={0.38} metalness={0.12} />
    </mesh>
  </group>
}

function CultureVessel({ position, volume, capacity, color }: { position: [number, number, number]; label: string; volume: number; capacity: number; color: string }) {
  const level = Math.max(0.03, Math.min(volume / Math.max(capacity, 1), 1))
  return <group position={position}>
    <mesh castShadow><cylinderGeometry args={[0.34, 0.5, 1.15, 28]} /><meshPhysicalMaterial color="#dcebea" transparent opacity={0.34} roughness={0.18} transmission={0.35} /></mesh>
    <mesh position={[0, -0.54 + (1.08 * level) / 2, 0]}><cylinderGeometry args={[0.31, 0.46, 1.08 * level, 28]} /><meshStandardMaterial color={color} transparent opacity={0.82} /></mesh>
    <mesh position={[0, 0.68, 0]}><cylinderGeometry args={[0.17, 0.17, 0.26, 24]} /><meshStandardMaterial color="#b9c8c5" /></mesh>
  </group>
}

function SampleMarker({ location }: { location: string }) {
  const positions: Record<string, [number, number, number]> = {
    manual_zone: [-3.9, 1.25, 1.4], spectrophotometer: [-2.25, 1.55, -0.25],
    liquid_handler: [0.2, 1.82, -0.35], incubator: [3.65, 1.65, -0.2], in_transit: [-0.5, 2.5, 0.8],
  }
  const target = positions[location] || positions.manual_zone
  const ref = useRef<THREE.Group>(null)
  useFrame(() => {
    if (!ref.current) return
    ref.current.position.lerp(new THREE.Vector3(...target), 0.08)
  })
  return <group ref={ref as any} position={target}>
    <mesh castShadow><sphereGeometry args={[0.12, 20, 20]} /><meshStandardMaterial color="#d1f05b" emissive="#7d9d20" emissiveIntensity={0.35} /></mesh>
    <pointLight color="#d1f05b" intensity={0.5} distance={1.2} />
  </group>
}

function LabScene({ snapshot, activeDevice, fault }: { snapshot: SimulationSnapshot; activeDevice?: string; fault: boolean }) {
  const source = snapshot.source_reactor || {}
  const target = snapshot.target_reactor || {}
  const media = snapshot.media_reservoir || {}
  const sample = snapshot.sample || {}
  const deviceFault = (name: string) => fault && (activeDevice === name || Boolean(snapshot.alarm))
  return <>
    <color attach="background" args={['#eef3f1']} />
    <ambientLight intensity={1.25} />
    <directionalLight castShadow position={[4, 8, 5]} intensity={2.1} shadow-mapSize={[1024, 1024]} />
    <mesh position={[0, -0.28, 0]} receiveShadow><boxGeometry args={[10, 0.45, 4.8]} /><meshStandardMaterial color="#c7d1ce" roughness={0.72} /></mesh>
    <mesh position={[0, -0.02, -1.95]} receiveShadow><boxGeometry args={[10, 0.12, 0.35]} /><meshStandardMaterial color="#63746f" /></mesh>
    <CultureVessel position={[-3.8, 0.55, 0.5]} label="原藻液瓶" volume={Number(source.volume_ml || 200)} capacity={Number(source.capacity_ml || 250)} color="#41935b" />
    <CultureVessel position={[2.1, 0.55, 0.7]} label="新培养瓶" volume={Number(target.volume_ml || 0)} capacity={Number(target.capacity_ml || 250)} color="#78bd6c" />
    <CultureVessel position={[-1.7, 0.55, 1.25]} label="培养基" volume={Number(media.volume_ml || 1000)} capacity={Number(media.capacity_ml || 1000)} color="#d8c66a" />
    <DeviceBox position={[0.1, 0.62, -0.5]} size={[2.25, 1.35, 1.55]} color="#4f8f87" label="液体处理器" active={activeDevice === 'liquid_handler' || activeDevice?.includes('pump')} fault={deviceFault('liquid_handler')} />
    <DeviceBox position={[-2.4, 0.58, -0.55]} size={[1.25, 1.2, 1.3]} color="#5686a6" label="分光光度计" active={activeDevice === 'spectrophotometer'} fault={deviceFault('spectrophotometer')} />
    <DeviceBox position={[3.65, 0.85, -0.6]} size={[1.8, 1.8, 1.55]} color="#8a8873" label="培养箱" active={activeDevice === 'incubator'} fault={deviceFault('incubator')} />
    <Line points={[[-1.7, 1.1, 1.1], [-0.65, 1.3, 0.2], [1.9, 1.05, 0.65]]} color={fault ? '#c62828' : '#d7b93e'} lineWidth={3} />
    <Line points={[[0, 1.3, -0.1], [-1.4, 1.55, 0.25], [-3.5, 1.2, 0.5]]} color={fault ? '#c62828' : '#4d9b67'} lineWidth={3} />
    <SampleMarker location={String(sample.location || 'manual_zone')} />
    <OrbitControls makeDefault enablePan={false} minDistance={7} maxDistance={13} minPolarAngle={0.72} maxPolarAngle={1.36} target={[0, 0.65, 0]} />
  </>
}

function FallbackView({ snapshot, step }: { snapshot: SimulationSnapshot; step?: string }) {
  const devices = ['spectrophotometer', 'liquid_handler', 'media_pump', 'seed_pump', 'incubator']
  return <Box sx={{ height: '100%', minHeight: 430, bgcolor: '#eef3f1', p: 3, display: 'grid', alignContent: 'center' }}>
    <Alert severity="info" sx={{ mb: 2 }}>当前环境使用设备状态图代替 3D 动画，仿真事件仍在实时推进。</Alert>
    <Typography variant="h2" mb={2}>{STEP_LABELS[step || ''] || step || '准备仿真'}</Typography>
    <Box display="grid" gridTemplateColumns={{ xs: '1fr 1fr', sm: 'repeat(3, 1fr)' }} gap={1.5}>{devices.map(device => <Box key={device} sx={{ bgcolor: '#fff', border: 1, borderColor: 'divider', p: 2, minHeight: 92 }}><Typography fontWeight={750}>{device}</Typography><Typography color="text.secondary" mt={1}>{String((snapshot as any)[device]?.status || 'STANDBY')}</Typography></Box>)}</Box>
  </Box>
}

export function SimulationPage() {
  const { runId = '' } = useParams()
  const id = decodeURIComponent(runId)
  const queryClient = useQueryClient()
  const theme = useTheme()
  const mobile = useMediaQuery(theme.breakpoints.down('md'))
  const reducedMotion = useMediaQuery('(prefers-reduced-motion: reduce)')
  const runQuery = useQuery({ queryKey: ['run', id], queryFn: () => api.run(id), refetchInterval: 2500 })
  const eventsQuery = useQuery({ queryKey: ['events', id], queryFn: () => api.events(id), refetchInterval: 2500 })
  const [paused, setPaused] = useState(false)
  const [eventIndex, setEventIndex] = useState(-1)
  const [speed, setSpeed] = useState(1)
  const events = eventsQuery.data?.events || []
  const run = runQuery.data?.run
  const completed = ['succeeded', 'failed', 'blocked'].includes(run?.status || '')

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
    if (!paused && events.length) setEventIndex(events.length - 1)
  }, [events.length, paused])

  useEffect(() => {
    if (!completed || paused || eventIndex < 0 || eventIndex >= events.length - 1) return
    const timer = window.setTimeout(() => setEventIndex(index => Math.min(index + 1, events.length - 1)), 700 / speed)
    return () => window.clearTimeout(timer)
  }, [completed, paused, eventIndex, events.length, speed])

  if (runQuery.isLoading) return <Box sx={{ minHeight: 500, display: 'grid', placeItems: 'center' }}><CircularProgress /></Box>
  if (!run || run.kind !== 'workflow') return <Alert severity="error">该 Run 不支持传代 3D 仿真。</Alert>
  const frame = selectSimulationFrame(events, eventIndex, {
    snapshot: run.simulation?.latest_snapshot,
    step: run.simulation?.current_step || run.raw?.current_step,
    progress: run.simulation?.progress ?? run.progress,
  })
  const selectedEvent = frame.event
  const payload = frame.payload
  const snapshot = frame.snapshot
  const step = frame.step
  const progress = frame.progress
  const fault = frame.fault
  const useFallback = reducedMotion || !hasWebGL()
  const timelineSteps = [...new Map(events.filter(item => item.payload.step).map(item => [item.payload.step, item])).values()]

  return <Box sx={{ mx: { xs: -2, sm: -3 }, mt: { xs: -2, sm: -3 }, minHeight: 'calc(100vh - 64px)', bgcolor: '#eef3f1' }}>
    <Stack direction={{ xs: 'column', md: 'row' }} justifyContent="space-between" spacing={1.5} sx={{ px: { xs: 2, md: 3 }, py: 2, bgcolor: '#fff', borderBottom: 1, borderColor: 'divider' }}>
      <Box><Stack direction="row" spacing={1} alignItems="center"><Typography variant="h1">传代可视化仿真</Typography><Chip size="small" color="info" label="simulation_only" /></Stack><Typography color="text.secondary" mt={0.5}>{run.title} · {id}</Typography></Box>
      <Stack direction="row" spacing={1} alignItems="center"><StatusChip status={run.status} /><Button component={RouterLink} to={`/runs/${encodeURIComponent(id)}`}>查看 Run</Button></Stack>
    </Stack>
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'minmax(0, 1fr) 340px' }, minHeight: { lg: 'calc(100vh - 154px)' } }}>
      <Box sx={{ position: 'relative', minHeight: { xs: 470, lg: 620 }, overflow: 'hidden' }}>
        {useFallback ? <FallbackView snapshot={snapshot} step={step} /> : <SceneBoundary fallback={<FallbackView snapshot={snapshot} step={step} />}><Canvas shadows={!mobile} dpr={mobile ? [1, 1.25] : [1, 1.75]} camera={{ position: [7.8, 6.2, 8.4], fov: 42 }} gl={{ antialias: !mobile, powerPreference: 'high-performance' }}><LabScene snapshot={snapshot} activeDevice={payload.device} fault={fault} /></Canvas></SceneBoundary>}
        <Box sx={{ position: 'absolute', left: 16, bottom: 16, right: 16, bgcolor: 'rgba(255,255,255,.92)', border: 1, borderColor: 'divider', p: 1.25, backdropFilter: 'blur(8px)' }}>
          <Stack direction="row" spacing={1} alignItems="center">
            <IconButton aria-label={paused ? '继续观看' : '暂停观看'} onClick={() => setPaused(value => !value)}>{paused ? <PlayArrowRounded /> : <PauseRounded />}</IconButton>
            <IconButton aria-label="追赶实时位置" onClick={() => { setEventIndex(events.length - 1); setPaused(false) }}><SkipNextRounded /></IconButton>
            {completed && <IconButton aria-label="重新播放" onClick={() => { setEventIndex(0); setPaused(false) }}><ReplayRounded /></IconButton>}
            <LinearProgress variant="determinate" value={Math.round(progress * 100)} sx={{ flex: 1, height: 7 }} />
            <Typography variant="caption" sx={{ minWidth: 42 }}>{Math.round(progress * 100)}%</Typography>
            {completed && <Select size="small" value={speed} onChange={event => setSpeed(Number(event.target.value))}>{[0.5, 1, 2, 4].map(value => <MenuItem key={value} value={value}>{value}×</MenuItem>)}</Select>}
          </Stack>
        </Box>
      </Box>
      <Box sx={{ bgcolor: '#fff', borderLeft: { lg: 1 }, borderTop: { xs: 1, lg: 0 }, borderColor: 'divider', p: 2.5, minWidth: 0 }}>
        <Typography className="eyebrow">实时业务状态</Typography><Typography variant="h2" mt={0.5}>{STEP_LABELS[step || ''] || step || '等待启动'}</Typography>
        <Typography color={fault ? 'error.main' : 'text.secondary'} mt={1}>{payload.message || (completed ? '仿真已结束' : '后台仿真正在推进')}</Typography>
        {!completed && <Alert severity="info" sx={{ mt: 2 }}>可以离开页面，仿真会在后台继续。暂停只影响观看。</Alert>}
        {fault && <Alert severity="error" sx={{ mt: 2 }}>检测到设备故障，系统正在执行安全停机；品系数据不会修改。</Alert>}
        <Divider sx={{ my: 2 }} />
        <Stack spacing={1.25}>
          <Stack direction="row" justifyContent="space-between"><Typography color="text.secondary">目标瓶体积</Typography><Typography fontWeight={750}>{Number(snapshot.target_reactor?.volume_ml || 0).toFixed(1)} mL</Typography></Stack>
          <Stack direction="row" justifyContent="space-between"><Typography color="text.secondary">培养基余量</Typography><Typography fontWeight={750}>{Number(snapshot.media_reservoir?.volume_ml || 0).toFixed(1)} mL</Typography></Stack>
          <Stack direction="row" justifyContent="space-between"><Typography color="text.secondary">吸光度</Typography><Typography fontWeight={750}>{snapshot.spectrophotometer?.absorbance ?? '—'}</Typography></Stack>
          <Stack direction="row" justifyContent="space-between"><Typography color="text.secondary">液体处理器</Typography><Typography fontWeight={750}>{String(snapshot.liquid_handler?.status || 'STANDBY')}</Typography></Stack>
          <Stack direction="row" justifyContent="space-between"><Typography color="text.secondary">培养箱</Typography><Typography fontWeight={750}>{String(snapshot.incubator?.status || 'STANDBY')}</Typography></Stack>
          <Stack direction="row" justifyContent="space-between"><Typography color="text.secondary">最近事件</Typography><Typography variant="caption">{formatDate(selectedEvent?.created_at || run.simulation?.last_event_at)}</Typography></Stack>
        </Stack>
        <Divider sx={{ my: 2 }} />
        <Typography variant="h3" mb={1}>步骤时间轴</Typography>
        <List dense sx={{ maxHeight: 280, overflowY: 'auto' }}>{timelineSteps.map(item => <ListItemButton key={`${item.sequence}-${item.payload.step}`} selected={item.sequence === selectedEvent?.sequence} onClick={() => { setEventIndex(events.findIndex(event => event.sequence === item.sequence)); setPaused(true) }}><ListItemText primary={STEP_LABELS[item.payload.step || ''] || item.payload.step} secondary={`${Math.round(Number(item.payload.progress || 0) * 100)}% · ${item.payload.status || item.event_type}`} /></ListItemButton>)}</List>
        {completed && !fault && <Stack spacing={1} mt={2}><Alert severity="success">仿真已完成，但不代表真实传代已经发生。</Alert><Button variant="contained" component={RouterLink} to={`/runs/${encodeURIComponent(id)}`}>确认已手工完成并更新资源</Button><Button startIcon={<ReplayRounded />} onClick={() => { setEventIndex(0); setPaused(false) }}>重新播放</Button></Stack>}
        {completed && fault && <Button fullWidth component={RouterLink} to={`/runs/${encodeURIComponent(id)}`} startIcon={<FastForwardRounded />} sx={{ mt: 2 }}>查看故障日志</Button>}
      </Box>
    </Box>
  </Box>
}
