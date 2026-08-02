import type { MotionCommand, MotionPhase, SimulationEvent, SimulationSnapshot } from './types'

export type Vec3Tuple = [number, number, number]

export interface MotionTimelineEntry {
  event: SimulationEvent
  eventIndex: number
  command: MotionCommand
  startMs: number
  endMs: number
}

export interface SimulationMotionFrame {
  entry?: MotionTimelineEntry
  event?: SimulationEvent
  eventIndex: number
  command?: MotionCommand
  phase?: MotionPhase
  phaseIndex: number
  phaseProgress: number
  easedPhaseProgress: number
  commandProgress: number
  playheadMs: number
  totalDurationMs: number
  snapshot: SimulationSnapshot
  attachmentParent?: string
}

const LEGACY_DURATIONS: Record<string, number> = {
  pick_labware: 2160,
  transport_labware: 2420,
  place_labware: 1920,
  prepare_measurement_plate: 1800,
  open_door: 1260,
  close_door: 1260,
  blank_plate: 2600,
  measure_plate_absorbance: 2400,
  dispense_medium: 720,
  transfer_seed: 680,
  mix_and_seal: 1400,
  configure: 1800,
  safe_shutdown: 2400,
  return_home: 1200,
  cleanup: 1200,
}

export function clamp01(value: number) {
  return Math.max(0, Math.min(Number.isFinite(value) ? value : 0, 1))
}

/** Quintic minimum-jerk interpolation with zero velocity and acceleration at both ends. */
export function minimumJerk(value: number) {
  const t = clamp01(value)
  return t * t * t * (10 + t * (-15 + t * 6))
}

/** Symmetric acceleration/deceleration profile for long transfers. */
export function sCurve(value: number) {
  const t = clamp01(value)
  if (t < 0.25) return 0.5 * minimumJerk(t / 0.25) * 0.25
  if (t > 0.75) return 0.875 + 0.125 * minimumJerk((t - 0.75) / 0.25)
  return 0.125 + (t - 0.25) * 1.5
}

export function applyMotionEasing(easing: string | undefined, value: number) {
  if (easing === 'hold') return value >= 1 ? 1 : 0
  if (easing === 'linear_cruise') return clamp01(value)
  if (easing === 's_curve') return sCurve(value)
  return minimumJerk(value)
}

function legacyPhases(action: string, durationMs: number): MotionPhase[] {
  const duration = Math.max(160, durationMs)
  if (action === 'transport_labware') {
    return [
      { name: 'accelerate', duration_ms: Math.round(duration * 0.28), easing: 's_curve' },
      { name: 'cruise', duration_ms: Math.round(duration * 0.44), easing: 'linear_cruise' },
      { name: 'decelerate', duration_ms: Math.round(duration * 0.28), easing: 's_curve' },
    ]
  }
  if (action === 'pick_labware') {
    return [
      { name: 'approach', duration_ms: Math.round(duration * 0.32), easing: 'minimum_jerk' },
      { name: 'fine_align', duration_ms: Math.round(duration * 0.18), easing: 'minimum_jerk' },
      { name: 'grip_close', duration_ms: Math.round(duration * 0.12), easing: 'minimum_jerk' },
      { name: 'grip_verify', duration_ms: Math.round(duration * 0.14), easing: 'hold' },
      { name: 'lift', duration_ms: Math.round(duration * 0.24), easing: 'minimum_jerk' },
    ]
  }
  if (action === 'place_labware') {
    return [
      { name: 'target_align', duration_ms: Math.round(duration * 0.22), easing: 'minimum_jerk' },
      { name: 'lower', duration_ms: Math.round(duration * 0.24), easing: 'minimum_jerk' },
      { name: 'seat_verify', duration_ms: Math.round(duration * 0.14), easing: 'hold' },
      { name: 'release', duration_ms: Math.round(duration * 0.12), easing: 'minimum_jerk' },
      { name: 'retreat', duration_ms: Math.round(duration * 0.28), easing: 'minimum_jerk' },
    ]
  }
  return [{ name: action || 'state_transition', duration_ms: duration, easing: 'minimum_jerk' }]
}

export function commandForEvent(event: SimulationEvent, fallbackStartMs = 0): MotionCommand {
  if (event.payload.motion_command) return event.payload.motion_command
  const action = String(event.payload.action || 'state_transition')
  const duration = LEGACY_DURATIONS[action] || 520
  const motion = event.payload.motion || event.payload.snapshot?.transport || {}
  return {
    schema_version: 'legacy-adapter-v1',
    command_id: `legacy-${event.sequence}`,
    actor_id: String(event.payload.device || 'workcell'),
    entity_id: motion.labware_id || undefined,
    action_type: action,
    simulation_start_ms: fallbackStartMs,
    nominal_duration_ms: duration,
    required_resources: [String(event.payload.device || 'workcell')],
    preconditions: [],
    postconditions: [],
    source_location: motion.source_location,
    target_location: motion.target_location,
    phases: legacyPhases(action, duration),
    interruptibility: 'phase_boundary',
  }
}

export function buildMotionTimeline(events: SimulationEvent[]): MotionTimelineEntry[] {
  let cursor = 0
  return events.map((event, eventIndex) => {
    const command = commandForEvent(event, cursor)
    const requestedStart = Number(command.simulation_start_ms)
    const startMs = Number.isFinite(requestedStart) && requestedStart >= cursor ? requestedStart : cursor
    const phaseDuration = command.phases.reduce((sum, phase) => sum + Math.max(0, Number(phase.duration_ms) || 0), 0)
    const duration = Math.max(120, phaseDuration || Number(command.nominal_duration_ms) || 0)
    const normalizedCommand = { ...command, simulation_start_ms: startMs, nominal_duration_ms: duration }
    const entry = { event, eventIndex, command: normalizedCommand, startMs, endMs: startMs + duration }
    cursor = entry.endMs
    return entry
  })
}

export function totalTimelineDuration(timeline: MotionTimelineEntry[]) {
  return timeline.length ? timeline[timeline.length - 1].endMs : 0
}

function phaseAt(command: MotionCommand, localMs: number) {
  let cursor = 0
  const phases = command.phases.length ? command.phases : legacyPhases(command.action_type, command.nominal_duration_ms)
  for (let index = 0; index < phases.length; index += 1) {
    const phase = phases[index]
    const duration = Math.max(1, Number(phase.duration_ms) || 1)
    if (localMs <= cursor + duration || index === phases.length - 1) {
      const progress = clamp01((localMs - cursor) / duration)
      return { phase, phaseIndex: index, phaseProgress: progress, elapsedBefore: cursor }
    }
    cursor += duration
  }
  return { phase: phases[phases.length - 1], phaseIndex: phases.length - 1, phaseProgress: 1, elapsedBefore: cursor }
}

export function motionFrameAt(
  timeline: MotionTimelineEntry[],
  playheadMs: number,
  fallbackSnapshot: SimulationSnapshot = {},
): SimulationMotionFrame {
  const totalDurationMs = totalTimelineDuration(timeline)
  const safeTime = Math.max(0, Math.min(playheadMs, totalDurationMs || playheadMs))
  if (!timeline.length) {
    return { eventIndex: -1, phaseIndex: -1, phaseProgress: 0, easedPhaseProgress: 0, commandProgress: 0, playheadMs: safeTime, totalDurationMs, snapshot: fallbackSnapshot }
  }
  const entry = timeline.find(item => safeTime < item.endMs) || timeline[timeline.length - 1]
  const localMs = Math.max(0, Math.min(safeTime - entry.startMs, entry.command.nominal_duration_ms))
  const selected = phaseAt(entry.command, localMs)
  const baseSnapshot = entry.event.payload.snapshot || fallbackSnapshot
  let attachmentParent = baseSnapshot.attachments?.[String(entry.command.entity_id || '')]
  const transition = entry.command.attachment_transition
  if (transition) {
    const transitionIndex = entry.command.phases.findIndex(phase => phase.name === transition.phase)
    const transitionReached = selected.phaseIndex > transitionIndex || (selected.phaseIndex === transitionIndex && selected.phaseProgress >= 1)
    attachmentParent = transitionReached ? transition.to_parent : transition.from_parent || attachmentParent
  }
  return {
    entry,
    event: entry.event,
    eventIndex: entry.eventIndex,
    command: entry.command,
    phase: selected.phase,
    phaseIndex: selected.phaseIndex,
    phaseProgress: selected.phaseProgress,
    easedPhaseProgress: applyMotionEasing(selected.phase.easing, selected.phaseProgress),
    commandProgress: clamp01(localMs / Math.max(1, entry.command.nominal_duration_ms)),
    playheadMs: safeTime,
    totalDurationMs,
    snapshot: baseSnapshot,
    attachmentParent,
  }
}

export function advanceSimulationClock(
  currentMs: number,
  deltaMs: number,
  speed: number,
  totalMs: number,
  paused: boolean,
) {
  if (paused || totalMs <= 0) return Math.max(0, Math.min(currentMs, totalMs || currentMs))
  return Math.max(0, Math.min(currentMs + Math.max(0, deltaMs) * Math.max(0, speed), totalMs))
}

export function timelineProgress(playheadMs: number, totalMs: number) {
  return totalMs > 0 ? clamp01(playheadMs / totalMs) : 0
}

export function lerpVec3(source: Vec3Tuple, target: Vec3Tuple, progress: number): Vec3Tuple {
  const eased = clamp01(progress)
  return source.map((value, index) => value + (target[index] - value) * eased) as Vec3Tuple
}

export function safeTransferPosition(source: Vec3Tuple, target: Vec3Tuple, progress: number): Vec3Tuple {
  const t = clamp01(progress)
  const clearance = Math.max(source[1], target[1]) + 1.15
  if (t < 0.22) return lerpVec3(source, [source[0], clearance, source[2]], minimumJerk(t / 0.22))
  if (t < 0.78) return lerpVec3([source[0], clearance, source[2]], [target[0], clearance, target[2]], sCurve((t - 0.22) / 0.56))
  return lerpVec3([target[0], clearance, target[2]], target, minimumJerk((t - 0.78) / 0.22))
}

