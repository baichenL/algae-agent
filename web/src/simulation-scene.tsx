import { OrbitControls } from '@react-three/drei'
import { useFrame, useThree } from '@react-three/fiber'
import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import type { SimulationEvent, SimulationSnapshot } from './types'
import {
  applyMotionEasing,
  lerpVec3,
  minimumJerk,
  safeTransferPosition,
  sCurve,
  type SimulationMotionFrame,
  type Vec3Tuple,
} from './simulation-motion'
import { DEVICE_TARGETS, LOCATION_POSITIONS, SCENE_DEVICES, type SceneDeviceId } from './simulation-config'
import {
  AnalyticalIsland,
  CultureFlask,
  FixedRobotArm,
  IncubatorUnit,
  LaboratoryRoom,
  LiquidHandlerUnit,
  MeasurementPlate,
  MobileRobotUnit,
  PlateCentrifugeUnit,
  PlateReaderUnit,
  RefrigeratedCentrifugeUnit,
  SampleStackUnit,
  ShakerUnit,
  SonicatorUnit,
  SpectrophotometerUnit,
  type RobotJoints,
} from './simulation-devices'

export { SCENE_DEVICES }
export type { SceneDeviceId }
type Position = Vec3Tuple

const DEVICE_ALIASES: Record<string, SceneDeviceId> = {
  incubator: 'incubator', liquid_handler: 'liquid_handler', media_pump: 'liquid_handler',
  seed_pump: 'liquid_handler', target_valve: 'liquid_handler', material_sensors: 'liquid_handler',
  spectrophotometer: 'plate_reader', plate_reader: 'plate_reader', robot_arm: 'robot_arm',
  human_operator: 'robot_arm', mobile_robot: 'mobile_robot', workcell: 'liquid_handler',
  safety_controller: 'robot_arm',
}

export function resolveActiveSceneDevice(device?: string, action?: string, step?: string): SceneDeviceId | undefined {
  if (device && DEVICE_ALIASES[device]) return DEVICE_ALIASES[device]
  if (action?.includes('plate')) return action === 'prepare_measurement_plate' ? 'liquid_handler' : 'plate_reader'
  if (step?.includes('Incubator')) return 'incubator'
  if (step?.includes('MeasurementPlate') || step?.includes('LiquidHandler')) return 'liquid_handler'
  if (step?.includes('Spectrophotometer') || step?.includes('Absorbance')) return 'plate_reader'
  return undefined
}

export function sceneLocationPosition(location?: string | null): Position {
  return LOCATION_POSITIONS[String(location || 'manual_zone')] || LOCATION_POSITIONS.manual_zone
}

function commandTransferProgress(frame?: SimulationMotionFrame) {
  if (!frame?.command) return undefined
  const action = frame.command.action_type
  if (action === 'pick_labware') return frame.commandProgress * 0.2
  if (action === 'transport_labware') return 0.2 + frame.commandProgress * 0.6
  if (action === 'place_labware') return 0.8 + frame.commandProgress * 0.2
  return undefined
}

export function deriveSceneState(
  snapshot: SimulationSnapshot,
  payload: SimulationEvent['payload'] = {},
  motionFrame?: SimulationMotionFrame,
) {
  const command = motionFrame?.command || payload.motion_command
  const transport = payload.motion || snapshot.transport || {}
  const sourceLocation = command?.source_location || transport.source_location
  const targetLocation = command?.target_location || transport.target_location
  const activeDevice = resolveActiveSceneDevice(payload.device || command?.actor_id, payload.action || command?.action_type, payload.step)
  const commandProgress = commandTransferProgress(motionFrame)
  const progress = Math.max(0, Math.min(Number(commandProgress ?? transport.progress ?? payload.step_progress ?? 0), 1))
  const source = sceneLocationPosition(sourceLocation)
  const target = sceneLocationPosition(targetLocation)
  return {
    activeDevice,
    transport,
    command,
    attachmentParent: motionFrame?.attachmentParent,
    transportPosition: safeTransferPosition(source, target, progress),
    cameraTarget: activeDevice ? DEVICE_TARGETS[activeDevice] : ([0.2, 1, 0] as Position),
  }
}

const HOME_JOINTS: RobotJoints = { base: 0, shoulder: -0.3, elbow: -0.52, wrist: 0.22, gripper: 1 }

function interpolateJoints(source: RobotJoints, target: RobotJoints, progress: number): RobotJoints {
  const t = minimumJerk(progress)
  return {
    base: source.base + (target.base - source.base) * t,
    shoulder: source.shoulder + (target.shoulder - source.shoulder) * t,
    elbow: source.elbow + (target.elbow - source.elbow) * t,
    wrist: source.wrist + (target.wrist - source.wrist) * t,
    gripper: source.gripper + (target.gripper - source.gripper) * t,
  }
}

export function robotJointsForFrame(frame?: SimulationMotionFrame): RobotJoints {
  if (!frame?.command || !['robot_arm', 'mobile_robot', 'human_operator'].includes(frame.command.actor_id)) return HOME_JOINTS
  const action = frame.command.action_type
  const phase = frame.phase?.name || ''
  const sourceWork: RobotJoints = { base: -0.72, shoulder: -0.02, elbow: -0.88, wrist: 0.48, gripper: 1 }
  const targetWork: RobotJoints = { base: 0.74, shoulder: 0.02, elbow: -0.92, wrist: 0.5, gripper: 0 }
  if (action === 'pick_labware') {
    const work = interpolateJoints(HOME_JOINTS, sourceWork, Math.min(1, frame.commandProgress * 1.8))
    if (phase === 'grip_close' || phase === 'grip_verify' || phase === 'lift') {
      return { ...work, gripper: phase === 'grip_close' ? 1 - frame.easedPhaseProgress : 0 }
    }
    return work
  }
  if (action === 'transport_labware') return interpolateJoints({ ...sourceWork, gripper: 0 }, targetWork, frame.commandProgress)
  if (action === 'place_labware') {
    const grip = phase === 'release' || phase === 'release_verify' || phase === 'retreat' ? frame.easedPhaseProgress : 0
    return { ...targetWork, gripper: grip }
  }
  if (action === 'return_home' || action.includes('load_') || action === 'move_to_incubator' || action === 'store_measurement_plate') {
    return interpolateJoints({ ...targetWork, gripper: 1 }, HOME_JOINTS, frame.commandProgress)
  }
  if (action === 'safe_shutdown') return interpolateJoints(targetWork, { ...HOME_JOINTS, shoulder: -0.12 }, frame.commandProgress)
  return HOME_JOINTS
}

function liquidHeadPosition(frame?: SimulationMotionFrame): Position {
  if (!frame?.command || !['liquid_handler', 'media_pump', 'seed_pump'].includes(frame.command.actor_id)) return [0, 0, 0]
  const phase = frame.phase?.name || ''
  const eased = frame.easedPhaseProgress
  const positions: Record<string, Position> = {
    pick_tip: [-1.12, 0.35, -0.08], approach_source: [-0.72, 0.32, -0.08], lower_to_liquid: [-0.72, -0.02, -0.08],
    aspirate: [-0.72, -0.02, -0.08], aspirate_dwell: [-0.72, -0.02, -0.08], retract_vertical: [-0.72, 0.34, -0.08],
    move_above_well: [0.15, 0.34, 0.05], lower_to_well: [0.15, -0.02, 0.05], dispense: [0.15, -0.02, 0.05],
    dispense_dwell: [0.15, -0.02, 0.05], approach_target: [0.42, 0.34, 0.02], lower_to_dispense: [0.42, -0.04, 0.02],
    metered_dispense: [0.42, -0.04, 0.02], open_valve: [0.42, -0.04, 0.02], close_valve: [0.42, -0.04, 0.02],
    drip_dwell: [0.42, -0.04, 0.02], move_to_target: [0.42, 0.32, 0.02], dispense_seed: [0.42, -0.04, 0.02],
  }
  const target = positions[phase] || [0, 0, 0]
  return lerpVec3([0, 0, 0], target, eased)
}

function plateReaderMotion(frame?: SimulationMotionFrame) {
  const action = frame?.command?.action_type
  const phase = frame?.phase?.name || ''
  const t = frame?.easedPhaseProgress || 0
  if (action === 'open_door') return { door: t, tray: t }
  if (action === 'close_door') return { door: 1 - t, tray: 1 - t }
  if (action === 'blank_plate' && phase === 'tray_retract') return { door: 0, tray: 1 - t }
  return { door: 0, tray: 0 }
}

function mobileRobotPosition(frame: SimulationMotionFrame | undefined, snapshot: SimulationSnapshot): Position {
  const command = frame?.command
  if (!command || command.actor_id !== 'mobile_robot' || command.action_type !== 'transport_labware') {
    return sceneLocationPosition(snapshot.mobile_robot?.location === 'incubator' ? 'incubator' : 'home').map((value, index) => index === 1 ? 0.28 : value) as Position
  }
  const source = sceneLocationPosition(command.source_location || 'home')
  const target = sceneLocationPosition(command.target_location || 'incubator')
  const start: Position = [source[0], 0.28, source[2]]
  const end: Position = [target[0] + 0.7, 0.28, target[2] + 0.8]
  const waypoint: Position = [(start[0] + end[0]) * 0.5, 0.28, 2.45]
  const t = sCurve(frame.commandProgress)
  return t < 0.5 ? lerpVec3(start, waypoint, t * 2) : lerpVec3(waypoint, end, (t - 0.5) * 2)
}

function CameraDirector({ target, guided, paused, controlsRef }: { target: Position; guided: boolean; paused: boolean; controlsRef: React.MutableRefObject<any> }) {
  const { camera } = useThree()
  useFrame((_, delta) => {
    if (!guided || paused) return
    const guidedTarget = new THREE.Vector3(target[0] * 0.38, 0.72, target[2] * 0.38)
    const desired = guidedTarget.clone().add(new THREE.Vector3(14.4, 8.2, 16.4))
    const cameraBlend = 1 - Math.exp(-delta * 2.4)
    const targetBlend = 1 - Math.exp(-delta * 3.2)
    camera.position.lerp(desired, cameraBlend)
    if (controlsRef.current) {
      controlsRef.current.target.lerp(guidedTarget, targetBlend)
      controlsRef.current.update()
    }
  })
  return null
}

function ManualTransferTray({ position, children }: { position: Position; children?: React.ReactNode }) {
  return <group position={position} name="simulation-manual-carrier">
    <mesh castShadow><boxGeometry args={[1.05, 0.08, 0.72]} /><meshStandardMaterial color="#8f9696" roughness={0.38} metalness={0.62} /></mesh>
    <mesh position={[-0.45, 0.08, 0]}><boxGeometry args={[0.12, 0.16, 0.62]} /><meshStandardMaterial color="#4d6f86" roughness={0.62} /></mesh>
    <mesh position={[0.45, 0.08, 0]}><boxGeometry args={[0.12, 0.16, 0.62]} /><meshStandardMaterial color="#4d6f86" roughness={0.62} /></mesh>
    <group position={[0, 0.18, 0]}>{children}</group>
  </group>
}

export function LabDigitalTwin({
  snapshot, payload, motionFrame, fault, paused, selectedDevice, resetCameraKey, onSelectDevice,
}: {
  snapshot: SimulationSnapshot
  payload: SimulationEvent['payload']
  motionFrame?: SimulationMotionFrame
  fault: boolean
  paused: boolean
  selectedDevice?: SceneDeviceId
  resetCameraKey: number
  onSelectDevice: (id: SceneDeviceId) => void
}) {
  const [guided, setGuided] = useState(true)
  const controlsRef = useRef<any>(null)
  const derived = useMemo(() => deriveSceneState(snapshot, payload, motionFrame), [snapshot, payload, motionFrame])
  const activeDevice = derived.activeDevice
  const focusDevice = selectedDevice || activeDevice
  const cameraTarget = focusDevice ? DEVICE_TARGETS[focusDevice] : derived.cameraTarget
  const faultDevice = fault ? activeDevice : undefined
  const source = snapshot.source_reactor || {}
  const target = snapshot.target_reactor || {}
  const media = snapshot.media_reservoir || {}
  const commandEntity = String(motionFrame?.command?.entity_id || '')
  const commandParent = motionFrame?.attachmentParent
  const plateParent = commandEntity === 'measurement_plate' && commandParent ? commandParent : snapshot.attachments?.measurement_plate
  const targetParent = commandEntity === 'target_flask' && commandParent ? commandParent : snapshot.attachments?.target_flask
  const plateReader = plateReaderMotion(motionFrame)
  const joints = robotJointsForFrame(motionFrame)
  const mobilePosition = mobileRobotPosition(motionFrame, snapshot)
  const manualCarrier = motionFrame?.command?.actor_id === 'human_operator'
  const plateOnRobot = plateParent?.includes('robot_arm.gripper')
  const plateOnReader = plateParent?.includes('plate_reader')
  const plateOnHandler = plateParent?.includes('liquid_handler')
  const plateOnManual = plateParent?.includes('manual_operator')
  const flaskOnMobile = targetParent?.includes('mobile_robot')
  const flaskInIncubator = targetParent?.includes('incubator')

  useEffect(() => setGuided(true), [resetCameraKey, payload.step])

  return <>
    <color attach="background" args={['#dfe6e6']} />
    <ambientLight intensity={0.48} />
    <hemisphereLight color="#fffdf2" groundColor="#71838a" intensity={0.72} />
    <directionalLight castShadow position={[-5, 10, 7]} intensity={1.65} color="#fff7e8" shadow-mapSize={[1024, 1024]} shadow-camera-far={25} />
    <rectAreaLight position={[0, 6.8, 0]} rotation={[-Math.PI / 2, 0, 0]} width={10} height={5} intensity={2.2} color="#fffdf4" />
    <LaboratoryRoom />

    <ShakerUnit active={activeDevice === 'shaker'} fault={faultDevice === 'shaker'} onSelect={onSelectDevice} />
    <IncubatorUnit active={activeDevice === 'incubator'} fault={faultDevice === 'incubator'} doorProgress={motionFrame?.command?.actor_id === 'incubator' ? minimumJerk(motionFrame.commandProgress < 0.5 ? motionFrame.commandProgress * 2 : (1 - motionFrame.commandProgress) * 2) : 0} onSelect={onSelectDevice}>
      {flaskInIncubator && <CultureFlask volume={Number(target.volume_ml || 0)} capacity={Number(target.capacity_ml || 250)} color="#6f9c69" scale={0.62} />}
    </IncubatorUnit>
    <LiquidHandlerUnit active={activeDevice === 'liquid_handler'} fault={faultDevice === 'liquid_handler'} headPosition={liquidHeadPosition(motionFrame)} onSelect={onSelectDevice}>
      {plateOnHandler && <MeasurementPlate filled={Number(snapshot.measurement_plate?.sample_volume_ml || 0) > 0} />}
    </LiquidHandlerUnit>
    <SampleStackUnit position={[-2.25, 1.15, -1.38]} active={activeDevice === 'sample_stack'} onSelect={onSelectDevice} />
    <SampleStackUnit position={[2.25, 1.15, -1.38]} active={activeDevice === 'sample_stack'} onSelect={onSelectDevice} />
    <FixedRobotArm position={[-2.35, 0.3, 0.42]} joints={joints} active={activeDevice === 'robot_arm' || activeDevice === 'liquid_handler'} fault={faultDevice === 'robot_arm'} onSelect={onSelectDevice} />
    <FixedRobotArm position={[2.35, 0.3, 0.42]} joints={joints} active={activeDevice === 'robot_arm' || activeDevice === 'plate_reader'} fault={faultDevice === 'robot_arm'} onSelect={onSelectDevice}>
      {plateOnRobot && <MeasurementPlate filled />}
    </FixedRobotArm>

    <AnalyticalIsland>
      <PlateCentrifugeUnit active={activeDevice === 'plate_centrifuge'} fault={faultDevice === 'plate_centrifuge'} onSelect={onSelectDevice} />
      <SpectrophotometerUnit active={activeDevice === 'spectrophotometer'} fault={faultDevice === 'spectrophotometer'} onSelect={onSelectDevice} />
      <PlateReaderUnit active={activeDevice === 'plate_reader'} fault={faultDevice === 'plate_reader'} trayProgress={plateReader.tray} doorProgress={plateReader.door} onSelect={onSelectDevice}>
        {plateOnReader && <MeasurementPlate filled />}
      </PlateReaderUnit>
      <RefrigeratedCentrifugeUnit active={activeDevice === 'refrigerated_centrifuge'} fault={faultDevice === 'refrigerated_centrifuge'} onSelect={onSelectDevice} />
      <SonicatorUnit active={activeDevice === 'sonicator'} fault={faultDevice === 'sonicator'} onSelect={onSelectDevice} />
      <FixedRobotArm position={[-1.95, 0.88, -0.35]} joints={joints} active={activeDevice === 'robot_arm' || activeDevice === 'plate_reader'} fault={faultDevice === 'robot_arm'} onSelect={onSelectDevice} />
    </AnalyticalIsland>

    <MobileRobotUnit position={mobilePosition} joints={joints} active={activeDevice === 'mobile_robot'} fault={faultDevice === 'mobile_robot'} onSelect={onSelectDevice}>
      {flaskOnMobile && <CultureFlask volume={Number(target.volume_ml || 0)} capacity={Number(target.capacity_ml || 250)} color="#6f9c69" scale={0.62} />}
    </MobileRobotUnit>

    <group position={[-0.72, 1.28, -0.15]}><CultureFlask volume={Number(source.volume_ml ?? 200)} capacity={Number(source.capacity_ml ?? 250)} color="#5e8f65" scale={0.72} /></group>
    {!flaskOnMobile && !flaskInIncubator && <group position={[0.15, 1.28, -0.1]}><CultureFlask volume={Number(target.volume_ml ?? 0)} capacity={Number(target.capacity_ml ?? 250)} color="#6f9c69" scale={0.72} /></group>}
    <group position={[0.9, 1.25, -0.12]}><CultureFlask volume={Number(media.volume_ml ?? 1000)} capacity={Number(media.capacity_ml ?? 1000)} color="#b7a96a" scale={0.66} /></group>
    {!plateOnRobot && !plateOnReader && !plateOnHandler && !plateOnManual && <group position={sceneLocationPosition(String(snapshot.measurement_plate?.location || 'plate_stack'))}><MeasurementPlate filled={Number(snapshot.measurement_plate?.sample_volume_ml || 0) > 0} /></group>}
    {manualCarrier && (plateOnManual || commandEntity === 'measurement_plate') && <ManualTransferTray position={derived.transportPosition}><MeasurementPlate filled /></ManualTransferTray>}

    <CameraDirector target={cameraTarget} guided={guided} paused={paused} controlsRef={controlsRef} />
    <OrbitControls ref={controlsRef} makeDefault enableDamping dampingFactor={0.08} enablePan={false} minDistance={6.5} maxDistance={21} minPolarAngle={0.55} maxPolarAngle={1.42} onStart={() => setGuided(false)} />
  </>
}
