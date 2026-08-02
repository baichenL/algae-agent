import { RoundedBox } from '@react-three/drei'
import type { ReactNode } from 'react'
import type { Vec3Tuple } from './simulation-motion'
import type { SceneDeviceId } from './simulation-config'

const paintedMetal = { color: '#eef1f2', roughness: 0.4, metalness: 0.22 }
const darkWorktop = { color: '#252a2e', roughness: 0.32, metalness: 0.45 }
const stainless = { color: '#aeb6b9', roughness: 0.28, metalness: 0.78 }
const polymer = { color: '#e5e8e9', roughness: 0.48, metalness: 0.04 }

type DeviceProps = {
  active?: boolean
  fault?: boolean
  onSelect: (id: SceneDeviceId) => void
}

export function StatusLamp({ active, fault, position }: { active?: boolean; fault?: boolean; position: Vec3Tuple }) {
  const color = fault ? '#b32632' : active ? '#2f7f54' : '#6e7d77'
  return <mesh position={position}>
    <cylinderGeometry args={[0.035, 0.035, 0.018, 16]} />
    <meshStandardMaterial color={color} emissive={color} emissiveIntensity={active || fault ? 0.7 : 0.06} />
  </mesh>
}

export function LaboratoryRoom() {
  return <group name="simulation-laboratory-room">
    <mesh position={[0, -0.14, 0]} receiveShadow>
      <boxGeometry args={[17.5, 0.25, 9.2]} />
      <meshStandardMaterial color="#7f9ca8" roughness={0.8} metalness={0.02} />
    </mesh>
    <mesh position={[0, 3.8, -4.55]} receiveShadow>
      <boxGeometry args={[17.5, 7.8, 0.18]} />
      <meshStandardMaterial color="#e9ece9" roughness={0.9} />
    </mesh>
    <mesh position={[-8.65, 3.8, 0]} receiveShadow>
      <boxGeometry args={[0.18, 7.8, 9.2]} />
      <meshStandardMaterial color="#e5e8e5" roughness={0.9} />
    </mesh>
    <mesh position={[0, 7.55, 0]} receiveShadow>
      <boxGeometry args={[17.5, 0.15, 9.2]} />
      <meshStandardMaterial color="#f3f3f0" roughness={0.92} />
    </mesh>
    {[[-5.5, -2.1], [-1.9, -2.1], [1.8, -2.1], [5.5, -2.1], [-3.7, 1.8], [0, 1.8], [3.7, 1.8]].map(([x, z], index) =>
      <group key={index} position={[x, 7.42, z]}>
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <planeGeometry args={[2.35, 0.58]} />
          <meshStandardMaterial color="#faf8ee" emissive="#fff9de" emissiveIntensity={0.38} />
        </mesh>
      </group>)}
    {Array.from({ length: 12 }, (_, index) => <mesh key={index} position={[-7.4 + index * 1.35, 0.005, 1.8]}>
      <boxGeometry args={[0.025, 0.012, 8.2]} />
      <meshStandardMaterial color="#668996" roughness={0.9} />
    </mesh>)}
  </group>
}

export function CultureFlask({ volume, capacity, color, scale = 1 }: { volume: number; capacity: number; color: string; scale?: number }) {
  const level = Math.max(0.02, Math.min(volume / Math.max(capacity, 1), 1))
  return <group scale={scale} name="simulation-culture-flask">
    <mesh castShadow><coneGeometry args={[0.34, 0.18, 0.72, 32]} /><meshPhysicalMaterial color="#e9f0f1" transparent opacity={0.38} transmission={0.5} thickness={0.03} roughness={0.22} /></mesh>
    <mesh position={[0, -0.31 + level * 0.3, 0]}><cylinderGeometry args={[0.17 + level * 0.11, 0.3, Math.max(0.035, level * 0.54), 32]} /><meshStandardMaterial color={color} transparent opacity={0.74} roughness={0.55} /></mesh>
    <mesh position={[0, 0.48, 0]}><cylinderGeometry args={[0.11, 0.14, 0.3, 24]} /><meshPhysicalMaterial color="#e4ebec" transparent opacity={0.5} transmission={0.4} /></mesh>
    <mesh position={[0, 0.67, 0]}><cylinderGeometry args={[0.14, 0.14, 0.1, 24]} /><meshStandardMaterial color="#75868c" roughness={0.65} /></mesh>
  </group>
}

export function MeasurementPlate({ filled = true }: { filled?: boolean }) {
  return <group name="simulation-measurement-plate">
    <RoundedBox args={[0.78, 0.09, 0.52]} radius={0.035} smoothness={2} castShadow>
      <meshStandardMaterial color="#e6e9e8" roughness={0.5} />
    </RoundedBox>
    {Array.from({ length: 96 }, (_, index) => {
      const col = index % 12
      const row = Math.floor(index / 12)
      const occupied = filled && (index === 0 || index === 1)
      return <mesh key={index} position={[-0.33 + col * 0.06, 0.057, -0.21 + row * 0.06]}>
        <cylinderGeometry args={[0.017, 0.017, 0.026, 10]} />
        <meshStandardMaterial color={occupied ? (index === 0 ? '#568760' : '#b5a35c') : '#9aa5a6'} roughness={0.62} />
      </mesh>
    })}
  </group>
}

export function ShakerUnit({ active, fault, onSelect }: DeviceProps) {
  return <group position={[-6.05, 0, 1.12]} onClick={() => onSelect('shaker')} name="simulation-model-shaker">
    <RoundedBox args={[1.45, 1.35, 1.25]} radius={0.08} smoothness={3} position={[0, 0.68, 0]} castShadow><meshStandardMaterial {...paintedMetal} color={fault ? '#ad7f82' : paintedMetal.color} /></RoundedBox>
    <RoundedBox args={[1.34, 0.68, 1.02]} radius={0.08} smoothness={3} position={[0, 1.5, 0]} castShadow>
      <meshPhysicalMaterial color="#d9dfe0" roughness={0.25} metalness={0.1} />
    </RoundedBox>
    <mesh position={[0, 1.52, 0.521]}><planeGeometry args={[1.06, 0.48]} /><meshPhysicalMaterial color="#8ea1a6" transparent opacity={0.58} transmission={0.25} roughness={0.18} /></mesh>
    <mesh position={[0, 1.25, 0]}><boxGeometry args={[0.92, 0.05, 0.72]} /><meshStandardMaterial {...stainless} /></mesh>
    <mesh position={[0.42, 0.42, 0.64]}><planeGeometry args={[0.35, 0.18]} /><meshStandardMaterial color="#323a3d" roughness={0.28} /></mesh>
    <StatusLamp active={active} fault={fault} position={[0.57, 0.2, 0.64]} />
  </group>
}

export function IncubatorUnit({ active, fault, doorProgress = 0, onSelect, children }: DeviceProps & { doorProgress?: number; children?: ReactNode }) {
  return <group position={[-5.02, 0, -1.38]} onClick={() => onSelect('incubator')} name="simulation-model-incubator">
    <RoundedBox args={[1.55, 2.75, 1.35]} radius={0.07} smoothness={3} position={[0, 1.38, 0]} castShadow><meshStandardMaterial {...paintedMetal} color={fault ? '#ad7f82' : paintedMetal.color} /></RoundedBox>
    <mesh position={[0, 1.55, 0.69]}><planeGeometry args={[1.22, 1.78]} /><meshPhysicalMaterial color="#718287" transparent opacity={0.48} transmission={0.28} roughness={0.2} /></mesh>
    {[-0.48, 0, 0.48].map(y => <mesh key={y} position={[0, 1.52 + y, 0.2]}><boxGeometry args={[1.15, 0.035, 0.9]} /><meshStandardMaterial {...stainless} /></mesh>)}
    <group position={[-0.7, 1.5, 0.72]} rotation={[0, -doorProgress * 1.25, 0]}>
      <mesh position={[0.7, 0, 0]}><boxGeometry args={[1.42, 2.05, 0.055]} /><meshPhysicalMaterial color="#dce2e2" transparent opacity={0.22} transmission={0.48} roughness={0.15} /></mesh>
      <mesh position={[1.34, 0, 0.05]}><boxGeometry args={[0.055, 1.3, 0.08]} /><meshStandardMaterial color="#464e50" /></mesh>
    </group>
    <StatusLamp active={active} fault={fault} position={[0.62, 0.28, 0.7]} />
    <group position={[0, 1.25, 0.18]}>{children}</group>
  </group>
}

export function SampleStackUnit({ position, active, onSelect }: DeviceProps & { position: Vec3Tuple }) {
  return <group position={position} onClick={() => onSelect('sample_stack')} name="simulation-model-sample-stack">
    <mesh castShadow><boxGeometry args={[0.78, 1.85, 0.74]} /><meshStandardMaterial color="#24292b" metalness={0.38} roughness={0.45} /></mesh>
    {[-0.65, -0.35, -0.05, 0.25, 0.55].map(y => <group key={y} position={[0, y, 0.39]}>
      <mesh><boxGeometry args={[0.66, 0.045, 0.08]} /><meshStandardMaterial {...stainless} /></mesh>
      <mesh position={[0.29, 0.08, 0]}><boxGeometry args={[0.025, 0.05, 0.02]} /><meshStandardMaterial color={active ? '#4b8a68' : '#6c7576'} /></mesh>
    </group>)}
  </group>
}

export function LiquidHandlerUnit({ active, fault, headPosition = [0, 0, 0], onSelect, children }: DeviceProps & { headPosition?: Vec3Tuple; children?: ReactNode }) {
  return <group position={[0, 0, -0.72]} onClick={() => onSelect('liquid_handler')} name="simulation-model-liquid-handler">
    <RoundedBox args={[3.6, 0.9, 1.7]} radius={0.07} smoothness={3} position={[0, 0.45, 0]} castShadow><meshStandardMaterial {...paintedMetal} color={fault ? '#ad7f82' : paintedMetal.color} /></RoundedBox>
    <mesh position={[0, 0.94, 0]} receiveShadow><boxGeometry args={[3.42, 0.12, 1.56]} /><meshStandardMaterial {...darkWorktop} /></mesh>
    <mesh position={[0, 2.05, -0.72]}><boxGeometry args={[3.5, 2.15, 0.1]} /><meshStandardMaterial {...paintedMetal} /></mesh>
    <mesh position={[-1.7, 2.05, 0]}><boxGeometry args={[0.1, 2.18, 1.52]} /><meshStandardMaterial {...paintedMetal} /></mesh>
    <mesh position={[1.7, 2.05, 0]}><boxGeometry args={[0.1, 2.18, 1.52]} /><meshStandardMaterial {...paintedMetal} /></mesh>
    <mesh position={[0, 3.15, 0]}><boxGeometry args={[3.5, 0.22, 1.56]} /><meshStandardMaterial {...paintedMetal} /></mesh>
    <mesh position={[0, 2.05, 0.78]}><planeGeometry args={[3.3, 1.92]} /><meshPhysicalMaterial color="#b8c5c7" transparent opacity={0.2} transmission={0.58} roughness={0.12} /></mesh>
    <mesh position={[0, 2.6, -0.2]}><boxGeometry args={[2.9, 0.07, 0.08]} /><meshStandardMaterial {...stainless} /></mesh>
    <group position={[headPosition[0], 1.82 + headPosition[1], headPosition[2]]} name="simulation-liquid-head">
      <RoundedBox args={[0.7, 0.32, 0.54]} radius={0.05} smoothness={2}><meshStandardMaterial {...polymer} /></RoundedBox>
      {[-0.2, -0.065, 0.065, 0.2].map(x => <mesh key={x} position={[x, -0.28, 0]}><cylinderGeometry args={[0.024, 0.012, 0.38, 12]} /><meshStandardMaterial color="#6f91a0" roughness={0.5} /></mesh>)}
    </group>
    <mesh position={[-1.15, 1.05, -0.1]}><boxGeometry args={[0.65, 0.09, 0.52]} /><meshStandardMaterial color="#a8afb0" /></mesh>
    <mesh position={[1.12, 1.08, -0.05]}><cylinderGeometry args={[0.28, 0.28, 0.2, 28]} /><meshStandardMaterial {...stainless} /></mesh>
    <StatusLamp active={active} fault={fault} position={[1.52, 0.72, 0.86]} />
    <group position={[0.1, 1.18, 0]}>{children}</group>
  </group>
}

export interface RobotJoints { base: number; shoulder: number; elbow: number; wrist: number; gripper: number }

export function FixedRobotArm({ position, joints, active, fault, onSelect, children, scale = 1 }: DeviceProps & { position: Vec3Tuple; joints: RobotJoints; children?: ReactNode; scale?: number }) {
  return <group position={position} scale={scale} onClick={() => onSelect('robot_arm')} name="simulation-model-fixed-robot">
    <mesh castShadow><cylinderGeometry args={[0.34, 0.42, 0.34, 32]} /><meshStandardMaterial {...paintedMetal} color={fault ? '#ad7f82' : paintedMetal.color} /></mesh>
    <group position={[0, 0.28, 0]} rotation={[0, joints.base, 0]}>
      <mesh><sphereGeometry args={[0.22, 24, 24]} /><meshStandardMaterial color="#718da5" roughness={0.4} /></mesh>
      <group rotation={[0, 0, joints.shoulder]}>
        <mesh position={[0, 0.5, 0]}><capsuleGeometry args={[0.14, 0.72, 12, 20]} /><meshStandardMaterial {...paintedMetal} /></mesh>
        <group position={[0, 0.95, 0]} rotation={[0, 0, joints.elbow]}>
          <mesh><sphereGeometry args={[0.19, 24, 24]} /><meshStandardMaterial color="#718da5" roughness={0.4} /></mesh>
          <mesh position={[0, 0.43, 0]}><capsuleGeometry args={[0.115, 0.64, 12, 20]} /><meshStandardMaterial {...paintedMetal} /></mesh>
          <group position={[0, 0.86, 0]} rotation={[0, 0, joints.wrist]} name="simulation-robot-gripper">
            <mesh><cylinderGeometry args={[0.12, 0.15, 0.22, 20]} /><meshStandardMaterial color="#59676b" roughness={0.38} /></mesh>
            {[-1, 1].map(side => <mesh key={side} position={[side * (0.12 + joints.gripper * 0.08), 0.18, 0]}><boxGeometry args={[0.045, 0.28, 0.12]} /><meshStandardMaterial color="#343b3d" /></mesh>)}
            <group position={[0, 0.4, 0]}>{children}</group>
          </group>
        </group>
      </group>
    </group>
    <StatusLamp active={active} fault={fault} position={[0.29, 0.08, 0.2]} />
  </group>
}

export function MobileRobotUnit({ position, joints, active, fault, onSelect, children }: DeviceProps & { position: Vec3Tuple; joints: RobotJoints; children?: ReactNode }) {
  return <group position={position} onClick={() => onSelect('mobile_robot')} name="simulation-model-mobile-robot">
    <RoundedBox args={[1.2, 0.42, 0.92]} radius={0.16} smoothness={4} castShadow><meshStandardMaterial {...polymer} color={fault ? '#ad7f82' : polymer.color} /></RoundedBox>
    <mesh position={[0, -0.22, 0]}><boxGeometry args={[1.24, 0.14, 0.95]} /><meshStandardMaterial color="#2d3335" roughness={0.68} /></mesh>
    {[-0.43, 0.43].map(x => <mesh key={x} position={[x, -0.3, 0.43]}><cylinderGeometry args={[0.12, 0.12, 0.08, 20]} /><meshStandardMaterial color="#303638" /></mesh>)}
    <mesh position={[0, 0.27, 0]}><boxGeometry args={[0.84, 0.08, 0.62]} /><meshStandardMaterial {...stainless} /></mesh>
    <group position={[0, 0.38, 0]}>{children}</group>
    <FixedRobotArm position={[0, 0.3, 0]} scale={0.56} joints={joints} active={active} fault={fault} onSelect={onSelect} />
    <StatusLamp active={active} fault={fault} position={[0.48, -0.03, 0.47]} />
  </group>
}

function IslandBench() {
  return <group>
    <mesh position={[0, 0.27, 0]} receiveShadow><cylinderGeometry args={[3.05, 3.05, 0.75, 72]} /><meshStandardMaterial {...paintedMetal} /></mesh>
    <mesh position={[0, 0.68, 0]}><cylinderGeometry args={[2.92, 2.92, 0.11, 72]} /><meshStandardMaterial {...darkWorktop} /></mesh>
  </group>
}

export function PlateCentrifugeUnit({ active, fault, onSelect }: DeviceProps) {
  return <group position={[-1.36, 1.06, 0.82]} onClick={() => onSelect('plate_centrifuge')} name="simulation-model-plate-centrifuge">
    <RoundedBox args={[1.12, 0.78, 0.9]} radius={0.12} smoothness={3} castShadow><meshStandardMaterial {...polymer} color={fault ? '#ad7f82' : polymer.color} /></RoundedBox>
    <mesh position={[0, 0.42, 0]} rotation={[Math.PI / 2, 0, 0]}><cylinderGeometry args={[0.36, 0.36, 0.06, 32]} /><meshStandardMaterial color="#737d7f" roughness={0.35} /></mesh>
    <mesh position={[0.38, 0.12, 0.46]}><planeGeometry args={[0.24, 0.13]} /><meshStandardMaterial color="#394143" /></mesh>
    <StatusLamp active={active} fault={fault} position={[0.48, -0.2, 0.46]} />
  </group>
}

export function SpectrophotometerUnit({ active, fault, onSelect }: DeviceProps) {
  return <group position={[-0.35, 1.04, -1.15]} onClick={() => onSelect('spectrophotometer')} name="simulation-model-spectrophotometer">
    <RoundedBox args={[1.0, 0.65, 0.78]} radius={0.08} smoothness={3} castShadow><meshStandardMaterial {...polymer} color={fault ? '#ad7f82' : polymer.color} /></RoundedBox>
    <mesh position={[-0.16, 0.36, 0]} rotation={[0, 0, -0.16]}><boxGeometry args={[0.48, 0.08, 0.56]} /><meshStandardMaterial color="#c8cdcd" /></mesh>
    <mesh position={[0.3, 0.05, 0.4]}><planeGeometry args={[0.24, 0.15]} /><meshStandardMaterial color="#374144" /></mesh>
    <StatusLamp active={active} fault={fault} position={[0.42, -0.18, 0.4]} />
  </group>
}

export function PlateReaderUnit({ active, fault, trayProgress = 0, doorProgress = 0, onSelect, children }: DeviceProps & { trayProgress?: number; doorProgress?: number; children?: ReactNode }) {
  return <group position={[0.38, 1.08, -0.05]} onClick={() => onSelect('plate_reader')} name="simulation-model-plate-reader">
    <RoundedBox args={[1.42, 0.86, 1.06]} radius={0.1} smoothness={3} castShadow><meshStandardMaterial {...polymer} color={fault ? '#ad7f82' : polymer.color} /></RoundedBox>
    <mesh position={[0, 0.08 + doorProgress * 0.22, 0.55]}><boxGeometry args={[0.88, 0.28, 0.055]} /><meshStandardMaterial color="#353d40" roughness={0.34} /></mesh>
    <group position={[0, -0.05, 0.26 + trayProgress * 0.72]} name="simulation-plate-reader-tray">
      <mesh><boxGeometry args={[0.86, 0.06, 0.62]} /><meshStandardMaterial {...stainless} /></mesh>
      <group position={[0, 0.09, 0]}>{children}</group>
    </group>
    <mesh position={[0.46, 0.22, 0.54]}><planeGeometry args={[0.28, 0.17]} /><meshStandardMaterial color="#323b3d" /></mesh>
    <StatusLamp active={active} fault={fault} position={[0.58, -0.27, 0.55]} />
  </group>
}

export function RefrigeratedCentrifugeUnit({ active, fault, onSelect }: DeviceProps) {
  return <group position={[1.42, 1.16, 0.92]} onClick={() => onSelect('refrigerated_centrifuge')} name="simulation-model-refrigerated-centrifuge">
    <RoundedBox args={[1.18, 1.1, 0.98]} radius={0.12} smoothness={3} castShadow><meshStandardMaterial {...polymer} color={fault ? '#ad7f82' : polymer.color} /></RoundedBox>
    <mesh position={[0, 0.59, 0]}><cylinderGeometry args={[0.44, 0.44, 0.1, 40]} /><meshStandardMaterial color="#7d8789" roughness={0.42} /></mesh>
    <mesh position={[0.36, 0.18, 0.51]}><planeGeometry args={[0.28, 0.18]} /><meshStandardMaterial color="#344043" /></mesh>
    <StatusLamp active={active} fault={fault} position={[0.5, -0.36, 0.51]} />
  </group>
}

export function SonicatorUnit({ active, fault, onSelect }: DeviceProps) {
  return <group position={[1.6, 1.05, -0.78]} onClick={() => onSelect('sonicator')} name="simulation-model-sonicator">
    <RoundedBox args={[0.86, 0.76, 0.72]} radius={0.06} smoothness={3} castShadow><meshStandardMaterial {...polymer} color={fault ? '#ad7f82' : polymer.color} /></RoundedBox>
    <mesh position={[0, 0.23, 0]}><boxGeometry args={[0.62, 0.24, 0.5]} /><meshPhysicalMaterial color="#7f9196" transparent opacity={0.5} transmission={0.22} /></mesh>
    <mesh position={[0.28, -0.12, 0.37]}><planeGeometry args={[0.2, 0.14]} /><meshStandardMaterial color="#384244" /></mesh>
    <StatusLamp active={active} fault={fault} position={[0.34, -0.27, 0.37]} />
  </group>
}

export function AnalyticalIsland({ children }: { children?: ReactNode }) {
  return <group position={[5.05, 0, -0.08]} name="simulation-analytical-island">
    <IslandBench />
    {children}
  </group>
}
