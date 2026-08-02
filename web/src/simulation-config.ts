import type { Vec3Tuple } from './simulation-motion'

export const SCENE_DEVICES = {
  incubator: { label: '细胞培养箱', role: '恒温、光照培养与传代后恢复' },
  shaker: { label: '恒温摇床', role: '恒温振荡培养与气液交换' },
  liquid_handler: { label: '液体处理工作站', role: '孔板制备、培养基加注与种液转移' },
  sample_stack: { label: '样品堆栈', role: '孔板、耗材与已检测板缓存' },
  robot_arm: { label: '固定式机械臂', role: '工作站与检测岛之间的自动搬运' },
  mobile_robot: { label: '移动机器人', role: '培养瓶跨区域运输' },
  plate_centrifuge: { label: '孔板离心机', role: '孔板快速离心，本流程保持待机' },
  spectrophotometer: { label: '分光光度计', role: '比色皿吸光检测，本流程保持待机' },
  plate_reader: { label: '酶标仪', role: '96 孔板空白校准与 680 nm 吸光读取' },
  refrigerated_centrifuge: { label: '冷冻离心机', role: '低温样品分离，本流程保持待机' },
  sonicator: { label: '超声破碎机', role: '细胞破碎与提取，本流程保持待机' },
} as const

export type SceneDeviceId = keyof typeof SCENE_DEVICES

export const LOCATION_POSITIONS: Record<string, Vec3Tuple> = {
  manual_zone: [-4.3, 0.88, 2.35],
  plate_stack: [2.35, 1.72, -1.5],
  liquid_handler: [0.1, 1.45, -0.45],
  plate_reader: [5.25, 1.34, -0.12],
  completed_plate_stack: [3.0, 1.62, -1.52],
  incubator: [-5.0, 1.15, -1.3],
  home: [-0.75, 0.35, 3.05],
  in_transit: [0, 2.1, 1.25],
}

export const DEVICE_TARGETS: Record<SceneDeviceId, Vec3Tuple> = {
  incubator: [-5.05, 1.1, -1.35],
  shaker: [-6.05, 0.9, 1.15],
  liquid_handler: [0, 1.35, -0.55],
  sample_stack: [2.45, 1.25, -1.5],
  robot_arm: [2.85, 1.15, 0.35],
  mobile_robot: [-0.8, 0.55, 3.0],
  plate_centrifuge: [3.55, 1.15, 0.82],
  spectrophotometer: [4.48, 1.12, -1.22],
  plate_reader: [5.25, 1.15, -0.08],
  refrigerated_centrifuge: [6.12, 1.15, 0.88],
  sonicator: [6.45, 1.12, -0.72],
}

