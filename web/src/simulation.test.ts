import { describe, expect, it } from 'vitest'
import { nextSimulationEventIndex, selectSimulationFrame, supportsSimulationAnimation } from './simulation'
import { deriveSceneState, resolveActiveSceneDevice, sceneLocationPosition } from './simulation-scene'
import type { SimulationEvent } from './types'
import {
  advanceSimulationClock, buildMotionTimeline, minimumJerk, motionFrameAt,
  safeTransferPosition,
} from './simulation-motion'

describe('传代仿真事件映射', () => {
  it('工作流和直接设备仿真共用动画页面', () => {
    expect(supportsSimulationAnimation('workflow')).toBe(true)
    expect(supportsSimulationAnimation('simulation')).toBe(true)
    expect(supportsSimulationAnimation('scientific')).toBe(false)
  })

  it('批量事件从当前游标逐帧推进而不是跳到末尾', () => {
    expect(nextSimulationEventIndex(-1, 4)).toBe(0)
    expect(nextSimulationEventIndex(0, 4)).toBe(1)
    expect(nextSimulationEventIndex(1, 4)).toBe(2)
    expect(nextSimulationEventIndex(3, 4)).toBe(3)
    expect(nextSimulationEventIndex(0, 0)).toBe(-1)
  })

  it('使用所选事件的步骤、进度和硬件快照', () => {
    const events: SimulationEvent[] = [{
      sequence: 7,
      event_type: 'simulation_snapshot',
      phase: 'DispenseMedium',
      level: 'info',
      payload: {
        step: 'DispenseMedium',
        device: 'media_pump',
        progress: 0.55,
        snapshot: { target_reactor: { volume_ml: 72 } },
      },
    }]
    const frame = selectSimulationFrame(events, 0)
    expect(frame.step).toBe('DispenseMedium')
    expect(frame.progress).toBe(0.55)
    expect(frame.snapshot.target_reactor?.volume_ml).toBe(72)
    expect(frame.fault).toBe(false)
  })

  it('故障与安全停机事件会映射为故障画面', () => {
    const frame = selectSimulationFrame([{
      sequence: 9,
      event_type: 'simulation_safe_shutdown',
      level: 'error',
      payload: { step: 'SafeShutdown', snapshot: { alarm: { code: 'MEDIA_PUMP_BLOCKED' } } },
    }], 0)
    expect(frame.fault).toBe(true)
    expect(frame.step).toBe('SafeShutdown')
  })

  it('将瓶板混合步骤确定性映射到真实设备', () => {
    expect(resolveActiveSceneDevice('media_pump', 'dispense_medium', 'DispenseMedium')).toBe('liquid_handler')
    expect(resolveActiveSceneDevice('plate_reader', 'blank_plate', 'BlankSpectrophotometer')).toBe('plate_reader')
    expect(resolveActiveSceneDevice('mobile_robot', 'transport_labware', 'ManualMoveToIncubator')).toBe('mobile_robot')
    expect(resolveActiveSceneDevice(undefined, 'prepare_measurement_plate', 'PrepareMeasurementPlate')).toBe('liquid_handler')
  })

  it('运输路径先抬升至安全高度再移动到目标设备', () => {
    const source = sceneLocationPosition('liquid_handler')
    const target = sceneLocationPosition('plate_reader')
    const scene = deriveSceneState({
      transport: {
        carrier: 'robot_arm', labware_id: 'measurement_plate',
        source_location: 'liquid_handler', target_location: 'plate_reader',
        motion_phase: 'TRANSPORTING', progress: 0.5,
      },
    }, { device: 'robot_arm', action: 'transport_labware' })
    expect(scene.transportPosition[0]).toBeCloseTo((source[0] + target[0]) / 2)
    expect(scene.transportPosition[1]).toBeGreaterThan(Math.max(source[1], target[1]))
    expect(scene.transportPosition[2]).toBeCloseTo((source[2] + target[2]) / 2)
  })

  it('最小跃度曲线在起止点连续且暂停不会推进仿真时钟', () => {
    expect(minimumJerk(0)).toBe(0)
    expect(minimumJerk(1)).toBe(1)
    expect(minimumJerk(0.25)).toBeLessThan(0.25)
    expect(minimumJerk(0.75)).toBeGreaterThan(0.75)
    expect(advanceSimulationClock(800, 500, 2, 4000, true)).toBe(800)
    expect(advanceSimulationClock(800, 500, 2, 4000, false)).toBe(1800)
  })

  it('motion-v2 按阶段切换抓取父级且相同时间得到相同状态', () => {
    const event: SimulationEvent = {
      sequence: 1,
      event_type: 'simulation_snapshot',
      level: 'info',
      payload: {
        action: 'pick_labware',
        device: 'robot_arm',
        snapshot: { attachments: { measurement_plate: 'liquid_handler.plate_slot' } },
        motion_command: {
          schema_version: 'motion-v2', command_id: 'motion-1', actor_id: 'robot_arm', entity_id: 'measurement_plate',
          action_type: 'pick_labware', simulation_start_ms: 0, nominal_duration_ms: 1000,
          required_resources: ['fixed_robot_arm', 'robot_gripper'], preconditions: [], postconditions: [],
          phases: [
            { name: 'approach', duration_ms: 400, easing: 'minimum_jerk' },
            { name: 'grip_verify', duration_ms: 200, easing: 'hold' },
            { name: 'lift', duration_ms: 400, easing: 'minimum_jerk' },
          ],
          attachment_transition: {
            entity_id: 'measurement_plate', phase: 'grip_verify',
            from_parent: 'liquid_handler.plate_slot', to_parent: 'robot_arm.gripper',
          },
        },
      },
    }
    const timeline = buildMotionTimeline([event])
    expect(motionFrameAt(timeline, 450).attachmentParent).toBe('liquid_handler.plate_slot')
    expect(motionFrameAt(timeline, 610).attachmentParent).toBe('robot_arm.gripper')
    expect(motionFrameAt(timeline, 750)).toEqual(motionFrameAt(timeline, 750))
  })

  it('安全运输轨迹不会在设备之间直线穿模', () => {
    const source = sceneLocationPosition('liquid_handler')
    const target = sceneLocationPosition('plate_reader')
    expect(safeTransferPosition(source, target, 0)).toEqual(source)
    expect(safeTransferPosition(source, target, 1)).toEqual(target)
    expect(safeTransferPosition(source, target, 0.5)[1]).toBeGreaterThan(Math.max(source[1], target[1]))
  })

  it('旧快照缺少孔板和机器人字段时仍可生成场景状态', () => {
    const scene = deriveSceneState({
      spectrophotometer: { status: 'READY' },
      sample: { location: 'spectrophotometer' },
    }, { step: 'MeasureAbsorbance', device: 'spectrophotometer' })
    expect(scene.activeDevice).toBe('plate_reader')
    expect(scene.transportPosition).toEqual(sceneLocationPosition('manual_zone'))
  })
})
