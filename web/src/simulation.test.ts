import { describe, expect, it } from 'vitest'
import { selectSimulationFrame } from './simulation'
import type { SimulationEvent } from './types'

describe('传代仿真事件映射', () => {
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
})
