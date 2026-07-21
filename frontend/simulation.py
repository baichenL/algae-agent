from __future__ import annotations

import html
import time

from frontend.http import (
    get_subculture_simulation,
    resolve_subculture_manual_task,
    start_subculture_simulation,
)


FAULT_OPTIONS = {
    "正常流程": None,
    "分光光度计超时": "spectrophotometer_timeout",
    "空白校准失败": "blank_calibration_failure",
    "吸光度读数无效": "invalid_absorbance",
    "移液平台不可用": "liquid_handler_unavailable",
    "培养基不足": "insufficient_medium",
    "吸液/排液失败": "aspirate_dispense_failure",
    "培养箱设置失败": "incubator_setpoint_failure",
}

INTERACTION_MODES = {
    "自动演示": "demo",
    "人工验证": "verification",
}

STEPS = [
    ("CheckSchedule", "计划"),
    ("ManualLoadSpectrophotometer", "分光上样"),
    ("BlankSpectrophotometer", "空白校准"),
    ("MeasureAbsorbance", "OD 测量"),
    ("ManualLoadLiquidHandler", "平台装载"),
    ("LoadMaterials", "耗材检查"),
    ("DispenseMedium", "加培养基"),
    ("TransferSeedCulture", "接种"),
    ("MixAndSeal", "混匀封口"),
    ("ManualMoveToIncubator", "移入培养箱"),
    ("MoveToIncubator", "环境设置"),
    ("RecordExperiment", "记录"),
    ("CleanupWorkspace", "完成"),
]


def _number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _status_text(status: str) -> str:
    return {
        "QUEUED": "排队中",
        "RUNNING": "运行中",
        "WAITING_MANUAL": "等待人工操作",
        "COMPLETED": "已完成",
        "FAILED": "已失败",
        "SKIPPED": "已跳过",
        "SUCCESS": "成功",
    }.get(status, status)


def _flask_markup(x: float, y: float, fill_ratio: float, label: str) -> str:
    liquid_y = 38 - 22 * fill_ratio
    liquid_h = 22 * fill_ratio
    return f"""
      <g class="sample-flask" transform="translate({x:.1f} {y:.1f})">
        <path d="M-8 0 h16 v9 l18 40 q4 10 -7 10 h-38 q-11 0 -7-10 l18-40z"
              fill="#ffffff" stroke="#334155" stroke-width="2"/>
        <path d="M-20 {liquid_y:.1f} q20 5 40 0 l6 {liquid_h:.1f} q3 8 -7 8 h-38 q-10 0 -7-8z"
              fill="#2f9e73" opacity="0.82"/>
        <text x="0" y="76" text-anchor="middle">{html.escape(label)}</text>
      </g>
    """


def build_digital_twin_html(run: dict) -> str:
    snapshot = run.get("hardware_state") or {}
    spectro = snapshot.get("spectrophotometer") or {}
    liquid_handler = snapshot.get("liquid_handler") or {}
    incubator = snapshot.get("incubator") or {}
    sample = snapshot.get("sample") or {}
    target = snapshot.get("target_reactor") or {}
    alarm = snapshot.get("alarm") or run.get("last_error")
    current_step = str(run.get("current_step") or "START")
    status = str(run.get("status") or "IDLE")
    latest_event = (run.get("events") or [{}])[-1]
    event_message = html.escape(
        str(latest_event.get("message") or "工作站已就绪，等待启动仿真")
    )

    locations = {
        "manual_zone": (95, 275),
        "spectrophotometer": (230, 165),
        "liquid_handler": (535, 170),
        "incubator": (855, 170),
        "in_transit": (385, 275),
    }
    flask_x, flask_y = locations.get(str(sample.get("location")), (95, 275))
    target_ratio = min(max(_number(target.get("volume_ml")) / 250.0, 0.0), 1.0)
    container = str(sample.get("container") or "source_flask")
    flask_label = "目标三角瓶" if "target" in container else "源三角瓶"

    spectro_active = spectro.get("status") == "RUNNING"
    handler_active = liquid_handler.get("status") == "RUNNING"
    incubator_active = incubator.get("status") == "RUNNING"
    spectro_class = "device active" if spectro_active else "device"
    handler_class = "device active" if handler_active else "device"
    incubator_class = "device active" if incubator_active else "device"

    step_index = next(
        (index for index, (name, _) in enumerate(STEPS) if name == current_step), -1
    )
    step_html = []
    for index, (name, label) in enumerate(STEPS):
        if status == "FAILED" and name == current_step:
            css_class = "failed"
        elif index < step_index or status == "COMPLETED":
            css_class = "done"
        elif index == step_index:
            css_class = "active"
        else:
            css_class = "waiting"
        step_html.append(
            f'<div class="step {css_class}"><span>{index + 1}</span>{html.escape(label)}</div>'
        )

    absorbance = spectro.get("absorbance")
    absorbance_text = "--" if absorbance is None else f"{_number(absorbance):.3f}"
    alarm_html = ""
    if alarm:
        alarm_message = html.escape(str(alarm.get("message") or alarm))
        alarm_html = f'<div class="alarm" role="alert">{alarm_message}</div>'

    return f"""
    <html><head><style>
      * {{ box-sizing: border-box; }}
      body {{ margin: 0; color: #17202a; background: #ffffff; font-family: Inter, "Segoe UI", sans-serif; }}
      .surface {{ border: 1px solid #d8dee6; border-radius: 8px; overflow: hidden; background: #f8fafb; }}
      .statusbar {{ display: flex; gap: 12px; align-items: center; justify-content: space-between;
                    padding: 12px 16px; background: #ffffff; border-bottom: 1px solid #d8dee6; }}
      .event {{ min-width: 0; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
      .run-status {{ flex: none; font-size: 12px; font-weight: 500; padding: 4px 8px; border-radius: 6px;
                     color: #165d43; background: #e6f4ee; border: 1px solid #9fd8c1; }}
      .workbench {{ width: 100%; height: auto; display: block; background: #f8fafb; }}
      .device {{ fill: #ffffff; stroke: #94a3b8; stroke-width: 2; }}
      .device.active {{ stroke: #087ea4; filter: drop-shadow(0 0 6px #60a5fa88); }}
      .bench {{ fill: #dce3ea; stroke: #64748b; stroke-width: 2; }}
      .device-label {{ font-size: 14px; font-weight: 500; fill: #17202a; text-anchor: middle; }}
      .device-value {{ font-size: 12px; fill: #52606d; text-anchor: middle; }}
      .route {{ fill: none; stroke: #a8b2bf; stroke-width: 2; stroke-dasharray: 6 7; }}
      .sample-flask {{ transition: transform .4s ease; }}
      .sample-flask text {{ font-size: 12px; fill: #17202a; font-weight: 500; }}
      .beam {{ stroke: #d97706; stroke-width: 4; opacity: {1 if spectro_active else .18}; }}
      .pipette {{ transform-box: fill-box; transform-origin: center; }}
      .handler-active .pipette {{ animation: pipette 1s ease-in-out infinite; }}
      .incubator-light {{ fill: #f4c430; opacity: {1 if incubator_active else .2}; }}
      @keyframes pipette {{ 50% {{ transform: translateY(16px); }} }}
      .steps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(88px, 1fr)); gap: 6px;
                padding: 10px 12px 12px; background: #ffffff; border-top: 1px solid #d8dee6; }}
      .step {{ min-height: 45px; padding: 6px; border-left: 3px solid #cbd5e1; color: #64748b;
               font-size: 11px; background: #f8fafb; }}
      .step span {{ display: block; font-size: 10px; margin-bottom: 2px; }}
      .step.active {{ border-color: #087ea4; color: #0b5168; background: #e8f4f7; }}
      .step.done {{ border-color: #2f9e73; color: #165d43; background: #edf8f3; }}
      .step.failed {{ border-color: #c2413b; color: #8d211d; background: #fbeceb; }}
      .alarm {{ margin: 0; padding: 9px 14px; color: #8d211d; background: #fbeceb;
                border-top: 1px solid #e7aaa6; font-size: 12px; }}
      @media (max-width: 620px) {{
        .statusbar {{ align-items: flex-start; }}
        .event {{ white-space: normal; }}
        .device-label {{ font-size: 12px; }}
        .device-value {{ font-size: 10px; }}
        .steps {{ grid-template-columns: repeat(3, 1fr); }}
      }}
      @media (prefers-reduced-motion: reduce) {{ .handler-active .pipette {{ animation: none; }} }}
    </style></head><body>
      <section class="surface" aria-label="传代设备仿真工作台">
        <div class="statusbar">
          <div class="event">{event_message}</div>
          <div class="run-status">{html.escape(_status_text(status))}</div>
        </div>
        <svg class="workbench" viewBox="0 0 1000 390" role="img" aria-label="三角烧瓶在人工区、分光光度计、移液平台和培养箱之间流转">
          <path class="route" d="M110 285 C180 285 180 210 230 210 S390 285 470 250 S740 210 855 210"/>
          <rect class="bench" x="32" y="92" width="936" height="235" rx="8"/>

          <rect x="52" y="230" width="120" height="80" rx="6" fill="#ffffff" stroke="#94a3b8" stroke-width="2"/>
          <text class="device-label" x="112" y="260">人工操作区</text>
          <text class="device-value" x="112" y="281">装载与搬运</text>

          <g>
            <rect class="{spectro_class}" x="174" y="112" width="190" height="125" rx="8"/>
            <rect x="205" y="137" width="128" height="54" rx="5" fill="#dfe7ee" stroke="#64748b"/>
            <line class="beam" x1="221" y1="164" x2="317" y2="164"/>
            <text class="device-label" x="269" y="215">T700AS 分光光度计</text>
            <text class="device-value" x="269" y="232">{_number(spectro.get('wavelength_nm'), 680):.0f} nm · OD {absorbance_text}</text>
          </g>

          <g class="{'handler-active' if handler_active else ''}">
            <rect class="{handler_class}" x="405" y="105" width="292" height="168" rx="8"/>
            <rect x="430" y="203" width="242" height="45" rx="4" fill="#e7edf2" stroke="#64748b"/>
            <g class="pipette">
              <rect x="484" y="126" width="130" height="30" rx="4" fill="#52606d"/>
              <path d="M505 156 v48 M548 156 v48 M591 156 v48" stroke="#087ea4" stroke-width="5"/>
            </g>
            <text class="device-label" x="551" y="293">自研移液平台</text>
            <text class="device-value" x="551" y="312">150 mL 培养基 + 1 mL 藻液</text>
          </g>

          <g>
            <rect class="{incubator_class}" x="754" y="105" width="190" height="205" rx="8"/>
            <rect x="782" y="135" width="134" height="118" rx="5" fill="#eef2f5" stroke="#64748b"/>
            <circle class="incubator-light" cx="815" cy="165" r="13"/>
            <path d="M790 210 h116 M790 230 h116" stroke="#94a3b8" stroke-width="2"/>
            <text class="device-label" x="849" y="278">智能光照培养箱</text>
            <text class="device-value" x="849" y="298">{_number(incubator.get('temperature_c'), 22):.1f} °C · {_number(incubator.get('light_lux')):.0f} lux</text>
          </g>

          {_flask_markup(flask_x, flask_y, target_ratio, flask_label)}
        </svg>
        <div class="steps">{''.join(step_html)}</div>
        {alarm_html}
      </section>
    </body></html>
    """


def _render_manual_task_dialog(st, backend_url: str, run: dict) -> None:
    task = run.get("pending_manual_task") or {}
    title = str(task.get("title") or "人工操作确认")

    @st.dialog(title)
    def manual_task_dialog():
        st.write(task.get("instruction") or "请完成当前人工操作。")
        col_source, col_target = st.columns(2)
        col_source.metric("来源", task.get("source") or "-")
        col_target.metric("目标", task.get("destination") or "-")
        st.caption(f"样品/容器：{task.get('container') or '-'}")

        if run.get("interaction_mode") == "demo":
            st.info("自动演示模式正在确认此操作。")
            return

        note = st.text_input("备注（可选）", key=f"manual_note_{task.get('task_id')}")
        completed_col, failed_col = st.columns(2)
        if completed_col.button("已完成", type="primary", use_container_width=True):
            response = resolve_subculture_manual_task(
                backend_url,
                run["run_id"],
                task["task_id"],
                {"resolution": "completed", "note": note or None},
            )
            if response.status_code >= 400:
                st.error(response.text)
            else:
                st.rerun()
        if failed_col.button("执行失败", use_container_width=True):
            response = resolve_subculture_manual_task(
                backend_url,
                run["run_id"],
                task["task_id"],
                {"resolution": "failed", "note": note or "人工操作报告失败"},
            )
            if response.status_code >= 400:
                st.error(response.text)
            else:
                st.rerun()

    manual_task_dialog()


def render_simulation_panel(st, backend_url: str, *, standalone: bool = False) -> None:
    import streamlit.components.v1 as components

    if not standalone:
        st.markdown("### 自动传代设备仿真")

    with st.container(border=True):
        row_one = st.columns([2, 1, 1, 2])
        with row_one[0]:
            strain_id = st.text_input(
                "仿真品系", value="Chlorella_01", key="simulation_strain"
            )
        with row_one[1]:
            generation = st.number_input(
                "当前代数",
                min_value=1,
                value=1,
                step=1,
                key="simulation_generation",
            )
        with row_one[2]:
            mode_label = st.selectbox(
                "交互模式", list(INTERACTION_MODES), key="simulation_mode"
            )
        with row_one[3]:
            fault_label = st.selectbox(
                "故障场景", list(FAULT_OPTIONS), key="simulation_fault"
            )

        row_two = st.columns([1, 1, 2])
        with row_two[0]:
            wavelength_nm = st.number_input(
                "测量波长 (nm)",
                min_value=190,
                max_value=1100,
                value=680,
                step=1,
                key="simulation_wavelength",
            )
        with row_two[1]:
            absorbance = st.number_input(
                "模拟吸光度",
                min_value=0.0,
                max_value=4.0,
                value=0.8,
                step=0.05,
                key="simulation_absorbance",
            )
        with row_two[2]:
            st.write("")
            start_clicked = st.button(
                "启动仿真", type="primary", use_container_width=True
            )

    if start_clicked:
        try:
            response = start_subculture_simulation(
                backend_url,
                {
                    "strain_id": strain_id,
                    "generation_number": int(generation),
                    "interaction_mode": INTERACTION_MODES[mode_label],
                    "fault_scenario": FAULT_OPTIONS[fault_label],
                    "wavelength_nm": float(wavelength_nm),
                    "simulated_absorbance": float(absorbance),
                },
            )
            response.raise_for_status()
            st.session_state["simulation_run_id"] = response.json()["run_id"]
            st.session_state["simulation_error"] = None
        except Exception as exc:
            st.session_state["simulation_error"] = str(exc)

    if st.session_state.get("simulation_error"):
        st.error(st.session_state["simulation_error"])

    run_id = st.session_state.get("simulation_run_id")
    if not run_id:
        components.html(build_digital_twin_html({}), height=590, scrolling=False)
        return

    try:
        response = get_subculture_simulation(backend_url, run_id)
        response.raise_for_status()
        run = response.json()
    except Exception as exc:
        st.error(f"读取仿真状态失败：{exc}")
        return

    summary_cols = st.columns(3)
    summary_cols[0].metric("运行状态", _status_text(str(run.get("status"))))
    od_value = run.get("measured_absorbance")
    summary_cols[1].metric("OD 结果", "--" if od_value is None else f"{od_value:.3f}")
    queue_position = run.get("queue_position")
    if run.get("status") in {"COMPLETED", "FAILED", "SKIPPED"}:
        queue_text = "已结束"
    elif queue_position == 0:
        queue_text = "执行中"
    else:
        queue_text = f"第 {queue_position} 位"
    summary_cols[2].metric(
        "工作站队列", queue_text
    )

    components.html(build_digital_twin_html(run), height=610, scrolling=False)
    st.progress(
        float(run.get("overall_progress") or 0.0),
        text=f"当前步骤：{run.get('current_step')}",
    )

    if run.get("status") == "WAITING_MANUAL" and run.get("pending_manual_task"):
        _render_manual_task_dialog(st, backend_url, run)
    elif run.get("status") == "COMPLETED":
        st.success("仿真传代完成。结果仅保存在仿真运行中，未写入生产状态。")
    elif run.get("status") == "FAILED":
        error = run.get("last_error") or {}
        st.error(f"仿真失败并已安全停机：{error.get('message', '未知错误')}")
    elif run.get("status") == "QUEUED":
        st.info(f"工作站忙，当前排队位置：{run.get('queue_position')}")

    with st.expander("设备事件", expanded=False):
        st.json((run.get("events") or [])[-16:])

    if run.get("status") in {"QUEUED", "RUNNING", "WAITING_MANUAL"}:
        time.sleep(0.25)
        st.rerun()
