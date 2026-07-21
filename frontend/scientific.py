from __future__ import annotations

import json

from frontend.http import (
    get_scientific_run,
    import_scientific_dataset,
    list_scientific_datasets,
    start_scientific_run,
)


ARTIFACT_LABELS = {
    "goal_contract": "目标契约",
    "plan_graph": "计划图",
    "observation": "观察",
    "verification_report": "验证报告",
    "plan_patch": "计划补丁",
    "diagnosis_report": "异常诊断",
    "experiment_design": "实验方案",
    "simulation_report": "数字孪生结果",
}


def _load_datasets(st, backend_url: str) -> list[dict]:
    response = list_scientific_datasets(backend_url)
    if response.status_code == 200:
        items = response.json().get("datasets") or []
        st.session_state["scientific_datasets"] = items
        return items
    st.error(f"读取数据集失败：{response.text}")
    return []


def _render_run(st, run: dict) -> None:
    st.markdown(f"### Scientific Run `{run.get('id')}`")
    cols = st.columns(4)
    cols[0].metric("状态", run.get("status", "-"))
    cols[1].metric("循环", int(run.get("cycle_index") or 0))
    cols[2].metric("Artifacts", len(run.get("artifacts") or []))
    cols[3].metric("虚拟结果", len(run.get("virtual_results") or []))
    if run.get("virtual_results"):
        st.warning("以下结果全部为 simulation_only；offline_replay/model_predicted 均不等同于真实湿实验结果。")
    for artifact in run.get("artifacts") or []:
        artifact_type = artifact.get("artifact_type")
        label = ARTIFACT_LABELS.get(artifact_type, artifact_type)
        expanded = artifact_type in {"diagnosis_report", "plan_patch", "experiment_design", "verification_report"}
        with st.expander(f"{label} · v{artifact.get('version')} · {str(artifact.get('content_hash'))[:12]}", expanded=expanded):
            payload = artifact.get("payload") or {}
            if artifact_type == "verification_report":
                verdict = payload.get("verdict")
                (st.success if verdict == "pass" else st.warning if verdict == "revise" else st.error)(f"Verifier: {verdict}")
            if artifact_type == "plan_patch":
                st.info(payload.get("reason") or "Observation-driven plan patch")
            st.json(payload)


def render_scientific_workbench(st, backend_url: str) -> None:
    st.title("🔬 微藻自驱动实验室 Scientific Workbench")
    st.caption("状态感知 → 异常诊断 → 证据验证 → 实验优化 → 约束修补 → 审批 → 数字孪生反馈")

    with st.expander("1. 导入 CSV/XLSX 生长数据", expanded=True):
        uploaded = st.file_uploader("科学数据文件", type=["csv", "xlsx"])
        col_a, col_b = st.columns(2)
        dataset_name = col_a.text_input("数据集名称", value="microalgae-growth")
        strain_id = col_b.text_input("strain_id", value="Chlorella_01")
        st.caption("字段留空时后端会尝试唯一映射；无法唯一识别时返回 mapping_required，不会猜测。")
        mapping_cols = st.columns(3)
        time_column = mapping_cols[0].text_input("时间列（可选）")
        response_column = mapping_cols[1].text_input("响应列（可选）")
        factor_columns = mapping_cols[2].text_input("因素列，逗号分隔（可选）")
        if st.button("导入数据集", disabled=uploaded is None, type="primary"):
            mapping = {"metric_name": "biomass"}
            if time_column:
                mapping["time_column"] = time_column
            if response_column:
                mapping["response_column"] = response_column
            if factor_columns:
                mapping["factor_columns"] = [item.strip() for item in factor_columns.split(",") if item.strip()]
            response = import_scientific_dataset(
                backend_url,
                filename=uploaded.name,
                content=uploaded.getvalue(),
                dataset_name=dataset_name,
                strain_id=strain_id,
                mapping_json=json.dumps(mapping, ensure_ascii=False),
            )
            if response.status_code == 200:
                st.success("数据已标准化为 dataset → culture batches → time-series measurements。")
                st.json(response.json())
                _load_datasets(st, backend_url)
            else:
                st.error(response.text)

    datasets = st.session_state.get("scientific_datasets") or _load_datasets(st, backend_url)
    if st.button("刷新科学数据集"):
        datasets = _load_datasets(st, backend_url)
    if not datasets:
        st.info("请先导入数据集。")
        return

    labels = {
        f"{item.get('name')} · {item.get('id')} · {item.get('batch_count', 0)} batches": item.get("id")
        for item in datasets
    }
    selected_label = st.selectbox("2. 选择数据集", list(labels))
    selected_id = labels[selected_label]
    controls = st.columns(4)
    mode = controls[0].selectbox("任务", ["diagnose_and_optimize", "diagnose"])
    direction = controls[1].selectbox("优化方向", ["maximize", "minimize"])
    offline_replay = controls[2].checkbox("17/8 离线回放", value=True)
    max_cycles = controls[3].number_input("最大循环", min_value=1, max_value=2, value=2)
    if st.button("启动自主闭环", type="primary"):
        with st.spinner("正在执行确定性分析、验证和计划修补……"):
            response = start_scientific_run(
                backend_url,
                {
                    "dataset_id": selected_id,
                    "mode": mode,
                    "direction": direction,
                    "offline_replay": offline_replay,
                    "max_cycles": int(max_cycles),
                    "session_id": "streamlit-scientific-workbench",
                },
            )
        if response.status_code == 202:
            run = response.json().get("scientific_run") or {}
            st.session_state["scientific_run_id"] = run.get("id")
            st.success("闭环已运行到完成或人工审批边界。")
        else:
            st.error(response.text)

    run_id = st.session_state.get("scientific_run_id")
    if run_id:
        if st.button("刷新当前 Scientific Run"):
            st.rerun()
        response = get_scientific_run(backend_url, run_id)
        if response.status_code == 200:
            _render_run(st, response.json().get("scientific_run") or {})
        else:
            st.error(response.text)

