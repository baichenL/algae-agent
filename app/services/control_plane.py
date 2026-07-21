from __future__ import annotations

from typing import Any

from app.core import database
from app.core.db import scientific as scientific_db
from app.core.db import workflow_runs
from app.core.db import control_plane as control_plane_db
from app.services.observability.agent_trace import build_agent_run_trace, list_agent_run_summaries
from app.services.workflows.simulation_service import simulation_runs
from app.core.workspaces import current_workspace, list_test_workspaces, workspace_for_run, workspace_scope


_ACTION_TO_SIMULATION_STEP = {
    "load_spectrophotometer": "ManualLoadSpectrophotometer",
    "blank": "BlankSpectrophotometer",
    "measure_absorbance": "MeasureAbsorbance",
    "load_liquid_handler": "ManualLoadLiquidHandler",
    "check": "LoadMaterials",
    "dispense_medium": "DispenseMedium",
    "transfer_seed": "TransferSeedCulture",
    "mix_and_seal": "MixAndSeal",
    "move_to_incubator": "ManualMoveToIncubator",
    "configure": "MoveToIncubator",
    "cleanup": "CleanupWorkspace",
    "safe_shutdown": "SafeShutdown",
}


PHASES: dict[str, list[tuple[str, str]]] = {
    "scientific": [
        ("data", "数据"), ("quality", "质量检查"), ("diagnosis", "诊断"),
        ("evidence", "证据"), ("design", "方案"), ("verify", "验证 / PlanPatch"),
        ("approval", "审批"), ("execute", "数字孪生"), ("feedback", "结果回流"),
    ],
    "agent": [
        ("message", "消息"), ("context", "Context"), ("decision", "Decision"),
        ("policy", "Policy"), ("tool", "Tool"), ("observation", "Observation"),
        ("response", "审批 / 答复"),
    ],
    "workflow": [
        ("configure", "配置"), ("queue", "排队"), ("execute", "设备步骤"),
        ("manual", "人工任务"), ("recovery", "恢复 / 安全停机"), ("complete", "完成"),
    ],
    "simulation": [
        ("configure", "配置"), ("queue", "排队"), ("execute", "设备步骤"),
        ("manual", "人工任务"), ("recovery", "恢复 / 安全停机"), ("complete", "完成"),
    ],
}


def _pending_items() -> list[dict[str, Any]]:
    return database.list_pending_actions(status="all")


def _approval_public(item: dict[str, Any], *, run_id: str | None = None) -> dict[str, Any]:
    payload = item.get("payload") or {}
    data = payload.get("data") or {}
    design = data.get("design") or {}
    return {
        "id": int(item["id"]),
        "run_id": run_id,
        "action_type": payload.get("type") or item.get("action_type"),
        "status": item.get("status", "pending"),
        "risk_level": item.get("risk_level", "medium"),
        "design_hash": data.get("design_hash") or design.get("design_hash"),
        "approval_version": int(item.get("approval_version") or item["id"]),
        "requested_by": item.get("requester"),
        "reviewed_by": item.get("reviewed_by"),
        "review_reason": item.get("review_reason"),
        "created_at": item.get("created_at"),
        "reviewed_at": item.get("reviewed_at"),
        "expires_at": item.get("expires_at"),
        "source": item.get("source"),
        "summary": approval_summary(item),
        "payload": payload,
        "executed_at": item.get("executed_at"),
        "execution_result": item.get("execution_result"),
        "execution_status": item.get("execution_status") or "not_started",
        "execution_started_at": item.get("execution_started_at"),
        "execution_error": item.get("execution_error"),
    }


def approval_summary(item: dict[str, Any]) -> str:
    payload = item.get("payload") or {}
    data = payload.get("data") or {}
    kind = payload.get("type") or item.get("action_type")
    if kind == "scientific_experiment_plan":
        design = data.get("design") or {}
        return f"数字孪生方案 · {len(design.get('conditions') or [])} 条件 · hash {str(data.get('design_hash') or '')[:12]}"
    if kind == "workflow_subculture":
        return f"传代 Workflow · {data.get('strain_id') or '-'}"
    if kind in {"add_strain", "update_strain", "delete_strain"}:
        labels = {"add_strain": "新增品系", "update_strain": "更新品系", "delete_strain": "删除品系"}
        return f"{labels[kind]} · {data.get('strain_id') or '-'}"
    return str(kind or "待审批操作")


def approval_run_id(item: dict[str, Any]) -> str | None:
    payload = item.get("payload") or {}
    data = payload.get("data") or {}
    if payload.get("type") == "scientific_experiment_plan" and data.get("scientific_run_id"):
        return f"scientific:{data['scientific_run_id']}"
    if item.get("agent_run_id"):
        return f"agent:{item['agent_run_id']}"
    workflow = workflow_runs.get_workflow_run_by_pending(int(item["id"]))
    if workflow:
        return f"workflow:{workflow['id']}"
    return None


def list_approvals(*, status: str = "all", kind: str | None = None, _include_workspaces: bool = True) -> list[dict[str, Any]]:
    items = _pending_items()
    result = []
    for item in items:
        payload_kind = ((item.get("payload") or {}).get("type") or item.get("action_type"))
        if status != "all" and item.get("status") != status:
            continue
        if kind and payload_kind != kind:
            continue
        result.append(_approval_public(item, run_id=approval_run_id(item)))
    if _include_workspaces and current_workspace().id == "shared":
        for workspace in list_test_workspaces():
            with workspace_scope(workspace):
                workspace_items = list_approvals(status=status, kind=kind, _include_workspaces=False)
            for item in workspace_items:
                item["workspace_id"] = workspace.id
            result.extend(workspace_items)
    return result


def _phase_payload(kind: str, active_key: str, *, terminal: bool = False, blocked: bool = False) -> list[dict[str, Any]]:
    definitions = PHASES[kind]
    active_index = next((index for index, (key, _) in enumerate(definitions) if key == active_key), 0)
    phases = []
    for index, (key, label) in enumerate(definitions):
        state = "completed" if terminal or index < active_index else "active" if index == active_index else "pending"
        if blocked and index == active_index:
            state = "blocked"
        phases.append({"key": key, "label": label, "state": state})
    return phases


def _scientific_status(run: dict[str, Any]) -> tuple[str, str, float]:
    status = str(run.get("status") or "running")
    mapping = {
        "waiting_approval": ("waiting_approval", "approval", 0.72),
        "ready_to_execute": ("ready_to_execute", "execute", 0.78),
        "simulation_failed": ("failed", "execute", 0.84),
        "blocked": ("blocked", "verify", 0.62),
        "denied": ("cancelled", "approval", 0.72),
        "succeeded": ("succeeded", "feedback", 1.0),
        "cycle_completed": ("running", "feedback", 0.92),
        "design_ready": ("running", "design", 0.58),
    }
    return mapping.get(status, ("running", "quality", 0.18))


def _scientific_summary(run: dict[str, Any]) -> dict[str, Any]:
    status, phase, progress = _scientific_status(run)
    goal = run.get("goal") or {}
    return {
        "id": f"scientific:{run['id']}", "kind": "scientific",
        "title": f"科学闭环 · {str(run['id'])[-8:]}", "workspace_id": current_workspace().id,
        "status": status, "phase": phase, "progress": progress,
        "cycle": int(run.get("cycle_index") or 0), "risk_level": "medium",
        "simulation_only": True, "source_ref": {"type": "scientific_run", "id": run["id"]},
        "created_at": run.get("created_at"), "updated_at": run.get("updated_at"),
        "summary": f"{goal.get('direction', 'maximize')} {goal.get('target_metric', 'biomass')}",
    }


def _agent_summary(run: dict[str, Any]) -> dict[str, Any]:
    raw = str(run.get("status") or "running")
    status = raw if raw in {"running", "waiting_approval", "failed", "blocked", "succeeded"} else "succeeded"
    phase = "response" if status in {"succeeded", "failed", "blocked", "waiting_approval"} else "context"
    return {
        "id": f"agent:{run['id']}", "kind": "agent",
        "title": (run.get("user_message") or "Agent Run")[:72], "workspace_id": current_workspace().id,
        "status": status, "phase": phase, "progress": 1.0 if status == "succeeded" else 0.82 if phase == "response" else 0.25,
        "cycle": 0, "risk_level": run.get("risk_level") or "low",
        "simulation_only": True, "source_ref": {"type": "agent_run", "id": run["id"]},
        "created_at": run.get("started_at"), "updated_at": run.get("finished_at") or run.get("started_at"),
        "summary": run.get("response_summary") or run.get("final_route") or "Agent orchestration",
    }


def _workflow_summary(run: dict[str, Any]) -> dict[str, Any]:
    raw = str(run.get("status") or "queued")
    status = "blocked" if raw == "stale" else raw
    phase = "queue" if status == "queued" else "execute" if status == "running" else "complete"
    return {
        "id": f"workflow:{run['id']}", "kind": "workflow", "title": f"传代 Workflow · {run.get('strain_id')}",
        "workspace_id": current_workspace().id, "status": status, "phase": phase,
        "progress": float(run.get("progress") or (0.15 if status == "queued" else 0.55 if status == "running" else 1.0)),
        "cycle": 0, "risk_level": "high", "simulation_only": not bool(run.get("physical_execution")),
        "source_ref": {"type": "workflow_run", "id": run["id"]},
        "created_at": run.get("created_at"), "updated_at": run.get("updated_at") or run.get("finished_at") or run.get("started_at") or run.get("created_at"),
        "summary": run.get("error") or run.get("current_step") or run.get("execution_mode") or "subculture",
    }


def _simulation_summary(run: dict[str, Any]) -> dict[str, Any]:
    raw = str(run.get("status") or "QUEUED").lower()
    status_map = {"waiting_manual": "waiting_input", "completed": "succeeded", "skipped": "cancelled"}
    status = status_map.get(raw, raw)
    phase = "manual" if status == "waiting_input" else "queue" if status == "queued" else "execute" if status == "running" else "complete"
    return {
        "id": f"simulation:{run['run_id']}", "kind": "simulation", "title": f"设备仿真 · {run.get('strain_id')}",
        "workspace_id": current_workspace().id, "status": status, "phase": phase,
        "progress": float(run.get("overall_progress") or 0), "cycle": 0, "risk_level": "low", "simulation_only": True,
        "source_ref": {"type": "simulation_run", "id": run["run_id"]}, "created_at": run.get("created_at"),
        "updated_at": run.get("updated_at"), "summary": run.get("current_step") or "simulation",
    }


def list_runs(*, kind: str | None = None, status: str | None = None, limit: int = 100, _include_workspaces: bool = True) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if kind in {None, "scientific"}:
        items.extend(_scientific_summary(item) for item in scientific_db.list_scientific_runs(limit=limit))
    if kind in {None, "agent"}:
        items.extend(_agent_summary(item) for item in list_agent_run_summaries(limit=limit))
    if kind in {None, "workflow"}:
        items.extend(_workflow_summary(item) for item in workflow_runs.list_workflow_runs(limit=limit))
    if kind in {None, "simulation"}:
        items.extend(_simulation_summary(item) for item in simulation_runs.list(limit=limit))
    if status:
        items = [item for item in items if item["status"] == status]
    for item in items:
        control_plane_db.upsert_run(item)
    if _include_workspaces and current_workspace().id == "shared":
        for workspace in list_test_workspaces():
            with workspace_scope(workspace):
                workspace_items = list_runs(kind=kind, status=status, limit=limit, _include_workspaces=False)
            for item in workspace_items:
                item["workspace_id"] = workspace.id
            items.extend(workspace_items)
    items.sort(key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), reverse=True)
    return items[: max(1, min(int(limit or 100), 500))]


def _actions_for(approvals: list[dict[str, Any]], status: str) -> list[dict[str, Any]]:
    pending = next((item for item in approvals if item["status"] == "pending"), None)
    approved = next((item for item in approvals if item["status"] == "approved" and not item.get("executed_at")), None)
    if pending:
        return [
            {"id": "approve", "label": "批准方案", "kind": "approval", "approval_id": pending["id"], "required_role": "approver", "requires_confirmation": True},
            {"id": "deny", "label": "退回 / 拒绝", "kind": "approval", "approval_id": pending["id"], "required_role": "approver", "requires_confirmation": True},
        ]
    if approved or status == "ready_to_execute":
        return [{"id": "execute", "label": "启动执行", "kind": "execution", "required_role": "approver", "requires_confirmation": True}]
    return []


def _agent_replans(trace_result: dict[str, Any]) -> list[dict[str, Any]]:
    trace = trace_result.get("trace") or {}
    raw_events = trace_result.get("raw_events") or []
    timestamps: dict[int, str | None] = {}
    for event in raw_events:
        if event.get("event_type") != "agent_replan_directive_created":
            continue
        payload = event.get("payload") or {}
        timestamps[int(payload.get("step_index") or 0)] = event.get("created_at")

    summaries: list[dict[str, Any]] = []
    for step in trace.get("steps") or []:
        directive = step.get("replan_directive") or {}
        source = directive.get("replan_source") or step.get("replan_source")
        if not directive or not (
            directive.get("is_dynamic")
            or source == "dynamic_observation"
            or str(directive.get("reason") or "").startswith("dynamic_")
        ):
            continue
        step_index = int(step.get("step_index") or 0)
        selected = directive.get("selected_candidate") or {}
        decision = directive.get("decision") or step.get("replan_decision") or {}
        observation = step.get("observation") or {}
        original_plan = step.get("plan_step") or step.get("decision") or trace.get("plan") or {}
        changes = []
        if selected.get("action_type"):
            changes.append(f"下一行动改为 {selected['action_type']}")
        if selected.get("target_id"):
            changes.append(f"目标锁定为 {selected['target_id']}")
        if directive.get("policy_boundary"):
            changes.append(f"保留安全边界 {directive['policy_boundary']}")
        rejected = directive.get("rejected_candidates") or []
        if rejected:
            changes.append(f"拒绝 {len(rejected)} 个不满足边界的候选行动")
        summaries.append({
            "id": f"agent-replan-{step_index}",
            "kind": "agent",
            "trigger_observation": observation,
            "original_plan": original_plan,
            "reason": selected.get("reason") or directive.get("reason") or "根据工具观察重新规划",
            "revised_plan": decision or selected,
            "key_changes": changes or ["根据最新观察调整下一行动"],
            "from_version": max(step_index, 1),
            "to_version": max(step_index, 1) + 1,
            "final_action": decision.get("action_name") or selected.get("action_type"),
            "occurred_at": timestamps.get(step_index),
            "safety_boundary": directive.get("policy_boundary") or directive.get("blocked_by"),
            "debug": {
                "rule_candidates": directive.get("rule_candidates") or [],
                "llm_candidates": directive.get("llm_candidates") or [],
                "rejected_candidates": rejected,
                "selection_reason": directive.get("selection_reason"),
                "confidence": selected.get("confidence"),
                "policy_boundary": directive.get("policy_boundary"),
                "raw_directive": directive,
            },
        })
    return summaries


def _scientific_replans(run: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = []
    for artifact in run.get("artifacts") or []:
        if artifact.get("artifact_type") != "plan_patch":
            continue
        patch = artifact.get("payload") or {}
        metadata = patch.get("metadata") or {}
        changes = []
        removed = metadata.get("removed_condition_ids") or patch.get("remove_node_ids") or []
        if removed:
            changes.append(f"移除条件：{', '.join(map(str, removed))}")
        retained = metadata.get("retained_condition_ids") or []
        if retained:
            changes.append(f"保留条件：{', '.join(map(str, retained))}")
        before_vessels = metadata.get("before_required_vessels")
        after_vessels = metadata.get("after_required_vessels")
        if before_vessels is not None and after_vessels is not None:
            changes.append(f"容器需求由 {before_vessels} 降至 {after_vessels}")
        summaries.append({
            "id": f"scientific-replan-{artifact.get('id')}",
            "kind": "scientific",
            "trigger_observation": {"code": str(patch.get("reason") or "").split(":", 1)[0], **metadata},
            "original_plan": {"design_hash": metadata.get("before_design_hash"), "condition_count": metadata.get("before_condition_count")},
            "reason": patch.get("reason") or "科学方案校验要求修订",
            "revised_plan": {"design_hash": metadata.get("after_design_hash"), "condition_count": metadata.get("after_condition_count"), "replace_nodes": patch.get("replace_nodes") or [], "add_nodes": patch.get("add_nodes") or []},
            "key_changes": changes or ["科学方案已按验证结果修订"],
            "from_version": int(patch.get("base_version") or max(int(artifact.get("version") or 1) - 1, 1)),
            "to_version": int(patch.get("new_version") or artifact.get("version") or 1),
            "final_action": ((patch.get("replace_nodes") or patch.get("add_nodes") or [{}])[0]).get("action"),
            "occurred_at": artifact.get("created_at"),
            "safety_boundary": "设备容量与对照组约束",
            "debug": {"raw_patch": patch},
        })
    return summaries


def get_run(canonical_id: str) -> dict[str, Any] | None:
    owner = workspace_for_run(canonical_id)
    if owner and owner.id != current_workspace().id:
        with workspace_scope(owner):
            result = get_run(canonical_id)
        if result:
            result["workspace_id"] = owner.id
        return result
    if ":" not in canonical_id:
        return None
    kind, source_id = canonical_id.split(":", 1)
    if kind == "scientific":
        run = scientific_db.get_scientific_run(source_id)
        if not run:
            return None
        summary = _scientific_summary(run)
        approvals = [
            _approval_public(item, run_id=canonical_id) for item in _pending_items()
            if (((item.get("payload") or {}).get("data") or {}).get("scientific_run_id") == source_id)
        ]
        status, phase, _ = _scientific_status(run)
        terminal = status == "succeeded"
        return {
            **summary, "phases": _phase_payload("scientific", phase, terminal=terminal, blocked=status in {"blocked", "failed"}),
            "current_action": (_actions_for(approvals, status) or [None])[0], "available_actions": _actions_for(approvals, status),
            "artifacts": run.get("artifacts") or [], "approvals": approvals,
            "executions": run.get("virtual_results") or [], "assertions": _scientific_assertions(run),
            "replans": _scientific_replans(run),
            "links": {"dataset_id": run.get("dataset_id"), "agent_run_id": run.get("agent_run_id")},
            "raw": run,
        }
    if kind == "agent":
        trace = build_agent_run_trace(source_id)
        if not trace:
            return None
        summary = _agent_summary(trace["run"])
        approvals = [_approval_public(item, run_id=canonical_id) for item in _pending_items() if item.get("agent_run_id") == source_id]
        terminal = summary["status"] == "succeeded"
        return {
            **summary, "phases": _phase_payload("agent", summary["phase"], terminal=terminal, blocked=summary["status"] in {"failed", "blocked"}),
            "current_action": (_actions_for(approvals, summary["status"]) or [None])[0], "available_actions": _actions_for(approvals, summary["status"]),
            "artifacts": [], "approvals": approvals, "executions": [], "assertions": [],
            "replans": _agent_replans(trace),
            "trace": trace.get("trace") or {}, "raw_events": trace.get("raw_events") or [], "raw": trace,
        }
    if kind == "workflow":
        try:
            run = workflow_runs.get_workflow_run(int(source_id))
        except ValueError:
            return None
        if not run:
            return None
        summary = _workflow_summary(run)
        pending = database.get_pending_action(int(run.get("pending_id"))) if run.get("pending_id") else None
        approvals = [_approval_public(pending, run_id=canonical_id)] if pending else []
        manual_commit = (
            summary["status"] == "succeeded"
            and not bool(run.get("persisted"))
            and not bool(run.get("physical_execution"))
        )
        available_actions = (
            [{"id": "execute", "label": "启动传代执行", "kind": "execution", "required_role": "approver", "requires_confirmation": True}]
            if summary["status"] == "queued"
            else [{"id": "manual_commit", "label": "确认已手工完成并更新资源", "kind": "manual_commit", "required_role": "approver", "requires_confirmation": True}]
            if manual_commit else []
        )
        return {
            **summary, "phases": _phase_payload("workflow", summary["phase"], terminal=summary["status"] == "succeeded", blocked=summary["status"] in {"failed", "blocked"}),
            "current_action": (available_actions or [None])[0],
            "available_actions": available_actions,
            "artifacts": [], "approvals": approvals, "executions": [run], "assertions": [], "replans": [],
            "simulation": {
                "current_step": run.get("current_step"),
                "progress": float(run.get("progress") or 0),
                "latest_snapshot": run.get("hardware_state") or (run.get("tool_result") or {}).get("hardware_state"),
                "event_count": int(run.get("last_event_sequence") or len((run.get("tool_result") or {}).get("simulation_events") or [])),
                "replay_available": bool(run.get("last_event_sequence") or (run.get("tool_result") or {}).get("simulation_events")),
                "last_event_at": run.get("last_event_at"),
            },
            "raw": run,
        }
    if kind == "simulation":
        run = simulation_runs.get(source_id)
        if not run:
            return None
        summary = _simulation_summary(run)
        action = None
        task = run.get("pending_manual_task")
        if task:
            action = {"id": "resolve_manual", "label": "完成人工任务", "kind": "manual_task", "task_id": task.get("task_id"), "required_role": "approver", "requires_confirmation": True}
        return {
            **summary, "phases": _phase_payload("simulation", summary["phase"], terminal=summary["status"] == "succeeded", blocked=summary["status"] == "failed"),
            "current_action": action, "available_actions": [action] if action else [], "artifacts": [], "approvals": [],
            "executions": [run], "assertions": [], "replans": [], "raw": run,
        }
    return None


def _scientific_assertions(run: dict[str, Any]) -> list[dict[str, Any]]:
    assertions = []
    for artifact in run.get("artifacts") or []:
        if artifact.get("artifact_type") == "verification_report":
            verdict = (artifact.get("payload") or {}).get("verdict")
            assertions.append({"name": f"Verifier v{artifact.get('version')}", "status": "passed" if verdict == "pass" else "failed", "details": verdict})
        if artifact.get("artifact_type") == "plan_patch":
            assertions.append({"name": "PlanPatch 已记录", "status": "passed", "details": (artifact.get("payload") or {}).get("reason")})
    return assertions


def get_events(canonical_id: str, *, after: int = 0) -> list[dict[str, Any]]:
    owner = workspace_for_run(canonical_id)
    if owner and owner.id != current_workspace().id:
        with workspace_scope(owner):
            return get_events(canonical_id, after=after)
    detail = get_run(canonical_id)
    if not detail:
        return []
    events: list[dict[str, Any]] = []
    if detail["kind"] == "agent":
        for item in detail.get("raw_events") or []:
            sequence = int(item.get("id") or 0)
            if sequence > after:
                events.append({"sequence": sequence, "event_type": item.get("event_type"), "phase": item.get("layer"), "level": "info", "payload": item.get("payload") or {}, "created_at": item.get("created_at")})
    elif detail["kind"] == "scientific":
        for item in detail.get("artifacts") or []:
            sequence = int(item.get("id") or 0)
            if sequence > after:
                events.append({"sequence": sequence, "event_type": item.get("artifact_type"), "phase": item.get("artifact_type"), "level": "info", "payload": item.get("payload") or {}, "created_at": item.get("created_at")})
    else:
        raw = detail.get("raw") or {}
        raw_events = raw.get("events") or (
            (raw.get("tool_result") or {}).get("simulation_events")
            if detail["kind"] == "workflow"
            else []
        ) or []
        for index, item in enumerate(raw_events, 1):
            if index > after:
                action = item.get("action") or item.get("event") or "workflow_event"
                step = _ACTION_TO_SIMULATION_STEP.get(action, item.get("step") or detail.get("phase"))
                payload = {
                    **item,
                    "step": step,
                    "step_progress": item.get("progress"),
                    "progress": min(index / max(len(raw_events), 1), 1.0),
                    "snapshot": item.get("snapshot") or item.get("hardware_state") or {},
                }
                events.append({"sequence": index, "event_type": "simulation_fault" if item.get("status") == "failed" or item.get("error") else "simulation_step_completed" if item.get("status") == "completed" else "simulation_snapshot", "phase": step, "level": "error" if item.get("error") or item.get("status") == "failed" else "info", "payload": payload, "created_at": item.get("timestamp")})
    control_plane_db.append_events(canonical_id, events)
    return control_plane_db.list_events(canonical_id, after=after)


def dashboard() -> dict[str, Any]:
    runs = list_runs(limit=100)
    approvals = list_approvals(status="pending")
    return {
        "workspace": {"id": "shared", "name": "共享实验室", "mode": "simulation_only"},
        "counts": {
            "active_runs": sum(item["status"] in {"queued", "running", "waiting_input", "waiting_approval", "ready_to_execute"} for item in runs),
            "pending_approvals": len(approvals),
            "failed_runs": sum(item["status"] in {"failed", "blocked"} for item in runs),
            "online_devices": sum(item.get("status") == "online" for item in scientific_db.list_lab_devices()),
        },
        "recent_runs": runs[:8], "pending_approvals": approvals[:5], "devices": scientific_db.list_lab_devices(),
    }
