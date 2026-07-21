from app.schemas.algae import ChatResponse
from app.services.chat.response_builder import _complete_chat_response
from app.services.chat.workflow_request_state import (
    build_workflow_request_state,
    clear_workflow_request_state,
    get_workflow_request_state,
    save_workflow_request_state,
)
from app.services.intent.routing_models import SpeechAct, WorkflowAuditDecision, WorkflowDecision
from app.services.memory.memory_service import append_decision_event
from app.services.protocols.models import ProtocolSimulationReport, ProtocolValidationReport
from app.services.protocols.pending_payload import load_frozen_protocol
from app.services.protocols.run_preview import (
    preview_rows,
    protocol_summary,
    simulation_summary,
    validation_summary,
)
from app.services.strains import strain_service
from app.services.workflows.workflow_audit_service import inspect_workflow_activity


def _workflow_protocol_response_fields(payload: dict) -> dict:
    frozen = payload.get("protocol") or {}
    protocol = load_frozen_protocol(payload)
    validation_report = (
        ProtocolValidationReport.from_dict(frozen.get("validation_report") or {})
        if frozen.get("validation_report")
        else None
    )
    simulation_report = (
        ProtocolSimulationReport.from_dict(frozen.get("simulation_report") or {})
        if frozen.get("simulation_report")
        else None
    )
    run_preview = frozen.get("run_preview") or {}
    commands = run_preview.get("commands") or []
    rows = [
        {
            "step_id": item.get("step_id"),
            "operation": item.get("operation"),
            "status": "planned",
            "device_or_location": item.get("device_or_location"),
            "source_container": item.get("source_container"),
            "target_container": item.get("target_container"),
            "error_code": None,
        }
        for item in commands
    ]
    if protocol and simulation_report and commands:
        from app.services.protocols.commands import ProtocolRunPreview

        rows = preview_rows(ProtocolRunPreview.from_dict(run_preview), simulation_report)
    return {
        "protocol_summary": protocol_summary(protocol) if protocol else {},
        "validation_summary": validation_summary(validation_report) if validation_report else {},
        "simulation_summary": simulation_summary(simulation_report) if simulation_report else {},
        "run_preview": rows,
    }


def handle_workflow_audit_intent(
    session_id: str,
    conversation_history: list,
    decision: WorkflowAuditDecision,
) -> ChatResponse:
    strain_id = decision.target.canonical_id if decision.target else None
    audit = inspect_workflow_activity(
        session_id=session_id,
        strain_id=strain_id,
        pending_id=decision.pending_id,
    )
    if decision.speech_act == SpeechAct.CONFIRM:
        if decision.pending_id is None:
            reply = "聊天路径不能审批或执行高风险 Workflow。请在待审批列表中选择明确的 pending_id 并通过审批按钮操作。"
        elif not audit["pending"]:
            reply = f"没有找到 pending {decision.pending_id}，不会执行任何 Workflow。"
        else:
            status = audit["pending"][0].get("status")
            reply = (
                f"pending {decision.pending_id} 当前状态为 {status}。"
                "聊天路径不会批准它；请使用前端审批按钮。"
            )
    elif audit["pending_count"] == 0:
        reply = "没有查询到与本会话或目标品系关联的传代 Pending，因此也没有可证明的执行记录。"
    else:
        pending_lines = [
            f"pending {item.get('id')}: status={item.get('status')}"
            for item in audit["pending"]
        ]
        run_lines = [
            f"run {item.get('id')}: status={item.get('status')}, "
            f"physical_execution={item.get('physical_execution')}, persisted={item.get('persisted')}"
            for item in audit["workflow_runs"]
        ]
        reply = "查询到真实 Workflow 记录：\n- " + "\n- ".join(pending_lines + run_lines)
    return _complete_chat_response(
        session_id,
        conversation_history,
        {
            "agent_output": {
                "action": "workflow_audit",
                "status": "success",
                **audit,
                "approval_via_chat": False,
            },
            "natural_reply": reply,
        },
    )


async def handle_workflow_intent(
    session_id: str,
    conversation_history: list,
    decision: WorkflowDecision,
    agent_run_id: str | None = None,
    source_message: str | None = None,
    tool_executor=None,
) -> ChatResponse:
    if decision.speech_act == SpeechAct.CANCEL:
        clear_workflow_request_state(session_id)
        return _complete_chat_response(
            session_id,
            conversation_history,
            {
                "agent_output": {
                    "action": "workflow_request_cancelled",
                    "status": "cancelled",
                },
                "natural_reply": "已取消尚未创建审批单的传代 Workflow 请求。",
            },
        )

    strain_id = decision.target.canonical_id if decision.target else None
    if not strain_id:
        state = build_workflow_request_state(
            session_id,
            agent_run_id or "",
            source_message or "",
        )
        save_workflow_request_state(session_id, state)
        append_decision_event({
            "event": "workflow_request_collecting",
            "session_id": session_id,
            "agent_run_id": agent_run_id,
            "route_kind": decision.kind.value,
            "workflow_name": "subculture",
            "missing_fields": ["strain_id"],
        })
        return _complete_chat_response(
            session_id,
            conversation_history,
            {
                "agent_output": {
                    "action": "workflow_request_clarification",
                    "status": "needs_more_info",
                    "risk_level": "high",
                    "candidates": [],
                    "missing_fields": ["strain_id"],
                },
                "natural_reply": "传代 Workflow 需要明确的 strain_id。补充目标后系统只会创建待审批请求，不会直接执行硬件。",
            },
        )
    active_request = get_workflow_request_state(session_id)
    pending_source_message = (
        active_request.get("source_message")
        if active_request
        else source_message
    )
    request_agent_run_id = (
        active_request.get("agent_run_id")
        if active_request and active_request.get("agent_run_id")
        else agent_run_id
    )
    try:
        if tool_executor is not None:
            tool_result = await tool_executor(
                "trigger_subculture_workflow",
                {
                    "strain_id": strain_id,
                    "session_id": session_id,
                    "agent_run_id": request_agent_run_id,
                    "source_message": pending_source_message or source_message,
                },
                allow_high_risk=True,
            )
            agent_report = tool_result.get("response_payload") or {}
            pending_id = agent_report.get("pending_id") or tool_result.get("pending_id")
            if agent_report.get("status") == "error" or not pending_id:
                raise RuntimeError(agent_report.get("message") or tool_result.get("message") or "workflow_tool_failed")
        else:
            pending_id = strain_service.create_pending_workflow_subculture(
                strain_id,
                requester="LLM_workflow",
                session_id=session_id,
                agent_run_id=request_agent_run_id,
                source_message=pending_source_message,
            )
        pending = strain_service.get_pending_action(pending_id)
        payload = (pending or {}).get("payload") or {}
        data = payload.get("data") or {}
        valid_pending = (
            pending
            and pending.get("status") == "pending"
            and payload.get("type") == "workflow_subculture"
            and data.get("strain_id") == strain_id
        )
        if not valid_pending:
            raise RuntimeError("pending_verification_failed")
        protocol_fields = _workflow_protocol_response_fields(payload)
    except Exception as exc:
        if not active_request:
            recovery_state = build_workflow_request_state(
                session_id,
                request_agent_run_id or "",
                pending_source_message or source_message or "",
            )
            recovery_state["state"] = "ready_to_create"
            recovery_state["required_slots"] = []
            recovery_state["collected_slots"] = {"strain_id": strain_id}
            save_workflow_request_state(session_id, recovery_state)
        response = {
            "agent_output": {
                "action": "workflow_request",
                "status": "error",
                "strain_id": strain_id,
                "reason": str(exc),
            },
            "natural_reply": f"未能为 {strain_id} 创建并验证传代审批请求；当前没有可确认的 pending_id。",
        }
        append_decision_event({
            "event": "workflow_pending_create_failed",
            "session_id": session_id,
            "agent_run_id": agent_run_id,
            "route_kind": decision.kind.value,
            "tool": "trigger_subculture_workflow",
            "strain_id": strain_id,
            "reason_code": str(exc),
        })
        return _complete_chat_response(session_id, conversation_history, response)

    clear_workflow_request_state(session_id)
    response = {
        "agent_output": {
            "action": "workflow_subculture",
            "status": "pending",
            "pending_id": pending_id,
            "strain_id": strain_id,
            "require_confirmation": True,
            "risk_level": "high",
            "confirm_endpoint": "/api/v1/strain/confirm",
            "confirm_payload_example": {"pending_id": pending_id, "approve": True},
        },
        "natural_reply": f"已创建传代 Workflow 待审批请求 pending {pending_id}。只有通过前端审批后，执行服务才会运行流程。",
    }
    response["agent_output"]["requires_approval"] = True
    response["agent_output"].update(protocol_fields)
    response["natural_reply"] = (
        f"已完成传代协议生成、确定性验证和模拟预览，并创建待审批请求 pending {pending_id}。"
        "只有通过前端审批后，执行服务才会从冻结协议恢复并运行流程。"
    )
    append_decision_event({
        "event": "workflow_pending_created",
        "session_id": session_id,
        "agent_run_id": agent_run_id,
        "route_kind": decision.kind.value,
        "tool": "trigger_subculture_workflow",
        "strain_id": strain_id,
        "pending_id": pending_id,
        "response_status": "pending",
    })
    return _complete_chat_response(session_id, conversation_history, response)
