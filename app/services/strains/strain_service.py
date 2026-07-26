from typing import Dict, Any, List
from app.core import database
from app.services.memory.memory_service import append_decision_event
from app.services.protocols import (
    ProtocolBuildError,
    build_protocol_run_preview,
    build_subculture_protocol,
    simulate_protocol,
    validate_protocol,
)
from app.services.protocols.pending_payload import build_protocol_payload


def _source_for_requester(requester: str) -> str:
    return "human_api" if requester == "human_api" else "chat"


def _last_subculture_time_for_update(data: Dict[str, Any], existing: Dict[str, Any]) -> str | None:
    if "days_since_last_subculture" in data:
        requested_days = int(data["days_since_last_subculture"])
        current_days = int(existing.get("days_since_last_subculture") or 0)
        if requested_days != current_days:
            # Let add_algae_strain derive a timestamp that matches the edited day count.
            return None
    if "last_subculture_time" in data:
        return data.get("last_subculture_time") or existing.get("last_subculture_time")
    return existing.get("last_subculture_time")


def _find_existing_pending(action_type: str, strain_id: str) -> Dict[str, Any] | None:
    if not strain_id:
        return None
    for item in database.list_pending_actions(status="pending"):
        payload = item.get("payload") or {}
        data = payload.get("data") or {}
        if payload.get("type") == action_type and data.get("strain_id") == strain_id:
            return item
    return None


def _record_pending_event(
    event: str,
    pending_id: int,
    action_type: str,
    strain_id: str,
    requester: str,
    duplicate: bool = False,
    agent_run_id: str = None,
    session_id: str = None,
) -> None:
    append_decision_event({
        "event": event,
        "agent_run_id": agent_run_id,
        "session_id": session_id,
        "pending_id": pending_id,
        "action_type": action_type,
        "strain_id": strain_id,
        "requester": requester,
        "source": _source_for_requester(requester),
        "duplicate": duplicate,
    })


def create_pending_add_strain(
    payload: Dict[str, Any],
    requester: str = "LLM",
    session_id: str = None,
    agent_run_id: str = None,
    source_message: str = None,
    graph_thread_id: str = None,
    execution_idempotency_key: str = None,
    domain_dedupe_key: str = None,
) -> int:
    """Create a pending action for adding a strain. Returns pending id."""
    strain_id = payload.get("strain_id")
    existing = _find_existing_pending("add_strain", strain_id)
    if existing:
        pid = existing.get("id")
        _record_pending_event(
            "pending_create_deduped",
            pid,
            "add_strain",
            strain_id,
            requester,
            duplicate=True,
            agent_run_id=agent_run_id,
            session_id=session_id,
        )
        return pid

    action_payload = {
        "type": "add_strain",
        "data": {
            **payload,
            "agent_run_id": agent_run_id,
            "session_id": session_id,
            "source_message": source_message,
            "graph_thread_id": graph_thread_id,
        }
    }
    pid = database.insert_pending_action(
        "add_strain",
        action_payload,
        requester=requester,
        risk_level="medium",
        source=_source_for_requester(requester),
        agent_run_id=agent_run_id,
        graph_thread_id=graph_thread_id,
        execution_idempotency_key=execution_idempotency_key,
        domain_dedupe_key=domain_dedupe_key,
    )
    _record_pending_event(
        "pending_created",
        pid,
        "add_strain",
        strain_id,
        requester,
        agent_run_id=agent_run_id,
        session_id=session_id,
    )
    return pid


def create_pending_update_strain(
    payload: Dict[str, Any],
    requester: str = "LLM",
    session_id: str = None,
    agent_run_id: str = None,
    source_message: str = None,
    graph_thread_id: str = None,
    execution_idempotency_key: str = None,
    domain_dedupe_key: str = None,
) -> int:
    """Create a pending action for updating a strain. Returns pending id."""
    strain_id = payload.get("strain_id")
    existing = _find_existing_pending("update_strain", strain_id)
    if existing:
        pid = existing.get("id")
        _record_pending_event(
            "pending_create_deduped",
            pid,
            "update_strain",
            strain_id,
            requester,
            duplicate=True,
            agent_run_id=agent_run_id,
            session_id=session_id,
        )
        return pid

    action_payload = {
        "type": "update_strain",
        "data": {
            **payload,
            "agent_run_id": agent_run_id,
            "session_id": session_id,
            "source_message": source_message,
            "graph_thread_id": graph_thread_id,
        }
    }
    pid = database.insert_pending_action(
        "update_strain",
        action_payload,
        requester=requester,
        risk_level="medium",
        source=_source_for_requester(requester),
        agent_run_id=agent_run_id,
        graph_thread_id=graph_thread_id,
        execution_idempotency_key=execution_idempotency_key,
        domain_dedupe_key=domain_dedupe_key,
    )
    _record_pending_event(
        "pending_created",
        pid,
        "update_strain",
        strain_id,
        requester,
        agent_run_id=agent_run_id,
        session_id=session_id,
    )
    return pid


def create_pending_delete_strain(
    strain_id: str,
    requester: str = "LLM",
    session_id: str = None,
    agent_run_id: str = None,
    source_message: str = None,
    graph_thread_id: str = None,
    execution_idempotency_key: str = None,
    domain_dedupe_key: str = None,
) -> int:
    """Create a pending action for deleting a strain. Returns pending id."""
    existing = _find_existing_pending("delete_strain", strain_id)
    if existing:
        pid = existing.get("id")
        _record_pending_event(
            "pending_create_deduped",
            pid,
            "delete_strain",
            strain_id,
            requester,
            duplicate=True,
            agent_run_id=agent_run_id,
            session_id=session_id,
        )
        return pid

    action_payload = {
        "type": "delete_strain",
        "data": {
            "strain_id": strain_id,
            "agent_run_id": agent_run_id,
            "session_id": session_id,
            "source_message": source_message,
            "graph_thread_id": graph_thread_id,
        }
    }
    pid = database.insert_pending_action(
        "delete_strain",
        action_payload,
        requester=requester,
        risk_level="medium",
        source=_source_for_requester(requester),
        agent_run_id=agent_run_id,
        graph_thread_id=graph_thread_id,
        execution_idempotency_key=execution_idempotency_key,
        domain_dedupe_key=domain_dedupe_key,
    )
    _record_pending_event(
        "pending_created",
        pid,
        "delete_strain",
        strain_id,
        requester,
        agent_run_id=agent_run_id,
        session_id=session_id,
    )
    return pid


def create_pending_workflow_subculture(
    strain_id: str,
    requester: str = "LLM_workflow",
    session_id: str = None,
    agent_run_id: str = None,
    source_message: str = None,
    graph_thread_id: str = None,
    execution_idempotency_key: str = None,
    domain_dedupe_key: str = None,
) -> int:
    """Create a high-risk pending workflow request. Returns pending id."""
    current_status = database.get_algae_status(strain_id)
    if not current_status:
        raise ValueError("strain_not_found")

    existing = _find_existing_pending("workflow_subculture", strain_id)
    if existing:
        pid = existing.get("id")
        _record_pending_event(
            "pending_create_deduped",
            pid,
            "workflow_subculture",
            strain_id,
            requester,
            duplicate=True,
            agent_run_id=agent_run_id,
            session_id=session_id,
        )
        return pid

    try:
        protocol = build_subculture_protocol(
            strain_id=strain_id,
            strain_snapshot=current_status,
            session_id=session_id,
            agent_run_id=agent_run_id,
            source_message=source_message,
            created_by=requester,
        )
    except ProtocolBuildError as exc:
        append_decision_event({
            "event": "protocol_build_failed",
            "agent_run_id": agent_run_id,
            "session_id": session_id,
            "strain_id": strain_id,
            **exc.to_dict(),
        })
        raise ValueError(exc.code) from exc

    append_decision_event({
        "event": "protocol_built",
        "agent_run_id": agent_run_id,
        "session_id": session_id,
        "strain_id": strain_id,
        "protocol_id": protocol.protocol_id,
        "protocol_hash": protocol.protocol_hash,
    })
    validation_report = validate_protocol(
        protocol,
        strain_snapshot=current_status,
        all_strains=database.list_algae_status(),
        pending_actions=database.list_pending_actions(status="pending"),
        enforce_duplicate_pending=False,
    )
    if not validation_report.valid:
        append_decision_event({
            "event": "protocol_validation_failed",
            "agent_run_id": agent_run_id,
            "session_id": session_id,
            "strain_id": strain_id,
            "protocol_id": protocol.protocol_id,
            "protocol_hash": protocol.protocol_hash,
            "error_codes": [issue.code for issue in validation_report.errors],
        })
        first_error = validation_report.errors[0].code if validation_report.errors else "protocol_invalid"
        raise ValueError(first_error)
    append_decision_event({
        "event": "protocol_validation_passed",
        "agent_run_id": agent_run_id,
        "session_id": session_id,
        "strain_id": strain_id,
        "protocol_id": protocol.protocol_id,
        "protocol_hash": protocol.protocol_hash,
    })
    append_decision_event({
        "event": "protocol_validated",
        "agent_run_id": agent_run_id,
        "session_id": session_id,
        "strain_id": strain_id,
        "protocol_id": protocol.protocol_id,
        "protocol_hash": protocol.protocol_hash,
        "valid": True,
        "check_count": len(validation_report.checks),
    })

    append_decision_event({
        "event": "protocol_simulation_started",
        "agent_run_id": agent_run_id,
        "session_id": session_id,
        "strain_id": strain_id,
        "protocol_id": protocol.protocol_id,
        "protocol_hash": protocol.protocol_hash,
    })
    simulation_report = simulate_protocol(protocol)
    if not simulation_report.success:
        append_decision_event({
            "event": "protocol_simulation_failed",
            "agent_run_id": agent_run_id,
            "session_id": session_id,
            "strain_id": strain_id,
            "protocol_id": protocol.protocol_id,
            "protocol_hash": protocol.protocol_hash,
            "failure_code": simulation_report.failure_code,
        })
        raise ValueError(simulation_report.failure_code or "SIMULATION_STEP_FAILED")
    append_decision_event({
        "event": "protocol_simulation_passed",
        "agent_run_id": agent_run_id,
        "session_id": session_id,
        "strain_id": strain_id,
        "protocol_id": protocol.protocol_id,
        "protocol_hash": protocol.protocol_hash,
        "simulation_id": simulation_report.simulation_id,
    })
    append_decision_event({
        "event": "protocol_simulated",
        "agent_run_id": agent_run_id,
        "session_id": session_id,
        "strain_id": strain_id,
        "protocol_id": protocol.protocol_id,
        "protocol_hash": protocol.protocol_hash,
        "success": True,
        "simulation_id": simulation_report.simulation_id,
        "step_count": len(simulation_report.step_results),
    })

    run_preview = build_protocol_run_preview(protocol)
    append_decision_event({
        "event": "run_preview_created",
        "agent_run_id": agent_run_id,
        "session_id": session_id,
        "strain_id": strain_id,
        "protocol_id": protocol.protocol_id,
        "protocol_hash": protocol.protocol_hash,
        "command_count": len(run_preview.commands),
    })

    expected_generation = int(current_status.get("generation_number") or 0)
    action_payload = {
        "type": "workflow_subculture",
        "action_type": "execute_experiment_protocol",
        "data": {
            "strain_id": strain_id,
            "workflow": "subculture",
            "agent_run_id": agent_run_id,
            "session_id": session_id,
            "source_message": source_message,
            "graph_thread_id": graph_thread_id,
            "expected_generation": expected_generation,
            "protocol_id": protocol.protocol_id,
            "protocol_hash": protocol.protocol_hash,
        },
        "protocol": build_protocol_payload(
            protocol=protocol,
            validation_report=validation_report,
            simulation_report=simulation_report,
            run_preview=run_preview,
            expected_generation=expected_generation,
            target_strain_id=strain_id,
        ),
    }
    pid = database.insert_pending_action(
        "workflow_subculture",
        action_payload,
        requester=requester,
        risk_level="high",
        source=_source_for_requester(requester),
        agent_run_id=agent_run_id,
        graph_thread_id=graph_thread_id,
        execution_idempotency_key=execution_idempotency_key,
        domain_dedupe_key=domain_dedupe_key,
    )
    _record_pending_event(
        "pending_created",
        pid,
        "workflow_subculture",
        strain_id,
        requester,
        agent_run_id=agent_run_id,
        session_id=session_id,
    )
    append_decision_event({
        "event": "protocol_pending_created",
        "agent_run_id": agent_run_id,
        "session_id": session_id,
        "pending_id": pid,
        "strain_id": strain_id,
        "protocol_id": protocol.protocol_id,
        "protocol_hash": protocol.protocol_hash,
    })
    return pid


def list_pending_actions(status: str = "pending") -> List[Dict[str, Any]]:
    return database.list_pending_actions(status=status)


def get_pending_action(action_id: int) -> Dict[str, Any]:
    return database.get_pending_action(action_id)


def approve_pending_action(action_id: int) -> Dict[str, Any]:
    pending = get_pending_action(action_id)
    if not pending:
        return {"status": "error", "reason": "not_found"}
    if pending.get("status", "pending") != "pending":
        return {"status": "error", "reason": "already_reviewed"}

    payload = pending.get("payload", {})
    action_type = payload.get("type")
    data = payload.get("data", {})
    try:
        if action_type == "workflow_subculture":
            return {"status": "error", "reason": "workflow_requires_async_approval"}
        if action_type == "add_strain":
            database.add_algae_strain(
                strain_id=data.get("strain_id"),
                name_cn=data.get("name_cn"),
                name_en=data.get("name_en"),
                generation_number=int(data.get("generation_number", 1)),
                days_since_last_subculture=int(data.get("days_since_last_subculture", 0))
            )
            database.update_pending_action_status(action_id, "approved", reviewed_by="human")
            append_decision_event({
                "event": "pending_reviewed",
                "pending_id": action_id,
                "action": "approved",
                "action_type": action_type,
                "strain_id": data.get("strain_id"),
                "reviewed_by": "human",
                "executed": True,
            })
            return {
                "status": "success",
                "action": "approved",
                "executed": True,
                "pending_id": action_id,
                "action_type": action_type,
                "strain_id": data.get("strain_id"),
                "msg": f"已添加/更新品系 {data.get('strain_id')}",
            }
        elif action_type == "update_strain":
            sid = data.get("strain_id")
            existing = database.get_algae_status(sid)
            if not existing:
                return {"status": "error", "reason": "not_found"}
            target_sid = data.get("new_strain_id") or sid
            database.add_algae_strain(
                strain_id=target_sid,
                name_cn=data.get("name_cn") or existing.get("name_cn"),
                name_en=data.get("name_en") or existing.get("name_en"),
                generation_number=int(data.get("generation_number", existing.get("generation_number", 1))),
                days_since_last_subculture=int(data.get("days_since_last_subculture", existing.get("days_since_last_subculture", 0))),
                last_subculture_time=_last_subculture_time_for_update(data, existing),
            )
            if target_sid != sid:
                database.delete_algae_strain(sid)
            database.update_pending_action_status(action_id, "approved", reviewed_by="human")
            append_decision_event({
                "event": "pending_reviewed",
                "pending_id": action_id,
                "action": "approved",
                "action_type": action_type,
                "strain_id": sid,
                "reviewed_by": "human",
                "executed": True,
            })
            return {
                "status": "success",
                "action": "approved",
                "executed": True,
                "pending_id": action_id,
                "action_type": action_type,
                "strain_id": sid,
                "msg": f"已更新品系 {sid}",
            }
        elif action_type == "delete_strain":
            sid = data.get("strain_id")
            ok = database.delete_algae_strain(sid)
            if ok:
                database.update_pending_action_status(action_id, "approved", reviewed_by="human")
                append_decision_event({
                    "event": "pending_reviewed",
                    "pending_id": action_id,
                    "action": "approved",
                    "action_type": action_type,
                    "strain_id": sid,
                    "reviewed_by": "human",
                    "executed": True,
                })
                return {
                    "status": "success",
                    "action": "approved",
                    "executed": True,
                    "pending_id": action_id,
                    "action_type": action_type,
                    "strain_id": sid,
                    "msg": f"已删除品系 {sid}",
                }
            else:
                return {"status": "error", "reason": "not_found"}
        else:
            return {"status": "error", "reason": "unsupported_action"}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def deny_pending_action(action_id: int) -> Dict[str, Any]:
    pending = get_pending_action(action_id)
    if not pending:
        return {"status": "error", "reason": "not_found"}
    if pending.get("status", "pending") != "pending":
        return {"status": "error", "reason": "already_reviewed"}

    ok = database.update_pending_action_status(action_id, "denied", reviewed_by="human")
    if ok:
        payload = pending.get("payload", {})
        data = payload.get("data", {})
        append_decision_event({
            "event": "pending_reviewed",
            "pending_id": action_id,
            "action": "denied",
            "action_type": payload.get("type"),
            "strain_id": data.get("strain_id"),
            "reviewed_by": "human",
            "executed": False,
        })
        return {
            "status": "success",
            "action": "denied",
            "executed": False,
            "pending_id": action_id,
            "action_type": payload.get("type"),
            "strain_id": data.get("strain_id"),
            "msg": "已拒绝待确认请求，未执行任何数据修改",
        }
    return {"status": "error", "reason": "not_found"}


async def approve_pending_action_async(action_id: int) -> Dict[str, Any]:
    pending = get_pending_action(action_id)
    if not pending:
        return {"status": "error", "reason": "not_found"}
    payload = pending.get("payload", {})
    action_type = payload.get("type")
    if action_type != "workflow_subculture":
        return approve_pending_action(action_id)
    from app.services.workflows.workflow_approval_service import approve_workflow_pending

    return await approve_workflow_pending(action_id)


async def execute_approved_pending_action(
    action_id: int,
    *,
    execution_idempotency_key: str | None = None,
    finalize_pending: bool = True,
) -> Dict[str, Any]:
    pending = get_pending_action(action_id)
    if not pending:
        return {"status": "error", "reason": "not_found", "pending_id": action_id}
    if pending.get("executed_at"):
        return {
            "status": "success",
            "action": "already_executed",
            "executed": False,
            "idempotent": True,
            "pending_id": action_id,
            "result": pending.get("execution_result"),
        }
    if pending.get("status") == "denied":
        return {
            "status": "success",
            "action": "denied",
            "executed": False,
            "pending_id": action_id,
            "action_type": pending.get("action_type"),
        }
    if pending.get("status") != "approved":
        return {
            "status": "error",
            "reason": "not_approved",
            "pending_id": action_id,
            "pending_status": pending.get("status"),
        }

    payload = pending.get("payload", {})
    action_type = payload.get("type")
    data = payload.get("data", {})
    if (
        execution_idempotency_key
        and action_type in {"add_strain", "update_strain", "delete_strain"}
    ):
        result = database.apply_strain_mutation_once(
            action_type=action_type,
            data=data,
            execution_idempotency_key=execution_idempotency_key,
        )
        if result.get("status") == "success":
            append_decision_event(
                {
                    "event": "approved_pending_executed",
                    "pending_id": action_id,
                    "action_type": action_type,
                    "strain_id": data.get("strain_id"),
                    "agent_run_id": data.get("agent_run_id"),
                    "session_id": data.get("session_id"),
                    "execution_idempotency_key": execution_idempotency_key,
                }
            )
            result["pending_id"] = action_id
            if finalize_pending:
                database.mark_pending_executed(action_id, result)
        return result
    if action_type == "workflow_subculture":
        from app.services.workflows.workflow_approval_service import approve_workflow_pending

        result = await approve_workflow_pending(action_id)
        if result.get("status") == "success":
            database.mark_pending_executed(action_id, result)
        return result

    try:
        if action_type == "add_strain":
            database.add_algae_strain(
                strain_id=data.get("strain_id"),
                name_cn=data.get("name_cn"),
                name_en=data.get("name_en"),
                generation_number=int(data.get("generation_number", 1)),
                days_since_last_subculture=int(data.get("days_since_last_subculture", 0)),
                last_subculture_time=data.get("last_subculture_time"),
            )
            result = {
                "status": "success",
                "action": "approved",
                "executed": True,
                "pending_id": action_id,
                "action_type": action_type,
                "strain_id": data.get("strain_id"),
            }
        elif action_type == "update_strain":
            sid = data.get("strain_id")
            existing = database.get_algae_status(sid)
            if not existing:
                return {"status": "error", "reason": "not_found", "pending_id": action_id}
            target_sid = data.get("new_strain_id") or sid
            database.add_algae_strain(
                strain_id=target_sid,
                name_cn=data.get("name_cn") or existing.get("name_cn"),
                name_en=data.get("name_en") or existing.get("name_en"),
                generation_number=int(data.get("generation_number", existing.get("generation_number", 1))),
                days_since_last_subculture=int(data.get("days_since_last_subculture", existing.get("days_since_last_subculture", 0))),
                last_subculture_time=_last_subculture_time_for_update(data, existing),
            )
            if target_sid != sid:
                database.delete_algae_strain(sid)
            result = {
                "status": "success",
                "action": "approved",
                "executed": True,
                "pending_id": action_id,
                "action_type": action_type,
                "strain_id": sid,
            }
        elif action_type == "delete_strain":
            sid = data.get("strain_id")
            if not database.delete_algae_strain(sid):
                return {"status": "error", "reason": "not_found", "pending_id": action_id}
            result = {
                "status": "success",
                "action": "approved",
                "executed": True,
                "pending_id": action_id,
                "action_type": action_type,
                "strain_id": sid,
            }
        else:
            return {"status": "error", "reason": "unsupported_action", "pending_id": action_id}
        append_decision_event({
            "event": "approved_pending_executed",
            "pending_id": action_id,
            "action_type": action_type,
            "strain_id": data.get("strain_id"),
            "agent_run_id": data.get("agent_run_id"),
            "session_id": data.get("session_id"),
        })
        if finalize_pending:
            database.mark_pending_executed(action_id, result)
        return result
    except Exception as exc:
        return {"status": "error", "reason": str(exc), "pending_id": action_id}
