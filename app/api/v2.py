from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Cookie, Depends, File, Form, Header, HTTPException, Response, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core import database
from app.core.config import load_memory
from app.core.model_registry import classify_provider_error
from app.core.db import schema as db_schema
from app.core.db import scientific as scientific_db
from app.core.db import assistant_conversations, conversation_tasks, operations, workflow_runs
from app.core.session_security import (
    SESSION_COOKIE,
    create_browser_session,
    delete_browser_session,
    get_browser_session,
    require_session_approver,
    require_session_scientist,
    require_session_viewer,
)
from app.core.security import ApiPrincipal
from app.schemas.algae import ChatRequest
from app.models.rag_schema import RagGenerationCreateRequest, RagQueryRequest
from app.services.chat.chat_service import handle_chat
from app.services.user_memory.service import process_completed_turn
from app.services.agent_runtime.state import RuntimeRequestContext
from app.services.control_plane import dashboard, get_events, get_run, list_approvals, list_runs
from app.services.scientific.service import run_scientific_task
from app.services.rag.service import answer_rag_question
from app.services.rag.knowledge_asset_service import index_source, list_sources, save_uploaded_source
from app.services.rag.generation_service import activate_generation, build_shadow_generation
from app.services.strains import strain_service
from app.services.workflows.simulation_service import simulation_runs
from app.services.workflows.workflow_approval_service import grant_workflow_approval
from app.services.proposal_execution_validation import validate_pending_for_execution
from app.services.trusted_execution_dispatch import dispatch_trusted_execution
from app.services.scientific.demo_data import scientific_demo_file
from app.services.scientific.importer import DatasetImportError, parse_scientific_dataset
from app.services.evaluation.reporting import (
    load_latest_report,
    load_report,
    sanitize_report,
)
from app.core.workspaces import (
    create_test_workspace,
    delete_test_workspace,
    get_workspace,
    list_test_workspaces,
    register_run_workspace,
    reset_test_workspace,
    workspace_for_run,
    workspace_scope,
    current_workspace,
    WorkspaceContext,
)


router = APIRouter(prefix="/api/v2", tags=["control-plane-v2"])


def _dispatch_trusted_execution(
    background_tasks: BackgroundTasks,
    approval_id: int,
    workflow_run_id: int | None = None,
    workspace: WorkspaceContext | None = None,
    operation_id: str | None = None,
) -> None:
    dispatch_trusted_execution(
        background_tasks,
        approval_id,
        workflow_run_id,
        workspace,
        operation_id,
    )


class LoginRequest(BaseModel):
    remember: bool = False


class RunCreateRequest(BaseModel):
    kind: Literal["agent", "scientific", "simulation"]
    input: dict[str, Any] = Field(default_factory=dict)


class ApprovalDecisionRequest(BaseModel):
    decision: Literal["approved", "denied"]
    reason: str | None = Field(default=None, max_length=500)
    run_id: str | None = Field(default=None, max_length=200)


class ExecuteRequest(BaseModel):
    idempotency_key: str = Field(min_length=8, max_length=200)


class ManualResolutionRequest(BaseModel):
    resolution: Literal["completed", "failed"]
    note: str | None = Field(default=None, max_length=500)


class ManualCommitRequest(BaseModel):
    completed_at: str = Field(min_length=10, max_length=40)
    note: str | None = Field(default=None, max_length=500)
    idempotency_key: str = Field(min_length=8, max_length=200)


class ScenarioRunRequest(BaseModel):
    parameters: dict[str, Any] = Field(default_factory=dict)


class ConversationCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=80)


class ConversationUpdateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=80)
    status: Literal["active", "archived"] | None = None


class ConversationMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=10000)
    client_message_id: str = Field(min_length=8, max_length=100)


class StrainResourceRequest(BaseModel):
    operation: Literal["add", "update", "delete"]
    data: dict[str, Any] = Field(default_factory=dict)


async def _process_assistant_message(
    *,
    operation_id: str,
    assistant_message_id: int,
    user_message_id: int,
    conversation_id: str,
    content: str,
    workspace: WorkspaceContext,
    owner: str,
    role: str,
) -> None:
    with workspace_scope(workspace):
        try:
            operations.update_operation(
                operation_id,
                status="running",
                phase="understanding",
                message="正在理解问题",
                progress=0.08,
            )
            result = await handle_chat(
                ChatRequest(message=content, session_id=conversation_id),
                request_context=RuntimeRequestContext(
                    owner=owner,
                    role=role,
                    workspace_id=workspace.id,
                    conversation_id=conversation_id,
                ),
            )
            operations.update_operation(
                operation_id,
                phase="response",
                message="正在整理回复",
                progress=0.9,
            )
            output = result.model_dump()
            task_id = (output.get("agent_output") or {}).get("task_id")
            task_status = (output.get("agent_output") or {}).get("task_status")
            agent_run_id = (output.get("agent_output") or {}).get("agent_run_id")
            if not agent_run_id:
                matches = list_runs(kind="agent", limit=1)
                agent_run_id = matches[0]["source_ref"]["id"] if matches else None
            canonical_run_id = f"agent:{agent_run_id}" if agent_run_id else None
            if canonical_run_id:
                run_detail = get_run(canonical_run_id)
                if run_detail and run_detail.get("replans"):
                    operations.update_operation(
                        operation_id,
                        phase="replan",
                        message="Agent 正在重新规划",
                        progress=0.82,
                        related_run_id=canonical_run_id,
                    )
            assistant_conversations.update_message(
                assistant_message_id,
                content=output.get("natural_reply") or "任务已处理。",
                run_id=canonical_run_id,
                structured=output.get("agent_output"),
                status="completed",
                operation_id=operation_id,
                task_id=task_id,
            )
            assistant_conversations.update_message(user_message_id, task_id=task_id)
            operation_metadata = dict((operations.get_operation(operation_id) or {}).get("metadata") or {})
            operation_metadata.update({"task_id": task_id, "task_status": task_status})
            operations.update_operation(
                operation_id,
                status="succeeded",
                phase="completed",
                message="回复已完成",
                progress=1.0,
                related_run_id=canonical_run_id,
                retryable=False,
                related_task_id=task_id,
                metadata=operation_metadata,
            )
            try:
                process_completed_turn(
                    owner_id=owner,
                    workspace_id=workspace.id,
                    conversation_id=conversation_id,
                    user_message_id=user_message_id,
                    user_text=content,
                    assistant_text=output.get("natural_reply") or "",
                    source_run_id=canonical_run_id,
                )
            except Exception:
                # User-memory enrichment is best effort and must never change a
                # successfully completed Assistant operation into a failure.
                pass
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            failure = classify_provider_error(exc)
            active_task = conversation_tasks.get_active_task(
                conversation_id,
                owner=owner,
                workspace_id=workspace.id,
            )
            if not active_task:
                latest_task = conversation_tasks.get_latest_task(conversation_id)
                if (
                    latest_task
                    and latest_task.get("owner") == owner
                    and latest_task.get("workspace_id") == workspace.id
                    and latest_task.get("status")
                    not in conversation_tasks.TERMINAL_STATUSES
                ):
                    active_task = latest_task
            failed_task_id = None
            if active_task and active_task.get("status") not in conversation_tasks.TERMINAL_STATUSES:
                try:
                    failed_task = conversation_tasks.update_task(
                        active_task["id"],
                        status="failed",
                        proposed_action={
                            **(active_task.get("proposed_action") or {}),
                            "failure": {
                                "code": failure["code"],
                                "category": failure["category"],
                                "error_type": type(exc).__name__,
                            },
                        },
                        event_type="task_failed",
                    )
                    failed_task_id = failed_task["id"]
                except Exception:
                    failed_task_id = active_task.get("id")
            assistant_conversations.update_message(
                assistant_message_id,
                content=failure["public_message"],
                status="failed",
                operation_id=operation_id,
                structured={
                    "action": "assistant_error",
                    "outcome_status": "failed",
                    "error_code": failure["code"],
                    "error_category": failure["category"],
                    "retryable": failure["retryable"],
                    "task_id": failed_task_id,
                },
            )
            operations.update_operation(
                operation_id,
                status="failed",
                phase="failed",
                message="回复生成失败",
                error_code=failure["code"],
                error_message=message,
                retryable=failure["retryable"],
                related_task_id=failed_task_id,
            )


def _index_knowledge_operation(
    *,
    operation_id: str,
    source_id: str,
    workspace: WorkspaceContext,
    rebuild: bool = False,
) -> None:
    with workspace_scope(workspace):
        try:
            operations.update_operation(
                operation_id,
                status="running",
                phase="extracting",
                message="正在提取文件内容",
                progress=0.18,
            )
            operations.update_operation(
                operation_id, phase="chunking", message="正在分块并建立索引", progress=0.55
            )
            index_source(source_id, rebuild=rebuild)
            source = database.get_rag_knowledge_source(source_id) or {}
            if source.get("ingestion_status") == "failed":
                raise RuntimeError(str(source.get("last_error") or "knowledge_index_failed"))
            operations.update_operation(
                operation_id, status="succeeded", phase="completed", message="知识文件索引完成",
                progress=1.0, retryable=False,
            )
        except Exception as exc:
            operations.update_operation(
                operation_id, status="failed", phase="failed", message="知识文件索引失败",
                error_code=exc.__class__.__name__, error_message=str(exc), retryable=True,
            )


def _build_rag_generation_operation(
    *, operation_id: str, generation_id: str | None, source_roots: list[str], workspace: WorkspaceContext
) -> None:
    with workspace_scope(workspace):
        try:
            operations.update_operation(
                operation_id, status="running", phase="discovering", message="正在发现正式知识源", progress=0.08
            )
            generation = build_shadow_generation(generation_id=generation_id, source_roots=source_roots or None)
            if generation.get("status") != "ready":
                raise RuntimeError(str(generation.get("error_message") or "generation_release_gates_failed"))
            operations.update_operation(
                operation_id, status="succeeded", phase="ready", message="影子索引已通过发布门禁",
                progress=1.0, retryable=False, metadata={"generation_id": generation["generation_id"]},
            )
        except Exception as exc:
            operations.update_operation(
                operation_id, status="failed", phase="failed", message="影子索引构建或门禁失败",
                error_code=exc.__class__.__name__, error_message=str(exc), retryable=True,
            )


def _public_chat_history(session_id: str) -> list[dict[str, str]]:
    history = load_memory(session_id)
    return [
        {"role": item["role"], "content": str(item.get("content") or "")}
        for item in history
        if isinstance(item, dict)
        and item.get("role") in {"user", "assistant"}
        and str(item.get("content") or "").strip()
    ]


def _conversation_for_owner(conversation_id: str, owner: str) -> dict[str, Any]:
    conversation = assistant_conversations.get_conversation(conversation_id)
    if (
        not conversation
        or conversation.get("owner") != owner
        or conversation.get("workspace_id") != current_workspace().id
    ):
        raise HTTPException(status_code=404, detail="conversation_not_found")
    return conversation


def _import_legacy_conversation(owner: str) -> None:
    if assistant_conversations.get_legacy_conversation("react-assistant"):
        return
    messages = _public_chat_history("react-assistant")
    if not messages:
        return
    conversation = assistant_conversations.create_conversation(
        owner,
        workspace_id=current_workspace().id,
        title="历史对话",
        legacy_session_id="react-assistant",
    )
    for index, message in enumerate(messages):
        assistant_conversations.append_message(
            conversation["id"],
            role=message["role"],
            content=message["content"],
            client_message_id=f"legacy-{index:06d}",
        )


TEST_SCENARIOS = [
    {
        "id": "scientific-two-cycle", "name": "两轮科学闭环", "kind": "scientific",
        "description": "质量检查、容量修补、审批、数字孪生与结果回流。", "requires_dataset": True,
        "assertions": ["Verifier pass", "PlanPatch recorded", "approval gate reached", "simulation_only"],
    },
    {
        "id": "scientific-diagnose", "name": "只诊断不执行", "kind": "scientific",
        "description": "验证只读诊断不会产生审批或执行。", "requires_dataset": True,
        "assertions": ["diagnosis artifact", "no pending approval"],
    },
    {
        "id": "subculture-happy", "name": "传代仿真 · 正常", "kind": "simulation",
        "description": "培养瓶传代、96 孔板检测与机器人搬运完整联动。", "fault": None,
        "assertions": ["simulation_only", "safe completion"],
    },
    {
        "id": "subculture-pump-fault", "name": "传代仿真 · 泵故障", "kind": "simulation",
        "description": "孔板检测后注入培养基泵堵塞，并验证机器人安全回零。", "fault": "pump_a_blocked",
        "assertions": ["fault observed", "safe shutdown"],
    },
    {
        "id": "subculture-manual", "name": "传代仿真 · 人工边界", "kind": "simulation",
        "description": "在工作站装载、孔板交接和培养箱交接处等待人工确认。", "fault": None,
        "assertions": ["WAITING_MANUAL", "manual task resolvable"],
    },
]


@router.post("/session/login")
async def login(
    payload: LoginRequest,
    response: Response,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    session = create_browser_session(x_api_key)
    response.set_cookie(
        SESSION_COOKIE,
        session.token,
        httponly=True,
        samesite="strict",
        secure=False,
        max_age=(12 * 60 * 60 if payload.remember else None),
        path="/",
    )
    return {
        "status": "success", "csrf_token": session.csrf_token,
        "principal": {"name": session.principal.name, "role": session.principal.role},
    }


@router.get("/session")
async def session_info(
    algae_session: str | None = Cookie(default=None, alias=SESSION_COOKIE),
    principal: ApiPrincipal = Depends(require_session_viewer),
):
    session = get_browser_session(algae_session)
    return {
        "status": "success", "csrf_token": session.csrf_token if session else None,
        "principal": {"name": principal.name, "role": principal.role},
        "workspace": {"id": "shared", "name": "共享实验室", "mode": "simulation_only"},
    }


@router.delete("/session")
async def logout(
    response: Response,
    algae_session: str | None = Cookie(default=None, alias=SESSION_COOKIE),
    _: ApiPrincipal = Depends(require_session_viewer),
):
    delete_browser_session(algae_session)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"status": "success"}


@router.get("/dashboard")
async def dashboard_endpoint(_: ApiPrincipal = Depends(require_session_viewer)):
    return {"status": "success", **dashboard()}


@router.get("/evaluations/latest")
async def latest_evaluation_endpoint(
    _: ApiPrincipal = Depends(require_session_viewer),
):
    report = load_latest_report()
    if report is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "evaluation_report_not_found", "message": "No completed evaluation report is available."},
        )
    return sanitize_report(report)


@router.get("/evaluations/{report_id}")
async def evaluation_report_endpoint(
    report_id: str,
    _: ApiPrincipal = Depends(require_session_viewer),
):
    try:
        report = load_report(report_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_evaluation_report_id", "message": "Invalid evaluation report id."},
        )
    if report is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "evaluation_report_not_found", "message": "Evaluation report was not found."},
        )
    return sanitize_report(report)


@router.get("/evaluations/{report_id}/download")
async def download_evaluation_report_endpoint(
    report_id: str,
    _: ApiPrincipal = Depends(require_session_viewer),
):
    try:
        report = load_report(report_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_evaluation_report_id", "message": "Invalid evaluation report id."},
        )
    if report is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "evaluation_report_not_found", "message": "Evaluation report was not found."},
        )
    body = json.dumps(sanitize_report(report), ensure_ascii=False, indent=2, allow_nan=False)
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{report_id}.json"'},
    )


@router.get("/runs")
async def runs_endpoint(
    kind: str | None = None,
    status: str | None = None,
    limit: int = 100,
    _: ApiPrincipal = Depends(require_session_viewer),
):
    return {"status": "success", "runs": list_runs(kind=kind, status=status, limit=limit)}


@router.post("/runs", status_code=202)
async def create_run(payload: RunCreateRequest, principal: ApiPrincipal = Depends(require_session_scientist)):
    values = payload.input
    if payload.kind == "agent":
        message = str(values.get("message") or "").strip()
        if not message:
            raise HTTPException(status_code=422, detail="message_required")
        result = await handle_chat(ChatRequest(message=message, session_id=str(values.get("session_id") or "react-assistant")))
        output = result.model_dump()
        agent_run_id = (output.get("agent_output") or {}).get("agent_run_id")
        if not agent_run_id:
            matches = list_runs(kind="agent", limit=1)
            agent_run_id = (matches[0]["source_ref"]["id"] if matches else None)
        return {"status": "success", "run_id": f"agent:{agent_run_id}" if agent_run_id else None, "response": output}
    if payload.kind == "scientific":
        dataset_id = str(values.get("dataset_id") or "")
        if not dataset_id:
            raise HTTPException(status_code=422, detail="dataset_id_required")
        result = run_scientific_task(
            dataset_id=dataset_id,
            mode=str(values.get("mode") or "diagnose_and_optimize"),
            direction=str(values.get("direction") or "maximize"),
            max_cycles=int(values.get("max_cycles") or 2),
            run_seed=int(values.get("run_seed") or 2025),
            session_id=str(values.get("session_id") or principal.name),
            offline_replay=bool(values.get("offline_replay", True)),
        )
        return {"status": "success", "run_id": f"scientific:{result['id']}", "run": get_run(f"scientific:{result['id']}")}
    run = simulation_runs.create(
        strain_id=str(values.get("strain_id") or "Chlorella_01"),
        generation_number=int(values.get("generation_number") or 1),
        interaction_mode=str(values.get("interaction_mode") or "demo"),
        fault_scenario=values.get("fault_scenario"),
        wavelength_nm=float(values.get("wavelength_nm") or 680),
        simulated_absorbance=float(values.get("simulated_absorbance") or 0.8),
    )
    return {"status": "success", "run_id": f"simulation:{run['run_id']}", "run": get_run(f"simulation:{run['run_id']}")}


@router.get("/assistant/history")
async def assistant_history(
    session_id: str = "react-assistant",
    _: ApiPrincipal = Depends(require_session_viewer),
):
    return {"status": "success", "session_id": session_id, "messages": _public_chat_history(session_id)}


@router.get("/assistant/conversations")
async def assistant_conversation_list(
    status: Literal["active", "archived"] = "active",
    principal: ApiPrincipal = Depends(require_session_viewer),
):
    _import_legacy_conversation(principal.name)
    return {
        "status": "success",
        "conversations": assistant_conversations.list_conversations(
            principal.name,
            workspace_id=current_workspace().id,
            status=status,
        ),
    }


@router.post("/assistant/conversations", status_code=201)
async def assistant_conversation_create(
    payload: ConversationCreateRequest,
    principal: ApiPrincipal = Depends(require_session_scientist),
):
    conversation = assistant_conversations.create_conversation(
        principal.name,
        workspace_id=current_workspace().id,
        title=payload.title or "新对话",
    )
    return {"status": "success", "conversation": conversation}


@router.get("/assistant/conversations/{conversation_id}/messages")
async def assistant_conversation_messages(
    conversation_id: str,
    principal: ApiPrincipal = Depends(require_session_viewer),
):
    conversation = _conversation_for_owner(conversation_id, principal.name)
    return {
        "status": "success",
        "conversation": conversation,
        "messages": assistant_conversations.list_messages(conversation_id),
        "active_task": conversation_tasks.get_active_task(
            conversation_id,
            owner=principal.name,
            workspace_id=current_workspace().id,
        ),
    }


@router.get("/assistant/conversations/{conversation_id}/tasks")
async def assistant_conversation_tasks(
    conversation_id: str,
    scope: Literal["active", "recent"] = "recent",
    principal: ApiPrincipal = Depends(require_session_viewer),
):
    _conversation_for_owner(conversation_id, principal.name)
    workspace = current_workspace()
    return {
        "status": "success",
        "tasks": conversation_tasks.list_tasks(
            conversation_id,
            owner=principal.name,
            workspace_id=workspace.id,
            scope=scope,
        ),
    }


def _task_for_owner(task_id: str, principal: ApiPrincipal) -> dict[str, Any]:
    task = conversation_tasks.get_task(task_id)
    workspace = current_workspace()
    if not task or task.get("owner") != principal.name or task.get("workspace_id") != workspace.id:
        raise HTTPException(status_code=404, detail="task_not_found")
    return task


@router.get("/assistant/tasks/{task_id}")
async def assistant_task_detail(
    task_id: str,
    principal: ApiPrincipal = Depends(require_session_viewer),
):
    task = _task_for_owner(task_id, principal)
    return {
        "status": "success",
        "task": task,
        "events": conversation_tasks.list_task_events(task_id),
    }


@router.post("/assistant/tasks/{task_id}/cancel")
async def assistant_task_cancel(
    task_id: str,
    principal: ApiPrincipal = Depends(require_session_scientist),
):
    task = _task_for_owner(task_id, principal)
    if task.get("status") in conversation_tasks.TERMINAL_STATUSES:
        return {"status": "success", "idempotent": True, "task": task}
    cancelled = conversation_tasks.cancel_task(task_id)
    return {"status": "success", "idempotent": False, "task": cancelled}


@router.post("/assistant/conversations/{conversation_id}/messages", status_code=202)
async def assistant_conversation_message(
    conversation_id: str,
    payload: ConversationMessageRequest,
    background_tasks: BackgroundTasks,
    principal: ApiPrincipal = Depends(require_session_scientist),
):
    conversation = _conversation_for_owner(conversation_id, principal.name)
    if conversation.get("status") != "active":
        raise HTTPException(status_code=409, detail="conversation_archived")
    existing = next(
        (
            item for item in assistant_conversations.list_messages(conversation_id)
            if item.get("client_message_id") == payload.client_message_id
        ),
        None,
    )
    if existing:
        related = next(
            (
                item for item in assistant_conversations.list_messages(conversation_id)
                if item.get("client_message_id") == f"{payload.client_message_id}:assistant"
            ),
            None,
        )
        return {
            "status": "success",
            "idempotent": True,
            "operation": operations.get_operation((related or {}).get("operation_id")) if related else None,
            "messages": assistant_conversations.list_messages(conversation_id),
        }
    workspace = current_workspace()
    operation = operations.create_operation(
        owner=principal.name,
        workspace_id=workspace.id,
        kind="assistant_message",
        label="生成 Assistant 回复",
        phase="queued",
        message="正在提交消息",
        related_entity_type="assistant_conversation",
        related_entity_id=conversation_id,
        metadata={
            "conversation_id": conversation_id,
            "content": payload.content.strip(),
            "client_message_id": payload.client_message_id,
        },
        retryable=True,
    )
    user_message = assistant_conversations.append_message(
        conversation_id,
        role="user",
        content=payload.content.strip(),
        client_message_id=payload.client_message_id,
    )
    assistant_message = assistant_conversations.append_message(
        conversation_id,
        role="assistant",
        content="Agent 正在理解问题…",
        client_message_id=f"{payload.client_message_id}:assistant",
        status="processing",
        operation_id=operation["id"],
    )
    metadata = dict(operation.get("metadata") or {})
    metadata["assistant_message_id"] = assistant_message["id"]
    metadata["user_message_id"] = user_message["id"]
    metadata["owner"] = principal.name
    metadata["role"] = principal.role
    operations.update_operation(operation["id"], metadata=metadata)
    background_tasks.add_task(
        _process_assistant_message,
        operation_id=operation["id"],
        assistant_message_id=int(assistant_message["id"]),
        user_message_id=int(user_message["id"]),
        conversation_id=conversation_id,
        content=payload.content.strip(),
        workspace=workspace,
        owner=principal.name,
        role=principal.role,
    )
    return {
        "status": "accepted",
        "idempotent": False,
        "operation_id": operation["id"],
        "operation": operations.get_operation(operation["id"]),
        "user_message": user_message,
        "assistant_message": assistant_message,
    }


@router.patch("/assistant/conversations/{conversation_id}")
async def assistant_conversation_update(
    conversation_id: str,
    payload: ConversationUpdateRequest,
    principal: ApiPrincipal = Depends(require_session_scientist),
):
    _conversation_for_owner(conversation_id, principal.name)
    conversation = assistant_conversations.update_conversation(
        conversation_id,
        title=payload.title,
        status=payload.status,
    )
    return {"status": "success", "conversation": conversation}


def _operation_for_owner(operation_id: str, principal: ApiPrincipal) -> dict[str, Any]:
    operation = operations.get_operation(operation_id)
    workspace = current_workspace()
    if (
        not operation
        or operation.get("owner") != principal.name
        or operation.get("workspace_id") != workspace.id
    ):
        raise HTTPException(status_code=404, detail="operation_not_found")
    return operation


@router.get("/operations")
async def operation_list(
    scope: Literal["active", "recent"] = "active",
    limit: int = 50,
    principal: ApiPrincipal = Depends(require_session_viewer),
):
    workspace = current_workspace()
    return {
        "status": "success",
        "operations": operations.list_operations(
            principal.name, workspace.id, scope=scope, limit=limit
        ),
    }


@router.get("/operations/stream")
async def operation_stream(principal: ApiPrincipal = Depends(require_session_viewer)):
    workspace = current_workspace()

    async def generate():
        last_digest = ""
        idle_ticks = 0
        while True:
            items = operations.list_operations(principal.name, workspace.id, scope="recent", limit=50)
            digest = json.dumps(
                [(item.get("id"), item.get("status"), item.get("phase"), item.get("updated_at")) for item in items],
                ensure_ascii=False,
            )
            if digest != last_digest:
                last_digest = digest
                idle_ticks = 0
                yield f"event: operations\ndata: {json.dumps(items, ensure_ascii=False, default=str)}\n\n"
            else:
                idle_ticks += 1
                if idle_ticks >= 15:
                    idle_ticks = 0
                    yield ": keep-alive\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/operations/{operation_id}")
async def operation_detail(
    operation_id: str,
    principal: ApiPrincipal = Depends(require_session_viewer),
):
    operation = _operation_for_owner(operation_id, principal)
    return {
        "status": "success",
        "operation": operation,
        "events": operations.list_events(operation_id),
    }


@router.post("/operations/{operation_id}/retry", status_code=202)
async def operation_retry(
    operation_id: str,
    background_tasks: BackgroundTasks,
    principal: ApiPrincipal = Depends(require_session_scientist),
):
    operation = _operation_for_owner(operation_id, principal)
    if operation.get("status") != "failed" or not operation.get("retryable"):
        raise HTTPException(status_code=409, detail="operation_not_retryable")
    metadata = operation.get("metadata") or {}
    if operation.get("kind") == "approval_execution":
        approval_id = int(metadata.get("approval_id") or 0)
        if not approval_id:
            raise HTTPException(status_code=409, detail="approval_not_found")
        pending = database.get_pending_action(approval_id)
        if not pending or pending.get("status") != "approved" or pending.get("executed_at"):
            raise HTTPException(status_code=409, detail="approval_not_retryable")
        database.mark_pending_execution_failed(approval_id, "operation_retry")
        database.queue_pending_execution(approval_id)
        operations.update_operation(
            operation_id,
            status="queued",
            phase="queued",
            message="正在重新启动审批执行",
            progress=0.0,
            retryable=True,
        )
        _dispatch_trusted_execution(
            background_tasks,
            approval_id,
            metadata.get("workflow_run_id"),
            current_workspace(),
            operation_id,
        )
        return {"status": "accepted", "operation_id": operation_id, "operation": operations.get_operation(operation_id)}
    if operation.get("kind") in {"knowledge_index", "knowledge_reindex"}:
        operations.update_operation(
            operation_id,
            status="queued",
            phase="queued",
            message="正在重新提交索引任务",
            progress=0.0,
            retryable=True,
        )
        background_tasks.add_task(
            _index_knowledge_operation,
            operation_id=operation_id,
            source_id=str(metadata.get("source_id") or operation.get("related_entity_id") or ""),
            workspace=current_workspace(),
            rebuild=bool(metadata.get("rebuild")),
        )
        return {"status": "accepted", "operation_id": operation_id, "operation": operations.get_operation(operation_id)}
    if operation.get("kind") != "assistant_message":
        raise HTTPException(status_code=409, detail="retry_not_supported")
    assistant_message_id = int(metadata.get("assistant_message_id") or 0)
    if not assistant_message_id:
        raise HTTPException(status_code=409, detail="assistant_message_not_found")
    assistant_conversations.update_message(
        assistant_message_id,
        content="Agent 正在重新处理…",
        status="processing",
        operation_id=operation_id,
    )
    operations.update_operation(
        operation_id,
        status="queued",
        phase="queued",
        message="正在重新提交",
        progress=0.0,
        retryable=True,
    )
    background_tasks.add_task(
        _process_assistant_message,
        operation_id=operation_id,
        assistant_message_id=assistant_message_id,
        user_message_id=int(metadata.get("user_message_id") or 0),
        conversation_id=str(metadata.get("conversation_id") or ""),
        content=str(metadata.get("content") or ""),
        workspace=current_workspace(),
        owner=str(metadata.get("owner") or principal.name),
        role=str(metadata.get("role") or principal.role),
    )
    return {
        "status": "accepted",
        "operation_id": operation_id,
        "operation": operations.get_operation(operation_id),
    }


@router.get("/runs/{run_id:path}/events")
async def run_events(run_id: str, after: int = 0, _: ApiPrincipal = Depends(require_session_viewer)):
    if not get_run(run_id):
        raise HTTPException(status_code=404, detail="run_not_found")
    return {"status": "success", "events": get_events(run_id, after=after)}


@router.get("/runs/{run_id:path}/stream")
async def run_stream(
    run_id: str,
    after: int = 0,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    _: ApiPrincipal = Depends(require_session_viewer),
):
    if not get_run(run_id):
        raise HTTPException(status_code=404, detail="run_not_found")

    try:
        resume_after = max(after, int(last_event_id or 0))
    except ValueError:
        resume_after = after

    async def generate():
        cursor = resume_after
        snapshot = get_run(run_id)
        yield f"event: snapshot\ndata: {json.dumps(snapshot, ensure_ascii=False, default=str)}\n\n"
        idle_ticks = 0
        while True:
            emitted = False
            for event in get_events(run_id, after=cursor):
                cursor = max(cursor, int(event["sequence"]))
                emitted = True
                yield f"id: {cursor}\nevent: run_event\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            if emitted:
                idle_ticks = 0
            else:
                idle_ticks += 1
                if idle_ticks >= 15:
                    idle_ticks = 0
                    yield ": keep-alive\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/runs/{run_id:path}")
async def run_detail(run_id: str, _: ApiPrincipal = Depends(require_session_viewer)):
    result = get_run(run_id)
    if not result:
        raise HTTPException(status_code=404, detail="run_not_found")
    return {"status": "success", "run": result}


@router.get("/approvals")
async def approvals_endpoint(
    status: str = "all", kind: str | None = None, _: ApiPrincipal = Depends(require_session_viewer),
):
    return {"status": "success", "approvals": list_approvals(status=status, kind=kind)}


@router.post("/approvals/{approval_id}/decision")
async def approval_decision(
    approval_id: int,
    payload: ApprovalDecisionRequest,
    background_tasks: BackgroundTasks,
    principal: ApiPrincipal = Depends(require_session_approver),
):
    owner = workspace_for_run(payload.run_id) if payload.run_id else None
    with workspace_scope(owner) if owner else nullcontext():
        pending = database.get_pending_action(approval_id)
        if not pending:
            raise HTTPException(status_code=404, detail="approval_not_found")
        if pending.get("status") not in {"pending", payload.decision}:
            raise HTTPException(status_code=409, detail="approval_already_decided")
        pending_payload = pending.get("payload") or {}
        pending_type = pending_payload.get("type")
        data = pending_payload.get("data") or {}
        if payload.decision == "denied":
            database.update_pending_action_status(approval_id, "denied", reviewed_by=principal.name, review_reason=payload.reason)
            from app.services.chat.task_coordinator import task_for_pending

            conversation_task = task_for_pending(approval_id)
            if conversation_task:
                conversation_tasks.update_task(
                    conversation_task["id"],
                    expected_version=int(conversation_task["version"]),
                    status="cancelled",
                    proposed_action={
                        **(conversation_task.get("proposed_action") or {}),
                        "cancel_reason": "approval_denied",
                    },
                    event_type="task_approval_denied",
                )
            if pending_type == "scientific_experiment_plan" and data.get("scientific_run_id"):
                scientific_db.update_scientific_run(str(data["scientific_run_id"]), status="denied")
            return {"status": "success", "decision": "denied", "approval_id": approval_id}
        if pending_type == "workflow_subculture":
            result = grant_workflow_approval(approval_id, reviewed_by=principal.name)
            if result.get("status") != "success":
                raise HTTPException(status_code=422, detail=result.get("reason"))
            workflow_id = f"workflow:{result['run_id']}"
            if owner:
                register_run_workspace(workflow_id, owner)
            operation = None
            if not result.get("idempotent"):
                database.queue_pending_execution(approval_id)
                execution_workspace = owner or current_workspace()
                operation = operations.create_operation(
                    owner=principal.name,
                    workspace_id=execution_workspace.id,
                    kind="approval_execution",
                    label="启动传代仿真",
                    phase="approved",
                    message="审批已通过，正在启动仿真",
                    related_run_id=workflow_id,
                    related_entity_type="approval",
                    related_entity_id=str(approval_id),
                    metadata={"approval_id": approval_id, "workflow_run_id": int(result["run_id"])},
                    retryable=True,
                )
                _dispatch_trusted_execution(
                    background_tasks,
                    approval_id,
                    int(result["run_id"]),
                    owner,
                    operation["id"],
                )
            return {
                "status": "success",
                "decision": "approved",
                "approval_id": approval_id,
                "run_id": workflow_id,
                "execution_scheduled": not bool(result.get("idempotent")),
                "operation_id": operation.get("id") if operation else None,
                "operation": operation,
            }
        database.update_pending_action_status(approval_id, "approved", reviewed_by=principal.name, review_reason=payload.reason)
        if pending_type == "scientific_experiment_plan" and data.get("scientific_run_id"):
            source_id = str(data["scientific_run_id"])
            scientific_db.update_scientific_run(source_id, status="ready_to_execute")
            database.queue_pending_execution(approval_id)
            execution_workspace = owner or current_workspace()
            operation = operations.create_operation(
                owner=principal.name, workspace_id=execution_workspace.id,
                kind="approval_execution", label="启动科学方案执行",
                phase="approved", message="审批已通过，正在启动数字孪生",
                related_run_id=f"scientific:{source_id}", related_entity_type="approval",
                related_entity_id=str(approval_id), metadata={"approval_id": approval_id}, retryable=True,
            )
            _dispatch_trusted_execution(
                background_tasks, approval_id, None, owner, operation["id"]
            )
            return {"status": "success", "decision": "approved", "approval_id": approval_id, "run_id": f"scientific:{source_id}", "execution_scheduled": True, "operation_id": operation["id"], "operation": operation}
        if pending_type in {"add_strain", "update_strain", "delete_strain"}:
            database.queue_pending_execution(approval_id)
            execution_workspace = owner or current_workspace()
            operation = operations.create_operation(
                owner=principal.name, workspace_id=execution_workspace.id,
                kind="approval_execution", label="更新品系资源",
                phase="approved", message="审批已通过，正在更新资源",
                related_entity_type="approval", related_entity_id=str(approval_id),
                metadata={"approval_id": approval_id}, retryable=True,
            )
            _dispatch_trusted_execution(
                background_tasks, approval_id, None, owner, operation["id"]
            )
            return {"status": "success", "decision": "approved", "approval_id": approval_id, "run_id": f"agent:{pending.get('agent_run_id')}" if pending.get("agent_run_id") else None, "execution_scheduled": True, "operation_id": operation["id"], "operation": operation}
        return {"status": "success", "decision": "approved", "approval_id": approval_id, "run_id": f"agent:{pending.get('agent_run_id')}" if pending.get("agent_run_id") else None, "execution_scheduled": False}


@router.post("/approvals/{approval_id}/retry", status_code=202)
async def retry_approval_execution(
    approval_id: int,
    background_tasks: BackgroundTasks,
    _: ApiPrincipal = Depends(require_session_approver),
):
    pending = database.get_pending_action(approval_id)
    if not pending:
        raise HTTPException(status_code=404, detail="approval_not_found")
    if pending.get("status") != "approved" or pending.get("executed_at"):
        raise HTTPException(status_code=409, detail="approval_not_retryable")
    database.mark_pending_execution_failed(approval_id, "manual_retry")
    database.queue_pending_execution(approval_id)
    _dispatch_trusted_execution(background_tasks, approval_id)
    return {"status": "success", "approval_id": approval_id, "execution_scheduled": True}


@router.post("/runs/{run_id:path}/execute", status_code=202)
async def execute_run(
    run_id: str,
    payload: ExecuteRequest,
    background_tasks: BackgroundTasks,
    principal: ApiPrincipal = Depends(require_session_approver),
):
    owner = workspace_for_run(run_id)
    with workspace_scope(owner) if owner else nullcontext():
        detail = get_run(run_id)
        if not detail:
            raise HTTPException(status_code=404, detail="run_not_found")
        kind, source_id = run_id.split(":", 1)
        if kind not in {"scientific", "workflow", "agent"}:
            raise HTTPException(status_code=422, detail="run_not_executable")
        workflow_run_id = None
        if kind == "workflow":
            workflow = workflow_runs.get_workflow_run(int(source_id))
            latest = database.get_pending_action(int(workflow["pending_id"])) if workflow else None
            workflow_run_id = int(source_id)
        else:
            latest = (detail.get("approvals") or [None])[0]
        if latest and latest.get("status") == "stale":
            raise HTTPException(
                status_code=422,
                detail={
                    "reason": latest.get("execution_error") or "proposal_stale"
                },
            )
        if not latest or latest.get("status") != "approved":
            raise HTTPException(status_code=409, detail="approved_proposal_required")
        invalid_reason = validate_pending_for_execution(latest)
        if invalid_reason:
            database.mark_pending_stale(int(latest["id"]), invalid_reason)
            raise HTTPException(status_code=422, detail={"reason": invalid_reason})
        if latest.get("execution_status") == "unknown":
            raise HTTPException(status_code=409, detail="execution_unknown_requires_manual_reconciliation")
        database.queue_pending_execution(int(latest["id"]))
        _dispatch_trusted_execution(
            background_tasks,
            int(latest["id"]),
            workflow_run_id,
            owner,
        )
        return {
            "status": "success",
            "run_id": run_id,
            "execution": {
                "status": "queued",
                "pending_id": int(latest["id"]),
                "idempotency_key": payload.idempotency_key,
                "canonical_execution_pending": True,
            },
        }


@router.post("/runs/{run_id:path}/manual-tasks/{task_id}/resolve")
async def resolve_manual_task(
    run_id: str,
    task_id: str,
    payload: ManualResolutionRequest,
    _: ApiPrincipal = Depends(require_session_approver),
):
    kind, source_id = run_id.split(":", 1) if ":" in run_id else ("", "")
    if kind != "simulation":
        raise HTTPException(status_code=422, detail="manual_task_not_supported")
    try:
        result = simulation_runs.resolve_manual_task(source_id, task_id, resolution=payload.resolution, note=payload.note)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run_not_found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "success", "run": result}


@router.post("/runs/{run_id:path}/manual-commit")
async def manual_commit_workflow(
    run_id: str,
    payload: ManualCommitRequest,
    principal: ApiPrincipal = Depends(require_session_approver),
):
    kind, source_id = run_id.split(":", 1) if ":" in run_id else ("", "")
    if kind != "workflow":
        raise HTTPException(status_code=422, detail="manual_commit_not_supported")
    try:
        completed = datetime.fromisoformat(payload.completed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="invalid_completed_at") from exc
    if completed.tzinfo is not None:
        completed_at = completed.astimezone().replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
    else:
        completed_at = completed.isoformat(sep=" ", timespec="seconds")
    from app.core.db import workflow_runs as workflow_run_db

    operation = operations.create_operation(
        owner=principal.name,
        workspace_id=current_workspace().id,
        kind="manual_subculture_commit",
        label="确认人工传代并更新资源",
        phase="validating",
        message="正在校验品系代数",
        related_run_id=run_id,
        related_entity_type="workflow_run",
        related_entity_id=source_id,
        retryable=False,
    )
    operations.update_operation(operation["id"], status="running", phase="persisting", message="正在更新资源并写入实验审计", progress=0.55)
    result = workflow_run_db.commit_manual_subculture(
        int(source_id),
        completed_at=completed_at,
        confirmed_by=principal.name,
        note=payload.note,
        idempotency_key=payload.idempotency_key,
    )
    if result.get("status") != "success":
        operations.update_operation(operation["id"], status="failed", phase="failed", message="人工传代记录未写入", error_code=str(result.get("reason") or "commit_failed"), error_message=str(result.get("reason") or result), retryable=False)
        status_code = 404 if result.get("reason") == "run_not_found" else 409
        raise HTTPException(status_code=status_code, detail=result)
    from app.services.chat.task_coordinator import task_for_workflow_run

    conversation_task = task_for_workflow_run(run_id)
    if conversation_task:
        conversation_tasks.update_task(
            conversation_task["id"],
            expected_version=int(conversation_task["version"]),
            status="completed",
            event_type="task_manual_confirmation_completed",
        )
    operations.update_operation(operation["id"], status="succeeded", phase="completed", message="资源、实验和审计记录已更新", progress=1.0, retryable=False)
    return {"status": "success", "run_id": run_id, "operation_id": operation["id"], "operation": operations.get_operation(operation["id"]), **result}


@router.post("/test-scenarios/assistant-presentation", status_code=201)
async def create_assistant_presentation_scenario(
    principal: ApiPrincipal = Depends(require_session_scientist),
):
    """Seed deterministic UI content, available only under explicit test auth."""
    if os.getenv("ALGAE_AUTH_MODE", "").strip().casefold() != "test":
        raise HTTPException(status_code=404, detail="scenario_not_found")
    workspace = current_workspace()
    conversation = assistant_conversations.create_conversation(
        principal.name,
        workspace_id=workspace.id,
        title="Assistant presentation UAT",
    )
    task = conversation_tasks.create_task(
        conversation_id=conversation["id"],
        owner=principal.name,
        workspace_id=workspace.id,
        goal_text="为 Chlamydomonas_01 准备邮件草稿并比较两个科学候选",
        task_type="email",
        status="collecting",
        state_changing=True,
        collected_slots={"target": "Chlamydomonas_01"},
        missing_slots=["recipient"],
        proposed_action={
            "available_actions": ["补充收件人", "取消任务"],
            "developer_only": {"checkpoint": "scenario-checkpoint"},
        },
    )
    assistant_conversations.append_message(
        conversation["id"],
        role="assistant",
        content=(
            "# 调查结论\n\n"
            "- 已生成两个候选\n"
            "- 邮件仍是草稿，**尚未发送**\n\n"
            "| 候选 | 风险 |\n| --- | --- |\n| A | 低 |\n| B | 中 |\n\n"
            "```text\nsimulation_only\n```"
        ),
        status="completed",
        task_id=task["id"],
        structured={
            "action": "email_draft",
            "status": "paused",
            "send": False,
            "answer_envelope": {
                "schema_version": "answer-envelope/v2",
                "outcome_status": "paused",
                "direct_answer": "调查结论",
                "confirmed_facts": [],
                "inferences": [],
                "simulation_results": [],
                "recommendations": [],
                "unknowns": [],
                "source_statuses": [],
                "completed_work": {"candidate_count": 2},
                "remaining_work": ["补充收件人"],
                "budget": {},
                "references": [],
                "next_actions": [],
                "presentation_blocks": [
                    {
                        "type": "email_draft",
                        "target": "Chlamydomonas_01",
                        "send": False,
                        "draft": {
                            "subject": "检查提醒",
                            "body": "请检查 Chlamydomonas_01。",
                            "recipients": ["test-lab@example.invalid"],
                        },
                    },
                    {
                        "type": "approval",
                        "pending_id": 900001,
                        "status": "pending",
                        "executable": False,
                        "requires_human_approval": True,
                    },
                    {
                        "type": "scientific_result",
                        "candidate_count": 2,
                        "candidates": [
                            {"candidate_id": "candidate-a"},
                            {"candidate_id": "candidate-b"},
                        ],
                        "validations": [{"candidate_id": "candidate-a"}],
                        "simulations": [{"candidate_id": "candidate-a"}],
                        "plan_patches": [{"patch_id": "patch-1"}],
                        "comparison": {"dimensions": ["benefit", "risk", "cost", "uncertainty"]},
                    },
                    {
                        "type": "paused_task",
                        "task_id": task["id"],
                        "reason": "awaiting_recipient",
                        "completed_work": {"candidate_count": 2},
                        "remaining_work": ["补充收件人"],
                        "budget": {},
                    },
                ],
            },
        },
    )
    return {
        "status": "success",
        "conversation_id": conversation["id"],
        "task_id": task["id"],
    }


@router.get("/test-scenarios")
async def test_scenarios(_: ApiPrincipal = Depends(require_session_viewer)):
    return {"status": "success", "workspace": {"id": "test-default", "name": "隔离测试工作区", "mode": "simulation_only"}, "scenarios": TEST_SCENARIOS}


@router.post("/test-scenarios/{scenario_id}/runs", status_code=202)
async def start_test_scenario(
    scenario_id: str,
    payload: ScenarioRunRequest,
    principal: ApiPrincipal = Depends(require_session_scientist),
):
    scenario = next((item for item in TEST_SCENARIOS if item["id"] == scenario_id), None)
    if not scenario:
        raise HTTPException(status_code=404, detail="scenario_not_found")
    source_dataset = None
    if scenario["kind"] == "scientific":
        selected_id = payload.parameters.get("dataset_id")
        use_demo = payload.parameters.get("use_demo_dataset") is True
        if bool(selected_id) == bool(use_demo):
            raise HTTPException(status_code=422, detail="choose_exactly_one_dataset_source")
        source_dataset = scientific_db.get_dataset(str(selected_id)) if selected_id else None
        if selected_id and not source_dataset:
            raise HTTPException(status_code=404, detail="scientific_dataset_not_found")
        if source_dataset and source_dataset.get("status") != "ready":
            raise HTTPException(status_code=409, detail="scientific_dataset_not_ready")
    operation = operations.create_operation(
        owner=principal.name,
        workspace_id=current_workspace().id,
        kind="test_lab_run",
        label="启动 Test Lab 场景",
        phase="workspace",
        message="正在创建隔离工作区",
        related_entity_type="test_scenario",
        related_entity_id=scenario_id,
        metadata={"scenario_id": scenario_id},
        retryable=False,
    )
    operations.update_operation(operation["id"], status="running", phase="workspace", message="正在创建隔离工作区", progress=0.18)
    workspace = create_test_workspace(scenario_id, seed=2025)
    operations.update_operation(operation["id"], phase="preparing", message="正在准备隔离数据与 Run", progress=0.45)
    with workspace_scope(workspace):
        db_schema.init_db()
        if scenario["kind"] == "scientific":
            if source_dataset:
                dataset_copy = dict(source_dataset)
                batches = dataset_copy.pop("batches", [])
                scientific_db.insert_dataset(dataset_copy, batches)
                dataset_id = dataset_copy["id"]
            else:
                content, filename, mapping = scientific_demo_file()
                parsed = parse_scientific_dataset(
                    content=content, filename=filename, strain_id="Chlorella_01", mapping=mapping,
                )
                scientific_db.insert_dataset(parsed.dataset, parsed.batches)
                dataset_id = parsed.dataset["id"]
            mode = "diagnose" if scenario_id == "scientific-diagnose" else "diagnose_and_optimize"
            result = run_scientific_task(
                dataset_id=str(dataset_id), mode=mode, max_cycles=2, run_seed=workspace.seed,
                session_id=f"{workspace.id}:{principal.name}", offline_replay=True,
                create_pending=mode == "diagnose_and_optimize",
            )
            canonical_id = f"scientific:{result['id']}"
        else:
            interaction_mode = "verification" if scenario_id == "subculture-manual" else "demo"
            run = simulation_runs.create(
                strain_id=str(payload.parameters.get("strain_id") or "Chlorella_01"),
                generation_number=int(payload.parameters.get("generation_number") or 1),
                interaction_mode=interaction_mode,
                fault_scenario=scenario.get("fault"),
            )
            canonical_id = f"simulation:{run['run_id']}"
    register_run_workspace(canonical_id, workspace)
    operations.update_operation(
        operation["id"],
        status="succeeded",
        phase="started",
        message="Test Lab Run 已创建",
        progress=1.0,
        related_run_id=canonical_id,
        retryable=False,
    )
    return {
        "status": "success", "run_id": canonical_id,
        "workspace": {"id": workspace.id, "name": workspace.name, "mode": "simulation_only", "seed": workspace.seed},
        "operation_id": operation["id"],
        "operation": operations.get_operation(operation["id"]),
    }


@router.get("/test-workspaces")
async def test_workspaces(_: ApiPrincipal = Depends(require_session_viewer)):
    return {
        "status": "success",
        "workspaces": [
            {"id": item.id, "name": item.name, "mode": "simulation_only", "seed": item.seed, "time_origin": item.time_origin}
            for item in list_test_workspaces()
        ],
    }


@router.post("/test-workspaces/{workspace_id}/reset")
async def reset_workspace(workspace_id: str, _: ApiPrincipal = Depends(require_session_approver)):
    try:
        workspace = reset_test_workspace(workspace_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="workspace_not_found") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail="workspace_busy") from exc
    with workspace_scope(workspace):
        db_schema.init_db()
    return {"status": "success", "workspace_id": workspace.id}


@router.delete("/test-workspaces/{workspace_id}")
async def delete_workspace(workspace_id: str, _: ApiPrincipal = Depends(require_session_approver)):
    try:
        delete_test_workspace(workspace_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="workspace_not_found") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail="workspace_busy") from exc
    return {"status": "success", "workspace_id": workspace_id}


@router.get("/resources/{resource_type}")
async def resources(resource_type: str, _: ApiPrincipal = Depends(require_session_viewer)):
    if resource_type == "strains":
        return {"status": "success", "items": database.list_algae_status()}
    if resource_type == "datasets":
        return {"status": "success", "items": scientific_db.list_datasets()}
    if resource_type == "devices":
        return {"status": "success", "items": scientific_db.list_lab_devices()}
    raise HTTPException(status_code=404, detail="resource_type_not_found")


@router.post("/resources/strains/requests", status_code=201)
async def create_strain_resource_request(
    payload: StrainResourceRequest,
    principal: ApiPrincipal = Depends(require_session_scientist),
):
    data = dict(payload.data or {})
    strain_id = str(data.get("strain_id") or "").strip()
    if not strain_id:
        raise HTTPException(status_code=422, detail="strain_id_required")
    existing = database.get_algae_status(strain_id)
    if payload.operation == "add":
        if existing:
            raise HTTPException(status_code=409, detail="strain_already_exists")
        if not str(data.get("name_cn") or "").strip() or not str(data.get("name_en") or "").strip():
            raise HTTPException(status_code=422, detail="strain_names_required")
        if int(data.get("generation_number", 1)) < 0 or int(data.get("days_since_last_subculture", 0)) < 0:
            raise HTTPException(status_code=422, detail="invalid_strain_counters")
        pending_id = strain_service.create_pending_add_strain(data, requester="human_api")
    elif payload.operation == "update":
        if not existing:
            raise HTTPException(status_code=404, detail="strain_not_found")
        if "generation_number" in data and int(data["generation_number"]) < 0:
            raise HTTPException(status_code=422, detail="invalid_generation_number")
        if "days_since_last_subculture" in data and int(data["days_since_last_subculture"]) < 0:
            raise HTTPException(status_code=422, detail="invalid_days_since_last_subculture")
        pending_id = strain_service.create_pending_update_strain(data, requester="human_api")
    else:
        if not existing:
            raise HTTPException(status_code=404, detail="strain_not_found")
        pending_id = strain_service.create_pending_delete_strain(strain_id, requester="human_api")
    return {
        "status": "success",
        "pending_id": pending_id,
        "approval": next((item for item in list_approvals(status="pending") if item["id"] == pending_id), None),
        "requested_by": principal.name,
    }


@router.post("/resources/datasets/import", status_code=202)
async def import_resource_dataset(
    file: UploadFile = File(...),
    dataset_name: str | None = Form(default=None),
    strain_id: str | None = Form(default=None),
    mapping_json: str | None = Form(default=None),
    sheet_name: str | None = Form(default=None),
    principal: ApiPrincipal = Depends(require_session_scientist),
):
    workspace = current_workspace()
    operation = operations.create_operation(
        owner=principal.name,
        workspace_id=workspace.id,
        kind="dataset_import",
        label="导入科学数据集",
        phase="uploading",
        message="文件已接收，正在解析数据集",
        related_entity_type="dataset",
        metadata={"filename": file.filename or "dataset"},
        retryable=False,
    )
    try:
        operations.update_operation(operation["id"], status="running", phase="parsing", message="正在解析数据与识别列", progress=0.35)
        mapping = json.loads(mapping_json) if mapping_json else None
        parsed = parse_scientific_dataset(
            content=await file.read(),
            filename=file.filename or "dataset",
            dataset_name=dataset_name,
            strain_id=strain_id,
            mapping=mapping,
            sheet_name=sheet_name,
        )
        existing = scientific_db.get_dataset(parsed.dataset["id"], include_measurements=False)
        if existing:
            operations.update_operation(operation["id"], status="succeeded", phase="completed", message="数据集已存在，已完成去重", progress=1.0, related_run_id=None)
            return {"status": "success", "idempotent": True, "dataset": existing, "operation_id": operation["id"], "operation": operations.get_operation(operation["id"])}
        operations.update_operation(operation["id"], phase="saving", message="列识别完成，正在保存数据集", progress=0.78)
        scientific_db.insert_dataset(parsed.dataset, parsed.batches)
        operations.update_operation(operation["id"], status="succeeded", phase="completed", message="数据集导入完成", progress=1.0, retryable=False)
        return {
            "status": "success",
            "idempotent": False,
            "dataset": scientific_db.get_dataset(parsed.dataset["id"], include_measurements=False),
            "imported_by": principal.name,
            "operation_id": operation["id"],
            "operation": operations.get_operation(operation["id"]),
        }
    except DatasetImportError as exc:
        waiting_mapping = exc.code == "mapping_required"
        operations.update_operation(
            operation["id"],
            status="waiting_input" if waiting_mapping else "failed",
            phase="column_mapping" if waiting_mapping else "failed",
            message="需要确认列映射" if waiting_mapping else "数据集解析失败",
            error_code=exc.code,
            error_message=str(exc),
            retryable=False,
        )
        raise HTTPException(
            status_code=422,
            detail={"code": exc.code, "message": str(exc), "operation_id": operation["id"], **exc.details},
        ) from exc
    except json.JSONDecodeError as exc:
        operations.update_operation(operation["id"], status="failed", phase="failed", message="列映射格式无效", error_code="invalid_mapping_json", error_message=str(exc), retryable=False)
        raise HTTPException(status_code=422, detail={"code": "invalid_mapping_json"}) from exc


@router.post("/knowledge/query")
async def knowledge_query(payload: RagQueryRequest, _: ApiPrincipal = Depends(require_session_viewer)):
    try:
        return {"status": "success", "answer": answer_rag_question(payload).model_dump()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {exc}") from exc


@router.post("/knowledge/index-generations", status_code=202)
async def create_knowledge_index_generation(
    payload: RagGenerationCreateRequest,
    background_tasks: BackgroundTasks,
    principal: ApiPrincipal = Depends(require_session_scientist),
):
    generation_id = payload.generation_id or f"rag-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    workspace = current_workspace()
    operation = operations.create_operation(
        owner=principal.name, workspace_id=workspace.id, kind="knowledge_generation_build",
        label="构建 RAG 影子索引", phase="queued", message="影子索引已排队",
        related_entity_type="rag_index_generation", related_entity_id=generation_id,
        metadata={"generation_id": generation_id, "source_roots": payload.source_roots}, retryable=True,
    )
    background_tasks.add_task(
        _build_rag_generation_operation, operation_id=operation["id"], generation_id=generation_id,
        source_roots=payload.source_roots, workspace=workspace,
    )
    return {"status": "accepted", "generation_id": generation_id, "operation_id": operation["id"], "operation": operation}


@router.get("/knowledge/index-generations/{generation_id}")
async def get_knowledge_index_generation(
    generation_id: str, _: ApiPrincipal = Depends(require_session_scientist)
):
    generation = database.get_rag_index_generation(generation_id)
    if not generation:
        raise HTTPException(status_code=404, detail="rag_generation_not_found")
    return {"status": "success", "generation": generation}


@router.post("/knowledge/index-generations/{generation_id}/activate")
async def activate_knowledge_index_generation(
    generation_id: str, _: ApiPrincipal = Depends(require_session_scientist)
):
    result = activate_generation(generation_id)
    if not result["activated"]:
        raise HTTPException(status_code=409, detail=result)
    return {"status": "success", **result}


@router.get("/knowledge/sources")
async def knowledge_sources(
    include_archived: bool = True,
    _: ApiPrincipal = Depends(require_session_viewer),
):
    return {"status": "success", **list_sources(include_archived=include_archived)}


@router.post("/knowledge/sources", status_code=202)
async def upload_knowledge_source(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    doc_type: str = Form(...),
    title: str | None = Form(default=None),
    version: str | None = Form(default=None),
    language: str | None = Form(default=None),
    asset_key: str | None = Form(default=None),
    effective_from: str | None = Form(default=None),
    effective_to: str | None = Form(default=None),
    supersedes_source_id: str | None = Form(default=None),
    principal: ApiPrincipal = Depends(require_session_scientist),
):
    try:
        source = save_uploaded_source(
            content=await file.read(),
            filename=file.filename or "document",
            doc_type=doc_type,
            owner=principal.name,
            title=title,
            version=version,
            language=language,
            asset_key=asset_key,
            effective_from=effective_from,
            effective_to=effective_to,
            supersedes_source_id=supersedes_source_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    workspace = current_workspace()
    operation = operations.create_operation(
        owner=principal.name,
        workspace_id=workspace.id,
        kind="knowledge_index",
        label="索引知识文件",
        phase="uploaded",
        message="文件已上传，正在准备索引",
        related_entity_type="knowledge_source",
        related_entity_id=source["source_id"],
        metadata={"source_id": source["source_id"], "rebuild": False},
        retryable=True,
    )
    background_tasks.add_task(
        _index_knowledge_operation,
        operation_id=operation["id"],
        source_id=source["source_id"],
        workspace=workspace,
    )
    return {"status": "accepted", "source": source, "indexing_scheduled": True, "operation_id": operation["id"], "operation": operation}


@router.post("/knowledge/sources/{source_id}/reindex", status_code=202)
async def reindex_knowledge_source(
    source_id: str,
    background_tasks: BackgroundTasks,
    principal: ApiPrincipal = Depends(require_session_scientist),
):
    source = database.get_rag_knowledge_source(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="knowledge_source_not_found")
    workspace = current_workspace()
    operation = operations.create_operation(
        owner=principal.name,
        workspace_id=workspace.id,
        kind="knowledge_reindex",
        label="重建知识索引",
        phase="queued",
        message="正在准备重建索引",
        related_entity_type="knowledge_source",
        related_entity_id=source_id,
        metadata={"source_id": source_id, "rebuild": True},
        retryable=True,
    )
    background_tasks.add_task(
        _index_knowledge_operation,
        operation_id=operation["id"],
        source_id=source_id,
        workspace=workspace,
        rebuild=True,
    )
    return {"status": "accepted", "source_id": source_id, "indexing_scheduled": True, "operation_id": operation["id"], "operation": operation}


@router.post("/knowledge/sources/{source_id}/archive")
async def archive_knowledge_source(
    source_id: str,
    principal: ApiPrincipal = Depends(require_session_scientist),
):
    if not database.set_rag_knowledge_source_lifecycle(source_id, "archived", actor=principal.name):
        raise HTTPException(status_code=404, detail="knowledge_source_not_found")
    return {"status": "success", "source_id": source_id, "lifecycle_status": "archived"}


@router.post("/knowledge/sources/{source_id}/restore")
async def restore_knowledge_source(
    source_id: str,
    _: ApiPrincipal = Depends(require_session_scientist),
):
    if not database.set_rag_knowledge_source_lifecycle(source_id, "active"):
        raise HTTPException(status_code=404, detail="knowledge_source_not_found")
    return {"status": "success", "source_id": source_id, "lifecycle_status": "active"}
