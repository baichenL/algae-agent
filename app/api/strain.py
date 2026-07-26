# app/api/strain.py 负责品系相关的 API 接口
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import JSONResponse

from app.schemas.algae import AddStrainRequest, ConfirmPendingRequest
from app.core import database
from app.core.db import workflow_runs
from app.services.strains import strain_service
from app.services.agent_runtime.approval_resume import review_pending_and_resume
from app.services.workflows.workflow_approval_service import grant_workflow_approval
from app.tools.executor import ToolExecutionContext, execute_registered_tool
from app.core.security import ApiPrincipal, require_approver, require_scientist, require_viewer
from app.services.trusted_execution_dispatch import dispatch_trusted_execution

router = APIRouter()


@router.post('/strain/add')
async def add_strain_endpoint(payload: AddStrainRequest, _: ApiPrincipal = Depends(require_scientist)):
    """Create a pending add-strain request which requires human confirmation."""
    try:
        pending_id = strain_service.create_pending_add_strain(payload.dict(), requester="human_api")
        return {"status": "pending", "pending_id": pending_id, "msg": "已创建待确认请求，请调用 /api/v1/strain/confirm 批准"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加品系失败: {str(e)}")


@router.get('/strain/list')
async def list_strains(_: ApiPrincipal = Depends(require_viewer)):
    """Return all strains in the database."""
    try:
        from app.core.database import list_algae_status
        from app.services.notifications.notification_service import attach_reminder_status_to_strains
        items = list_algae_status()
        items = attach_reminder_status_to_strains(items)
        return {"status": "success", "strains": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询品系失败: {str(e)}")


@router.get('/strain/last_operation')
async def last_operation(_: ApiPrincipal = Depends(require_viewer)):
    """Return the last database operation (add/update/delete) performed on algae_status."""
    try:
        from app.core.database import get_last_db_operation
        last = get_last_db_operation()
        return {"status": "success", "last_operation": last}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询最后操作失败: {str(e)}")


@router.get('/strain/pending')
async def list_pending(status: str = "pending", _: ApiPrincipal = Depends(require_viewer)):
    items = strain_service.list_pending_actions(status=status)
    # 格式化为更友好的摘要，便于人工审批 UI 展示
    formatted = []
    for it in items:
        payload = it.get("payload", {}) or {}
        summary = ""
        if payload.get("type") == "add_strain":
            data = payload.get("data", {})
            summary = (
                f"添加品系: {data.get('strain_id')} / {data.get('name_cn')} ({data.get('name_en')}) | "
                f"代数={data.get('generation_number')}, 距传代={data.get('days_since_last_subculture')}天"
            )
        elif payload.get("type") == "update_strain":
            data = payload.get("data", {})
            fields = [key for key in data.keys() if key != "strain_id"]
            summary = f"更新品系: {data.get('strain_id')} | 字段={fields}"
        elif payload.get("type") == "delete_strain":
            data = payload.get("data", {})
            summary = f"删除品系: {data.get('strain_id')}"
        elif payload.get("type") == "workflow_subculture":
            data = payload.get("data", {})
            summary = f"传代 workflow 审批: {data.get('strain_id')} | workflow={data.get('workflow', 'subculture')}"
        elif payload.get("type") == "scientific_experiment_plan":
            data = payload.get("data", {})
            design = data.get("design") or {}
            summary = (
                f"数字孪生实验方案: run={data.get('scientific_run_id')} | "
                f"conditions={len(design.get('conditions') or [])} | simulation_only=true | "
                f"hash={str(data.get('design_hash') or '')[:12]}"
            )
        else:
            summary = f"操作: {it.get('action_type')}"

        # 提供更清晰的目标对象明细（object），便于审批者直接看到将被操作的实体字段
        object_details = payload.get("data") if isinstance(payload.get("data"), dict) else {}

        formatted.append({
            "id": it.get("id"),
            "action_type": it.get("action_type"),
            "requester": it.get("requester"),
            "created_at": it.get("created_at"),
            "status": it.get("status", "pending"),
            "reviewed_at": it.get("reviewed_at"),
            "reviewed_by": it.get("reviewed_by"),
            "review_reason": it.get("review_reason"),
            "risk_level": it.get("risk_level"),
            "source": it.get("source"),
            "summary": summary,
            "object": object_details,
            "payload": payload
        })

    return {"status": "success", "pending": formatted}


@router.post('/strain/confirm')
async def confirm_pending(
    payload: ConfirmPendingRequest,
    background_tasks: BackgroundTasks,
    principal: ApiPrincipal = Depends(require_approver),
):
    pending = strain_service.get_pending_action(payload.pending_id)
    if not pending:
        raise HTTPException(status_code=404, detail="pending_not_found")
    governed_type = (pending.get("payload") or {}).get("type")
    if not payload.approve:
        governed_result = await review_pending_and_resume(
            payload.pending_id,
            approve=False,
            reviewed_by=principal.name,
        )
        if governed_result.get("status") != "success":
            raise HTTPException(
                status_code=422,
                detail=governed_result.get("reason", "denial_failed"),
            )
        if governed_type == "scientific_experiment_plan":
            from app.core.db import scientific as scientific_db

            governed_run_id = (
                ((pending.get("payload") or {}).get("data") or {}).get("scientific_run_id")
            )
            if governed_run_id:
                scientific_db.update_scientific_run(str(governed_run_id), status="denied")
        return governed_result

    governed_workflow_run_id = None
    if governed_type == "workflow_subculture":
        governed_result = grant_workflow_approval(
            payload.pending_id,
            reviewed_by=principal.name,
        )
        governed_workflow_run_id = governed_result.get("run_id")
    else:
        governed_result = await review_pending_and_resume(
            payload.pending_id,
            approve=True,
            reviewed_by=principal.name,
        )
    if governed_result.get("status") != "success":
        raise HTTPException(
            status_code=422,
            detail=governed_result.get("reason", "approval_failed"),
        )
    database.queue_pending_execution(payload.pending_id)
    dispatch_trusted_execution(
        background_tasks,
        payload.pending_id,
        (
            int(governed_workflow_run_id)
            if governed_workflow_run_id is not None
            else None
        ),
    )
    return JSONResponse(
        status_code=202,
        content={
            **governed_result,
            "pending_id": payload.pending_id,
            "decision": "approved",
            "execution_scheduled": True,
        },
    )

@router.get('/workflow-runs/{run_id}')
async def get_workflow_run(run_id: int, _: ApiPrincipal = Depends(require_viewer)):
    run = workflow_runs.get_workflow_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="workflow_run_not_found")
    return {"status": "success", "workflow_run": run}


@router.post('/strain/submit_tool')
async def submit_tool(payload: dict, _: ApiPrincipal = Depends(require_scientist)):
    """Submit a frontend form tool request through the shared policy gateway."""
    try:
        tool = payload.get("tool")
        args = payload.get("args", {})
        if not tool:
            raise HTTPException(status_code=400, detail="tool is required")
        result = await execute_registered_tool(
            tool,
            args,
            context=ToolExecutionContext(
                caller="frontend_form",
                session_id=args.get("session_id"),
                agent_run_id=args.get("agent_run_id"),
                source="strain_submit_tool_api",
            ),
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"工具提交失败: {str(e)}")
