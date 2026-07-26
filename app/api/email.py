# app/api/email.py 负责邮件相关的 API 接口
from fastapi import APIRouter, Depends, HTTPException

from app.core.security import ApiPrincipal, require_approver, require_scientist, require_viewer
from app.models.email_schema import EmailDraftRequest, EmailSendRequest, EmailTestRequest
from app.services.context.context_builder import build_context_snapshot
from app.services.email.email_log_service import read_recent_email_logs
from app.tools.email_tool import create_email_draft_from_user_message

router = APIRouter()


@router.post('/email/draft')
async def create_email_draft(payload: EmailDraftRequest, _: ApiPrincipal = Depends(require_scientist)):
    try:
        snapshot = build_context_snapshot("email_api")
        draft = create_email_draft_from_user_message(payload.message, snapshot)
        if payload.recipients:
            draft.recipients = payload.recipients
        if payload.metadata:
            draft.metadata.update(payload.metadata)
        if payload.template_type:
            draft.template_type = payload.template_type
        return {"status": "success", "draft": draft.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"邮件草稿生成失败: {str(e)}")


@router.post('/email/send')
async def send_email_endpoint(payload: EmailSendRequest, _: ApiPrincipal = Depends(require_approver)):
    raise HTTPException(
        status_code=409,
        detail=(
            "Direct email sending is disabled. Create an immutable email proposal, "
            "approve its pending id, and let the trusted worker send it."
        ),
    )


@router.get('/email/logs')
async def email_logs(limit: int = 50, _: ApiPrincipal = Depends(require_viewer)):
    return {"status": "success", "logs": read_recent_email_logs(limit)}


@router.post('/email/test')
async def email_test(payload: EmailTestRequest, _: ApiPrincipal = Depends(require_approver)):
    raise HTTPException(
        status_code=409,
        detail="Direct SMTP tests are disabled in the API process; use the trusted worker diagnostic.",
    )
