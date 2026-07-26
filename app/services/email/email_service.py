import datetime
import os
import smtplib
import ssl
import json
from pathlib import Path
from dataclasses import dataclass
from email.message import EmailMessage
from typing import List, Optional

from app.models.email_schema import EmailLogRecord, EmailSendRequest, EmailSendResponse
from app.services.email.email_log_service import append_email_log
from app.services.email.recipient_policy import get_default_recipients
from app.services.observability.error_events import record_error_event
from app.core.workspaces import current_workspace


@dataclass(frozen=True)
class EmailSettings:
    host: str
    port: int
    username: str
    password: str
    email_from: str
    use_tls: bool
    allowed_recipients: List[str]
    timeout_seconds: int = 15


def build_email_settings_from_env() -> Optional[EmailSettings]:
    host = _env("EMAIL_HOST", "SMTP_HOST")
    username = _env("EMAIL_USER", "SMTP_USERNAME")
    password = _env("EMAIL_PASSWORD", "SMTP_PASSWORD")
    email_from = _env("EMAIL_FROM")
    allowed = _env("EMAIL_ALLOWED_RECIPIENTS", "EMAIL_TO")
    if not all([host, username, password, email_from, allowed]):
        return None
    return EmailSettings(
        host=host,
        port=_int_env("EMAIL_PORT", _int_env("SMTP_PORT", 587)),
        username=username,
        password=password,
        email_from=email_from,
        use_tls=_bool_env("EMAIL_USE_TLS", _bool_env("SMTP_USE_TLS", True)),
        allowed_recipients=[item.strip() for item in allowed.split(",") if item.strip()],
        timeout_seconds=_int_env("EMAIL_TIMEOUT_SECONDS", _int_env("SMTP_TIMEOUT_SECONDS", 15)),
    )


def validate_recipients(recipients: List[str], settings: EmailSettings) -> None:
    allowed = {item.lower() for item in settings.allowed_recipients}
    denied = [item for item in recipients if item.lower() not in allowed]
    if denied:
        raise ValueError(f"收件人不在白名单中: {denied}")


def send_email(request: EmailSendRequest) -> EmailSendResponse:
    template_type = str(request.metadata.get("template_type", "unknown"))
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    workspace = current_workspace()
    if workspace.email_mode == "test_outbox":
        outbox_path = Path(workspace.db_path).with_name("test_outbox.jsonl")
        outbox_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "created_at": created_at,
            "workspace_id": workspace.id,
            "recipients": request.recipients,
            "subject": request.subject,
            "body": request.body,
            "source": request.source,
            "metadata": request.metadata,
        }
        with outbox_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return EmailSendResponse(
            status="success",
            sent=False,
            message="Queued in simulation-only test outbox; no SMTP adapter was called.",
            recipients=request.recipients,
            log_id=f"outbox:{workspace.id}:{created_at}",
        )
    settings = build_email_settings_from_env()
    if not settings:
        record_error_event(
            layer="email_notification",
            component="email_service",
            operation="send_email",
            severity="warning",
            error_type="EmailSettingsMissing",
            error_message="SMTP settings are not configured.",
            metadata={"source": request.source, "recipients": request.recipients},
        )
        return _log_and_response(request, created_at, template_type, False, "发送失败：SMTP 未配置")

    try:
        validate_recipients(request.recipients, settings)
        message = EmailMessage()
        message["Subject"] = request.subject
        message["From"] = settings.email_from
        message["To"] = ", ".join(request.recipients)
        if request.metadata.get("message_id"):
            message["Message-ID"] = str(request.metadata["message_id"])
        message.set_content(request.body)

        with smtplib.SMTP(settings.host, settings.port, timeout=settings.timeout_seconds) as server:
            if settings.use_tls:
                server.starttls(context=ssl.create_default_context())
            server.login(settings.username, settings.password)
            server.send_message(message)
        return _log_and_response(request, created_at, template_type, True, "已发送")
    except Exception as exc:
        record_error_event(
            layer="email_notification",
            component="email_service",
            operation="send_email",
            severity="error",
            error_type=type(exc).__name__,
            error_message=str(exc),
            metadata={"source": request.source, "recipients": request.recipients},
        )
        return _log_and_response(request, created_at, template_type, False, f"发送失败：{str(exc)}")


def test_email_connection(recipients: List[str]) -> EmailSendResponse:
    target_recipients = recipients or get_default_recipients()
    request = EmailSendRequest(
        subject="[Algae Agent] SMTP 测试邮件",
        body="这是一封 Algae Agent SMTP 配置测试邮件。",
        recipients=target_recipients,
        source="email_test",
        metadata={"template_type": "smtp_test"},
    )
    return send_email(request)


def _log_and_response(
    request: EmailSendRequest,
    created_at: str,
    template_type: str,
    sent: bool,
    message: str,
) -> EmailSendResponse:
    status = "success" if sent else "error"
    try:
        log_id = append_email_log(
            EmailLogRecord(
                created_at=created_at,
                status=status,
                source=request.source,
                recipients=request.recipients,
                subject=request.subject,
                template_type=template_type,
                error=None if sent else message,
                metadata={k: v for k, v in request.metadata.items() if "password" not in k.lower()},
            )
        )
    except Exception as exc:
        record_error_event(
            layer="email_notification",
            component="email_service",
            operation="append_email_log",
            severity="warning",
            error_type=type(exc).__name__,
            error_message=str(exc),
            metadata={"source": request.source, "status": status},
        )
        log_id = created_at
    return EmailSendResponse(
        status=status,
        sent=sent,
        message=message,
        recipients=request.recipients,
        log_id=log_id,
    )


def _env(name: str, fallback: str | None = None) -> str:
    value = os.getenv(name, "").strip()
    if value or not fallback:
        return value
    return os.getenv(fallback, "").strip()


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
