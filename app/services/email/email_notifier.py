from dataclasses import dataclass
from typing import Optional

from app.models.email_schema import EmailSendRequest
from app.services.email.email_service import build_email_settings_from_env, send_email as send_email_via_service


@dataclass(frozen=True)
class EmailSettings:
    smtp_host: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    email_from: str
    email_to: str
    use_ssl: bool
    use_tls: bool
    timeout_seconds: int

    @classmethod
    def from_env(cls) -> Optional["EmailSettings"]:
        settings = build_email_settings_from_env()
        if not settings:
            return None
        return cls(
            smtp_host=settings.host,
            smtp_port=settings.port,
            smtp_username=settings.username,
            smtp_password=settings.password,
            email_from=settings.email_from,
            email_to=",".join(settings.allowed_recipients),
            use_ssl=False,
            use_tls=settings.use_tls,
            timeout_seconds=settings.timeout_seconds,
        )


def send_email(subject: str, body: str, settings: Optional[EmailSettings] = None) -> bool:
    """Compatibility wrapper. Real SMTP sending is centralized in email_service.py."""
    recipients = []
    if settings and settings.email_to:
        recipients = [item.strip() for item in settings.email_to.split(",") if item.strip()]
    else:
        resolved = EmailSettings.from_env()
        if resolved and resolved.email_to:
            recipients = [item.strip() for item in resolved.email_to.split(",") if item.strip()]

    result = send_email_via_service(EmailSendRequest(
        subject=subject,
        body=body,
        recipients=recipients,
        source="email_notifier",
        metadata={"template_type": "scheduled_reminder"},
    ))
    return result.sent
