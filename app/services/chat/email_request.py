from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from app.services.intent.input_normalizer import normalize_text_value


@dataclass(frozen=True)
class EmailRequestSpec:
    action: Literal["draft"] = "draft"
    target: str | None = None
    recipient: str | None = None
    create_approval: bool = False
    send: bool = False
    missing_fields: tuple[str, ...] = ()
    source_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_TARGET = re.compile(r"\b([A-Za-z][A-Za-z0-9_-]*[-_][0-9]+)\b")
_EMAIL = re.compile(r"(?:邮件|邮箱|写封|写一封|email|mail)", re.I)
_APPROVAL = re.compile(r"(?:待审批|待确认|审批请求|先给我审阅|先审阅|提交审批|approval|review)", re.I)
_SEND = re.compile(r"(?:立即发送|直接发送|现在发送|马上发送|send\s+(?:it\s+)?now)", re.I)
_NO_SEND = re.compile(r"(?:不要|不得|不能|仅|只).{0,12}(?:发送|已发送|send|sent)", re.I)


def is_email_request(message: str) -> bool:
    return bool(_EMAIL.search(normalize_text_value(message)))


def parse_email_request(message: str, *, fallback_target: str | None = None) -> EmailRequestSpec:
    text = normalize_text_value(message)
    target_match = _TARGET.search(text)
    target = target_match.group(1) if target_match else fallback_target
    recipient = None
    if "实验员" in text:
        recipient = "实验员"
    elif "lab technician" in text.casefold() or "operator" in text.casefold():
        recipient = "lab technician"
    create_approval = bool(_APPROVAL.search(text))
    send = bool(_SEND.search(text)) and not bool(_NO_SEND.search(text))
    missing = ("target",) if not target else ()
    return EmailRequestSpec(
        target=target,
        recipient=recipient,
        create_approval=create_approval,
        send=send,
        missing_fields=missing,
        source_message=message,
    )


def resume_email_request(task: dict[str, Any], message: str) -> EmailRequestSpec:
    previous = dict((task.get("proposed_action") or {}).get("email_request_spec") or {})
    combined = "\n".join(
        item
        for item in (
            str(previous.get("source_message") or task.get("goal_text") or ""),
            message,
        )
        if item
    )
    return parse_email_request(combined, fallback_target=previous.get("target"))
