from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EmailDraft(BaseModel):
    subject: str
    body: str
    recipients: List[str] = Field(default_factory=list)
    template_type: str = "manual_check"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    requires_confirmation: bool = True


class EmailSendRequest(BaseModel):
    subject: str
    body: str
    recipients: List[str]
    source: str = "api"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmailSendResponse(BaseModel):
    status: str
    sent: bool
    message: str
    recipients: List[str] = Field(default_factory=list)
    log_id: Optional[str] = None


class EmailLogRecord(BaseModel):
    created_at: str
    status: str
    source: str
    recipients: List[str] = Field(default_factory=list)
    subject: str
    template_type: str = "unknown"
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmailDraftRequest(BaseModel):
    template_type: str = "manual_check"
    message: str = ""
    recipients: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmailTestRequest(BaseModel):
    recipients: List[str] = Field(default_factory=list)
