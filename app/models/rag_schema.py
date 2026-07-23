from datetime import datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class RagQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    doc_types: List[str] = Field(default_factory=list)
    retrieval_mode: Literal["auto", "simple", "agentic"] = "auto"
    source_ids: List[str] = Field(default_factory=list)
    version_policy: Literal["current", "all", "as_of"] = "current"
    as_of: datetime | None = None

    @model_validator(mode="after")
    def validate_version_policy(self):
        if self.version_policy == "as_of" and self.as_of is None:
            raise ValueError("as_of is required when version_policy='as_of'")
        return self


class RagCitation(BaseModel):
    source_id: int
    chunk_id: str
    file_name: str
    source_path: str
    doc_type: str
    title: Optional[str] = None
    section: Optional[str] = None
    page_number: Optional[int] = None
    sheet_name: Optional[str] = None
    row_start: Optional[int] = None
    row_end: Optional[int] = None
    topic: Optional[str] = None
    version: Optional[str] = None
    year: Optional[str] = None
    language: Optional[str] = None
    evidence_type: Optional[str] = None
    confidence: Optional[float] = None
    extraction_method: Optional[str] = None
    organism: Optional[str] = None
    medium: Optional[str] = None
    task_type: Optional[str] = None
    equipment: Optional[str] = None
    measurement: Optional[str] = None
    knowledge_source_id: Optional[str] = None
    generation_id: Optional[str] = None
    evidence_id: Optional[str] = None
    document_version: Optional[str] = None
    content_hash: Optional[str] = None
    source_locator: dict[str, Any] = Field(default_factory=dict)


class RagGenerationCreateRequest(BaseModel):
    generation_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._:-]{1,100}$")
    source_roots: List[str] = Field(default_factory=list)


class RagEvidence(BaseModel):
    source_id: int
    file_name: str
    doc_type: str
    location: str
    quote_summary: str


class RagAnswerSegment(BaseModel):
    text: str
    citation_ids: List[int] = Field(default_factory=list)


class RagAnswer(BaseModel):
    conclusion: str
    evidence: List[RagEvidence] = Field(default_factory=list)
    facts: List[RagAnswerSegment] = Field(default_factory=list)
    explanations: List[RagAnswerSegment] = Field(default_factory=list)
    suggestions: List[RagAnswerSegment] = Field(default_factory=list)
    uncertainty: List[str] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


class RagQueryResponse(BaseModel):
    status: str
    blocked: bool = False
    blocked_reason: Optional[str] = None
    answer: Optional[RagAnswer] = None
    citations: List[RagCitation] = Field(default_factory=list)
    query_log_id: Optional[int] = None
    debug: dict[str, Any] = Field(default_factory=dict)


class RagSourceResponse(BaseModel):
    status: str
    sources: List[dict[str, Any]] = Field(default_factory=list)


class RagIndexStatusResponse(BaseModel):
    status: str
    index: dict[str, Any]
