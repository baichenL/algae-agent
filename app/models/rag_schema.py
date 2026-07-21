from typing import Any, List, Optional

from pydantic import BaseModel, Field


class RagQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = 5
    doc_types: List[str] = Field(default_factory=list)


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
