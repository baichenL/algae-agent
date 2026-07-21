import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


QuestionType = Literal[
    "existence",
    "value",
    "list",
    "procedure",
    "comparison",
    "suitability",
    "explanation",
    "analysis",
    "optimization",
    "overview",
]

AnswerShape = Literal["yes_no", "value", "list", "procedure", "overview", "not_found"]

EvidenceRequirement = Literal[
    "exact_match",
    "explicit_statement",
    "structured_schema",
    "structured_data_analysis",
    "citation_required",
    "structured_overview",
]

AnswerabilityStatus = Literal["answered", "partial", "not_found", "blocked"]


class QueryFrame(BaseModel):
    original_question: str
    question_type: QuestionType
    target_entity: str | None = None
    target_attribute: str | None = None
    target_value: str | None = None
    answer_shape: AnswerShape
    evidence_requirement: EvidenceRequirement = "citation_required"
    allow_inference: bool = False
    source_constraint: list[str] = Field(default_factory=list)
    raw_terms: list[str] = Field(default_factory=list)


class EvidenceUnit(BaseModel):
    evidence_id: str
    document_id: str | int | None = None
    document_version: str | None = None
    evidence_type: str | None = None
    content: str = ""
    content_hash: str = ""
    source_type: str
    source_file: str
    fact_type: str
    source_id: str | None = None
    entity: str | None = None
    relation: str | None = None
    target_entity: str | None = None
    attribute: str | None = None
    value: str | None = None
    unit: str | None = None
    text_span: str | None = None
    page_number: int | None = None
    section_path: list[str] = Field(default_factory=list)
    parent_id: str | None = None
    previous_id: str | None = None
    next_id: str | None = None
    source_locator: dict[str, Any] | str | None = None
    parser_version: str | None = None
    element_type: str | None = None
    section: str | None = None
    sheet_name: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    extraction_method: str | None = None
    location: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    citation: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize_modern_fields(self) -> "EvidenceUnit":
        if not self.evidence_type:
            self.evidence_type = self.fact_type
        if not self.document_version:
            self.document_version = (
                self.metadata.get("version")
                or self.citation.get("version")
                or self.metadata.get("document_version")
            )
        if self.page_number is None:
            self.page_number = self.location.get("page_number") or self.citation.get("page_number")
        if not self.section_path:
            raw_section = (
                self.location.get("section_path")
                or self.metadata.get("section_path")
                or self.location.get("section")
                or self.section
            )
            self.section_path = _coerce_section_path(raw_section)
        if not self.parent_id:
            self.parent_id = self.metadata.get("parent_id")
        if not self.previous_id:
            self.previous_id = self.metadata.get("previous_id")
        if not self.next_id:
            self.next_id = self.metadata.get("next_id")
        if not self.element_type:
            self.element_type = self.metadata.get("element_type") or self.fact_type
        if not self.parser_version:
            self.parser_version = self.metadata.get("parser_version")
        if not self.source_locator:
            self.source_locator = _default_source_locator(self)
        if not self.content:
            self.content = _default_content(self)
        if not self.content_hash:
            self.content_hash = build_evidence_content_hash(
                content=self.content,
                source_file=self.source_file,
                document_version=self.document_version,
                source_locator=self.source_locator,
            )
        return self


class KnowledgeSource(BaseModel):
    source_id: str | None = None
    source_type: str = "local_file"
    doc_type: str
    source_path: str
    file_name: str | None = None
    version: str | None = None
    year: str | None = None
    language: str | None = None
    trust_level: str = "lab_internal"
    owner: str | None = None
    ingestion_status: str = "pending"
    content_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    source: KnowledgeSource
    units: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractionReport(BaseModel):
    extractor_name: str
    status: Literal["success", "partial", "failed"] = "success"
    evidence_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class EvidencePlan(BaseModel):
    required_capabilities: list[str] = Field(default_factory=list)
    preferred_evidence_types: list[str] = Field(default_factory=list)
    source_constraints: list[str] = Field(default_factory=list)
    fallback_allowed: bool = True
    closed_world_required: bool = False


class AnswerabilityResult(BaseModel):
    status: AnswerabilityStatus
    reason: str
    usable_evidence_ids: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    closed_world_negative: bool = False


class GroundedCitation(BaseModel):
    source_id: int
    evidence_id: str
    source_file: str
    source_type: str
    location: dict[str, Any] = Field(default_factory=dict)


class GroundedClaim(BaseModel):
    claim_type: Literal["fact", "background", "explanation", "suggestion", "uncertainty"]
    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class GroundedAnswer(BaseModel):
    answerability: AnswerabilityResult
    direct_answer: str
    evidence: list[EvidenceUnit] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)
    citations: list[GroundedCitation] = Field(default_factory=list)
    claims: list[GroundedClaim] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


class AtomicQuestion(BaseModel):
    question_id: str
    text: str
    inherited_context: str | None = None
    original_span: str | None = None


class QuestionAspects(BaseModel):
    subject: str | None = None
    condition: str | None = None
    relation: str | None = None
    target: str | None = None
    expected_answer_type: str = "general"
    required_support: str = "relevant_context"
    raw_terms: list[str] = Field(default_factory=list)


class EvidenceAssessment(BaseModel):
    evidence_id: str
    supports_direct_answer: bool = False
    supports_background: bool = False
    missing_aspects: list[str] = Field(default_factory=list)
    contradicted_aspects: list[str] = Field(default_factory=list)
    reason: str = ""


class SufficiencyResult(BaseModel):
    status: Literal["sufficient", "background_only", "insufficient", "contradicted"] = "insufficient"
    direct_evidence_ids: list[str] = Field(default_factory=list)
    background_evidence_ids: list[str] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)
    contradicted_aspects: list[str] = Field(default_factory=list)
    reason: str = ""
    assessments: list[EvidenceAssessment] = Field(default_factory=list)


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_evidence_content_hash(
    *,
    content: str,
    source_file: str = "",
    document_version: str | None = None,
    source_locator: Any = None,
) -> str:
    payload = {
        "content": content or "",
        "source_file": source_file or "",
        "document_version": document_version or "",
        "source_locator": source_locator or {},
    }
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def _coerce_section_path(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item or "").strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item or "").strip()]
    return [part.strip() for part in str(value).replace(">", "/").split("/") if part.strip()]


def _default_content(unit: EvidenceUnit) -> str:
    return str(
        unit.text_span
        or unit.value
        or unit.attribute
        or unit.fact_type
        or ""
    )


def _default_source_locator(unit: EvidenceUnit) -> dict[str, Any]:
    locator: dict[str, Any] = {
        "source_file": unit.source_file,
        "source_type": unit.source_type,
    }
    for key in (
        "page_number",
        "section",
        "sheet_name",
        "row_start",
        "row_end",
        "row_index",
        "column_index",
        "chunk_index",
        "table_index",
    ):
        value = unit.location.get(key) or unit.citation.get(key)
        if value is not None:
            locator[key] = value
    if unit.section_path:
        locator["section_path"] = unit.section_path
    return locator
