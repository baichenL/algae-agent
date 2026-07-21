from typing import Literal

from pydantic import BaseModel, Field


SemanticEffect = Literal[
    "knowledge_read",
    "state_read",
    "write_proposal",
    "workflow_execution",
    "communication",
]

EntityType = Literal[
    "culture_medium",
    "medium_variant",
    "chemical",
    "recipe_group",
    "organism",
    "experimental_condition",
    "protocol_material",
    "manual",
    "paper",
    "experiment_data",
    "unknown",
]

RelationType = Literal[
    "overview_of",
    "has_component",
    "has_component_amount",
    "has_component_group",
    "used_in_step",
    "suitable_for",
    "standard_equivalence",
    "supports_conclusion",
    "schema_query",
    "data_analysis",
    "unknown",
]


class TypedEntity(BaseModel):
    entity_type: EntityType
    surface: str
    canonical_id: str
    role: Literal["subject", "object", "condition", "source"] = "object"
    confidence: float = 1.0
    metadata: dict = Field(default_factory=dict)


class EvidenceContract(BaseModel):
    relation: RelationType
    required_fact_types: list[str] = Field(default_factory=list)
    required_slots: list[str] = Field(default_factory=list)
    preferred_sources: list[str] = Field(default_factory=list)
    direct_statement_required: bool = False


class RequestedOutput(BaseModel):
    shape: Literal["yes_no", "value", "list", "procedure", "overview", "comparison", "general"] = "general"
    language: str = "zh"
    concise: bool = False
    max_points: int | None = None


class SemanticSubquery(BaseModel):
    subquery_id: str
    original_text: str
    intent: str
    relation: RelationType
    entities: list[TypedEntity] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    requested_output: RequestedOutput = Field(default_factory=RequestedOutput)
    evidence_contract: EvidenceContract
    confidence: float = 1.0


class SemanticQuery(BaseModel):
    original_question: str
    effect: SemanticEffect = "knowledge_read"
    subqueries: list[SemanticSubquery] = Field(default_factory=list)
    parser: str = "typed_local"
    confidence: float = 1.0

