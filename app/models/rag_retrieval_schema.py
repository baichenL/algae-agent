from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


class RetrievalQuery(BaseModel):
    query_id: str = Field(default_factory=lambda: f"retrieval:{uuid.uuid4().hex}")
    original_query: str
    normalized_query: str | None = None
    query_type: str | None = None
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = 5
    sparse_top_k: int = 30
    dense_top_k: int = 30
    rerank_top_k: int = 5
    parent_expansion_enabled: bool = True
    trace_context: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.normalized_query is None:
            self.normalized_query = " ".join(self.original_query.split()).lower()


class RetrievalHit(BaseModel):
    evidence_id: str
    document_id: str | int | None = None
    content: str
    source: str
    sparse_rank: int | None = None
    dense_rank: int | None = None
    semantic_rank: int | None = None
    sparse_score: float | None = None
    dense_score: float | None = None
    semantic_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    final_rank: int | None = None
    selected: bool = False
    rejection_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalBundle(BaseModel):
    query: RetrievalQuery
    sparse_hits: list[RetrievalHit] = Field(default_factory=list)
    dense_hits: list[RetrievalHit] = Field(default_factory=list)
    semantic_hits: list[RetrievalHit] = Field(default_factory=list)
    fused_hits: list[RetrievalHit] = Field(default_factory=list)
    reranked_hits: list[RetrievalHit] = Field(default_factory=list)
    expanded_hits: list[RetrievalHit] = Field(default_factory=list)
    selected_hits: list[RetrievalHit] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
