from __future__ import annotations

import os

from app.models.rag_retrieval_schema import RetrievalHit


def rrf_k() -> int:
    return max(int(os.getenv("RAG_RRF_K", "60")), 1)


def sparse_weight() -> float:
    return float(os.getenv("RAG_SPARSE_WEIGHT", "1.0"))


def dense_weight() -> float:
    return float(os.getenv("RAG_DENSE_WEIGHT", "1.0"))


def reciprocal_rank_fusion(
    sparse_hits: list[RetrievalHit],
    dense_hits: list[RetrievalHit],
    semantic_hits: list[RetrievalHit] | None = None,
    *,
    k: int | None = None,
    sparse: float | None = None,
    dense: float | None = None,
    semantic: float | None = None,
) -> list[RetrievalHit]:
    rank_constant = k or rrf_k()
    weights = {
        "sparse": sparse_weight() if sparse is None else sparse,
        "dense": dense_weight() if dense is None else dense,
        "semantic": (dense_weight() * 0.5) if semantic is None else semantic,
    }
    merged: dict[str, RetrievalHit] = {}
    scores: dict[str, float] = {}
    for channel, hits in (
        ("sparse", sparse_hits),
        ("dense", dense_hits),
        ("semantic", semantic_hits or []),
    ):
        for rank, hit in enumerate(hits, start=1):
            existing = merged.get(hit.evidence_id)
            if existing is None:
                existing = hit.model_copy(deep=True)
                merged[hit.evidence_id] = existing
            else:
                existing.metadata.update({key: value for key, value in hit.metadata.items() if value is not None})
            if channel == "sparse":
                existing.sparse_rank = existing.sparse_rank or rank
                existing.sparse_score = existing.sparse_score if existing.sparse_score is not None else hit.sparse_score
            elif channel == "dense":
                existing.dense_rank = existing.dense_rank or rank
                existing.dense_score = max(existing.dense_score or 0.0, hit.dense_score or 0.0)
            else:
                existing.semantic_rank = existing.semantic_rank or rank
                existing.semantic_score = max(existing.semantic_score or 0.0, hit.semantic_score or 0.0)
            scores[hit.evidence_id] = scores.get(hit.evidence_id, 0.0) + weights[channel] / (rank_constant + rank)

    fused = []
    for evidence_id, hit in merged.items():
        hit.fusion_score = scores.get(evidence_id, 0.0)
        fused.append(hit)
    fused.sort(
        key=lambda item: (
            -float(item.fusion_score or 0.0),
            str(item.metadata.get("file_name") or ""),
            int(item.metadata.get("chunk_index") or 0),
            item.evidence_id,
        )
    )
    for index, hit in enumerate(fused, start=1):
        hit.final_rank = index
    return fused
