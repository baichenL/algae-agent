from __future__ import annotations

from app.services.rag.retrieval.hybrid_retriever import retrieve_hybrid_bundle
from app.services.rag.retrieval.parent_expansion import expand_parent_context
from app.services.rag.retrieval.reranker import rerank_chunks


def retrieve_chunks(
    question: str,
    top_k: int = 5,
    doc_types: list[str] | None = None,
    metadata_filters: dict | None = None,
) -> list[dict]:
    final_top_k = min(max(int(top_k or 5), 1), 20)
    candidate_top_k = max(30, final_top_k)
    bundle = retrieve_hybrid_bundle(
        question,
        top_k=candidate_top_k,
        doc_types=doc_types,
        metadata_filters=metadata_filters,
    )
    candidates = [_hit_to_chunk(hit) for hit in bundle.selected_hits]
    reranked = rerank_chunks(question, candidates[:30], top_k=min(30, len(candidates)))
    diversified = _diversify_final(
        reranked,
        final_top_k,
        per_document_limit=None if _single_document_query(metadata_filters) else 2,
    )
    expanded = expand_parent_context(diversified)
    final_ids = {f"chunk:{item.get('chunk_id')}" for item in diversified}
    rerank_by_id = {
        f"chunk:{item.get('chunk_id')}": {
            "rerank_score": item.get("rerank_score"), "final_rank": item.get("final_rank"),
            "reranker_backend": item.get("reranker_backend"), "reranker_version": item.get("reranker_version"),
        }
        for item in reranked
    }
    trace = {
        "query": bundle.query.model_dump(),
        "warnings": bundle.warnings,
        "timings_ms": bundle.timings_ms,
        "candidates": [
            {
                "evidence_id": hit.evidence_id,
                "sparse_rank": hit.sparse_rank,
                "dense_rank": hit.dense_rank,
                "semantic_rank": hit.semantic_rank,
                "fusion_score": hit.fusion_score,
                "rerank_score": (rerank_by_id.get(hit.evidence_id) or {}).get("rerank_score"),
                "final_rank": (rerank_by_id.get(hit.evidence_id) or {}).get("final_rank"),
                "document_id": hit.document_id,
                "selected": hit.evidence_id in final_ids,
                "rejection_reason": None if hit.evidence_id in final_ids else "not_final_after_rerank_or_diversity",
                "model_versions": {
                    "reranker_backend": (rerank_by_id.get(hit.evidence_id) or {}).get("reranker_backend"),
                    "reranker_version": (rerank_by_id.get(hit.evidence_id) or {}).get("reranker_version"),
                },
            }
            for hit in bundle.fused_hits
        ],
        "reranked": [
            {
                "chunk_id": item.get("chunk_id"),
                "rerank_score": item.get("rerank_score"),
                "final_rank": item.get("final_rank"),
            }
            for item in reranked
        ],
    }
    for item in expanded:
        item["retrieval_trace"] = trace
        item["retrieval_degraded"] = bool(bundle.warnings)
        item["retrieval_degraded_reasons"] = list(bundle.warnings)
    return expanded


def _hit_to_chunk(hit) -> dict:
    chunk = dict(hit.metadata)
    channels = set(chunk.get("retrieval_channels") or [])
    if hit.sparse_rank is not None:
        channels.add("fts")
    if hit.dense_rank is not None:
        channels.add("vector")
    if hit.semantic_rank is not None:
        channels.add("semantic_fallback")
    chunk.update(
        {
            "retrieval_channels": sorted(channels),
            "sparse_rank": hit.sparse_rank,
            "dense_rank": hit.dense_rank,
            "semantic_rank": hit.semantic_rank,
            "sparse_score": hit.sparse_score,
            "dense_score": hit.dense_score,
            "vector_score": hit.dense_score or chunk.get("vector_score") or 0.0,
            "semantic_score": hit.semantic_score or chunk.get("semantic_score") or 0.0,
            "fusion_score": hit.fusion_score,
            "hybrid_score": hit.fusion_score or 0.0,
            "retrieval_hit": hit.model_dump(),
        }
    )
    return chunk


def _diversify_final(candidates: list[dict], top_k: int, per_document_limit: int | None) -> list[dict]:
    result: list[dict] = []
    counts: dict[str, int] = {}
    seen_content: set[str] = set()
    rejected: list[dict] = []
    for item in candidates:
        fingerprint = str(item.get("content_hash") or item.get("content") or "").strip().casefold()
        if fingerprint in seen_content:
            item["rejection_reason"] = "duplicate_content"
            rejected.append(item)
            continue
        seen_content.add(fingerprint)
        document_key = str(item.get("document_id") or item.get("source_id") or item.get("source_path") or "")
        if per_document_limit is not None and counts.get(document_key, 0) >= per_document_limit:
            item["rejection_reason"] = "per_document_limit"
            rejected.append(item)
            continue
        counts[document_key] = counts.get(document_key, 0) + 1
        item["selected"] = True
        item["final_rank"] = len(result) + 1
        result.append(item)
        if len(result) >= top_k:
            break
    return result


def _single_document_query(filters: dict | None) -> bool:
    filters = filters or {}
    source_ids = filters.get("source_ids") or ([] if not filters.get("source_id") else [filters["source_id"]])
    return len(source_ids) == 1
