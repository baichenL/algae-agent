import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass

from app.models.rag_retrieval_schema import RetrievalBundle, RetrievalHit, RetrievalQuery
from app.core.database import (
    list_rag_chunks_for_semantic,
    list_rag_knowledge_sources,
    search_rag_chunk_embeddings,
    search_rag_chunks,
)
from app.services.rag.embedding_service import embed_query, embedding_runtime_status, get_embedding_config
from app.services.rag.retrieval.rrf import reciprocal_rank_fusion


DOC_TYPE_PRIORITY = {
    "media_recipe": 0,
    "manual": 1,
    "experiment_data": 2,
    "paper": 3,
    "document": 4,
}


DOMAIN_EXPANSIONS = {
    "tap": ["tap", "tris", "acetate", "phosphate", "medium", "recipe"],
    "medium": ["medium", "recipe", "composition", "component"],
    "recipe": ["recipe", "composition", "component", "amount"],
    "composition": ["composition", "component", "recipe"],
    "component": ["component", "composition", "ingredient"],
    "phosphate": ["phosphate", "k2hpo4", "kh2po4"],
    "salt": ["salt", "nh4cl", "mgso4", "cacl2"],
    "trace": ["trace", "hutner", "edta", "znso4", "feso4"],
    "manual": ["manual", "sop", "protocol", "procedure", "step"],
    "sop": ["sop", "manual", "protocol", "procedure", "step"],
    "protocol": ["protocol", "manual", "sop", "procedure", "step"],
    "procedure": ["procedure", "step", "operation", "protocol"],
    "detect": ["detect", "check", "inspection", "microscopy"],
    "polluted": ["polluted", "contamination", "contaminated"],
    "contaminated": ["contaminated", "contamination", "polluted"],
    "contamination": ["contamination", "contaminated", "polluted", "microscopy", "inspection"],
    "paper": ["paper", "literature", "claim", "method"],
    "literature": ["literature", "paper", "claim", "method"],
    "machine": ["machine", "learning", "model", "prediction"],
    "learning": ["learning", "machine", "model", "prediction"],
    "od750": ["od750", "od", "optical", "density", "biomass"],
    "od": ["od", "od750", "optical", "density"],
    "biomass": ["biomass", "growth"],
    "growth": ["growth", "curve", "biomass"],
    "ph": ["ph"],
    "co2": ["co2"],
    "temperature": ["temperature", "temp"],
    "light": ["light", "irradiance"],
}


@dataclass
class HybridCandidate:
    chunk: dict
    lexical_score: float | None = None
    semantic_score: float = 0.0
    vector_score: float = 0.0
    metadata_boost: float = 0.0
    hybrid_score: float = 0.0


def retrieve_hybrid_chunks(
    question: str,
    top_k: int = 5,
    doc_types: list[str] | None = None,
    metadata_filters: dict | None = None,
) -> list[dict]:
    bundle = retrieve_hybrid_bundle(question, top_k=top_k, doc_types=doc_types, metadata_filters=metadata_filters)
    return [_hit_to_chunk(hit) for hit in bundle.selected_hits]


def retrieve_hybrid_bundle(
    question: str,
    top_k: int = 5,
    doc_types: list[str] | None = None,
    metadata_filters: dict | None = None,
) -> RetrievalBundle:
    requested = max(int(top_k or 5), 1)
    query = RetrievalQuery(
        original_query=question,
        metadata_filters=metadata_filters or {},
        top_k=requested,
        sparse_top_k=max(requested * 6, 12),
        dense_top_k=max(requested * 6, 12),
        rerank_top_k=requested,
    )
    timings = {}
    warnings = []

    started = time.perf_counter()
    lexical_rows = search_rag_chunks(question, top_k=query.sparse_top_k, doc_types=doc_types)
    lexical_rows = _filter_rows(lexical_rows, metadata_filters)
    sparse_hits = [_row_to_hit(row, "sparse", rank) for rank, row in enumerate(lexical_rows, start=1)]
    timings["sparse"] = _elapsed_ms(started)

    started = time.perf_counter()
    vector_rows = _embedding_search(question, top_k=query.dense_top_k, doc_types=doc_types)
    vector_rows = _filter_rows(vector_rows, metadata_filters)
    if not vector_rows:
        warnings.append(f"dense_retrieval_unavailable:{embedding_runtime_status()['status']}")
    elif embedding_runtime_status().get("degraded"):
        warnings.append(f"dense_retrieval_degraded:{embedding_runtime_status().get('degraded_reason')}")
    dense_hits = [_row_to_hit(row, "dense", rank) for rank, row in enumerate(vector_rows, start=1)]
    timings["dense"] = _elapsed_ms(started)

    started = time.perf_counter()
    semantic_rows = _semantic_search(question, top_k=query.sparse_top_k, doc_types=doc_types)
    semantic_rows = _filter_rows(semantic_rows, metadata_filters)
    semantic_hits = [_row_to_hit(row, "semantic_fallback", rank) for rank, row in enumerate(semantic_rows, start=1)]
    timings["semantic_fallback"] = _elapsed_ms(started)

    started = time.perf_counter()
    fused = reciprocal_rank_fusion(sparse_hits, dense_hits, semantic_hits)
    fused.sort(
        key=lambda hit: (
            DOC_TYPE_PRIORITY.get(hit.metadata.get("doc_type"), 99),
            -float(hit.fusion_score or 0.0),
            hit.metadata.get("file_name") or "",
            int(hit.metadata.get("chunk_index") or 0),
            hit.evidence_id,
        )
    )
    timings["fusion"] = _elapsed_ms(started)

    selected = _dedupe_hits(fused, requested)
    selected_ids = {hit.evidence_id for hit in selected}
    selected = [hit.model_copy(update={"selected": True, "final_rank": index}) for index, hit in enumerate(selected, start=1)]
    fused = [
        hit.model_copy(update={"selected": hit.evidence_id in selected_ids})
        for hit in fused
    ]
    return RetrievalBundle(
        query=query,
        sparse_hits=sparse_hits,
        dense_hits=dense_hits,
        semantic_hits=semantic_hits,
        fused_hits=fused,
        selected_hits=selected,
        warnings=warnings,
        timings_ms=timings,
    )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def _dense_retrieval_enabled() -> bool:
    return os.getenv("RAG_DENSE_RETRIEVAL_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def _row_to_hit(row: dict, source: str, rank: int) -> RetrievalHit:
    chunk_id = row.get("chunk_id") or row.get("evidence_id") or f"{source}:{rank}"
    metadata = {**row, "retrieval_source": source}
    if source == "sparse":
        metadata["retrieval_channels"] = ["fts"]
    elif source == "dense":
        metadata["retrieval_channels"] = ["vector"]
    else:
        metadata["retrieval_channels"] = ["semantic_fallback"]
    return RetrievalHit(
        evidence_id=f"chunk:{chunk_id}",
        document_id=row.get("document_id"),
        content=row.get("content") or "",
        source=source,
        sparse_rank=rank if source == "sparse" else None,
        dense_rank=rank if source == "dense" else None,
        semantic_rank=rank if source == "semantic_fallback" else None,
        sparse_score=float(row.get("score")) if source == "sparse" and row.get("score") is not None else None,
        dense_score=float(row.get("vector_score")) if source == "dense" and row.get("vector_score") is not None else None,
        semantic_score=float(row.get("semantic_score")) if source == "semantic_fallback" and row.get("semantic_score") is not None else None,
        metadata=metadata,
    )


def _hit_to_chunk(hit: RetrievalHit) -> dict:
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
            "selected": hit.selected,
            "rejection_reason": hit.rejection_reason,
            "retrieval_hit": hit.model_dump(),
        }
    )
    return chunk


def _dedupe_hits(hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
    selected = []
    seen = set()
    for hit in hits:
        fingerprint = (
            str(hit.metadata.get("file_name") or "").casefold(),
            str(hit.metadata.get("doc_type") or "").casefold(),
            str(hit.metadata.get("section") or "").casefold(),
            hit.metadata.get("page_number"),
            hit.metadata.get("sheet_name"),
            hit.metadata.get("row_start"),
            hit.metadata.get("row_end"),
            hit.metadata.get("content_hash") or hit.content.strip(),
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        selected.append(hit)
        if len(selected) >= top_k:
            break
    return selected


def _filter_rows(rows: list[dict], metadata_filters: dict | None) -> list[dict]:
    filters = metadata_filters or {}
    if not filters:
        return rows
    sources_by_path = {}
    if filters.get("trust_level") or filters.get("source_type"):
        sources_by_path = {
            row.get("source_path"): row
            for row in list_rag_knowledge_sources()
        }
    return [
        row
        for row in rows
        if _matches_metadata_filter(row, filters, sources_by_path)
    ]


def _matches_metadata_filter(row: dict, filters: dict, sources_by_path: dict) -> bool:
    metadata = row.get("metadata") or {}
    for key in ("doc_type", "year", "version", "language"):
        wanted = filters.get(key)
        if wanted and str(row.get(key) or metadata.get(key) or "") != str(wanted):
            return False
    source = sources_by_path.get(row.get("source_path")) or {}
    for key in ("trust_level", "source_type"):
        wanted = filters.get(key)
        if wanted and str(source.get(key) or "") != str(wanted):
            return False
    # evidence_type is reserved for unified EvidenceUnit retrieval; chunk rows are text evidence.
    wanted_evidence = filters.get("evidence_type")
    if wanted_evidence and wanted_evidence != "text_chunk":
        return False
    return True


def _merge_candidates(
    lexical_rows: list[dict],
    semantic_rows: list[dict],
    vector_rows: list[dict],
    question: str,
    metadata_filters: dict | None = None,
) -> dict[str, HybridCandidate]:
    query_vector = _semantic_vector(question)
    merged: dict[str, HybridCandidate] = {}
    for rank, row in enumerate(lexical_rows, start=1):
        chunk_id = row.get("chunk_id")
        if not chunk_id:
            continue
        score = _lexical_rank_score(rank, row.get("score"))
        semantic = _cosine(query_vector, _chunk_vector(row))
        merged[chunk_id] = HybridCandidate(
            chunk={**row, "retrieval_channels": ["fts"]},
            lexical_score=score,
            semantic_score=semantic,
        )

    for row in vector_rows:
        chunk_id = row.get("chunk_id")
        if not chunk_id:
            continue
        vector_score = float(row.get("vector_score") or 0.0)
        if chunk_id in merged:
            merged[chunk_id].vector_score = max(merged[chunk_id].vector_score, vector_score)
            merged[chunk_id].chunk["vector_backend"] = row.get("vector_backend")
            merged[chunk_id].chunk["embedding_model"] = row.get("embedding_model")
            merged[chunk_id].chunk.setdefault("retrieval_channels", ["fts"])
            if "vector" not in merged[chunk_id].chunk["retrieval_channels"]:
                merged[chunk_id].chunk["retrieval_channels"].append("vector")
        else:
            merged[chunk_id] = HybridCandidate(
                chunk={**row, "retrieval_channels": ["vector"]},
                lexical_score=0.0,
                semantic_score=_cosine(query_vector, _chunk_vector(row)),
                vector_score=vector_score,
            )

    for rank, row in enumerate(semantic_rows, start=1):
        chunk_id = row.get("chunk_id")
        if not chunk_id:
            continue
        semantic = float(row.get("semantic_score") or 0.0)
        if chunk_id in merged:
            merged[chunk_id].semantic_score = max(merged[chunk_id].semantic_score, semantic)
            merged[chunk_id].chunk.setdefault("retrieval_channels", ["fts"])
            if "semantic" not in merged[chunk_id].chunk["retrieval_channels"]:
                merged[chunk_id].chunk["retrieval_channels"].append("semantic")
        else:
            merged[chunk_id] = HybridCandidate(
                chunk={**row, "retrieval_channels": ["semantic"]},
                lexical_score=_lexical_rank_score(rank, None) * 0.35,
                semantic_score=semantic,
            )

    for candidate in merged.values():
        lexical = candidate.lexical_score or 0.0
        vector = candidate.vector_score
        semantic_fallback = candidate.semantic_score if vector <= 0 else 0.0
        candidate.metadata_boost = _metadata_boost(candidate.chunk, metadata_filters)
        # True hybrid fusion: FTS handles exact lab terms, embedding handles semantic recall,
        # and metadata/evidence boosts keep answers inside the right lab source boundary.
        candidate.hybrid_score = (
            (0.50 * lexical)
            + (0.40 * vector)
            + (0.20 * semantic_fallback)
            + (0.10 * candidate.metadata_boost)
        )
        candidate.chunk["semantic_score"] = candidate.semantic_score
        candidate.chunk["vector_score"] = candidate.vector_score
        candidate.chunk["metadata_boost"] = candidate.metadata_boost
        candidate.chunk["hybrid_score"] = candidate.hybrid_score
    return merged


def _embedding_search(question: str, top_k: int, doc_types: list[str] | None = None) -> list[dict]:
    if not _dense_retrieval_enabled():
        return []
    status = embedding_runtime_status()
    if _strict_mode() and status.get("degraded"):
        raise RuntimeError(f"Strict production mode rejects degraded dense retrieval: {status.get('degraded_reason')}")
    if status["status"] in {"disabled", "disabled_missing_api_key"}:
        if _strict_mode():
            raise RuntimeError(f"Strict production mode requires dense retrieval embeddings: {status['status']}")
        return []
    query_vector = embed_query(question)
    if not query_vector:
        return []
    rows = search_rag_chunk_embeddings(
        query_vector,
        embedding_model=get_embedding_config().model,
        top_k=top_k,
        doc_types=doc_types,
    )
    return [
        {
            **row,
            "retrieval_channels": ["vector"],
            "vector_backend": status["vector_backend_active"],
            "vector_backend_degraded": bool(status.get("degraded")),
            "vector_backend_degraded_reason": status.get("degraded_reason"),
        }
        for row in rows
    ]


def _strict_mode() -> bool:
    return os.getenv("RAG_STRICT_PRODUCTION", "false").strip().lower() in {"1", "true", "yes", "on"}


def _metadata_boost(chunk: dict, metadata_filters: dict | None) -> float:
    boost = 0.0
    metadata = chunk.get("metadata") or {}
    if chunk.get("doc_type") in {"media_recipe", "manual", "experiment_data"}:
        boost += 0.35
    for key in ("year", "version", "language"):
        if chunk.get(key) or metadata.get(key):
            boost += 0.08
    if metadata_filters:
        for key, value in metadata_filters.items():
            if not value:
                continue
            if str(chunk.get(key) or metadata.get(key) or "") == str(value):
                boost += 0.12
    return min(boost, 1.0)


def _semantic_search(question: str, top_k: int, doc_types: list[str] | None = None) -> list[dict]:
    query_vector = _semantic_vector(question)
    if not query_vector:
        return []
    rows = list_rag_chunks_for_semantic(doc_types=doc_types, limit=1000)
    scored = []
    for row in rows:
        score = _cosine(query_vector, _chunk_vector(row))
        if score <= 0:
            continue
        scored.append({**row, "semantic_score": score})
    scored.sort(
        key=lambda row: (
            DOC_TYPE_PRIORITY.get(row.get("doc_type"), 99),
            -float(row.get("semantic_score") or 0.0),
            row.get("file_name") or "",
            int(row.get("chunk_index") or 0),
        )
    )
    return scored[: max(int(top_k or 5), 1)]


def _diversify(candidates: list[HybridCandidate], top_k: int) -> list[HybridCandidate]:
    seen_documents: set[int] = set()
    seen_content: set[tuple] = set()
    diversified: list[HybridCandidate] = []
    remaining: list[HybridCandidate] = []
    for item in candidates:
        fingerprint = _content_fingerprint(item.chunk)
        if fingerprint in seen_content:
            continue
        seen_content.add(fingerprint)
        document_id = int(item.chunk.get("document_id") or 0)
        if document_id and document_id not in seen_documents:
            diversified.append(item)
            seen_documents.add(document_id)
        else:
            remaining.append(item)
        if len(diversified) >= top_k:
            break
    for item in remaining:
        if len(diversified) >= top_k:
            break
        diversified.append(item)
    return diversified


def _content_fingerprint(chunk: dict) -> tuple:
    return (
        str(chunk.get("file_name") or "").casefold(),
        str(chunk.get("doc_type") or "").casefold(),
        str(chunk.get("section") or "").casefold(),
        chunk.get("page_number"),
        chunk.get("sheet_name"),
        chunk.get("row_start"),
        chunk.get("row_end"),
        chunk.get("content_hash") or str(chunk.get("content") or "").strip(),
    )


def _lexical_rank_score(rank: int, bm25_score: object) -> float:
    base = 1.0 / max(rank, 1)
    try:
        bm25 = abs(float(bm25_score))
    except (TypeError, ValueError):
        return base
    return base + min(bm25 / 10.0, 0.25)


def _chunk_vector(chunk: dict) -> Counter:
    text = " ".join(
        str(part or "")
        for part in [
            chunk.get("file_name"),
            chunk.get("title"),
            chunk.get("section"),
            chunk.get("topic"),
            chunk.get("content"),
        ]
    )
    return _semantic_vector(text)


def _semantic_vector(text: str) -> Counter:
    tokens = _tokens(text)
    vector: Counter = Counter(tokens)
    for token in list(tokens):
        for expanded in DOMAIN_EXPANSIONS.get(token, []):
            vector[expanded] += 0.65
    for gram in _char_ngrams(text):
        vector[gram] += 0.35
    return vector


def _tokens(text: str) -> list[str]:
    raw = str(text or "").lower()
    raw = raw.replace("鈧?", "").replace("渭", "u").replace("碌", "u")
    tokens = re.findall(r"[a-z0-9_+\-]+|[\u4e00-\u9fff]{2,}", raw)
    normalized = []
    for token in tokens:
        item = token.strip("_-+")
        if len(item) >= 2 or item in {"n", "p"}:
            normalized.append(item)
    return normalized


def _char_ngrams(text: str, size: int = 3) -> list[str]:
    compact = re.sub(r"\s+", "", str(text or "").lower())
    if len(compact) < size:
        return []
    return [compact[index : index + size] for index in range(0, min(len(compact) - size + 1, 160))]


def _cosine(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    numerator = sum(float(left[key]) * float(right[key]) for key in shared)
    if numerator <= 0:
        return 0.0
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left.values()))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
