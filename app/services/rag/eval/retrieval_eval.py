from __future__ import annotations

import time

from app.core.database import search_rag_chunk_embeddings, search_rag_chunks
from app.services.rag.embedding_service import embed_query, embedding_runtime_status, get_embedding_config
from app.services.rag.eval.mini_eval import seed_default_mini_eval_cases
from app.core.database import list_rag_eval_cases
from app.services.rag.retrieval.hybrid_retriever import retrieve_hybrid_chunks


EVIDENCE_TYPE_DOC_TYPES = {
    "recipe_component": {"media_recipe"},
    "sop_fact": {"manual"},
    "paper_claim": {"paper"},
    "table_column": {"experiment_data"},
    "data_value": {"experiment_data"},
    "aggregation_trace": {"experiment_data"},
}


def run_retrieval_eval(top_k: int = 20, generation_id: str | None = None) -> dict:
    cases = list_rag_eval_cases()
    if not cases:
        seed_default_mini_eval_cases()
        cases = list_rag_eval_cases()
    scored_cases = [case for case in cases if not case.get("expected_refusal")]
    modes = ["fts", "vector", "hybrid"]
    results_by_mode = {mode: [] for mode in modes}
    for case in scored_cases:
        for mode in modes:
            started = time.perf_counter()
            candidates = _retrieve_for_mode(case["question"], mode, top_k=top_k, generation_id=generation_id)
            result = _score_case(case, mode, candidates, top_k=top_k)
            result["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            results_by_mode[mode].append(result)
    return {
        "case_count": len(scored_cases),
        "top_k": top_k,
        "generation_id": generation_id,
        "embedding": embedding_runtime_status(),
        "metrics": {
            mode: _aggregate(items)
            for mode, items in results_by_mode.items()
        },
        "results": results_by_mode,
    }


def run_agentic_retrieval_eval(generation_id: str | None = None, top_k: int = 20) -> dict:
    """Evaluate only cases explicitly labelled for the bounded agentic path."""
    from app.services.rag.agentic import retrieve_with_budget

    cases = [
        case for case in list_rag_eval_cases()
        if case.get("retrieval_mode") == "agentic" and not case.get("expected_refusal")
    ]
    results = []
    for case in cases:
        metadata_filters = {"generation_id": generation_id} if generation_id else None
        result = retrieve_with_budget(
            case["question"], top_k=top_k, metadata_filters=metadata_filters,
        )
        scored = _score_case(case, "agentic", result.chunks, top_k=top_k)
        scored["latency_ms"] = float(result.debug.get("elapsed_ms") or 0.0)
        scored["rounds"] = int(result.debug.get("rounds") or 0)
        scored["retrieval_calls"] = int(result.debug.get("retrieval_calls") or 0)
        scored["stop_reason"] = result.debug.get("stop_reason")
        results.append(scored)
    return {"case_count": len(cases), "metrics": _aggregate(results), "results": results}


def _retrieve_for_mode(question: str, mode: str, top_k: int, generation_id: str | None = None) -> list[dict]:
    if mode == "fts":
        rows = search_rag_chunks(question, top_k=top_k, generation_id=generation_id)
        return [{**row, "retrieval_channels": ["fts_eval"]} for row in rows]
    if mode == "vector":
        query_vector = embed_query(question)
        if not query_vector:
            return []
        rows = search_rag_chunk_embeddings(
            query_vector,
            embedding_model=get_embedding_config().model,
            top_k=top_k,
            generation_id=generation_id,
        )
        return [{**row, "retrieval_channels": ["vector_eval"]} for row in rows]
    return retrieve_hybrid_chunks(question, top_k=top_k, metadata_filters={"generation_id": generation_id} if generation_id else None)


def _score_case(case: dict, mode: str, candidates: list[dict], top_k: int) -> dict:
    expected_tokens = [str(item).lower() for item in case.get("expected_answer_contains") or []]
    expected_labels = case.get("expected_evidence_labels") or []
    required_doc_types = _required_doc_types(case.get("required_evidence_types") or [])
    first_hit_rank = None
    first_type_rank = None
    relevance_by_rank = []
    for index, candidate in enumerate(candidates[:top_k], start=1):
        label_match = _candidate_matches_evidence_labels(candidate, expected_labels)
        token_match = _candidate_matches_tokens(candidate, expected_tokens)
        type_match = _candidate_matches_doc_types(candidate, required_doc_types)
        relevant = label_match if expected_labels else (token_match or type_match)
        relevance_by_rank.append(1.0 if relevant else 0.0)
        if first_hit_rank is None and (label_match if expected_labels else token_match):
            first_hit_rank = index
        if first_type_rank is None and type_match:
            first_type_rank = index
    hit_rank = first_hit_rank or first_type_rank
    top_candidate = candidates[0] if candidates else {}
    citation_ready = (
        _candidate_matches_evidence_labels(top_candidate, expected_labels)
        if expected_labels
        else bool(
            top_candidate.get("source_path")
            and (
                top_candidate.get("section")
                or top_candidate.get("page_number")
                or top_candidate.get("sheet_name")
                or top_candidate.get("file_name")
            )
        )
    )
    return {
        "id": case["id"],
        "category": case["category"],
        "mode": mode,
        "retrieved_count": len(candidates),
        "hit_rank": hit_rank,
        "recall_at_k": bool(hit_rank),
        "mrr": 1.0 / hit_rank if hit_rank and hit_rank <= 10 else 0.0,
        "ndcg": _ndcg(relevance_by_rank),
        "citation_accuracy": bool(hit_rank and citation_ready),
        "evidence_type_accuracy": bool(first_type_rank) if required_doc_types else None,
        "retrieved_ids": [item.get("chunk_id") for item in candidates[:top_k]],
        "retrieval_channels": sorted(
            {
                channel
                for item in candidates[:top_k]
                for channel in (item.get("retrieval_channels") or [])
            }
        ),
    }


def _aggregate(items: list[dict]) -> dict:
    if not items:
        return {
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "citation_accuracy": 0.0,
            "evidence_type_accuracy": 0.0,
            "p95_latency_ms": 0.0,
        }
    evidence_items = [item for item in items if item.get("evidence_type_accuracy") is not None]
    return {
        "recall_at_k": _mean(1.0 if item["recall_at_k"] else 0.0 for item in items),
        "mrr": _mean(float(item["mrr"]) for item in items),
        "ndcg": _mean(float(item.get("ndcg") or 0.0) for item in items),
        "citation_accuracy": _mean(1.0 if item["citation_accuracy"] else 0.0 for item in items),
        "evidence_type_accuracy": _mean(
            1.0 if item["evidence_type_accuracy"] else 0.0
            for item in evidence_items
        )
        if evidence_items
        else 0.0,
        "p95_latency_ms": _percentile([float(item.get("latency_ms") or 0) for item in items], 0.95),
    }


def _candidate_matches_tokens(candidate: dict, expected_tokens: list[str]) -> bool:
    if not expected_tokens:
        return False
    text = " ".join(
        str(value or "")
        for value in [
            candidate.get("file_name"),
            candidate.get("title"),
            candidate.get("section"),
            candidate.get("content"),
            candidate.get("answer_context"),
            candidate.get("topic"),
            (candidate.get("metadata") or {}).get("topic"),
        ]
    ).lower()
    return all(token in text for token in expected_tokens)


def _candidate_matches_evidence_labels(candidate: dict, labels: list[dict]) -> bool:
    """Match generation-stable labels without relying on internal chunk ids."""
    if not labels:
        return False
    metadata = candidate.get("metadata") or {}
    source_id = candidate.get("source_id") or candidate.get("knowledge_source_id") or metadata.get("source_id")
    content_hash = candidate.get("content_hash") or metadata.get("content_hash")
    locators = {
        str(value)
        for value in (
            candidate.get("source_locator"),
            candidate.get("section"),
            candidate.get("page_number"),
            candidate.get("sheet_name"),
            metadata.get("source_locator"),
        )
        if value not in (None, "")
    }
    for label in labels:
        expected_source = label.get("knowledge_source_id") or label.get("source_id")
        expected_hash = label.get("content_hash")
        expected_locator = label.get("source_locator")
        if expected_source and str(source_id or "") != str(expected_source):
            continue
        if expected_hash and str(content_hash or "") != str(expected_hash):
            continue
        if expected_locator and str(expected_locator) not in locators:
            continue
        if expected_source or expected_hash or expected_locator:
            return True
    return False


def _candidate_matches_doc_types(candidate: dict, required_doc_types: set[str]) -> bool:
    return bool(required_doc_types) and candidate.get("doc_type") in required_doc_types


def _required_doc_types(required_evidence_types: list[str]) -> set[str]:
    doc_types: set[str] = set()
    for evidence_type in required_evidence_types:
        doc_types.update(EVIDENCE_TYPE_DOC_TYPES.get(evidence_type, set()))
    return doc_types


def _mean(values) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _ndcg(relevance_by_rank: list[float]) -> float:
    if not relevance_by_rank:
        return 0.0
    dcg = sum(rel / _log2(rank + 1) for rank, rel in enumerate(relevance_by_rank, start=1))
    ideal = sorted(relevance_by_rank, reverse=True)
    idcg = sum(rel / _log2(rank + 1) for rank, rel in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def _log2(value: int) -> float:
    import math

    return math.log2(value)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round((len(ordered) - 1) * quantile)), len(ordered) - 1)
    return ordered[index]
