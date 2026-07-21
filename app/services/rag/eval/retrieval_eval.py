from __future__ import annotations

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


def run_retrieval_eval(top_k: int = 5) -> dict:
    cases = list_rag_eval_cases()
    if not cases:
        seed_default_mini_eval_cases()
        cases = list_rag_eval_cases()
    scored_cases = [case for case in cases if not case.get("expected_refusal")]
    modes = ["fts", "vector", "hybrid"]
    results_by_mode = {mode: [] for mode in modes}
    for case in scored_cases:
        for mode in modes:
            candidates = _retrieve_for_mode(case["question"], mode, top_k=top_k)
            results_by_mode[mode].append(_score_case(case, mode, candidates, top_k=top_k))
    return {
        "case_count": len(scored_cases),
        "top_k": top_k,
        "embedding": embedding_runtime_status(),
        "metrics": {
            mode: _aggregate(items)
            for mode, items in results_by_mode.items()
        },
        "results": results_by_mode,
    }


def _retrieve_for_mode(question: str, mode: str, top_k: int) -> list[dict]:
    if mode == "fts":
        rows = search_rag_chunks(question, top_k=top_k)
        return [{**row, "retrieval_channels": ["fts_eval"]} for row in rows]
    if mode == "vector":
        query_vector = embed_query(question)
        if not query_vector:
            return []
        rows = search_rag_chunk_embeddings(
            query_vector,
            embedding_model=get_embedding_config().model,
            top_k=top_k,
        )
        return [{**row, "retrieval_channels": ["vector_eval"]} for row in rows]
    return retrieve_hybrid_chunks(question, top_k=top_k)


def _score_case(case: dict, mode: str, candidates: list[dict], top_k: int) -> dict:
    expected_tokens = [str(item).lower() for item in case.get("expected_answer_contains") or []]
    required_doc_types = _required_doc_types(case.get("required_evidence_types") or [])
    first_hit_rank = None
    first_type_rank = None
    relevance_by_rank = []
    for index, candidate in enumerate(candidates[:top_k], start=1):
        token_match = _candidate_matches_tokens(candidate, expected_tokens)
        type_match = _candidate_matches_doc_types(candidate, required_doc_types)
        relevance_by_rank.append(1.0 if (token_match or type_match) else 0.0)
        if first_hit_rank is None and token_match:
            first_hit_rank = index
        if first_type_rank is None and type_match:
            first_type_rank = index
    hit_rank = first_hit_rank or first_type_rank
    top_candidate = candidates[0] if candidates else {}
    citation_ready = bool(
        top_candidate.get("source_path")
        and (
            top_candidate.get("section")
            or top_candidate.get("page_number")
            or top_candidate.get("sheet_name")
            or top_candidate.get("file_name")
        )
    )
    return {
        "id": case["id"],
        "category": case["category"],
        "mode": mode,
        "retrieved_count": len(candidates),
        "hit_rank": hit_rank,
        "recall_at_k": bool(hit_rank),
        "mrr": 1.0 / hit_rank if hit_rank else 0.0,
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
