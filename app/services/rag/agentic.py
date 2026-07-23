from __future__ import annotations

import time
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from dataclasses import dataclass, field

from app.services.rag.evidence.decomposer import decompose_question
from app.services.rag.query_normalizer import normalize_rag_query
from app.services.rag.retrieval.retriever import retrieve_chunks


MAX_ROUNDS = 3
MAX_SUBQUESTIONS = 6
MAX_RETRIEVAL_CALLS = 8
MAX_EVIDENCE = 40
TOTAL_BUDGET_SECONDS = 12.0


@dataclass
class AgenticRetrievalResult:
    chunks: list[dict] = field(default_factory=list)
    debug: dict = field(default_factory=dict)


def select_retrieval_mode(question: str, requested: str = "auto") -> str:
    if os.getenv("RAG_AGENTIC_ENABLED", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        return "simple"
    if requested in {"simple", "agentic"}:
        return requested
    return "agentic" if normalize_rag_query(question).intent == "complex" else "simple"


def retrieve_with_budget(
    question: str,
    *,
    top_k: int,
    doc_types: list[str] | None = None,
    metadata_filters: dict | None = None,
) -> AgenticRetrievalResult:
    started = time.perf_counter()
    atomics = decompose_question(question)[:MAX_SUBQUESTIONS]
    pending = [item.text for item in atomics] or [question]
    candidates: dict[str, dict] = {}
    calls = 0
    rounds = 0
    stop_reason = "budget_exhausted"
    round_trace = []

    while pending and rounds < MAX_ROUNDS and calls < MAX_RETRIEVAL_CALLS:
        remaining = TOTAL_BUDGET_SECONDS - (time.perf_counter() - started)
        if remaining <= 0:
            stop_reason = "time_budget_exhausted"
            break
        rounds += 1
        batch = pending[: MAX_RETRIEVAL_CALLS - calls]
        pending = []
        round_hits = []
        executor = ThreadPoolExecutor(max_workers=min(4, len(batch)))
        try:
            future_map = {
                executor.submit(
                    retrieve_chunks,
                    subquery,
                    min(max(top_k * 2, 6), 20),
                    doc_types,
                    metadata_filters,
                ): subquery
                for subquery in batch
            }
            calls += len(future_map)
            try:
                for future in as_completed(future_map, timeout=max(remaining, 0.1)):
                    subquery = future_map[future]
                    try:
                        hits = future.result()
                    except Exception as exc:
                        round_hits.append({"query": subquery, "error": str(exc), "count": 0})
                        continue
                    round_hits.append({"query": subquery, "count": len(hits)})
                    for hit in hits:
                        key = str(hit.get("evidence_id") or hit.get("chunk_id") or hit.get("content_hash"))
                        current = candidates.get(key)
                        if current is None or float(hit.get("rerank_score") or 0) > float(current.get("rerank_score") or 0):
                            candidates[key] = hit
            except FuturesTimeoutError:
                stop_reason = "time_budget_exhausted"
                pending = []
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        ranked = sorted(candidates.values(), key=lambda item: -float(item.get("rerank_score") or item.get("hybrid_score") or 0))[:MAX_EVIDENCE]
        sufficient, missing = _evidence_sufficient(question, ranked)
        round_trace.append({"round": rounds, "queries": round_hits, "candidate_count": len(ranked), "missing": missing})
        if sufficient:
            stop_reason = "evidence_sufficient"
            break
        if calls >= MAX_RETRIEVAL_CALLS:
            stop_reason = "retrieval_call_budget_exhausted"
            break
        pending = _focused_rewrites(question, missing, rounds)[: MAX_RETRIEVAL_CALLS - calls]

    ranked = sorted(candidates.values(), key=lambda item: -float(item.get("rerank_score") or item.get("hybrid_score") or 0))
    selected = _diversify(ranked, min(max(top_k, 1), 20))
    return AgenticRetrievalResult(
        chunks=selected,
        debug={
            "mode": "agentic", "rounds": rounds, "retrieval_calls": calls,
            "subquestion_count": len(atomics), "candidate_count": min(len(ranked), MAX_EVIDENCE),
            "stop_reason": stop_reason, "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "budgets": {"max_rounds": MAX_ROUNDS, "max_subquestions": MAX_SUBQUESTIONS,
                        "max_retrieval_calls": MAX_RETRIEVAL_CALLS, "max_evidence": MAX_EVIDENCE,
                        "total_seconds": TOTAL_BUDGET_SECONDS},
            "round_trace": round_trace,
        },
    )


def _evidence_sufficient(question: str, chunks: list[dict]) -> tuple[bool, list[str]]:
    normalized = normalize_rag_query(question)
    missing = []
    if not chunks:
        missing.append("evidence")
    sources = {item.get("source_id") or item.get("source_path") for item in chunks if item.get("source_path")}
    if normalized.intent == "complex" and len(sources) < 2:
        missing.append("source_diversity")
    corpus = " ".join(str(item.get("index_text") or item.get("content") or "") for item in chunks).casefold()
    if normalized.exact_terms and not all(term.casefold() in corpus for term in normalized.exact_terms):
        missing.append("exact_terms")
    return not missing, missing


def _focused_rewrites(question: str, missing: list[str], round_index: int) -> list[str]:
    rewrites = []
    if "exact_terms" in missing:
        rewrites.append(f"{question} 精确参数 单位 原文")
    if "source_diversity" in missing:
        rewrites.extend([f"{question} 实验手册 SOP", f"{question} 论文 文献 数据"])
    if "evidence" in missing:
        rewrites.append(f"{question} 同义词 相关术语")
    if not rewrites:
        rewrites.append(f"{question} 补充证据 第{round_index + 1}轮")
    return list(dict.fromkeys(rewrites))


def _diversify(chunks: list[dict], top_k: int) -> list[dict]:
    selected = []
    counts: dict[str, int] = {}
    for chunk in chunks:
        source = str(chunk.get("document_id") or chunk.get("source_path") or "")
        if counts.get(source, 0) >= 2:
            continue
        counts[source] = counts.get(source, 0) + 1
        selected.append(chunk)
        if len(selected) >= top_k:
            break
    return selected
