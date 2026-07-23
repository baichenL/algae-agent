from __future__ import annotations

import os
import re
import threading
import time
from typing import Protocol

from app.models.rag_retrieval_schema import RetrievalHit, RetrievalQuery


EVIDENCE_TYPE_HINTS = {
    "recipe_component": {"recipe", "component", "amount", "medium", "tap"},
    "table_column": {"field", "column", "schema", "od750", "biomass"},
    "data_value": {"max", "min", "mean", "row", "value", "trend", "biomass"},
    "sop_fact": {"sop", "manual", "protocol", "procedure", "step"},
    "paper_claim": {"paper", "literature", "method", "machine", "learning"},
}

DOC_TYPE_BOOST = {
    "media_recipe": 0.8,
    "manual": 0.55,
    "experiment_data": 0.45,
    "paper": 0.25,
}

DEFAULT_BGE_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_MAX_PASSAGE_CHARS = 2400
DEFAULT_CALIBRATION_MAX_DELTA = 0.05
DEFAULT_LOAD_COOLDOWN_SECONDS = 300

_RERANKER_LOCK = threading.Lock()
_BGE_SINGLETON: "BgeReranker | None" = None
_ACTIVE_RERANKER: "Reranker | None" = None


class Reranker(Protocol):
    backend_name: str
    version: str

    def rerank(self, query: RetrievalQuery, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        ...


class NoOpReranker:
    backend_name = "noop"
    version = "2026-07-phase3-v1"

    def rerank(self, query: RetrievalQuery, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        return [
            hit.model_copy(update={"rerank_score": hit.fusion_score or 0.0, "final_rank": index})
            for index, hit in enumerate(hits, start=1)
        ]


class DeterministicFakeReranker:
    backend_name = "fake"
    version = "2026-07-phase3-v1"

    def rerank(self, query: RetrievalQuery, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        ranked = sorted(
            hits,
            key=lambda hit: (
                -len(set(_terms(query.original_query)) & set(_terms(hit.content))),
                -float(hit.fusion_score or 0.0),
                hit.evidence_id,
            ),
        )
        return [
            hit.model_copy(
                update={
                    "rerank_score": float(len(set(_terms(query.original_query)) & set(_terms(hit.content)))),
                    "final_rank": index,
                    "metadata": {
                        **hit.metadata,
                        "rerank_components": {
                            "backend": "fake",
                            "term_overlap": len(set(_terms(query.original_query)) & set(_terms(hit.content))),
                            "total": float(len(set(_terms(query.original_query)) & set(_terms(hit.content)))),
                        },
                    },
                }
            )
            for index, hit in enumerate(ranked, start=1)
        ]


class RuleBasedReranker:
    backend_name = "rule"
    version = "2026-07-phase3-v1"

    def rerank(self, query: RetrievalQuery, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        scored = []
        for hit in hits:
            chunk = _hit_to_chunk(hit)
            score, components = _rerank_score(query.original_query, chunk)
            metadata = {**hit.metadata, "rerank_components": components}
            scored.append(hit.model_copy(update={"rerank_score": score, "metadata": metadata}))
        ranked = sorted(
            scored,
            key=lambda item: (
                -float(item.rerank_score or 0.0),
                item.metadata.get("file_name") or "",
                int(item.metadata.get("chunk_index") or 0),
                item.evidence_id,
            ),
        )
        return [hit.model_copy(update={"final_rank": index}) for index, hit in enumerate(ranked, start=1)]


class BgeReranker:
    backend_name = "bge"
    version = "2026-07-phase4-v1"

    def __init__(self) -> None:
        self.model_name = os.getenv("RAG_BGE_RERANKER_MODEL", DEFAULT_BGE_MODEL)
        self.max_passage_chars = max(int(os.getenv("RAG_RERANKER_MAX_PASSAGE_CHARS", str(DEFAULT_MAX_PASSAGE_CHARS))), 400)
        self._model = None
        self._load_error: str | None = None
        self._load_error_at: float | None = None

    def rerank(self, query: RetrievalQuery, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        if not hits:
            return []
        model = self._load_model()
        pairs = [
            [query.original_query, _truncate_passage(query.original_query, hit.content, self.max_passage_chars)]
            for hit in hits
        ]
        scores = self._compute_scores(model, pairs)
        scored = []
        for hit, score in zip(hits, scores):
            metadata = {
                **hit.metadata,
                "bge_score": float(score),
                "bge_model": self.model_name,
                "bge_score_scope": "same_query_ranking_only",
                "rerank_input_chars": len(_truncate_passage(query.original_query, hit.content, self.max_passage_chars)),
            }
            scored.append(hit.model_copy(update={"rerank_score": float(score), "metadata": metadata}))
        ranked = sorted(
            scored,
            key=lambda item: (
                -float(item.rerank_score or 0.0),
                item.metadata.get("file_name") or "",
                int(item.metadata.get("chunk_index") or 0),
                item.evidence_id,
            ),
        )
        total = max(len(ranked), 1)
        result = []
        for index, hit in enumerate(ranked, start=1):
            metadata = {
                **hit.metadata,
                "bge_rank_score": (total - index + 1) / total,
            }
            result.append(hit.model_copy(update={"final_rank": index, "metadata": metadata}))
        return result

    def _load_model(self):
        if self._model is not None:
            return self._model
        cooldown = max(int(os.getenv("RAG_RERANKER_LOAD_COOLDOWN_SECONDS", str(DEFAULT_LOAD_COOLDOWN_SECONDS))), 1)
        if self._load_error_at and time.monotonic() - self._load_error_at < cooldown:
            raise ParserlessRerankerUnavailable(self._load_error or "reranker_load_cooldown")
        try:
            if os.getenv("RAG_BGE_LOCAL_FILES_ONLY", "true").strip().lower() in {"1", "true", "yes", "on"}:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from FlagEmbedding import FlagReranker

            use_fp16 = _cuda_available() and os.getenv("RAG_BGE_RERANKER_USE_FP16", "true").strip().lower() in {"1", "true", "yes", "on"}
            self._model = FlagReranker(self.model_name, use_fp16=use_fp16)
            self._load_error = None
            self._load_error_at = None
            return self._model
        except Exception as exc:
            self._load_error = str(exc)
            self._load_error_at = time.monotonic()
            if _strict_mode():
                raise RuntimeError(f"BGE reranker is required for this query but unavailable: {exc}") from exc
            raise ParserlessRerankerUnavailable(str(exc)) from exc

    def _compute_scores(self, model, pairs: list[list[str]]) -> list[float]:
        scores = model.compute_score(pairs)
        if not isinstance(scores, list):
            scores = [scores]
        return [float(score) for score in scores]


class ParserlessRerankerUnavailable(RuntimeError):
    pass


class BusinessCalibrationReranker:
    backend_name = "bge_business_calibration"
    version = "2026-07-phase4-v1"

    def __init__(self, base: Reranker | None = None, degraded_reason: str | None = None) -> None:
        self.base = base
        self.degraded_reason = degraded_reason
        if base is None:
            self.backend_name = "business_calibration_only"
        elif getattr(base, "backend_name", "") != "bge":
            self.backend_name = f"{base.backend_name}_business_calibration"

    def rerank(self, query: RetrievalQuery, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        if self.base is None:
            base_ranked = RuleBasedReranker().rerank(query, hits)
        else:
            base_ranked = self.base.rerank(query, hits)
        max_delta = max(float(os.getenv("RAG_BUSINESS_CALIBRATION_MAX_DELTA", str(DEFAULT_CALIBRATION_MAX_DELTA))), 0.0)
        calibrated = []
        total = max(len(base_ranked), 1)
        for index, hit in enumerate(base_ranked, start=1):
            chunk = _hit_to_chunk(hit)
            raw_business_score, components = _rerank_score(query.original_query, chunk)
            business_delta = _bounded_business_delta(raw_business_score, max_delta)
            base_rank_score = float(hit.metadata.get("bge_rank_score") or ((total - index + 1) / total))
            final_score = base_rank_score + business_delta
            metadata = {
                **hit.metadata,
                "business_score": raw_business_score,
                "business_delta": business_delta,
                "business_calibration_max_delta": max_delta,
                "business_components": components,
                "rerank_components": {**components, "raw_total": components.get("total"), "total": final_score},
                "final_score": final_score,
            }
            if self.degraded_reason:
                metadata["reranker_degraded"] = True
                metadata["reranker_degraded_reason"] = self.degraded_reason
            calibrated.append(hit.model_copy(update={"rerank_score": final_score, "metadata": metadata}))
        ranked = sorted(
            calibrated,
            key=lambda item: (
                -float(item.rerank_score or 0.0),
                item.metadata.get("file_name") or "",
                int(item.metadata.get("chunk_index") or 0),
                item.evidence_id,
            ),
        )
        return [hit.model_copy(update={"final_rank": index}) for index, hit in enumerate(ranked, start=1)]


def rerank_chunks(question: str, candidates: list[dict], top_k: int) -> list[dict]:
    query = RetrievalQuery(original_query=question, top_k=top_k, rerank_top_k=top_k)
    hits = [_chunk_to_hit(item) for item in candidates]
    reranker = get_reranker()
    ranked_hits = reranker.rerank(query, hits)[: max(int(top_k or 5), 1)]
    result = []
    for hit in ranked_hits:
        chunk = _hit_to_chunk(hit)
        chunk["reranker_backend"] = reranker.backend_name
        chunk["reranker_version"] = reranker.version
        result.append(chunk)
    return result


def get_reranker() -> Reranker:
    global _ACTIVE_RERANKER
    if os.getenv("RAG_RERANKER_ENABLED", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        return NoOpReranker()
    backend = os.getenv("RAG_RERANKER_BACKEND", "bge_calibrated").strip().lower()
    if backend == "noop":
        return NoOpReranker()
    if backend == "fake":
        return DeterministicFakeReranker()
    if backend in {"bge", "bge_calibrated", "bge_business", "bge_business_calibration"}:
        if _ACTIVE_RERANKER is not None:
            return _ACTIVE_RERANKER
        bge = _bge_singleton()
        if bge._model is not None:
            _ACTIVE_RERANKER = bge if backend == "bge" else BusinessCalibrationReranker(bge)
            return _ACTIVE_RERANKER
        reason = bge._load_error or "bge_not_prewarmed"
        if _strict_mode():
            raise RuntimeError(f"BGE reranker is required but unavailable: {reason}")
        return BusinessCalibrationReranker(None, degraded_reason=reason)
    return RuleBasedReranker()


def prewarm_reranker() -> dict:
    """Load the local model once during application startup, never on a user query."""
    global _ACTIVE_RERANKER
    if os.getenv("RAG_RERANKER_PREWARM_ENABLED", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        return reranker_runtime_status(reason="prewarm_disabled")
    backend = os.getenv("RAG_RERANKER_BACKEND", "bge_calibrated").strip().lower()
    if backend not in {"bge", "bge_calibrated", "bge_business", "bge_business_calibration"}:
        _ACTIVE_RERANKER = get_reranker()
        return reranker_runtime_status()
    bge = _bge_singleton()
    try:
        bge._load_model()
        _ACTIVE_RERANKER = bge if backend == "bge" else BusinessCalibrationReranker(bge)
    except ParserlessRerankerUnavailable:
        _ACTIVE_RERANKER = None
    return reranker_runtime_status()


def reranker_runtime_status(*, reason: str | None = None) -> dict:
    backend = os.getenv("RAG_RERANKER_BACKEND", "bge_calibrated").strip().lower()
    if backend not in {"bge", "bge_calibrated", "bge_business", "bge_business_calibration"}:
        return {"status": "ready", "backend": backend, "model": None, "degraded": False, "reason": reason}
    bge = _bge_singleton()
    ready = bge._model is not None
    return {
        "status": "ready" if ready else "degraded",
        "backend": backend,
        "model": bge.model_name,
        "degraded": not ready,
        "reason": reason or bge._load_error or "bge_not_prewarmed",
        "cooldown_active": bool(bge._load_error_at),
    }


def _bge_singleton() -> BgeReranker:
    global _BGE_SINGLETON
    if _BGE_SINGLETON is None:
        with _RERANKER_LOCK:
            if _BGE_SINGLETON is None:
                _BGE_SINGLETON = BgeReranker()
    return _BGE_SINGLETON


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _chunk_to_hit(chunk: dict) -> RetrievalHit:
    evidence_id = chunk.get("evidence_id") or f"chunk:{chunk.get('chunk_id')}"
    return RetrievalHit(
        evidence_id=evidence_id,
        document_id=chunk.get("document_id"),
        content=chunk.get("answer_context") or chunk.get("content") or "",
        source="reranker_input",
        sparse_rank=chunk.get("sparse_rank"),
        dense_rank=chunk.get("dense_rank"),
        semantic_rank=chunk.get("semantic_rank"),
        sparse_score=chunk.get("sparse_score"),
        dense_score=chunk.get("dense_score") or chunk.get("vector_score"),
        semantic_score=chunk.get("semantic_score"),
        fusion_score=chunk.get("fusion_score") or chunk.get("hybrid_score"),
        metadata={**chunk},
    )


def _hit_to_chunk(hit: RetrievalHit) -> dict:
    chunk = dict(hit.metadata)
    chunk["rerank_score"] = hit.rerank_score
    chunk["final_rank"] = hit.final_rank
    if "rerank_components" in hit.metadata:
        chunk["rerank_components"] = hit.metadata["rerank_components"]
    return chunk


def _rerank_score(question: str, item: dict) -> tuple[float, dict]:
    query_terms = _terms(question)
    metadata = item.get("metadata") or {}
    text = " ".join(
        str(value or "")
        for value in [
            item.get("file_name"),
            item.get("title"),
            item.get("section"),
            metadata.get("topic"),
            metadata.get("parent_type"),
            item.get("content"),
        ]
    )
    content_terms = _terms(text)
    overlap = len(query_terms & content_terms)
    channel_bonus = 0.25 * len(item.get("retrieval_channels") or [])
    hybrid = float(item.get("hybrid_score") or 0.0)
    doc_bonus = DOC_TYPE_BOOST.get(item.get("doc_type"), 0.0)
    intent_bonus = _intent_bonus(query_terms, item)
    citation_bonus = _citation_completeness_bonus(item)
    source_bonus = _source_boundary_bonus(item)
    total = overlap + channel_bonus + hybrid + doc_bonus + intent_bonus + citation_bonus + source_bonus
    return total, {
        "term_overlap": overlap,
        "channel_bonus": channel_bonus,
        "hybrid_score": hybrid,
        "doc_type_bonus": doc_bonus,
        "intent_bonus": intent_bonus,
        "citation_bonus": citation_bonus,
        "source_boundary_bonus": source_bonus,
        "total": total,
    }


def _bounded_business_delta(raw_score: float, max_delta: float) -> float:
    if max_delta <= 0:
        return 0.0
    # Convert an unbounded rule score into a small, monotonic adjustment.
    normalized = raw_score / (abs(raw_score) + 8.0) if raw_score else 0.0
    return max(min(normalized * max_delta, max_delta), -max_delta)


def _truncate_passage(query: str, passage: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(passage or "")).strip()
    if len(text) <= max_chars:
        return text
    head_chars = max_chars // 3
    tail_chars = max_chars // 5
    middle_chars = max_chars - head_chars - tail_chars - 10
    query_terms = sorted(_terms(query), key=len, reverse=True)
    lower = text.lower()
    hit_index = -1
    for term in query_terms:
        if len(term) < 2:
            continue
        hit_index = lower.find(term.lower())
        if hit_index >= 0:
            break
    if hit_index < 0:
        hit_index = len(text) // 2
    middle_start = max(hit_index - middle_chars // 2, head_chars)
    middle_end = min(middle_start + middle_chars, len(text) - tail_chars)
    return f"{text[:head_chars]} ... {text[middle_start:middle_end]} ... {text[-tail_chars:]}"


def _strict_mode() -> bool:
    return os.getenv("RAG_STRICT_PRODUCTION", "false").strip().lower() in {"1", "true", "yes", "on"}


def _intent_bonus(query_terms: set[str], item: dict) -> float:
    metadata = item.get("metadata") or {}
    evidence_type = metadata.get("evidence_type") or metadata.get("fact_type") or "text_chunk"
    doc_type = item.get("doc_type")
    if evidence_type == "text_chunk":
        evidence_type = {
            "media_recipe": "recipe_component",
            "manual": "sop_fact",
            "experiment_data": "data_value",
            "paper": "paper_claim",
        }.get(doc_type, "text_chunk")
    hints = EVIDENCE_TYPE_HINTS.get(evidence_type, set())
    return 0.35 if query_terms & hints else 0.0


def _citation_completeness_bonus(item: dict) -> float:
    fields = [
        item.get("section"),
        item.get("page_number"),
        item.get("sheet_name"),
        item.get("row_start"),
        item.get("version") or (item.get("metadata") or {}).get("version"),
        item.get("year") or (item.get("metadata") or {}).get("year"),
        item.get("language") or (item.get("metadata") or {}).get("language"),
    ]
    return min(sum(1 for value in fields if value not in {None, ""}) * 0.05, 0.25)


def _source_boundary_bonus(item: dict) -> float:
    doc_type = item.get("doc_type")
    if doc_type in {"media_recipe", "manual", "experiment_data"}:
        return 0.2
    if doc_type == "paper":
        return 0.05
    return 0.0


def _terms(text: str) -> set[str]:
    return {
        token.strip("_-+").lower()
        for token in re.findall(r"[a-z0-9_+\-]+|[\u4e00-\u9fff]{2,}", str(text or ""))
        if token.strip("_-+")
    }
