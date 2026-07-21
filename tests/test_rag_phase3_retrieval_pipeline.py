from app.core import database
from app.models.rag_retrieval_schema import RetrievalHit, RetrievalQuery
from app.models.rag_schema import RagQueryRequest
from app.services.rag.ingestion.index_store import ingest_file
from app.services.rag.retrieval.hybrid_retriever import retrieve_hybrid_bundle, retrieve_hybrid_chunks
from app.services.rag.eval.retrieval_eval import _score_case
from app.services.rag.retrieval.reranker import (
    BusinessCalibrationReranker,
    DeterministicFakeReranker,
    NoOpReranker,
    RuleBasedReranker,
    rerank_chunks,
)
from app.services.rag.retrieval.rrf import reciprocal_rank_fusion
from app.services.rag.service import answer_rag_question


def test_retrieval_query_defaults_and_hit_serialization():
    query = RetrievalQuery(original_query=" TAP  medium ")
    hit = RetrievalHit(evidence_id="chunk:1", content="TAP medium", source="sparse", sparse_rank=1)

    assert query.normalized_query == "tap medium"
    assert query.sparse_top_k == 30
    assert hit.model_dump()["evidence_id"] == "chunk:1"


def test_rrf_calculation_weights_and_stable_tie_breaking():
    sparse = [
        RetrievalHit(evidence_id="chunk:a", content="A", source="sparse", sparse_rank=1),
        RetrievalHit(evidence_id="chunk:b", content="B", source="sparse", sparse_rank=2),
    ]
    dense = [
        RetrievalHit(evidence_id="chunk:b", content="B", source="dense", dense_rank=1),
        RetrievalHit(evidence_id="chunk:a", content="A", source="dense", dense_rank=2),
    ]

    fused = reciprocal_rank_fusion(sparse, dense, k=60, sparse=1.0, dense=2.0)

    assert fused[0].evidence_id == "chunk:b"
    assert fused[0].fusion_score > fused[1].fusion_score
    assert fused[0].sparse_rank == 2
    assert fused[0].dense_rank == 1


def test_reranker_backends_are_deterministic(monkeypatch):
    hits = [
        RetrievalHit(evidence_id="chunk:1", content="alpha beta", source="fusion", fusion_score=0.1),
        RetrievalHit(evidence_id="chunk:2", content="gamma", source="fusion", fusion_score=0.2),
    ]
    query = RetrievalQuery(original_query="alpha", top_k=2)

    assert NoOpReranker().rerank(query, hits)[0].evidence_id == "chunk:1"
    assert DeterministicFakeReranker().rerank(query, hits)[0].evidence_id == "chunk:1"
    assert RuleBasedReranker().rerank(query, hits)[0].rerank_score is not None

    monkeypatch.setenv("RAG_RERANKER_ENABLED", "false")
    chunks = [
        {"chunk_id": "1", "content": "alpha beta", "file_name": "a.txt", "hybrid_score": 0.1},
        {"chunk_id": "2", "content": "gamma", "file_name": "b.txt", "hybrid_score": 0.2},
    ]
    reranked = rerank_chunks("alpha", chunks, top_k=2)
    assert reranked[0]["reranker_backend"] == "noop"


def test_business_calibration_records_bounded_scores(monkeypatch):
    monkeypatch.setenv("RAG_BUSINESS_CALIBRATION_MAX_DELTA", "0.03")
    hits = [
        RetrievalHit(
            evidence_id="chunk:1",
            content="TAP medium recipe component amount",
            source="bge",
            metadata={"file_name": "a.txt", "chunk_index": 0, "bge_rank_score": 1.0, "doc_type": "media_recipe"},
        ),
        RetrievalHit(
            evidence_id="chunk:2",
            content="unrelated background",
            source="bge",
            metadata={"file_name": "b.txt", "chunk_index": 1, "bge_rank_score": 0.5, "doc_type": "paper"},
        ),
    ]
    ranked = BusinessCalibrationReranker(None).rerank(RetrievalQuery(original_query="TAP recipe amount"), hits)

    assert ranked[0].metadata["business_delta"] <= 0.03
    assert ranked[0].metadata["final_score"] is not None
    assert "business_components" in ranked[0].metadata


def test_bge_unavailable_degrades_to_business_calibration(monkeypatch):
    monkeypatch.setenv("RAG_RERANKER_ENABLED", "true")
    monkeypatch.setenv("RAG_RERANKER_BACKEND", "bge_calibrated")
    chunks = [
        {"chunk_id": "1", "content": "alpha beta", "file_name": "a.txt", "hybrid_score": 0.1},
        {"chunk_id": "2", "content": "gamma", "file_name": "b.txt", "hybrid_score": 0.2},
    ]
    reranked = rerank_chunks("alpha", chunks, top_k=2)

    assert reranked[0]["reranker_backend"] in {"business_calibration_only", "bge_business_calibration"}
    if reranked[0]["reranker_backend"] == "business_calibration_only":
        assert reranked[0]["reranker_degraded"] is True


def test_hybrid_bundle_exposes_sparse_dense_semantic_and_rrf(isolated_sqlite_db, tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_ENABLED", "true")
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "fake")
    source = tmp_path / "manual__retrieval_demo__v1__en.txt"
    source.write_text("Photobioreactor setup protocol with sterile tubing.", encoding="utf-8")
    ingest_file(source)

    bundle = retrieve_hybrid_bundle("pbr setup protocol", top_k=2, doc_types=["manual"])
    chunks = retrieve_hybrid_chunks("pbr setup protocol", top_k=1, doc_types=["manual"])

    assert bundle.sparse_hits
    assert bundle.dense_hits
    assert bundle.semantic_hits
    assert bundle.fused_hits[0].fusion_score is not None
    assert chunks[0]["fusion_score"] == chunks[0]["hybrid_score"]
    assert "vector" in chunks[0]["retrieval_channels"]


def test_dense_retrieval_disabled_is_marked_as_fallback(isolated_sqlite_db, tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DENSE_RETRIEVAL_ENABLED", "false")
    source = tmp_path / "manual__dense_disabled__v1__en.txt"
    source.write_text("Contamination protocol microscopy inspection.", encoding="utf-8")
    ingest_file(source)

    bundle = retrieve_hybrid_bundle("contamination microscopy", top_k=1, doc_types=["manual"])

    assert bundle.dense_hits == []
    assert any(item.startswith("dense_retrieval_unavailable") for item in bundle.warnings)
    assert bundle.selected_hits


def test_metadata_filter_and_hit_level_trace(isolated_sqlite_db, tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_ENABLED", "true")
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "fake")
    source = tmp_path / "manual__trace_demo__2025__en.txt"
    source.write_text("Traceable contamination microscopy protocol.", encoding="utf-8")
    ingest_file(source)

    chunks = retrieve_hybrid_chunks("contamination microscopy", top_k=1, metadata_filters={"year": "2025"})
    response = answer_rag_question(RagQueryRequest(question="How to check contamination?", top_k=1))
    trace_id = response.debug["trace_id"]
    rows = database.list_rag_retrieval_trace_rows(query_id=trace_id)

    assert chunks[0]["year"] == "2025"
    assert chunks[0]["retrieval_hit"]["fusion_score"] is not None
    assert rows
    assert rows[0]["stage"] == "selected"
    assert rows[0]["fusion_score"] is not None


def test_retrieval_eval_reports_ndcg():
    case = {
        "id": "ndcg-demo",
        "category": "retrieval",
        "expected_answer_contains": ["tap"],
        "required_evidence_types": [],
    }
    candidates = [
        {"chunk_id": "a", "content": "unrelated"},
        {"chunk_id": "b", "content": "TAP medium recipe"},
    ]

    scored = _score_case(case, "hybrid", candidates, top_k=2)

    assert 0.0 < scored["ndcg"] < 1.0
