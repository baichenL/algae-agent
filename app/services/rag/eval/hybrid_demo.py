from __future__ import annotations

from app.services.rag.embedding_service import embedding_runtime_status
from app.services.rag.retrieval.retriever import retrieve_chunks


def build_hybrid_demo_report(question: str, top_k: int = 5) -> dict:
    chunks = retrieve_chunks(question, top_k=top_k)
    return {
        "question": question,
        "embedding": embedding_runtime_status(),
        "retrieved_count": len(chunks),
        "retrieval_channels": sorted(
            {
                channel
                for chunk in chunks
                for channel in (chunk.get("retrieval_channels") or [])
            }
        ),
        "top_chunks": [
            {
                "chunk_id": chunk.get("chunk_id"),
                "file_name": chunk.get("file_name"),
                "doc_type": chunk.get("doc_type"),
                "section": chunk.get("section"),
                "hybrid_score": chunk.get("hybrid_score"),
                "vector_score": chunk.get("vector_score"),
                "semantic_score": chunk.get("semantic_score"),
                "rerank_score": chunk.get("rerank_score"),
                "rerank_components": chunk.get("rerank_components"),
                "retrieval_channels": chunk.get("retrieval_channels") or [],
                "vector_backend": chunk.get("vector_backend"),
            }
            for chunk in chunks
        ],
    }
