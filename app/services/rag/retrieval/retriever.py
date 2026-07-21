from app.services.rag.retrieval.hybrid_retriever import retrieve_hybrid_chunks
from app.services.rag.retrieval.reranker import rerank_chunks
from app.services.rag.retrieval.parent_expansion import expand_parent_context


DOC_TYPE_PRIORITY = {
    "media_recipe": 0,
    "manual": 1,
    "experiment_data": 2,
    "paper": 3,
    "document": 4,
}


def retrieve_chunks(
    question: str,
    top_k: int = 5,
    doc_types: list[str] | None = None,
    metadata_filters: dict | None = None,
) -> list[dict]:
    candidates = retrieve_hybrid_chunks(
        question,
        top_k=max(top_k * 3, top_k),
        doc_types=doc_types,
        metadata_filters=metadata_filters,
    )
    candidates.sort(
        key=lambda item: (
            DOC_TYPE_PRIORITY.get(item.get("doc_type"), 99),
            -float(item.get("hybrid_score") or 0),
            item.get("file_name") or "",
            int(item.get("chunk_index") or 0),
        )
    )
    seen_documents: set[int] = set()
    seen_content: set[tuple] = set()
    diversified: list[dict] = []
    remaining: list[dict] = []
    for item in candidates:
        fingerprint = (
            str(item.get("file_name") or "").casefold(),
            str(item.get("section") or "").casefold(),
            item.get("page_number"),
            item.get("sheet_name"),
            item.get("row_start"),
            item.get("row_end"),
            item.get("content_hash") or str(item.get("content") or "").strip(),
        )
        if fingerprint in seen_content:
            continue
        seen_content.add(fingerprint)
        document_id = int(item.get("document_id") or 0)
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
    diversified = expand_parent_context(diversified)
    # Small-to-big retrieval keeps small chunks searchable while reranking the answer context.
    return rerank_chunks(question, diversified, top_k)
