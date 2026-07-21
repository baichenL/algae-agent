from app.models.rag_evidence_schema import EvidenceUnit


def chunk_to_evidence_unit(chunk: dict) -> EvidenceUnit:
    location = {}
    if chunk.get("page_number"):
        location["page_number"] = chunk.get("page_number")
    if chunk.get("sheet_name"):
        location["sheet_name"] = chunk.get("sheet_name")
    if chunk.get("row_start"):
        location["row_start"] = chunk.get("row_start")
    if chunk.get("row_end"):
        location["row_end"] = chunk.get("row_end")
    if chunk.get("section"):
        location["section"] = chunk.get("section")
    location["chunk_index"] = chunk.get("chunk_index")

    citation = {
        "chunk_id": chunk.get("chunk_id"),
        "file_name": chunk.get("file_name"),
        "source_path": chunk.get("source_path"),
        "doc_type": chunk.get("doc_type"),
        **location,
    }

    metadata = {
        **(chunk.get("metadata") or {}),
        "retrieval_channels": chunk.get("retrieval_channels") or [],
        "hybrid_score": chunk.get("hybrid_score"),
        "vector_score": chunk.get("vector_score"),
        "semantic_score": chunk.get("semantic_score"),
        "parent_id": (chunk.get("metadata") or {}).get("parent_id"),
        "parent_type": (chunk.get("metadata") or {}).get("parent_type"),
        "retrieval_hit": chunk.get("retrieval_hit"),
    }

    return EvidenceUnit(
        evidence_id=f"chunk:{chunk.get('chunk_id')}",
        document_id=chunk.get("document_id"),
        document_version=metadata.get("version"),
        content=chunk.get("answer_context") or chunk.get("content") or "",
        content_hash=chunk.get("content_hash") or "",
        source_type=chunk.get("doc_type") or "document",
        source_file=chunk.get("file_name") or "",
        fact_type="text_chunk",
        entity=chunk.get("title") or chunk.get("topic"),
        text_span=chunk.get("answer_context") or chunk.get("content") or "",
        page_number=chunk.get("page_number"),
        section_path=_section_path(metadata.get("section_path") or chunk.get("section")),
        parent_id=metadata.get("parent_id"),
        previous_id=metadata.get("previous_id"),
        next_id=metadata.get("next_id"),
        source_locator={**citation},
        parser_version=metadata.get("parser_version"),
        element_type=metadata.get("element_type") or "paragraph",
        location=location,
        confidence=0.45,
        citation=citation,
        metadata=metadata,
    )


def chunks_to_evidence_units(chunks: list[dict]) -> list[EvidenceUnit]:
    return [chunk_to_evidence_unit(chunk) for chunk in chunks]


def _section_path(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item or "").strip()]
    if not value:
        return []
    return [str(value)]
