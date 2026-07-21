def format_rag_citation(citation) -> str:
    location = _format_location(citation)
    metadata_parts = []
    for key in ("version", "year", "language", "evidence_type"):
        value = getattr(citation, key, None)
        if value:
            metadata_parts.append(f"{key}={value}")
    suffix = f"; {', '.join(metadata_parts)}" if metadata_parts else ""
    return f"{location}{suffix}"


def _format_location(citation) -> str:
    if getattr(citation, "page_number", None):
        return f"page {citation.page_number}"
    if getattr(citation, "sheet_name", None):
        if getattr(citation, "row_start", None) and getattr(citation, "row_end", None):
            return f"sheet {citation.sheet_name}, rows {citation.row_start}-{citation.row_end}"
        if getattr(citation, "row_start", None):
            return f"sheet {citation.sheet_name}, row {citation.row_start}"
        return f"sheet {citation.sheet_name}"
    if getattr(citation, "section", None):
        return str(citation.section)
    return getattr(citation, "chunk_id", None) or "source"
