import hashlib
import re

from app.services.rag.ingestion.loaders import LoadedUnit


def build_chunks(
    units: list[LoadedUnit],
    source_metadata: dict,
    max_chars: int = 1200,
) -> list[dict]:
    chunks: list[dict] = []
    for unit in units:
        section_path = unit.section or unit.title or source_metadata.get("topic") or source_metadata["file_name"]
        parent_type = _parent_type(source_metadata["doc_type"])
        parent_id = hashlib.sha256(
            f"{source_metadata['source_path']}:{parent_type}:{section_path}".encode("utf-8")
        ).hexdigest()[:24]
        for local_index, part in enumerate(
            _split_unit_text(unit.content, source_metadata["doc_type"], max_chars=max_chars),
            start=1,
        ):
            content = _normalize_text(part)
            if not content:
                continue
            chunk_index = len(chunks)
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            chunk_id = hashlib.sha256(
                f"{source_metadata['source_path']}:{chunk_index}:{content_hash}".encode("utf-8")
            ).hexdigest()[:24]
            chunks.append({
                "chunk_id": chunk_id,
                "doc_type": source_metadata["doc_type"],
                "source_path": source_metadata["source_path"],
                "file_name": source_metadata["file_name"],
                "title": unit.title or source_metadata.get("topic"),
                "section": unit.section,
                "page_number": unit.page_number,
                "sheet_name": unit.sheet_name,
                "row_start": unit.row_start,
                "row_end": unit.row_end,
                "chunk_index": chunk_index,
                "content": content,
                "content_hash": content_hash,
                "metadata": {
                    "topic": source_metadata.get("topic"),
                    "version": source_metadata.get("version"),
                    "year": source_metadata.get("year"),
                    "language": source_metadata.get("language"),
                    "parent_id": parent_id,
                    "parent_type": parent_type,
                    "section_path": section_path,
                    "heading": unit.section or unit.title,
                    "child_index": local_index,
                    "chunk_strategy": _chunk_strategy(source_metadata["doc_type"]),
                    **(unit.metadata or {}),
                },
            })
    return chunks


def _parent_type(doc_type: str) -> str:
    if doc_type == "manual":
        return "sop_section"
    if doc_type == "paper":
        return "paper_section"
    if doc_type == "experiment_data":
        return "table_block"
    return "source_section"


def _chunk_strategy(doc_type: str) -> str:
    if doc_type == "manual":
        return "sop_step_chunking"
    if doc_type == "paper":
        return "paper_section_chunking"
    if doc_type == "experiment_data":
        return "table_block_chunking"
    return "paragraph_chunking"


def _split_unit_text(text: str, doc_type: str, max_chars: int) -> list[str]:
    normalized = _normalize_text(text)
    if doc_type == "manual":
        step_parts = _split_sop_steps(normalized)
        if len(step_parts) > 1:
            return _pack_parts(step_parts, max_chars=max_chars)
    if doc_type == "paper":
        section_parts = _split_paper_sections(normalized)
        if len(section_parts) > 1:
            return _pack_parts(section_parts, max_chars=max_chars)
    return _split_text(normalized, max_chars=max_chars)


def _split_sop_steps(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    parts: list[str] = []
    current: list[str] = []
    step_pattern = re.compile(r"^(?:step\s*)?\d+[\).、]\s+|^[-*]\s+|^(?:步骤|第\s*\d+\s*步)", re.I)
    for line in lines:
        if step_pattern.search(line) and current:
            parts.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        parts.append("\n".join(current))
    return parts


def _split_paper_sections(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    parts: list[str] = []
    current: list[str] = []
    heading_pattern = re.compile(
        r"^(abstract|introduction|materials?\s+and\s+methods|methods?|results?|discussion|conclusions?)\b",
        re.I,
    )
    for line in lines:
        if heading_pattern.search(line) and current:
            parts.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        parts.append("\n".join(current))
    return parts


def _pack_parts(parts: list[str], max_chars: int) -> list[str]:
    packed: list[str] = []
    current = ""
    for part in parts:
        if len(part) > max_chars:
            if current:
                packed.append(current)
                current = ""
            packed.extend(_split_text(part, max_chars=max_chars))
            continue
        candidate = f"{current}\n\n{part}".strip() if current else part
        if len(candidate) > max_chars:
            if current:
                packed.append(current)
            current = part
        else:
            current = candidate
    if current:
        packed.append(current)
    return packed


def _split_text(text: str, max_chars: int) -> list[str]:
    normalized = _normalize_text(text)
    if len(normalized) <= max_chars:
        return [normalized]

    paragraphs = [item.strip() for item in re.split(r"\n{2,}", normalized) if item.strip()]
    parts: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                parts.append(current)
                current = ""
            parts.extend(paragraph[i : i + max_chars] for i in range(0, len(paragraph), max_chars))
            continue
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) > max_chars:
            parts.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()
