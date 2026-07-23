from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterable

from app.services.rag.ingestion.loaders import LoadedUnit


TARGET_TOKENS = 512
MIN_TOKENS = 256
MAX_TOKENS = 768
PREFIX_MAX_TOKENS = 120


def build_chunks(
    units: list[LoadedUnit],
    source_metadata: dict,
    max_chars: int | None = None,
    *,
    generation_id: str = "legacy",
    target_tokens: int = TARGET_TOKENS,
) -> list[dict]:
    """Build structure-aware child chunks while preserving verbatim citation text.

    ``max_chars`` remains accepted for API compatibility. New callers should use
    token targets; the optional character ceiling is only a final safety guard.
    """
    chunks: list[dict] = []
    source_path = str(source_metadata["source_path"])
    doc_type = str(source_metadata["doc_type"])
    for unit in units:
        section_path = unit.section or unit.title or source_metadata.get("topic") or source_metadata["file_name"]
        parent_type = _parent_type(doc_type)
        parent_id = hashlib.sha256(
            f"{generation_id}:{source_path}:{parent_type}:{section_path}".encode("utf-8")
        ).hexdigest()[:24]
        parts = _split_unit_text(
            unit.content,
            doc_type,
            target_tokens=max(64, int(target_tokens)),
            max_tokens=MAX_TOKENS,
            max_chars=max_chars,
        )
        for local_index, part in enumerate(parts, start=1):
            content = _normalize_text(part)
            if not content:
                continue
            chunk_index = len(chunks)
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            chunk_id = hashlib.sha256(
                f"{generation_id}:{source_path}:{chunk_index}:{content_hash}".encode("utf-8")
            ).hexdigest()[:24]
            prefix = (
                _context_prefix(source_metadata, unit, section_path)
                if os.getenv("RAG_CONTEXTUAL_CHUNKS_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
                else ""
            )
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "generation_id": generation_id,
                    "doc_type": doc_type,
                    "source_path": source_path,
                    "file_name": source_metadata["file_name"],
                    "title": unit.title or source_metadata.get("topic"),
                    "section": unit.section,
                    "page_number": unit.page_number,
                    "sheet_name": unit.sheet_name,
                    "row_start": unit.row_start,
                    "row_end": unit.row_end,
                    "chunk_index": chunk_index,
                    "content": content,
                    "context_prefix": prefix,
                    "index_text": f"{prefix}\n\n{content}".strip(),
                    "content_hash": content_hash,
                    "metadata": {
                        "topic": source_metadata.get("topic"),
                        "version": source_metadata.get("version"),
                        "year": source_metadata.get("year"),
                        "language": source_metadata.get("language"),
                        "source_id": source_metadata.get("source_id"),
                        "generation_id": generation_id,
                        "parent_id": parent_id,
                        "parent_type": parent_type,
                        "section_path": section_path,
                        "heading": unit.section or unit.title,
                        "child_index": local_index,
                        "chunk_strategy": _chunk_strategy(doc_type),
                        "token_count": _token_count(content),
                        "context_prefix_policy": "deterministic-v1",
                        **(unit.metadata or {}),
                    },
                }
            )
    return chunks


def _parent_type(doc_type: str) -> str:
    return {
        "manual": "sop_section",
        "paper": "paper_section",
        "experiment_data": "table_block",
        "media_recipe": "recipe_group",
    }.get(doc_type, "source_section")


def _chunk_strategy(doc_type: str) -> str:
    return {
        "manual": "sop_step_token_chunking",
        "paper": "paper_section_token_chunking",
        "experiment_data": "table_header_row_chunking",
        "media_recipe": "recipe_component_group_chunking",
    }.get(doc_type, "recursive_token_chunking")


def _split_unit_text(
    text: str,
    doc_type: str,
    *,
    target_tokens: int,
    max_tokens: int,
    max_chars: int | None,
) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    if doc_type == "manual":
        atoms = _split_sop_steps(normalized)
        overlap = False
    elif doc_type == "paper":
        atoms = _split_paper_sections(normalized)
        overlap = True
    elif doc_type in {"experiment_data", "media_recipe"}:
        atoms = _split_table_or_recipe(normalized)
        overlap = False
    else:
        atoms = _recursive_atoms(normalized)
        overlap = True
    chunks = _pack_token_parts(atoms, target_tokens, max_tokens, overlap=overlap)
    if max_chars:
        chunks = [piece for chunk in chunks for piece in _hard_split_chars(chunk, max_chars)]
    return chunks


def _split_sop_steps(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    parts: list[str] = []
    current: list[str] = []
    step_pattern = re.compile(r"^(?:(?:step|步骤)\s*)?\d+[.)、]\s+|^[-*]\s+|^第\s*\d+\s*步", re.I)
    for line in lines:
        if step_pattern.search(line) and current:
            parts.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        parts.append("\n".join(current))
    return _explode_oversized(parts)


def _split_paper_sections(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    parts: list[str] = []
    current: list[str] = []
    heading_pattern = re.compile(
        r"^(abstract|introduction|materials?\s+and\s+methods|methods?|results?|discussion|conclusions?|摘要|引言|材料与方法|结果|讨论|结论)\b",
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
    return _explode_oversized(parts)


def _split_table_or_recipe(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return _recursive_atoms(text)
    header = lines[0]
    return [header if index == 0 else f"{header}\n{line}" for index, line in enumerate(lines)]


def _recursive_atoms(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    return _explode_oversized(paragraphs or [text])


def _explode_oversized(parts: Iterable[str], max_tokens: int = MAX_TOKENS) -> list[str]:
    result: list[str] = []
    for part in parts:
        if _token_count(part) <= max_tokens:
            result.append(part)
            continue
        sentences = [item.strip() for item in re.split(r"(?<=[。！？!?；;.!])\s*|\n+", part) if item.strip()]
        if len(sentences) == 1:
            result.extend(_hard_split_tokens(part, max_tokens))
        else:
            result.extend(sentences)
    return result


def _pack_token_parts(parts: list[str], target: int, maximum: int, *, overlap: bool) -> list[str]:
    packed: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for raw in parts:
        for part in _hard_split_tokens(raw, maximum):
            part_tokens = _token_count(part)
            if current and current_tokens + part_tokens > target and current_tokens >= MIN_TOKENS:
                packed.append("\n\n".join(current))
                carry = _tail_for_overlap(current, max(1, int(target * 0.10))) if overlap else []
                current = carry
                current_tokens = sum(_token_count(item) for item in current)
            current.append(part)
            current_tokens += part_tokens
            if current_tokens >= maximum:
                packed.append("\n\n".join(current))
                current = []
                current_tokens = 0
    if current:
        packed.append("\n\n".join(current))
    return packed


def _tail_for_overlap(parts: list[str], budget: int) -> list[str]:
    selected: list[str] = []
    used = 0
    for part in reversed(parts):
        count = _token_count(part)
        if selected and used + count > budget:
            break
        selected.insert(0, part)
        used += count
        if used >= budget:
            break
    return selected


def _hard_split_tokens(text: str, maximum: int) -> list[str]:
    if _token_count(text) <= maximum:
        return [text]
    tokens = re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*|\S", text)
    return [" ".join(tokens[index : index + maximum]).strip() for index in range(0, len(tokens), maximum)]


def _hard_split_chars(text: str, maximum: int) -> list[str]:
    return [text[index : index + maximum] for index in range(0, len(text), maximum)]


def _context_prefix(source: dict, unit: LoadedUnit, section_path: str) -> str:
    fields = [
        f"文档：{unit.title or source.get('topic') or source.get('file_name')}",
        f"类型：{source.get('doc_type')}",
        f"版本：{source.get('version')}" if source.get("version") else None,
        f"章节：{section_path}" if section_path else None,
        f"年份：{source.get('year')}" if source.get("year") else None,
        f"页码：{unit.page_number}" if unit.page_number is not None else None,
        f"工作表：{unit.sheet_name}" if unit.sheet_name else None,
        f"行：{unit.row_start}-{unit.row_end}" if unit.row_start is not None else None,
    ]
    prefix = "；".join(str(field) for field in fields if field)
    if _token_count(prefix) <= PREFIX_MAX_TOKENS:
        return prefix
    return " ".join(re.findall(r"\S+", prefix)[:PREFIX_MAX_TOKENS])


def _token_count(text: str) -> int:
    return len(re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*|\S", str(text or "")))


def _normalize_text(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()
